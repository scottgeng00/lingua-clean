"""Decontaminate fineweb_edu_subsets/fwedu_heldout against the 4 x 8M training subsets, as dclm_heldout was.

Candidates, in order: the current held-out docs, then --extra fresh docs drawn uniformly from corpus rows
used by neither the training subsets nor the held-out set (fetched from the Hub by row group). A candidate
is dropped if, in the training subsets or among earlier kept candidates,
  1. its URL, exact text, or normalized text (lowercased, whitespace-collapsed) occurs, or
  2. it has a near-duplicate: MinHash over word 5-gram shingle sets (128 hashes, LSH 32 bands x 4 rows),
     estimated Jaccard >= 0.7.
The first --n surviving candidates replace the val file (atomically); dataset.json records the counts.

    .venv-kd/bin/python -m setup.decontaminate_fineweb_edu_heldout --n 10000 --extra 6000
"""

import argparse
import glob
import hashlib
import json
import os
import re
import zlib
from multiprocessing import Pool

import numpy as np
import orjson
from huggingface_hub import HfFileSystem

from setup.extend_fineweb_edu_heldout import (EXTEND_SEED, OUT, corpus_meta, draw_unused, fetch_rows,
                                              original_draw)
from setup.prepare_fineweb_edu_subsets import HELDOUT, N_SUBSETS, SUBSET_ROWS

DECONTAM_SEED = EXTEND_SEED + 1
N_HASH, BANDS, ROWS, SHINGLE, THRESH = 128, 32, 4, 5, 0.7
P31 = (1 << 31) - 1
_rng = np.random.default_rng(12345)
A = _rng.integers(1, P31, size=N_HASH, dtype=np.uint64)
B = _rng.integers(0, P31, size=N_HASH, dtype=np.uint64)
MIX = np.uint64(0x9E3779B97F4A7C15)
WORD = re.compile(r"\w+")


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "little")


def exact_keys(rec):
    t = rec["text"]
    return h64(rec.get("url") or ""), h64(t), h64(" ".join(t.split()).lower())


def signature(text):
    """128 MinHash values over the doc's word 5-gram shingles (deterministic across processes)."""
    w = np.array([zlib.crc32(x.encode()) for x in WORD.findall(text.lower())], dtype=np.uint64)
    if len(w) < SHINGLE:
        sh = np.array([zlib.crc32(" ".join(WORD.findall(text.lower())).encode())], dtype=np.uint64)
    else:
        sh = np.zeros(len(w) - SHINGLE + 1, dtype=np.uint64)
        for k in range(SHINGLE):  # polynomial combine of 5 consecutive word hashes, mod 2^32
            sh = (sh * np.uint64(1000003) + w[k:len(w) - SHINGLE + 1 + k]) & np.uint64(0xFFFFFFFF)
        sh = np.unique(sh)
    return ((A[:, None] * sh[None, :] + B[:, None]) % np.uint64(P31)).min(axis=1)


def band_keys(sig):
    s = sig.reshape(BANDS, ROWS)
    k = np.zeros(BANDS, dtype=np.uint64)
    for r in range(ROWS):
        k = (k * MIX) ^ s[:, r]
    return (k * MIX) ^ np.arange(BANDS, dtype=np.uint64)


def _init(cand_exact, cand_bands, cand_sigs):
    global G
    G = (cand_exact, cand_bands, cand_sigs)


def scan(path):
    """Candidates hit by one training chunk: {cand idx: reason}."""
    cand_exact, cand_bands, cand_sigs = G
    hits = {}
    with open(path, "rb", buffering=1 << 24) as fh:
        for line in fh:
            rec = orjson.loads(line)
            for kind, key in zip(("url", "text", "norm_text"), exact_keys(rec)):
                for c in cand_exact[kind].get(key, ()):
                    hits.setdefault(c, f"exact {kind}")
            for bk in band_keys(sig := signature(rec["text"])).tolist():
                for c in cand_bands.get(bk, ()):
                    if c not in hits:
                        j = float((cand_sigs[c] == sig).mean())
                        if j >= THRESH:
                            hits[c] = f"near-dup J~{j:.2f}"
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--extra", type=int, default=6_000)
    ap.add_argument("--workers", type=int, default=120)
    a = ap.parse_args()
    val = f"{a.out}/{HELDOUT}/{HELDOUT}.val.jsonl"
    current = [orjson.loads(l) for l in open(val, "rb")]

    fs = HfFileSystem()
    metas, offsets, total = corpus_meta(fs)
    n_orig_held = 1000  # held-out rows that came from the original draw (the rest from the extension)
    used = set(original_draw(total, n_orig_held).tolist()) | {r["src_row"] for r in current}
    extra_rows = draw_unused(total, np.sort(np.fromiter(used, dtype=np.int64)), a.extra, DECONTAM_SEED)
    recs = fetch_rows(fs, metas, offsets, extra_rows)
    cands = current + [recs[r] for r in extra_rows]
    print(f"{len(cands):,} candidates ({len(current):,} current + {a.extra:,} fresh, seed {DECONTAM_SEED})", flush=True)

    cand_exact = {"url": {}, "text": {}, "norm_text": {}}
    cand_bands, sigs = {}, []
    for i, rec in enumerate(cands):
        for kind, key in zip(("url", "text", "norm_text"), exact_keys(rec)):
            cand_exact[kind].setdefault(key, []).append(i)
        sig = signature(rec["text"])
        sigs.append(sig)
        for bk in band_keys(sig).tolist():
            cand_bands.setdefault(bk, []).append(i)
    sigs = np.stack(sigs)

    train = sorted(glob.glob(f"{a.out}/fwedu_8m_*/*.chunk.*.jsonl"))
    assert len(train) == N_SUBSETS * 16
    reason = {}
    with Pool(a.workers, initializer=_init, initargs=(cand_exact, cand_bands, sigs)) as pool:
        for k, hits in enumerate(pool.imap_unordered(scan, train)):
            for c, r in hits.items():
                reason.setdefault(c, r)
            if k % 8 == 0:
                print(f"scanned {k + 1}/{len(train)} training chunks; {len(reason):,} candidates hit so far", flush=True)

    # Within the candidate list: drop docs duplicating an earlier kept candidate.
    kept, seen_exact = [], {"url": set(), "text": set(), "norm_text": set()}
    kept_bands = {}
    for i, rec in enumerate(cands):
        if i in reason:
            continue
        ks = dict(zip(("url", "text", "norm_text"), exact_keys(rec)))
        dup = next((f"self exact {k}" for k, v in ks.items() if v in seen_exact[k]), None)
        if dup is None:
            for bk in band_keys(sigs[i]).tolist():
                for c in kept_bands.get(bk, ()):
                    if (sigs[c] == sigs[i]).mean() >= THRESH:
                        dup = "self near-dup"
                        break
                if dup:
                    break
        if dup:
            reason[i] = dup
            continue
        kept.append(i)
        for k, v in ks.items():
            seen_exact[k].add(v)
        for bk in band_keys(sigs[i]).tolist():
            kept_bands.setdefault(bk, []).append(i)
        if len(kept) == a.n:
            break
    assert len(kept) == a.n, f"only {len(kept)} clean candidates; raise --extra"

    stats = {}
    for i in range(len(current)):
        if i in reason:
            key = reason[i].split(" J~")[0]
            stats[key] = stats.get(key, 0) + 1
    n_cur_kept = sum(1 for i in kept if i < len(current))
    print(f"current held-out: {len(current) - n_cur_kept:,} of {len(current):,} removed {stats}", flush=True)

    rows = [cands[i] for i in kept]
    tmp = val + ".tmp"
    with open(tmp, "wb") as f:
        for r in rows:
            f.write(orjson.dumps(r) + b"\n")
    os.replace(tmp, val)
    mpath = f"{a.out}/{HELDOUT}/dataset.json"
    meta = json.load(open(mpath))
    meta.update({
        "rows": a.n,
        "token_count_sum (dataset's own token_count field)": sum(r["token_count"] for r in rows),
        "decontamination": (f"vs all {N_SUBSETS} x {SUBSET_ROWS:,} training docs and itself: URL / exact text / normalized "
                            f"text, and MinHash (word {SHINGLE}-grams, {N_HASH} hashes, LSH {BANDS}x{ROWS}) Jaccard >= {THRESH}. "
                            f"{len(current) - n_cur_kept} of the previous {len(current)} docs removed {stats}; refilled from "
                            f"fresh uniform draws over unused rows (seed {DECONTAM_SEED}). "
                            "setup/decontaminate_fineweb_edu_heldout.py"),
    })
    json.dump(meta, open(mpath, "w"), indent=1)
    print(f"{HELDOUT}: {a.n:,} decontaminated docs written", flush=True)


if __name__ == "__main__":
    main()
