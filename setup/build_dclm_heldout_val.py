"""Build a DCLM validation set with no duplicates in the training subsets.

dclm_shuffled/dclm.val.jsonl is disjoint from the training chunks by line, but DCLM has many
duplicate pages: ~15% of its 1000 docs share a URL or the exact text with docs inside each
8M-doc dclm_subsets/dclm_8m_k. This draws docs, in file order, from a chunk no subset uses
(default dclm.chunk.15.jsonl) and keeps the first N whose id (URL), exact text and normalized
text (lowercased, whitespace-collapsed) occur in none of the subsets and in no earlier kept doc.
Near-duplicates that differ after normalization are not caught.

Output: <data_root>/dclm_heldout/dclm_heldout.val.jsonl (+ dataset.json), picked up by the
validation eval as source "dclm_heldout". Also reports how much of the old val set overlaps.

    python setup/build_dclm_heldout_val.py
"""

import argparse
import glob
import hashlib
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import orjson

DATA = "/home/sgeng/scottg/pretrain/assets/data"


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "little")


def keys(rec):
    t = rec["text"]
    return h64(rec["id"]), h64(t), h64(" ".join(t.split()).lower())


def hash_file(path):
    """(id, text, norm_text) hashes of every row of one chunk file, as three uint64 arrays."""
    cols = ([], [], [])
    with open(path, "rb", buffering=1 << 24) as fh:
        for line in fh:
            for c, k in zip(cols, keys(orjson.loads(line))):
                c.append(k)
    return [np.array(c, dtype=np.uint64) for c in cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=sorted(glob.glob(f"{DATA}/dclm_subsets/dclm_8m_*")))
    ap.add_argument("--pool", default=f"{DATA}/dolmino_splits/dclm_shuffled/dclm.chunk.15.jsonl")
    ap.add_argument("--old-val", default=f"{DATA}/dolmino_splits/dclm_shuffled/dclm.val.jsonl")
    ap.add_argument("--out-dir", default=f"{DATA}/dolmino_splits/dclm_heldout")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    t0 = time.time()
    if os.path.exists(os.path.join(args.out_dir, "dclm_heldout.val.jsonl")):
        raise SystemExit(f"{args.out_dir} already has a val file")

    files = [f for sub in args.subsets for f in sorted(glob.glob(os.path.join(sub, "*.chunk.*.jsonl")))]
    cols = ([], [], [])
    with Pool(args.workers) as pool:
        for i, arrs in enumerate(pool.imap_unordered(hash_file, files)):
            for c, a in zip(cols, arrs):
                c.append(a)
            print(f"hashed {i + 1}/{len(files)} chunk files, {time.time() - t0:.0f}s", flush=True)
    rows = sum(len(a) for a in cols[0])
    train = [np.unique(np.concatenate(c)) for c in cols]
    del cols
    print(f"{rows:,} training rows hashed", flush=True)
    names = ("id", "text", "norm_text")

    def hits(k):  # train arrays are sorted (np.unique)
        out = []
        for x, t in zip(k, train):
            i = np.searchsorted(t, np.uint64(x))
            out.append(bool(i < len(t) and t[i] == x))
        return out

    old = [orjson.loads(l) for l in open(args.old_val, "rb")]
    old_hits = np.array([hits(keys(r)) for r in old])
    old_report = {n: int(old_hits[:, i].sum()) for i, n in enumerate(names)}
    old_report["any"] = int(old_hits.any(1).sum())
    print(f"old val ({len(old)} docs) overlapping the subsets: {old_report}", flush=True)

    kept, seen, scanned, rejected = [], (set(), set(), set()), 0, {n: 0 for n in names}
    rejected["dup_within_val"] = rejected["empty"] = 0
    with open(args.pool, "rb") as fh:
        for line in fh:
            scanned += 1
            rec = orjson.loads(line)
            if not rec.get("text", "").strip():
                rejected["empty"] += 1
                continue
            k = keys(rec)
            hk = hits(k)
            if any(hk):
                for n, h in zip(names, hk):
                    rejected[n] += h
                continue
            if any(x in s for x, s in zip(k, seen)):
                rejected["dup_within_val"] += 1
                continue
            for x, s in zip(k, seen):
                s.add(x)
            kept.append(line)
            if len(kept) == args.n:
                break
    if len(kept) < args.n:
        raise SystemExit(f"only {len(kept)} clean docs in {args.pool}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "dclm_heldout.val.jsonl"), "wb") as f:
        f.writelines(kept)
    json.dump({
        "description": f"First {args.n} docs of {args.pool} whose id, exact text and normalized text occur in "
                       "none of the training subsets (nor earlier in this file).",
        "subsets": args.subsets, "training_rows_checked": rows, "pool": args.pool,
        "pool_lines_scanned": scanned, "rejected": rejected,
        "old_val": args.old_val, "old_val_overlap_with_subsets": old_report,
        "elapsed_seconds": round(time.time() - t0),
    }, open(os.path.join(args.out_dir, "dataset.json"), "w"), indent=2)
    print(f"kept {len(kept)} of {scanned} scanned; rejected {rejected}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
