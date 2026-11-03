"""Exact-deduplicate a pre-tokenized DCLM stream (setup/pretokenize_dclm_qwen3.py): keep a document only if it is the
first occurrence, in stream order, of both its exact text and its id (URL). Dolmino's DCLM has ~102M unique texts in
~611M docs, so a random draw repeats documents: 14.6% of dclm_8m_0..3 are exact-text duplicates of earlier docs.

Hashes come from the lingua jsonl subsets (global doc d = subset k row j, row j in chunk j % 16, line j // 16), so the
source stream must have been built from exactly --subsets in order. Output: tokens.bin, doc_offsets.npy (same format,
read by lingua/token_stream.py), kept_docs.npy (source doc index of every kept doc) and meta.json.

    .venv-kd/bin/python setup/dedup_token_stream.py --src .../dclm_qwen3_tokens_64m --out .../dclm_qwen3_tokens_64m_dedup
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import orjson

from setup.extend_dclm_subsets import id_hash, text_hash

DATA = "/home/sgeng/scottg/pretrain/assets/data/dclm_subsets"
N_CHUNKS = 16


def hash_chunk(task):
    s, c = task
    ids, texts = [], []
    with open(f"{DATA}/{s}/{s}.chunk.{c:02d}.jsonl", "rb", buffering=1 << 24) as f:
        for line in f:
            d = orjson.loads(line)
            ids.append(id_hash(d["id"]))
            texts.append(text_hash(d["text"]))
    return s, c, np.asarray(ids, dtype=np.uint64), np.asarray(texts, dtype=np.uint64)


def copy_runs(task):
    """Copy runs of consecutive kept docs: src tokens [s_lo, s_hi) -> dst tokens starting at d_lo."""
    src, dst, runs = task
    s = np.memmap(f"{src}/tokens.bin", dtype=np.uint32, mode="r")
    d = np.memmap(f"{dst}/tokens.bin", dtype=np.uint32, mode="r+")
    for s_lo, s_hi, d_lo in runs:
        d[d_lo:d_lo + (s_hi - s_lo)] = s[s_lo:s_hi]
    d.flush()
    return len(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=96)
    a = ap.parse_args()
    meta = json.load(open(f"{a.src}/meta.json"))
    subsets = list(meta["subsets"])
    src_off = np.load(f"{a.src}/doc_offsets.npy")
    n = len(src_off) - 1

    tasks = [(s, c) for s in subsets for c in range(N_CHUNKS)]
    hid, htx = np.zeros(n, dtype=np.uint64), np.zeros(n, dtype=np.uint64)
    with Pool(min(a.workers, len(tasks))) as pool:
        for s, c, i, t in pool.imap_unordered(hash_chunk, tasks):
            first, rows = meta["subsets"][s]["first_doc"], meta["subsets"][s]["docs"]
            d = first + c + N_CHUNKS * np.arange(len(i))
            assert len(i) == len(range(c, rows, N_CHUNKS)), (s, c)
            hid[d], htx[d] = i, t

    # First occurrence (lowest doc index) of each text and of each id; keep docs that are both.
    _, ft = np.unique(htx, return_index=True)
    _, fi = np.unique(hid, return_index=True)
    first_t, first_i = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    first_t[ft] = True
    first_i[fi] = True
    keep = first_t & first_i
    kept = np.flatnonzero(keep)
    lens = np.diff(src_off)[kept]
    off = np.concatenate([[0], np.cumsum(lens)])
    os.makedirs(a.out, exist_ok=True)
    np.save(f"{a.out}/doc_offsets.npy", off)
    np.save(f"{a.out}/kept_docs.npy", kept)
    np.memmap(f"{a.out}/tokens.bin", dtype=np.uint32, mode="w+", shape=(int(off[-1]),)).flush()

    # Runs of consecutive kept source docs are contiguous token spans in both streams.
    brk = np.flatnonzero(np.diff(kept) != 1) + 1
    starts, ends = np.concatenate([[0], brk]), np.concatenate([brk, [len(kept)]])
    runs = list(zip(src_off[kept[starts]].tolist(), src_off[kept[ends - 1] + 1].tolist(), off[starts].tolist()))
    batches = [runs[i::a.workers] for i in range(a.workers)]
    with Pool(a.workers) as pool:
        pool.map(copy_runs, [(a.src, a.out, b) for b in batches])

    # Spot check: 2000 random kept docs equal their source tokens.
    s_tok = np.memmap(f"{a.src}/tokens.bin", dtype=np.uint32, mode="r")
    d_tok = np.memmap(f"{a.out}/tokens.bin", dtype=np.uint32, mode="r")
    for j in np.random.default_rng(0).choice(len(kept), 2000, replace=False):
        k = kept[j]
        assert np.array_equal(s_tok[src_off[k]:src_off[k + 1]], d_tok[off[j]:off[j + 1]]), j

    per = {}
    for s in subsets:
        lo, hi = meta["subsets"][s]["first_doc"], meta["subsets"][s]["first_doc"] + meta["subsets"][s]["docs"]
        per[s] = {"source_docs": hi - lo, "kept_docs": int(keep[lo:hi].sum())}
    out_meta = {"source": a.src, "dedup": "keep first occurrence (stream order) of exact text and of id (URL)",
                "docs": int(len(kept)), "tokens": int(off[-1]), "source_docs": int(n), "source_tokens": int(src_off[-1]),
                "dropped_text_dups": int((~first_t).sum()), "dropped_id_only": int((first_t & ~first_i).sum()),
                "subsets": per, "tokenizer": meta["tokenizer"], "eos_id": meta["eos_id"], "dtype": "uint32",
                "script": "setup/dedup_token_stream.py"}
    json.dump(out_meta, open(f"{a.out}/meta.json", "w"), indent=1)
    print(json.dumps(out_meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
