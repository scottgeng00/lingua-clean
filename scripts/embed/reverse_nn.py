"""Reverse nearest neighbours: for every chunk of a pretraining corpus, its nearest Paloma chunk (cosine), both
within each Paloma source separately and over all of Paloma pooled. Exact search on one GPU.
Output OUT/<corpus>/: <source>.idx.npy (int32, index into that source's chunks), <source>.sim.npy (float16), one pair
per Paloma source plus "pooled" (index into the concatenation of all sources in sorted order); paloma_sizes.json.

    .venv-kd/bin/python scripts/embed/reverse_nn.py --emb .../embeddings --corpus rw_gacha --out .../embeddings/reverse_nn
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=32768)
    a = ap.parse_args()
    out = f"{a.out}/{a.corpus}"
    os.makedirs(out, exist_ok=True)
    srcs, keys = [], []
    for d in sorted(glob.glob(f"{a.emb}/paloma/*/shard_0000.npy")):
        srcs.append(d.split("/")[-2])
        keys.append(np.load(d))
    sizes = [len(k) for k in keys]
    bounds = np.cumsum([0] + sizes)
    K = torch.from_numpy(np.concatenate(keys)).cuda()  # [170k, 1024] fp16
    json.dump(dict(zip(srcs, sizes)), open(f"{out}/paloma_sizes.json", "w"))
    files = sorted(glob.glob(f"{a.emb}/{a.corpus}/shard_????.npy"))
    n = sum(np.load(f, mmap_mode="r").shape[0] for f in files)
    names = srcs + ["pooled"]
    idx = {s: np.zeros(n, dtype=np.int32) for s in names}
    sim = {s: np.zeros(n, dtype=np.float16) for s in names}
    t0, row = time.time(), 0
    with torch.inference_mode():
        for f in files:
            E = torch.from_numpy(np.load(f)).cuda()
            for b in range(0, len(E), a.block):
                S = E[b:b + a.block] @ K.T  # [block, 170k]
                r = slice(row + b, row + b + len(S))
                v, i = S.max(1)
                idx["pooled"][r], sim["pooled"][r] = i.int().cpu().numpy(), v.cpu().numpy()
                for j, s in enumerate(srcs):
                    v, i = S[:, bounds[j]:bounds[j + 1]].max(1)
                    idx[s][r], sim[s][r] = i.int().cpu().numpy(), v.cpu().numpy()
            row += len(E)
            print(f"{row:,}/{n:,} chunks, {time.time() - t0:.0f}s", flush=True)
    for s in names:
        np.save(f"{out}/{s}.idx.npy", idx[s])
        np.save(f"{out}/{s}.sim.npy", sim[s])
    print(f"done {a.corpus}: {n:,} chunks in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
