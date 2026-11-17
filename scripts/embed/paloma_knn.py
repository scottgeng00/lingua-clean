"""Nearest neighbours of every Paloma chunk in each embedded corpus (scripts/embed/embed_tokens.py outputs), exact
cosine search on one GPU. For each corpus: the full corpus, and a random subset with the same number of chunks as the
smallest corpus (size-matched, seed 0), since the best-match similarity grows with corpus size.
Output: OUT/knn.npz with, per (corpus, variant): top-K similarities [n_paloma_chunks, K] and neighbour chunk indices;
plus the Paloma chunk source names and doc ids.

    .venv-kd/bin/python scripts/embed/paloma_knn.py --emb /checkpoint/mnm/sgeng/pretrain/embeddings \
        --corpora dclm_2x_w1152 rw_gacha rw_direct --out .../embeddings/analysis
"""
import argparse
import glob
import os
import time

import numpy as np
import torch


def load_corpus(path):
    files = sorted(glob.glob(f"{path}/shard_????.npy"))
    return np.concatenate([np.load(f) for f in files])


def topk(q, base, k, block=2_000_000, qblock=8192):
    """q [nq, d], base [n, d] (both fp16 on GPU) -> (sims [nq, k] float32, idx [nq, k] int64)."""
    sims = torch.full((len(q), k), -2.0, device=q.device)
    idx = torch.zeros((len(q), k), dtype=torch.long, device=q.device)
    for b0 in range(0, len(base), block):
        bb = base[b0:b0 + block]
        for q0 in range(0, len(q), qblock):
            s = q[q0:q0 + qblock] @ bb.T
            v, i = s.float().topk(min(k, s.shape[1]), dim=1)
            allv = torch.cat([sims[q0:q0 + qblock], v], 1)
            alli = torch.cat([idx[q0:q0 + qblock], i + b0], 1)
            vv, j = allv.topk(k, dim=1)
            sims[q0:q0 + qblock], idx[q0:q0 + qblock] = vv, alli.gather(1, j)
    return sims.cpu().numpy(), idx.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--corpora", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    srcs, qs, qdoc = [], [], []
    for d in sorted(glob.glob(f"{a.emb}/paloma/*/shard_0000.npy")):
        e = np.load(d)
        src = d.split("/")[-2]
        qs.append(e); srcs += [src] * len(e); qdoc.append(np.load(d.replace(".npy", ".meta.npz"))["doc"])
    q = torch.from_numpy(np.concatenate(qs)).cuda()
    print(f"{len(q):,} Paloma chunks from {len(set(srcs))} sources", flush=True)
    sizes = {c: sum(np.load(f, mmap_mode="r").shape[0] for f in glob.glob(f"{a.emb}/{c}/shard_????.npy"))
             for c in a.corpora}
    n_match = min(sizes.values())
    print("corpus chunks:", sizes, "size-matched to", n_match, flush=True)
    out = {"paloma_source": np.array(srcs), "paloma_doc": np.concatenate(qdoc)}
    for c in a.corpora:
        t0 = time.time()
        base = torch.from_numpy(load_corpus(f"{a.emb}/{c}")).cuda()
        s, i = topk(q, base, a.k)
        out[f"{c}/full/sim"], out[f"{c}/full/idx"] = s, i
        sub = np.sort(np.random.default_rng(0).choice(len(base), n_match, replace=False))
        s, i = topk(q, base[torch.from_numpy(sub).cuda()], a.k)
        out[f"{c}/matched/sim"], out[f"{c}/matched/idx"] = s, sub[i]
        print(f"{c}: {len(base):,} chunks, top-1 sim mean full {out[f'{c}/full/sim'][:, 0].mean():.4f} "
              f"matched {s[:, 0].mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
        del base
        torch.cuda.empty_cache()
    np.savez(f"{a.out}/knn.npz", **out)
    print("wrote", f"{a.out}/knn.npz")


if __name__ == "__main__":
    main()
