"""Vendi score (cosine kernel) of each embedded corpus (scripts/embed/embed_tokens.py outputs): mean and std over
--reps random samples of --n documents (each document = its first chunk's embedding). With L2-normalized rows X,
VS = exp(H(eigenvalues of X^T X / n)), the same spectrum as the n x n kernel K / n.

    .venv-kd/bin/python scripts/embed/vendi.py --emb /home/sgeng/scottg/pretrain/embeddings --n 1024 --reps 50
"""
import argparse
import glob

import numpy as np


def vendi(x):
    x = x.astype(np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    ev = np.clip(np.linalg.eigvalsh(x.T @ x / len(x)), 0, None)
    ev = ev[ev > 1e-12] / ev.sum()
    return float(np.exp(-(ev * np.log(ev)).sum()))


def first_chunks(path):
    """memmapped shard arrays + (shard, row) of each document's first chunk"""
    arrs, rows = [], []
    for i, f in enumerate(sorted(glob.glob(f"{path}/shard_????.npy"))):
        arrs.append(np.load(f, mmap_mode="r"))
        ch = np.load(f.replace(".npy", ".meta.npz"))["chunk"]
        r = np.flatnonzero(ch == 0)
        rows.append(np.stack([np.full(len(r), i), r], 1))
    return arrs, np.concatenate(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--corpora", nargs="+", default=["dclm_2x_w1152", "rw_gacha", "rw_direct"])
    ap.add_argument("--n", type=int, nargs="+", default=[1024])
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    groups = [(c, f"{a.emb}/{c}") for c in a.corpora]
    pal = sorted(glob.glob(f"{a.emb}/paloma/*"))
    print("| corpus | docs | " + " | ".join(f"Vendi @ n={n} (mean ± std over {a.reps})" for n in a.n) + " |")
    print("|---|---|" + "---|" * len(a.n))
    for name, path in groups + [("paloma (pooled)", None)] + [(f"paloma/{p.split('/')[-1]}", p) for p in pal]:
        if path is None:  # pool all Paloma sources
            parts = [first_chunks(p) for p in pal]
            arrs = [x for ar, _ in parts for x in ar]
            off = np.cumsum([0] + [len(ar) for ar, _ in parts])
            rows = np.concatenate([r + [o, 0] for (_, r), o in zip(parts, off[:-1])])
        else:
            arrs, rows = first_chunks(path)
        cells = []
        for n in a.n:
            if len(rows) < n:
                cells.append("—")
                continue
            v = []
            for _ in range(a.reps):
                sel = rows[rng.choice(len(rows), n, replace=False)]
                v.append(vendi(np.stack([arrs[s][r] for s, r in sel])))
            cells.append(f"{np.mean(v):.1f} ± {np.std(v):.1f}")
        print(f"| {name} | {len(rows):,} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
