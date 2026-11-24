"""Why are rewrites of *different* source documents similar? For every cross-document pair in the Vendi batches
(analysis/multisample_vendi.py, same seed / documents / views), compare the rewrites' cosine with their sources'
cosine: high rewrite similarity with low source similarity = similarity added by the rewrite (format / template);
both high = the sources really are about the same thing.

Prints per arm: corr(rewrite cos, source cos), mean excess (rewrite - source), and for the top decile of rewrite
pairs their mean source cos and the share whose sources are also top-decile; then writes example pairs (at the
given percentiles of rewrite cos, plus the largest-excess pairs) with text excerpts to --out (markdown).

    .venv-kd/bin/python analysis/multisample_pairs.py --out /tmp/pairs.md
"""
import argparse
import random
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import multisample_vendi as mv  # noqa: E402

KINDS = ["article", "discussion", "faq", "math", "narrative", "table", "tutorial", "wiki"]


def replay(seed=0, n_chunks=48, groups=256, batches=5, batch_size=256):
    """The documents, batches and per-document view picks of multisample_vendi.py's main()."""
    rng = random.Random(seed)
    chunks = rng.sample(range(mv.POOL_DOCS // mv.CHUNK), n_chunks)
    views = {arm: mv.load_chunks(arm, chunks) for arm in mv.ARMS}
    docs = sorted(set.intersection(*[{r for r, v in views[arm].items() if len(v) == 8 and all(v)} for arm in mv.ARMS]))
    rng.sample(docs, groups)
    bs = [rng.sample(docs, batch_size) for _ in range(batches)]
    pick = {(b, r): rng.randrange(8) for b, rows in enumerate(bs) for r in rows}
    return views, bs, pick


def excerpt(t, n):
    t = " ".join(t.split())
    return t[:n] + (" …" if len(t) > n else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="analysis/multisample_vendi.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--percentiles", type=float, nargs="+", default=[50, 90, 99, 99.9])
    ap.add_argument("--chars", type=int, default=700)
    a = ap.parse_args()
    z = np.load(a.npz)
    E = z["E"].astype(np.float32)
    at = {(k.split("|")[0], int(k.split("|")[1]), int(k.split("|")[2])): i for i, k in enumerate(z["keys"])}
    views, batches, pick = replay()
    src = mv.source_texts({r for rows in batches for r in rows})

    def text(arm, r, v):
        return src[r] if arm == "real" else views[arm.split(":")[0]][r][v]

    # arm -> list of (b, rows, view per row); mix also per format (every document in that format)
    specs = {arm: [(b, rows, [-1 if arm == "real" else pick[b, r] for r in rows]) for b, rows in enumerate(batches)]
             for arm in mv.ARMS + ["real"]}
    for v, k in enumerate(KINDS):
        if k in ("math", "discussion", "wiki"):
            specs[f"mix:{k}"] = [(b, rows, [v] * len(rows)) for b, rows in enumerate(batches)]
    md = ["# Cross-document rewrite pairs\n",
          "Pairs of rewrites of *different* source documents from the 5 x 256 Vendi batches. `rw` = cosine between "
          "the two rewrites, `src` = cosine between their two source documents (Qwen3-Embedding-4B).\n"]
    print(f"{'arm':16s} {'corr':>6s} {'excess':>7s} {'top10% rw: src cos':>19s} {'src also top10%':>16s}")
    for arm, bl in specs.items():
        pairs = []  # (rw, src, (arm, r1, v1), (arm, r2, v2))
        for b, rows, vs in bl:
            X = np.stack([E[at[arm.split(":")[0], r, v]] for r, v in zip(rows, vs)])
            S = np.stack([E[at["real", r, -1]] for r in rows])
            K, Ks = X @ X.T, S @ S.T
            iu = np.triu_indices(len(rows), 1)
            for i, j, rw, s in zip(*iu, K[iu], Ks[iu]):
                pairs.append((float(rw), float(s), (rows[i], vs[i]), (rows[j], vs[j])))
        rw = np.array([p[0] for p in pairs])
        s = np.array([p[1] for p in pairs])
        top = rw >= np.quantile(rw, 0.9)
        print(f"{arm:16s} {np.corrcoef(rw, s)[0, 1]:6.3f} {np.mean(rw - s):+7.3f} {s[top].mean():19.3f} "
              f"{np.mean(s[top] >= np.quantile(s, 0.9)):16.2f}")
        md.append(f"\n## {arm}\n\ncorr(rw, src) = {np.corrcoef(rw, s)[0, 1]:.3f}; mean rw = {rw.mean():.3f}, "
                  f"mean src = {s.mean():.3f}; top-decile rw pairs: mean src = {s[top].mean():.3f}, "
                  f"{np.mean(s[top] >= np.quantile(s, 0.9)):.0%} have top-decile src.\n")
        order = np.argsort(rw)
        chosen = [(f"p{q:g} of rw", order[min(len(order) - 1, int(q / 100 * len(order)))]) for q in a.percentiles]
        chosen.append(("largest rw - src", int(np.argmax(rw - s))))
        for label, i in chosen:
            p = pairs[i]
            md.append(f"\n### {label}: rw {p[0]:.3f}, src {p[1]:.3f}\n")
            for n, (r, v) in enumerate([p[2], p[3]], 1):
                md.append(f"**rewrite {n}** (source row {r}, view {v}):\n\n> {excerpt(text(arm, r, v), a.chars)}\n")
                if arm != "real":
                    md.append(f"source {n}: _{excerpt(src[r], 250)}_\n")
    open(a.out, "w").write("\n".join(md))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
