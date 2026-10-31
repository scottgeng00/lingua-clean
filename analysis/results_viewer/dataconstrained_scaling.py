"""Scaling of the data-constrained ladder sweeps across widths (body params): optimal LR, optimal batch and best loss
per evaluation point, plus how much loss varies across configs.

Optimal LR per (width, eval point, batch) is the minimum of a parabola in log LR through the grid minimum and its two
neighbours (the grid argmin, flagged, when the minimum is at the grid edge). The width's optimum is taken at its best
batch. Stage 1 uses wd 0.1 (the full grid), stage 2 wd 0.3. Logs to project lingua-clean-dataconstrained, run "scaling".

    source scripts/env.sh && .venv-kd/bin/python analysis/results_viewer/dataconstrained_scaling.py [--no-wandb]
"""
import math
import os
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NO_WANDB = "--no-wandb" in sys.argv
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../setup"))
import dataconstrained_grid as G  # noqa: E402
import make_dataconstrained_configs as MC  # noqa: E402

WIDTHS = [512, 768, 1152]
BODY = {w: MC.shape(w)["body"] for w in WIDTHS}
POINTS = [(1, "d1x", 0.1, "unique 1x"), (1, "2x", 0.1, "unique 2x"), (2, "p4", 0.3, "repeat pass 4"),
          (2, "p16", 0.3, "repeat pass 16")]
COLORS = {"unique 1x": "#2a78d6", "unique 2x": "#eb6834", "repeat pass 4": "#1baf7a", "repeat pass 16": "#eda100"}
MARKERS = {"unique 1x": "o", "unique 2x": "s", "repeat pass 4": "^", "repeat pass 16": "D"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
OUT = "/checkpoint/mnm/sgeng/pretrain/logs/reports/dataconstrained_scaling.png"


def grid(width, stage, pt, wd, metric="val_ce"):
    """{bs: {lr: loss}} for seed 0 sweep runs."""
    out = defaultdict(dict)
    for n, e in G.M.items():
        if re.match(rf"w{width}_s{stage}_bs", n) and e["eval_point"] == pt and e["wd"] == wd and e["seed"] == 0:
            v = G.metric(n, metric)
            if v is not None:
                out[e["bs"]][e["lr"]] = v
    return out


def lr_opt(curve):
    """(optimal lr, loss at optimum, at_edge) from {lr: loss}."""
    lrs = sorted(curve)
    i = min(range(len(lrs)), key=lambda j: curve[lrs[j]])
    if i == 0 or i == len(lrs) - 1 or len(lrs) < 3:
        return lrs[i], curve[lrs[i]], True
    x = np.log10(lrs[i - 1:i + 2])
    y = np.array([curve[l] for l in lrs[i - 1:i + 2]])
    a, b, c = np.polyfit(x, y, 2)
    if a <= 0:
        return lrs[i], curve[lrs[i]], False
    xs = -b / (2 * a)
    return 10 ** xs, c - b * b / (4 * a), False


def fit(xs, ys):
    """Power law y = A x^k by least squares in log space."""
    k, logA = np.polyfit(np.log(xs), np.log(ys), 1)
    return k, math.exp(logA)


def main():
    res = {}  # (label, width) -> dict
    print("## Optimal LR per batch (parabola in log LR; * = grid edge)\n")
    for stage, pt, wd, label in POINTS:
        for w in WIDTHS:
            g = grid(w, stage, pt, wd)
            per_bs = {bs: lr_opt(c) for bs, c in g.items() if len(c) >= 3}
            if not per_bs:
                continue
            bs_best = min(per_bs, key=lambda b: per_bs[b][1])
            lr, loss, edge = per_bs[bs_best]
            vals = sorted(v for c in g.values() for v in c.values())
            res[(label, w)] = dict(lr=lr, loss=loss, edge=edge, bs=bs_best, vals=vals, per_bs=per_bs, grid=g)
            print(f"{label:15s} w{w:<5d} " + "  ".join(
                f"bs{b}: {per_bs[b][0]:.2e}{'*' if per_bs[b][2] else ''} ({per_bs[b][1]:.4f})" for b in sorted(per_bs)))

    print("\n## Variation across configs (val loss)\n")
    print("| point | width | best (fit) | best run | 2nd | 5th | median | configs | LR ÷1.5 / ×1.5 | batch ÷2 or ×2 (best) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for stage, pt, wd, label in POINTS:
        for w in WIDTHS:
            r = res.get((label, w))
            if not r:
                continue
            v, g, bsb = r["vals"], r["grid"], r["bs"]
            c = g[bsb]
            lrs = np.log10(sorted(c))
            ys = np.array([c[l] for l in sorted(c)])
            # interpolated loss at optimal LR / 1.5 and x 1.5 along the best batch's curve
            off = [np.interp(math.log10(r["lr"] * f), lrs, ys, left=np.nan, right=np.nan) for f in (1 / 1.5, 1.5)]
            off_s = " / ".join("-" if np.isnan(o) else f"+{o - r['loss']:.3f}" for o in off)
            nb = [r["per_bs"][b][1] for b in (bsb // 2, bsb * 2) if b in r["per_bs"]]
            print(f"| {label} | {w} | {r['loss']:.4f} | {v[0]:.4f} | {v[1]:.4f} | {v[4]:.4f} | {np.median(v):.4f} | "
                  f"{len(v)} | {off_s} | "
                  f"{'+%.3f' % (min(nb) - r['loss']) if nb else '-'} |")

    print("\n## Power-law fits vs body params (exponent k: y ∝ N^k)\n")
    fits = {}
    for _, _, _, label in POINTS:
        pts = [(BODY[w], res[(label, w)]["lr"]) for w in WIDTHS if (label, w) in res]
        if len(pts) == 3:
            k, A = fit(*zip(*pts))
            fits[label] = (k, A)
            print(f"{label:15s} LR_opt ∝ N^{k:+.2f}   (" + ", ".join(f"w{w}: {res[(label, w)]['lr']:.2e}" for w in WIDTHS)
                  + f")  best batch: " + "/".join(str(res[(label, w)]["bs"]) for w in WIDTHS))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for _, _, _, label in POINTS:
        ws = [w for w in WIDTHS if (label, w) in res]
        n = [BODY[w] for w in ws]
        style = dict(color=COLORS[label], marker=MARKERS[label], markersize=8, linewidth=2, markeredgecolor="white",
                     markeredgewidth=1.2, label=label, zorder=3)
        axes[0].plot(n, [res[(label, w)]["lr"] for w in ws], **style)
        if label in fits:
            k, A = fits[label]
            xx = np.geomspace(min(n), max(n), 20)
            axes[0].plot(xx, A * xx ** k, color=COLORS[label], linewidth=1, linestyle=":", zorder=2)
            axes[0].annotate(f"N^{k:+.2f}", (n[-1], res[(label, ws[-1])]["lr"]), xytext=(6, 0),
                             textcoords="offset points", fontsize=8, color=INK, va="center")
        for w in ws:
            if res[(label, w)]["edge"]:
                axes[0].scatter([BODY[w]], [res[(label, w)]["lr"]], s=150, facecolors="none", edgecolors=INK, zorder=4)
        axes[1].plot(n, [res[(label, w)]["bs"] * MC.SEQ for w in ws], **style)
        axes[2].plot(n, [res[(label, w)]["loss"] for w in ws], **style)
    titles = ["optimal peak LR (parabola fit at best batch)", "best batch (tokens per step)",
              "best dclm_heldout val loss (fit)"]
    for ax, t in zip(axes, titles):
        ax.set_xscale("log")
        ax.set_xticks([BODY[w] for w in WIDTHS])
        ax.set_xticklabels([f"w{w}\n{BODY[w] / 1e6:.0f}M" for w in WIDTHS])
        ax.minorticks_off()
        ax.set_xlabel("body parameters (log scale)", color=MUTED, fontsize=9)
        ax.set_title(t, color=INK, fontsize=10, loc="left")
        ax.grid(color=GRID, linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log", base=2)
    axes[1].set_yticks([262144, 524288])
    axes[1].set_yticklabels(["262k (64 seq)", "524k (128 seq)"])
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Data-constrained ladder: hyperparameter scaling (stage 1 wd 0.1, stage 2 wd 0.3; dotted = power-law "
                 "fit; circled = optimum at grid edge)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    if not NO_WANDB:
        import wandb
        run = wandb.init(project="lingua-clean-dataconstrained", name="scaling", id="scaling", resume="allow",
                         job_type="analysis")
        rows = [[label, w, BODY[w], r["lr"], r["edge"], r["bs"], r["loss"]] for (label, w), r in res.items()]
        run.log({"scaling_png": wandb.Image(OUT), "optima": wandb.Table(
            columns=["point", "width", "body_params", "lr_opt", "at_edge", "best_batch", "best_val"], data=rows)})
        print(run.url)
        run.finish()


if __name__ == "__main__":
    main()
