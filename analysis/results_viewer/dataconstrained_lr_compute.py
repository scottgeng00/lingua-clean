"""Optimal peak LR scaling for the stage 1 (unique data, Chinchilla 1x and 2x, wd 0.1) runs of the data-constrained
ladder: LR* vs compute (LR* = A C^b) and as a function of model size and tokens (LR* = A N^alpha D^beta, N = body
params, D = tokens).

Optimal LR per (width, eval point, batch) = minimum of a parabola in log LR through the grid minimum and its two
neighbours (dataconstrained_scaling.lr_opt), taken at the best batch. Compute as in dataconstrained_compute.py.
Logs to project lingua-clean-dataconstrained, run "lr-scaling".

    source scripts/env.sh && .venv-kd/bin/python analysis/results_viewer/dataconstrained_lr_compute.py [--no-wandb]
"""
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

NO_WANDB = "--no-wandb" in sys.argv
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataconstrained_compute as DC  # noqa: E402
import dataconstrained_scaling as DS  # noqa: E402

POINTS = [("d1x", "1x", "o", 20), ("2x", "2x", "s", 40)]  # eval point, label, marker, tokens per body param
WD = 0.1
COLORS = {512: "#2a78d6", 768: "#eb6834", 1152: "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
FIT, FIT2 = "#c0392b", "#8a4fd0"
OUT = "/checkpoint/mnm/sgeng/pretrain/logs/reports/dataconstrained_lr_scaling.png"
LR_TICKS = [1e-3, 1.5e-3, 2e-3, 3e-3, 4e-3, 6e-3]


def collect():
    rows = []
    for pt, tag, marker, ratio in POINTS:
        for w in DS.WIDTHS:
            g = DS.grid(w, 1, pt, WD)
            per_bs = {bs: DS.lr_opt(c) for bs, c in g.items() if len(c) >= 3}
            bs = min(per_bs, key=lambda b: per_bs[b][1])
            e = next(e for e in DS.G.M.values() if e["width"] == w and e["stage"] == 1 and e["eval_point"] == pt)
            rows.append(dict(w=w, tag=tag, marker=marker, ratio=ratio, N=DS.BODY[w], D=e["tokens"],
                             c=DC.flops(w, e["tokens"]), bs=bs, lr=per_bs[bs][0], edge=per_bs[bs][2],
                             loss=per_bs[bs][1]))
    return rows


def style(ax, xlabel):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks(LR_TICKS)
    ax.set_yticklabels([f"{t:g}" for t in LR_TICKS])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(7e-4, 6e-3)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel("optimal peak LR (log scale)", color=MUTED, fontsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    rows = collect()
    assert not any(r["edge"] for r in rows), "an optimum is at the grid edge"
    lr = np.log([r["lr"] for r in rows])
    # LR* = A C^b
    b, logA = np.polyfit(np.log([r["c"] for r in rows]), lr, 1)
    res_c = lr - (logA + b * np.log([r["c"] for r in rows]))
    # LR* = A N^alpha D^beta (least squares in log space)
    X = np.column_stack([np.ones(len(rows)), np.log([r["N"] for r in rows]), np.log([float(r["D"]) for r in rows])])
    (k0, alpha, beta), *_ = np.linalg.lstsq(X, lr, rcond=None)
    res_nd = lr - X @ np.array([k0, alpha, beta])
    rms = lambda r: math.exp(np.sqrt(np.mean(r ** 2)))

    print("| width | body N | tokens | D | compute | best batch | LR* | val loss |")
    print("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: r["c"]):
        print(f"| {r['w']} | {r['N'] / 1e6:.1f}M | {r['tag']} | {r['D'] / 1e9:.2f}B | {r['c']:.2e} | {r['bs']} | "
              f"{r['lr']:.2e} | {r['loss']:.4f} |")
    print(f"\nLR* = {math.exp(logA):.3g} * C^{b:.3f}                   rms error x{rms(res_c):.3f}")
    print(f"LR* = {math.exp(k0):.3g} * N^{alpha:.3f} * D^{beta:.3f}     rms error x{rms(res_nd):.3f}")
    print(f"  equivalently at fixed tokens/param r = D/N: LR* ∝ N^{alpha + beta:.3f} * r^{beta:.3f}")
    for w in (1536, 2048):
        N = DS.MC.shape(w)["body"]
        print(f"  predicted w{w} (body {N / 1e6:.0f}M): 1x {math.exp(k0) * N ** alpha * (20 * N) ** beta:.2e}, "
              f"2x {math.exp(k0) * N ** alpha * (40 * N) ** beta:.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    # Panel 1: vs compute
    ax = axes[0]
    xx = np.geomspace(min(r["c"] for r in rows) * 0.6, 1e21, 100)
    ax.plot(xx, math.exp(logA) * xx ** b, color=FIT, linewidth=3, alpha=0.9, zorder=2,
            label=f"LR* = {math.exp(logA):.3g} · C^{b:.3f}  (rms error ×{rms(res_c):.2f})")
    # Panel 2: vs N, lines of constant tokens/param from the N, D fit
    ax2 = axes[1]
    nn = np.geomspace(min(r["N"] for r in rows) * 0.7, 2e9, 100)
    for (pt, tag, marker, ratio), ls in zip(POINTS, ("-", "--")):
        ax2.plot(nn, math.exp(k0) * nn ** alpha * (ratio * nn) ** beta, color=FIT, linewidth=2.5, linestyle=ls,
                 alpha=0.9, zorder=2, label=f"fit at D = {ratio}N ({tag} Chinchilla)")
    for r in rows:
        for a, x in ((ax, r["c"]), (ax2, r["N"])):
            a.scatter([x], [r["lr"]], s=110, marker=r["marker"], color=COLORS[r["w"]], edgecolors="white",
                      linewidths=1.2, zorder=4)
            a.annotate(f"w{r['w']} {r['tag']} (bs {r['bs']}): {r['lr']:.2e}", (x, r["lr"]),
                       xytext=(8, 6 if r["tag"] == "1x" else -12), textcoords="offset points", fontsize=8, color=INK)
    for w in (1536, 2048):  # Delphi rungs 6, 7: where the fit points next
        N = DS.MC.shape(w)["body"]
        for ratio, m in ((20, "o"), (40, "s")):
            ax2.scatter([N], [math.exp(k0) * N ** alpha * (ratio * N) ** beta], s=60, marker=m, facecolors="none",
                        edgecolors=MUTED, linewidths=1.2, zorder=3)
        ax2.annotate(f"w{w} (predicted)", (N, math.exp(k0) * N ** alpha * (20 * N) ** beta), xytext=(6, 6),
                     textcoords="offset points", fontsize=8, color=MUTED)
    for a in axes:
        for w in DS.WIDTHS:
            a.scatter([], [], s=70, color=COLORS[w], label=f"width {w}")
        for _, tag, m, _ in POINTS:
            a.scatter([], [], s=70, marker=m, color=MUTED, label=f"{tag} Chinchilla tokens")
        a.legend(fontsize=8, frameon=False, loc="lower left")
    style(ax, "training compute C, FLOPs = 6 x (body + output head params) x tokens (log scale)")
    style(ax2, "body parameters N (log scale)")
    ax.set_title("optimal LR vs compute", color=INK, fontsize=10, loc="left")
    ax2.set_title(f"LR*(N, D) = {math.exp(k0):.3g} · N^{alpha:.3f} · D^{beta:.3f}  (rms error ×{rms(res_nd):.2f})",
                  color=INK, fontsize=10, loc="left")
    fig.suptitle("Optimal LR scaling, stage 1 unique-data runs only (1x and 2x Chinchilla, wd 0.1; parabola fit in "
                 "log LR at the best batch)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    if not NO_WANDB:
        import wandb
        run = wandb.init(project="lingua-clean-dataconstrained", name="lr-scaling", id="lr-scaling", resume="allow",
                         job_type="analysis")
        run.log({"lr_scaling_png": wandb.Image(OUT), "lr_opt": wandb.Table(
            columns=["width", "tokens", "body_params", "D", "flops", "best_batch", "lr_opt", "val_loss"],
            data=[[r["w"], r["tag"], r["N"], r["D"], r["c"], r["bs"], r["lr"], r["loss"]] for r in rows])})
        print(run.url)
        run.finish()


if __name__ == "__main__":
    main()
