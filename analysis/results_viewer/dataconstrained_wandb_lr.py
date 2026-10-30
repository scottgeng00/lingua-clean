"""LR curves of the data-constrained ladder on wandb: val loss and Paloma BPB vs learning rate, one curve per batch
size (seed 0), per width. Columns: stage 1 at 1x and 2x Chinchilla (wd 0.1), stage 2 at pass 4 and pass 16 (wd 0.3).

Logs to project lingua-clean-dataconstrained, run "lr-curves": a rendered PNG per width and a table per panel.

    source scripts/env.sh && .venv-kd/bin/python analysis/results_viewer/dataconstrained_wandb_lr.py
"""
import os
import re
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import wandb  # noqa: E402

sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataconstrained_grid as G  # noqa: E402

# Fixed slot per batch size, shared by both widths.
COLORS = {8: "#2a78d6", 16: "#eb6834", 32: "#1baf7a", 64: "#eda100", 128: "#e87ba4", 256: "#008300"}
MARKERS = {8: "o", 16: "s", 32: "^", 64: "D", 128: "v", 256: "P"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
PANELS = [(1, "d1x", 0.1, "stage 1, 1x Chinchilla"), (1, "2x", 0.1, "stage 1, 2x Chinchilla"),
          (2, "p4", 0.3, "stage 2, pass 4"), (2, "p16", 0.3, "stage 2, pass 16")]
LABEL = {"val_ce": "dclm_heldout val loss (nats/token)", "paloma": "Paloma BPB (mean of 16)"}
OUT = "/checkpoint/mnm/sgeng/pretrain/logs/reports/dataconstrained_lr_w{}.png"


def points(width, stage, pt, wd, metric):
    """{bs: [(lr, value)]} for seed 0 sweep runs (not the final / extension runs)."""
    out = {}
    for n, e in G.M.items():
        if (re.match(rf"w{width}_s{stage}_bs", n) and e["eval_point"] == pt and e["wd"] == wd and e["seed"] == 0):
            v = G.metric(n, metric)
            if v is not None:
                out.setdefault(e["bs"], []).append((e["lr"], v))
    return {b: sorted(p) for b, p in sorted(out.items())}


def main():
    run = wandb.init(project="lingua-clean-dataconstrained", name="lr-curves", id="lr-curves", resume="allow",
                     job_type="analysis")
    log = {}
    for width in (512, 1152):
        fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
        for i, metric in enumerate(("val_ce", "paloma")):
            for j, (stage, pt, wd, title) in enumerate(PANELS):
                ax = axes[i, j]
                data = points(width, stage, pt, wd, metric)
                best = min(((v, lr, b) for b, p in data.items() for lr, v in p), default=None)
                for b, p in data.items():
                    xs, ys = zip(*p)
                    ax.plot(xs, ys, color=COLORS[b], marker=MARKERS[b], linewidth=2, markersize=7,
                            markeredgecolor="white", markeredgewidth=1.2, label=f"bs {b}", zorder=3)
                    ax.annotate(f"bs {b}", (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points", fontsize=8,
                                color=INK, va="center")
                if best:
                    ax.scatter([best[1]], [best[0]], s=180, facecolors="none", edgecolors=INK, linewidths=1.5, zorder=4)
                    ax.annotate(f"best {best[0]:.4f} (lr {best[1]:g}, bs {best[2]})", xy=(0.98, 0.04),
                                xycoords="axes fraction", ha="right", fontsize=8, color=INK)
                    lo = best[0]
                    ys_all = sorted(v for p in data.values() for _, v in p)
                    hi = ys_all[min(len(ys_all) - 1, len(ys_all) * 3 // 4)] + 0.02
                    ax.set_ylim(lo - 0.12 * (hi - lo), hi)
                ax.set_xscale("log")
                ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
                ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
                ax.set_title(f"{title} (wd {wd})", color=INK, fontsize=10, loc="left")
                ax.set_xlabel("learning rate (log scale)", color=MUTED, fontsize=9)
                if j == 0:
                    ax.set_ylabel(LABEL[metric], color=MUTED, fontsize=9)
                ax.grid(color=GRID, linewidth=0.8)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
                for s in ("left", "bottom"):
                    ax.spines[s].set_color(GRID)
                ax.tick_params(colors=MUTED, labelsize=8)
                log[f"w{width}_{metric}_{pt}_table"] = wandb.Table(
                    columns=["batch", "lr", metric], data=[[b, lr, v] for b, p in data.items() for lr, v in p])
        fig.suptitle(f"width {width}: LR sweep, one curve per batch size (seed 0; circled = best in panel; "
                     f"y-axis clipped at the 75th percentile, diverged high-LR points off-chart)", fontsize=11,
                     color=INK)
        handles = [plt.Line2D([], [], color=COLORS[b], marker=MARKERS[b], linewidth=2, markersize=7, label=f"batch {b}")
                   for b in sorted({b for b in COLORS if any(re.match(rf"w{width}_s\d_bs{b}_", n) for n in G.M)})]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(OUT.format(width), dpi=130)
        log[f"w{width}_lr_curves_png"] = wandb.Image(OUT.format(width))
    run.log(log)
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
