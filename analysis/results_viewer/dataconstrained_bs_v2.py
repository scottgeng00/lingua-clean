"""Batch-size plots for ladder data_constrainedv2 (unique DCLM data), dclm_heldout val loss.

For every (width, Chinchilla multiple, batch) the loss is the best over LR and wd (parabola vertex in log LR when
bracketed and not extrapolated below the best point, else the best measured point). Figures:
  1. loss penalty vs batch (tokens per step), one subplot per width, colour = Chinchilla multiple;
  2. the same, one subplot per multiple, colour = width;
  3. best batch vs training tokens D (all settings), with ties (within 0.003) as error bars, our power-law fit and
     Step Law's B = 0.58 D^0.571.

    .venv-kd/bin/python analysis/results_viewer/dataconstrained_bs_v2.py [--out temp/bs_v2] [--wandb]
"""
import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

from dataconstrained_lr_v2 import COLORS, MARKERS, MULTS, ROOT, WIDTHS, load, vertex

SEQ = 4096
BODY_HEAD = {512: 102_965_248, 768: 192_199_424, 1152: 429_866_112, 1536: 837_408_256}
TIE = 0.003


def per_batch(data):
    out = defaultdict(dict)  # (w, mult) -> bs -> (loss, lr, wd)
    for (w, m, bs, wd), g in data.items():
        if len(g) < 3:
            continue
        lo = min(g.values())
        v = vertex(g)
        val, lr = (v[1], v[0]) if v and v[1] >= lo - 0.003 else (lo, min(g, key=g.get))
        if bs not in out[(w, m)] or val < out[(w, m)][bs][0]:
            out[(w, m)][bs] = (val, lr, wd)
    return out


def tok_fmt(x, _):
    return f"bs{round(x / SEQ)}\n{x / 2**20:.2g}M tok"


def penalty_fig(B, by, out):
    panels, series = (WIDTHS, MULTS) if by == "width" else (MULTS, WIDTHS)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), sharey=True)
    for ax, p in zip(axes, panels):
        for i, s in enumerate(series):
            k = (p, s) if by == "width" else (s, p)
            if k not in B or len(B[k]) < 2:
                continue
            best = min(v[0] for v in B[k].values())
            xs = sorted(B[k])
            lbl = f"{k[1]}x (D={k[1] * 20 * BODY_HEAD[k[0]] / 1e9:.1f}B)" if by == "width" else f"w{k[0]}"
            ax.plot([b * SEQ for b in xs], [B[k][b][0] - best for b in xs], color=COLORS[i], marker=MARKERS[i],
                    ms=7, lw=2, label=lbl)
            bb = min(xs, key=lambda b: B[k][b][0])
            ax.plot([bb * SEQ], [0], marker="*", ms=16, color=COLORS[i], mec="white", mew=1.2, zorder=5)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(tok_fmt))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([b * SEQ for b in (32, 64, 128, 256, 512)])
        ax.axhline(TIE, color="#b9b8b0", lw=1, ls="--")
        ax.set_ylim(-0.004, 0.08)
        ax.set_title(f"width {p}" if by == "width" else f"{p}x Chinchilla", fontsize=12)
        ax.set_xlabel("batch (tokens / step)")
        ax.grid(True, which="major", color="#e6e6e3", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, title="Chinchilla mult." if by == "width" else "width",
                  title_fontsize=8)
    axes[0].set_ylabel("val loss − best batch's loss (nats/token)")
    fig.suptitle("Batch size: loss penalty at each batch's best LR and wd (★ = best batch; dashed = 0.003 tie band)",
                 fontsize=12)
    fig.tight_layout()
    path = f"{out}_penalty_by_{by}.png"
    fig.savefig(path, dpi=130)
    return path


def optimum_fig(B, out):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs, ys = [], []
    for i, w in enumerate(WIDTHS):
        for m in MULTS:
            k = (w, m)
            if k not in B or len(B[k]) < 2:
                continue
            best = min(v[0] for v in B[k].values())
            ties = [b * SEQ for b, v in B[k].items() if v[0] - best <= TIE]
            bb = min(B[k], key=lambda b: B[k][b][0]) * SEQ
            D = m * 20 * BODY_HEAD[w]
            ax.errorbar([D], [bb], yerr=[[bb - min(ties)], [max(ties) - bb]], color=COLORS[i], marker=MARKERS[i],
                        ms=8, lw=1.5, capsize=3, label=f"w{w}" if m == min(mm for (ww, mm) in B if ww == w) else None)
            xs.append(np.log(D)); ys.append(np.log(bb))
    b, a = np.polyfit(xs, ys, 1)
    D = np.logspace(9.2, 11, 50)
    ax.plot(D, np.exp(a) * D ** b, color="#52514e", lw=2, label=f"our fit: {np.exp(a):.3g}·D^{b:.2f}")
    ax.plot(D, 0.58 * D ** 0.571, color="#52514e", lw=2, ls="--", label="Step Law: 0.58·D^0.571")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(tok_fmt)); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([b * SEQ for b in (32, 64, 128, 256, 512)])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e9:g}B"))
    ax.set_xlabel("training tokens D")
    ax.set_ylabel("best batch (tokens / step)")
    ax.set_title("Best batch vs training tokens (bars: batches within 0.003 of best)", fontsize=12)
    ax.grid(True, which="major", color="#e6e6e3", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    path = f"{out}_optimum_vs_D.png"
    fig.savefig(path, dpi=130)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/temp/bs_v2")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    B = per_batch(load())
    paths = [penalty_fig(B, "width", a.out), penalty_fig(B, "mult", a.out), optimum_fig(B, a.out)]
    print("\n".join(paths))
    if a.wandb:
        import wandb
        run = wandb.init(project="lingua-clean-data_constrainedv2", name="batch-size", id="batch-size", resume="allow")
        run.log({os.path.basename(p)[:-4]: wandb.Image(p) for p in paths})
        run.finish()
