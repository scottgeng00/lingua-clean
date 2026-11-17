"""Step-by-step picture of the partial correlation in paloma_subset_corr.py (panel c), on the 102 fixed-compute
differences (FineWeb-Edu - DCLM twin, mix - real, KD - CE). One row per (Paloma source, downstream metric):
  1. raw: change in source BPB vs change in downstream metric;
  2. change in source BPB vs change in the mean BPB of the other 15 sources ("overall loss"), with the fitted line;
  3. same for the downstream metric;
  4. the leftovers (residuals) of 2 and 3 against each other: their correlation is the partial r.

    .venv-kd/bin/python analysis/results_viewer/partial_corr_explainer.py [--wandb]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from dataconstrained_lr_v2 import COLORS, ROOT
from paloma_subset_corr import SOURCES, load

PAIRS_NAMES = [("m2d2_s2orc_unsplit", "mmlu_bpb", "s2orc BPB", "MMLU BPB"),
               ("dolma_100_programing_languages", "he", "code BPB (Paloma)", "HumanEval BPB"),
               ("twitterAAE_HELM_fixed", "mmlu_bpb", "twitterAAE BPB", "MMLU BPB")]
GROUPS = {"fe": ("FineWeb-Edu − DCLM twin (93)", COLORS[0]), "mix": ("pt17b mix − real (4)", COLORS[1]),
          "kd": ("KD − CE (5)", COLORS[2])}


def differences(rows):
    pairs = [("fe", n.replace("_fe_", "_u_"), n) for n in rows if "_fe_" in n and n.replace("_fe_", "_u_") in rows]
    pairs += [("mix", "jhe/real", f"jhe/{m}") for m in ("mix_fs_gacha", "mix_fs_direct", "mix_rw_gacha", "mix_rw_direct")]
    pairs += [("kd", "kd/b_ce", f"kd/{r}") for r in ("b_kd0p1", "b_kd0p3", "b_kd0p5", "b_kd0p7", "b_kd0p9")]
    return pairs


def row(axes, rows, pairs, src, ev, sl, el):
    g = np.array([p[0] for p in pairs])
    dx = np.array([rows[b]["pal"][src] - rows[a]["pal"][src] for _, a, b in pairs])
    dy = np.array([rows[b][ev] - rows[a][ev] for _, a, b in pairs])
    dm = np.array([np.mean([rows[b]["pal"][s] - rows[a]["pal"][s] for s in SOURCES if s != src]) for _, a, b in pairs])
    fit = lambda v: np.polyval(np.polyfit(dm, v, 1), dm)
    rx, ry = dx - fit(dx), dy - fit(dy)

    def sc(ax, x, y, xl, yl, title, line=None):
        for k, (lbl, c) in GROUPS.items():
            m = g == k
            ax.scatter(x[m], y[m], s=26 if k == "fe" else 60, color=c, edgecolor="white", lw=0.6, zorder=3,
                       marker="o" if k == "fe" else ("s" if k == "mix" else "^"))
        if line is not None:
            o = np.argsort(x)
            ax.plot(x[o], line[o], color="#52514e", lw=1.5, zorder=2)
        ax.axhline(0, color="#d0cfc8", lw=0.8, zorder=1)
        ax.axvline(0, color="#d0cfc8", lw=0.8, zorder=1)
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, color="#eeeeea", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    sc(axes[0], dx, dy, f"Δ {sl}", f"Δ {el}", f"1. raw: r = {np.corrcoef(dx, dy)[0, 1]:+.2f}\n(mostly 'better overall')")
    sc(axes[1], dm, dx, "Δ overall loss (mean BPB, other 15 sources)", f"Δ {sl}",
       "2. how much of Δ" + f" {sl} overall loss explains", fit(dx))
    sc(axes[2], dm, dy, "Δ overall loss (mean BPB, other 15 sources)", f"Δ {el}",
       f"3. how much of Δ {el} overall loss explains", fit(dy))
    sc(axes[3], rx, ry, f"{sl}: leftover after removing overall loss", f"{el}: leftover",
       f"4. leftovers: partial r = {np.corrcoef(rx, ry)[0, 1]:+.2f}\n(specific to {sl.split(' ')[0]})")
    b = np.polyfit(rx, ry, 1)
    xs = np.linspace(rx.min(), rx.max(), 10)
    axes[3].plot(xs, np.polyval(b, xs), color="#52514e", lw=1.5, ls="--")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/temp/partial_corr_explainer.png")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    rows = load()
    pairs = differences(rows)
    fig, axes = plt.subplots(len(PAIRS_NAMES), 4, figsize=(22, 5.2 * len(PAIRS_NAMES)))
    for i, (s, e, sl, el) in enumerate(PAIRS_NAMES):
        row(axes[i], rows, pairs, s, e, sl, el)
    handles = [Line2D([], [], marker=m, ls="none", color=c, ms=8, label=l)
               for (l, c), m in zip(GROUPS.values(), ("o", "s", "^"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.suptitle("What 'overall loss accounted for' means: each point = one change at fixed compute (data or loss only). "
                 "Rows: s2orc→MMLU BPB (a real link), code→HumanEval (strong), twitterAAE→MMLU BPB (no link)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(a.out, dpi=110)
    print(a.out)
    if a.wandb:
        import wandb
        run = wandb.init(project="lingua-clean-data_constrainedv2", name="paloma-subset-corr", id="paloma-subset-corr",
                         resume="allow")
        run.log({"partial_corr_explainer": wandb.Image(a.out)})
        run.finish()
