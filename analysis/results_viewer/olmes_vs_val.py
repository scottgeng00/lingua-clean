"""Downstream OLMES scores vs dclm_heldout val loss and vs Paloma BPB (mean of 16 sources) for every data_constrainedv2
run with an OLMES suite eval (scripts/launch_olmes_suite.py) and a Paloma score at its final step. Groups: QA error
(100 - mean acc of ARC-E/C, HellaSwag, PIQA, Winogrande, SciQ), MMLU error, MMLU BPB, math BPB (mean of 7 Minerva MATH
subjects), code BPB (mean of HumanEval, MBPP); plus Paloma vs dclm_heldout and MMLU error vs MMLU BPB.
Each panel: least-squares line over all points, Pearson r and Spearman rho.

    .venv-kd/bin/python analysis/results_viewer/olmes_vs_val.py [--out temp/olmes_vs_val.png] [--wandb]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from dataconstrained_lr_v2 import COLORS, ROOT, RUNS, WIDTHS

QA = ["arc_easy:rc::olmes:full", "arc_challenge:rc::olmes:full", "hellaswag:rc::olmes:full", "piqa:rc::olmes:full",
      "winogrande:rc::olmes:full", "sciq:rc::olmo3"]


def load():
    M = json.load(open(f"{ROOT}/apps/main/configs/experiments/data_constrainedv2/manifest.json"))
    rows = []
    for path in sorted(glob.glob(f"{RUNS}/*/evals/*_olmes/olmes_results/metrics.json")):
        name = path.split("/evals/")[0].rsplit("/", 1)[1]
        e = M[name]
        v = f"{RUNS}/{name}/evals/{e['steps']:010d}/validation.json"
        if not os.path.exists(v):
            continue
        pal = None
        for line in open(f"{RUNS}/{name}/metrics.paloma_np.jsonl"):
            r = json.loads(line)
            if r["global_step"] == e["steps"]:
                p = [v for k, v in r.items() if k.endswith("primary_score") and "/paloma::" not in k]
                pal = np.mean(p) if len(p) == 16 else None
        if pal is None:
            continue
        t = {x["alias"]: x["metrics"]["primary_score"] for x in json.load(open(path))["tasks"]}
        rows.append(dict(name=name, width=e["width"], rep="_r_" in name,
                         val=json.load(open(v))["dclm_heldout"]["ce_per_token"], pal=pal,
                         qa=100 - 100 * np.mean([t[a] for a in QA]), mmlu=100 - 100 * t["mmlu:rc::olmes"],
                         mmlu_bpb=t["mmlu:rc:bpb::olmes"], math=t["olmo3:base_easy:math_bpb"],
                         code=np.mean([t["codex_humaneval:3shot:bpb::none"], t["mbpp:3shot:bpb::none"]])))
    return rows


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return np.corrcoef(rx, ry)[0, 1]


def panel(ax, rows, xk, yk, xlabel, ylabel):
    x, y = np.array([r[xk] for r in rows]), np.array([r[yk] for r in rows])
    for r in rows:
        c = COLORS[WIDTHS.index(r["width"])]
        ax.plot(r[xk], r[yk], marker="^" if r["rep"] else "o", ms=8, ls="none",
                mfc="white" if r["rep"] else c, mec=c, mew=1.8, zorder=3)
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, a + b * xs, color="#52514e", lw=1.5, zorder=2)
    resid = np.std(y - (a + b * x))
    ax.set_title(f"{ylabel}\nr = {np.corrcoef(x, y)[0, 1]:.3f}, ρ = {spearman(x, y):.3f}, "
                 f"resid σ = {resid:.3g}", fontsize=10)
    ax.set_xlabel(xlabel)
    ax.grid(True, color="#e6e6e3", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    return np.corrcoef(x, y)[0, 1], spearman(x, y)
    ax.set_title(f"{ylabel}\nPearson r = {np.corrcoef(x, y)[0, 1]:.3f}, Spearman ρ = {spearman(x, y):.3f}, "
                 f"resid σ = {resid:.3g}", fontsize=10)
    ax.set_xlabel(xlabel)
    ax.grid(True, color="#e6e6e3", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/temp/olmes_vs_val.png")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    rows = load()
    groups = [("qa", "QA error % (ARC-E/C, HSwag, PIQA, Wino, SciQ)"), ("mmlu", "MMLU error % (5-shot RC)"),
              ("mmlu_bpb", "MMLU BPB (gold answer)"), ("math", "math BPB (Minerva MATH)"), ("code", "code BPB (HumanEval, MBPP)")]
    xs = [("val", "dclm_heldout val loss (nats/token)"), ("pal", "Paloma BPB (mean of 16)")]
    fig, axes = plt.subplots(3, 5, figsize=(26, 15))
    corr = {}
    for i, (xk, xl) in enumerate(xs):
        for j, (yk, yl) in enumerate(groups):
            corr[(xk, yk)] = panel(axes[i, j], rows, xk, yk, xl, yl)
    corr[("val", "pal")] = panel(axes[2, 0], rows, "val", "pal", xs[0][1], "Paloma BPB vs dclm_heldout")
    panel(axes[2, 1], rows, "mmlu_bpb", "mmlu", "MMLU BPB (gold answer)", "MMLU error % vs MMLU BPB")
    ax = axes[2, 2]
    ax.axis("off")
    cells = [[yl.split(" (")[0], f"{corr[('val', yk)][0]:.3f} / {corr[('val', yk)][1]:.3f}",
              f"{corr[('pal', yk)][0]:.3f} / {corr[('pal', yk)][1]:.3f}"] for yk, yl in groups]
    tb = ax.table(cellText=cells, colLabels=["eval", "vs dclm val: r / ρ", "vs Paloma: r / ρ"], loc="center",
                  cellLoc="center")
    tb.scale(1.25, 2.2)
    tb.set_fontsize(11)
    ax.set_title(f"Correlations (Paloma vs dclm val: r = {corr[('val', 'pal')][0]:.3f})", fontsize=11)
    for ax in axes[2, 3:]:
        ax.axis("off")
    handles = [Line2D([], [], marker="o", ls="none", color=COLORS[i], ms=8, label=f"w{w}") for i, w in enumerate(WIDTHS)]
    handles += [Line2D([], [], marker="o", ls="none", mfc="#888", mec="#888", ms=8, label="fresh data"),
                Line2D([], [], marker="^", ls="none", mfc="white", mec="#888", mew=1.8, ms=8, label="repetition (w768)")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=10)
    fig.suptitle(f"Downstream evals vs DCLM held-out loss (row 1) and Paloma BPB (row 2); {len(rows)} final checkpoints, "
                 "data_constrainedv2; line = least-squares fit", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(a.out, dpi=120)
    print(a.out)
    if a.wandb:
        import wandb
        run = wandb.init(project="lingua-clean-data_constrainedv2", name="olmes-vs-val", id="olmes-vs-val",
                         resume="allow")
        run.log({"olmes_vs_val": wandb.Image(a.out)})
        run.finish()
