"""Does held-out loss predict downstream scores when the training data changes? FineWeb-Edu runs (w{w}_fe_*) vs
their DCLM twins (w{w}_u_*: same width, horizon, batch, LR, wd) in data_constrainedv2, all with an OLMES suite eval.
Rows: eval vs DCLM held-out loss; eval vs Paloma BPB (mean of 16); per pair, (FineWeb-Edu - DCLM) eval difference vs
Paloma difference. One least-squares line per data source (rows 1-2) and the within-source / pooled Pearson r.

    .venv-kd/bin/python analysis/results_viewer/olmes_data_corr.py [--out temp/olmes_data_corr.png] [--wandb]
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

from dataconstrained_lr_v2 import COLORS, ROOT, RUNS
from olmes_vs_val import QA

DATA = {"dclm": ("DCLM", COLORS[0]), "fe": ("FineWeb-Edu", COLORS[1])}
MARK = {512: "o", 768: "s"}
EVALS = [("qa", "QA error %"), ("mmlu", "MMLU error %"), ("mmlu_bpb", "MMLU BPB"), ("math", "math BPB (Minerva)"),
         ("code", "code BPB (HumanEval, MBPP)")]


def row(name, e):
    f = f"{RUNS}/{name}/evals/{e['steps']:010d}/validation.json"
    m = glob.glob(f"{RUNS}/{name}/evals/*_olmes/olmes_results/metrics.json")
    if not (os.path.exists(f) and m):
        return None
    pal = None
    for line in open(f"{RUNS}/{name}/metrics.paloma_np.jsonl"):
        r = json.loads(line)
        if r["global_step"] == e["steps"]:
            p = [v for k, v in r.items() if k.endswith("primary_score") and "/paloma::" not in k]
            pal = np.mean(p) if len(p) == 16 else None
    if pal is None:
        return None
    t = {x["alias"]: x["metrics"]["primary_score"] for x in json.load(open(m[0]))["tasks"]}
    return dict(name=name, width=e["width"], data="fe" if "_fe_" in name else "dclm",
                val=json.load(open(f))["dclm_heldout"]["ce_per_token"], pal=pal,
                qa=100 - 100 * np.mean([t[a] for a in QA]), mmlu=100 - 100 * t["mmlu:rc::olmes"],
                mmlu_bpb=t["mmlu:rc:bpb::olmes"], math=t["olmo3:base_easy:math_bpb"],
                code=np.mean([t["codex_humaneval:3shot:bpb::none"], t["mbpp:3shot:bpb::none"]]))


def load():
    M = json.load(open(f"{ROOT}/apps/main/configs/experiments/data_constrainedv2/manifest.json"))
    pairs = []
    for n in sorted(M):
        if "_fe_" not in n:
            continue
        a, b = row(n, M[n]), row(n.replace("_fe_", "_u_"), M[n.replace("_fe_", "_u_")])
        if a and b:
            pairs.append((b, a))
    return pairs


def style(ax, xlabel):
    ax.set_xlabel(xlabel)
    ax.grid(True, color="#e6e6e3", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def scatter(ax, rows, xk, yk, ylabel, xlabel):
    rs = []
    for d, (lbl, c) in DATA.items():
        sub = [r for r in rows if r["data"] == d]
        x, y = np.array([r[xk] for r in sub]), np.array([r[yk] for r in sub])
        for r in sub:
            ax.plot(r[xk], r[yk], marker=MARK[r["width"]], ms=6, ls="none", color=c, alpha=0.8, mec="white", mew=0.6)
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 20)
        ax.plot(xs, a + b * xs, color=c, lw=2)
        rs.append(f"{lbl} {np.corrcoef(x, y)[0, 1]:.2f}")
    x, y = np.array([r[xk] for r in rows]), np.array([r[yk] for r in rows])
    ax.set_title(f"{ylabel}\nr: {', '.join(rs)}, pooled {np.corrcoef(x, y)[0, 1]:.2f}", fontsize=10)
    style(ax, xlabel)


def deltas(ax, pairs, yk, ylabel):
    dx = np.array([f["pal"] - d["pal"] for d, f in pairs])
    dy = np.array([f[yk] - d[yk] for d, f in pairs])
    for (d, f), x, y in zip(pairs, dx, dy):
        ax.plot(x, y, marker=MARK[d["width"]], ms=6, ls="none", color="#52514e", alpha=0.7, mec="white", mew=0.6)
    ax.axhline(0, color="#b9b8b0", lw=1)
    ax.axvline(0, color="#b9b8b0", lw=1)
    better = np.mean(dy < 0)
    ax.set_title(f"Δ {ylabel} (FineWeb-Edu − DCLM twin)\nr = {np.corrcoef(dx, dy)[0, 1]:.2f}; median Δ = "
                 f"{np.median(dy):+.3g}; FW-Edu better in {100 * better:.0f}% of pairs", fontsize=10)
    style(ax, "Δ Paloma BPB (FineWeb-Edu − DCLM twin)")


def pair_lines(ax, pairs, xk, yk, ylabel, xlabel):
    """Each FineWeb-Edu run joined to its DCLM twin: the segment is a pure data change at fixed compute. Title: median
    segment slope (data) vs the slope of the DCLM-only least-squares line (compute)."""
    for d, f in pairs:
        ax.plot([d[xk], f[xk]], [d[yk], f[yk]], color="#b9b8b0", lw=0.7, zorder=1)
    for key, (lbl, c) in DATA.items():
        sub = [p[0] if key == "dclm" else p[1] for p in pairs]
        ax.scatter([r[xk] for r in sub], [r[yk] for r in sub], s=22, color=c, edgecolor="white", lw=0.5, zorder=3)
    x, y = np.array([d[xk] for d, _ in pairs]), np.array([d[yk] for d, _ in pairs])
    b_compute = np.polyfit(x, y, 1)[0]
    b_data = np.median([(f[yk] - d[yk]) / (f[xk] - d[xk]) for d, f in pairs])
    xs = np.linspace(x.min(), max(f[xk] for _, f in pairs), 20)
    a = np.mean(y - b_compute * x)
    ax.plot(xs, a + b_compute * xs, color=DATA["dclm"][1], lw=1.5, ls="--", zorder=2)
    ax.set_title(f"{ylabel}\nslope: data {b_data:.3g} vs compute {b_compute:.3g} (ratio {b_data / b_compute:.2f})",
                 fontsize=10)
    style(ax, xlabel)
    return b_data, b_compute


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/temp/olmes_data_corr.png")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    pairs = load()
    rows = [r for p in pairs for r in p]
    fig, axes = plt.subplots(3, 5, figsize=(26, 15))
    for j, (k, lbl) in enumerate(EVALS):
        scatter(axes[0, j], rows, "val", k, lbl, "DCLM held-out loss (nats/token)")
        scatter(axes[1, j], rows, "pal", k, lbl, "Paloma BPB (mean of 16)")
        deltas(axes[2, j], pairs, k, lbl)
    handles = [Line2D([], [], marker="o", ls="none", color=c, ms=8, label=l) for l, c in DATA.values()]
    handles += [Line2D([], [], marker=m, ls="none", color="#888", ms=8, label=f"w{w}") for w, m in MARK.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11)
    fig.suptitle(f"Data as the varying factor: {len(pairs)} FineWeb-Edu runs vs their DCLM twins (same width, horizon, "
                 "batch, LR, wd). Rows: vs DCLM loss; vs Paloma; paired differences", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(a.out, dpi=110)
    print(a.out)
    fig2, axes2 = plt.subplots(2, 5, figsize=(26, 10))
    slopes = {}
    for i, (xk, xl) in enumerate([("val", "DCLM held-out loss (nats/token)"), ("pal", "Paloma BPB (mean of 16)")]):
        for j, (k, lbl) in enumerate(EVALS):
            slopes[(xk, k)] = pair_lines(axes2[i, j], pairs, xk, k, lbl, xl)
    fig2.legend(handles=handles[:2] + [Line2D([], [], color="#b9b8b0", lw=1, label="same settings, data changed"),
                                       Line2D([], [], color=DATA["dclm"][1], ls="--", lw=1.5,
                                              label="DCLM compute trend (fit)")],
                loc="lower center", ncol=4, frameon=False, fontsize=11)
    fig2.suptitle("Lines across data mixes: each grey segment joins a FineWeb-Edu run to its DCLM twin (fixed compute). "
                  "Ratio = data slope / compute slope (1 = data moves models along the compute trend)", fontsize=13)
    fig2.tight_layout(rect=(0, 0.04, 1, 0.96))
    out2 = a.out.replace(".png", "_pairs.png")
    fig2.savefig(out2, dpi=110)
    print(out2)
    for (xk, k), (bd, bc) in slopes.items():
        print(f"{xk:4s} {k:9s} data slope {bd:+.3f} compute slope {bc:+.3f} ratio {bd / bc:.2f}")
    for k, lbl in EVALS:
        dy = np.array([f[k] - d[k] for d, f in pairs])
        print(f"{lbl:28s} median Δ(FE-DCLM) {np.median(dy):+.4f}  FE better {100 * np.mean(dy < 0):.0f}%")
    print("median Δ DCLM val", np.median([f["val"] - d["val"] for d, f in pairs]),
          "median Δ Paloma", np.median([f["pal"] - d["pal"] for d, f in pairs]))
    if a.wandb:
        import wandb
        run = wandb.init(project="lingua-clean-data_constrainedv2", name="olmes-data-corr", id="olmes-data-corr",
                         resume="allow")
        run.log({"olmes_data_corr": wandb.Image(a.out), "olmes_data_pairs": wandb.Image(out2)})
        run.finish()
