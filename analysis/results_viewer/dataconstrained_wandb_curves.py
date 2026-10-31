"""Repetition curves of the data-constrained ladder on wandb: validation loss and Paloma vs pass for the stage 2 final
runs (both seeds, every pass, plus the 32-pass extension points once evaluated), with the best stage 1 (unique data)
result drawn at the same token count (1x Chinchilla = pass 4, 2x = pass 8).

Logs to project lingua-clean-dataconstrained, run "repetition-curves": native wandb line charts (hover) and a
rendered PNG of the small multiples. Re-run to refresh.

    source scripts/env.sh && .venv-kd/bin/python analysis/results_viewer/dataconstrained_wandb_curves.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import wandb  # noqa: E402

sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataconstrained_grid as G  # noqa: E402

FINALS = {512: "w512_s2final_bs64_lr6e-3_wd0p3", 768: "w768_s2final_bs64_lr2e-3_wd0p3", 1152: "w1152_s2final_bs128_lr1p5e-3_wd0p3"}
CONFIG = {512: "LR 6e-3, batch 64, wd 0.3", 768: "LR 2e-3, batch 64, wd 0.3", 1152: "LR 1.5e-3, batch 128, wd 0.3"}
SERIES = {"seed 0": "#2a78d6", "seed 1": "#eb6834", "unique data (stage 1 best)": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
OUT = "/checkpoint/mnm/sgeng/pretrain/logs/reports/dataconstrained_repetition_curves.png"


def curve(width, seed, metric):
    """[(pass, value)] over passes 1-16 of the final run and the extension (20, 24, 28, 32) where evaluated. A width
    whose final run is 32 passes from scratch (w768) has every point under ..._x32[_pK]."""
    base = FINALS[width] + ("_seed1" if seed else "")
    names = {k: (base if k == 16 else f"{base}_p{k}") for k in range(1, 17)}
    names.update({k: (f"{base}_x32" if k == 32 else f"{base}_x32_p{k}") for k in (20, 24, 28, 32)})
    for k in range(1, 17):
        if names[k] not in G.M:
            names[k] = f"{base}_x32_p{k}"
    pts = []
    for k, n in sorted(names.items()):
        if n in G.M:
            v = G.metric(n, metric)
            if v is not None:
                pts.append((k, v))
    return pts


def unique_ref(width, metric):
    """Best stage 1 value at 1x (plotted at pass 4) and 2x (pass 8), selected by val CE as in the sweep."""
    out = []
    for pt, x in (("d1x", 4), ("2x", 8)):
        c = [(G.metric(n, "val_ce"), n) for n, e in G.M.items() if e["width"] == width and e["stage"] == 1
             and e["eval_point"] == pt]
        _, n = min(t for t in c if t[0])
        out.append((x, G.metric(n, metric)))
    return out


def main():
    data = {(w, m): {"seed 0": curve(w, 0, m), "seed 1": curve(w, 1, m), "unique data (stage 1 best)": unique_ref(w, m)}
            for w in FINALS for m in ("val_ce", "paloma")}
    label = {"val_ce": "dclm_heldout val loss (nats/token)", "paloma": "Paloma BPB (mean of 16)"}

    fig, axes = plt.subplots(2, len(FINALS), figsize=(6 * len(FINALS), 8.5))
    for j, w in enumerate(FINALS):
        for i, m in enumerate(("val_ce", "paloma")):
            ax = axes[i, j]
            for name, pts in data[(w, m)].items():
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ref = name.startswith("unique")
                ax.plot(xs, ys, linestyle="--" if ref else "-", linewidth=2, marker="o", markersize=5 if not ref else 8,
                        color=SERIES[name], label=name, markeredgecolor="white", markeredgewidth=1.2, zorder=3)
                ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8,
                            color=INK, va="center")
            ax.set_xscale("log", base=2)
            ticks = [1, 2, 4, 8, 16, 32] if any(p[0] > 16 for p in data[(w, m)]["seed 0"]) else [1, 2, 4, 8, 16]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
            ax.set_xlabel("passes over the 0.25x-Chinchilla pool (log scale)", color=MUTED, fontsize=9)
            ax.set_ylabel(label[m], color=MUTED, fontsize=9)
            ax.set_title(f"width {w}: {label[m].split(' (')[0]}  ({CONFIG[w]})", color=INK, fontsize=10, loc="left")
            ax.grid(color=GRID, linewidth=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Repetition curves: final stage 2 runs, decay branch after every pass (unique-data reference at "
                 "equal tokens: 1x Chinchilla = pass 4, 2x = pass 8)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)

    run = wandb.init(project="lingua-clean-dataconstrained", name="repetition-curves", id="repetition-curves",
                     resume="allow", job_type="analysis")
    log = {"repetition_curves_png": wandb.Image(OUT)}
    for (w, m), series in data.items():
        keys = [k for k, v in series.items() if v]
        log[f"w{w}_{m}_vs_pass"] = wandb.plot.line_series(
            xs=[[p[0] for p in series[k]] for k in keys], ys=[[p[1] for p in series[k]] for k in keys], keys=keys,
            title=f"width {w}: {label[m]} vs pass", xname="pass")
        log[f"w{w}_{m}_table"] = wandb.Table(columns=["series", "pass", m],
                                             data=[[k, p[0], p[1]] for k in keys for p in series[k]])
    run.log(log)
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
