"""Loss vs training compute for every evaluated run of the data-constrained ladder (all widths, stages, passes, seeds,
decay branches), with the Pareto frontier (lowest loss at <= compute) labelled.

Compute = 6 * (body + output head params) * tokens: the output head matmul is real compute (and 3x the body at width
512 with the 152k vocab); the input embedding is a lookup. Attention-score FLOPs are left out.
Logs to project lingua-clean-dataconstrained, run "compute-scaling".

    source scripts/env.sh && .venv-kd/bin/python analysis/results_viewer/dataconstrained_compute.py [--no-wandb]
"""
import math
import os
import sys

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NO_WANDB = "--no-wandb" in sys.argv
UNIQUE_ONLY = "--unique-only" in sys.argv  # only runs that never repeat data: stage 1 and pass 1 of stage 2
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../setup"))
import dataconstrained_grid as G  # noqa: E402
import make_dataconstrained_configs as MC  # noqa: E402

WIDTHS = [512, 768, 1152]
COLORS = {512: "#2a78d6", 768: "#eb6834", 1152: "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
LABEL = {"val_ce": "dclm_heldout val loss (nats/token)", "paloma": "Paloma BPB (mean of 16)"}
OUT = "/checkpoint/mnm/sgeng/pretrain/logs/reports/dataconstrained_compute_{}" + ("_unique" if UNIQUE_ONLY else "") + ".png"


def is_unique(e):
    """Unique-data point: stage 1, or the first pass over a stage 2 pool."""
    return e["stage"] == 1 or e["eval_point"] == "p1"


def flops(width, tokens):
    s = MC.shape(width)
    return 6.0 * (s["body"] + s["embedding"]) * tokens


def describe(e):
    return f"{short(e)}, lr {e['lr']:g} bs {e['bs']} wd {e['wd']:g}" + (" s1" if e["seed"] else "")


def points(metric):
    out = []
    for n, e in G.M.items():
        if e["width"] in WIDTHS:
            v = G.metric(n, metric)
            if v is not None and (is_unique(e) or not UNIQUE_ONLY):
                out.append(dict(name=n, c=flops(e["width"], e["tokens"]), y=v, e=e))
    return out


def frontier(pts, tol=1.02):
    """Points with no other point at <= compute (within tol, so runs whose step counts round differently count as
    equal compute) and lower loss."""
    pts = sorted(pts, key=lambda p: p["c"])
    out = []
    for p in pts:
        if all(q["y"] >= p["y"] for q in pts if q["c"] <= p["c"] * tol and q is not p):
            out.append(p)
    return out


def short(e):
    if e["eval_point"] == "p1":
        return f"w{e['width']} unique 0.25x (pass 1)"
    if e["stage"] == 1:
        return f"w{e['width']} unique {'1x' if e['eval_point'] == 'd1x' else '2x'}"
    return f"w{e['width']} rep {e['eval_point'].replace('-independent', '')}"


def fit_curves(fr):
    """Power law L = A C^-a and saturating L = E + A C^-a, fit (in log loss) to the unique-data frontier points."""
    from scipy.optimize import curve_fit
    pts = [p for p in fr if is_unique(p["e"])]
    c = np.array([float(p["c"]) for p in pts]) / 1e18
    y = np.array([p["y"] for p in pts])
    a, logA = np.polyfit(np.log(c), np.log(y), 1)
    fits = {"power": (lambda x, A=math.exp(logA), a=a: A * (x / 1e18) ** a, f"L = {math.exp(logA):.3f} (C/1e18)^{a:.4f}")}
    try:
        (E, A, b), _ = curve_fit(lambda x, E, A, b: np.log(E + A * x ** -b), c, np.log(y), p0=(y.min() * 0.6, 1.0, 0.2),
                                 bounds=([0, 0, 0], [y.min(), 100, 2]), maxfev=20000)
        fits["sat"] = (lambda x, E=E, A=A, b=b: E + A * (x / 1e18) ** -b,
                       f"L = {E:.3f} + {A:.3f} (C/1e18)^-{b:.3f}")
    except RuntimeError:
        pass
    return pts, fits


def plot(metric):
    pts = points(metric)
    fr = frontier(pts)
    fig, ax = plt.subplots(figsize=(15, 8.5))
    for w in WIDTHS:
        for stage, filled in ((1, True), (2, False)):
            sel = [p for p in pts if p["e"]["width"] == w and p["e"]["stage"] == stage]
            ax.scatter([p["c"] for p in sel], [p["y"] for p in sel], s=22, linewidths=1.1, alpha=0.75,
                       facecolors=COLORS[w] if filled else "none", edgecolors=COLORS[w], zorder=2,
                       label=f"w{w} {'unique (stage 1: 1x, 2x)' if stage == 1 else ('unique (stage 2 pass 1: 0.25x)' if UNIQUE_ONLY else 'repeated (stage 2)')} ({len(sel)})")
    ax.plot([p["c"] for p in fr], [p["y"] for p in fr], color=MUTED, linewidth=0.9, drawstyle="steps-post", zorder=3)
    ax.scatter([p["c"] for p in fr], [p["y"] for p in fr], s=60, facecolors="none", edgecolors=INK, linewidths=1.5,
               zorder=4, label=f"Pareto frontier ({len(fr)})")
    # Short labels, cycling through offsets so neighbours don't collide; full configs go in the table.
    offsets = [(10, 8), (10, -14), (-12, -18), (14, 2)]
    for i, p in enumerate(fr):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(f"{i + 1} {short(p['e'])}", (p["c"], p["y"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, color=INK, ha="left" if dx > 0 else "right")
    fit_pts, fits = fit_curves(fr)
    xx = np.geomspace(min(p["c"] for p in pts) * 0.8, 1e21, 200)
    fit_label = "unique-data frontier"
    for key, style, color, lw in (("power", ":", "#8a4fd0", 2.2), ("sat", "-", "#c0392b", 3.0)):
        if key in fits:
            fn, eq = fits[key]
            ax.plot(xx, fn(xx), color=color, linewidth=lw, linestyle=style, zorder=6, alpha=0.9,
                    label=f"{'power law' if key == 'power' else 'power law + floor'} fit to unique-data frontier: {eq}")
    ax.scatter([p["c"] for p in fit_pts], [p["y"] for p in fit_pts], s=14, color=INK, zorder=5,
               label=f"points used in fit ({len(fit_pts)}: stage 1 and pass 1 on the frontier)")
    print(f"\n## {metric} fits ({len(fit_pts)} unique-data frontier points)")
    for key, (fn, eq) in fits.items():
        resid = [math.log(p["y"] / fn(p["c"])) for p in fit_pts]
        print(f"  {key:5s} {eq}   rms log resid {np.sqrt(np.mean(np.square(resid))):.4f}")
        for c in (1e20, 1e21):
            print(f"        predicted at {c:.0e}: {fn(c):.4f}")
    last = fr[-1]
    for key, (fn, _) in fits.items():
        print(f"  {key:5s} frontier end {short(last['e'])} {last['y']:.4f} at {last['c']:.2e} vs curve {fn(last['c']):.4f}")
    ys = sorted(p["y"] for p in pts)
    lo = min(fr[-1]["y"], *(fn(1e21) for fn, _ in fits.values()))
    hi = fr[0]["y"]
    ax.set_ylim(lo - 0.04 * (hi - lo), hi + 0.08 * (hi - lo))
    ax.set_xlim(min(p["c"] for p in pts) * 0.8, 1e21)
    ax.set_xscale("log")
    ax.set_yscale("log")
    yt = [t for t in np.arange(1.0, 5.0, 0.1) if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{t:.1f}" for t in yt])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("training compute, FLOPs = 6 x (body + output head params) x tokens (log scale)", color=MUTED,
                  fontsize=9)
    ax.set_ylabel(LABEL[metric] + " (log scale)", color=MUTED, fontsize=9)
    ax.set_title(f"{LABEL[metric]} vs compute: {'unique-data runs only' if UNIQUE_ONLY else 'every evaluated run'} ({len(pts)} points, all widths, "
                 f"{'stage 1 + stage 2 pass 1' if UNIQUE_ONLY else 'stages, passes'}, seeds, decay branches)", color=INK, fontsize=10, loc="left")
    ax.grid(color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT.format(metric), dpi=140)
    print(f"\n## {metric} Pareto frontier")
    for i, p in enumerate(fr):
        print(f"  {i + 1:2d}  {p['c']:.2e} FLOPs  {p['y']:.4f}  {describe(p['e'])}")
    return fr


def main():
    frs = {m: plot(m) for m in ("val_ce", "paloma")}
    if not NO_WANDB:
        import wandb
        run = wandb.init(project="lingua-clean-dataconstrained", name="compute-scaling" + ("-unique" if UNIQUE_ONLY else ""),
                         id="compute-scaling" + ("-unique" if UNIQUE_ONLY else ""),
                         resume="allow", job_type="analysis")
        log = {f"{m}_vs_compute_png": wandb.Image(OUT.format(m)) for m in frs}
        for m, fr in frs.items():
            log[f"{m}_frontier"] = wandb.Table(columns=["#", "flops", m, "config", "run"],
                                         data=[[i + 1, p["c"], p["y"], describe(p["e"]), p["name"]]
                                               for i, p in enumerate(fr)])
        run.log(log)
        print(run.url)
        run.finish()


if __name__ == "__main__":
    main()
