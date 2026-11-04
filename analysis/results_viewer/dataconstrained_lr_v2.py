"""LR vs dclm_heldout val loss on ladder data_constrainedv2 (unique DCLM data), at the best (batch, wd) per setting.

For every (width, Chinchilla multiple) we pick the (batch, wd) slice whose best LR gives the lowest val loss (seed means;
only slices with >= 4 LRs), then plot that slice's loss vs LR. 1x/2x/4x come from the 4x main runs and their decay
branches (w{w}_u_...), 8x from the 8x main runs (w{w}_u8x_...); a missing 1x-4x cell falls back to the 8x run's branch.
Two figures: one subplot per width (colour = Chinchilla multiple) and one subplot per multiple (colour = width).

    .venv-kd/bin/python analysis/results_viewer/dataconstrained_lr_v2.py [--out temp/lr_v2] [--wandb]
"""
import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GROUP = "data_constrainedv2"
RUNS = f"/checkpoint/mnm/sgeng/pretrain/lingua-runs/runs/midtrain/experiments/{GROUP}"
WIDTHS = [512, 768, 1152, 1536]
MULTS = [1, 2, 4, 8]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4 (validated); markers add redundancy
MARKERS = ["o", "s", "^", "D"]
NAME = re.compile(r"w(\d+)_u(8x)?_bs(\d+)_lr([\dp]+e-\d)_wd([\dp]+?)(?:_seed(\d))?(?:_d(\d)x)?$")


def load():
    """(width, mult, bs, wd) -> lr -> [val per seed]; 4x-run family preferred, 8x-run branches fill gaps."""
    M = json.load(open(f"{ROOT}/apps/main/configs/experiments/{GROUP}/manifest.json"))
    prim, fill = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for n, e in M.items():
        m = NAME.match(n)
        if not m:
            continue
        f = f"{RUNS}/{n}/evals/{e['steps']:010d}/validation.json"
        if not os.path.exists(f):
            continue
        w, u8, bs, lr, wd, seed, d = m.groups()
        mult = int(d) if d else (8 if u8 else 4)
        key = (int(w), mult, int(bs), float(wd.replace("p", ".")))
        v = json.load(open(f))["dclm_heldout"]["ce_per_token"]
        tgt = prim if (mult == 8) == bool(u8) else fill
        tgt[key][float(lr.replace("p", "."))].append(v)
    out = {}
    for key in set(prim) | set(fill):
        g = {lr: np.mean(v) for lr, v in fill.get(key, {}).items()}
        g.update({lr: np.mean(v) for lr, v in prim.get(key, {}).items()})
        out[key] = dict(sorted(g.items()))
    return out


def vertex(g):
    lrs, ys = np.array(list(g)), np.array(list(g.values()))
    j = int(np.argmin(ys))
    if j == 0 or j == len(ys) - 1:
        return None
    a, b, c = np.polyfit(np.log(lrs[j - 1:j + 2]), ys[j - 1:j + 2], 2)
    return float(np.exp(-b / (2 * a))), float(c - b * b / (4 * a))


def pick(data, bs=None):
    """(width, mult) -> (best loss, bs, wd, curve): the (bs, wd) slice with the lowest best-LR loss; bs fixed if given."""
    best = {}
    for (w, mult, b, wd), g in data.items():
        if len(g) < 4 or (bs is not None and b != bs):
            continue
        lo = min(g.values())
        if (w, mult) not in best or lo < best[(w, mult)][0]:
            best[(w, mult)] = (lo, b, wd, g)
    return best


def pick_panel_bs(data, by):
    """One batch size per subplot: among batch sizes available (>= 4 LRs) for every series in the panel, the one with
    the smallest mean loss penalty vs each series' unconstrained best; wd stays the best one at that batch."""
    free = pick(data)
    panels, series = (WIDTHS, MULTS) if by == "width" else (MULTS, WIDTHS)
    out, chosen = {}, {}
    for p in panels:
        keys = [(p, s) if by == "width" else (s, p) for s in series]
        keys = [k for k in keys if k in free]
        cands = {}
        for b in sorted({k[2] for k in data}):
            sel = pick(data, b)
            if all(k in sel for k in keys):
                cands[b] = np.mean([sel[k][0] - free[k][0] for k in keys])
        b = min(cands, key=cands.get)
        chosen[p] = (b, cands[b])
        sel = pick(data, b)
        out.update({k: sel[k] for k in keys})
    return out, chosen


def plot(best, by, out):
    """by="width": raw loss, all multiples in one panel (y spans each series' basin, best + 0.08).
    by="mult": widths differ by ~0.7 nats, so y = loss minus that width's best loss (basin shape / location)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    panels, series = (WIDTHS, MULTS) if by == "width" else (MULTS, WIDTHS)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    for ax, p in zip(axes, panels):
        tops, bots = [], []
        for i, s in enumerate(series):
            w, mult = (p, s) if by == "width" else (s, p)
            if (w, mult) not in best:
                continue
            lo, bs, wd, g = best[(w, mult)]
            off = lo if by == "mult" else 0.0
            lbl = (f"{mult}x" if by == "width" else f"w{w}") + f"  (wd {wd:g})"
            panel_bs = bs
            ax.plot(list(g), [y - off for y in g.values()], color=COLORS[i], marker=MARKERS[i], ms=6, lw=2, label=lbl)
            v = vertex(g)
            if v:
                ax.plot([v[0]], [v[1] - off], marker="*", ms=15, color=COLORS[i], mec="white", mew=1.2, zorder=5)
            bots.append(lo - off)
            tops.append(lo - off + 0.08)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0e}".replace("e-0", "e-")))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title((f"width {p}" if by == "width" else f"{p}x Chinchilla") + f"  —  batch {panel_bs}", fontsize=12)
        ax.set_xlabel("learning rate")
        ax.grid(True, which="major", color="#e6e6e3", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, title="Chinchilla mult." if by == "width" else "width", title_fontsize=8)
        if bots:
            ax.set_ylim(min(bots) - 0.01, max(tops))
    axes[0].set_ylabel("dclm_heldout val loss (nats/token)" if by == "width"
                       else "val loss − best loss at that width (nats/token)")
    fig.suptitle("LR vs held-out loss, one batch size per subplot (best wd at that batch)  (★ = parabola vertex"
                 + ("; y clipped at best + 0.08 per series)" if by == "width" else ")"), fontsize=12)
    fig.tight_layout()
    path = f"{out}_by_{by}.png"
    fig.savefig(path, dpi=130)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/temp/lr_v2")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    data = load()
    best = pick(data)
    print("Unconstrained best (batch, wd) per setting:")
    print("| width | mult | batch | wd | best LR (vertex) | val loss | #LRs |\n|---|---|---|---|---|---|---|")
    for (w, mult) in sorted(best):
        lo, bs, wd, g = best[(w, mult)]
        v = vertex(g)
        print(f"| {w} | {mult}x | {bs} | {wd:g} | {f'{v[0]:.2e}' if v else 'edge ' + f'{min(g, key=g.get):.2e}'} | "
              f"{v[1] if v else lo:.4f} | {len(g)} |")
    paths = []
    for by in ("width", "mult"):
        sel, chosen = pick_panel_bs(data, by)
        print(f"\nby {by}: one batch per subplot (mean penalty vs free choice):",
              {p: f"bs{b} (+{pen:.4f})" for p, (b, pen) in chosen.items()})
        for k in sorted(sel):
            lo, bs, wd, g = sel[k]; v = vertex(g)
            print(f"  w{k[0]} {k[1]}x bs{bs} wd{wd:g}: {f'{v[0]:.2e} -> {v[1]:.4f}' if v else f'edge {min(g, key=g.get):.2e} -> {lo:.4f}'}")
        paths.append(plot(sel, by, a.out))
    print("\n".join(paths))
    if a.wandb:
        import wandb
        run = wandb.init(project="lingua-clean-data_constrainedv2", name="lr-curves-best-bs-wd", id="lr-curves-best-bs-wd",
                         resume="allow")
        run.log({os.path.basename(p)[:-4]: wandb.Image(p) for p in paths})
        run.finish()
