"""Hyperparameter transfer DCLM -> FineWeb-Edu on data_constrainedv2: w{w}_u_* (DCLM) vs w{w}_fe_* (FineWeb-Edu) runs
with identical width, horizon, batch, LR, wd. Each dataset is scored on its own held-out set (dclm_heldout /
fwedu_heldout). Figures:
  1. LR curves at wd 0.1, one subplot per (width, batch), loss minus that curve's best, both datasets, each horizon;
  2. best LR (parabola vertex in log LR) FineWeb-Edu vs DCLM over every (width, batch, wd, horizon) slice with >= 3
     shared LRs, on own held-out and on Paloma.

    .venv-kd/bin/python analysis/results_viewer/hparam_transfer.py [--out temp/hparam_transfer] [--wandb]
"""
import argparse
import collections
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

from dataconstrained_lr_v2 import COLORS, MARKERS, ROOT, RUNS, vertex

NAME = re.compile(r"w(\d+)_(fe|u)_bs(\d+)_lr([\dp]+e-\d)_wd([\dp]+?)(?:_d(\d)x)?$")
DATA = {"u": ("DCLM", "-"), "fe": ("FineWeb-Edu", "--")}
MULTS = [1, 2, 4]


def paloma(n, e):
    f = f"{RUNS}/{n}/metrics.paloma_np.jsonl"
    for line in open(f) if os.path.exists(f) else []:
        r = json.loads(line)
        if r["global_step"] == e["steps"]:
            p = [v for k, v in r.items() if k.endswith("primary_score") and "/paloma::" not in k]
            return np.mean(p) if len(p) == 16 else None


def load():
    """(w, bs, wd, mult) -> metric ("own" / "pal") -> data -> {lr: value}"""
    M = json.load(open(f"{ROOT}/apps/main/configs/experiments/data_constrainedv2/manifest.json"))
    G = collections.defaultdict(lambda: collections.defaultdict(dict))
    for n, e in M.items():
        m = NAME.match(n)
        if not m:
            continue
        f = f"{RUNS}/{n}/evals/{e['steps']:010d}/validation.json"
        if not os.path.exists(f):
            continue
        w, d, bs, lr, wd, mult = m.groups()
        k = (int(w), int(bs), float(wd.replace("p", ".")), int(mult) if mult else 4)
        lr = float(lr.replace("p", "."))
        v = json.load(open(f))
        G[k]["own"].setdefault(d, {})[lr] = v["fwedu_heldout" if d == "fe" else "dclm_heldout"]["ce_per_token"]
        p = paloma(n, e)
        if p is not None:
            G[k]["pal"].setdefault(d, {})[lr] = p
    return G


def shared(G, k, metric):
    g = G[k][metric]
    if "fe" not in g or "u" not in g:
        return None
    lrs = sorted(set(g["fe"]) & set(g["u"]))
    return {d: {l: g[d][l] for l in lrs} for d in ("u", "fe")} if len(lrs) >= 3 else None


def best_lr(g):
    v = vertex(dict(sorted(g.items())))
    return (v[0], False) if v else (min(g, key=g.get), True)


def curves_fig(G, out):
    panels = sorted({(k[0], k[1]) for k in G if k[2] == 0.1 and any(shared(G, (k[0], k[1], 0.1, m), "own")
                                                                        for m in MULTS)})
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.8), squeeze=False)
    for ax, (w, bs) in zip(axes[0], panels):
        for i, m in enumerate(MULTS):
            s = shared(G, (w, bs, 0.1, m), "own")
            if not s:
                continue
            for d, (lbl, ls) in DATA.items():
                g = s[d]
                lo = min(g.values())
                ax.plot(list(g), [y - lo for y in g.values()], color=COLORS[i], ls=ls, marker=MARKERS[i], ms=5, lw=1.8,
                        mfc="white" if d == "fe" else COLORS[i])
                b, edge = best_lr(g)
                if not edge:
                    ax.axvline(b, color=COLORS[i], ls=ls, lw=0.8, alpha=0.6)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1e}".replace("e-0", "e-")))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(-0.003, 0.06)
        ax.set_title(f"width {w}, batch {bs}, wd 0.1", fontsize=11)
        ax.set_xlabel("learning rate")
        ax.grid(True, color="#e6e6e3", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("own held-out loss − curve's best (nats/token)")
    handles = [Line2D([], [], color=COLORS[i], marker=MARKERS[i], lw=1.8, label=f"{m}x") for i, m in enumerate(MULTS)]
    handles += [Line2D([], [], color="#52514e", ls=ls, lw=1.8, label=lbl) for lbl, ls in DATA.values()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10)
    fig.suptitle("LR transfer DCLM → FineWeb-Edu: each dataset on its own held-out set (solid = DCLM, dashed = "
                 "FineWeb-Edu; vertical lines = parabola vertex)", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    path = f"{out}_lr_curves.png"
    fig.savefig(path, dpi=120)
    return path


def scatter_fig(G, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, metric, title in zip(axes, ("own", "pal"), ("own held-out loss", "Paloma BPB")):
        xs, ys = [], []
        for k in sorted(G):
            s = shared(G, k, metric)
            if not s:
                continue
            (bd, ed), (bf, ef) = best_lr(s["u"]), best_lr(s["fe"])
            c = COLORS[[0.01, 0.03, 0.1, 0.3].index(k[2])]
            ax.plot(bd, bf, marker={512: "o", 768: "s"}[k[0]], ms=8 if k[1] != 512 else 6, ls="none", color=c,
                    mfc="white" if (ed or ef) else c, mew=1.5, alpha=0.85)
            xs.append(bd); ys.append(bf)
        r = np.array(ys) / np.array(xs)
        lim = (min(xs + ys) / 1.3, max(xs + ys) * 1.3)
        ax.plot(lim, lim, color="#52514e", lw=1.2)
        for f in (0.8, 1.25):
            ax.plot(lim, [l * f for l in lim], color="#b9b8b0", lw=1, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(f"Best LR by {title}: median FW-Edu/DCLM ratio {np.median(r):.2f}; "
                     f"{100 * np.mean(np.abs(np.log(r)) < np.log(1.25)):.0f}% within ±25% (n = {len(r)})", fontsize=10)
        ax.set_xlabel("best LR, DCLM")
        ax.set_ylabel("best LR, FineWeb-Edu")
        ax.grid(True, color="#e6e6e3", lw=0.6, which="both")
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], marker="o", ls="none", color=COLORS[i], ms=8, label=f"wd {wd:g}")
               for i, wd in enumerate([0.01, 0.03, 0.1, 0.3])]
    handles += [Line2D([], [], marker="o", ls="none", color="#888", label="w512"),
                Line2D([], [], marker="s", ls="none", color="#888", label="w768"),
                Line2D([], [], marker="o", ls="none", mfc="white", mec="#888", label="optimum at grid edge"),
                Line2D([], [], color="#b9b8b0", ls="--", label="±25%")]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=9)
    fig.suptitle("Best LR, FineWeb-Edu vs DCLM, per (width, batch, wd, horizon) slice", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    path = f"{out}_best_lr.png"
    fig.savefig(path, dpi=120)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/temp/hparam_transfer")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    G = load()
    paths = [curves_fig(G, a.out), scatter_fig(G, a.out)]
    print("\n".join(paths))
    if a.wandb:
        import wandb
        run = wandb.init(project="lingua-clean-data_constrainedv2", name="hparam-transfer-fwedu",
                         id="hparam-transfer-fwedu", resume="allow")
        run.log({os.path.basename(p)[:-4]: wandb.Image(p) for p in paths})
        run.finish()
