"""Histograms of teacher entropy per source at T=1 and T=2.

Generates `teacher_entropy_hist_T{1,2}.png`. Three panels:
  (1) per-source overlaid density
  (2) pooled marginal with q=0.30 gate threshold drawn
  (3) math vs non-math pooled overlay (the bimodality claim)
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Inputs/outputs live in $ANALYSIS_OUT_DIR/idx139/ by default; fall back to the
# directory containing this script for backwards compatibility.
HERE = Path(os.environ.get("ANALYSIS_OUT_DIR", Path(__file__).parent.parent.parent)) / "idx139"
if not (HERE / "teacher_entropy_by_source.parquet").exists():
    HERE = Path(__file__).parent
df = pd.read_parquet(HERE / "teacher_entropy_by_source.parquet")

SOURCES = ["math", "flan", "wiki", "dclm", "stackexchange", "pes2o"]
COLORS = {
    "math": "#d62728",
    "flan": "#ff7f0e",
    "wiki": "#2ca02c",
    "dclm": "#1f77b4",
    "stackexchange": "#9467bd",
    "pes2o": "#8c564b",
}


def plot_for_temperature(T: float, outfile: Path) -> None:
    sub = df[df.temperature == T]
    xmax = float(sub.entropy.quantile(0.999))
    bins = np.linspace(0, xmax, 80)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for src in SOURCES:
        vals = sub[sub.source == src].entropy.values
        ax.hist(
            vals, bins=bins, density=True, histtype="step",
            linewidth=1.8, label=f"{src} (n={len(vals)})", color=COLORS[src],
        )
    ax.set_xlabel(f"teacher entropy H(p_T)  [nats, T={T}]")
    ax.set_ylabel("density")
    ax.set_title(f"Per-source teacher entropy (T={T})")
    ax.legend(fontsize=9, loc="upper right")

    ax = axes[1]
    pooled = sub.entropy.values
    tau = float(np.quantile(pooled, 0.30))
    ax.hist(pooled, bins=bins, density=True, color="#444", alpha=0.7)
    ax.axvline(tau, color="red", linestyle="--", linewidth=2,
               label=f"q=0.30 gate threshold τ={tau:.2f}")
    ax.set_xlabel(f"teacher entropy H(p_T)  [nats, T={T}]")
    ax.set_ylabel("density")
    ax.set_title(f"Pooled marginal + gate threshold (T={T})")
    ax.legend(fontsize=10)

    ax = axes[2]
    math_vals = sub[sub.source == "math"].entropy.values
    nonmath_vals = sub[sub.source != "math"].entropy.values
    ax.hist(math_vals, bins=bins, density=True, color="#d62728",
            alpha=0.55, label=f"math (n={len(math_vals)})")
    ax.hist(nonmath_vals, bins=bins, density=True, color="#1f77b4",
            alpha=0.55, label=f"non-math (n={len(nonmath_vals)})")
    ax.axvline(tau, color="red", linestyle="--", linewidth=2,
               label=f"τ={tau:.2f}")
    ax.set_xlabel(f"teacher entropy H(p_T)  [nats, T={T}]")
    ax.set_ylabel("density")
    ax.set_title(f"Math vs non-math (T={T})  — the bimodality claim")
    ax.legend(fontsize=10)

    fig.suptitle(
        f"OLMo-2-7B-Instruct teacher entropy, T={T} (idx 139 diagnostic)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=140, bbox_inches="tight")
    plt.close(fig)

    fired_frac_math = float((math_vals <= tau).mean())
    fired_frac_nonmath = float((nonmath_vals <= tau).mean())
    print(f"[T={T}] τ@q0.30 = {tau:.3f} nats")
    print(f"  math      fired_frac = {fired_frac_math:.3f}")
    print(f"  non-math  fired_frac = {fired_frac_nonmath:.3f}")
    print(f"  saved → {outfile}")


if __name__ == "__main__":
    for T in (1.0, 2.0):
        plot_for_temperature(T, HERE / f"teacher_entropy_hist_T{int(T)}.png")
