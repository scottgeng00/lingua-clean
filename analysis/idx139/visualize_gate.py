"""Visualize idx 139's RKL gate behavior — final clean version."""
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Inputs/outputs live in $ANALYSIS_OUT_DIR/idx139/ by default; fall back to the
# directory containing this script for backwards compatibility.
HERE = Path(os.environ.get("ANALYSIS_OUT_DIR", Path(__file__).parent.parent.parent)) / "idx139"
if not (HERE / "per_token_with_strings.parquet").exists():
    HERE = Path(__file__).parent

OUT = str(HERE / "gate_visualization.png")
df = pd.read_parquet(HERE / "per_token_with_strings.parquet")

tau = float(np.quantile(df['entropy'].values, 0.30))

src_order = ['math', 'flan', 'wiki', 'stackexchange', 'pes2o', 'dclm']
colors = {
    'math':          '#d62728',
    'flan':          '#ff7f0e',
    'wiki':          '#2ca02c',
    'stackexchange': '#1f77b4',
    'pes2o':         '#9467bd',
    'dclm':          '#7f7f7f',
}

fig = plt.figure(figsize=(16, 13.5))
gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 0.9, 1.4], width_ratios=[1.4, 1.0],
                      hspace=0.85, wspace=0.30)

# =========================
# Panel A: ridgeline — per-source normalization (so each ridge is independently visible)
# =========================
axA = fig.add_subplot(gs[0, 0])

bins = np.linspace(0, 12, 121)
y_offset_step = 1.05

for i, src in enumerate(src_order):
    H = df[df['source'] == src]['entropy'].values
    hist, _ = np.histogram(H, bins=bins, density=True)
    # Per-source normalization to ridge height of ~0.85
    if hist.max() > 0:
        hist = hist / hist.max() * 0.85
    y0 = i * y_offset_step
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    fire_mask = bin_centers < tau
    axA.fill_between(bin_centers, y0, y0 + hist, where=fire_mask,
                     color=colors[src], alpha=0.85, linewidth=0)
    axA.fill_between(bin_centers, y0, y0 + hist, where=~fire_mask,
                     color=colors[src], alpha=0.18, linewidth=0)
    axA.plot(bin_centers, y0 + hist, color=colors[src], linewidth=1.2)
    axA.axhline(y0, color='black', linewidth=0.4, alpha=0.3)
    axA.text(-0.4, y0 + 0.35, src,
             ha='right', va='center', fontsize=12, fontweight='bold',
             color=colors[src])

axA.axvline(tau, color='red', linestyle='--', linewidth=1.6, alpha=0.75, zorder=10)
axA.text(tau, len(src_order) * y_offset_step + 0.20,
         f'τ = {tau:.2f} nats   (q = 0.30)',
         ha='center', va='bottom', fontsize=10.5, color='red', fontweight='bold')

axA.set_xlim(-0.2, 12)
axA.set_ylim(-0.1, len(src_order) * y_offset_step + 0.95)
axA.set_xlabel('teacher entropy H (nats, T=2.0)', fontsize=11)
axA.set_yticks([])
axA.set_title('A. Per-source teacher-entropy distribution',
              fontsize=12, loc='left', pad=28, fontweight='bold')
axA.text(0, 1.02, 'Each ridge normalized independently · Solid = below τ (RKL fires) · Faded = above τ (CE only)',
         transform=axA.transAxes, fontsize=9.5, color='#444', style='italic')
for spine in ['left', 'right', 'top']:
    axA.spines[spine].set_visible(False)

# =========================
# Panel B: per-source fire rate
# =========================
axB = fig.add_subplot(gs[0, 1])

fire_rates = [100 * df[df['source'] == s]['fired'].mean() for s in src_order]
mean_Hs    = [df[df['source'] == s]['entropy'].mean() for s in src_order]
bar_colors = [colors[s] for s in src_order]
y_pos = np.arange(len(src_order))
axB.barh(y_pos, fire_rates, color=bar_colors, alpha=0.85,
         edgecolor='black', linewidth=0.5)

for i, (rate, mH) in enumerate(zip(fire_rates, mean_Hs)):
    axB.text(rate + 2, i, f'{rate:.1f}%   meanH={mH:.2f}',
             va='center', fontsize=10, fontweight='bold')

axB.set_yticks(y_pos)
axB.set_yticklabels(src_order, fontsize=11)
axB.invert_yaxis()
axB.set_xlabel('fire rate (% tokens with RKL)', fontsize=10.5)
axB.axvline(30, color='red', linestyle='--', linewidth=1, alpha=0.6)
axB.text(30, -0.65, 'global\nq=30%', ha='center', va='bottom', fontsize=8.5, color='red')
axB.set_xlim(0, 145)
axB.set_title('B. Fire rate by source', fontsize=12, loc='left', pad=28, fontweight='bold')
for spine in ['right', 'top']:
    axB.spines[spine].set_visible(False)

# =========================
# Panel C: composition stacked bars
# =========================
axC = fig.add_subplot(gs[1, :])

total = len(df)
mix_share = {s: len(df[df['source'] == s]) / total for s in src_order}
fired_total = df['fired'].sum()
unfired_total = (~df['fired']).sum()
fired_share = {s: df[(df['source'] == s) & df['fired']].shape[0] / fired_total for s in src_order}
unfired_share = {s: df[(df['source'] == s) & ~df['fired']].shape[0] / unfired_total for s in src_order}

categories = [
    'mix\n(equal-doc baseline)',
    f'fired bucket\n(n={fired_total:,}, RKL applied)',
    f'unfired bucket\n(n={unfired_total:,}, CE only)',
]
data_rows = [mix_share, fired_share, unfired_share]
y_pos = np.arange(len(categories))
left = np.zeros(len(categories))
for src in src_order:
    vals = np.array([row[src] for row in data_rows])
    axC.barh(y_pos, vals, left=left, color=colors[src],
             edgecolor='white', linewidth=1.5, alpha=0.92, height=0.7)
    for i, v in enumerate(vals):
        if v > 0.035:
            txt_color = 'white' if src in ['math', 'stackexchange', 'pes2o'] else 'black'
            axC.text(left[i] + v/2, i, f'{src}\n{100*v:.0f}%',
                     ha='center', va='center', fontsize=9.5,
                     color=txt_color, fontweight='bold')
    left += vals

axC.set_yticks(y_pos)
axC.set_yticklabels(categories, fontsize=10.5)
axC.invert_yaxis()
axC.set_xlim(0, 1)
axC.set_xlabel('share of bucket', fontsize=10.5)
axC.set_title('C. Composition: where do fired vs. skipped tokens come from?',
              fontsize=12, loc='left', pad=28, fontweight='bold')
axC.text(0, 1.02,
         'Math: 12% of mix → 37% of fires. DCLM: 24% of mix → 29% of skips.',
         transform=axC.transAxes, fontsize=9.5, color='#444', style='italic')
for spine in ['right', 'top']:
    axC.spines[spine].set_visible(False)

# =========================
# Panel D: example tokens — one row per source, single clean label layer
# =========================
axD = fig.add_subplot(gs[2, :])

# Hand-picked exemplars per source, sorted by H, with context, target, fired flag
example_rows = [
    ('math', [
        (0.00, '"   "→"100"',         True),
        (0.64, '"Exactly"→"!"',       True),
        (0.64, '":\\n"→"```"',        True),
        (4.89, '"have"→" specific"',  True),
        (6.60, '"111"→"112"',         False),
        (10.95, '"\\n\\n"→"Student"', False),
        (11.04, '"A"→" regular"',     False),
    ]),
    ('wiki', [
        (0.01, '"Lanc"→"ashire"',          True),
        (0.05, '"-S"→"aint"',              True),
        (2.79, '"Wilhelm"→" Ernst"',       True),
        (4.90, '"developing"→" countries"',True),
        (7.80, '"reportedly"→" changed"',  False),
        (11.01, '"Al"→"god"',              False),
        (11.02, '"V"→"aren"',              False),
    ]),
    ('dclm', [
        (0.02, '"\\n"→"   " (indent)',  True),
        (2.97, '"local"→" community"',  True),
        (2.98, '"aspect"→" ratio"',     True),
        (4.90, '"smell"→" of"',         True),
        (7.85, '"is"→" affected"',      False),
        (11.03, '"4"→" Easy"',          False),
        (11.03, '"Self"→" Awareness"',  False),
    ]),
]

n_rows = len(example_rows)
row_height = 1.0
axD.set_xlim(-0.5, 12.5)
axD.set_ylim(-0.5, n_rows * row_height + 0.55)

for ri, (src, examples) in enumerate(example_rows):
    y = (n_rows - 1 - ri) * row_height + 0.15
    # Source label on the left
    axD.text(-0.35, y, src.upper(), ha='right', va='center',
             fontsize=13, fontweight='bold', color=colors[src])
    # Row separator
    axD.axhline(y - row_height/2 + 0.05, color='gray', linewidth=0.3, alpha=0.3)
    # Plot each example
    for i, (H, label, fired) in enumerate(examples):
        marker = 'o' if fired else 'X'
        size = 130 if fired else 110
        face = colors[src] if fired else 'white'
        edge = 'black' if fired else colors[src]
        axD.scatter(H, y, marker=marker, s=size,
                    facecolor=face, edgecolor=edge, linewidth=1.6, zorder=3)
        # Alternating label position above/below
        dy = 0.32 if (i % 2 == 0) else -0.32
        va = 'bottom' if dy > 0 else 'top'
        axD.annotate(label, (H, y + dy), ha='center', va=va,
                     fontsize=8.5, family='monospace',
                     bbox=dict(boxstyle='round,pad=0.22',
                               facecolor='white',
                               edgecolor=colors[src],
                               linewidth=0.6, alpha=0.95),
                     zorder=4)

# τ line
axD.axvline(tau, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
axD.text(tau, n_rows * row_height + 0.30, f'τ={tau:.2f}', fontsize=10,
         color='red', ha='center', fontweight='bold')

axD.set_yticks([])
axD.set_xlabel('teacher entropy H (nats)', fontsize=10.5)
axD.set_title('D. Example tokens   (● filled = fires (RKL applied),  ✕ outlined = skipped (CE only))',
              fontsize=12, loc='left', pad=28, fontweight='bold')
for spine in ['left', 'right', 'top']:
    axD.spines[spine].set_visible(False)
axD.grid(True, axis='x', alpha=0.15)

# Suptitle
fig.suptitle(
    'Idx 139 RKL gate — which tokens does the entropy threshold fire on?\n'
    f'OLMo-2-7B-Instruct teacher · Dolmino sources · T=2.0 · q=0.30 · n={len(df):,} tokens',
    fontsize=13, y=0.98, fontweight='bold'
)

plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
print(f'wrote {OUT}')
print(f'size: {os.path.getsize(OUT):,} bytes')
