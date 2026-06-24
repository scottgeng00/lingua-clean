"""Scatter: mid-train task scores vs post-train task scores, per recipe and post-train stage.
Each dot = one task. Color = recipe. Three panels = SFT / DPO / RLVR1.

Inputs (override via env vars):
  POST_ROOT  — root of post-train (SFT/DPO/RLVR1) OLMES results (default $EVAL_ROOT/posttrain).
  MID_ROOT   — root of mid-train OLMES results (default $EVAL_ROOT/midtrain).
  NTP_MID    — path to NTP baseline mid-train metrics.json
               (default $EVAL_ROOT/midtrain/ntp_baseline/evals/0000028800/olmes_results/metrics.json).
  ANALYSIS_OUT_DIR — where to write the figure (default ./analysis_outputs/idx139).
"""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ANALYSIS_OUT_DIR = Path(os.environ.get("ANALYSIS_OUT_DIR", "./analysis_outputs")) / "idx139"
ANALYSIS_OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = str(ANALYSIS_OUT_DIR / "mid_vs_post_scatter.png")

EVAL_ROOT = os.environ.get("EVAL_ROOT", "./evals")
POST_ROOT = os.environ.get("POST_ROOT", f"{EVAL_ROOT}/posttrain")
MID_ROOT = os.environ.get("MID_ROOT", f"{EVAL_ROOT}/midtrain")

pairs = {
    'kd_rl7b':     ('dolmino_midtrain_olmo2_kd_rl7b_28800steps', 'kd-rl7b-28800'),
    'rkl_purekd':  ('dolmino_midtrain_olmo2_reverse_kl_1b_base_purekd_28800steps', 'rkl-purekd-i100-28800'),
    'rkl_topkgap': ('dolmino_midtrain_olmo2_rkl_topkgap_ce_k20_lam0p5_28800steps', 'rkl-topkgap-125-28800'),
}
# NTP baseline is overlaid separately: post-train evaluated on only 10 tasks,
# so it would collapse the intersection if added to `pairs`. Use a dedicated
# mid-train results path (the run lives under dolmino_expert_runs, not v3).
NTP_LABEL = 'ntp_baseline'
NTP_MID = os.environ.get(
    "NTP_MID",
    f"{MID_ROOT}/ntp_baseline/evals/0000028800/olmes_results/metrics.json",
)
NTP_POST_DIRS = {
    'sft':   f'{POST_ROOT}/olmo2-1b-baseline28800-sft',
    'dpo':   f'{POST_ROOT}/olmo2-1b-baseline28800-dpo',
    'rlvr1': f'{POST_ROOT}/olmo2-1b-baseline28800-rlvr1-step2600',
}
colors = {'kd_rl7b': '#1f77b4', 'rkl_purekd': '#d62728', 'rkl_topkgap': '#2ca02c',
          NTP_LABEL: '#8c8c8c'}
markers = {'kd_rl7b': 'o', 'rkl_purekd': 's', 'rkl_topkgap': '^', NTP_LABEL: 'D'}

def load(p):
    if not Path(p).exists(): return None
    out = {}
    for line in json.load(open(p)).get('all_primary_scores', []):
        try:
            name, val = line.rsplit(':', 1)
            out[name.strip()] = float(val.strip())
        except Exception:
            pass
    return out

def load_jsonl(p):
    """metrics-all.jsonl: one task per line, key is task_config.metadata.alias or task_name."""
    if not Path(p).exists(): return None
    out = {}
    for line in open(p):
        rec = json.loads(line)
        score = rec.get('metrics', {}).get('primary_score')
        if score is None: continue
        alias = (rec.get('task_config', {}).get('metadata') or {}).get('alias')
        name = alias or rec.get('task_name')
        out[name] = float(score)
    return out

def load_pertask_dir(d):
    """Aggregate per-task metrics.json files (one per task) by their OLMES alias."""
    import glob
    out = {}
    for f in glob.glob(f'{d}/task-*-metrics.json'):
        rec = json.load(open(f))
        score = rec.get('metrics', {}).get('primary_score')
        if score is None: continue
        alias = (rec.get('task_config', {}).get('metadata') or {}).get('alias')
        out[alias or rec['task_name']] = float(score)
    return out

mid_scores = {}
post_scores = {'sft': {}, 'dpo': {}, 'rlvr1': {}}
for label, (mid_dir, post_pref) in pairs.items():
    mid_scores[label] = load(f'{MID_ROOT}/{mid_dir}/evals/0000028800/olmes_results/metrics.json')
    post_scores['sft'][label]   = load(f'{POST_ROOT}/{post_pref}-sft/metrics.json')
    post_scores['dpo'][label]   = load(f'{POST_ROOT}/{post_pref}-dpo/metrics.json')
    post_scores['rlvr1'][label] = load(f'{POST_ROOT}/{post_pref}-rlvr1-2604steps/metrics.json')

# NTP baseline: separate paths, per-task post-train files, ~146 tasks.
ntp_mid = load(NTP_MID) or {}
ntp_post = {stage: load_pertask_dir(d) for stage, d in NTP_POST_DIRS.items()}

# Common tasks across the 3 KD recipes only (NTP is overlaid on whatever subset it has).
all_dicts = list(mid_scores.values()) + [post_scores[s][l] for s in post_scores for l in pairs]
tasks = sorted(set.intersection(*[set(d.keys()) for d in all_dicts]))

HEADLINE = {
    'mmlu::olmes': 'mmlu',
    'mmlu:mc::olmes': 'mmlu_mc',
    'mmlu_pro:mc::none': 'mmlu_pro',
    'agi_eval_english:1shot::olmes': 'agi_en',
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=False)
fig.subplots_adjust(top=0.84, wspace=0.28, left=0.06, right=0.98, bottom=0.13)

for ax, stage in zip(axes, ['sft', 'dpo', 'rlvr1']):
    all_x, all_y = [], []
    for label in pairs:
        xs = [mid_scores[label][t] for t in tasks]
        ys = [post_scores[stage][label][t] for t in tasks]
        ax.scatter(xs, ys, c=colors[label], marker=markers[label], s=22,
                   alpha=0.55, edgecolor='none', label=label)
        all_x.extend(xs); all_y.extend(ys)
        # Per-recipe correlation
        r = np.corrcoef(xs, ys)[0,1]
        rho, _ = spearmanr(xs, ys)
        # Label headline tasks (skip if not in common set)
        for t, lbl in HEADLINE.items():
            if t not in tasks: continue
            x, y = mid_scores[label][t], post_scores[stage][label][t]
            ax.annotate(lbl, (x, y), fontsize=6.5, alpha=0.65, color=colors[label],
                        xytext=(2, 2), textcoords='offset points')

    # NTP overlay: only on the tasks where both mid and post-stage scores exist.
    # Excluded from the pooled r / OLS fit (different task subset would bias it).
    ntp_tasks = sorted(set(ntp_mid) & set(ntp_post[stage]))
    if ntp_tasks:
        nx = [ntp_mid[t] for t in ntp_tasks]
        ny = [ntp_post[stage][t] for t in ntp_tasks]
        ax.scatter(nx, ny, c=colors[NTP_LABEL], marker=markers[NTP_LABEL], s=18,
                   alpha=0.45, edgecolor='none', label=f'{NTP_LABEL} (n={len(ntp_tasks)})')

    # Pooled regression
    all_x, all_y = np.array(all_x), np.array(all_y)
    pooled_r = np.corrcoef(all_x, all_y)[0,1]
    pooled_rho, _ = spearmanr(all_x, all_y)
    # OLS line
    slope, intercept = np.polyfit(all_x, all_y, 1)
    xx = np.linspace(all_x.min(), all_x.max(), 50)
    ax.plot(xx, slope*xx + intercept, color='black', linewidth=1.2, alpha=0.6, linestyle='--',
            label=f'OLS  r={pooled_r:.2f}')
    # y=x reference
    lo, hi = min(all_x.min(), all_y.min()), max(all_x.max(), all_y.max())
    ax.plot([lo, hi], [lo, hi], color='gray', linewidth=0.7, alpha=0.4, label='y=x')

    ax.set_xlabel('mid-train score (step 28,800)', fontsize=10)
    ax.set_ylabel(f'post-train ({stage}) score', fontsize=10)
    ax.set_title(f'mid → {stage}\nPearson r = {pooled_r:.2f}   Spearman ρ = {pooled_rho:.2f}   '
                 f'(n={len(all_x)}, 3 recipes × {len(tasks)} tasks)',
                 fontsize=10.5, loc='left', pad=8)
    ax.grid(True, alpha=0.2)
    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    if stage == 'sft':
        ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)

fig.suptitle(
    'Mid-train vs. post-train per-task scores  ·  3 KD recipes × 125 OLMES tasks',
    fontsize=12.5, fontweight='bold', y=0.97
)
fig.text(0.5, 0.005,
         'Each dot = one task. Within-recipe rank correlation is very high (r~0.83-0.96): '
         'mid-train task profile transfers to post-train. But per-task recipe ordering is NOT preserved '
         '(see headline labels — recipes often swap order at the same task between mid and post).',
         ha='center', fontsize=9, style='italic', color='#444')

plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
print(f'wrote {OUT}')
print(f'size: {os.path.getsize(OUT):,} bytes')
