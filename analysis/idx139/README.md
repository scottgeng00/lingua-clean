# analysis/idx139

Per-token (H_T, H_S) gate-overlap analysis backing the paper's "entropy-gating
is real" claim for recipe `idx139` (originally `rkl_entgate_q30_lam1p0`).

## Prerequisites

Source the env script first so the scripts can find the teachers, data, and
output dirs:

```bash
source scripts/env.sh
```

Required env vars (see `scripts/env.sh`):
- `TEACHER_7B_PATH` — OLMo-2-1124-7B-Instruct (teacher used in idx 139)
- `TEACHER_1B_RLVR1_PATH` — OLMo-2-0425-1B-RLVR1 (1B-RLVR1 baseline teacher)
- `STUDENT_HF_PATH` — OLMo-2-0425-1B-stage1-4001B HF dir (used by `per_token_HT_HS.py` to load the student via `AutoModelForCausalLM`)
- `DATA_ROOT` — directory holding `dclm_shuffled/`, `math_shuffled/`, etc.
- `ANALYSIS_OUT_DIR` — where parquets/jsons/pngs land (default
  `${HOME}/lingua-runs/analysis`); the scripts write into
  `${ANALYSIS_OUT_DIR}/idx139/`.

## Scripts

| File                                | What it does |
|-------------------------------------|--------------|
| `teacher_entropy_by_source.py`      | Computes per-token H(p_T) for the 7B-Instruct teacher across all six Dolmino sources at T={1.0, 2.0}. Decides go/no-go for entropy-gated RKL by checking the factual-vs-reasoning entropy gap at T=2.0. Writes `teacher_entropy_by_source.parquet` and `teacher_entropy_summary.json`. |
| `teacher_entropy_by_source.sbatch`  | SLURM wrapper for the above (1 GPU, 1 hour, 200G). |
| `teacher_entropy_per_token.py`      | Same teacher / sources / T=2.0, but keeps the decoded token string at every position so we can sort by H and inspect fired vs. unfired tokens. Writes `per_token_with_strings.parquet` and `fired_vs_unfired_examples.json`. |
| `teacher_entropy_per_token.sbatch`  | SLURM wrapper. |
| `teacher_entropy_RLVR1.py`          | Same as `teacher_entropy_per_token.py` but with the 1B-RLVR1 teacher — checks whether the entropy-gate intuition generalizes to a small same-family teacher. Writes `per_token_RLVR1_teacher.parquet` and `per_token_RLVR1_teacher_summary.json`. |
| `teacher_entropy_RLVR1.sbatch`      | SLURM wrapper. |
| `per_token_HT_HS.py`                | Joint per-token analysis of teacher entropy (H_T) and student entropy (H_S) on the 1B init. Computes the gate-overlap between idx 139's gate (bottom-30% H_T) and idx 143's gate (top-30% H_S). Writes `per_token_HT_HS.parquet` and `HT_HS_summary.json`. |
| `per_token_HT_HS.sbatch`            | SLURM wrapper. |
| `plot_entropy_histogram.py`         | Reads `teacher_entropy_by_source.parquet` and renders per-source + pooled + math-vs-non-math overlays at T=1 and T=2. Writes `teacher_entropy_hist_T{1,2}.png`. |
| `visualize_gate.py`                 | Reads `per_token_with_strings.parquet` and renders the four-panel "what does the gate fire on" figure (ridgeline / fire-rate / composition / exemplar tokens). Writes `gate_visualization.png`. |
| `plot_mid_vs_post.py`               | Scatter of mid-train vs. post-train OLMES scores across SFT/DPO/RLVR1 stages. Reads OLMES `metrics.json` files from `${MID_ROOT}` and `${POST_ROOT}` (defaults under `${EVAL_ROOT}`); see the docstring for the exact recipe-to-directory map. Writes `mid_vs_post_scatter.png`. |

## Typical workflow

```bash
source scripts/env.sh

# (1) Go/no-go diagnostic — does H separate factual vs reasoning at T=2.0?
sbatch analysis/idx139/teacher_entropy_by_source.sbatch

# (2) Per-token exemplars + global tau (needed by visualize_gate.py).
sbatch analysis/idx139/teacher_entropy_per_token.sbatch

# (3) Same for the 1B-RLVR1 teacher (pre-flight for idx 145).
sbatch analysis/idx139/teacher_entropy_RLVR1.sbatch

# (4) Teacher-vs-student joint analysis (needed for idx 143 prediction).
sbatch analysis/idx139/per_token_HT_HS.sbatch

# (5) Figures (cheap, run interactively).
python analysis/idx139/plot_entropy_histogram.py
python analysis/idx139/visualize_gate.py
python analysis/idx139/plot_mid_vs_post.py   # only if you have post-train OLMES dumps
```

All outputs land in `${ANALYSIS_OUT_DIR}/idx139/`.
