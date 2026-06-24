# lingua-clean

A surface-level cleanup of [lingua](https://github.com/facebookresearch/lingua)
focused on a self-contained subset of knowledge-distillation (KD) ablations
for **OLMo-2-0425-1B mid-training on the Dolmino splits at 28800 steps**.

This is a *port*, not a rewrite: same training driver, same APIs, same outputs
as the upstream research repo. The only difference is that we drop ~40 loss
functions and their dispatch paths that aren't needed for the supported
recipes, and we provide a single launch-script entry point.

## Supported recipes

All recipes live under `apps/main/configs/recipes/`. They all share the same
canonical config (OLMo-2-0425-1B student, dolmino-splits dataset, 28800 steps,
4-node FSDP, bf16, FP8 7B teacher where applicable). They differ only in the
`teacher_model_path` and the `data.use_<X>_distillation` block.

### Headline recipes

| Recipe                          | Teacher                  | Loss form |
|---------------------------------|--------------------------|-----------|
| `ntp_baseline`                  | — (no teacher)           | Vanilla next-token-prediction CE. |
| `fkd_1b`                        | OLMo-2-0425-1B-Instruct  | Forward KL distillation, α=0.5, T=2. |
| `fkd_7b`                        | OLMo-2-1124-7B-Instruct  | Forward KL distillation, α=0.5, T=2. |
| `rkl_entgate_q30_lam1p0` (`idx139`) | OLMo-2-1124-7B-Instruct  | (idx 139) Entropy-gated reverse KL. L = λ·CE + 1[H(p_T)≤τ]·T²·KL(p_S‖p_T). q=0.30, λ=1, T=2. |
| `rkl_entswitch_q30_lam1p0` (`idx148`) | OLMo-2-1124-7B-Instruct | (idx 148) Entropy-switched RKL. L = (1−m_t)·λ·CE + m_t·T²·KL(p_S‖p_T). |
| `entband_q30_q30_lam1p0`        | OLMo-2-1124-7B-Instruct  | (idx 151) Entropy-band KD: CE + T²·RKL(low 30%) + T²·FKL(high 30%). |

### Ablation controls

| Recipe                          | Notes |
|---------------------------------|-------|
| `fkl_top30_lam1p0`              | (idx 152) Entropy-band scaffold with `entband_low_quantile=0`; FKL on top-30% teacher-H tokens only. |
| `fkl_entgate_q30_lam1p0`        | (idx 140) FKL counterpart of `idx139`. |
| `rkl_randmask_p30_lam1p0`       | Replaces the entgate predicate with a uniform random top-30% mask. |
| `rkl_stentgate_q30_lam1p0`      | Gate on **student** entropy (top-30%), not teacher entropy. |
| `rkl_uniform_lam{0p1,0p3,0p5,1p0}` | RKL purekd + uniform-CE-on-all-tokens sweep (no token selection). |
| `rkl_purekd`                    | (idx 100) Pure RKL (MiniLLM-style), α=1, no CE. |
| `rkl_fkl_mix_alpha{0p5,0p8}`    | RKL+FKL pure-KD mixture (idx 103/104). |

## Layout

```
lingua-clean/
├── README.md                                  ← this file
├── LICENSE                                    ← upstream Llama-2 license
├── requirements.txt                           ← plain-pip deps (no torch/xformers/flash-attn — see bin/)
├── bin/
│   └── install_requirements.sh                ← env bootstrap (pip + cu121 wheels)
├── lingua/                                    ← lingua library (unchanged)
│   └── (args, data, distributed, optim, ...)
├── apps/main/
│   ├── train.py                               ← PRUNED (15791 → ~7600 lines)
│   ├── eval.py, eval_olmes.py, generate.py    ← unchanged
│   ├── transformer.py                         ← unchanged
│   └── configs/
│       ├── dolmino_midtrain_olmo2_v2.yaml     ← canonical 9600-step config
│       ├── dolmino_expert_cpt_baseline_28800.yaml  ← canonical 28800-step config
│       └── recipes/                           ← one YAML per supported recipe
├── setup/
│   ├── consolidate.py
│   ├── convert_consolidated_lingua_ckpt_to_hf.py
│   ├── download_hf_ckpt.py
│   └── download_prepare_hf_data.py
├── analysis/
│   └── idx139/                               ← per-token (H_T, H_S) gate-overlap analysis
└── scripts/
    ├── README.md
    ├── env.sh                                ← single source of truth for paths
    ├── launch_midtrain.sh                     ← sbatch wrapper (the entry point)
    └── _sbatch_inner.sh                       ← runs INSIDE sbatch / srun
```

The `apps/main/train.py` here is the same training driver as upstream lingua
with the unused `compute_*_loss` functions removed. The removed names are kept
as stubs that raise `NotImplementedError` so the dispatch chain inside `train()`
still parses cleanly (the corresponding `use_<X>` flags default to `False` and
are not exposed in any supported recipe, so the stub bodies are unreachable at
runtime).

Note: `lingua/data.py` was left intact (the `DataArgs` dataclass has dozens of
deeply-commented interdependent fields; pruning them was high-risk for low
benefit since unused flags default to `False` and have no runtime cost).

## Setup

1. Activate the existing `lingua` conda env (the one the original repo uses).
   No separate env bootstrap needed:
   ```bash
   conda activate lingua
   ```
   If you don't have it yet, `bash bin/install_requirements.sh` (after a
   fresh `conda create -n lingua python=3.11 -y && conda activate lingua`)
   installs everything — pip deps from `requirements.txt`, then `torch==2.5.0`
   + `xformers==0.0.28.post2` (cu121) + `flash-attn==2.7.4.post1`. Override
   `LINGUA_CONDA_ENV=<name>` if your env has a different name.

2. Edit `scripts/env.sh` (or export the variables in your shell beforehand)
   so the paths point at where you've actually downloaded the data and
   checkpoints. By default everything lives under one deletable sandbox
   dir, `${LINGUA_SANDBOX_ROOT}` (default `~/lingua-clean-sandbox`):
   - `${LINGUA_SANDBOX_ROOT}/data/dolmino_splits/`             (Dolmino mid-train splits)
   - `${LINGUA_SANDBOX_ROOT}/teachers/OLMo-2-0425-1B-Instruct/`
   - `${LINGUA_SANDBOX_ROOT}/teachers/OLMo-2-1124-7B-Instruct/`
   - `${LINGUA_SANDBOX_ROOT}/students/OLMo-2-0425-1B-stage1-4001B/`     (Lingua DCP root, `STUDENT_INIT_PATH`)
   - `${LINGUA_SANDBOX_ROOT}/students/OLMo-2-0425-1B-stage1-4001B/hf/`  (HF mirror, `STUDENT_HF_PATH`)
   - `${LINGUA_SANDBOX_ROOT}/runs/{midtrain,evals,analysis,slurm_logs}/`

   To start over: `rm -rf ${LINGUA_SANDBOX_ROOT}`. Override individual paths
   (e.g. point `DATA_ROOT` at a long-lived cache) to share artifacts across
   sandboxes.

   The student is split because Lingua's training driver loads the init via
   `lingua.checkpoint.load_from_checkpoint`, which expects a DCP directory
   (`.metadata` + `__0_0.distcp`), while the tokenizer / OLMES eval / idx139
   analysis path loads HF format via `AutoModelForCausalLM.from_pretrained`.
   `scripts/fetch_models.sh` populates both layouts in one shot.

   Optional cluster-specific knobs in `scripts/env.sh` (unset by default,
   slurm uses site defaults if you leave them empty):
   - `SLURM_ACCOUNT`, `SLURM_QOS`, `SLURM_PARTITION` — appended to the
     `launch_midtrain.sh` sbatch command if non-empty.
   - `SLURM_QOS_TEST` — dev-tier QOS used by `scripts/run_diagnostic.sh` and
     `scripts/smoke_test.py` (default `h100_dev`).
   - `SLURM_QOS_ANALYSIS` — small-job QOS for `analysis/idx139/*.sbatch`
     (defaults to `SLURM_QOS_TEST`).
   - `WANDB_ENTITY` — wandb entity for all runs (leave empty to use your
     personal default).

3. Source the env script (must be done in every fresh shell):
   ```bash
   source scripts/env.sh
   ```

4. Download teachers (`OLMo-2-0425-1B-Instruct`, `OLMo-2-1124-7B-Instruct`) and
   the student init checkpoint (`OLMo-2-0425-1B-stage1-4001B`) into
   `${TEACHER_1B_PATH}`, `${TEACHER_7B_PATH}`, and `${STUDENT_INIT_PATH}`
   respectively. Use `setup/download_hf_ckpt.py` or download manually from the
   AI2 Hugging Face org.

5. Download and shard the Dolmino mid-training mix (DCLM / FLAN / Math / Wiki /
   pes2o / StackExchange splits) into `${DATA_ROOT}`. Use
   `setup/download_prepare_hf_data.py`.

6. Confirm the launch script can find the recipe you want and renders a sane
   sbatch invocation:
   ```bash
   RECIPE=ntp_baseline DRY_RUN=1 bash scripts/launch_midtrain.sh
   ```

## Launching a run

```bash
source scripts/env.sh

# Real submit (4-node, ~56h wall clock for the 7B-teacher recipes):
RECIPE=idx139 bash scripts/launch_midtrain.sh

# Single-node smoke test (override nodes; will still run for ~weeks at this size):
RECIPE=ntp_baseline NNODES=1 bash scripts/launch_midtrain.sh

# Just print the sbatch invocation, don't submit:
RECIPE=fkd_7b DRY_RUN=1 bash scripts/launch_midtrain.sh
```

Outputs:
- Checkpoints / wandb-offline dumps land at the `dump_dir` declared in the
  recipe YAML (default: `${MIDTRAIN_ROOT}/<recipe_name>`).
- Slurm logs land at `${SLURM_LOG_DIR}` (default `~/lingua-runs/slurm_logs`).
- OLMES evaluation runs automatically every 1200 steps via the
  `async_eval_gpus` path (see the `eval:` block of any recipe YAML).

## Analysis

Per-token (H_T, H_S) gate-overlap analysis backing the paper's "entropy-gating
is real" claim for the `idx139` recipe (originally `rkl_entgate_q30_lam1p0`) lives under
`analysis/idx139/`. See `analysis/idx139/README.md` for the workflow.

## What was dropped

Compared to upstream `apps/main/train.py`:
- Removed 39 `compute_*_loss` functions that none of the supported recipes use.
  Each removed name is kept as a `_removed_loss(name)` stub at the top of
  `apps/main/train.py` so the dispatch chain still parses; calling one raises
  `NotImplementedError`. The full removed list (from `train.py`):
  - **Two-teacher / merged-teacher KD:** `compute_two_teacher_geometric_kd_loss`,
    `compute_projected_two_teacher_kd_loss`,
    `compute_agreement_gated_two_teacher_kd_loss`,
    `compute_competence_routed_two_teacher_kd_loss`,
    `compute_intersection_projected_kd_loss`.
  - **Residual / hybrid / projected / geometric KD:**
    `compute_hybrid_residual_kd_loss`, `compute_projected_kd_loss`,
    `compute_geometric_kd_loss`, `compute_akl_distillation_loss`,
    `compute_bucket_aware_kd_loss`, `compute_entropy_gated_kd_loss`,
    `compute_selective_kd_loss`.
  - **Rho1 family:** `compute_rho1_loss`, `compute_rho1_kd_loss`,
    `compute_rho1_expert_stratified_loss`.
  - **RKL-with-extra-CE-gate family:** `compute_rkl_with_gated_ce_loss`,
    `compute_rkl_with_lowent_ce_loss`, `compute_rkl_with_source_ce_loss`,
    `compute_rkl_with_teacher_disagree_ce_loss`,
    `compute_rkl_with_teacher_fail_ce_loss`,
    `compute_rkl_with_teacher_success_ce_loss`,
    `compute_rkl_with_topk_gap_ce_loss`,
    `compute_rkl_with_topk_gap_ce_replace_loss`,
    `compute_rkl_with_gradagree_gap_ce_loss`.
  - **Frontier / EMA / margin family:** `compute_frontier_band_loss`,
    `compute_frontierv2_loss`, `compute_frontierv3_loss`,
    `compute_frontierv4_loss`, `compute_ema_frontier_loss`,
    `compute_ema_ref_loss`, `compute_margin_constraint_loss`,
    `compute_entropy_aware_margin_loss`.
  - **Best-expert / SIW / LWT / MILE / REMIT / teacher-critic:**
    `compute_best_expert_loss`, `compute_best_expert_seq_kd_loss`,
    `compute_siw_loss`, `compute_lwt_loss`, `compute_mile_loss`,
    `compute_remit_loss`, `compute_teacher_critic_loss`.
- Kept the dispatch chain inside `train()` unchanged so callers and config
  flags behave identically to upstream; the removed `use_<X>` flags simply
  default to `False` in the supported recipes.
- All scripts in the upstream `lingua/` root (`ce_kl_*`, `diag_*`, `rho1_*`,
  `s3_forensics_*`, `analyze_*`, `bakd_*`, `merge_*`, `precompute_*`, all
  `relaunch_*`, all `synth_*`, the dozens of one-off `*.sh` launchers and
  Python analysis utilities) are not included. Only the four `setup/` scripts
  needed for env bootstrap are copied.
- `lingua/custom_data.py` (factuality / retrieved-context dataloader used only
  by a separate JH project) was also dropped; no supported recipe imports it.
