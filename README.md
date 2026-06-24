# lingua-clean

A build on top of [lingua](https://github.com/facebookresearch/lingua).

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
   dir, `${CACHE_DIR}` (default `~/lingua-clean-sandbox`):
   - `${CACHE_DIR}/data/dolmino_splits/`             (Dolmino mid-train splits)
   - `${CACHE_DIR}/teachers/OLMo-2-0425-1B-Instruct/`
   - `${CACHE_DIR}/teachers/OLMo-2-1124-7B-Instruct/`
   - `${CACHE_DIR}/students/OLMo-2-0425-1B-stage1-4001B/`     (Lingua DCP root, `STUDENT_INIT_PATH`)
   - `${CACHE_DIR}/students/OLMo-2-0425-1B-stage1-4001B/hf/`  (HF mirror, `STUDENT_HF_PATH`)
   - `${CACHE_DIR}/runs/{midtrain,evals,analysis,slurm_logs}/`

   To start over: `rm -rf ${CACHE_DIR}`. Override individual paths
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
   - `WANDB_ENTITY` — wandb entity for all runs (leave empty to use your
     personal default).

3. Source the env script (must be done in every fresh shell):
   ```bash
   source scripts/env.sh
   ```

4. Download teachers (`OLMo-2-0425-1B-Instruct`, `OLMo-2-1124-7B-Instruct`) and
   the student init checkpoint (`OLMo-2-0425-1B-stage1-4001B`) into
   `${TEACHER_1B_PATH}`, `${TEACHER_7B_PATH}`, and `${STUDENT_INIT_PATH}`
   respectively. Use `setup/download_hf_ckpt.py` or download manually from HuggingFace.

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

