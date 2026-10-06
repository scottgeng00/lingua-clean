#!/usr/bin/env bash
# lingua-clean environment configuration. Source before training, eval, or analysis:
#   source scripts/env.sh
# Override any variable by exporting it before sourcing, or edit defaults below.
# All artifacts root under ${CACHE_DIR} → one `rm -rf` cleans everything.
#
# === OFF-CLUSTER USERS: MUST OVERRIDE THESE BEFORE SOURCING ===
#   DATA_ROOT          path to your Dolmino splits dir (~2 TB; don't put under $HOME)
#   SLURM_ACCOUNT      defaulted to `comem` — wrong elsewhere; sbatch will reject
#   SLURM_QOS          defaulted to `h100_comem_high` — wrong elsewhere
#   OLMES_CONDA_ENV    only if you'll run OLMES evals (mid-train recipes do, every 1200 steps)

# export CACHE_DIR="${CACHE_DIR:-${HOME}/lingua-clean-sandbox}"
export CACHE_DIR="/checkpoint/dream-sc-3/scottg/pretrain/lingua-runs"

# Data + checkpoints, all under /home/sgeng/scottg/pretrain/assets. Override by exporting the var before sourcing.
export DATA_ROOT="${DATA_ROOT:-/home/sgeng/scottg/pretrain/assets/data/dolmino_splits}"
export TEACHER_1B_PATH="${TEACHER_1B_PATH:-/home/sgeng/scottg/pretrain/assets/models/OLMo-2-0425-1B-Instruct}"
export TEACHER_7B_PATH="${TEACHER_7B_PATH:-/home/sgeng/scottg/pretrain/assets/models/OLMo-2-1124-7B-Instruct}"
export STUDENT_INIT_PATH="${STUDENT_INIT_PATH:-/home/sgeng/scottg/pretrain/assets/models/OLMo-2-0425-1B-stage1-4001B}"  # Lingua DCP root
export STUDENT_HF_PATH="${STUDENT_HF_PATH:-${STUDENT_INIT_PATH}/hf}"                                # HF mirror
export TOKENIZER_PATH="${TOKENIZER_PATH:-${STUDENT_HF_PATH}}"

# Output roots.
export MIDTRAIN_ROOT="${MIDTRAIN_ROOT:-${CACHE_DIR}/runs/midtrain}"
export EVAL_ROOT="${EVAL_ROOT:-${CACHE_DIR}/runs/evals}"
export SLURM_LOG_DIR="${SLURM_LOG_DIR:-${CACHE_DIR}/runs/slurm_logs}"

# Python envs. LINGUA_VENV is the uv venv built by bin/install_requirements.sh.
# OLMES env is separate because it pins different vLLM/transformers versions;
# it may be either a venv or a conda env.
export LINGUA_VENV="${LINGUA_VENV:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/.venv}"
export OLMES_CONDA_ENV="${OLMES_CONDA_ENV:-/home/sgeng/scottg/pretrain/olmes/.venv}"

# SLURM. Leave a value empty to omit the flag and use the site default.
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-dream}"
export SLURM_QOS="${SLURM_QOS:-h200_dream_high}"

# Wandb. Empty → user's default entity.
export WANDB_ENTITY="${WANDB_ENTITY:-}"

mkdir -p "${CACHE_DIR}" "${MIDTRAIN_ROOT}" "${EVAL_ROOT}" "${SLURM_LOG_DIR}"

echo "[env] CACHE_DIR=${CACHE_DIR}  (delete with: rm -rf \$CACHE_DIR)"
echo "[env] DATA_ROOT=${DATA_ROOT}"
echo "[env] SLURM_ACCOUNT=${SLURM_ACCOUNT:-<unset>}  SLURM_QOS=${SLURM_QOS:-<unset>}"
