#!/usr/bin/env bash
# lingua-clean environment configuration.
# Source this before running training, eval, or analysis:
#   source scripts/env.sh
#
# Override any of these variables in your shell before sourcing if you
# want a different layout. By default ALL downloads, run outputs, eval
# artifacts, and slurm logs are rooted at ${LINGUA_SANDBOX_ROOT} so a
# single `rm -rf ${LINGUA_SANDBOX_ROOT}` cleans everything up.

# === Repository root ===
export LINGUA_CLEAN_ROOT="${LINGUA_CLEAN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# === Sandbox root ===
# Single deletable dir that holds every artifact this repo writes: HF
# downloads, Lingua DCP checkpoints, Dolmino splits, training dumps, eval
# artifacts, analysis outputs, slurm logs. Override to share data across
# sandboxes (e.g. point DATA_ROOT/TEACHER_*_PATH at a long-lived cache and
# keep MIDTRAIN_ROOT inside the sandbox).
export LINGUA_SANDBOX_ROOT="${LINGUA_SANDBOX_ROOT:-${HOME}/lingua-clean-sandbox}"

# === Data ===
# Dolmino mid-train tokenized splits (~2 TB shuffled corpus).
# See README for download instructions.
export DATA_ROOT="${DATA_ROOT:-${LINGUA_SANDBOX_ROOT}/data/dolmino_splits}"

# === Teachers ===
# Pre-downloaded HF checkpoints of the teacher models (loaded via
# `AutoModelForCausalLM.from_pretrained`, so these point at HF dirs directly).
# See README for hf-hub download commands.
export TEACHER_1B_PATH="${TEACHER_1B_PATH:-${LINGUA_SANDBOX_ROOT}/teachers/OLMo-2-0425-1B-Instruct}"
export TEACHER_7B_PATH="${TEACHER_7B_PATH:-${LINGUA_SANDBOX_ROOT}/teachers/OLMo-2-1124-7B-Instruct}"

# RLVR1-stage teacher used by the idx139 analysis scripts (1B same-family teacher).
export TEACHER_1B_RLVR1_PATH="${TEACHER_1B_RLVR1_PATH:-${LINGUA_SANDBOX_ROOT}/teachers/OLMo-2-0425-1B-RLVR1}"

# === Student init ===
# STUDENT_INIT_PATH is the Lingua DCP checkpoint root (contains `.metadata`
# and `__0_0.distcp`). Fed to `checkpoint.init_ckpt_path` in the recipe YAMLs
# and consumed by `lingua.checkpoint.load_from_checkpoint`.
# STUDENT_HF_PATH is the HF-format mirror (config.json, *.safetensors,
# tokenizer) used by the OLMES eval path, the idx139 analysis scripts, and
# anywhere else `AutoModelForCausalLM.from_pretrained` / `AutoTokenizer.from_pretrained`
# is called on the student.
export STUDENT_INIT_PATH="${STUDENT_INIT_PATH:-${LINGUA_SANDBOX_ROOT}/students/OLMo-2-0425-1B-stage1-4001B}"
export STUDENT_HF_PATH="${STUDENT_HF_PATH:-${STUDENT_INIT_PATH}/hf}"

# === Tokenizer ===
# Defaults to the student HF dir (which ships the tokenizer alongside weights).
export TOKENIZER_PATH="${TOKENIZER_PATH:-${STUDENT_HF_PATH}}"

# === Output roots ===
export MIDTRAIN_ROOT="${MIDTRAIN_ROOT:-${LINGUA_SANDBOX_ROOT}/runs/midtrain}"
export EVAL_ROOT="${EVAL_ROOT:-${LINGUA_SANDBOX_ROOT}/runs/evals}"
export ANALYSIS_OUT_DIR="${ANALYSIS_OUT_DIR:-${LINGUA_SANDBOX_ROOT}/runs/analysis}"
export SLURM_LOG_DIR="${SLURM_LOG_DIR:-${LINGUA_SANDBOX_ROOT}/runs/slurm_logs}"

# === Conda envs (caller-specific) ===
# `LINGUA_CONDA_ENV` is the training/eval env (torch + transformers + lingua deps).
# Default `lingua` reuses the existing env from the original lingua repo — no
# need to bootstrap a separate `lingua-clean` env. Override if you've built one.
# `OLMES_CONDA_ENV` is the vLLM/OLMES eval env (a separate environment because
# OLMES pins different vLLM/transformers versions than the training env).
export LINGUA_CONDA_ENV="${LINGUA_CONDA_ENV:-lingua}"
export OLMES_CONDA_ENV="${OLMES_CONDA_ENV:-/checkpoint/comem/jacquelinehe/miniconda3/envs/olmes}"

# Path to conda's `etc/profile.d/conda.sh` for `_sbatch_inner.sh`.
# Cluster-default points at the comem miniconda; override for elsewhere.
export CONDA_PROFILE_SH="${CONDA_PROFILE_SH:-/checkpoint/comem/jacquelinehe/miniconda3/etc/profile.d/conda.sh}"

# === SLURM submission knobs ===
# Cluster defaults are baked in below. Override by exporting before sourcing
# this file (or by editing them in place if you're permanently on a different
# cluster). Leave a value empty to omit the flag and let slurm pick its site
# default.
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-comem}"
export SLURM_QOS="${SLURM_QOS:-h100_comem_high}"             # production tier, full-size runs
# QOS for short jobs (smoke test, idx139 analysis, run_diagnostic.sh).
# Falls back to SLURM_QOS if unset.
export SLURM_QOS_TEST="${SLURM_QOS_TEST:-h100_comem_high}"   # dev/test tier
export SLURM_QOS_ANALYSIS="${SLURM_QOS_ANALYSIS:-${SLURM_QOS_TEST:-${SLURM_QOS}}}"

# === Weights & Biases ===
# Unset by default — wandb falls back to the user's default entity. Set this
# if you want runs to log under a shared team entity.
export WANDB_ENTITY="${WANDB_ENTITY:-}"

mkdir -p "${LINGUA_SANDBOX_ROOT}" "${MIDTRAIN_ROOT}" "${EVAL_ROOT}" "${ANALYSIS_OUT_DIR}" "${SLURM_LOG_DIR}"

echo "[env] LINGUA_SANDBOX_ROOT=${LINGUA_SANDBOX_ROOT}"
echo "[env] LINGUA_CLEAN_ROOT=${LINGUA_CLEAN_ROOT}"
echo "[env] DATA_ROOT=${DATA_ROOT}"
echo "[env] TEACHER_1B_PATH=${TEACHER_1B_PATH}"
echo "[env] TEACHER_7B_PATH=${TEACHER_7B_PATH}"
echo "[env] STUDENT_INIT_PATH=${STUDENT_INIT_PATH}"
echo "[env] STUDENT_HF_PATH=${STUDENT_HF_PATH}"
echo "[env] MIDTRAIN_ROOT=${MIDTRAIN_ROOT}"
echo "[env] SLURM_ACCOUNT=${SLURM_ACCOUNT:-<unset>}  SLURM_QOS=${SLURM_QOS:-<unset>}  SLURM_QOS_TEST=${SLURM_QOS_TEST:-<unset>}"
echo "[env] (delete with: rm -rf ${LINGUA_SANDBOX_ROOT})"
