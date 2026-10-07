#!/bin/bash
# scripts/_sbatch_inner.sh — runs INSIDE sbatch / srun. Sets up env, then
# torchruns apps.main.train with the chosen recipe YAML.
#
# Expects launch_midtrain.sh to have sourced scripts/env.sh first, so
# LINGUA_VENV / etc. are already exported into this script's environment
# via sbatch --export=ALL,...
set -euo pipefail

# sbatch runs a spooled copy of this file, so BASH_SOURCE points into
# /var/spool; prefer the ROOT_DIR exported by the launcher, then the submit dir.
ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"

# Re-source env.sh as a belt-and-braces guard for re-runs from an interactive
# session where the caller forgot to source it.
source "${ROOT_DIR}/scripts/env.sh"

# shellcheck disable=SC1091
source "${LINGUA_VENV}/bin/activate"

# Wandb / tuning env — mirror the in-house relaunch_rho1_2m_tokens.sh defaults.
# wandb credentials come from ~/.netrc for the host in ~/.config/wandb/settings;
# an exported WANDB_API_KEY would take precedence even if it belongs to another
# wandb host (e.g. public api.wandb.ai vs a self-hosted server). Set
# WANDB_USE_ENV_KEY=1 to use the environment key instead.
if [ "${WANDB_USE_ENV_KEY:-0}" != "1" ]; then
    unset WANDB_API_KEY
fi
# Online by default (needs wandb credentials, see above); WANDB_MODE=offline logs locally only.
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_INIT_TIMEOUT=300
export WANDB__SERVICE_WAIT=300
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8

NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}
NNODES=${SLURM_NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-$((29501 + RANDOM % 5000))}
NODE_RANK=${SLURM_NODEID:-0}

if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
fi

echo "NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NODE_RANK=$NODE_RANK"
echo "RECIPE=$RECIPE CONFIG=$RECIPE_YAML STEPS=$STEPS_OVERRIDE FP8=$LINGUA_TEACHER_FP8 COMPILE=$LINGUA_COMPILE_TEACHER"

cd "${ROOT_DIR}"

TRAIN_CMD=(torchrun
    --nnodes="${NNODES}"
    --nproc-per-node="${NPROC_PER_NODE}"
    --node-rank="${NODE_RANK}"
    --master-addr="${MASTER_ADDR}"
    --master-port="${MASTER_PORT}"
    -m apps.main.train
    "config=${RECIPE_YAML}"
    "steps=${STEPS_OVERRIDE}")

if [ -n "${SLURM_JOB_NODELIST:-}" ] && [ "$NNODES" -gt 1 ]; then
    # Multi-node fan-out (mirrors the in-house launcher). Each task must use its
    # own rank, so swap the already-expanded --node-rank for a literal
    # ${SLURM_PROCID} that the per-task shell resolves.
    RANK_ARG="--node-rank=\${SLURM_PROCID}"
    MULTI_CMD=("${TRAIN_CMD[@]/#--node-rank=*/$RANK_ARG}")
    srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 \
        --export=ALL,NPROC_PER_NODE,NNODES,MASTER_ADDR,MASTER_PORT,LINGUA_TEACHER_FP8,LINGUA_COMPILE_TEACHER,RECIPE,RECIPE_YAML,STEPS_OVERRIDE,ROOT_DIR,DATA_ROOT,TEACHER_1B_PATH,TEACHER_7B_PATH,STUDENT_INIT_PATH,STUDENT_HF_PATH,TOKENIZER_PATH,MIDTRAIN_ROOT,EVAL_ROOT,SLURM_LOG_DIR,LINGUA_VENV,OLMES_CONDA_ENV,WANDB_API_KEY,WANDB_MODE,WANDB_ENTITY,CUDA_DEVICE_MAX_CONNECTIONS,TORCH_NCCL_AVOID_RECORD_STREAMS,PYTORCH_CUDA_ALLOC_CONF \
        bash -c "export NODE_RANK=\${SLURM_PROCID}; ${MULTI_CMD[*]}"
else
    "${TRAIN_CMD[@]}"
fi
