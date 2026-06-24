#!/bin/bash
# scripts/launch_pretrain.sh — launch a from-scratch lingua-clean pretraining
# run via sbatch.
#
# This is a small wrapper around the same `_sbatch_inner.sh` that
# `launch_midtrain.sh` uses; the only differences are (1) defaults are tuned
# for pretraining (fewer nodes, fewer steps), and (2) the YAML lives in
# `apps/main/configs/<CONFIG>.yaml` (top-level configs/) rather than
# `apps/main/configs/recipes/<RECIPE>.yaml`. This keeps the from-scratch
# `dolmino_pretrain.yaml` separate from the mid-training KD recipe matrix.
#
# Usage:
#   CONFIG=dolmino_pretrain bash scripts/launch_pretrain.sh                # submit
#   CONFIG=dolmino_pretrain DRY_RUN=1 bash scripts/launch_pretrain.sh      # print sbatch
#
# Environment overrides:
#   CONFIG=<name>           default `dolmino_pretrain`. Resolves to apps/main/configs/<CONFIG>.yaml.
#   NNODES=<int>            default 2 (matches the original pretrain_from_scratch_run.sh).
#   STEPS_OVERRIDE=<int>    default 9600 (~26B Dolmino tokens at the configured batch/seq).
#   DRY_RUN=0|1             if 1, print the sbatch command and exit.

set -euo pipefail

CONFIG="${CONFIG:-dolmino_pretrain}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/env.sh"

CONFIG_YAML="${ROOT_DIR}/apps/main/configs/${CONFIG}.yaml"

if [ ! -f "${CONFIG_YAML}" ]; then
    echo "ERROR: pretrain config not found: ${CONFIG_YAML}" >&2
    echo "Available top-level configs:" >&2
    ls "${ROOT_DIR}/apps/main/configs/" 2>/dev/null | grep -E '\.yaml$' | sed 's/\.yaml$//' >&2
    exit 1
fi

NNODES="${NNODES:-2}"
STEPS_OVERRIDE="${STEPS_OVERRIDE:-9600}"

# From-scratch pretraining has no teacher, so the FP8/compile toggles are
# vestigial here. Kept at 0 so the inner script doesn't try to FP8-convert
# something that doesn't exist.
LINGUA_TEACHER_FP8=0
LINGUA_COMPILE_TEACHER=0

SLURM_LOG_DIR="${SLURM_LOG_DIR:-${HOME}/lingua-runs/slurm_logs}"
mkdir -p "${SLURM_LOG_DIR}"

JOBNAME="lingua_clean_pt_${CONFIG}"

SBATCH_CMD=(
    sbatch
    --job-name="${JOBNAME}"
    --output="${SLURM_LOG_DIR}/${JOBNAME}-%j.out"
    --error="${SLURM_LOG_DIR}/${JOBNAME}-%j.err"
    --time=122:00:00
    --cpus-per-task=64
    --nodes="${NNODES}"
    --ntasks-per-node=1
    --gres=gpu:8
    --mem=456G
    --requeue
)

if [ -n "${SLURM_ACCOUNT:-}" ]; then
    SBATCH_CMD+=(--account="${SLURM_ACCOUNT}")
fi
if [ -n "${SLURM_QOS:-}" ]; then
    SBATCH_CMD+=(--qos="${SLURM_QOS}")
fi
if [ -n "${SLURM_PARTITION:-}" ]; then
    SBATCH_CMD+=(--partition="${SLURM_PARTITION}")
fi

# The inner script keys off RECIPE / RECIPE_YAML — pass our pretrain config
# values through those names so we don't have to fork the inner script.
RECIPE="${CONFIG}"
RECIPE_YAML="${CONFIG_YAML}"

SBATCH_CMD+=(
    --export=ALL,RECIPE,STEPS_OVERRIDE,LINGUA_TEACHER_FP8,LINGUA_COMPILE_TEACHER,ROOT_DIR,RECIPE_YAML,DATA_ROOT,TEACHER_1B_PATH,TEACHER_7B_PATH,STUDENT_INIT_PATH,STUDENT_HF_PATH,TOKENIZER_PATH,MIDTRAIN_ROOT,EVAL_ROOT,SLURM_LOG_DIR,LINGUA_CONDA_ENV,OLMES_CONDA_ENV,WANDB_ENTITY
    "${ROOT_DIR}/scripts/_sbatch_inner.sh"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1; would run:"
    printf '  %q' "${SBATCH_CMD[@]}"
    printf '\n'
    echo
    echo "Inner script: ${ROOT_DIR}/scripts/_sbatch_inner.sh"
    echo "Config YAML:  ${CONFIG_YAML}"
    exit 0
fi

"${SBATCH_CMD[@]}"
