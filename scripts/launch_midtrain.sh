#!/bin/bash
# scripts/launch_midtrain.sh — launch a lingua-clean mid-training run via sbatch.
#
# Usage:
#   RECIPE=<name> bash scripts/launch_midtrain.sh                # submit via sbatch
#   RECIPE=<name> DRY_RUN=1 bash scripts/launch_midtrain.sh      # print the sbatch
#                                                                # command without
#                                                                # submitting
#
# Available RECIPE values mirror the YAML basenames under
# apps/main/configs/recipes/, e.g.:
#   ntp_baseline, fkd_1b, fkd_7b,
#   idx139 (rkl_entgate_q30_lam1p0), idx148 (rkl_entswitch_q30_lam1p0),
#   entband_q30_q30_lam1p0, fkl_top30_lam1p0,
#   fkl_entgate_q30_lam1p0, rkl_randmask_p30_lam1p0,
#   rkl_stentgate_q30_lam1p0,
#   rkl_uniform_lam0p1, rkl_uniform_lam0p3, rkl_uniform_lam0p5, rkl_uniform_lam1p0,
#   rkl_purekd, rkl_fkl_mix_alpha0p5, rkl_fkl_mix_alpha0p8.
#
# Environment overrides:
#   RECIPE=<name>            (required) recipe basename
#   NNODES=<int>             default 4 (matches in-house FP8 fast path)
#   STEPS_OVERRIDE=<int>     default 28800
#   LINGUA_TEACHER_FP8=0|1   default 1 for 7B teachers, 0 for 1B teacher / NTP
#   LINGUA_COMPILE_TEACHER=0|1   default 0
#   DRY_RUN=0|1              if 1, print the sbatch command and exit
#
set -euo pipefail

if [ -z "${RECIPE:-}" ]; then
    echo "ERROR: set RECIPE=<recipe_name>. See scripts/launch_midtrain.sh header." >&2
    exit 1
fi

CLEAN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Source the env script so DATA_ROOT, TEACHER_*_PATH, MIDTRAIN_ROOT, etc. are
# set before sbatch hands them to the inner script (and before downstream tools
# like OmegaConf's ${oc.env:...} resolve them).
source "${CLEAN_ROOT}/scripts/env.sh"

RECIPE_YAML="${CLEAN_ROOT}/apps/main/configs/recipes/${RECIPE}.yaml"

if [ ! -f "${RECIPE_YAML}" ]; then
    echo "ERROR: recipe YAML not found: ${RECIPE_YAML}" >&2
    echo "Available recipes:" >&2
    ls "${CLEAN_ROOT}/apps/main/configs/recipes/" | sed 's/\.yaml$//' >&2
    exit 1
fi

NNODES="${NNODES:-4}"
STEPS_OVERRIDE="${STEPS_OVERRIDE:-28800}"

# FP8 default: enabled for 7B-Instruct teacher recipes, disabled for the
# 1B-Instruct teacher (matches the launch-script defaults in the upstream
# `relaunch_rho1_2m_tokens.sh`). NTP / unspecified-teacher recipes inherit 0.
if [ -z "${LINGUA_TEACHER_FP8:-}" ]; then
    if grep -q "7B-Instruct" "${RECIPE_YAML}"; then
        LINGUA_TEACHER_FP8=1
    else
        LINGUA_TEACHER_FP8=0
    fi
fi
LINGUA_COMPILE_TEACHER="${LINGUA_COMPILE_TEACHER:-0}"

SLURM_LOG_DIR="${SLURM_LOG_DIR:-${HOME}/lingua-runs/slurm_logs}"
mkdir -p "${SLURM_LOG_DIR}"

JOBNAME="lingua_clean_${RECIPE}"

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

# Append cluster-specific flags only when the env vars are set, so collaborators
# on other clusters don't have to edit this file.
if [ -n "${SLURM_ACCOUNT:-}" ]; then
    SBATCH_CMD+=(--account="${SLURM_ACCOUNT}")
fi
if [ -n "${SLURM_QOS:-}" ]; then
    SBATCH_CMD+=(--qos="${SLURM_QOS}")
fi
if [ -n "${SLURM_PARTITION:-}" ]; then
    SBATCH_CMD+=(--partition="${SLURM_PARTITION}")
fi

SBATCH_CMD+=(
    --export=ALL,RECIPE,STEPS_OVERRIDE,LINGUA_TEACHER_FP8,LINGUA_COMPILE_TEACHER,CLEAN_ROOT,RECIPE_YAML,LINGUA_CLEAN_ROOT,DATA_ROOT,TEACHER_1B_PATH,TEACHER_7B_PATH,STUDENT_INIT_PATH,STUDENT_HF_PATH,TOKENIZER_PATH,MIDTRAIN_ROOT,EVAL_ROOT,SLURM_LOG_DIR,LINGUA_CONDA_ENV,OLMES_CONDA_ENV,CONDA_PROFILE_SH,WANDB_ENTITY
    "${CLEAN_ROOT}/scripts/_sbatch_inner.sh"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1; would run:"
    printf '  %q' "${SBATCH_CMD[@]}"
    printf '\n'
    echo
    echo "Inner script: ${CLEAN_ROOT}/scripts/_sbatch_inner.sh"
    echo "Recipe YAML:  ${RECIPE_YAML}"
    exit 0
fi

"${SBATCH_CMD[@]}"
