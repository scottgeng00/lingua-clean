#!/bin/bash
#
# Per-token (H_T, H_S) joint analysis to predict idx 143 vs 139 gate overlap.
#
# Source scripts/env.sh first (sets LINGUA_CLEAN_ROOT, SLURM_LOG_DIR,
# ANALYSIS_OUT_DIR, TEACHER_*_PATH, STUDENT_HF_PATH, DATA_ROOT,
# LINGUA_VENV).
#
#
# This file does not set --account / --qos / --partition; site defaults
# apply. If your cluster requires them, either add them to the #SBATCH
# block above or pass them on the command line:
#   sbatch --account="$SLURM_ACCOUNT" --qos="$SLURM_QOS_ANALYSIS" <this file>
#SBATCH --job-name=pertoken-HT-HS
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=1:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -e
: "${LINGUA_CLEAN_ROOT:?source scripts/env.sh first}"
source "${LINGUA_CLEAN_ROOT}/scripts/env.sh"

source "${LINGUA_VENV}/bin/activate"

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

python "${LINGUA_CLEAN_ROOT}/analysis/idx139/per_token_HT_HS.py" \
    --docs-per-source 20 \
    --max-tokens 1024 \
    --temperature 2.0 \
    --gate-quantile 0.30
