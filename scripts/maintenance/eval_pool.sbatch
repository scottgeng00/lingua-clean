#!/bin/bash
#SBATCH --job-name=eval_pool
#SBATCH --account=mnm
#SBATCH --qos=g3_mnm_high
#SBATCH --partition=g3
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=700G
#SBATCH --time=08:00:00
#SBATCH --output=/checkpoint/mnm/sgeng/pretrain/logs/evalpool/%j.out
# 8 GPUs (2 nodes x 4) on g3_mnm_high, user-approved cap. Steals pending olmes_evals jobs (see eval_pool.py).
source /checkpoint/mnm/sgeng/pretrain/lingua-clean/scripts/env.sh >/dev/null
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}
srun --ntasks-per-node=1 /checkpoint/mnm/sgeng/pretrain/lingua-clean/.venv-kd/bin/python -u \
    /checkpoint/mnm/sgeng/pretrain/lingua-clean/scripts/maintenance/eval_pool.py --hours 8 --gpus 4
