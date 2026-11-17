#!/bin/bash
# Account / partition / QoS: from the cluster profile (source scripts/env.sh exports SBATCH_*); add -q "$SLURM_HIGH_QOS" for high priority.
#SBATCH --job-name=eval_pool
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=700G
#SBATCH --time=08:00:00
#SBATCH --output=/home/sgeng/scottg/pretrain/logs/evalpool/%j.out
# Steals pending olmes_evals jobs (see eval_pool.py). Defaults: 8 GPUs as 2 nodes x 4 (Dalton). Elsewhere size it from the
# profile, e.g. on 8-GPU nodes: sbatch -N 1 --gres=gpu:"$GPUS_PER_NODE" scripts/maintenance/eval_pool.sbatch
source /home/sgeng/scottg/pretrain/lingua-clean/scripts/env.sh >/dev/null
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}
srun --ntasks-per-node=1 /home/sgeng/scottg/pretrain/lingua-clean/.venv-kd/bin/python -u \
    /home/sgeng/scottg/pretrain/lingua-clean/scripts/maintenance/eval_pool.py --hours 8 --gpus "${SLURM_GPUS_ON_NODE:-4}"
