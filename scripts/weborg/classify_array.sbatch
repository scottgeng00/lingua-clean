#!/bin/bash
# One-GPU array over document shards: WebOrganizer topic + format classification (scripts/weborg/classify.py).
#   sbatch --array=0-39 scripts/weborg/classify_array.sbatch STREAM_DIR OUT_DIR NUM_SHARDS [extra args]
# Account / partition / QoS: from the cluster profile (source scripts/env.sh exports SBATCH_*); add -q "$SLURM_HIGH_QOS" for high priority.
#SBATCH --job-name=weborg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/home/sgeng/scottg/pretrain/logs/weborg/%x-%A_%a.out
STREAM=$1; OUT=$2; NSH=$3; shift 3
cd /home/sgeng/scottg/pretrain/lingua-clean
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 RAYON_NUM_THREADS=16 TOKENIZERS_PARALLELISM=true
.venv-kd/bin/python scripts/weborg/classify.py --stream "$STREAM" --out "$OUT" --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$NSH" "$@"
