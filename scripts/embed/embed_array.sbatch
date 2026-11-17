#!/bin/bash
# One-GPU array job over document shards of one pre-tokenized corpus. Per shard: Qwen3-Embedding-0.6B embeddings
# (embed_tokens.py) then the FineWeb-Edu classifier (quality_edu.py) on the GPU, and the DCLM fastText quality score
# (quality_dclm.py) on the CPUs in parallel. Each step skips its output if it already exists.
#   sbatch --array=0-39 scripts/embed/embed_array.sbatch STREAM_DIR OUT_DIR NUM_SHARDS [extra args, e.g. --docs N]
# Account / partition / QoS: from the cluster profile (source scripts/env.sh exports SBATCH_*); add -q "$SLURM_HIGH_QOS" for high priority.
#SBATCH --job-name=embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/home/sgeng/scottg/pretrain/logs/embed/%x-%A_%a.out

STREAM=$1; OUT=$2; NSH=$3; shift 3
cd /home/sgeng/scottg/pretrain/lingua-clean
export PYTHONPATH=$PWD OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
ARGS=(--stream "$STREAM" --out "$OUT" --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$NSH" "$@")
/home/sgeng/scottg/pretrain/venvs/quality/bin/python scripts/embed/quality_dclm.py "${ARGS[@]}" --procs 6 &
FT=$!
.venv-kd/bin/python scripts/embed/embed_tokens.py "${ARGS[@]}" || exit 1
.venv-kd/bin/python scripts/embed/quality_edu.py "${ARGS[@]}" || exit 1
wait $FT
