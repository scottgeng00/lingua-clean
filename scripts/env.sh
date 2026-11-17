#!/usr/bin/env bash
# lingua-clean environment configuration. Source before training, eval, or analysis:
#   source scripts/env.sh
# Override any variable by exporting it before sourcing, or edit defaults below.
# All artifacts root under ${CACHE_DIR} → one `rm -rf` cleans everything.
#
#
# Cluster: Slurm account / partition / QoS, GPUs / CPUs / memory per node and cluster-specific paths come from the
# profile lingua/cluster.py detects (Slurm ClusterName, else GPU type; LINGUA_CLUSTER=<profile> forces one). It also
# exports SBATCH_ACCOUNT / SBATCH_PARTITION / SBATCH_QOS so plain `sbatch file.sbatch` lands in the right place.
# Storage: ~/scottg/pretrain must point at this cluster's storage root (PRETRAIN_ROOT); configs use that path.

ROOT_DIR="${ROOT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
_cluster_exports="$(python3 "${ROOT_DIR}/lingua/cluster.py" --shell)" || return 1 2>/dev/null || exit 1
eval "${_cluster_exports}"
unset _cluster_exports

export CACHE_DIR="${CACHE_DIR:-${PRETRAIN_ROOT}/lingua-runs}"

# Data + checkpoints, all under ${PRETRAIN_ROOT}/assets. Override by exporting the var before sourcing.
export DATA_ROOT="${DATA_ROOT:-${PRETRAIN_ROOT}/assets/data/dolmino_splits}"
export TEACHER_1B_PATH="${TEACHER_1B_PATH:-${PRETRAIN_ROOT}/assets/models/OLMo-2-0425-1B-Instruct}"
export TEACHER_7B_PATH="${TEACHER_7B_PATH:-${PRETRAIN_ROOT}/assets/models/OLMo-2-1124-7B-Instruct}"
export STUDENT_INIT_PATH="${STUDENT_INIT_PATH:-${PRETRAIN_ROOT}/assets/models/OLMo-2-0425-1B-stage1-4001B}"  # Lingua DCP root
export STUDENT_HF_PATH="${STUDENT_HF_PATH:-${STUDENT_INIT_PATH}/hf}"                                # HF mirror
export TOKENIZER_PATH="${TOKENIZER_PATH:-${STUDENT_HF_PATH}}"

# Output roots.
export MIDTRAIN_ROOT="${MIDTRAIN_ROOT:-${CACHE_DIR}/runs/midtrain}"
export EVAL_ROOT="${EVAL_ROOT:-${CACHE_DIR}/runs/evals}"
export SLURM_LOG_DIR="${SLURM_LOG_DIR:-${CACHE_DIR}/runs/slurm_logs}"

# Python envs. LINGUA_VENV is the uv venv built by bin/install_requirements.sh.
# OLMES env is separate because it pins different vLLM/transformers versions;
# it may be either a venv or a conda env.
# .venv-kd (bin/install_kd_requirements.sh, per architecture) runs training and evals on every cluster.
export LINGUA_VENV="${LINGUA_VENV:-${ROOT_DIR}/.venv-kd}"
export OLMES_CONDA_ENV="${OLMES_CONDA_ENV:-${PRETRAIN_ROOT}/olmes/.venv-lmeval}"  # OLMES ported to lm_eval 0.4.13 (branch lm-eval-0.4.13)
# FlashAttention 4 and its dependencies, installed outside the venv (see lingua/fa4_attention.py; only the training
# process adds this directory to its sys.path). Set by the cluster profile where FA4 is installed (Blackwell);
# unset, not importable, or a pre-Blackwell GPU -> training uses flash-attn 2.
export LINGUA_FA4_SITE="${LINGUA_FA4_SITE:-}"
# Same for FlashAttention 3 on Hopper (lingua/fa3_attention.py, built by bin/build_fa3_h100.sh).
export LINGUA_FA3_SITE="${LINGUA_FA3_SITE:-}"

# Wandb. Empty → user's default entity.
export WANDB_ENTITY="${WANDB_ENTITY:-}"
# Credentials for this host are in ~/.netrc.
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://meta-fair.wandb.io}"

mkdir -p "${CACHE_DIR}" "${MIDTRAIN_ROOT}" "${EVAL_ROOT}" "${SLURM_LOG_DIR}"

echo "[env] CACHE_DIR=${CACHE_DIR}  (delete with: rm -rf \$CACHE_DIR)"
echo "[env] DATA_ROOT=${DATA_ROOT}"
echo "[env] cluster=${LINGUA_CLUSTER}  SLURM_ACCOUNT=${SLURM_ACCOUNT:-<unset>}  SLURM_PARTITION=${SLURM_PARTITION:-<unset>}  SLURM_QOS=${SLURM_QOS:-<unset>}  GPUS_PER_NODE=${GPUS_PER_NODE}"
