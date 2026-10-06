#!/bin/bash
# scripts/prepare_dolmino.sbatch — download allenai/dolmino-mix-1124 and shuffle
# it into per-domain chunks under ${DATA_ROOT} via setup/download_prepare_dolmino.py.
#
# Submit from the repo root (account/log path come from scripts/env.sh; qos is set below):
#   source scripts/env.sh
#   sbatch --account=$SLURM_ACCOUNT \
#       --output=$SLURM_LOG_DIR/%x-%j.out scripts/prepare_dolmino.sbatch
#
# Environment overrides (export before sbatch):
#   NCHUNKS=<int>      chunks per domain, default 16. Must be <= and divide the
#                      training GPU count, or the loader drops files (16 → 2 or 4 nodes).
#   MEMORY_GB=<int>    terashuf RAM budget, default 512 (keep below --mem).
#   DOMAINS="a b"      subset of: dclm flan math pes2o stackexchange wiki (default all).
#   EXTRA_ARGS="..."   passed through, e.g. "--skip-download" when resuming.
#SBATCH --job-name=dolmino-prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=72:00:00
#SBATCH --qos=cpu_lowest

set -euo pipefail

# sbatch runs a spooled copy of this file, so locate the repo via the submit dir.
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f "${ROOT_DIR}/scripts/env.sh" ]; then
    echo "ERROR: submit from the lingua-clean repo root (no scripts/env.sh in ${ROOT_DIR})" >&2
    exit 1
fi
cd "${ROOT_DIR}"
source scripts/env.sh
# shellcheck disable=SC1091
source "${LINGUA_VENV}/bin/activate"

NCHUNKS="${NCHUNKS:-16}"
MEMORY_GB="${MEMORY_GB:-512}"
DOMAINS="${DOMAINS:-dclm flan math pes2o stackexchange wiki}"

# terashuf spills to $TMPDIR; /tmp on the node can't hold ~2 TB.
export TMPDIR="${DATA_ROOT}/tmp"
mkdir -p "${DATA_ROOT}" "${TMPDIR}"

echo "[prep] DATA_ROOT=${DATA_ROOT} NCHUNKS=${NCHUNKS} MEMORY_GB=${MEMORY_GB} DOMAINS=${DOMAINS}"
# shellcheck disable=SC2086
python setup/download_prepare_dolmino.py \
    --data-dir "${DATA_ROOT}" \
    --memory "${MEMORY_GB}" \
    --nchunks "${NCHUNKS}" \
    ${EXTRA_ARGS:-}

echo "[prep] chunk counts:"
for d in ${DOMAINS}; do
    echo "  ${d}: $(ls "${DATA_ROOT}/${d}_shuffled/"*.chunk.*.jsonl 2>/dev/null | wc -l) chunks, val: $(ls "${DATA_ROOT}/${d}_shuffled/"*.val.jsonl 2>/dev/null | wc -l)"
done
