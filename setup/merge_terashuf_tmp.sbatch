#!/bin/bash
# setup/merge_terashuf_tmp.sbatch — run setup/merge_terashuf_tmp.py on one full
# CPU node, then independently recount the output lines.
#
# Submit from the repo root, e.g. for dclm:
#   source scripts/env.sh
#   TMP_FILES="$DATA_ROOT/tmp_dclm/terashuftmp*" OUT_DIR=$DATA_ROOT/dclm_shuffled PREFIX=dclm \
#   EXPECTED_LINES=606266382 \
#   sbatch --account=$SLURM_ACCOUNT --output=$SLURM_LOG_DIR/%x-%j.out setup/merge_terashuf_tmp.sbatch
#
# EXPECTED_LINES is the "Read N lines" figure from terashuf's log; the job fails
# unless both the merger's count and the recount match it.
#SBATCH --job-name=terashuf-merge
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --qos=cpu_lowest

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source .venv/bin/activate

: "${TMP_FILES:?set TMP_FILES to a glob of terashuf tmp files}"
: "${OUT_DIR:?set OUT_DIR}"
: "${PREFIX:?set PREFIX, e.g. dclm}"
: "${EXPECTED_LINES:?set EXPECTED_LINES to the Read N lines count from the terashuf log}"
NCHUNKS="${NCHUNKS:-16}"

# shellcheck disable=SC2086
python setup/merge_terashuf_tmp.py --tmp-files ${TMP_FILES} --out-dir "${OUT_DIR}" \
    --prefix "${PREFIX}" --nchunks "${NCHUNKS}" --val-docs "${VAL_DOCS:-34}" \
    --seed "${SEED:-42}" --expected-lines "${EXPECTED_LINES}"

echo "[recount] independent wc -l of outputs:"
PAR=$((NCHUNKS + 1))
COUNTS="${OUT_DIR}/${PREFIX}.recount.txt"
ls "${OUT_DIR}/${PREFIX}".chunk.*.jsonl "${OUT_DIR}/${PREFIX}.val.jsonl" | xargs -P "${PAR}" -n 1 wc -l > "${COUNTS}"
cat "${COUNTS}"
n=$(awk '{s += $1} END {print s}' "${COUNTS}")
echo "[recount] total ${n}, expected ${EXPECTED_LINES}"
if [ "${n}" != "${EXPECTED_LINES}" ]; then
    echo "[recount] MISMATCH" >&2
    exit 1
fi
echo "[recount] OK"
