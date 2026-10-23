#!/bin/bash
# setup/submit_dclm_rewrites.sh — generate 1 rewrite per doc of a DCLM subset with simplescale,
# for each (generator, prompt) pair. Dry run by default: prints the simplescale submit commands.
#
#   bash setup/submit_dclm_rewrites.sh               # dry run
#   SUBMIT=1 bash setup/submit_dclm_rewrites.sh      # submit
#
# Then, per finished run, export the row-aligned training split:
#   .venv/bin/python setup/prepare_dclm_rewrites.py --run <run_dir> \
#       --subset ${DATA}/dclm_subsets/${SUBSET} --details <model>_<prompt>_g1_max${MAX_NEW}
#
# Environment overrides:
#   SUBSET=dclm_8m_1        subset under dclm_subsets/ to rewrite
#   MODELS="2b 9b"          Qwen3.5 sizes (assets/models/Qwen3.5-<SIZE^^>)
#   PROMPTS="wiki guided"   handlers in simplescale.harnesses.rewrite
#   POOLS="h200_comm_shared=64"   space-separated QOS=COUNT worker pools
#   TP=1                    GPUs per worker (2 is plenty for 27B at 16k context)
#   MAX_NEW=4096 MAX_DOC=11000 CONTEXT=16384
set -euo pipefail

SUBSET="${SUBSET:-dclm_8m_1}"
MODELS="${MODELS:-2b 9b}"
PROMPTS="${PROMPTS:-wiki guided}"
POOLS="${POOLS:-h200_comm_shared=64}"
MAX_NEW="${MAX_NEW:-4096}"
MAX_DOC="${MAX_DOC:-11000}"
CONTEXT="${CONTEXT:-16384}"
TP="${TP:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA=/home/sgeng/scottg/pretrain/assets/data
MODELS_DIR=/home/sgeng/scottg/pretrain/assets/models
SS=/checkpoint/dream-sc-3/scottg/pretrain/simplescale
MANIFEST="${SS}/runs/manifests/${SUBSET}.g1.max${MAX_NEW}.manifest.jsonl"

if [ ! -f "${MANIFEST}" ]; then
    "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/setup/build_dclm_rewrite_manifest.py" \
        "${DATA}/dclm_subsets/${SUBSET}" "${MANIFEST}" \
        --max-new-tokens "${MAX_NEW}" --max-document-tokens "${MAX_DOC}"
fi

POOL_ARGS=()
for p in ${POOLS}; do POOL_ARGS+=(--pool "${p}"); done

for size in ${MODELS}; do
    for prompt in ${PROMPTS}; do
        run="${SUBSET}-qwen35-${size}-${prompt}-g1"
        cmd=(
            "${SS}/.venv/bin/simplescale" submit "${MANIFEST}"
            --model "${MODELS_DIR}/Qwen3.5-${size^^}"
            --handler "simplescale.harnesses.rewrite:${prompt}"
            --run-dir "${SS}/runs/${run}"
            "${POOL_ARGS[@]}"
            --tp "${TP}" --context-length "${CONTEXT}"
            --wandb-project simplescale --job-name "ss-${run}"
            --repo "${SS}"
        )
        [ "${SUBMIT:-0}" = "1" ] || cmd+=(--dry-run)
        echo "== ${run} -> export as ${SUBSET}_qwen35_${size}_${prompt}_g1_max${MAX_NEW}"
        (cd "${SS}" && "${cmd[@]}")
    done
done
