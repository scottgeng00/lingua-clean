#!/bin/bash
# setup/submit_dclm_multirewrites.sh — multi-rewrite study generation: Qwen3.5-9B rewrites of the
# first 4M rows of dclm_8m_1, one simplescale run per method (manifests from
# setup/build_multirewrite_manifests.py). Dry run by default.
#
#   bash setup/submit_dclm_multirewrites.sh                       # dry run, all methods
#   SUBMIT=1 METHODS="iid t12" bash setup/submit_dclm_multirewrites.sh
#
# Then per finished run: .venv/bin/python setup/prepare_dclm_multirewrites.py --method <m> --run <run_dir>
#
# Environment overrides:
#   METHODS="iid t12 gacha nemo wofmt"
#   POOLS_<method>="<high pool>=64 <bulk pool>=128"   worker pools per method (default: the cluster profile's)
#   TAG=rewrite_first4m      manifest / run-dir tag
set -euo pipefail

METHODS="${METHODS:-iid t12 gacha nemo wofmt}"
TAG="${TAG:-rewrite_first4m}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SS="$(cd -- "${ROOT_DIR}/../simplescale" && pwd)"
MODEL=/home/sgeng/scottg/pretrain/assets/models/Qwen3.5-9B

declare -A HANDLER=(
    [iid]=simplescale.harnesses.rewrite:wiki [t12]=simplescale.harnesses.rewrite:wiki
    [gacha]=simplescale.harnesses.gacha:rewrite_wiki
    [nemo]=simplescale.harnesses.rewrite:nemotron [wofmt]=simplescale.harnesses.rewrite:wofmt
)
# Gacha's decision turns accumulate the source plus four 32-option lists in one conversation.
declare -A CONTEXT=([iid]=16384 [t12]=16384 [gacha]=32768 [nemo]=16384 [wofmt]=16384)
# Gacha tasks are ~6 sequential turns x 4 parallel samples, and Qwen3.5's hybrid (GDN) state caps a
# worker at ~210 running requests: keep ~4 x 64 requests in flight in small chunks so chunks finish
# every few minutes instead of all together (workers here are preempted every ~20-30 min).
declare -A EXTRA=([gacha]="--chunk-size 8 --task-concurrency 64")
# Default worker pools follow the cluster profile (simplescale/clusters.py): high-priority + bulk (preemptible).
read -r HIGH DEF < <("${SS}/.venv/bin/python" -c \
    'from simplescale.clusters import detect; c = detect(); print(c.high_pool or c.bulk_pool, c.bulk_pool)')
declare -A POOLS=(
    [iid]="${POOLS_iid:-${HIGH}=64 ${DEF}=128}"
    [t12]="${POOLS_t12:-${HIGH}=128 ${DEF}=128}"
    [gacha]="${POOLS_gacha:-${HIGH}=256 ${DEF}=512}"
    [nemo]="${POOLS_nemo:-${HIGH}=32 ${DEF}=128}"
    [wofmt]="${POOLS_wofmt:-${HIGH}=32 ${DEF}=128}"
)

for m in ${METHODS}; do
    manifest="${SS}/runs/manifests/dclm_8m_1_${TAG}.${m}.manifest.jsonl"
    [ -f "${manifest}" ] || { echo "missing ${manifest}; run setup/build_multirewrite_manifests.py" >&2; exit 1; }
    run="dclm_8m_1_${TAG}-qwen35-9b-${m}"
    pool_args=()
    for p in ${POOLS[$m]}; do pool_args+=(--pool "${p}"); done
    cmd=(
        "${SS}/.venv/bin/simplescale" submit "${manifest}"
        --model "${MODEL}" --handler "${HANDLER[$m]}" --run-dir "${SS}/runs/${run}"
        "${pool_args[@]}" --tp 1 --context-length "${CONTEXT[$m]}"
        --wandb-project simplescale --job-name "ss-${run}" --repo "${SS}"
        ${EXTRA[$m]:-}
    )
    [ "${SUBMIT:-0}" = "1" ] || cmd+=(--dry-run)
    echo "== ${run} (${POOLS[$m]})"
    (cd "${SS}" && "${cmd[@]}")
done
