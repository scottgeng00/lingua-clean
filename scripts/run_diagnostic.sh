#!/usr/bin/env bash
# scripts/run_diagnostic.sh — fan out the analysis/idx139 diagnostic jobs.
#
# Submits the four sbatch analysis jobs that back the idx139 (entropy-gated RKL)
# results. The plot scripts are cheap and run interactively when --plots is set.
#
# Usage:
#   source scripts/env.sh
#   bash scripts/run_diagnostic.sh                # submit all sbatch jobs
#   bash scripts/run_diagnostic.sh --plots        # also render the figures
#                                                  # (only useful AFTER the
#                                                  # sbatch jobs finish)
#   bash scripts/run_diagnostic.sh --only by_source,per_token   # subset
#   DRY_RUN=1 bash scripts/run_diagnostic.sh      # print sbatch commands only
#
# Job IDs (keys for --only):
#   by_source   teacher_entropy_by_source.sbatch    (go/no-go diagnostic; ~1 GPU·h)
#   per_token   teacher_entropy_per_token.sbatch    (per-token strings + gate exemplars)
#   rlvr1       teacher_entropy_RLVR1.sbatch        (1B RLVR1 teacher counterpart)
#   ht_hs       per_token_HT_HS.sbatch              (joint teacher+student entropy)

set -euo pipefail

: "${TEACHER_7B_PATH:?source scripts/env.sh first}"
: "${STUDENT_HF_PATH:?source scripts/env.sh first}"
: "${DATA_ROOT:?source scripts/env.sh first}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

ONLY="${ONLY:-by_source,per_token,rlvr1,ht_hs}"
RUN_PLOTS=0
for arg in "$@"; do
    case "$arg" in
        --plots) RUN_PLOTS=1 ;;
        --only) shift; ONLY="${1:?--only takes a comma-separated list}" ;;
        --only=*) ONLY="${arg#--only=}" ;;
        *) echo "unknown arg: $arg" >&2; exit 1 ;;
    esac
done

want() { [[ ",${ONLY}," == *",$1,"* ]]; }

ANALYSIS_DIR="${ROOT_DIR}/analysis/idx139"

declare -A SBATCH_FILES=(
    [by_source]="${ANALYSIS_DIR}/teacher_entropy_by_source.sbatch"
    [per_token]="${ANALYSIS_DIR}/teacher_entropy_per_token.sbatch"
    [rlvr1]="${ANALYSIS_DIR}/teacher_entropy_RLVR1.sbatch"
    [ht_hs]="${ANALYSIS_DIR}/per_token_HT_HS.sbatch"
)

# rlvr1 needs the 1B-RLVR1 teacher path; gate it.
if want rlvr1 && [ -z "${TEACHER_1B_RLVR1_PATH:-}" ]; then
    echo "[diagnostic] WARN: TEACHER_1B_RLVR1_PATH unset; skipping rlvr1." >&2
    echo "             (Run with INCLUDE_RLVR1=1 bash scripts/fetch_models.sh first.)" >&2
    ONLY="${ONLY//,rlvr1,/,}"
    ONLY="${ONLY//rlvr1,/}"
    ONLY="${ONLY//,rlvr1/}"
fi

SBATCH_BASE=(sbatch)
[ -n "${SLURM_ACCOUNT:-}" ] && SBATCH_BASE+=(--account="${SLURM_ACCOUNT}")
[ -n "${SLURM_QOS:-}" ] && SBATCH_BASE+=(--qos="${SLURM_QOS}")
[ -n "${SLURM_PARTITION:-}" ] && SBATCH_BASE+=(--partition="${SLURM_PARTITION}")

SUBMITTED=()
for key in by_source per_token rlvr1 ht_hs; do
    want "$key" || continue
    sbatch_file="${SBATCH_FILES[$key]}"
    [ -f "$sbatch_file" ] || { echo "[diagnostic] missing: $sbatch_file" >&2; exit 1; }

    cmd=("${SBATCH_BASE[@]}" "$sbatch_file")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[dry-run] ${cmd[*]}"
        continue
    fi

    echo "[diagnostic] submitting $key"
    out=$("${cmd[@]}")
    echo "             $out"
    # Capture job ID for the optional --plots wait barrier.
    jobid="$(echo "$out" | grep -oE '[0-9]+' | tail -n1 || true)"
    [ -n "$jobid" ] && SUBMITTED+=("$jobid")
done

if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

if [ "$RUN_PLOTS" -eq 1 ]; then
    if [ ${#SUBMITTED[@]} -gt 0 ]; then
        DEP="afterok:$(IFS=:; echo "${SUBMITTED[*]}")"
        echo "[diagnostic] --plots: rendering after $DEP via sbatch --dependency"
        PLOT_CMD=("${SBATCH_BASE[@]}"
            --job-name=idx139-plots
            --output="${SLURM_LOG_DIR:-${HOME}/lingua-runs/slurm_logs}/idx139-plots-%j.out"
            --error="${SLURM_LOG_DIR:-${HOME}/lingua-runs/slurm_logs}/idx139-plots-%j.err"
            --time=00:20:00 --cpus-per-task=4 --mem=32G
            --dependency="$DEP"
            --wrap "set -e; cd '${ROOT_DIR}'; \
                source scripts/env.sh; \
                source \"\${LINGUA_VENV}/bin/activate\"; \
                python analysis/idx139/plot_entropy_histogram.py; \
                python analysis/idx139/visualize_gate.py")
        "${PLOT_CMD[@]}"
    else
        echo "[diagnostic] --plots: no sbatch jobs to wait on; rendering now."
        cd "${ROOT_DIR}"
        python analysis/idx139/plot_entropy_histogram.py
        python analysis/idx139/visualize_gate.py
    fi
fi

echo "[diagnostic] done. Outputs land in ${ANALYSIS_OUT_DIR:-${HOME}/lingua-runs/analysis}/idx139/"
