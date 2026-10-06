#!/usr/bin/env bash
# bin/install_requirements.sh
#
# Set up the uv-managed venv for lingua-clean training, eval, and analysis.
# Dependencies live in pyproject.toml and are pinned in uv.lock; this just runs
# `uv sync` and sanity-checks imports. Safe to re-run -- uv only installs
# what's missing or out of date.
#
# Pre-reqs:
#   * uv is installed and on PATH (curl -LsSf https://astral.sh/uv/install.sh | sh).
#     uv downloads Python 3.11 itself if the host doesn't have it.
#   * CUDA 12.1-compatible drivers on the host (matches the torch wheels).
#
# Usage:
#   bash bin/install_requirements.sh                  # venv at $LINGUA_VENV (default: <repo>/.venv)
#   LINGUA_VENV=/path/to/venv bash bin/install_requirements.sh
#
# Afterwards, activate the venv yourself for interactive work:
#   source .venv/bin/activate        (or prefix commands with `uv run`)
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LINGUA_VENV="${LINGUA_VENV:-${ROOT_DIR}/.venv}"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not on PATH. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# uv's cache usually sits on a different filesystem than /checkpoint, where
# hardlinks fail; copy instead of warning on every install.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_PROJECT_ENVIRONMENT="${LINGUA_VENV}"

echo "[install] ROOT_DIR=${ROOT_DIR}"
echo "[install] LINGUA_VENV=${LINGUA_VENV}"
echo "[install] uv sync --frozen"
uv sync --frozen --project "${ROOT_DIR}"

echo
echo "[install] Done. Sanity-check imports:"
"${LINGUA_VENV}/bin/python" - <<'PY'
import importlib, sys
mods = [
    "torch", "xformers", "flash_attn",
    "transformers", "tokenizers", "huggingface_hub", "datasets",
    "omegaconf", "wandb", "tiktoken", "datatrove",
    "numpy", "pandas", "matplotlib",
]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, repr(e)))
if missing:
    print("MISSING / broken:")
    for m, e in missing:
        print(f"  - {m}: {e}")
    sys.exit(1)
print("All checked imports OK.")
PY

echo "[install] Finished installing lingua-clean requirements into ${LINGUA_VENV}."
echo "[install] Activate it for interactive work:  source ${LINGUA_VENV}/bin/activate"
