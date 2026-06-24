#!/usr/bin/env bash
# bin/install_requirements.sh
#
# Install the Python deps required for lingua-clean training, eval, and
# analysis. Modelled on BoLT/bin/install_requirements.sh but without the
# data-prep section -- this script ONLY installs packages.
#
# Pre-reqs:
#   * conda env is already created and activated, e.g.:
#       conda create -n lingua-clean python=3.11 -y
#       conda activate lingua-clean
#   * CUDA 12.1-compatible drivers on the host (matches the torch wheels below).
#
# Usage:
#   conda activate lingua-clean
#   bash bin/install_requirements.sh
#
# Skips work that has already been done -- safe to re-run.
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: no conda env active. Activate one first:" >&2
    echo "  conda create -n lingua-clean python=3.11 -y && conda activate lingua-clean" >&2
    exit 1
fi

CLEAN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[install] CONDA_PREFIX=${CONDA_PREFIX}"
echo "[install] CLEAN_ROOT=${CLEAN_ROOT}"

# --- 1. Lingua deps from requirements.txt (everything except the CUDA wheels) ---
echo "[install] pip install -r requirements.txt"
pip install -r "${CLEAN_ROOT}/requirements.txt"

# --- 2. Torch + xformers (CUDA 12.1 wheels, matching the in-house setup) ---
echo "[install] torch 2.5.0 + xformers 0.0.28.post2 (cu121)"
pip install torch==2.5.0 xformers==0.0.28.post2 \
    --index-url https://download.pytorch.org/whl/cu121

# --- 3. Flash-attention (needs torch pre-installed) ---
echo "[install] flash-attn 2.7.4.post1"
pip install flash-attn==2.7.4.post1 --no-build-isolation

echo
echo "[install] Done. Sanity-check imports:"
python - <<'PY'
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

echo "[install] Finished installing lingua-clean requirements."
