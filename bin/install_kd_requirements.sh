#!/usr/bin/env bash
# bin/install_kd_requirements.sh — build .venv-kd, the environment for runs whose
# online teacher needs a newer transformers than the main .venv (e.g. Qwen3.5,
# transformers >= 5, which itself needs torch >= 2.6).
#
# Same direct dependencies as pyproject.toml, except:
#   torch 2.8.0 (cu128), xformers 0.0.32.post2, flash-attn 2.8.3 (torch 2.8 wheel),
#   transformers 5.19.0 (+ the tokenizers / huggingface_hub / safetensors it needs),
#   flash-linear-attention 0.5.2 (fast kernels for Qwen3.5's linear-attention layers).
# xformers 0.0.32 accepts flash-attn <= 2.8.2 but there is no 2.8.2 wheel for torch 2.8;
# lingua only imports xformers for op lists / the profiler, so the venv's
# sitecustomize.py sets XFORMERS_IGNORE_FLASH_VERSION_CHECK=1.
#
# Usage:  bash bin/install_kd_requirements.sh        # then launch with LINGUA_VENV=$PWD/.venv-kd

set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${KD_VENV:-${ROOT_DIR}/.venv-kd}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

REQ="$(mktemp)"
trap 'rm -f "$REQ"' EXIT
python3 - "$ROOT_DIR/pyproject.toml" "$REQ" <<'PY'
import re, sys
try:
    import tomllib
except ImportError:  # python < 3.11
    import tomli as tomllib
deps = tomllib.load(open(sys.argv[1], "rb"))["project"]["dependencies"]
skip = {"torch", "xformers", "flash-attn", "transformers", "tokenizers", "huggingface-hub", "safetensors"}
name = lambda x: re.split(r"[=<>!~\[; ]", x, maxsplit=1)[0].strip().lower().replace("_", "-")
open(sys.argv[2], "w").write("\n".join(d for d in deps if name(d) not in skip) + "\n")
PY

uv venv --python 3.11 "$VENV"
uv pip install --python "$VENV/bin/python" \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r "$REQ" \
    "torch==2.8.0" "xformers==0.0.32.post2" \
    "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl" \
    "transformers==5.19.0" "flash-linear-attention==0.5.2" accelerate

SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cat > "$SITE/sitecustomize.py" <<'PY'
# Written by bin/install_kd_requirements.sh: xformers 0.0.32 pins flash-attn <= 2.8.2,
# but the torch 2.8 wheel is 2.8.3 (lingua uses flash-attn directly, xformers only for op lists).
import os
os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")
PY

"$VENV/bin/python" - <<'PY'
import torch, transformers, flash_attn, xformers.ops, fla
print(f"torch {torch.__version__} | transformers {transformers.__version__} | "
      f"flash_attn {flash_attn.__version__} | xformers {xformers.__version__} | fla {fla.__version__}")
PY
echo "[install] Done: ${VENV}. Launch with LINGUA_VENV=${VENV}"
