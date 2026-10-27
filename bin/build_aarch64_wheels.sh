#!/usr/bin/env bash
# bin/build_aarch64_wheels.sh — build the CUDA wheels that have no aarch64 release
# (flash-attn 2.8.3, xformers 0.0.32.post2) against .venv-kd's torch 2.8.0+cu129,
# for Blackwell (GB200/GB300, sm_100; sm_100 cubins run on sm_103).
# Needs a CUDA 12.9 toolkit at $CUDA_HOME (conda-forge cuda-nvcc / cuda-libraries-dev).
# Run on a compute node (lots of CPUs, ~20 min):
#   sbatch -A mnm -q g3_comm_shared --gres=gpu:1 -c 64 --mem=400G -t 2:00:00 bin/build_aarch64_wheels.sh
set -euo pipefail
ROOT_DIR="${ROOT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${KD_VENV:-${ROOT_DIR}/.venv-kd}"
WHEELHOUSE="${WHEELHOUSE:-${ROOT_DIR}/../wheelhouse}"
export CUDA_HOME="${CUDA_HOME:-${ROOT_DIR}/../cuda-12.9}"
# venv bin for ninja: without it torch.utils.cpp_extension compiles one file at a time.
export PATH="${CUDA_HOME}/bin:${VENV}/bin:${PATH}"
export UV_LINK_MODE=copy
mkdir -p "${WHEELHOUSE}"
uv pip install --python "$VENV/bin/python" pip ninja packaging psutil wheel setuptools
BUILD="$(mktemp -d /tmp/fa-build.XXXX)"
trap 'rm -rf "$BUILD"' EXIT
cd "$BUILD"

# flash-attn: build for sm_100 only.
FLASH_ATTN_CUDA_ARCHS=100 MAX_JOBS="${MAX_JOBS:-48}" NVCC_THREADS=2 FLASH_ATTENTION_FORCE_BUILD=TRUE \
    "$VENV/bin/python" -m pip wheel --no-build-isolation --no-deps -w "${WHEELHOUSE}" "flash-attn==2.8.3" -v 2>&1 | tail -20

# xformers: lingua only uses its Python API (fmha types, op lists, profiler) and its
# flash-attn dispatch (which uses the flash-attn package above), so skip the CUDA kernels.
FORCE_CUDA=0 XFORMERS_DISABLE_FLASH_ATTN=1 TORCH_CUDA_ARCH_LIST="10.0" MAX_JOBS="${MAX_JOBS:-48}" \
    "$VENV/bin/python" -m pip wheel --no-build-isolation --no-deps -w "${WHEELHOUSE}" "xformers==0.0.32.post2" 2>&1 | tail -5
ls -la "${WHEELHOUSE}"
