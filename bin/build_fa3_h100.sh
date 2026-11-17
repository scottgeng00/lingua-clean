#!/usr/bin/env bash
# bin/build_fa3_h100.sh — build FlashAttention 3 (Hopper, sm_90a) for .venv-kd's torch 2.8.0+cu128 and install it into
# a self-contained site directory that only the training process puts on sys.path (lingua/fa3_attention.py,
# $LINGUA_FA3_SITE). .venv-kd itself is not modified.
#
#   sbatch -A dream -p h100_mdf -q h100_mdf_explore -N1 --gres=gpu:1 -c 32 --mem=128G -t 0:30:00 \
#       -o fa3_build-%j.out bin/build_fa3_h100.sh [PREFIX]
#
# PREFIX defaults to $PRETRAIN_ROOT/speed-extras-h100; the result is PREFIX/fa3_site (flash_attn_3/_C.abi3.so +
# flash_attn_interface.py), to be set as LINGUA_FA3_SITE. Scratch (sources, build venv, CUDA toolchain) goes to
# PREFIX/build and can be deleted. The build needs internet (git, pip, NVIDIA's redist server) and CUDA 12.8: the
# cluster modules stop at 12.1, so nvcc comes from NVIDIA's cuda_nvcc redist archive and the headers from the pip
# wheels torch uses. Only bf16, head_dim 128, sm_90 kernels are compiled (no fp16 / fp8, paged KV, split-KV,
# appendKV, local windows, softcap, PackGQA or clusters): 4 translation units, ~4 min, ~7 GB RAM.
# A GPU is not needed to compile; the job uses one to smoke-test the result.
set -euo pipefail

FA_TAG=v2.8.3  # the flash-attn release installed in .venv-kd; its hopper/ builds against torch 2.8
PREFIX="${1:-${PRETRAIN_ROOT:-/home/sgeng/scottg/pretrain}/speed-extras-h100}"
BUILD="${PREFIX}/build"
SITE="${PREFIX}/fa3_site"
mkdir -p "${BUILD}"
cd "${BUILD}"

# Build venv: same python / torch as .venv-kd, plus CCCL headers and nvcc 12.8.
if [ ! -x venv/bin/python ]; then
    uv venv -q --python 3.11 venv
fi
uv pip install -q --python venv/bin/python --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple torch==2.8.0 setuptools wheel packaging ninja \
    "nvidia-cuda-cccl-cu12==12.8.90"
# (the nvidia-cuda-nvcc-cu12 wheel has ptxas but no nvcc front end)
NVCC_ARCHIVE=cuda_nvcc-linux-x86_64-12.8.93-archive
if [ ! -x "${NVCC_ARCHIVE}/bin/nvcc" ]; then
    curl -sSfL "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/${NVCC_ARCHIVE}.tar.xz" \
        | tar -xJ
fi

# CUDA_HOME for torch.utils.cpp_extension: nvcc's bin/ and nvvm/, its headers plus every nvidia wheel's, libcudart.so.
NV="$(venv/bin/python -c 'import nvidia, os; print(os.path.dirname(nvidia.__file__))')"
CUDA_HOME="${BUILD}/cuda-12.8"
rm -rf "${CUDA_HOME}"
mkdir -p "${CUDA_HOME}/include" "${CUDA_HOME}/lib64"
ln -s "${BUILD}/${NVCC_ARCHIVE}/bin" "${CUDA_HOME}/bin"
ln -s "${BUILD}/${NVCC_ARCHIVE}/nvvm" "${CUDA_HOME}/nvvm"
for inc in "${BUILD}/${NVCC_ARCHIVE}/include" "${NV}"/*/include; do
    for f in "${inc}"/*; do
        [ -e "${CUDA_HOME}/include/$(basename "$f")" ] || ln -s "$f" "${CUDA_HOME}/include/"
    done
done
ln -s "${NV}/cuda_runtime/lib/libcudart.so.12" "${CUDA_HOME}/lib64/libcudart.so"
export CUDA_HOME PATH="${CUDA_HOME}/bin:${BUILD}/venv/bin:${PATH}"  # venv/bin: ninja
nvcc --version | tail -2

if [ ! -d flash-attention ]; then
    git clone -q --depth 1 --branch "${FA_TAG}" https://github.com/Dao-AILab/flash-attention.git
    git -C flash-attention submodule update -q --init --depth 1 csrc/cutlass
fi
cd flash-attention/hopper
# xformers (imported by lingua.transformer) bundles its own FA3 build that registers the torch op namespace
# flash_attn_3, and a second TORCH_LIBRARY for the same namespace aborts the process; register ours as lingua_fa3.
git checkout -q flash_api.cpp flash_attn_interface.py
sed -i 's/TORCH_LIBRARY\(_IMPL\)\?(flash_attn_3,/TORCH_LIBRARY\1(lingua_fa3,/' flash_api.cpp
sed -i 's/torch\.ops\.flash_attn_3$/torch.ops.lingua_fa3/' flash_attn_interface.py
grep -q "TORCH_LIBRARY(lingua_fa3," flash_api.cpp && grep -q "torch.ops.lingua_fa3" flash_attn_interface.py

export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE FLASH_ATTENTION_DISABLE_FP16=TRUE \
    FLASH_ATTENTION_DISABLE_FP8=TRUE FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE FLASH_ATTENTION_DISABLE_SPLIT=TRUE \
    FLASH_ATTENTION_DISABLE_APPENDKV=TRUE FLASH_ATTENTION_DISABLE_LOCAL=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
    FLASH_ATTENTION_DISABLE_PACKGQA=TRUE FLASH_ATTENTION_DISABLE_CLUSTER=TRUE FLASH_ATTENTION_DISABLE_HDIM64=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM96=TRUE FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE
# Each nvcc process peaks at a few GB; the bwd unit is the largest.
export MAX_JOBS="${MAX_JOBS:-4}" NVCC_THREADS="${NVCC_THREADS:-4}" TORCH_CUDA_ARCH_LIST=9.0
rm -rf build dist
../../venv/bin/python setup.py -q bdist_wheel

rm -rf "${SITE}"
mkdir -p "${SITE}"
../../venv/bin/python -m zipfile -e dist/flash_attn_3-*.whl "${SITE}"
echo "flash-attention ${FA_TAG} ($(git rev-parse HEAD)) hopper/, bf16 hdim128 sm90 only, ops as torch.ops.lingua_fa3" \
    > "${SITE}/BUILD_INFO"

# Smoke test with the training venv (needs a GPU), next to xformers' FA3.
if nvidia-smi -L >/dev/null 2>&1; then
    LINGUA_VENV="${LINGUA_VENV:-${PRETRAIN_ROOT:-/home/sgeng/scottg/pretrain}/lingua-clean/.venv-kd}"
    PYTHONPATH="${SITE}" "${LINGUA_VENV}/bin/python" - <<'EOF'
import torch, xformers.ops, flash_attn_interface as fa3
q, k, v = (torch.randn(4096, 9, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
cu = torch.tensor([0, 1000, 4096], device="cuda", dtype=torch.int32)
out = fa3.flash_attn_varlen_func(q, k, v, cu, cu, 3096, 3096, causal=True)
out = out[0] if isinstance(out, tuple) else out
out.sum().backward()
print("FA3 smoke test ok:", tuple(out.shape), q.grad.abs().mean().item())
EOF
fi
echo "FA3 site: ${SITE}"
