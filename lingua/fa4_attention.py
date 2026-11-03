"""Causal attention over packed documents with FlashAttention 4 (CuTe DSL kernels for Blackwell).

Same semantics as
    flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True)
On GB300 (head_dim 128, real document lengths) FA4 runs the forward at ~930 TFLOP/s and the backward at ~540 TFLOP/s,
vs ~330 / ~260 for flash_attn 2.

FA4 is not part of the venv: it is imported as package `fa4`, a copy of flash-attention's flash_attn/cute (upstream
commit 4616f51), installed with its dependencies (nvidia-cutlass-dsl, quack-kernels, apache-tvm-ffi, cuda-bindings)
into the directory $LINGUA_FA4_SITE (set in scripts/env.sh). fa4_available() appends that directory to sys.path of
this process only (other programs, e.g. the OLMES env, never see it) and imports FA4 on first call.
(Not `flash_attn.cute`: the installed flash_attn 2.8.3 wheel ships an old, incompatible snapshot under that name.)
Kernels are JIT-compiled on first use (~7 s for forward + backward) and cached on disk (node-local
/tmp/$USER/flash_attention_cute_dsl_cache), so later runs on the same node load them.
"""

import logging
import os
import sys
from typing import Tuple

import torch

logger = logging.getLogger()

_fa4 = None  # fa4.interface once imported, False if it does not import


def fa4_available() -> bool:
    global _fa4
    if _fa4 is None:
        site = os.environ.get("LINGUA_FA4_SITE")
        if site:
            for path in (site, os.path.join(site, "nvidia_cutlass_dsl", "dsl_packages")):
                if os.path.isdir(path) and path not in sys.path:
                    sys.path.append(path)
        os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
        try:
            import fa4.interface
            _fa4 = fa4.interface
        except Exception as e:
            logger.warning(f"FlashAttention 4 does not import (LINGUA_FA4_SITE={site}): {type(e).__name__}: {e}")
            _fa4 = False
    return _fa4 is not False


@torch.library.custom_op("lingua::fa4_varlen_causal_attn", mutates_args=())
def _attn_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor,
             deterministic: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    out, lse, *_ = _fa4._flash_attn_fwd(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, causal=True,
                                        return_lse=True)
    return out, lse


@_attn_op.register_fake
def _(q, k, v, cu_seqlens, deterministic):
    return torch.empty_like(q), q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)


@torch.library.custom_op("lingua::fa4_varlen_causal_attn_bwd", mutates_args=())
def _bwd_op(do: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor,
            cu_seqlens: torch.Tensor, deterministic: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = _fa4._flash_attn_bwd(q, k, v, o, do, lse, causal=True, cu_seqlens_q=cu_seqlens,
                                      cu_seqlens_k=cu_seqlens, deterministic=deterministic)
    return dq, dk, dv


@_bwd_op.register_fake
def _(do, q, k, v, o, lse, cu_seqlens, deterministic):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_context(ctx, inputs, output):
    q, k, v, cu_seqlens, deterministic = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse, cu_seqlens)
    ctx.deterministic = deterministic


def _backward(ctx, do, dlse):
    q, k, v, o, lse, cu_seqlens = ctx.saved_tensors
    dq, dk, dv = _bwd_op(do, q, k, v, o, lse, cu_seqlens, ctx.deterministic)
    return dq, dk, dv, None, None


_attn_op.register_autograd(_backward, setup_context=_setup_context)


def fa4_varlen_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor,
                                deterministic: bool = True) -> torch.Tensor:
    """softmax(q k^T / sqrt(d)) v within each document, causal (requires fa4_available()). q, k, v: (total_tokens,
    heads, head_dim); cu_seqlens: int32 document offsets [0, ..., total_tokens]. No max_seqlen: on Blackwell, FA4's varlen kernels schedule work from
    cu_seqlens on the device (same speed without the hint), and leaving it out keeps a per-batch Python int out of
    torch.compile graphs. deterministic: backward without atomics in the dQ accumulation (~6% slower backward)."""
    return _attn_op(q.contiguous(), k.contiguous(), v.contiguous(), cu_seqlens, deterministic)[0]
