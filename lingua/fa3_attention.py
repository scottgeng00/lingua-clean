"""Causal attention over packed documents with FlashAttention 3 (Hopper, sm_90).

Same semantics as
    flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True)
On H100 (head_dim 128, 9 heads, packed 4096-token sequences) FA3 runs the forward at ~450 TFLOP/s and the backward
at ~230 TFLOP/s, vs ~215 / ~150 for flash_attn 2: +6.5% tokens/s for w1152 at 1 and 4 nodes, same loss.

FA3 is not part of the venv: bin/build_fa3_h100.sh compiles flash-attention v2.8.3's hopper/ (bf16, head_dim 128,
sm_90 kernels only) against torch 2.8 and installs `flash_attn_3._C` + `flash_attn_interface` into the directory
$LINGUA_FA3_SITE (set by the cluster profile). fa3_available() appends that directory to sys.path of this process
only (other programs, e.g. the OLMES env, never see it) and imports FA3 on first call. The build registers its ops
as torch.ops.lingua_fa3, since xformers' own bundled FA3 already holds the flash_attn_3 namespace.
The kernels are wrapped in torch custom ops with fake implementations, so torch.compile traces through them
without graph breaks.
"""

import logging
import os
import sys
from typing import Tuple

import torch

logger = logging.getLogger()

_fa3 = None  # flash_attn_interface once imported, False if it does not import


def fa3_available() -> bool:
    global _fa3
    if _fa3 is None and torch.cuda.is_available() and torch.cuda.get_device_capability() != (9, 0):
        # The build only has sm_90a kernels (H100 / H200).
        logger.info(f"FlashAttention 3 skipped on {torch.cuda.get_device_name()} (needs sm_90)")
        _fa3 = False
    if _fa3 is None:
        site = os.environ.get("LINGUA_FA3_SITE")
        if not site:
            _fa3 = False
            return False
        if os.path.isdir(site) and site not in sys.path:
            sys.path.append(site)
        try:
            import flash_attn_interface
            _fa3 = flash_attn_interface
        except Exception as e:
            logger.warning(f"FlashAttention 3 does not import (LINGUA_FA3_SITE={site}): {type(e).__name__}: {e}")
            _fa3 = False
    return _fa3 is not False


@torch.library.custom_op("lingua::fa3_varlen_causal_attn", mutates_args=())
def _attn_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor,
             max_seqlen: int) -> Tuple[torch.Tensor, torch.Tensor]:
    out, lse, *_ = _fa3._flash_attn_forward(
        q, k, v, None, None, None, None, cu_seqlens, cu_seqlens, None, None, None, max_seqlen, max_seqlen,
        None, None, None, None, None, None, None, None, None, q.shape[-1] ** -0.5, causal=True)
    return out, lse


@_attn_op.register_fake
def _(q, k, v, cu_seqlens, max_seqlen):
    return torch.empty_like(q), q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)


@torch.library.custom_op("lingua::fa3_varlen_causal_attn_bwd", mutates_args=())
def _bwd_op(do: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor,
            cu_seqlens: torch.Tensor, max_seqlen: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _fa3._flash_attn_backward(do, q, k, v, o, lse, cu_seqlens, cu_seqlens, None, None, max_seqlen, max_seqlen,
                              dq, dk, dv, q.shape[-1] ** -0.5, True)
    return dq, dk, dv


@_bwd_op.register_fake
def _(do, q, k, v, o, lse, cu_seqlens, max_seqlen):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_context(ctx, inputs, output):
    q, k, v, cu_seqlens, max_seqlen = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse, cu_seqlens)
    ctx.max_seqlen = max_seqlen


def _backward(ctx, do, dlse):
    q, k, v, o, lse, cu_seqlens = ctx.saved_tensors
    dq, dk, dv = _bwd_op(do.contiguous(), q, k, v, o, lse, cu_seqlens, ctx.max_seqlen)
    return dq, dk, dv, None, None


_attn_op.register_autograd(_backward, setup_context=_setup_context)


def fa3_varlen_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor,
                                max_seqlen: int) -> torch.Tensor:
    """softmax(q k^T / sqrt(d)) v within each document, causal (requires fa3_available()). q, k, v: (total_tokens,
    heads, head_dim) bf16; cu_seqlens: int32 document offsets [0, ..., total_tokens]; max_seqlen: an upper bound on
    the document length (sizes the launch grid; the kernels read the real lengths from cu_seqlens)."""
    return _attn_op(q.contiguous(), k.contiguous(), v.contiguous(), cu_seqlens, max_seqlen)[0]
