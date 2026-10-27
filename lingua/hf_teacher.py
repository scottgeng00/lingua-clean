"""Document-masked forward passes through HF teacher models on packed batches.

The student attends within documents only (data.disable_cross_doc_attn), via flash-attn
varlen over the flattened batch. `teacher_logits_packed` gives an HF teacher the same view:
the (B, S) batch is flattened to one packed row with per-document position ids and
`cu_seq_lens_*` kwargs, which transformers' flash-attention path (and, for Qwen3.5, the
gated-delta-rule linear attention) honours.

Qwen3.5's short causal conv in front of its linear attention has a pure-PyTorch fallback
(used without the `causal_conv1d` package) that ignores document boundaries; the leak at
each document start then propagates through the recurrent state and later layers.
`patch_qwen3_5_packed_conv` swaps in a boundary-aware version when `cu_seq_lens_q` is given:
flash-linear-attention's Triton varlen conv when available (3x faster than the PyTorch version,
~50 ms per 16k-token Qwen3.5-9B forward on GB300), else `_segmented_causal_conv1d`.
"""

import torch
import torch.nn.functional as F


def packed_position_ids(cu_seqlens: torch.Tensor, total: int) -> torch.Tensor:
    """Position within its document for each of the `total` packed tokens, shape (1, total)."""
    pos = torch.arange(total, device=cu_seqlens.device)
    starts = cu_seqlens[:-1].long()
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    doc_start = torch.repeat_interleave(starts, lengths, output_size=total)
    return (pos - doc_start).unsqueeze(0)


def teacher_logits_packed(model, input_ids: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int):
    """Teacher logits (B, S, V) for input_ids (B, S) with attention restricted to documents.

    cu_seqlens: int32 document boundaries over the flattened B*S tokens (as built for the
    student's flash-attn varlen call); max_seqlen: longest document.
    """
    bsz, seq_len = input_ids.shape
    flat = input_ids.reshape(1, bsz * seq_len)
    cu = cu_seqlens.to(device=input_ids.device, dtype=torch.int32)
    out = model(
        flat,
        position_ids=packed_position_ids(cu, bsz * seq_len),
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=int(max_seqlen),
        max_length_k=int(max_seqlen),
        use_cache=False,
    )
    return out.logits.view(bsz, seq_len, -1)


def _segmented_causal_conv1d(hidden_states, weight, bias=None, activation=None, cu_seqlens=None):
    """Depthwise causal conv (x: (1, C, T), weight: (C, K)) that does not cross documents."""
    from transformers.activations import ACT2FN

    _, _, total = hidden_states.shape
    k = weight.shape[-1]
    x = hidden_states.to(weight.dtype)
    doc = torch.repeat_interleave(
        torch.arange(len(cu_seqlens) - 1, device=x.device),
        (cu_seqlens[1:] - cu_seqlens[:-1]).long(),
        output_size=total,
    )
    out = torch.zeros_like(x)
    for shift in range(k):  # out[t] += w[:, k-1-shift] * x[t-shift] if t-shift is in t's document
        if shift == 0:
            out += weight[:, k - 1, None] * x
            continue
        same = (doc[shift:] == doc[:-shift]).to(x.dtype)
        out[:, :, shift:] += weight[:, k - 1 - shift, None] * x[:, :, :-shift] * same
    if bias is not None:
        out += bias[:, None]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def _fla_varlen_causal_conv1d(hidden_states, weight, bias=None, activation=None, cu_seqlens=None):
    """Same contract as `_segmented_causal_conv1d` (x: (1, C, T)), via fla's Triton kernel."""
    from fla.modules.convolution import causal_conv1d

    out, _ = causal_conv1d(hidden_states.transpose(1, 2).contiguous(), weight, bias,
                           activation=activation, cu_seqlens=cu_seqlens.long())
    return out.transpose(1, 2)


def patch_qwen3_5_packed_conv():
    """Make Qwen3.5's causal conv respect document boundaries when cu_seq_lens_q is passed.

    Only the PyTorch fallback is replaced; if the `causal_conv1d` package provides the kernel
    it is left alone (it is not used in our environment). Returns True if patched.
    """
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    except ImportError:
        return False
    if getattr(mq, "_lingua_packed_conv", False):
        return True
    original = mq.causal_conv1d_fn
    try:
        import fla.modules.convolution  # noqa: F401
        packed_conv = _fla_varlen_causal_conv1d
    except ImportError:
        packed_conv = _segmented_causal_conv1d

    def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        cu = kwargs.get("cu_seq_lens_q")
        if cu is None or hidden_states.shape[0] != 1:
            return original(hidden_states, weight, bias, activation=activation, **kwargs)
        return packed_conv(hidden_states, weight, bias, activation, cu)

    mq.causal_conv1d_fn = causal_conv1d_fn
    mq._lingua_packed_conv = True
    return True
