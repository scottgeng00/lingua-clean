"""Linear (LM head) + cross-entropy with the softmax gradient computed in the forward pass.

    loss = mean_{i: target_i != ignore_index} [ logsumexp(x_i W^T) - (x_i W^T)[target_i] ]

The bf16 logits x W^T are materialized as usual, then one Triton kernel per row computes the loss (fp32 math) and
overwrites the logits with d loss / d logits = (softmax - onehot) / n_valid in bf16, so the backward is just the two
GEMMs. Same precision as `cross_entropy(F.linear(x, W), target)` (bf16 logits, fp32 log-softmax, bf16 logit
gradients) but one read + one write of the logits instead of the several passes the unfused version makes.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ce_fwd_bwd_kernel(logits_ptr, stride_row, target_ptr, loss_ptr, inv_n_ptr, n_cols, ignore_index,
                       BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    ptr = logits_ptr + row * stride_row
    tgt = tl.load(target_ptr + row)
    valid = tgt != ignore_index
    # pass 1: online max / sum of exponentials (one exp per element)
    m = float("-inf")
    s = 0.0
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        z = tl.load(ptr + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(z, 0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(z - m_new), 0)
        m = m_new
    lse = m + tl.log(s)
    z_t = tl.load(ptr + tl.where(valid, tgt, 0)).to(tl.float32)
    tl.store(loss_ptr + row, tl.where(valid, lse - z_t, 0.0))
    # pass 2: gradient, in place
    scale = tl.where(valid, tl.load(inv_n_ptr), 0.0)
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        z = tl.load(ptr + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        p = tl.exp(z - lse)
        g = tl.where(cols == tgt, p - 1.0, p) * scale
        tl.store(ptr + cols, g.to(logits_ptr.dtype.element_ty), mask=cols < n_cols)


class _LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, target, ignore_index):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        t = target.reshape(-1)
        logits = torch.mm(x2, weight.t())
        n_valid = (t != ignore_index).sum().float()
        inv_n = 1.0 / n_valid
        losses = torch.empty(t.numel(), device=x.device, dtype=torch.float32)
        _ce_fwd_bwd_kernel[(t.numel(),)](logits, logits.stride(0), t, losses, inv_n, logits.shape[1], ignore_index,
                                         BLOCK=4096, num_warps=4)
        ctx.save_for_backward(x2, weight, logits)  # logits now hold d loss / d logits
        ctx.x_shape = shape
        return losses.sum() / n_valid

    @staticmethod
    def backward(ctx, grad_out):
        x2, weight, grad_logits = ctx.saved_tensors
        grad_x = grad_w = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.mm(grad_logits, weight)
            grad_x = (grad_x * grad_out.to(grad_x.dtype)).reshape(ctx.x_shape)
        if ctx.needs_input_grad[1]:
            grad_w = torch.mm(grad_logits.t(), x2)
            grad_w = grad_w * grad_out.to(grad_w.dtype)
        return grad_x, grad_w, None, None


@torch._dynamo.disable
def linear_cross_entropy(x: torch.Tensor, weight: torch.Tensor, target: torch.Tensor,
                         ignore_index: int = -100) -> torch.Tensor:
    """Mean cross-entropy of the logits x @ weight.T against target (tokens equal to ignore_index excluded)."""
    return _LinearCrossEntropy.apply(x, weight, target, ignore_index)
