"""AdamW whose update is a single Triton launch over all tensors, with fp32 per-element arithmetic.

torch.optim.AdamW(fused=True) passes the hyperparameters to its kernel as doubles, so every element is updated in fp64.
On GB300 (~1 TFLOP/s fp64) that makes the step compute-bound: 3.4 ms for 77M parameters, ~10x the time to stream
them. This class keeps AdamW's semantics, state layout (checkpoints load either way) and fused-only features
(found_inf skips the update and leaves the step count unchanged), and runs the same update in fp32:

    p -= lr * wd * p
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g^2
    p -= lr / (1 - b1^t) * m / (sqrt(v) / sqrt(1 - b2^t) + eps)

with the bias corrections computed in fp64 from the on-device step, as torch does. Results differ from torch's
kernel only by fp32 rounding (~1 ulp).
"""

import math

import torch
import triton
import triton.language as tl
from torch.distributed.tensor import DTensor

BLOCK = 1024
NUM_WARPS = 8


@triton.jit
def _adamw_coef_kernel(step_ptrs, coefs, log2_betas, lr, n_tensors, BLOCK_T: tl.constexpr):
    """Per tensor, from its (already incremented) step: lr / (1 - b1^t) and sqrt(1 - b2^t), computed in fp64 like
    torch's fused kernel, but once per tensor instead of in every thread."""
    i = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = i < n_tensors
    step_ptr = tl.load(step_ptrs + i, mask=mask, other=0).to(tl.pointer_type(tl.float32))
    step = tl.load(step_ptr, mask=mask, other=1.0).to(tl.float64)
    bc1 = 1.0 - tl.exp2(step * tl.load(log2_betas))
    bc2 = 1.0 - tl.exp2(step * tl.load(log2_betas + 1))
    tl.store(coefs + 2 * i, (lr.to(tl.float64) / bc1).to(tl.float32), mask=mask)
    tl.store(coefs + 2 * i + 1, tl.sqrt(bc2).to(tl.float32), mask=mask)


@triton.jit
def _adamw_kernel(param_ptrs, grad_ptrs, exp_avg_ptrs, exp_avg_sq_ptrs, numels, wds, coefs,
                  chunk_tensor, chunk_start, found_inf_ptr, lr, beta1, beta2, eps,
                  HAS_FOUND_INF: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    skip = False
    if HAS_FOUND_INF:
        skip = tl.load(found_inf_ptr) == 1.0
    if not skip:
        t = tl.load(chunk_tensor + pid)
        offs = tl.load(chunk_start + pid) + tl.arange(0, BLOCK)
        mask = offs < tl.load(numels + t)
        p_ptr = tl.load(param_ptrs + t).to(tl.pointer_type(tl.float32))
        g_ptr = tl.load(grad_ptrs + t).to(tl.pointer_type(tl.float32))
        m_ptr = tl.load(exp_avg_ptrs + t).to(tl.pointer_type(tl.float32))
        v_ptr = tl.load(exp_avg_sq_ptrs + t).to(tl.pointer_type(tl.float32))
        step_size = tl.load(coefs + 2 * t)
        bc2_sqrt = tl.load(coefs + 2 * t + 1)
        lr_wd = lr * tl.load(wds + t)

        p = tl.load(p_ptr + offs, mask=mask)
        g = tl.load(g_ptr + offs, mask=mask)
        m = tl.load(m_ptr + offs, mask=mask)
        v = tl.load(v_ptr + offs, mask=mask)
        p = p - lr_wd * p
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        denom = tl.sqrt(v) / bc2_sqrt + eps
        p = p - step_size * m / denom
        tl.store(p_ptr + offs, p, mask=mask)
        tl.store(m_ptr + offs, m, mask=mask)
        tl.store(v_ptr + offs, v, mask=mask)


def _local(t):
    return t._local_tensor if isinstance(t, DTensor) else t


class FastAdamW(torch.optim.AdamW):
    """Drop-in for torch.optim.AdamW(fused=True) on fp32 CUDA parameters (see module docstring)."""

    def __init__(self, params, **kwargs):
        kwargs["fused"] = True  # state layout (on-device fp32 step) of the fused implementation
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            assert not group["amsgrad"] and not group["maximize"], "FastAdamW: amsgrad / maximize not supported"
        self._layout = None

    @staticmethod
    def _key(entries):
        return [(p.data_ptr(), m.data_ptr(), v.data_ptr(), st.data_ptr(), p.numel(), wd)
                for p, _, m, v, st, wd in entries]

    def _build_layout(self, entries, betas):
        """Per-tensor pointer tables and the chunk -> (tensor, start) map, for tensors whose storage never moves
        (parameters, moments, steps); gradient pointers are refreshed every step."""
        dev = entries[0][0].device
        chunk_tensor, chunk_start = [], []
        for i, (p, *_rest) in enumerate(entries):
            starts = range(0, p.numel(), BLOCK)
            chunk_tensor += [i] * len(starts)
            chunk_start += list(starts)

        def table(vals, dtype):
            return torch.tensor(vals, dtype=dtype, device=dev)

        self._layout = dict(
            key=self._key(entries),
            params=table([e[0].data_ptr() for e in entries], torch.int64),
            exp_avgs=table([e[2].data_ptr() for e in entries], torch.int64),
            exp_avg_sqs=table([e[3].data_ptr() for e in entries], torch.int64),
            steps=table([e[4].data_ptr() for e in entries], torch.int64),
            numels=table([e[0].numel() for e in entries], torch.int64),
            wds=table([e[5] for e in entries], torch.float32),
            log2_betas=table([math.log2(b) for b in betas], torch.float64),
            chunk_tensor=table(chunk_tensor, torch.int32),
            chunk_start=table(chunk_start, torch.int64),
            coefs=torch.empty(2 * len(entries), dtype=torch.float32, device=dev),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        entries, steps, hyper = [], [], None
        for group in self.param_groups:
            params, grads, exp_avgs, exp_avg_sqs, state_steps = [], [], [], [], []
            self._init_group(group, params, grads, exp_avgs, exp_avg_sqs, [], state_steps)
            h = (float(group["lr"]), *map(float, group["betas"]), float(group["eps"]))
            assert hyper is None or hyper == h, "FastAdamW: lr / betas / eps must be the same in all groups"
            hyper = h
            for p, g, m, v, st in zip(params, grads, exp_avgs, exp_avg_sqs, state_steps):
                p, g, m, v = _local(p), _local(g), _local(m), _local(v)
                if p.numel() == 0:
                    continue
                assert p.dtype == g.dtype == m.dtype == v.dtype == torch.float32 and st.dtype == torch.float32
                assert p.is_contiguous() and g.is_contiguous() and m.is_contiguous() and v.is_contiguous()
                entries.append((p, g, m, v, st, float(group["weight_decay"])))
            steps += state_steps
        if not entries:
            return loss
        lr, beta1, beta2, eps = hyper
        if self._layout is None or self._layout["key"] != self._key(entries):
            self._build_layout(entries, (beta1, beta2))
        lay = self._layout
        grad_ptrs = torch.tensor([e[1].data_ptr() for e in entries], dtype=torch.int64).pin_memory()
        grad_ptrs = grad_ptrs.to(entries[0][0].device, non_blocking=True)
        found_inf = getattr(self, "found_inf", None)
        # Like torch's fused AdamW: increment the steps first and undo it when the update is skipped.
        torch._foreach_add_(steps, 1)
        n = len(entries)
        _adamw_coef_kernel[(triton.cdiv(n, 128),)](lay["steps"], lay["coefs"], lay["log2_betas"], lr, n, BLOCK_T=128)
        _adamw_kernel[(lay["chunk_tensor"].numel(),)](
            lay["params"], grad_ptrs, lay["exp_avgs"], lay["exp_avg_sqs"], lay["numels"], lay["wds"], lay["coefs"],
            lay["chunk_tensor"], lay["chunk_start"], found_inf if found_inf is not None else lay["wds"],
            lr, beta1, beta2, eps, HAS_FOUND_INF=found_inf is not None, BLOCK=BLOCK, num_warps=NUM_WARPS)
        if found_inf is not None:
            torch._foreach_sub_(steps, [found_inf] * len(steps))
        return loss
