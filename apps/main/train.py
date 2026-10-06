# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from copy import deepcopy
import gc
import json
import logging
import math
import os
import re
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint
import xformers.profiler
from torch.optim import lr_scheduler
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed._tensor import DTensor

from lingua.args import dataclass_from_dict, dump_config, flatten_dict
from lingua.checkpoint import CheckpointArgs, CheckpointManager, load_from_checkpoint
from lingua.data import (
    DataArgs,
    PackTokensState,
    build_dataloader_from_args,
    init_dataloader_state_from_args,
)
from lingua.online_data import (
    ExcessLossController,
    OnlineReweightingArgs,
    ReweightableDataLoader,
)
from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    init_signal_handler,
    dist_mean_dict,
    get_device_mesh,
    get_global_rank,
    get_is_master,
    get_world_size,
    parallelize_model,
    setup_env,
    setup_torch_distributed,
    clean_env,
    requeue_slurm_job,
    check_model_value_range,
)
from lingua.logger import init_logger
from lingua.metrics import (
    GPUMemoryMonitor,
    LoggingArgs,
    MetricLogger,
    get_num_params,
)
from lingua.optim import OptimArgs, build_optimizer
from lingua.profiling import ProfilerArgs, maybe_run_profiler
from lingua.tokenizer import build_tokenizer
from apps.main.transformer import (
    LMTransformerArgs,
    LMTransformer,
    get_num_flop_per_token,
    build_fsdp_grouping_plan,
    tp_parallelize,
    get_no_recompute_ops,
)
from lingua.probe import AutoProbeD
from lingua.stool import StoolArgs, launch_job

from transformers import AutoModelForCausalLM

import wandb

logger = logging.getLogger()


# ---------------------------------------------------------------------------
# Stubs for loss functions removed during the lingua-clean port.
#
# The dispatch chain inside `train()` still mentions these names inside
# `elif use_<X>_distillation:` branches that are gated by data-flags which
# the cleaned DataArgs no longer exposes (the flags default to False and
# cannot be set from the supported recipe YAMLs). The branches are therefore
# dead code at runtime; we keep the names defined here so the module imports
# cleanly and so any future readers can grep the dispatch chain and see
# what was dropped. Calling any of these will raise NotImplementedError —
# this is intentional: it surfaces immediately if a user tries to reach a
# removed recipe rather than silently doing the wrong thing.
# ---------------------------------------------------------------------------
def _removed_loss(name):
    def _fn(*args, **kwargs):
        raise NotImplementedError(
            f"{name} was removed in lingua-clean. The supported recipes are "
            "listed in README.md / apps/main/configs/recipes/. If you need "
            "this loss, restore it from the upstream lingua/ checkpoint."
        )
    _fn.__name__ = name
    return _fn


compute_projected_kd_loss = _removed_loss("compute_projected_kd_loss")
compute_rkl_with_gated_ce_loss = _removed_loss("compute_rkl_with_gated_ce_loss")
compute_rkl_with_lowent_ce_loss = _removed_loss("compute_rkl_with_lowent_ce_loss")
compute_rkl_with_source_ce_loss = _removed_loss("compute_rkl_with_source_ce_loss")
compute_rkl_with_teacher_disagree_ce_loss = _removed_loss(
    "compute_rkl_with_teacher_disagree_ce_loss"
)
compute_rkl_with_teacher_fail_ce_loss = _removed_loss(
    "compute_rkl_with_teacher_fail_ce_loss"
)
compute_rkl_with_teacher_success_ce_loss = _removed_loss(
    "compute_rkl_with_teacher_success_ce_loss"
)
compute_rkl_with_topk_gap_ce_loss = _removed_loss("compute_rkl_with_topk_gap_ce_loss")
compute_rkl_with_topk_gap_ce_replace_loss = _removed_loss(
    "compute_rkl_with_topk_gap_ce_replace_loss"
)
compute_rkl_with_gradagree_gap_ce_loss = _removed_loss(
    "compute_rkl_with_gradagree_gap_ce_loss"
)
compute_rho1_kd_loss = _removed_loss("compute_rho1_kd_loss")
compute_akl_distillation_loss = _removed_loss("compute_akl_distillation_loss")
compute_projected_two_teacher_kd_loss = _removed_loss(
    "compute_projected_two_teacher_kd_loss"
)
compute_agreement_gated_two_teacher_kd_loss = _removed_loss(
    "compute_agreement_gated_two_teacher_kd_loss"
)
compute_competence_routed_two_teacher_kd_loss = _removed_loss(
    "compute_competence_routed_two_teacher_kd_loss"
)
compute_intersection_projected_kd_loss = _removed_loss(
    "compute_intersection_projected_kd_loss"
)
compute_geometric_kd_loss = _removed_loss("compute_geometric_kd_loss")
compute_two_teacher_geometric_kd_loss = _removed_loss(
    "compute_two_teacher_geometric_kd_loss"
)
compute_selective_kd_loss = _removed_loss("compute_selective_kd_loss")
compute_bucket_aware_kd_loss = _removed_loss("compute_bucket_aware_kd_loss")
compute_hybrid_residual_kd_loss = _removed_loss("compute_hybrid_residual_kd_loss")
compute_entropy_gated_kd_loss = _removed_loss("compute_entropy_gated_kd_loss")
compute_frontier_band_loss = _removed_loss("compute_frontier_band_loss")
compute_frontierv2_loss = _removed_loss("compute_frontierv2_loss")
compute_frontierv3_loss = _removed_loss("compute_frontierv3_loss")
compute_frontierv4_loss = _removed_loss("compute_frontierv4_loss")
compute_ema_frontier_loss = _removed_loss("compute_ema_frontier_loss")
compute_remit_loss = _removed_loss("compute_remit_loss")
compute_teacher_critic_loss = _removed_loss("compute_teacher_critic_loss")
compute_margin_constraint_loss = _removed_loss("compute_margin_constraint_loss")
compute_entropy_aware_margin_loss = _removed_loss("compute_entropy_aware_margin_loss")
compute_siw_loss = _removed_loss("compute_siw_loss")
compute_best_expert_loss = _removed_loss("compute_best_expert_loss")
compute_best_expert_seq_kd_loss = _removed_loss("compute_best_expert_seq_kd_loss")
compute_rho1_loss = _removed_loss("compute_rho1_loss")
compute_rho1_expert_stratified_loss = _removed_loss(
    "compute_rho1_expert_stratified_loss"
)
compute_ema_ref_loss = _removed_loss("compute_ema_ref_loss")
compute_weighted_loss_with_teacher = _removed_loss("compute_weighted_loss_with_teacher")
compute_lwt_loss = _removed_loss("compute_lwt_loss")
compute_mile_loss = _removed_loss("compute_mile_loss")


def swap_ema_ref_checkpoint(*args, **kwargs):
    raise NotImplementedError(
        "swap_ema_ref_checkpoint was removed in lingua-clean (EMA-frontier "
        "recipe not supported)."
    )


def _hybrid_residual_kd_chunk(*args, **kwargs):
    raise NotImplementedError(
        "_hybrid_residual_kd_chunk was removed in lingua-clean (hybrid-residual "
        "KD recipe not supported)."
    )


def _build_token_domain_ids(
    labels: torch.Tensor,
    cu_seqlens: Optional[List[Optional[List[int]]]],
    doc_sources: Optional[List[Optional[List[str]]]],
    domain_labels: Optional[List[Any]] = None,
):
    """Build per-token (B, S) domain_ids + id_to_name map from packed-doc metadata.

    Mirrors the per-token domain attribution at line ~2845 of `compute_rho1_loss`
    but returns the result so the KD paths can share the same convention. When
    cu_seqlens/doc_sources are unavailable, falls back to per-sequence
    `domain_labels` if those exist; otherwise returns (None, {}).
    """
    has_doc_sources = (
        doc_sources is not None
        and cu_seqlens is not None
        and len(doc_sources) == labels.size(0)
    )
    has_domain_labels = domain_labels is not None and len(domain_labels) == labels.size(0)
    if not (has_doc_sources or has_domain_labels):
        return None, {}

    source_to_idx: Dict[str, int] = {}
    domain_ids = torch.zeros(
        labels.size(0), labels.size(1), device=labels.device, dtype=torch.long
    )
    if has_doc_sources:
        for b in range(labels.size(0)):
            cu = cu_seqlens[b] if cu_seqlens[b] is not None else [0, labels.size(1)]
            dsrcs = doc_sources[b] if doc_sources[b] is not None else []
            for d in range(len(cu) - 1):
                src_key = dsrcs[d] if d < len(dsrcs) else (
                    str(domain_labels[b]) if has_domain_labels else "unknown"
                )
                src_key = str(src_key)
                if src_key not in source_to_idx:
                    source_to_idx[src_key] = len(source_to_idx)
                domain_ids[b, cu[d]:cu[d + 1]] = source_to_idx[src_key]
    else:
        for i, src in enumerate(domain_labels):
            src_key = str(src)
            if src_key not in source_to_idx:
                source_to_idx[src_key] = len(source_to_idx)
            domain_ids[i, :] = source_to_idx[src_key]

    id_to_name = {v: k for k, v in source_to_idx.items()}
    return domain_ids, id_to_name


def _concentration_stats(prefix: str, vals: torch.Tensor) -> Dict[str, float]:
    """Return concentration metrics (top-k% mass share, gini, median, p99, mean) for
    a 1-D non-negative `vals` tensor. Designed for token-level signal mass diagnostics
    (per-token KL, JS, H_s, etc.). Fully no-op outside of `torch.no_grad()`.
    """
    out: Dict[str, float] = {}
    n = int(vals.numel())
    if n == 0:
        return out
    v = vals.float().abs()
    sorted_desc = torch.sort(v, descending=True).values
    total = sorted_desc.sum().clamp(min=1e-12)
    k10 = max(1, int(n * 0.10))
    k20 = max(1, int(n * 0.20))
    out[f"{prefix}_top10pct_share"] = (sorted_desc[:k10].sum() / total).item()
    out[f"{prefix}_top20pct_share"] = (sorted_desc[:k20].sum() / total).item()

    asc = sorted_desc.flip(0)
    idx = torch.arange(1, n + 1, device=asc.device, dtype=asc.dtype)
    out[f"{prefix}_gini"] = (
        (2.0 * (idx * asc).sum() - (n + 1) * asc.sum())
        / (n * asc.sum().clamp(min=1e-12))
    ).item()

    out[f"{prefix}_mean"] = v.mean().item()
    out[f"{prefix}_median"] = v.median().item()
    out[f"{prefix}_p99"] = (
        torch.quantile(v, 0.99).item() if n > 1 else v.max().item()
    )
    out[f"{prefix}_n_tokens"] = float(n)
    return out


def _compute_kd_concentration_diagnostics(
    student_logits: torch.Tensor,    # (B, S, V) - WILL be .detach()ed internally
    teacher_logits: torch.Tensor,    # (B, S, V) - the 1B-anchor teacher; WILL be .detach()ed
    mask: torch.Tensor,              # (B, S) float, 1 = valid
    temperature: float = 1.0,
    domain_ids: Optional[torch.Tensor] = None,
    id_to_name: Optional[Dict[int, str]] = None,
    chunk_size: int = 128,
    max_domains_to_log: int = 16,
) -> Dict[str, float]:
    """Concentration of per-token KD signal — read-only diagnostic, fully no_grad.

    For every valid token computes:
      * KL_t  = KL(p_teacher || p_student)   (the kd-rl objective term)
      * JS_t  = JS(p_teacher,   p_student)   (symmetric disagreement)
      * Hs_t  = H(p_student)                 (student-only uncertainty signal)
    where p_* = softmax(logits / temperature).

    Returns a flat dict of scalar wandb stats, prefixed `selective/`:
      * global: {kl,js,Hs}_{top10pct_share, top20pct_share, gini, mean, median, p99}
      * per-domain: {kl,js,Hs}_by_domain/<name>_{median,mean,frac_of_tokens}
                   (limited to the largest `max_domains_to_log` domains by token count)
      * a 'selective/_temperature' tag for the temperature used

    NUMERICAL GUARANTEE: this function never mutates inputs and runs entirely under
    `torch.no_grad()`. Returned scalar stats are detached items; no autograd
    interaction with the loss path. Caller is responsible for invoking only when
    diagnostics are enabled (gating flag lives at the call site).
    """
    with torch.no_grad():
        s_logits = student_logits.detach()
        t_logits = teacher_logits.detach()
        valid = mask.detach().bool()
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            return {}

        B, S, V = s_logits.shape
        kl_buf = torch.empty(B, S, device=s_logits.device, dtype=torch.float32)
        js_buf = torch.empty(B, S, device=s_logits.device, dtype=torch.float32)
        hs_buf = torch.empty(B, S, device=s_logits.device, dtype=torch.float32)

        for cs in range(0, S, chunk_size):
            ce = min(cs + chunk_size, S)
            s_chunk = s_logits[:, cs:ce, :]
            t_chunk = t_logits[:, cs:ce, :]

            log_s = F.log_softmax(s_chunk / temperature, dim=-1)
            log_t = F.log_softmax(t_chunk / temperature, dim=-1)
            p_s = log_s.exp()
            p_t = log_t.exp()

            kl_chunk = (p_t * (log_t - log_s)).sum(dim=-1)
            m = 0.5 * (p_s + p_t)
            log_m = m.clamp_min(1e-12).log()
            js_chunk = 0.5 * (
                (p_s * (log_s - log_m)).sum(dim=-1)
                + (p_t * (log_t - log_m)).sum(dim=-1)
            )
            hs_chunk = -(p_s * log_s).sum(dim=-1)

            kl_buf[:, cs:ce] = kl_chunk.float()
            js_buf[:, cs:ce] = js_chunk.float()
            hs_buf[:, cs:ce] = hs_chunk.float()

            del log_s, log_t, p_s, p_t, m, log_m, kl_chunk, js_chunk, hs_chunk

        kl_valid = kl_buf[valid]
        js_valid = js_buf[valid]
        hs_valid = hs_buf[valid]

        stats: Dict[str, float] = {"selective/_temperature": float(temperature)}
        stats.update(_concentration_stats("selective/kl_1b_student", kl_valid))
        stats.update(_concentration_stats("selective/js_1b_student", js_valid))
        stats.update(_concentration_stats("selective/Hs_student",    hs_valid))

        if domain_ids is not None and id_to_name:
            flat_dom = domain_ids.detach()[valid]
            unique_ids, counts = torch.unique(flat_dom, return_counts=True)
            order = torch.argsort(counts, descending=True)
            unique_ids = unique_ids[order][:max_domains_to_log].tolist()
            for did in unique_ids:
                name = id_to_name.get(did, f"d{did}")
                dmask = flat_dom == did
                n_dom = int(dmask.sum().item())
                if n_dom == 0:
                    continue
                stats[f"selective/kl_by_domain/{name}_median"] = kl_valid[dmask].median().item()
                stats[f"selective/kl_by_domain/{name}_mean"]   = kl_valid[dmask].mean().item()
                stats[f"selective/js_by_domain/{name}_median"] = js_valid[dmask].median().item()
                stats[f"selective/Hs_by_domain/{name}_median"] = hs_valid[dmask].median().item()
                stats[f"selective/by_domain/{name}_frac"]      = n_dom / n_valid
        return stats


@torch.compile(dynamic=False)
def compute_kl_distillation_loss(
    student_logits: torch.Tensor,   # (batch_size, seq_len, vocab_size)
    teacher_logits: torch.Tensor,   # (batch_size, seq_len, vocab_size)
    labels: torch.Tensor,           # (batch_size, seq_len)
    temperature: float = 1.0,
    alpha: float = 0.5,
    chunk_size: int = 128,  # Process this many tokens at a time to save memory
) -> tuple:
    """
    Memory-efficient Classic Knowledge Distillation: minimize KL divergence between student and teacher.

    Loss = α * KL(teacher || student) * τ² + (1 - α) * CE(student, labels)

    This implementation processes tokens in chunks along the sequence dimension
    to avoid materializing full (B, S, V) softmax tensors simultaneously.

    Args:
        student_logits: Student model logits (B, S, V)
        teacher_logits: Teacher model logits (B, S, V)
        labels: Target token IDs for hard label loss
        temperature: Temperature for softening distributions (higher = softer)
        alpha: Weight for distillation loss (1 - alpha = weight for hard label loss)
        chunk_size: Number of tokens to process at once (lower = less memory)

    Returns:
        (total_loss, stats_dict)
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    mask = (labels != -100).float()
    n_valid = mask.sum().clamp(min=1)

    # Hard label cross-entropy loss (memory efficient - doesn't need full vocab materialization)
    ce_loss = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        labels.view(-1),
        reduction='none',
        ignore_index=-100
    ).view(batch_size, seq_len)
    ce_loss_mean = ce_loss.sum() / n_valid

    # Memory-efficient KL computation: process in chunks along sequence dimension
    kl_per_token = torch.zeros(batch_size, seq_len, device=student_logits.device, dtype=student_logits.dtype)

    # For logging - accumulate entropy stats
    teacher_entropy_sum = 0.0
    student_entropy_sum = 0.0
    n_entropy_samples = 0

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)

        # Get chunk slices
        student_chunk = student_logits[:, chunk_start:chunk_end, :]  # (B, chunk, V)
        teacher_chunk = teacher_logits[:, chunk_start:chunk_end, :]  # (B, chunk, V)
        mask_chunk = mask[:, chunk_start:chunk_end]  # (B, chunk)

        # Compute scaled log-softmax for student (needs gradients)
        student_log_soft = F.log_softmax(student_chunk / temperature, dim=-1)

        # Compute log-softmax for teacher (no gradients needed).
        # Using log_target=True avoids materializing a separate softmax tensor
        # and allows torch.compile to fuse exp(log_softmax) internally.
        with torch.no_grad():
            teacher_log_soft = F.log_softmax(teacher_chunk / temperature, dim=-1)

        # KL(teacher || student) = sum_v exp(teacher_log_soft) * (teacher_log_soft - student_log_soft)
        kl_chunk = F.kl_div(
            student_log_soft,
            teacher_log_soft,
            reduction='none',
            log_target=True,
        ).sum(dim=-1)  # (B, chunk)

        kl_per_token[:, chunk_start:chunk_end] = kl_chunk

        # Accumulate entropy statistics for logging (sampled from first chunk only to save compute)
        if chunk_start == 0:
            with torch.no_grad():
                chunk_mask = mask_chunk.bool()
                if chunk_mask.any():
                    # Reuse already-computed log_softmax for entropy — avoids extra log() call.
                    teacher_ent = -(teacher_log_soft * torch.exp(teacher_log_soft)).sum(dim=-1)
                    student_ent = -(student_log_soft * torch.exp(student_log_soft)).sum(dim=-1)
                    teacher_entropy_sum = teacher_ent[chunk_mask].sum().item()
                    student_entropy_sum = student_ent[chunk_mask].sum().item()
                    n_entropy_samples = chunk_mask.sum().item()

        # Free intermediate tensors
        del student_log_soft, teacher_log_soft, kl_chunk

    # Apply mask and compute mean KL
    kl_loss_mean = (kl_per_token * mask).sum() / n_valid

    # Scale KL by temperature² to maintain gradient magnitude
    kl_loss_scaled = kl_loss_mean * (temperature ** 2)

    # Combined loss
    total_loss = alpha * kl_loss_scaled + (1 - alpha) * ce_loss_mean

    with torch.no_grad():
        valid_kl = kl_per_token[mask.bool()]
        valid_ce = ce_loss[mask.bool()]

        stats = {
            "kd/kl_loss": kl_loss_scaled.item(),
            "kd/ce_loss": ce_loss_mean.item(),
            "kd/total_loss": total_loss.item(),
            "kd/kl_per_token_mean": valid_kl.mean().item(),
            "kd/kl_per_token_std": valid_kl.std().item(),
            "kd/ce_per_token_mean": valid_ce.mean().item(),
            "kd/teacher_entropy_mean": teacher_entropy_sum / max(n_entropy_samples, 1),
            "kd/student_entropy_mean": student_entropy_sum / max(n_entropy_samples, 1),
            "kd/temperature": temperature,
            "kd/alpha": alpha,
        }

    return total_loss, stats


def compute_reverse_kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
    chunk_size: int = 128,
) -> tuple:
    """Reverse-KL distillation (MiniLLM-style), same shape as kd-rl7b but with
    the *student* expectation:

        L = (1 - alpha) * CE(z_S, y)  +  alpha * T^2 * KL(p_S^T || p_T^T)

    where p_S^T = softmax(z_S / T) and p_T^T = softmax(z_T / T) are both at
    temperature T. The KL takes its expectation under p_S, not p_T:

        KL(p_S || p_T) = sum_v p_S(v) * (log p_S(v) - log p_T(v))

    Motivation (Gu et al. 2024, "MiniLLM"): forward KL is mode-covering -- it
    punishes the student for placing low mass where the teacher has high
    mass, which forces the student to spend capacity on *every* mode of the
    teacher, including high-entropy regions a smaller student cannot
    faithfully represent. Reverse KL is mode-seeking -- it punishes the
    student for placing high mass where the teacher has low mass, so the
    student can confidently concentrate on the dominant teacher mode(s) and
    ignore the long tail. For an LM with a capacity gap (1B student, 7B
    teacher), reverse KL is the theoretically-motivated divergence: it does
    not ask the student to do something it cannot do.

    Implementation:
      * Gradients flow through both p_S (target of the KL expectation) and
        log p_S (the denominator). Teacher log-softmax is detached.
      * Chunked along the seq dim, same memory profile as
        compute_kl_distillation_loss.
      * Per-chunk RKL uses F.kl_div(teacher_log_soft, student_log_soft,
        log_target=True) which is mathematically identical to the manual
        sum_v p_S * (log p_S - log p_T) (verified: bit-exact value AND
        gradient match the manual implementation). The fused kernel
        avoids explicitly materializing student_soft=exp(log_soft) in the
        autograd graph, reducing per-chunk memory footprint.
      * The RKL forward+backward is wrapped in torch.utils.checkpoint, so
        the per-chunk log-softmax / KL materializations are recomputed
        during backward instead of pinned across all chunks. This lets us
        train with a larger sequence-length / chunk-size budget without
        running into peak-activation OOM.
      * Telemetry (entropy, fkl-for-comparison) is computed under no_grad
        from the recomputed activations on the first chunk only, so it
        does not contribute to backward memory.

    Telemetry (kd_rkl/*):
      * rkl_loss      = alpha * T^2 * E[KL(p_S || p_T)]   (the actual loss term)
      * fkl_loss      = alpha * T^2 * E[KL(p_T || p_S)]   (the forward-KL
                        value at the same T on the same batch -- what
                        kd-rl7b sees. For comparison only; not in the loss.)
      * rkl_per_token_mean, fkl_per_token_mean, rkl/fkl ratio
      * teacher_entropy, student_entropy
      * temperature, alpha
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    mask = (labels != -100).float()
    n_valid = mask.sum().clamp(min=1)

    ce_loss = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        labels.view(-1),
        reduction='none',
        ignore_index=-100,
    ).view(batch_size, seq_len)
    ce_loss_mean = ce_loss.sum() / n_valid

    rkl_per_token = torch.zeros(batch_size, seq_len, device=student_logits.device, dtype=student_logits.dtype)
    fkl_per_token = torch.zeros(batch_size, seq_len, device=student_logits.device, dtype=student_logits.dtype)

    teacher_entropy_sum = 0.0
    student_entropy_sum = 0.0
    n_entropy_samples = 0

    # Inner KL kernel — wrapped in checkpoint to drop intermediate softmax
    # activations from peak memory. Returns the per-token RKL for this chunk.
    # NOTE: the function captures `temperature` from the enclosing scope; that
    # is safe because temperature is a Python float and not a tensor.
    def _rkl_chunk_fn(student_chunk, teacher_log_soft_detached):
        student_log_soft = F.log_softmax(student_chunk / temperature, dim=-1)
        # KL(p_S || p_T) via PyTorch's fused kernel. F.kl_div(input, target,
        # log_target=True, reduction='none') computes target.exp() * (target - input)
        # so passing (teacher_log_soft, student_log_soft) gives
        #   exp(student_log_soft) * (student_log_soft - teacher_log_soft)
        # which is the reverse-KL summand. Verified bit-exact vs manual.
        return F.kl_div(
            teacher_log_soft_detached,
            student_log_soft,
            reduction='none',
            log_target=True,
        ).sum(dim=-1)

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        student_chunk = student_logits[:, chunk_start:chunk_end, :]
        teacher_chunk = teacher_logits[:, chunk_start:chunk_end, :]
        mask_chunk = mask[:, chunk_start:chunk_end]

        with torch.no_grad():
            teacher_log_soft = F.log_softmax(teacher_chunk / temperature, dim=-1)

        # Activation-checkpointed RKL: forward materializes student_log_soft
        # inside the checkpoint and discards it; backward recomputes from
        # student_chunk. use_reentrant=False is required because we have
        # detached tensors in the closure (teacher_log_soft).
        rkl_chunk = torch.utils.checkpoint.checkpoint(
            _rkl_chunk_fn,
            student_chunk,
            teacher_log_soft,
            use_reentrant=False,
        )
        rkl_per_token[:, chunk_start:chunk_end] = rkl_chunk

        # Forward KL for telemetry only. No-grad path so it does not affect
        # backward memory. Re-derive student_log_soft under no_grad just for
        # this telemetry compute.
        with torch.no_grad():
            student_log_soft_nograd = F.log_softmax(student_chunk / temperature, dim=-1)
            fkl_chunk = (teacher_log_soft.exp() * (teacher_log_soft - student_log_soft_nograd)).sum(dim=-1)
            fkl_per_token[:, chunk_start:chunk_end] = fkl_chunk

            if chunk_start == 0:
                chunk_mask = mask_chunk.bool()
                if chunk_mask.any():
                    teacher_ent = -(teacher_log_soft * teacher_log_soft.exp()).sum(dim=-1)
                    student_ent = -(student_log_soft_nograd * student_log_soft_nograd.exp()).sum(dim=-1)
                    teacher_entropy_sum = teacher_ent[chunk_mask].sum().item()
                    student_entropy_sum = student_ent[chunk_mask].sum().item()
                    n_entropy_samples = int(chunk_mask.sum().item())
                    del teacher_ent, student_ent
            del student_log_soft_nograd

        del teacher_log_soft, rkl_chunk, fkl_chunk

    rkl_loss_mean = (rkl_per_token * mask).sum() / n_valid
    rkl_loss_scaled = rkl_loss_mean * (temperature ** 2)
    total_loss = alpha * rkl_loss_scaled + (1.0 - alpha) * ce_loss_mean

    with torch.no_grad():
        valid_mask = mask.bool()
        valid_rkl = rkl_per_token[valid_mask]
        valid_fkl = fkl_per_token[valid_mask]
        valid_ce  = ce_loss[valid_mask]
        ent_norm  = max(n_entropy_samples, 1)
        rkl_mean  = valid_rkl.mean().item() if valid_rkl.numel() > 0 else 0.0
        fkl_mean  = valid_fkl.mean().item() if valid_fkl.numel() > 0 else 0.0
        # Reverse / forward asymmetry. <1 means RKL is "easier" than FKL on
        # this batch (teacher has long-tail mass the student can ignore);
        # >1 means RKL is "harder" (student has mass on tokens the teacher
        # gives low probability to -- usually means student is wrong here).
        rkl_to_fkl_ratio = (rkl_mean / fkl_mean) if fkl_mean > 1e-12 else 0.0

        stats = {
            "kd_rkl/rkl_loss": rkl_loss_scaled.item(),
            "kd_rkl/fkl_loss": (fkl_mean * (temperature ** 2)) * alpha,
            "kd_rkl/ce_loss": ce_loss_mean.item(),
            "kd_rkl/total_loss": total_loss.item(),
            "kd_rkl/rkl_per_token_mean": rkl_mean,
            "kd_rkl/rkl_per_token_std": valid_rkl.std().item() if valid_rkl.numel() > 1 else 0.0,
            "kd_rkl/fkl_per_token_mean": fkl_mean,
            "kd_rkl/rkl_to_fkl_ratio": rkl_to_fkl_ratio,
            "kd_rkl/ce_per_token_mean": valid_ce.mean().item() if valid_ce.numel() > 0 else 0.0,
            "kd_rkl/teacher_entropy_mean": teacher_entropy_sum / ent_norm,
            "kd_rkl/student_entropy_mean": student_entropy_sum / ent_norm,
            "kd_rkl/temperature": float(temperature),
            "kd_rkl/alpha": float(alpha),
        }

    return total_loss, stats


def compute_rkl_fkl_mix_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
    chunk_size: int = 128,
) -> tuple:
    """Pure-KD mixture of reverse-KL and forward-KL (no CE term):

        L = T^2 * [ alpha * KL(p_S || p_T)  +  (1 - alpha) * KL(p_T || p_S) ]

    Motivation (paper-side): reverse-KL is mode-seeking and wins on multiple-
    choice / reasoning tasks but under-allocates mass to rare gold tokens
    (factual recall). Forward-KL is mass-covering, so a small fwdKL addback
    re-injects pressure to keep mass on positions the teacher likes that the
    student would otherwise discard. This *keeps* the no-CE design (the
    teacher carries all supervision, the student is never directly anchored
    to potentially-noisy gold tokens) while restoring the coverage property
    that pure reverse-KL gives up.

    Efficiency notes:
      * Both KL directions share student_log_soft and teacher_log_soft on
        the same chunk, so we compute both inside the SAME chunked loop and
        the SAME activation-checkpoint wrapper. This is ~2x cheaper than
        calling compute_reverse_kl_distillation_loss + a separate fwd-KL
        pass (which would double student forward-softmax work and double
        the saved-for-backward footprint).
      * The fused F.kl_div kernel is used in both directions:
          - Reverse: F.kl_div(teacher_log_soft, student_log_soft,
                              reduction='none', log_target=True).sum(-1)
            = exp(student_log_soft) * (student_log_soft - teacher_log_soft)
            = sum_v p_S(v) * (log p_S(v) - log p_T(v))   ✓ reverse-KL
          - Forward: F.kl_div(student_log_soft, teacher_log_soft,
                              reduction='none', log_target=True).sum(-1)
            = exp(teacher_log_soft) * (teacher_log_soft - student_log_soft)
            = sum_v p_T(v) * (log p_T(v) - log p_S(v))   ✓ forward-KL
        Both are mathematically identical to the manual forms (verified
        offline; the fwd-KL formula is the same one the canonical kd-rl7b
        path uses).
      * The chunked inner kernel is wrapped in torch.utils.checkpoint so
        student_log_soft is dropped from the saved-for-backward tensors
        and recomputed on the backward pass. Backward memory footprint
        ≈ one chunk's softmax, not seq_len // chunk_size of them.
      * Gradient path: only the (alpha * rkl + (1-alpha) * fkl) term in
        each chunk is in the autograd graph. Teacher log-softmax is
        computed once per chunk under no_grad and passed into the
        checkpointed function as a detached input.

    Telemetry (kd_rfkl/*): per-token RKL and FKL means (so we can see the
    relative magnitudes the two terms contribute), the scaled loss
    components, teacher/student entropy on the first chunk.
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    mask = (labels != -100).float()
    n_valid = mask.sum().clamp(min=1)

    rkl_per_token = torch.zeros(batch_size, seq_len, device=student_logits.device, dtype=student_logits.dtype)
    fkl_per_token = torch.zeros(batch_size, seq_len, device=student_logits.device, dtype=student_logits.dtype)

    teacher_entropy_sum = 0.0
    student_entropy_sum = 0.0
    n_entropy_samples = 0

    # Inner kernel — both KL directions in one pass, activation-checkpointed.
    # Returns (per-token RKL, per-token FKL) for this chunk. Both share the
    # same student_log_soft tensor, which is the load-bearing optimization.
    def _mix_chunk_fn(student_chunk, teacher_log_soft_detached):
        student_log_soft = F.log_softmax(student_chunk / temperature, dim=-1)
        # Reverse: sum_v p_S * (log p_S - log p_T)
        rkl = F.kl_div(
            teacher_log_soft_detached,
            student_log_soft,
            reduction='none',
            log_target=True,
        ).sum(dim=-1)
        # Forward: sum_v p_T * (log p_T - log p_S)
        fkl = F.kl_div(
            student_log_soft,
            teacher_log_soft_detached,
            reduction='none',
            log_target=True,
        ).sum(dim=-1)
        return rkl, fkl

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        student_chunk = student_logits[:, chunk_start:chunk_end, :]
        teacher_chunk = teacher_logits[:, chunk_start:chunk_end, :]
        mask_chunk = mask[:, chunk_start:chunk_end]

        with torch.no_grad():
            teacher_log_soft = F.log_softmax(teacher_chunk / temperature, dim=-1)

        rkl_chunk, fkl_chunk = torch.utils.checkpoint.checkpoint(
            _mix_chunk_fn,
            student_chunk,
            teacher_log_soft,
            use_reentrant=False,
        )
        rkl_per_token[:, chunk_start:chunk_end] = rkl_chunk
        fkl_per_token[:, chunk_start:chunk_end] = fkl_chunk

        # Entropy telemetry on the first chunk only, under no_grad.
        if chunk_start == 0:
            with torch.no_grad():
                chunk_mask = mask_chunk.bool()
                if chunk_mask.any():
                    student_log_soft_nograd = F.log_softmax(student_chunk / temperature, dim=-1)
                    teacher_ent = -(teacher_log_soft * teacher_log_soft.exp()).sum(dim=-1)
                    student_ent = -(student_log_soft_nograd * student_log_soft_nograd.exp()).sum(dim=-1)
                    teacher_entropy_sum = teacher_ent[chunk_mask].sum().item()
                    student_entropy_sum = student_ent[chunk_mask].sum().item()
                    n_entropy_samples = int(chunk_mask.sum().item())
                    del student_log_soft_nograd, teacher_ent, student_ent

        del teacher_log_soft, rkl_chunk, fkl_chunk

    rkl_loss_mean = (rkl_per_token * mask).sum() / n_valid
    fkl_loss_mean = (fkl_per_token * mask).sum() / n_valid
    rkl_loss_scaled = rkl_loss_mean * (temperature ** 2)
    fkl_loss_scaled = fkl_loss_mean * (temperature ** 2)
    total_loss = alpha * rkl_loss_scaled + (1.0 - alpha) * fkl_loss_scaled

    with torch.no_grad():
        valid_mask = mask.bool()
        valid_rkl = rkl_per_token[valid_mask]
        valid_fkl = fkl_per_token[valid_mask]
        ent_norm = max(n_entropy_samples, 1)
        rkl_mean = valid_rkl.mean().item() if valid_rkl.numel() > 0 else 0.0
        fkl_mean = valid_fkl.mean().item() if valid_fkl.numel() > 0 else 0.0
        rkl_to_fkl_ratio = (rkl_mean / fkl_mean) if fkl_mean > 1e-12 else 0.0

        stats = {
            "kd_rfkl/rkl_loss": rkl_loss_scaled.item(),
            "kd_rfkl/fkl_loss": fkl_loss_scaled.item(),
            "kd_rfkl/total_loss": total_loss.item(),
            "kd_rfkl/rkl_per_token_mean": rkl_mean,
            "kd_rfkl/rkl_per_token_std": valid_rkl.std().item() if valid_rkl.numel() > 1 else 0.0,
            "kd_rfkl/fkl_per_token_mean": fkl_mean,
            "kd_rfkl/fkl_per_token_std": valid_fkl.std().item() if valid_fkl.numel() > 1 else 0.0,
            "kd_rfkl/rkl_to_fkl_ratio": rkl_to_fkl_ratio,
            "kd_rfkl/teacher_entropy_mean": teacher_entropy_sum / ent_norm,
            "kd_rfkl/student_entropy_mean": student_entropy_sum / ent_norm,
            "kd_rfkl/temperature": float(temperature),
            "kd_rfkl/alpha": float(alpha),
        }

    return total_loss, stats


def compute_rkl_with_uniform_ce_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    chunk_size: int = 128,
    lambda_ce: float = 0.3,
) -> tuple:
    """Reverse-KL purekd on every token + uniform additive CE on every token.

        L_t = T² · KL(p_S || p_T)_t  +  λ · CE(y_t)_t

    The "no selection" control for the selective-CE family (idx 125, 126,
    129, 130, 131). All those recipes fire CE on a subset; this fires CE
    uniformly with constant weight λ. Used to isolate "selection matters"
    from "CE amount matters."

    Effective corpus-wide CE pressure = λ (vs idx 125's λ·topk_frac ≈ 0.10
    or idx 131's 1.0·topk_frac = 0.20). Sweeping λ ∈ {0.1, 0.3, 0.5, 1.0}
    maps the uniform-CE curve directly against the selective-CE points.

    Mechanism prediction:
      λ=0.1 → close to idx 100 (RKL purekd, NQ floor)
      λ=0.3 → comparable to idx 125 (selective at K=20%, λ=0.5 → 10% effective)
      λ=0.5 → comparable to idx 131 (replace at K=20% → 20% effective)
      λ=1.0 → close to canon (FKL+CE) NQ ceiling but with RKL backbone

    If results land on a smooth curve matching the predictions, "selection
    is irrelevant; only CE amount matters" is confirmed. If selective points
    sit ABOVE the uniform curve at matched effective CE, selection adds value.

    Reference idx (relaunch_rho1_2m_tokens.sh): 132 (λ=0.1), 133 (λ=0.3),
    134 (λ=0.5), 135 (λ=1.0).
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    mask = (labels != -100).float()
    n_valid = mask.sum().clamp(min=1)

    ce_per_token = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        labels.view(-1),
        reduction='none',
        ignore_index=-100,
    ).view(batch_size, seq_len)

    rkl_per_token = torch.zeros(
        batch_size, seq_len,
        device=student_logits.device, dtype=student_logits.dtype,
    )

    teacher_entropy_sum = 0.0
    student_entropy_sum = 0.0
    n_entropy_samples = 0

    def _rkl_chunk_fn(student_chunk, teacher_log_soft_detached):
        student_log_soft = F.log_softmax(student_chunk / temperature, dim=-1)
        return F.kl_div(
            teacher_log_soft_detached,
            student_log_soft,
            reduction='none',
            log_target=True,
        ).sum(dim=-1)

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        student_chunk = student_logits[:, chunk_start:chunk_end, :]
        teacher_chunk = teacher_logits[:, chunk_start:chunk_end, :]
        mask_chunk = mask[:, chunk_start:chunk_end]

        with torch.no_grad():
            teacher_log_soft = F.log_softmax(teacher_chunk / temperature, dim=-1)

        rkl_chunk = torch.utils.checkpoint.checkpoint(
            _rkl_chunk_fn,
            student_chunk,
            teacher_log_soft,
            use_reentrant=False,
        )
        rkl_per_token[:, chunk_start:chunk_end] = rkl_chunk

        if chunk_start == 0:
            with torch.no_grad():
                chunk_mask = mask_chunk.bool()
                if chunk_mask.any():
                    student_log_soft_nograd = F.log_softmax(student_chunk / temperature, dim=-1)
                    teacher_ent = -(teacher_log_soft * teacher_log_soft.exp()).sum(dim=-1)
                    student_ent = -(student_log_soft_nograd * student_log_soft_nograd.exp()).sum(dim=-1)
                    teacher_entropy_sum = teacher_ent[chunk_mask].sum().item()
                    student_entropy_sum = student_ent[chunk_mask].sum().item()
                    n_entropy_samples = int(chunk_mask.sum().item())
                    del student_log_soft_nograd, teacher_ent, student_ent

        del teacher_log_soft, rkl_chunk

    rkl_loss_mean = (rkl_per_token * mask).sum() / n_valid
    rkl_loss_scaled = rkl_loss_mean * (temperature ** 2)
    ce_loss_mean = (ce_per_token * mask).sum() / n_valid
    total_loss = rkl_loss_scaled + lambda_ce * ce_loss_mean

    with torch.no_grad():
        ent_norm = max(n_entropy_samples, 1)
        valid_mask = mask.bool()
        valid_rkl = rkl_per_token[valid_mask]
        valid_ce = ce_per_token[valid_mask]
        rkl_mean = valid_rkl.mean().item() if valid_rkl.numel() > 0 else 0.0
        ce_mean = valid_ce.mean().item() if valid_ce.numel() > 0 else 0.0

        stats = {
            "kd_rkl_uniform/rkl_loss": rkl_loss_scaled.item(),
            "kd_rkl_uniform/ce_term": (lambda_ce * ce_loss_mean).item(),
            "kd_rkl_uniform/total_loss": total_loss.item(),
            "kd_rkl_uniform/rkl_per_token_mean": rkl_mean,
            "kd_rkl_uniform/ce_per_token_mean": ce_mean,
            "kd_rkl_uniform/teacher_entropy_mean": teacher_entropy_sum / ent_norm,
            "kd_rkl_uniform/student_entropy_mean": student_entropy_sum / ent_norm,
            "kd_rkl_uniform/temperature": float(temperature),
            "kd_rkl_uniform/lambda_ce": float(lambda_ce),
        }

    return total_loss, stats


def compute_entropy_gated_rkl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    chunk_size: int = 128,
    lambda_ce: float = 1.0,
    gate_quantile: float = 0.30,
    entropy_temp: float = 2.0,
) -> tuple:
    """CE on every token; RKL fires only on tokens where the teacher is sharp.

        H_t   = H(p_T^{T=entropy_temp})_t
        tau   = quantile(H_valid, gate_quantile)
        m_t   = 1[H_t <= tau]                     # ~gate_quantile fraction of tokens fire
        L_t   = lambda_ce * CE(y_t)_t + m_t * T^2 * KL(p_S^{T} || p_T^{T})_t

    Motivation (idx 138):
      Teacher-entropy diagnostic on dolmino sources (T=2.0):
        math median H = 0.79 nats         (teacher confidently knows next token)
        wiki/pes2o/dclm median H = 7.3-7.6 (teacher near-maximum entropy)
      RKL on high-entropy tokens drags the student toward an *uninformed*
      teacher distribution -- structurally bad for factual/NQ tokens that
      have a sharp gold target. Gating RKL by teacher entropy stops that
      contamination while preserving RKL signal on math/structural tokens
      where the teacher is informative.

      All prior selective-CE recipes (125/126/129/130/131/132-135/137)
      shape the CE term and leave RKL on every token. This is the first
      recipe that shapes RKL itself.

    Adaptive threshold:
      tau = per-batch quantile so exactly gate_quantile of valid tokens
      fire RKL. Robust to batch composition (math-heavy batches fire on
      more tokens; wiki-heavy batches fire on fewer, which is correct).

    Hyperparameters:
      gate_quantile = 0.30  (math + structural tokens are ~math share 0.21 +
                             some flan/code; 0.30 is a safe upper bound)
      lambda_ce     = 1.0   (full NTP weight; the whole point is to let CE
                             drive NQ without RKL contamination)
      temperature   = 2.0   (matches all other RKL kernels; T^2 scaling preserved)
      entropy_temp  = 2.0   (entropy computed at the same T as KL for consistency
                             with the diagnostic measurement)

    Predicted outcome vs idx 100 (RKL-only):
      NQ:    13.9 -> rises toward NTP ceiling ~19-20 (CE unblocked on factual)
      gsm8k: 30   -> rises toward idx 125 ~62-65 (RKL preserved on math)
      m8:    37   -> rises toward idx 125 ~47 (mixed contributions)

    Reference idx (relaunch_rho1_2m_tokens.sh): 138.
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    mask = (labels != -100).float()
    n_valid = mask.sum().clamp(min=1)

    ce_per_token = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        labels.view(-1),
        reduction='none',
        ignore_index=-100,
    ).view(batch_size, seq_len)

    rkl_per_token = torch.zeros(
        batch_size, seq_len,
        device=student_logits.device, dtype=student_logits.dtype,
    )
    teacher_ent_per_token = torch.zeros(
        batch_size, seq_len,
        device=student_logits.device, dtype=torch.float32,
    )

    teacher_entropy_sum = 0.0
    student_entropy_sum = 0.0
    n_entropy_samples = 0

    def _rkl_chunk_fn(student_chunk, teacher_log_soft_detached):
        student_log_soft = F.log_softmax(student_chunk / temperature, dim=-1)
        return F.kl_div(
            teacher_log_soft_detached,
            student_log_soft,
            reduction='none',
            log_target=True,
        ).sum(dim=-1)

    # Pass 1: compute per-token teacher entropy (no_grad) and per-token RKL
    # (grad through student via activation checkpointing). Entropy is computed
    # for every chunk so the gate quantile sees the full batch distribution.
    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        student_chunk = student_logits[:, chunk_start:chunk_end, :]
        teacher_chunk = teacher_logits[:, chunk_start:chunk_end, :]
        mask_chunk = mask[:, chunk_start:chunk_end]

        with torch.no_grad():
            teacher_log_soft = F.log_softmax(teacher_chunk / temperature, dim=-1)
            # Entropy at the same temperature used by the KL term. If we ever
            # decouple, recompute log_softmax at entropy_temp here.
            teacher_ent_chunk = -(teacher_log_soft.exp() * teacher_log_soft).sum(dim=-1)
            teacher_ent_per_token[:, chunk_start:chunk_end] = teacher_ent_chunk.float()

        rkl_chunk = torch.utils.checkpoint.checkpoint(
            _rkl_chunk_fn,
            student_chunk,
            teacher_log_soft,
            use_reentrant=False,
        )
        rkl_per_token[:, chunk_start:chunk_end] = rkl_chunk

        # Telemetry-only entropy sums on first chunk
        if chunk_start == 0:
            with torch.no_grad():
                chunk_mask = mask_chunk.bool()
                if chunk_mask.any():
                    student_log_soft_nograd = F.log_softmax(student_chunk / temperature, dim=-1)
                    student_ent = -(student_log_soft_nograd * student_log_soft_nograd.exp()).sum(dim=-1)
                    teacher_entropy_sum = teacher_ent_chunk[chunk_mask].sum().item()
                    student_entropy_sum = student_ent[chunk_mask].sum().item()
                    n_entropy_samples = int(chunk_mask.sum().item())
                    del student_log_soft_nograd, student_ent
            del teacher_ent_chunk

        del teacher_log_soft, rkl_chunk

    # Compute adaptive gate. tau = quantile over valid tokens only.
    valid_mask = mask.bool()
    with torch.no_grad():
        valid_ent = teacher_ent_per_token[valid_mask]
        if valid_ent.numel() > 0:
            # gate_quantile = 0.30 means RKL fires on the 30% lowest-entropy
            # (most-confident-teacher) tokens.
            tau = torch.quantile(valid_ent.float(), gate_quantile).item()
        else:
            tau = 0.0
        gate = (teacher_ent_per_token <= tau).to(rkl_per_token.dtype) * mask
        n_fired = gate.sum().clamp(min=1)
        fired_frac = (gate.sum() / n_valid).item()

    # Gated RKL: average over fired tokens only (so loss magnitude is
    # comparable to idx 100's full-token RKL). T^2 scaling preserved.
    gated_rkl_per_token = rkl_per_token * gate
    rkl_loss_mean = gated_rkl_per_token.sum() / n_fired
    rkl_loss_scaled = rkl_loss_mean * (temperature ** 2)

    # CE: uniform on every valid token.
    ce_loss_mean = (ce_per_token * mask).sum() / n_valid

    total_loss = lambda_ce * ce_loss_mean + rkl_loss_scaled

    with torch.no_grad():
        ent_norm = max(n_entropy_samples, 1)
        valid_rkl_all = rkl_per_token[valid_mask]
        valid_rkl_fired = rkl_per_token[gate.bool()]
        valid_ce = ce_per_token[valid_mask]
        # Fired-token entropy stats for diagnosis: should be << global mean.
        fired_ent = teacher_ent_per_token[gate.bool()] if gate.sum() > 0 else valid_ent[:0]
        unfired_mask = (mask.bool()) & (~gate.bool())
        unfired_ent = teacher_ent_per_token[unfired_mask] if unfired_mask.sum() > 0 else valid_ent[:0]

        stats = {
            "kd_rkl_entgate/rkl_loss": rkl_loss_scaled.item(),
            "kd_rkl_entgate/ce_term": (lambda_ce * ce_loss_mean).item(),
            "kd_rkl_entgate/total_loss": total_loss.item(),
            "kd_rkl_entgate/rkl_per_token_mean_all": valid_rkl_all.mean().item() if valid_rkl_all.numel() > 0 else 0.0,
            "kd_rkl_entgate/rkl_per_token_mean_fired": valid_rkl_fired.mean().item() if valid_rkl_fired.numel() > 0 else 0.0,
            "kd_rkl_entgate/ce_per_token_mean": valid_ce.mean().item() if valid_ce.numel() > 0 else 0.0,
            "kd_rkl_entgate/teacher_entropy_mean": teacher_entropy_sum / ent_norm,
            "kd_rkl_entgate/student_entropy_mean": student_entropy_sum / ent_norm,
            "kd_rkl_entgate/teacher_entropy_full_mean": valid_ent.mean().item() if valid_ent.numel() > 0 else 0.0,
            "kd_rkl_entgate/teacher_entropy_fired_mean": fired_ent.mean().item() if fired_ent.numel() > 0 else 0.0,
            "kd_rkl_entgate/teacher_entropy_unfired_mean": unfired_ent.mean().item() if unfired_ent.numel() > 0 else 0.0,
            "kd_rkl_entgate/tau": float(tau),
            "kd_rkl_entgate/fired_frac": fired_frac,
            "kd_rkl_entgate/gate_quantile": float(gate_quantile),
            "kd_rkl_entgate/lambda_ce": float(lambda_ce),
            "kd_rkl_entgate/temperature": float(temperature),
        }

    return total_loss, stats

def compute_entropy_switched_rkl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    chunk_size: int = 128,
    lambda_ce: float = 1.0,
    gate_quantile: float = 0.30,
) -> tuple:
    """Mirror of compute_entropy_gated_rkl_loss but PARTITIONS CE and RKL
    rather than stacking them.

        H_t  = H(p_T^{T})_t
        tau  = quantile(H_t, gate_quantile)
        m_t  = 1[H_t <= tau]
        L_t  = (1 - m_t) * lambda_ce * CE(y_t) + m_t * T^2 * KL(p_S || p_T)_t

    Compared to idx 139 (which is `lambda_ce*CE + m_t*RKL`), this drops the
    CE term on the *fired* (low-teacher-entropy) tokens. Test: does CE on
    math/structural tokens help (139) or just dilute the RKL signal there
    (idx 142 partially supports the dilution hypothesis at the global
    level — this isolates it per-token). Reference idx: 148.

    Loss magnitudes follow the same averaging convention as idx 139:
      RKL averaged over fired tokens (so per-token weight is comparable
      across q-values), CE averaged over UNFIRED tokens (so the per-token
      CE weight is comparable to the unfired regime in idx 139).
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    mask = (labels != -100).float()
    n_valid = mask.sum().clamp(min=1)

    ce_per_token = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        labels.view(-1),
        reduction='none',
        ignore_index=-100,
    ).view(batch_size, seq_len)

    rkl_per_token = torch.zeros(
        batch_size, seq_len,
        device=student_logits.device, dtype=student_logits.dtype,
    )
    teacher_ent_per_token = torch.zeros(
        batch_size, seq_len,
        device=student_logits.device, dtype=torch.float32,
    )

    teacher_entropy_sum = 0.0
    student_entropy_sum = 0.0
    n_entropy_samples = 0

    def _rkl_chunk_fn(student_chunk, teacher_log_soft_detached):
        student_log_soft = F.log_softmax(student_chunk / temperature, dim=-1)
        return F.kl_div(
            teacher_log_soft_detached,
            student_log_soft,
            reduction='none',
            log_target=True,
        ).sum(dim=-1)

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        student_chunk = student_logits[:, chunk_start:chunk_end, :]
        teacher_chunk = teacher_logits[:, chunk_start:chunk_end, :]
        mask_chunk = mask[:, chunk_start:chunk_end]

        with torch.no_grad():
            teacher_log_soft = F.log_softmax(teacher_chunk / temperature, dim=-1)
            teacher_ent_chunk = -(teacher_log_soft.exp() * teacher_log_soft).sum(dim=-1)
            teacher_ent_per_token[:, chunk_start:chunk_end] = teacher_ent_chunk.float()

        rkl_chunk = torch.utils.checkpoint.checkpoint(
            _rkl_chunk_fn,
            student_chunk,
            teacher_log_soft,
            use_reentrant=False,
        )
        rkl_per_token[:, chunk_start:chunk_end] = rkl_chunk

        if chunk_start == 0:
            with torch.no_grad():
                chunk_mask = mask_chunk.bool()
                if chunk_mask.any():
                    student_log_soft_nograd = F.log_softmax(student_chunk / temperature, dim=-1)
                    student_ent = -(student_log_soft_nograd * student_log_soft_nograd.exp()).sum(dim=-1)
                    teacher_entropy_sum = teacher_ent_chunk[chunk_mask].sum().item()
                    student_entropy_sum = student_ent[chunk_mask].sum().item()
                    n_entropy_samples = int(chunk_mask.sum().item())
                    del student_log_soft_nograd, student_ent
            del teacher_ent_chunk

        del teacher_log_soft, rkl_chunk

    valid_mask = mask.bool()
    with torch.no_grad():
        valid_ent = teacher_ent_per_token[valid_mask]
        if valid_ent.numel() > 0:
            tau = torch.quantile(valid_ent.float(), gate_quantile).item()
        else:
            tau = 0.0
        gate = (teacher_ent_per_token <= tau).to(rkl_per_token.dtype) * mask
        ce_gate = (1.0 - gate) * mask        # CE fires on unfired tokens only
        n_fired = gate.sum().clamp(min=1)
        n_unfired = ce_gate.sum().clamp(min=1)
        fired_frac = (gate.sum() / n_valid).item()

    gated_rkl_per_token = rkl_per_token * gate
    rkl_loss_mean = gated_rkl_per_token.sum() / n_fired
    rkl_loss_scaled = rkl_loss_mean * (temperature ** 2)

    # CE averaged over UNFIRED tokens only (per-token weight comparable to
    # idx 139's unfired regime).
    ce_loss_mean = (ce_per_token * ce_gate).sum() / n_unfired

    total_loss = lambda_ce * ce_loss_mean + rkl_loss_scaled

    with torch.no_grad():
        ent_norm = max(n_entropy_samples, 1)
        valid_rkl_all = rkl_per_token[valid_mask]
        valid_rkl_fired = rkl_per_token[gate.bool()]
        valid_ce = ce_per_token[valid_mask]
        ce_unfired = ce_per_token[ce_gate.bool()] if ce_gate.sum() > 0 else valid_ce[:0]
        fired_ent = teacher_ent_per_token[gate.bool()] if gate.sum() > 0 else valid_ent[:0]
        unfired_mask = (mask.bool()) & (~gate.bool())
        unfired_ent = teacher_ent_per_token[unfired_mask] if unfired_mask.sum() > 0 else valid_ent[:0]

        stats = {
            "kd_rkl_entswitch/rkl_loss": rkl_loss_scaled.item(),
            "kd_rkl_entswitch/ce_term": (lambda_ce * ce_loss_mean).item(),
            "kd_rkl_entswitch/total_loss": total_loss.item(),
            "kd_rkl_entswitch/rkl_per_token_mean_all": valid_rkl_all.mean().item() if valid_rkl_all.numel() > 0 else 0.0,
            "kd_rkl_entswitch/rkl_per_token_mean_fired": valid_rkl_fired.mean().item() if valid_rkl_fired.numel() > 0 else 0.0,
            "kd_rkl_entswitch/ce_per_token_mean_all": valid_ce.mean().item() if valid_ce.numel() > 0 else 0.0,
            "kd_rkl_entswitch/ce_per_token_mean_unfired": ce_unfired.mean().item() if ce_unfired.numel() > 0 else 0.0,
            "kd_rkl_entswitch/teacher_entropy_mean": teacher_entropy_sum / ent_norm,
            "kd_rkl_entswitch/student_entropy_mean": student_entropy_sum / ent_norm,
            "kd_rkl_entswitch/teacher_entropy_full_mean": valid_ent.mean().item() if valid_ent.numel() > 0 else 0.0,
            "kd_rkl_entswitch/teacher_entropy_fired_mean": fired_ent.mean().item() if fired_ent.numel() > 0 else 0.0,
            "kd_rkl_entswitch/teacher_entropy_unfired_mean": unfired_ent.mean().item() if unfired_ent.numel() > 0 else 0.0,
            "kd_rkl_entswitch/tau": float(tau),
            "kd_rkl_entswitch/fired_frac": fired_frac,
            "kd_rkl_entswitch/gate_quantile": float(gate_quantile),
            "kd_rkl_entswitch/lambda_ce": float(lambda_ce),
            "kd_rkl_entswitch/temperature": float(temperature),
        }

    return total_loss, stats


def compute_entropy_gated_fkl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    chunk_size: int = 128,
    lambda_ce: float = 1.0,
    gate_quantile: float = 0.30,
    entropy_temp: float = 2.0,
) -> tuple:
    """Mirror of compute_entropy_gated_rkl_loss but with FKL = KL(p_T || p_S).

    Matches the kd-rl baseline FKL form (see compute_kl_distillation_loss):
    both distributions softened at T, loss scaled by T^2.
    Ablation purpose: isolate the contribution of the *reverse* direction in
    KL. If gains under entgate persist with FKL, the win is from masking, not
    from RKL specifically. Reference idx: 140.
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    mask = (labels != -100).float()
    n_valid = mask.sum().clamp(min=1)

    ce_per_token = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        labels.view(-1),
        reduction='none',
        ignore_index=-100,
    ).view(batch_size, seq_len)

    fkl_per_token = torch.zeros(
        batch_size, seq_len,
        device=student_logits.device, dtype=student_logits.dtype,
    )
    teacher_ent_per_token = torch.zeros(
        batch_size, seq_len,
        device=student_logits.device, dtype=torch.float32,
    )

    teacher_entropy_sum = 0.0
    student_entropy_sum = 0.0
    n_entropy_samples = 0

    def _fkl_chunk_fn(student_chunk, teacher_log_soft_detached):
        student_log_soft = F.log_softmax(student_chunk / temperature, dim=-1)
        return F.kl_div(
            student_log_soft,
            teacher_log_soft_detached,
            reduction='none',
            log_target=True,
        ).sum(dim=-1)

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        student_chunk = student_logits[:, chunk_start:chunk_end, :]
        teacher_chunk = teacher_logits[:, chunk_start:chunk_end, :]
        mask_chunk = mask[:, chunk_start:chunk_end]

        with torch.no_grad():
            teacher_log_soft = F.log_softmax(teacher_chunk / temperature, dim=-1)
            teacher_ent_chunk = -(teacher_log_soft.exp() * teacher_log_soft).sum(dim=-1)
            teacher_ent_per_token[:, chunk_start:chunk_end] = teacher_ent_chunk.float()

        fkl_chunk = torch.utils.checkpoint.checkpoint(
            _fkl_chunk_fn,
            student_chunk,
            teacher_log_soft,
            use_reentrant=False,
        )
        fkl_per_token[:, chunk_start:chunk_end] = fkl_chunk

        if chunk_start == 0:
            with torch.no_grad():
                chunk_mask = mask_chunk.bool()
                if chunk_mask.any():
                    student_log_soft_nograd = F.log_softmax(student_chunk / temperature, dim=-1)
                    student_ent = -(student_log_soft_nograd * student_log_soft_nograd.exp()).sum(dim=-1)
                    teacher_entropy_sum = teacher_ent_chunk[chunk_mask].sum().item()
                    student_entropy_sum = student_ent[chunk_mask].sum().item()
                    n_entropy_samples = int(chunk_mask.sum().item())
                    del student_log_soft_nograd, student_ent
            del teacher_ent_chunk

        del teacher_log_soft, fkl_chunk

    valid_mask = mask.bool()
    with torch.no_grad():
        valid_ent = teacher_ent_per_token[valid_mask]
        if valid_ent.numel() > 0:
            tau = torch.quantile(valid_ent.float(), gate_quantile).item()
        else:
            tau = 0.0
        gate = (teacher_ent_per_token <= tau).to(fkl_per_token.dtype) * mask
        n_fired = gate.sum().clamp(min=1)
        fired_frac = (gate.sum() / n_valid).item()

    gated_fkl_per_token = fkl_per_token * gate
    fkl_loss_mean = gated_fkl_per_token.sum() / n_fired
    fkl_loss_scaled = fkl_loss_mean * (temperature ** 2)

    ce_loss_mean = (ce_per_token * mask).sum() / n_valid
    total_loss = lambda_ce * ce_loss_mean + fkl_loss_scaled

    with torch.no_grad():
        ent_norm = max(n_entropy_samples, 1)
        valid_fkl_all = fkl_per_token[valid_mask]
        valid_fkl_fired = fkl_per_token[gate.bool()]
        valid_ce = ce_per_token[valid_mask]
        fired_ent = teacher_ent_per_token[gate.bool()] if gate.sum() > 0 else valid_ent[:0]
        unfired_mask = (mask.bool()) & (~gate.bool())
        unfired_ent = teacher_ent_per_token[unfired_mask] if unfired_mask.sum() > 0 else valid_ent[:0]

        stats = {
            "kd_fkl_entgate/fkl_loss": fkl_loss_scaled.item(),
            "kd_fkl_entgate/ce_term": (lambda_ce * ce_loss_mean).item(),
            "kd_fkl_entgate/total_loss": total_loss.item(),
            "kd_fkl_entgate/fkl_per_token_mean_all": valid_fkl_all.mean().item() if valid_fkl_all.numel() > 0 else 0.0,
            "kd_fkl_entgate/fkl_per_token_mean_fired": valid_fkl_fired.mean().item() if valid_fkl_fired.numel() > 0 else 0.0,
            "kd_fkl_entgate/ce_per_token_mean": valid_ce.mean().item() if valid_ce.numel() > 0 else 0.0,
            "kd_fkl_entgate/teacher_entropy_mean": teacher_entropy_sum / ent_norm,
            "kd_fkl_entgate/student_entropy_mean": student_entropy_sum / ent_norm,
            "kd_fkl_entgate/teacher_entropy_full_mean": valid_ent.mean().item() if valid_ent.numel() > 0 else 0.0,
            "kd_fkl_entgate/teacher_entropy_fired_mean": fired_ent.mean().item() if fired_ent.numel() > 0 else 0.0,
            "kd_fkl_entgate/teacher_entropy_unfired_mean": unfired_ent.mean().item() if unfired_ent.numel() > 0 else 0.0,
            "kd_fkl_entgate/tau": float(tau),
            "kd_fkl_entgate/fired_frac": fired_frac,
            "kd_fkl_entgate/gate_quantile": float(gate_quantile),
            "kd_fkl_entgate/lambda_ce": float(lambda_ce),
            "kd_fkl_entgate/temperature": float(temperature),
        }

    return total_loss, stats


def compute_entropy_band_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    chunk_size: int = 128,
    lambda_ce: float = 1.0,
    low_band_quantile: float = 0.30,
    high_band_quantile: float = 0.30,
    entropy_temp: float = 2.0,
) -> tuple:
    """Complement-band KD: RKL on the bottom-low_band fraction of teacher
    entropy tokens, FKL on the top-high_band fraction; CE on every valid token.
    Middle band gets CE only.

    Diagnostic question: do high-entropy teacher distributions contain useful
    "soft target" information that CE alone misses? Equivalent compute design:
      bottom 30% by H(p_T): RKL  (mode-seeking on confident teacher tokens)
      top    30% by H(p_T): FKL  (mode-covering on uncertain teacher tokens)
      middle 40%:           CE only (no KD signal at all)

    Loss:
        H_t       = H(p_T^{T=entropy_temp})_t
        tau_low   = quantile(H_valid, low_band_quantile)
        tau_high  = quantile(H_valid, 1 - high_band_quantile)
        m_low_t   = 1[H_t <= tau_low]   (~low_band_quantile fire RKL)
        m_high_t  = 1[H_t >= tau_high]  (~high_band_quantile fire FKL)
        L = lambda_ce * CE(y) + T^2 * [mean_over_fired(m_low * RKL) + mean_over_fired(m_high * FKL)]

    Budget control: each direction's KL term is averaged over its own fired
    token count (not total). This matches idx 139's convention (RKL term scaled
    by 1/n_fired) so the per-direction loss magnitude is comparable to a single
    pure-direction recipe. Total KL pressure roughly 2x a single-direction
    entgate recipe (because both terms fire); compensated by smaller fired
    fractions per direction (30% vs 30% = same per-direction budget as idx 139).

    Note this differs from AKL/RKL+FKL-mix recipes by:
      (a) tokens are partitioned by entropy, not blended on every token
      (b) the divergence direction is conditional on entropy band
      (c) middle band has NO KD pressure (deliberate; CE-only control)

    Hyperparameters mirrored from idx 139:
      low_band_quantile  = 0.30  (matches idx 139 RKL fire-rate)
      high_band_quantile = 0.30  (symmetric high-end band)
      lambda_ce          = 1.0
      temperature        = 2.0
      entropy_temp       = 2.0
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    mask = (labels != -100).float()
    n_valid = mask.sum().clamp(min=1)

    ce_per_token = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        labels.view(-1),
        reduction='none',
        ignore_index=-100,
    ).view(batch_size, seq_len)

    rkl_per_token = torch.zeros(
        batch_size, seq_len,
        device=student_logits.device, dtype=student_logits.dtype,
    )
    fkl_per_token = torch.zeros(
        batch_size, seq_len,
        device=student_logits.device, dtype=student_logits.dtype,
    )
    teacher_ent_per_token = torch.zeros(
        batch_size, seq_len,
        device=student_logits.device, dtype=torch.float32,
    )

    teacher_entropy_sum = 0.0
    student_entropy_sum = 0.0
    n_entropy_samples = 0

    def _rkl_chunk_fn(student_chunk, teacher_log_soft_detached):
        student_log_soft = F.log_softmax(student_chunk / temperature, dim=-1)
        # RKL = KL(p_S || p_T) = sum_v p_S * (log p_S - log p_T)
        # F.kl_div(input=log_target, target=log_input, log_target=True) computes
        # sum_v target.exp() * (log target - input). With input=p_T's log and
        # target=p_S's log: sum p_S * (log p_S - log p_T) = KL(p_S||p_T). ✓
        return F.kl_div(
            teacher_log_soft_detached,
            student_log_soft,
            reduction='none',
            log_target=True,
        ).sum(dim=-1)

    def _fkl_chunk_fn(student_chunk, teacher_log_soft_detached):
        student_log_soft = F.log_softmax(student_chunk / temperature, dim=-1)
        # FKL = KL(p_T || p_S) = sum p_T * (log p_T - log p_S)
        return F.kl_div(
            student_log_soft,
            teacher_log_soft_detached,
            reduction='none',
            log_target=True,
        ).sum(dim=-1)

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        student_chunk = student_logits[:, chunk_start:chunk_end, :]
        teacher_chunk = teacher_logits[:, chunk_start:chunk_end, :]
        mask_chunk = mask[:, chunk_start:chunk_end]

        with torch.no_grad():
            teacher_log_soft = F.log_softmax(teacher_chunk / temperature, dim=-1)
            teacher_ent_chunk = -(teacher_log_soft.exp() * teacher_log_soft).sum(dim=-1)
            teacher_ent_per_token[:, chunk_start:chunk_end] = teacher_ent_chunk.float()

        rkl_chunk = torch.utils.checkpoint.checkpoint(
            _rkl_chunk_fn,
            student_chunk,
            teacher_log_soft,
            use_reentrant=False,
        )
        rkl_per_token[:, chunk_start:chunk_end] = rkl_chunk

        fkl_chunk = torch.utils.checkpoint.checkpoint(
            _fkl_chunk_fn,
            student_chunk,
            teacher_log_soft,
            use_reentrant=False,
        )
        fkl_per_token[:, chunk_start:chunk_end] = fkl_chunk

        if chunk_start == 0:
            with torch.no_grad():
                chunk_mask = mask_chunk.bool()
                if chunk_mask.any():
                    student_log_soft_nograd = F.log_softmax(student_chunk / temperature, dim=-1)
                    student_ent = -(student_log_soft_nograd * student_log_soft_nograd.exp()).sum(dim=-1)
                    teacher_entropy_sum = teacher_ent_chunk[chunk_mask].sum().item()
                    student_entropy_sum = student_ent[chunk_mask].sum().item()
                    n_entropy_samples = int(chunk_mask.sum().item())
                    del student_log_soft_nograd, student_ent
            del teacher_ent_chunk

        del teacher_log_soft, rkl_chunk, fkl_chunk

    valid_mask = mask.bool()
    with torch.no_grad():
        valid_ent = teacher_ent_per_token[valid_mask]
        if valid_ent.numel() > 0:
            tau_low = torch.quantile(valid_ent.float(), low_band_quantile).item()
            tau_high = torch.quantile(valid_ent.float(), 1.0 - high_band_quantile).item()
        else:
            tau_low = 0.0
            tau_high = 0.0
        low_gate = (teacher_ent_per_token <= tau_low).to(rkl_per_token.dtype) * mask
        high_gate = (teacher_ent_per_token >= tau_high).to(fkl_per_token.dtype) * mask
        n_low = low_gate.sum().clamp(min=1)
        n_high = high_gate.sum().clamp(min=1)
        low_frac = (low_gate.sum() / n_valid).item()
        high_frac = (high_gate.sum() / n_valid).item()

    # Per-direction average (matches idx 139's convention: KL averaged over
    # fired tokens, scaled by T^2). Each direction contributes one comparable
    # KL term.
    rkl_loss_mean = (rkl_per_token * low_gate).sum() / n_low
    fkl_loss_mean = (fkl_per_token * high_gate).sum() / n_high
    rkl_loss_scaled = rkl_loss_mean * (temperature ** 2)
    fkl_loss_scaled = fkl_loss_mean * (temperature ** 2)

    ce_loss_mean = (ce_per_token * mask).sum() / n_valid

    total_loss = lambda_ce * ce_loss_mean + rkl_loss_scaled + fkl_loss_scaled

    with torch.no_grad():
        ent_norm = max(n_entropy_samples, 1)
        valid_rkl_all = rkl_per_token[valid_mask]
        valid_fkl_all = fkl_per_token[valid_mask]
        valid_ce = ce_per_token[valid_mask]
        low_ent_fired = teacher_ent_per_token[low_gate.bool()] if low_gate.sum() > 0 else valid_ent[:0]
        high_ent_fired = teacher_ent_per_token[high_gate.bool()] if high_gate.sum() > 0 else valid_ent[:0]
        middle_mask = (mask.bool()) & (~low_gate.bool()) & (~high_gate.bool())
        middle_ent = teacher_ent_per_token[middle_mask] if middle_mask.sum() > 0 else valid_ent[:0]

        stats = {
            "kd_band/rkl_loss": rkl_loss_scaled.item(),
            "kd_band/fkl_loss": fkl_loss_scaled.item(),
            "kd_band/ce_term": (lambda_ce * ce_loss_mean).item(),
            "kd_band/total_loss": total_loss.item(),
            "kd_band/rkl_per_token_mean_all": valid_rkl_all.mean().item() if valid_rkl_all.numel() > 0 else 0.0,
            "kd_band/fkl_per_token_mean_all": valid_fkl_all.mean().item() if valid_fkl_all.numel() > 0 else 0.0,
            "kd_band/ce_per_token_mean": valid_ce.mean().item() if valid_ce.numel() > 0 else 0.0,
            "kd_band/teacher_entropy_mean": teacher_entropy_sum / ent_norm,
            "kd_band/student_entropy_mean": student_entropy_sum / ent_norm,
            "kd_band/teacher_entropy_full_mean": valid_ent.mean().item() if valid_ent.numel() > 0 else 0.0,
            "kd_band/teacher_entropy_low_mean": low_ent_fired.mean().item() if low_ent_fired.numel() > 0 else 0.0,
            "kd_band/teacher_entropy_high_mean": high_ent_fired.mean().item() if high_ent_fired.numel() > 0 else 0.0,
            "kd_band/teacher_entropy_middle_mean": middle_ent.mean().item() if middle_ent.numel() > 0 else 0.0,
            "kd_band/tau_low": float(tau_low),
            "kd_band/tau_high": float(tau_high),
            "kd_band/low_frac": low_frac,
            "kd_band/high_frac": high_frac,
            "kd_band/middle_frac": 1.0 - low_frac - high_frac,
            "kd_band/low_band_quantile": float(low_band_quantile),
            "kd_band/high_band_quantile": float(high_band_quantile),
            "kd_band/lambda_ce": float(lambda_ce),
            "kd_band/temperature": float(temperature),
        }

    return total_loss, stats


def compute_random_gated_rkl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    chunk_size: int = 128,
    lambda_ce: float = 1.0,
    gate_quantile: float = 0.30,
    base_seed: int = 0,
    global_step: int = 0,
    rank: int = 0,
) -> tuple:
    """Mirror of compute_entropy_gated_rkl_loss but with a *random* binary mask
    matched to the entgate fire count instead of an entropy-quantile gate.

    Control for "is the win from entropy specifically, or any 30% sparsity?"
    Mask is a per-batch uniform-random subset of valid tokens of size
    floor(gate_quantile * n_valid). Seed = base_seed + global_step + rank, so
    masks are reproducible within a run and uncorrelated across ranks/steps.
    Reference idx: 141.
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    mask = (labels != -100).float()
    n_valid = mask.sum().clamp(min=1)

    ce_per_token = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        labels.view(-1),
        reduction='none',
        ignore_index=-100,
    ).view(batch_size, seq_len)

    rkl_per_token = torch.zeros(
        batch_size, seq_len,
        device=student_logits.device, dtype=student_logits.dtype,
    )
    teacher_ent_per_token = torch.zeros(
        batch_size, seq_len,
        device=student_logits.device, dtype=torch.float32,
    )
    teacher_entropy_sum = 0.0
    student_entropy_sum = 0.0
    n_entropy_samples = 0

    def _rkl_chunk_fn(student_chunk, teacher_log_soft_detached):
        student_log_soft = F.log_softmax(student_chunk / temperature, dim=-1)
        return F.kl_div(
            teacher_log_soft_detached,
            student_log_soft,
            reduction='none',
            log_target=True,
        ).sum(dim=-1)

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        student_chunk = student_logits[:, chunk_start:chunk_end, :]
        teacher_chunk = teacher_logits[:, chunk_start:chunk_end, :]
        mask_chunk = mask[:, chunk_start:chunk_end]

        with torch.no_grad():
            teacher_log_soft = F.log_softmax(teacher_chunk / temperature, dim=-1)
            teacher_ent_chunk = -(teacher_log_soft.exp() * teacher_log_soft).sum(dim=-1)
            teacher_ent_per_token[:, chunk_start:chunk_end] = teacher_ent_chunk.float()

        rkl_chunk = torch.utils.checkpoint.checkpoint(
            _rkl_chunk_fn,
            student_chunk,
            teacher_log_soft,
            use_reentrant=False,
        )
        rkl_per_token[:, chunk_start:chunk_end] = rkl_chunk

        if chunk_start == 0:
            with torch.no_grad():
                chunk_mask = mask_chunk.bool()
                if chunk_mask.any():
                    student_log_soft_nograd = F.log_softmax(student_chunk / temperature, dim=-1)
                    student_ent = -(student_log_soft_nograd * student_log_soft_nograd.exp()).sum(dim=-1)
                    teacher_entropy_sum = teacher_ent_chunk[chunk_mask].sum().item()
                    student_entropy_sum = student_ent[chunk_mask].sum().item()
                    n_entropy_samples = int(chunk_mask.sum().item())
                    del student_log_soft_nograd, student_ent
            del teacher_ent_chunk

        del teacher_log_soft, rkl_chunk

    # Random mask matched to entgate fire count: top-k over uniform-random scores
    # restricted to valid (non-ignore) tokens. Identical |fired| to entgate so
    # the only differing variable is *which* tokens fire, not how many.
    with torch.no_grad():
        valid_mask = mask.bool()
        n_valid_int = int(n_valid.item())
        n_fire = max(int(gate_quantile * n_valid_int), 1)
        gen = torch.Generator(device=student_logits.device)
        gen.manual_seed(int(base_seed) + int(global_step) + int(rank))
        scores = torch.empty(batch_size, seq_len, device=student_logits.device).uniform_(0.0, 1.0, generator=gen)
        scores_valid = torch.where(valid_mask, scores, torch.full_like(scores, float('-inf')))
        flat_scores = scores_valid.flatten()
        topk_vals, topk_idx = torch.topk(flat_scores, n_fire, sorted=False)
        gate = torch.zeros_like(scores)
        gate.view(-1)[topk_idx] = 1.0
        gate = gate * mask
        n_fired = gate.sum().clamp(min=1)
        fired_frac = (gate.sum() / n_valid).item()

    gated_rkl_per_token = rkl_per_token * gate
    rkl_loss_mean = gated_rkl_per_token.sum() / n_fired
    rkl_loss_scaled = rkl_loss_mean * (temperature ** 2)

    ce_loss_mean = (ce_per_token * mask).sum() / n_valid
    total_loss = lambda_ce * ce_loss_mean + rkl_loss_scaled

    with torch.no_grad():
        ent_norm = max(n_entropy_samples, 1)
        valid_rkl_all = rkl_per_token[valid_mask]
        valid_rkl_fired = rkl_per_token[gate.bool()]
        valid_ce = ce_per_token[valid_mask]
        valid_ent = teacher_ent_per_token[valid_mask]
        fired_ent = teacher_ent_per_token[gate.bool()] if gate.sum() > 0 else valid_ent[:0]
        unfired_mask = (mask.bool()) & (~gate.bool())
        unfired_ent = teacher_ent_per_token[unfired_mask] if unfired_mask.sum() > 0 else valid_ent[:0]

        stats = {
            "kd_rkl_randmask/rkl_loss": rkl_loss_scaled.item(),
            "kd_rkl_randmask/ce_term": (lambda_ce * ce_loss_mean).item(),
            "kd_rkl_randmask/total_loss": total_loss.item(),
            "kd_rkl_randmask/rkl_per_token_mean_all": valid_rkl_all.mean().item() if valid_rkl_all.numel() > 0 else 0.0,
            "kd_rkl_randmask/rkl_per_token_mean_fired": valid_rkl_fired.mean().item() if valid_rkl_fired.numel() > 0 else 0.0,
            "kd_rkl_randmask/ce_per_token_mean": valid_ce.mean().item() if valid_ce.numel() > 0 else 0.0,
            "kd_rkl_randmask/teacher_entropy_mean": teacher_entropy_sum / ent_norm,
            "kd_rkl_randmask/student_entropy_mean": student_entropy_sum / ent_norm,
            "kd_rkl_randmask/teacher_entropy_full_mean": valid_ent.mean().item() if valid_ent.numel() > 0 else 0.0,
            "kd_rkl_randmask/teacher_entropy_fired_mean": fired_ent.mean().item() if fired_ent.numel() > 0 else 0.0,
            "kd_rkl_randmask/teacher_entropy_unfired_mean": unfired_ent.mean().item() if unfired_ent.numel() > 0 else 0.0,
            "kd_rkl_randmask/fired_frac": fired_frac,
            "kd_rkl_randmask/gate_quantile": float(gate_quantile),
            "kd_rkl_randmask/lambda_ce": float(lambda_ce),
            "kd_rkl_randmask/temperature": float(temperature),
            "kd_rkl_randmask/base_seed": float(base_seed),
        }

    return total_loss, stats


def compute_student_entropy_gated_rkl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    chunk_size: int = 128,
    lambda_ce: float = 1.0,
    gate_quantile: float = 0.30,
) -> tuple:
    """Mirror of compute_entropy_gated_rkl_loss but gates on STUDENT entropy
    (top-K most uncertain) instead of teacher entropy (bottom-K most confident).

        H_S_t = H(p_S^{T=temperature})_t           # student entropy (no_grad)
        tau   = quantile(H_S_valid, 1 - gate_quantile)
        m_t   = 1[H_S_t >= tau]                    # ~gate_quantile fraction fire
        L_t   = lambda_ce * CE(y_t) + m_t * T^2 * KL(p_S || p_T)_t

    Active-learning framing of the gate: spend KD signal where the student
    is least confident (highest entropy), instead of where the teacher is
    most confident. Reference idx: 143.

    Gate selection is computed under no_grad so that the student's entropy
    choosing which tokens fire does NOT backprop into the student logits.
    Only the gated KL term provides student-side gradient.
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    mask = (labels != -100).float()
    n_valid = mask.sum().clamp(min=1)

    ce_per_token = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        labels.view(-1),
        reduction='none',
        ignore_index=-100,
    ).view(batch_size, seq_len)

    rkl_per_token = torch.zeros(
        batch_size, seq_len,
        device=student_logits.device, dtype=student_logits.dtype,
    )
    student_ent_per_token = torch.zeros(
        batch_size, seq_len,
        device=student_logits.device, dtype=torch.float32,
    )
    teacher_entropy_sum = 0.0
    student_entropy_sum = 0.0
    n_entropy_samples = 0

    def _rkl_chunk_fn(student_chunk, teacher_log_soft_detached):
        student_log_soft = F.log_softmax(student_chunk / temperature, dim=-1)
        return F.kl_div(
            teacher_log_soft_detached,
            student_log_soft,
            reduction='none',
            log_target=True,
        ).sum(dim=-1)

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        student_chunk = student_logits[:, chunk_start:chunk_end, :]
        teacher_chunk = teacher_logits[:, chunk_start:chunk_end, :]
        mask_chunk = mask[:, chunk_start:chunk_end]

        with torch.no_grad():
            teacher_log_soft = F.log_softmax(teacher_chunk / temperature, dim=-1)
            # Student entropy for the gate; no grad so the gate selection
            # itself doesn't backprop into the student logits.
            student_log_soft_nograd = F.log_softmax(student_chunk / temperature, dim=-1)
            student_ent_chunk = -(student_log_soft_nograd.exp() * student_log_soft_nograd).sum(dim=-1)
            student_ent_per_token[:, chunk_start:chunk_end] = student_ent_chunk.float()
            teacher_ent_chunk = -(teacher_log_soft.exp() * teacher_log_soft).sum(dim=-1)

        rkl_chunk = torch.utils.checkpoint.checkpoint(
            _rkl_chunk_fn,
            student_chunk,
            teacher_log_soft,
            use_reentrant=False,
        )
        rkl_per_token[:, chunk_start:chunk_end] = rkl_chunk

        if chunk_start == 0:
            with torch.no_grad():
                chunk_mask = mask_chunk.bool()
                if chunk_mask.any():
                    teacher_entropy_sum = teacher_ent_chunk[chunk_mask].sum().item()
                    student_entropy_sum = student_ent_chunk[chunk_mask].sum().item()
                    n_entropy_samples = int(chunk_mask.sum().item())
            del teacher_ent_chunk, student_ent_chunk

        del teacher_log_soft, student_log_soft_nograd, rkl_chunk

    # Gate fires on HIGHEST student-entropy tokens (top-K).
    # tau = quantile(H_S, 1 - gate_quantile) so that the top gate_quantile
    # fraction of tokens (by student entropy) fire.
    valid_mask = mask.bool()
    with torch.no_grad():
        valid_sent = student_ent_per_token[valid_mask]
        if valid_sent.numel() > 0:
            tau = torch.quantile(valid_sent.float(), 1.0 - gate_quantile).item()
        else:
            tau = 0.0
        gate = (student_ent_per_token >= tau).to(rkl_per_token.dtype) * mask
        n_fired = gate.sum().clamp(min=1)
        fired_frac = (gate.sum() / n_valid).item()

    gated_rkl_per_token = rkl_per_token * gate
    rkl_loss_mean = gated_rkl_per_token.sum() / n_fired
    rkl_loss_scaled = rkl_loss_mean * (temperature ** 2)

    ce_loss_mean = (ce_per_token * mask).sum() / n_valid
    total_loss = lambda_ce * ce_loss_mean + rkl_loss_scaled

    with torch.no_grad():
        ent_norm = max(n_entropy_samples, 1)
        valid_rkl_all = rkl_per_token[valid_mask]
        valid_rkl_fired = rkl_per_token[gate.bool()]
        valid_ce = ce_per_token[valid_mask]
        fired_sent = student_ent_per_token[gate.bool()] if gate.sum() > 0 else valid_sent[:0]
        unfired_mask = (mask.bool()) & (~gate.bool())
        unfired_sent = student_ent_per_token[unfired_mask] if unfired_mask.sum() > 0 else valid_sent[:0]

        stats = {
            "kd_rkl_stentgate/rkl_loss": rkl_loss_scaled.item(),
            "kd_rkl_stentgate/ce_term": (lambda_ce * ce_loss_mean).item(),
            "kd_rkl_stentgate/total_loss": total_loss.item(),
            "kd_rkl_stentgate/rkl_per_token_mean_all": valid_rkl_all.mean().item() if valid_rkl_all.numel() > 0 else 0.0,
            "kd_rkl_stentgate/rkl_per_token_mean_fired": valid_rkl_fired.mean().item() if valid_rkl_fired.numel() > 0 else 0.0,
            "kd_rkl_stentgate/ce_per_token_mean": valid_ce.mean().item() if valid_ce.numel() > 0 else 0.0,
            "kd_rkl_stentgate/teacher_entropy_mean": teacher_entropy_sum / ent_norm,
            "kd_rkl_stentgate/student_entropy_mean": student_entropy_sum / ent_norm,
            "kd_rkl_stentgate/student_entropy_full_mean": valid_sent.mean().item() if valid_sent.numel() > 0 else 0.0,
            "kd_rkl_stentgate/student_entropy_fired_mean": fired_sent.mean().item() if fired_sent.numel() > 0 else 0.0,
            "kd_rkl_stentgate/student_entropy_unfired_mean": unfired_sent.mean().item() if unfired_sent.numel() > 0 else 0.0,
            "kd_rkl_stentgate/tau": float(tau),
            "kd_rkl_stentgate/fired_frac": fired_frac,
            "kd_rkl_stentgate/gate_quantile": float(gate_quantile),
            "kd_rkl_stentgate/lambda_ce": float(lambda_ce),
            "kd_rkl_stentgate/temperature": float(temperature),
        }

    return total_loss, stats


@dataclass
class TrainArgs:
    name: str = "lingua"
    dump_dir: str = ""

    seed: int = 42

    # Number of gradient accumulation steps
    # Total batch size is batch_size*grad_acc_steps
    grad_acc_steps: int = 1
    # Periodic gradient-geometry probe across source domains (e.g., dclm/math/flan).
    # Disabled when None.
    grad_probe_freq: Optional[int] = None
    # Minimum sequences per domain in a batch to include that domain in the probe.
    grad_probe_min_seqs: int = 2
    # Probe uses only a subset of params for efficiency; cap total elements here.
    grad_probe_max_param_elems: int = 5_000_000
    # Max number of parameter tensors included in the probe subset.
    grad_probe_max_tensors: int = 16
    # Domain name fragments used to bucket source labels.
    grad_probe_domains: List[str] = field(default_factory=lambda: ["dclm", "math", "flan"])

    gc_collect_freq: int = 1000
    probe_freq: Optional[int] = None

    # Nb optimizer steps to take
    steps: int = 1000

    # On-the-fly teacher model for token-level delta weighting.
    # When set, loads a frozen HF model and computes teacher logprobs each step
    # instead of reading pre-cached logprobs from data.
    teacher_model_path: Optional[str] = None

    # Multi-expert online inference: list of HF model paths.
    # When set, runs a no-grad forward pass through each model per batch,
    # takes the per-token min NLL across all experts, and uses that as the
    # reference signal (Δ = student_nll - min_expert_nll).
    # Takes precedence over teacher_model_path if both are set.
    teacher_model_paths: Optional[List[str]] = None

    data: DataArgs = field(default_factory=DataArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: LMTransformerArgs = field(default_factory=LMTransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)

    # Online excess-loss reweighting controller (set enabled=True to activate).
    online_reweighting: OnlineReweightingArgs = field(default_factory=OnlineReweightingArgs)

    # If set to None, eval is run locally otherwise it launches a new job with the given number of gpus
    async_eval_gpus: Optional[int] = None
    eval: Optional[Any] = None
    eval_backend: str = "harness"  # "harness" (lm-eval) or "olmes"


@dataclass
class TrainState(Stateful):
    step: int  # Nb of steps taken by the optimizer
    acc_step: int  # Nb of accumulation steps done since last optimizer step
    scheduler: lr_scheduler.LambdaLR
    data_loader_state: PackTokensState

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "acc_step": self.acc_step,
            "data_loader_state": self.data_loader_state,
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.data_loader_state = PackTokensState(**state_dict["data_loader_state"])
        self.scheduler.load_state_dict(state_dict["scheduler"])


def validate_train_args(args: TrainArgs, output_size: int):
    if args.model.vocab_size < 0:
        logger.info(f"Setting model output size to {output_size}")
        args.model.vocab_size = output_size
    assert (
        args.model.vocab_size == output_size
    ), "Vocab size should be the same as output size"

    assert args.dump_dir, "Dump dir not set"

    if args.checkpoint.path is None:
        logger.info(f"Setting checkpoint path to {str(Path(args.dump_dir) / 'checkpoints')}")
        args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")

    for source in args.data.sources:
        data_path = os.path.join(args.data.root_dir, source)
        assert os.path.exists(data_path), f"{data_path} doesn't exist"

    if (
        args.distributed.dp_replicate
        * args.distributed.dp_shard
        * args.distributed.tp_size
        != get_world_size()
    ):
        assert get_world_size() % args.distributed.dp_shard == 0
        args.distributed.dp_replicate = get_world_size() // args.distributed.dp_shard

        assert args.distributed.dp_replicate % args.distributed.tp_size == 0
        args.distributed.dp_replicate = (
            args.distributed.dp_replicate // args.distributed.tp_size
        )

        logger.warning(
            f"Setting Data Parallel size to {args.distributed.dp_replicate * args.distributed.dp_shard}"
        )
        assert (
            args.distributed.dp_replicate
            * args.distributed.dp_shard
            * args.distributed.tp_size
            == get_world_size()
        )

        if args.distributed.fsdp_type == "no_shard":
            assert (
                args.distributed.dp_shard == 1
                and args.distributed.dp_replicate == get_world_size()
            )

    args.model.max_seqlen = args.data.seq_len

    if args.distributed.tp_size == 1:
        logger.warning(
            "Tensor parallelism has not been tested for a while, use at your own risk"
        )

    assert (
        args.probe_freq != args.profiling.mem_steps
    ), "Don't profile during probe step"
    assert (
        args.probe_freq != args.profiling.profile_steps
    ), "Don't profile during probe step"

    if args.logging.wandb is not None:
        args.logging.wandb.name = args.name

    if args.probe_freq is not None:
        assert (
            args.distributed.tp_size == 1
        ), "Probing not supported with tensor parallelism"
        assert (
            args.distributed.selective_activation_checkpointing is False
        ), "Probing not supported with selective activation checkpointing"


preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption ! checkpointing asap and exiting.")
    preemption_flag["flag"] = True


def every_n_steps(train_state, freq, acc_step=None, acc_freq=None):
    test = train_state.step % freq == 0
    if acc_step is not None:
        test = test and (train_state.acc_step == acc_step)
    elif acc_freq is not None:
        test = test and ((train_state.acc_step % acc_freq) == 0)
    return test


def _parse_source_weight_schedule(
    raw_schedule: Optional[Dict[Any, Dict[str, float]]],
) -> Dict[int, Dict[str, float]]:
    if not raw_schedule:
        return {}

    parsed: Dict[int, Dict[str, float]] = {}
    for step_key, overrides in raw_schedule.items():
        step = int(step_key)
        if step < 0:
            raise ValueError(f"source_weight_schedule step must be >= 0, got {step}")
        if not isinstance(overrides, dict):
            raise ValueError(
                f"source_weight_schedule[{step_key}] must be a mapping of source->weight"
            )
        parsed[step] = {str(k): float(v) for k, v in overrides.items()}
    return dict(sorted(parsed.items(), key=lambda kv: kv[0]))


def _get_loader_sources_dict(data_loader_state: Dict[str, Any]) -> Dict[str, float]:
    # PrefetchState -> PackTokensState -> TokenizerState -> MultiChoiceState
    return data_loader_state["it_state"]["it_state"]["it_state"]["sources"]


def _apply_source_weight_overrides(
    data_loader_state: Dict[str, Any],
    overrides: Dict[str, float],
    step: int,
) -> Dict[str, float]:
    sources = _get_loader_sources_dict(data_loader_state)
    updated = dict(sources)
    updated.update(overrides)

    if any(v < 0 for v in updated.values()):
        raise ValueError(f"source_weight_schedule produced negative weights at step {step}: {updated}")
    if sum(updated.values()) <= 0:
        raise ValueError(f"source_weight_schedule produced zero total weight at step {step}: {updated}")

    # Mutate in-place so the running choose_source iterator sees new values.
    sources.clear()
    sources.update(updated)
    return updated


def _load_source_weights_file(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Source weights file must be a JSON object: {path}")
    weights = {str(k): float(v) for k, v in payload.items()}
    if any(v < 0 for v in weights.values()):
        raise ValueError(f"Negative weights are not allowed in {path}: {weights}")
    if sum(weights.values()) <= 0:
        raise ValueError(f"Sum of weights must be > 0 in {path}: {weights}")
    return weights


def _select_grad_probe_params(
    model: torch.nn.Module,
    max_elems: int,
    max_tensors: int,
) -> List[tuple[str, torch.nn.Parameter]]:
    """Select a representative parameter subset for inexpensive gradient probes."""
    candidates: List[tuple[str, torch.nn.Parameter]] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Skip tiny scalars/vectors that tend to be noisy and uninformative.
        if p.numel() < 1024:
            continue
        candidates.append((name, p))

    if not candidates:
        return []

    # Prefer one middle transformer layer when we can infer layer indices.
    # This keeps probe scope stable and cheap versus sampling across the model.
    layer_candidates: List[tuple[int, str, torch.nn.Parameter]] = []
    layer_pat = re.compile(r"\.(?:layers|blocks|h)\.(\d+)\.")
    for name, p in candidates:
        m = layer_pat.search(name)
        if m is None:
            continue
        layer_candidates.append((int(m.group(1)), name, p))

    if layer_candidates:
        layer_ids = sorted({lid for lid, _, _ in layer_candidates})
        middle_layer = layer_ids[len(layer_ids) // 2]
        middle_params = [
            (name, p) for lid, name, p in layer_candidates if lid == middle_layer
        ]
        # Start with larger tensors from the middle layer for stronger signal.
        middle_params.sort(key=lambda x: x[1].numel(), reverse=True)
        picked_middle: List[tuple[str, torch.nn.Parameter]] = []
        total_middle = 0
        for name, p in middle_params:
            if len(picked_middle) >= max_tensors:
                break
            if total_middle + p.numel() > max_elems and picked_middle:
                break
            picked_middle.append((name, p))
            total_middle += p.numel()
        if picked_middle:
            return picked_middle

    # Prefer broad model coverage by taking evenly-spaced tensors.
    step = max(1, len(candidates) // max(1, max_tensors))
    picked: List[tuple[str, torch.nn.Parameter]] = []
    total = 0
    for i in range(0, len(candidates), step):
        name, p = candidates[i]
        if len(picked) >= max_tensors:
            break
        if total + p.numel() > max_elems and picked:
            break
        picked.append((name, p))
        total += p.numel()

    # Fallback to first candidate if budget is extremely small.
    if not picked:
        picked.append(candidates[0])
    return picked


def _build_domain_index_map(
    source_labels: Optional[List[str]],
    batch_size: int,
    domain_keys: List[str],
) -> Dict[str, List[int]]:
    """Map domain key -> sequence indices using substring matching on source labels."""
    domain_to_indices: Dict[str, List[int]] = {k: [] for k in domain_keys}
    if not source_labels or len(source_labels) != batch_size:
        return domain_to_indices

    for i, raw in enumerate(source_labels):
        label = str(raw).lower()
        for key in domain_keys:
            if key in label:
                domain_to_indices[key].append(i)
                break
    return domain_to_indices


def _grad_probe_stats_for_batch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    source_labels: Optional[List[str]],
    probe_named_params: List[tuple[str, torch.nn.Parameter]],
    domain_keys: List[str],
    min_seqs: int,
) -> Dict[str, float]:
    """
    Compute pairwise gradient cosine similarities across requested domains.
    Uses CE loss on current batch subsets and a parameter subset for efficiency.
    """
    if not probe_named_params:
        return {"grad_probe/available": 0.0, "grad_probe/reason_no_params": 1.0}

    bsz = int(labels.shape[0])
    domain_to_indices = _build_domain_index_map(source_labels, bsz, domain_keys)
    stats: Dict[str, float] = {
        "grad_probe/available": 1.0,
        "grad_probe/domains_requested": float(len(domain_keys)),
        "grad_probe/domains_usable": float(
            sum(1 for k in domain_keys if len(domain_to_indices.get(k, [])) >= min_seqs)
        ),
    }
    for k in domain_keys:
        stats[f"grad_probe/nseq/{k}"] = float(len(domain_to_indices.get(k, [])))

    params = [p for _, p in probe_named_params]
    grads_by_domain: Dict[str, List[Optional[torch.Tensor]]] = {}
    real_domain_grad: Dict[str, bool] = {}
    fallback_idx = list(range(min(max(1, min_seqs), max(1, bsz))))
    if not fallback_idx:
        fallback_idx = [0]

    # IMPORTANT: run the same number of forward/backward-style calls on every rank
    # to keep distributed collectives aligned.
    for dom in domain_keys:
        idxs = domain_to_indices.get(dom, [])
        has_real = len(idxs) >= min_seqs
        real_domain_grad[dom] = has_real
        use_indices = idxs if has_real else fallback_idx
        stats[f"grad_probe/has_real_batch/{dom}"] = 1.0 if has_real else 0.0
        idx = torch.tensor(idxs, device=input_ids.device, dtype=torch.long)
        if not has_real:
            idx = torch.tensor(use_indices, device=input_ids.device, dtype=torch.long)
        dom_loss = model(input_ids[idx], labels[idx])
        grads = torch.autograd.grad(
            dom_loss,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        cached: List[Optional[torch.Tensor]] = []
        sq = 0.0
        for g in grads:
            if g is None:
                cached.append(None)
                continue
            gc = g.detach().float().cpu()
            cached.append(gc)
            sq += float((gc * gc).sum().item())
        grads_by_domain[dom] = cached
        stats[f"grad_probe/grad_norm/{dom}"] = sq**0.5

    # Pairwise cosine on the probe subset.
    doms = list(domain_keys)
    for i in range(len(doms)):
        for j in range(i + 1, len(doms)):
            a, b = doms[i], doms[j]
            dot = 0.0
            n1 = 0.0
            n2 = 0.0
            for ga, gb in zip(grads_by_domain[a], grads_by_domain[b]):
                if ga is None or gb is None:
                    continue
                dot += float((ga * gb).sum().item())
                n1 += float((ga * ga).sum().item())
                n2 += float((gb * gb).sum().item())
            denom = max((n1 * n2) ** 0.5, 1e-12)
            stats[f"grad_probe/cosine/{a}_vs_{b}"] = dot / denom
            stats[f"grad_probe/cosine_valid/{a}_vs_{b}"] = 1.0 if (real_domain_grad[a] and real_domain_grad[b]) else 0.0

    return stats


def train(args: TrainArgs):
    with ExitStack() as context_stack:
        tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
        validate_train_args(
            args,
            tokenizer.n_words,
        )
        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")
        init_logger(Path(args.dump_dir) / "train.log")
        init_signal_handler(set_preemption_flag)  # For handling preemption signals.
        setup_env(args.env)
        setup_torch_distributed(args.distributed)
        world_mesh = get_device_mesh(args.distributed)
        logger.info(f"Starting job: {args.name}")

        # build dataloader
        # need dp world size and rank
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        if args.distributed.dp_shard > 1:
            dp_rank = dp_rank * world_mesh["dp_shard"].size() + world_mesh["dp_shard"].get_local_rank()
            dp_degree *= world_mesh["dp_shard"].size()

        logger.info(f"Running on dp rank : {dp_rank}")
        logger.info(f"Running on dp size : {dp_degree}")

        torch.manual_seed(args.seed)
        logger.info("Building model")

        # Initializing Model in meta device allows us to initialize models much bigger than 1 gpu's memory
        with torch.device("meta"):
            model = LMTransformer(args.model)
        logger.info("Model is built !")

        model_param_count = get_num_params(model)

        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=build_fsdp_grouping_plan(args.model),
            tp_parallelize=tp_parallelize,
            no_recompute_ops=get_no_recompute_ops(),
        )

        # Once we shard the model on different gpus we can actually initialize the model
        # First we create empty tensors of the correct shapes
        model = model.to_empty(device="cuda")
        # Then we init the model. Please make sure this function initializes *ALL* parameters
        # and buffers, otherwise you will have random values in the unitialized tensors
        # which will silently fail (give nan gradients for example)

        if args.checkpoint.init_ckpt_path:
            if "olmo" in args.checkpoint.init_ckpt_path.lower():
                assert args.model.qk_norm, (
                    f"OLMo checkpoint requires qk_norm=true, got {args.model.qk_norm}"
                )
                assert args.model.post_norm, (
                    f"OLMo checkpoint requires post_norm=true, got {args.model.post_norm}"
                )
            logger.info(f"Loading initial model from {args.checkpoint.init_ckpt_path}")
            load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="") # Put model_key="" if its directly the model checkpoint
            model.rope_embeddings.reset_parameters() # For RoPe initialization since it's a buffer it might not be loaded
        else:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.model.seed)
                model.init_weights()
        check_model_value_range(model, range=10.0, std=1.0)

        # log model size

        logger.info(f"Model size: {model_param_count:,} total parameters")

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")

        # Load frozen teacher model(s) for on-the-fly delta weighting.
        # teacher_model_paths (list) takes precedence over teacher_model_path (single).
        def _load_teacher_hf(path: str):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                )
                logger.info(f"Teacher load: using flash_attention_2 for {path}")
            except Exception as e:
                logger.warning(
                    f"Teacher load: flash_attention_2 unavailable for {path} ({type(e).__name__}: {e}); "
                    "falling back to default attention."
                )
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.bfloat16,
                )
            model.config.use_cache = False
            model = model.cuda().eval()

            # Opt-in: replace the layer-stack Linear modules with rowwise-scaled
            # FP8 (e4m3fn) on H200. Teacher forwards run under torch.inference_mode
            # so no backward path is exercised; rowwise FP8 inference quality
            # difference vs bf16 is well below KD signal floor (typical < 0.1pp on
            # task avg for Llama-style models). Skips lm_head and embed_tokens by
            # filter — only the per-layer attention/FFN projections are converted.
            # Driven by env var so in-flight jobs reading this code path on requeue
            # see identical behavior unless LINGUA_TEACHER_FP8=1 is set.
            if os.environ.get("LINGUA_TEACHER_FP8", "0") == "1":
                try:
                    from lingua.float8 import convert_linears_to_fp8
                    fp8_filter = os.environ.get(
                        "LINGUA_TEACHER_FP8_FILTER", r"layers\.[0-9]+\."
                    )
                    n_lin_before = sum(
                        1 for m in model.modules() if isinstance(m, torch.nn.Linear)
                    )
                    model = convert_linears_to_fp8(model, "rowwise", fp8_filter)
                    n_fp8 = sum(
                        1 for m in model.modules() if m.__class__.__name__ == "Fp8Linear"
                    )
                    logger.info(
                        f"Teacher load: FP8 (rowwise) applied to {path}: "
                        f"{n_fp8}/{n_lin_before} Linear modules converted "
                        f"(filter={fp8_filter!r})."
                    )
                except Exception as e:
                    logger.warning(
                        f"Teacher load: FP8 conversion failed for {path} "
                        f"({type(e).__name__}: {e}); falling back to bf16 teacher."
                    )

            # Opt-in: wrap teacher in torch.compile for ~15-30% faster fwd. Off
            # by default so in-flight jobs (which read this code path on requeue)
            # see identical behavior; flip on via env var in the launch script.
            # NOTE: when stacked with FP8, compile happens AFTER FP8 conversion
            # so Inductor sees the Fp8Linear forward path and can fuse around it.
            if os.environ.get("LINGUA_COMPILE_TEACHER", "0") == "1":
                try:
                    model = torch.compile(model, dynamic=False, fullgraph=False)
                    logger.info(f"Teacher load: torch.compile wrapped {path}")
                except Exception as e:
                    logger.warning(
                        f"Teacher load: torch.compile failed for {path} "
                        f"({type(e).__name__}: {e}); using eager teacher."
                    )
            return model

        teacher_model = None
        teacher_models = []  # populated when teacher_model_paths is set
        if args.teacher_model_paths:
            for path in args.teacher_model_paths:
                logger.info(f"Loading frozen expert model: {path}")
                m = _load_teacher_hf(path)
                for p in m.parameters():
                    p.requires_grad_(False)
                teacher_models.append(m)
            total_params = sum(sum(p.numel() for p in m.parameters()) for m in teacher_models)
            logger.info(f"Loaded {len(teacher_models)} expert models ({total_params/1e9:.2f}B params total, frozen)")
            logger.info(f"GPU memory after expert load: {gpu_memory_monitor}")
        elif args.teacher_model_path:
            logger.info(f"Loading frozen teacher model: {args.teacher_model_path}")
            teacher_model = _load_teacher_hf(args.teacher_model_path)
            for p in teacher_model.parameters():
                p.requires_grad_(False)
            teacher_param_count = sum(p.numel() for p in teacher_model.parameters())
            logger.info(f"Teacher model loaded: {teacher_param_count:,} parameters (frozen)")
            logger.info(f"GPU memory after teacher load: {gpu_memory_monitor}")

            # Load EMA self-reference model.
        ema_ref_model = None
        if getattr(args.data, 'use_ema_ref', False) or getattr(args.data, 'use_ema_frontier', False):
                if getattr(args.data, 'ema_frontier_lagged', False):
                    # For lagged swaps, we need a model that can ingest distcp
                    # checkpoints written by the training loop.
                    with torch.device("meta"):
                        ema_ref_model = LMTransformer(args.model)
                    ema_ref_model = ema_ref_model.to_empty(device="cuda")
                    load_from_checkpoint(
                        args.checkpoint.init_ckpt_path, ema_ref_model, model_key=""
                    )
                    ema_ref_model.rope_embeddings.reset_parameters()
                    ema_ref_model = ema_ref_model.eval()
                    for p in ema_ref_model.parameters():
                        p.requires_grad_(False)
                    ema_ref_param_count = sum(p.numel() for p in ema_ref_model.parameters())
                    logger.info(
                        f"EMA ref model initialized from init checkpoint: "
                        f"{ema_ref_param_count:,} parameters (frozen)"
                    )
                else:
                    # Non-lagged mode keeps the previous HF-loading behavior.
                    # The init checkpoint is in Lingua format; the HF-format model
                    # lives in the hf/ subdirectory.
                    ema_ref_path = os.path.join(args.checkpoint.init_ckpt_path, "hf")
                    if not os.path.isdir(ema_ref_path):
                        ema_ref_path = args.checkpoint.init_ckpt_path
                    logger.info(f"Loading EMA self-reference model from: {ema_ref_path}")
                    ema_ref_model = AutoModelForCausalLM.from_pretrained(
                        ema_ref_path,
                        torch_dtype=torch.bfloat16,
                    ).cuda().eval()
                    for p in ema_ref_model.parameters():
                        p.requires_grad_(False)
                    ema_ref_param_count = sum(p.numel() for p in ema_ref_model.parameters())
                    logger.info(
                        f"EMA ref model loaded: {ema_ref_param_count:,} parameters "
                        f"(frozen, capacity-matched)"
                    )
                logger.info(f"GPU memory after EMA ref load: {gpu_memory_monitor}")

        # build optimizer after apply parallelisms to the model
        optimizer, scheduler = build_optimizer(model, args.optim, args.steps)
        data_loader_state = init_dataloader_state_from_args(
            args.data, dp_rank, dp_degree
        )

        train_state = TrainState(
            step=0,
            acc_step=0,
            data_loader_state=data_loader_state,
            scheduler=scheduler,
        )

        checkpoint = CheckpointManager.instantiate_and_make_dir(args.checkpoint)
        checkpoint.load(model, optimizer, train_state, world_mesh)
        source_weight_schedule = _parse_source_weight_schedule(
            getattr(args.data, "source_weight_schedule", None)
        )
        applied_source_weight_steps: set[int] = set()
        first_micro_acc_step = 1 % args.grad_acc_steps
        source_weight_refresh_every = int(getattr(args.checkpoint.dump, "every", 0) or 0)
        source_weights_file = os.path.join(args.dump_dir, "source_weights.json")
        source_weights_file_mtime: Optional[float] = None

        if source_weight_schedule:
            # Catch up to the latest milestone on resume so a restarted job adopts
            # the intended active domain mix immediately.
            resume_target_step = max(
                (s for s in source_weight_schedule if s <= train_state.step),
                default=None,
            )
            if resume_target_step is not None:
                updated = _apply_source_weight_overrides(
                    train_state.data_loader_state,
                    source_weight_schedule[resume_target_step],
                    resume_target_step,
                )
                applied_source_weight_steps.add(resume_target_step)
                if get_is_master():
                    norm = {k: v / sum(updated.values()) for k, v in updated.items()}
                    logger.info(
                        f"Applied source_weight_schedule catch-up at step {resume_target_step}: "
                        f"raw={updated} norm={norm}"
                    )
        if os.path.isfile(source_weights_file):
            try:
                file_weights = _load_source_weights_file(source_weights_file)
                updated = _apply_source_weight_overrides(
                    train_state.data_loader_state,
                    file_weights,
                    train_state.step,
                )
                source_weights_file_mtime = os.path.getmtime(source_weights_file)
                if get_is_master():
                    norm = {k: v / sum(updated.values()) for k, v in updated.items()}
                    logger.info(
                        f"Applied source weights from file at startup ({source_weights_file}): "
                        f"raw={updated} norm={norm}"
                    )
            except Exception as e:
                if get_is_master():
                    logger.warning(
                        f"Failed to apply startup source weights file {source_weights_file}: {e}"
                    )
        # Either load from latest checkpoint or start from scratch
        if args.probe_freq is not None:
            if get_is_master():
                os.makedirs(Path(args.dump_dir) / "probe", exist_ok=True)
            torch.distributed.barrier()
            probe = AutoProbeD(
                model,
                (
                    Path(args.dump_dir) / "probe" / f"probe.{dp_rank}.jsonl"
                    if (dp_rank % 128 == 0)
                    else None
                ),
            )

        grad_probe_named_params: List[tuple[str, torch.nn.Parameter]] = []
        grad_probe_warned_no_labels = False
        pending_grad_probe_stats: Dict[str, float] = {}
        if args.grad_probe_freq is not None:
            grad_probe_named_params = _select_grad_probe_params(
                model,
                max_elems=max(1, int(args.grad_probe_max_param_elems)),
                max_tensors=max(1, int(args.grad_probe_max_tensors)),
            )
            if get_is_master():
                n_elems = sum(p.numel() for _, p in grad_probe_named_params)
                logger.info(
                    f"[GradProbe] enabled freq={args.grad_probe_freq}, "
                    f"params={len(grad_probe_named_params)} tensors, elems={n_elems}"
                )

        gc.disable()

        # train loop
        model.train()
        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )

        # ── dataloader: online reweighting path vs original path ─────────────
        _online_rw = args.online_reweighting
        _rho1_domain_normalize = bool(getattr(args.data, "rho1_domain_normalize", False))
        _rho1_stratified = bool(getattr(args.data, "rho1_stratified_by_domain", False))
        # Selective-KD per-domain normalization / telemetry also needs the
        # source-labeled loader so `doc_sources` reaches the KD loss path.
        # Without this trigger the offline loader runs and `domain_ids` is
        # silently None inside compute_selective_kd_loss → use_domain_norm
        # evaluates False even when the user asked for it. See the
        # 7186923_46 (s3-js-guarded-domnorm) post-mortem.
        _selective_kd_needs_sources = (
            bool(getattr(args.data, "use_selective_kd", False))
            and (
                bool(getattr(args.data, "selective_kd_domain_normalize", False))
                or bool(getattr(args.data, "selective_kd_emit_domain_stats", False))
            )
        )
        _use_source_labeled_loader = (
            _online_rw.enabled
            or _rho1_domain_normalize
            or _rho1_stratified
            or _selective_kd_needs_sources
            or (args.grad_probe_freq is not None)
            or bool(getattr(args.data, "use_oracle_source_routing", False))
            or bool(getattr(args.data, "use_rkl_with_source_ce_distillation", False))
        )
        if _use_source_labeled_loader:
            # ReweightableDataLoader manages its own subprocess lifecycle.
            online_loader = ReweightableDataLoader(args.data, train_state.data_loader_state)
            context_stack.callback(online_loader.close)
            data_loader = None  # not used in online path

            if _online_rw.enabled:
                # Controller accumulates per-source stats and emits new weights.
                _base_weights = dict(
                    train_state.data_loader_state["it_state"]["it_state"]["it_state"]["sources"]
                )
                online_ctrl = ExcessLossController(_base_weights, _online_rw)
                _ctrl_state_path = os.path.join(args.dump_dir, "online_ctrl_state.json")
                if os.path.exists(_ctrl_state_path):
                    import json as _json
                    with open(_ctrl_state_path) as _f:
                        _saved_ctrl = _json.load(_f)
                    online_ctrl.load_state_dict(_saved_ctrl)
                    if get_is_master():
                        logger.info(
                            f"[OnlineReweighting] restored controller state from step {_saved_ctrl.get('step', '?')}"
                        )
                if get_is_master():
                    logger.info(
                        f"[OnlineReweighting] enabled — update_interval={_online_rw.update_interval} "
                        f"eta={_online_rw.eta} ema_beta={_online_rw.ema_beta} "
                        f"floor={_online_rw.floor_mul}x ceil={_online_rw.ceil_mul}x"
                    )
                    logger.info(f"[OnlineReweighting] base weights: {_base_weights}")
            else:
                online_ctrl = None
                if get_is_master():
                    logger.info("[Rho1] Using source-labeled dataloader for domain-aware token selection")
        else:
            online_loader = None
            online_ctrl = None
            data_loader = context_stack.enter_context(
                build_dataloader_from_args(
                    args.data,
                    state=train_state.data_loader_state,
                )
            )
        # ─────────────────────────────────────────────────────────────────────
        torch_profiler = context_stack.enter_context(
            maybe_run_profiler(args.dump_dir, model, args.profiling)
        )

        nwords_since_last_log = 0
        time_last_log = timer()
        gc.collect()
        saved = False
        routing_buffer_cache = {
            "padded": None,
            "labels": None,
            "valid_mask": None,
            "capacity_docs": 0,
            "capacity_len": 0,
        }


        # def ids_for(txt: str):
        #     # Encode without BOS/EOS so we get exactly the token pieces of the string.
        #     return tokenizer.encode(txt, add_bos=False, add_eos=False)

        # IGNORE_IDS = set()
        # if args.data.add_special_tokens:
        #     for s in (SpecialTokens.FACTUAL_TOKEN, SpecialTokens.NONFACTUAL_TOKEN, SpecialTokens.PARTIAL_FACTUAL_TOKEN):
        #         IGNORE_IDS.update(ids_for(s.value))
        #         assert len(ids_for(s.value)) == 1, f"{s.value} splits into {ids_for(s.value)}; register as a single special token!"
        # IGNORE_IDS_T = torch.tensor(list(IGNORE_IDS), dtype=torch.long)

        # attn_impl = "flex_attention" if args.data.mask_cross_doc_loss else "sdpa"
        # logger.info(f"Attn implementation: {attn_impl}")
        while train_state.step < args.steps:
            # We constrain train_state.acc_step to be in range 0 to args.grad_acc_steps - 1
            train_state.acc_step = (train_state.acc_step + 1) % args.grad_acc_steps

            # Optional runtime domain/source reweighting (no-op when schedule is empty).
            if (
                source_weight_schedule
                and train_state.acc_step == first_micro_acc_step
                and train_state.step in source_weight_schedule
                and train_state.step not in applied_source_weight_steps
            ):
                updated = _apply_source_weight_overrides(
                    train_state.data_loader_state,
                    source_weight_schedule[train_state.step],
                    train_state.step,
                )
                applied_source_weight_steps.add(train_state.step)
                if get_is_master():
                    norm = {k: v / sum(updated.values()) for k, v in updated.items()}
                    logger.info(
                        f"Applied source_weight_schedule at step {train_state.step}: "
                        f"raw={updated} norm={norm}"
                    )
            # Config-free periodic source reweighting:
            # If dump_dir/source_weights.json exists, refresh it at fixed intervals.
            # This allows updating domain weights mid-run without editing YAML.
            if (
                source_weight_refresh_every > 0
                and train_state.acc_step == first_micro_acc_step
                and train_state.step > 0
                and (train_state.step % source_weight_refresh_every == 0)
                and os.path.isfile(source_weights_file)
            ):
                try:
                    curr_mtime = os.path.getmtime(source_weights_file)
                    if source_weights_file_mtime is None or curr_mtime != source_weights_file_mtime:
                        file_weights = _load_source_weights_file(source_weights_file)
                        updated = _apply_source_weight_overrides(
                            train_state.data_loader_state,
                            file_weights,
                            train_state.step,
                        )
                        source_weights_file_mtime = curr_mtime
                        if get_is_master():
                            norm = {k: v / sum(updated.values()) for k, v in updated.items()}
                            logger.info(
                                f"Applied source weights from file at step {train_state.step} "
                                f"({source_weights_file}): raw={updated} norm={norm}"
                            )
                except Exception as e:
                    if get_is_master():
                        logger.warning(
                            f"Failed to refresh source weights from {source_weights_file} "
                            f"at step {train_state.step}: {e}"
                        )

            # get batch
            curr_lr = float(optimizer.param_groups[0]["lr"])
            data_load_start = timer()
            if online_loader is not None:
                batch, train_state.data_loader_state, _source_labels = next(online_loader)
            else:
                batch, train_state.data_loader_state = next(data_loader)
                _source_labels = None

            # Handle new dict format with tokens and cu_seqlens
            if isinstance(batch, dict):
                batch_tokens = batch['tokens']
                batch_cu_seqlens = batch.get('cu_seqlens', None)  # List of cu_seqlens per batch item
                batch_doc_sources = batch.get('doc_sources', None)  # List of per-doc source lists
            else:
                batch_tokens = batch
                batch_cu_seqlens = None
                batch_doc_sources = None

            # Avoid an unconditional copy when the loader already returns a tensor.
            batch_tokens = torch.as_tensor(batch_tokens, dtype=torch.long)

            # JH ADD - Support both legacy ndarray payload and dict payload with doc_ids
            # if isinstance(batch, dict):
            #     np_tokens = batch["tokens"]                      # (B, S, V)
            #     np_docids = batch.get("doc_ids", None)          # (B, S, V) or None
            # else:
            #     np_tokens = batch
            #     np_docids = None

            # tokens_t = torch.tensor(np_tokens, dtype=torch.long)
            # input_ids = tokens_t[:, :, 0].cuda(non_blocking=True)
            # labels    = tokens_t[:, :, 1].cuda(non_blocking=True)

            # doc_ids shape must be [B, S] aligned with the *input* view (V=0)
            # doc_ids_t = None
            # if np_docids is not None:
            #     doc_ids_t = torch.tensor(np_docids[:, :, 0], dtype=torch.int32).cuda(non_blocking=True)

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                logger.info("garbage collection")
                # we do garbage collection manually otherwise different processes
                # run the GC at different times so they slow down the whole pipeline
                gc.collect()

            # Extract input_ids and labels (views 0 and 1)
            input_ids = batch_tokens[:, :, 0].to(device="cuda", non_blocking=True)
            labels = batch_tokens[:, :, 1].to(device="cuda", non_blocking=True)

            # Prepare cu_seqlens for FA2 varlen if cross-doc attention masking is enabled
            cu_seqlens_tensor = None
            max_seqlen = None
            if getattr(args.data, 'disable_cross_doc_attn', False) and batch_cu_seqlens is not None:
                # Convert per-batch cu_seqlens to a single flattened tensor for FA2 varlen
                # FA2 varlen expects cu_seqlens to mark document boundaries across the flattened batch
                # For batch_size B and seq_len S, we have B*S total tokens
                # Each item in batch_cu_seqlens is a list like [0, doc1_end, doc2_end, ..., S]

                bsz = batch_tokens.shape[0]
                seq_len = batch_tokens.shape[1]

                # Build global cu_seqlens: offset each batch item's boundaries by batch_idx * seq_len
                global_cu_seqlens = [0]
                max_doc_len = 0
                for batch_idx, cu_seqs in enumerate(batch_cu_seqlens):
                    offset = batch_idx * seq_len
                    # Add all boundaries except the first (0) since we already have 0 or previous end
                    for i, pos in enumerate(cu_seqs):
                        if i == 0:
                            continue  # Skip the leading 0
                        global_pos = offset + pos
                        global_cu_seqlens.append(global_pos)

                        # Track max document length
                        prev_pos = cu_seqs[i-1]
                        doc_len = pos - prev_pos
                        max_doc_len = max(max_doc_len, doc_len)

                cu_seqlens_tensor = torch.tensor(global_cu_seqlens, dtype=torch.int32, device='cuda')
                max_seqlen = max_doc_len

            # Extract precomputed teacher signals from data batch views (if present).
            # View layout when precomputed signals are available:
            #   view 2: teacher NLL      (always present when use_teacher_logprobs=True)
            #   view 3: teacher entropy  (present when teacher_entropy_field is set)
            #   view 4: teacher margin   (present when teacher_margin_field is set)
            teacher_logprobs = None
            teacher_entropy_precomputed = None
            teacher_margin_precomputed = None
            # Always initialize per-step routing stats; some training paths
            # (e.g., no teacher routing) do not populate this later.
            expert_routing_stats = {}
            n_batch_views = batch_tokens.shape[2]
            if args.data.use_teacher_logprobs and n_batch_views > 2:
                teacher_logprobs = batch_tokens[:, :, 2].to(device="cuda", dtype=torch.float32, non_blocking=True)
                if n_batch_views > 3:
                    teacher_entropy_precomputed = batch_tokens[:, :, 3].to(device="cuda", dtype=torch.float32, non_blocking=True)
                if n_batch_views > 4:
                    teacher_margin_precomputed = batch_tokens[:, :, 4].to(device="cuda", dtype=torch.float32, non_blocking=True)

            # On-the-fly teacher: only run if teacher model loaded AND signals not already precomputed.
            # Methods using full logits (KL distillation, entropy-gated KD) always need on-the-fly.
            needs_kl = (
                (hasattr(args.data, 'use_kl_distillation') and args.data.use_kl_distillation) or
                (hasattr(args.data, 'use_entropy_gated_kd') and args.data.use_entropy_gated_kd) or
                (hasattr(args.data, 'use_selective_kd') and args.data.use_selective_kd)
            )
            teacher_logits_for_eam = None  # Store teacher logits for methods needing full logits
            best_expert_logits = None       # Store winning expert's full logits for seq-level KD
            all_expert_nlls = None          # (E, B, T) per-expert NLLs for expert-stratified selection
            bucket_kd_main_logits = None    # (B, T, V) main-teacher full logits for bucket-aware KD
            bucket_kd_gate_logits = None    # (B, T, V) gate-teacher full logits for bucket-aware KD

            if teacher_models and teacher_logprobs is None:
                # Multi-expert: run each expert, then route per-token or per-sequence.
                _use_seq_kd      = getattr(args.data, 'use_best_expert_seq_kd', False)
                _use_seq_rho1    = getattr(args.data, 'use_best_expert_seq_rho1', False)
                _use_oracle_routing = getattr(args.data, 'use_oracle_source_routing', False)
                _use_seq_routing = getattr(args.data, 'use_best_expert_seq', False) or _use_seq_kd or _use_seq_rho1 or _use_oracle_routing
                if _use_seq_routing:
                    assert batch_cu_seqlens is not None, \
                        "use_best_expert_seq / use_best_expert_seq_kd requires batch_cu_seqlens to avoid crossing doc boundaries"
                expert_routing_stats = {}
                routing_debug_line = None
                _nll_chunk_size = int(getattr(args.data, "teacher_nll_chunk_size", 131072))

                def _chunked_token_nll(
                    logits_3d: torch.Tensor,
                    targets_2d: torch.Tensor,
                    ignore_index: int = -100,
                ) -> torch.Tensor:
                    flat_logits = logits_3d.reshape(-1, logits_3d.size(-1))
                    flat_targets = targets_2d.reshape(-1)
                    n_tokens = flat_targets.numel()
                    nll_flat = torch.empty(
                        n_tokens, device=flat_logits.device, dtype=torch.float32
                    )
                    for start in range(0, n_tokens, _nll_chunk_size):
                        end = min(start + _nll_chunk_size, n_tokens)
                        nll_flat[start:end] = F.cross_entropy(
                            flat_logits[start:end],
                            flat_targets[start:end],
                            reduction="none",
                            ignore_index=ignore_index,
                        ).to(torch.float32)
                    return nll_flat.view_as(targets_2d)

                def _ensure_routing_buffers(required_docs: int, required_len: int) -> None:
                    cap_docs = routing_buffer_cache["capacity_docs"]
                    cap_len = routing_buffer_cache["capacity_len"]
                    needs_realloc = (
                        routing_buffer_cache["padded"] is None
                        or required_docs > cap_docs
                        or required_len > cap_len
                        or routing_buffer_cache["padded"].device != input_ids.device
                        or routing_buffer_cache["padded"].dtype != input_ids.dtype
                    )
                    if needs_realloc:
                        new_cap_docs = max(required_docs, max(cap_docs * 2, 1))
                        new_cap_len = max(required_len, max(cap_len * 2, 1))
                        routing_buffer_cache["padded"] = torch.empty(
                            (new_cap_docs, new_cap_len),
                            dtype=input_ids.dtype,
                            device=input_ids.device,
                        )
                        routing_buffer_cache["labels"] = torch.empty(
                            (new_cap_docs, new_cap_len),
                            dtype=torch.long,
                            device=input_ids.device,
                        )
                        routing_buffer_cache["valid_mask"] = torch.empty(
                            (new_cap_docs, new_cap_len),
                            dtype=torch.bool,
                            device=input_ids.device,
                        )
                        routing_buffer_cache["capacity_docs"] = new_cap_docs
                        routing_buffer_cache["capacity_len"] = new_cap_len

                with torch.inference_mode():
                    if _use_seq_routing and batch_cu_seqlens is not None:
                        # Document-level routing with no cross-doc attention.
                        # Batch ALL documents across the whole batch into one padded
                        # tensor per expert (3 forward passes total instead of ~240),
                        # then unpack results — keeps GPU well-utilised.
                        B, T = labels.shape
                        E = len(teacher_models)
                        teacher_logprobs = torch.zeros(B, T, device=input_ids.device, dtype=torch.float32)
                        expert_wins = torch.zeros(E, dtype=torch.float32)

                        # Collect full-doc metadata and slices in one Python pass.
                        doc_meta = []   # (b, doc_start, doc_end) for each doc
                        doc_inputs_full = []  # raw token slices, variable length
                        doc_labels_full = []
                        doc_sources_flat = []  # per-doc source name (for oracle routing)
                        for b, cu_seqs in enumerate(batch_cu_seqlens):
                            for d in range(len(cu_seqs) - 1):
                                doc_start = cu_seqs[d]
                                doc_end   = cu_seqs[d + 1]
                                doc_meta.append((b, doc_start, doc_end))
                                doc_inputs_full.append(input_ids[b, doc_start:doc_end])
                                doc_labels_full.append(labels[b, doc_start:doc_end])
                                if batch_doc_sources is not None and b < len(batch_doc_sources) and batch_doc_sources[b] is not None and d < len(batch_doc_sources[b]):
                                    doc_sources_flat.append(batch_doc_sources[b][d])
                                elif _source_labels is not None and b < len(_source_labels):
                                    doc_sources_flat.append(_source_labels[b])
                                else:
                                    doc_sources_flat.append("unknown")

                        n_docs = len(doc_meta)
                        pad_id = 0

                        if _use_oracle_routing:
                            # Oracle routing: assign each document to an expert based on
                            # its data source label rather than running a prefix NLL pass.
                            # Expert 0 = math RLVR expert (math sources)
                            # Expert 1 = instruction expert (all other sources)
                            _math_sources = set(getattr(args.data, 'oracle_math_sources', ['math_shuffled']))
                            best_e_per_doc = torch.tensor(
                                [0 if src in _math_sources else 1 for src in doc_sources_flat],
                                dtype=torch.long, device=input_ids.device,
                            )
                            expert_wins = torch.bincount(best_e_per_doc, minlength=E).to(torch.float32)
                        else:
                            # One routing pass per expert over prefix chunks grouped by
                            # similar lengths. This keeps routing exact while reducing
                            # padding waste from a single global max length.
                            _EXPERT_SUB_BSZ = max(
                                1, int(getattr(args.data, "routing_expert_sub_bsz", 32))
                            )
                            routing_doc_indices = list(range(n_docs))
                            routing_doc_indices.sort(
                                key=lambda doc_idx: routing_inputs[doc_idx].shape[0],
                                reverse=True,
                            )
                            doc_mean_nlls = torch.empty(
                                E, n_docs, device=input_ids.device, dtype=torch.float32
                            )
                            for expert_idx, m in enumerate(teacher_models):
                                for start in range(0, n_docs, _EXPERT_SUB_BSZ):
                                    end = min(start + _EXPERT_SUB_BSZ, n_docs)
                                    chunk_doc_indices = routing_doc_indices[start:end]
                                    chunk_size = len(chunk_doc_indices)
                                    chunk_inputs = [routing_inputs[i] for i in chunk_doc_indices]
                                    chunk_labels = [routing_labels[i] for i in chunk_doc_indices]
                                    max_len_chunk = max(tok.shape[0] for tok in chunk_inputs)

                                    _ensure_routing_buffers(chunk_size, max_len_chunk)
                                    padded_chunk = routing_buffer_cache["padded"][:chunk_size, :max_len_chunk]
                                    lbl_chunk = routing_buffer_cache["labels"][:chunk_size, :max_len_chunk]
                                    valid_chunk = routing_buffer_cache["valid_mask"][:chunk_size, :max_len_chunk]

                                    padded_chunk.fill_(pad_id)
                                    lbl_chunk.fill_(-100)
                                    for j, (tok, lbl) in enumerate(zip(chunk_inputs, chunk_labels)):
                                        dl = tok.shape[0]
                                        padded_chunk[j, :dl] = tok
                                        lbl_chunk[j, :dl] = lbl

                                    out = m(padded_chunk, use_cache=False)
                                    chunk_nll = _chunked_token_nll(
                                        logits_3d=out.logits,
                                        targets_2d=lbl_chunk,
                                        ignore_index=-100,
                                    )
                                    del out

                                    torch.ne(lbl_chunk, -100, out=valid_chunk)
                                    valid_count = valid_chunk.sum(dim=-1).clamp(min=1).to(chunk_nll.dtype)
                                    mean_nll = chunk_nll.masked_fill(~valid_chunk, 0.0).sum(dim=-1) / valid_count
                                    chunk_idx_tensor = torch.as_tensor(
                                        chunk_doc_indices, device=input_ids.device, dtype=torch.long
                                    )
                                    doc_mean_nlls[expert_idx, chunk_idx_tensor] = mean_nll

                            best_e_per_doc = doc_mean_nlls.argmin(dim=0)  # (n_docs,)
                            expert_wins = torch.bincount(best_e_per_doc, minlength=E).to(torch.float32)

                        # Full-NLL pass uses full doc lengths — keep sub-bsz small to
                        # avoid OOM on long DCLM docs.
                        _FULL_NLL_SUB_BSZ = max(
                            1, int(getattr(args.data, "routing_full_nll_sub_bsz", 8))
                        )

                        # Compute teacher NLL on full docs for the chosen expert only.
                        # This keeps the "route by prefix" speedup while avoiding
                        # proxy targets on the document tail.
                        # When _use_seq_kd, also keep the winning expert's full logits.
                        best_expert_logits_by_doc = {} if _use_seq_kd else None
                        for expert_idx, m in enumerate(teacher_models):
                            assigned = (best_e_per_doc == expert_idx).nonzero(as_tuple=False).flatten()
                            if assigned.numel() == 0:
                                continue

                            assigned_list = assigned.tolist()
                            # Sort by length so each microbatch has tighter padding.
                            assigned_list.sort(
                                key=lambda doc_idx: doc_inputs_full[doc_idx].shape[0],
                                reverse=True,
                            )
                            n_assigned = len(assigned_list)
                            for start in range(0, n_assigned, _FULL_NLL_SUB_BSZ):
                                end = min(start + _FULL_NLL_SUB_BSZ, n_assigned)
                                chunk_doc_indices = assigned_list[start:end]
                                chunk_size = len(chunk_doc_indices)
                                chunk_inputs = [doc_inputs_full[i] for i in chunk_doc_indices]
                                chunk_labels = [doc_labels_full[i] for i in chunk_doc_indices]
                                max_len_chunk = max(tok.shape[0] for tok in chunk_inputs)

                                _ensure_routing_buffers(chunk_size, max_len_chunk)

                                padded_chunk = routing_buffer_cache["padded"][:chunk_size, :max_len_chunk]
                                lbl_chunk = routing_buffer_cache["labels"][:chunk_size, :max_len_chunk]
                                padded_chunk.fill_(pad_id)
                                lbl_chunk.fill_(-100)
                                for j, (tok, lbl) in enumerate(zip(chunk_inputs, chunk_labels)):
                                    dl = tok.shape[0]
                                    padded_chunk[j, :dl] = tok
                                    lbl_chunk[j, :dl] = lbl

                                out = m(padded_chunk, use_cache=False)
                                chunk_nll = _chunked_token_nll(
                                    logits_3d=out.logits,
                                    targets_2d=lbl_chunk,
                                    ignore_index=-100,
                                )
                                if _use_seq_kd:
                                    # Store the winning expert's logits for each doc in
                                    # this chunk; only valid (non-padded) positions are kept.
                                    for j, global_doc_idx in enumerate(chunk_doc_indices):
                                        b_j, ds, de = doc_meta[global_doc_idx]
                                        dl = de - ds
                                        best_expert_logits_by_doc[global_doc_idx] = (
                                            out.logits[j, :dl].clone()
                                        )
                                del out

                                # Scatter routed full-doc targets back to [B, T].
                                for j, global_doc_idx in enumerate(chunk_doc_indices):
                                    b, doc_start, doc_end = doc_meta[global_doc_idx]
                                    dl = doc_end - doc_start
                                    doc_nll = chunk_nll[j, :dl]
                                    doc_nll = doc_nll.masked_fill(
                                        doc_labels_full[global_doc_idx] == -100, 0.0
                                    )
                                    teacher_logprobs[b, doc_start:doc_end] = doc_nll

                        # Assemble best_expert_logits (B, T, V) from per-doc slices.
                        if _use_seq_kd and best_expert_logits_by_doc:
                            _sample = next(iter(best_expert_logits_by_doc.values()))
                            best_expert_logits = torch.zeros(
                                B, T, _sample.size(-1),
                                device=input_ids.device,
                                dtype=_sample.dtype,
                            )
                            for global_doc_idx, doc_logits in best_expert_logits_by_doc.items():
                                b_j, ds, de = doc_meta[global_doc_idx]
                                best_expert_logits[b_j, ds:de] = doc_logits
                            best_expert_logits_by_doc.clear()

                        expert_routing_stats = {
                            f"best_expert/routing_frac_expert_{i}": (expert_wins[i] / max(n_docs, 1)).item()
                            for i in range(E)
                        }

                        # Per-source routing fractions: routing_fraction/{source}/{expert}
                        # Uses _source_labels[b] as the source for each document in batch item b.
                        if _source_labels is not None:
                            # IMPORTANT: Keep key set identical across ranks.
                            # dist_mean_dict() performs one all_reduce per key, so
                            # rank-dependent source presence must be zero-filled.
                            _known_sources = list(
                                _get_loader_sources_dict(train_state.data_loader_state).keys()
                            )
                            _src_expert_counts: Dict[str, List[int]] = {
                                _src: [0] * E for _src in _known_sources
                            }
                            _src_expert_counts["unknown"] = [0] * E

                            for _doc_idx, (_b, _, _) in enumerate(doc_meta):
                                _src = _source_labels[_b] if _b < len(_source_labels) else "unknown"
                                # Never introduce new metric keys from per-rank data.
                                # Any unexpected source is folded into "unknown" so the
                                # key set remains identical across all ranks.
                                _src_key = _src if _src in _src_expert_counts else "unknown"
                                _src_expert_counts[_src_key][int(best_e_per_doc[_doc_idx].item())] += 1

                            for _src in sorted(_src_expert_counts.keys()):
                                _counts = _src_expert_counts[_src]
                                _total = sum(_counts)
                                _tag = _src.replace("_shuffled", "")
                                for _ei, _count in enumerate(_counts):
                                    expert_routing_stats[f"routing_fraction/{_tag}/expert_{_ei}"] = (
                                        _count / max(_total, 1)
                                    )

                        fracs_str = "/".join(
                            f"e{i}={expert_wins[i].item():.0f}({(expert_wins[i]/max(n_docs,1)).item():.2f})"
                            for i in range(E)
                        )
                        routing_debug_line = f"[routing] step={train_state.step} mode=doc n_docs={n_docs} max_doc_len={max_doc_len} {fracs_str}"

                    else:
                        # Full packed-sequence forward pass (token-level routing or
                        # fallback sequence-level routing when cu_seqlens unavailable).
                        all_nlls = []
                        _hybrid_kd_enabled = bool(
                            getattr(args.data, "use_hybrid_residual_kd", False)
                        )
                        _2tgkd_enabled = bool(
                            getattr(args.data, "use_two_teacher_geometric_kd", False)
                        )
                        _p2tkd_enabled = bool(
                            getattr(args.data, "use_projected_two_teacher_kd", False)
                        )
                        _a2tkd_enabled = bool(
                            getattr(args.data, "use_agreement_gated_two_teacher_kd", False)
                        )
                        _crt2tkd_enabled = bool(
                            getattr(args.data, "use_competence_routed_two_teacher_kd", False)
                        )
                        _ipkd_enabled = bool(
                            getattr(args.data, "use_intersection_projected_kd", False)
                        )
                        _bakd_enabled = bool(
                            getattr(args.data, "use_bucket_aware_kd", False)
                        ) or _hybrid_kd_enabled or _2tgkd_enabled or _p2tkd_enabled or _a2tkd_enabled or _crt2tkd_enabled or _ipkd_enabled
                        # BAKD, Hybrid-Residual KD, Two-Teacher Geometric KD,
                        # Projected Two-Teacher KD, Agreement-Gated Two-Teacher
                        # KD, Competence-Routed Two-Teacher KD, and
                        # Intersection-Projected KD all need *both* teachers'
                        # full logits captured during the multi-teacher fwd loop.
                        # Priority of index ownership: intersection-projected,
                        # competence-routed-2t, agreement-gated-2t, projected-2t,
                        # two-teacher-geomkd, hybrid-KD, BAKD as fallback.
                        if _ipkd_enabled:
                            _bakd_main_idx = int(
                                getattr(args.data, "intersection_projected_kd_main_idx", 1)
                            )
                            _bakd_gate_idx = int(
                                getattr(args.data, "intersection_projected_kd_gate_idx", 0)
                            )
                        elif _crt2tkd_enabled:
                            _bakd_main_idx = int(
                                getattr(args.data, "competence_routed_two_teacher_kd_main_idx", 1)
                            )
                            _bakd_gate_idx = int(
                                getattr(args.data, "competence_routed_two_teacher_kd_gate_idx", 0)
                            )
                        elif _a2tkd_enabled:
                            _bakd_main_idx = int(
                                getattr(args.data, "agreement_gated_two_teacher_kd_main_idx", 1)
                            )
                            _bakd_gate_idx = int(
                                getattr(args.data, "agreement_gated_two_teacher_kd_gate_idx", 0)
                            )
                        elif _p2tkd_enabled:
                            _bakd_main_idx = int(
                                getattr(args.data, "projected_two_teacher_kd_main_idx", 1)
                            )
                            _bakd_gate_idx = int(
                                getattr(args.data, "projected_two_teacher_kd_gate_idx", 0)
                            )
                        elif _2tgkd_enabled:
                            _bakd_main_idx = int(
                                getattr(args.data, "two_teacher_geometric_kd_main_idx", 1)
                            )
                            _bakd_gate_idx = int(
                                getattr(args.data, "two_teacher_geometric_kd_gate_idx", 0)
                            )
                        elif _hybrid_kd_enabled:
                            _bakd_main_idx = int(
                                getattr(args.data, "hybrid_residual_kd_main_idx", 1)
                            )
                            _bakd_gate_idx = int(
                                getattr(args.data, "hybrid_residual_kd_gate_idx", 0)
                            )
                        else:
                            _bakd_main_idx = int(
                                getattr(args.data, "bucket_aware_kd_main_idx", 1)
                            )
                            _bakd_gate_idx = int(
                                getattr(args.data, "bucket_aware_kd_gate_idx", 0)
                            )
                        for _ti, m in enumerate(teacher_models):
                            out = m(input_ids, use_cache=False)
                            nll = _chunked_token_nll(
                                logits_3d=out.logits,
                                targets_2d=labels,
                                ignore_index=-100,
                            )
                            nll[labels == -100] = float("inf")
                            all_nlls.append(nll)
                            if _bakd_enabled and _ti == _bakd_main_idx:
                                bucket_kd_main_logits = out.logits.detach()
                            if _bakd_enabled and _ti == _bakd_gate_idx:
                                bucket_kd_gate_logits = out.logits.detach()
                            del out
                        stacked = torch.stack(all_nlls, dim=0)  # (E, B, T)

                        if _use_seq_routing:
                            # Fallback: no cu_seqlens, route at packed-sequence level
                            valid_mask  = (labels != -100).float()
                            valid_count = valid_mask.sum(dim=-1).clamp(min=1)
                            seq_sum = stacked.masked_fill(valid_mask.unsqueeze(0) == 0, 0.0).sum(dim=-1)
                            seq_avg = seq_sum / valid_count.unsqueeze(0)
                            best_idx = seq_avg.argmin(dim=0)
                            best_idx_exp = best_idx.unsqueeze(0).unsqueeze(-1).expand(
                                1, stacked.size(1), stacked.size(2))
                            min_nll = stacked.gather(0, best_idx_exp).squeeze(0)
                            n_experts = stacked.size(0)
                            expert_routing_stats = {
                                f"best_expert/routing_frac_expert_{i}": (best_idx == i).float().mean().item()
                                for i in range(n_experts)
                            }
                            fracs_str = "/".join(
                                f"e{i}={expert_routing_stats.get(f'best_expert/routing_frac_expert_{i}', 0.0):.2f}"
                                for i in range(n_experts)
                            )
                            routing_debug_line = (
                                f"[routing] step={train_state.step} mode=packed_seq "
                                f"n_seq={best_idx.numel()} {fracs_str}"
                            )
                        elif getattr(args.data, 'rho1_round_robin_experts', False):
                            E = stacked.size(0)
                            expert_idx = train_state.step % E
                            min_nll = stacked[expert_idx]
                            expert_routing_stats = {
                                "round_robin/active_expert": float(expert_idx),
                            }
                            routing_debug_line = (
                                f"[routing] step={train_state.step} mode=round_robin "
                                f"active_expert={expert_idx}/{E}"
                            )
                        elif getattr(args.data, 'rho1_stage_boundaries', None):
                            E = stacked.size(0)
                            boundaries = list(args.data.rho1_stage_boundaries)
                            assert len(boundaries) == E - 1, (
                                f"rho1_stage_boundaries must have len(teacher_model_paths)-1={E-1} "
                                f"entries, got {len(boundaries)} ({boundaries})."
                            )
                            assert all(boundaries[i] < boundaries[i + 1] for i in range(len(boundaries) - 1)) and (
                                len(boundaries) == 0 or boundaries[0] > 0
                            ), f"rho1_stage_boundaries must be strictly increasing positive ints, got {boundaries}"
                            step = train_state.step
                            expert_idx = sum(1 for b in boundaries if step >= b)
                            expert_idx = min(expert_idx, E - 1)
                            min_nll = stacked[expert_idx]
                            expert_routing_stats = {
                                "staged/active_expert": float(expert_idx),
                            }
                            routing_debug_line = (
                                f"[routing] step={step} mode=staged "
                                f"active_expert={expert_idx}/{E} boundaries={boundaries}"
                            )
                        elif getattr(args.data, 'rho1_capgap_enable', False):
                            # Capacity-gap penalty mode: rank by main-expert excess but penalize
                            # tokens where the gate expert is also far behind the main one.
                            # teacher_logprobs becomes the main expert's NLL so excess_loss in
                            # compute_rho1_loss naturally reads as (L_student - L_main); the
                            # penalty itself is applied by compute_rho1_loss using all_expert_nlls.
                            E = stacked.size(0)
                            _main_idx = int(getattr(args.data, 'rho1_capgap_main_expert_idx', 1))
                            _gate_idx = int(getattr(args.data, 'rho1_capgap_gate_expert_idx', 0))
                            if not (0 <= _main_idx < E and 0 <= _gate_idx < E):
                                raise RuntimeError(
                                    f"rho1_capgap: main_idx={_main_idx} or gate_idx={_gate_idx} "
                                    f"out of range for E={E} experts in teacher_model_paths."
                                )
                            if _main_idx == _gate_idx:
                                raise RuntimeError(
                                    "rho1_capgap: main_idx and gate_idx must differ "
                                    "(main = strong-capacity ref, gate = same-capacity ref)."
                                )
                            min_nll = stacked[_main_idx]
                            _lambda = float(getattr(args.data, 'rho1_capgap_lambda', 0.5))
                            _hard_gate = bool(getattr(args.data, 'rho1_capgap_hard_x_gate', False))
                            _tau_x = float(getattr(args.data, 'rho1_capgap_hard_x_gate_margin', 0.0))
                            expert_routing_stats = {
                                "capgap/main_idx": float(_main_idx),
                                "capgap/gate_idx": float(_gate_idx),
                                "capgap/lambda": _lambda,
                                "capgap/hard_x_gate": 1.0 if _hard_gate else 0.0,
                                "capgap/hard_x_gate_margin": _tau_x,
                            }
                            routing_debug_line = (
                                f"[routing] step={train_state.step} mode=capgap "
                                f"main={_main_idx} gate={_gate_idx} lambda={_lambda:.3f}"
                                + (f" hard_x_gate>tau={_tau_x:.3f}" if _hard_gate else "")
                            )
                        else:
                            # Per-token reduction across experts (default=min).
                            _reduce_mode = str(getattr(args.data, "rho1_expert_reduce", "min")).lower()
                            if _reduce_mode in ("avg", "mean"):
                                min_nll = stacked.mean(dim=0)
                                _reduce_mode = "avg"
                            elif _reduce_mode == "max":
                                min_nll = stacked.max(dim=0).values
                            else:
                                min_nll = stacked.min(dim=0).values
                                _reduce_mode = "min"
                            expert_routing_stats = {}

                        min_nll[labels == -100] = 0.0
                        teacher_logprobs = min_nll

                        if (
                            getattr(args.data, 'rho1_expert_stratified', False)
                            or getattr(args.data, 'rho1_distinctive_advantage', False)
                            or getattr(args.data, 'rho1_capgap_enable', False)
                        ):
                            all_expert_nlls = stacked.clone()
                            all_expert_nlls[stacked == float("inf")] = 0.0
                routing_freq = getattr(args.logging, "routing_freq", 10)
                should_log_routing = (
                    routing_debug_line is not None
                    and get_is_master()
                    and routing_freq is not None
                    and routing_freq > 0
                    and train_state.acc_step == 0
                    and ((train_state.step + 1) % routing_freq == 0)
                )
                if should_log_routing:
                    logger.info(routing_debug_line.replace(f"step={train_state.step}", f"step={train_state.step + 1}"))

            elif teacher_model is not None and (needs_kl or teacher_logprobs is None):
                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_lp = F.log_softmax(teacher_out.logits, dim=-1)

                    if teacher_logprobs is None:
                        teacher_logprobs = -teacher_lp.gather(
                            dim=-1, index=labels.clamp(min=0).unsqueeze(-1)
                        ).squeeze(-1)
                        teacher_logprobs[labels == -100] = 0.0

                    # Compute entropy only if not already provided by precomputed view
                    if teacher_entropy_precomputed is None:
                        needs_entropy = (
                            (hasattr(args.data, 'use_entropy_delta') and args.data.use_entropy_delta) or
                            (hasattr(args.data, 'use_entropy_aware_margin') and args.data.use_entropy_aware_margin)
                        )
                        if needs_entropy:
                            teacher_probs = torch.exp(teacher_lp)
                            teacher_entropy_precomputed = -torch.sum(teacher_probs * teacher_lp, dim=-1)
                            del teacher_probs  # Free immediately

                    # Store teacher logits for methods needing full logits (margin computation).
                    # Not needed when teacher_margin_precomputed is already available.
                    if teacher_margin_precomputed is None:
                        needs_teacher_logits = (
                            (hasattr(args.data, 'use_entropy_aware_margin') and args.data.use_entropy_aware_margin) or
                            (hasattr(args.data, 'use_margin_constraint') and args.data.use_margin_constraint)
                        )
                        if needs_teacher_logits:
                            teacher_logits_for_eam = teacher_out.logits.detach()

                    del teacher_out, teacher_lp  # Free full vocab tensors

            # EMA self-reference: compute reference NLL from frozen/EMA model
            ema_ref_nll = None
            if ema_ref_model is not None:
                with torch.inference_mode():
                    ema_ref_out = ema_ref_model(input_ids)
                    ema_ref_logits = (
                        ema_ref_out.logits if hasattr(ema_ref_out, "logits") else ema_ref_out
                    )
                    ema_ref_lp = F.log_softmax(ema_ref_logits.float(), dim=-1)
                    ema_ref_nll = -ema_ref_lp.gather(
                        dim=-1, index=labels.clamp(min=0).unsqueeze(-1)
                    ).squeeze(-1)
                    ema_ref_nll[labels == -100] = 0.0
                    del ema_ref_out, ema_ref_lp

            # Log dataloader output at start of training to verify masking
            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                total_labels = labels.numel()
                masked_labels = (labels == -100).sum().item()
                valid_labels = total_labels - masked_labels
                mask_pct = 100.0 * masked_labels / total_labels if total_labels > 0 else 0

                logger.info("=" * 60)
                logger.info("DATALOADER DEBUG - First batch sample")
                logger.info(f"  Batch shape: {labels.shape} (batch_size, seq_len)")
                logger.info(f"  Total labels: {total_labels}")
                logger.info(f"  Masked labels (-100): {masked_labels} ({mask_pct:.1f}%)")
                logger.info(f"  Valid labels (loss computed): {valid_labels} ({100-mask_pct:.1f}%)")
                logger.info(f"  mask_synth_context_loss: {args.data.mask_synth_context_loss}")

                # Decode first sequence to show what's masked vs not
                if hasattr(args.data, 'mask_synth_context_loss') and args.data.mask_synth_context_loss:
                    try:
                        # Get first sequence
                        seq_input = input_ids[0].cpu().tolist()
                        seq_labels = labels[0].cpu().tolist()

                        # Find boundary where masking ends
                        first_valid_idx = next((i for i, l in enumerate(seq_labels) if l != -100), len(seq_labels))

                        logger.info(f"  First sequence - mask ends at token index: {first_valid_idx}")
                        logger.info(f"  Masked region (synth context): tokens 0-{first_valid_idx-1}")
                        logger.info(f"  Valid region (original text): tokens {first_valid_idx}-{len(seq_labels)-1}")

                        # Decode a snippet if tokenizer available
                        masked_snippet = seq_input[:min(50, first_valid_idx)]
                        valid_snippet = seq_input[first_valid_idx:first_valid_idx+50] if first_valid_idx < len(seq_input) else []

                        logger.info(f"  Masked token IDs (first 50): {masked_snippet}")
                        logger.info(f"  Valid token IDs (first 50): {valid_snippet}")

                        # Try to decode
                        decoded_masked = tokenizer.decode(masked_snippet)
                        decoded_valid = tokenizer.decode(valid_snippet) if valid_snippet else "(empty)"
                        logger.info(f"  Masked text preview: {decoded_masked[:200]}...")
                        logger.info(f"  Valid text preview: {decoded_valid[:200]}...")
                    except Exception as e:
                        logger.info(f"  Could not decode tokens: {e}")

                logger.info("=" * 60)

            # # A) pre-existing -100s from the loader (doc boundaries, etc.) BEFORE you touch labels
            # preexisting = (labels == -100).sum().item()

            # # B) metadata tokens present in labels BEFORE masking
            # specials_present = torch.isin(labels, IGNORE_IDS_T.to(labels.device)).sum().item()

            # # Apply your metadata mask (only when enabled)
            # if args.data.add_special_tokens:
            #     ignore_mask = torch.isin(labels, IGNORE_IDS_T.to(labels.device))
            #     labels = labels.masked_fill(ignore_mask, -100)

            # # C) total -100 AFTER your mask
            # total_after = (labels == -100).sum().item()

            # if get_is_master():
            #     logger.info(f"masked(preexisting)={preexisting} "
            #                 f"specials_present={specials_present} "
            #                 f"masked(total_after)={total_after} "
            #                 f"add_specials={args.data.add_special_tokens}")
            data_load_time = round(timer() - data_load_start, 4)
            nwords_since_last_log += input_ids.numel()

            bsz, seqlen = labels.shape
            grad_probe_stats: Dict[str, float] = {}

            # forward
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            # This is an automatic probe that will compute statistics
            # of all linears' inputs, weights and outputs
            # along with attention logits and entropy
            # both in forward and backward pass
            if (args.probe_freq is not None) and every_n_steps(
                train_state, args.probe_freq, acc_step=1 % args.grad_acc_steps
            ):
                # Here we do a fake forward and backward pass on a smaller
                # batch size to avoid OOM
                # This assumes the model has no stateful layers (batch norm..)
                assert (
                    next(model.parameters()).grad is None
                ), "Can't probe model if grads are not reset"

                with probe:
                    probe.metadata = {
                        "it": train_state.step,
                        "global_step": train_state.step,
                        "loop": "lingua",
                    }
                    # Non compiled model uses roughly 2x memory in our exps
                    # So we divide bsz by 2 or seqlen by 2
                    probe_bsz = max(1, bsz // 2)
                    probe_seq = seqlen if (bsz // 2 >= 1) else (seqlen // 2)
                    probe_loss = model(
                        input_ids[:probe_bsz, :probe_seq],
                        labels[:probe_bsz, :probe_seq],
                    )

                    # if doc_ids_t is not None:
                    #     probe_doc = doc_ids_t[:probe_bsz, :probe_seq]
                    #     probe_attn_impl = "flex_attention"
                    # else:
                    #     probe_doc = None
                    #     probe_attn_impl = "sdpa"

                    # probe_loss = model(
                    #     input_ids[:probe_bsz, :probe_seq],
                    #     labels[:probe_bsz, :probe_seq],
                    #     attn_impl=probe_attn_impl,
                    #     doc_ids=probe_doc,
                    #     mask_cross_doc_loss=args.data.mask_cross_doc_loss,
                    # )
                    probe_loss.backward()
                    # We zero grads to cancel this fake step
                    optimizer.zero_grad()

                assert (
                    next(model.parameters()).grad is None
                ), "Probe model shouldn't have grads at this point"

            if (
                args.grad_probe_freq is not None
                and train_state.acc_step == first_micro_acc_step
                and every_n_steps(train_state, args.grad_probe_freq, acc_step=first_micro_acc_step)
            ):
                if _source_labels is None and not grad_probe_warned_no_labels and get_is_master():
                    logger.warning(
                        "[GradProbe] source labels are unavailable for this dataloader path; "
                        "domain cosine metrics require source-labeled batches."
                    )
                    grad_probe_warned_no_labels = True
                try:
                    assert (
                        next(model.parameters()).grad is None
                    ), "Grad probe requires clean grads (run at optimizer-step boundary)."
                    grad_probe_stats = _grad_probe_stats_for_batch(
                        model=model,
                        input_ids=input_ids,
                        labels=labels,
                        source_labels=_source_labels,
                        probe_named_params=grad_probe_named_params,
                        domain_keys=[str(x).lower() for x in args.grad_probe_domains],
                        min_seqs=max(1, int(args.grad_probe_min_seqs)),
                    )
                    optimizer.zero_grad()
                except Exception as e:
                    grad_probe_stats = {"grad_probe/error": 1.0}
                    if get_is_master():
                        logger.warning(f"[GradProbe] failed at step {train_state.step}: {e}")
                pending_grad_probe_stats = grad_probe_stats

            #loss = model(input_ids, labels, doc_ids=doc_ids_t, mask_cross_doc_loss=args.data.mask_cross_doc_loss, attn_impl=attn_impl)

            # Use weighted loss when teacher logprobs are available (cached or on-the-fly)
            use_weighted = teacher_logprobs is not None and (
                args.data.use_teacher_logprobs or teacher_model is not None
            )
            # If we're past the RKL→NTP stage switch (idx 124), force-disable
            # use_weighted so the dispatch falls through to vanilla NTP (the
            # bare `else: loss = model(input_ids, labels)`) instead of the
            # legacy delta-weighted CE path (compute_weighted_loss_with_teacher).
            # See the bug fix in the reverse_kl_stage_schedule_enabled block
            # below for context.
            if (
                getattr(args.data, 'use_reverse_kl_distillation', False)
                and getattr(args.data, 'reverse_kl_stage_schedule_enabled', False)
                and teacher_model is not None
            ):
                _switch_step = int(args.steps * float(getattr(args.data, 'reverse_kl_stage_switch_frac', 0.8)))
                if train_state.step >= _switch_step:
                    use_weighted = False
            use_rho1 = getattr(args.data, 'use_rho1', False)
            use_best_expert = getattr(args.data, 'use_best_expert', False)
            use_best_expert_rho1 = getattr(args.data, 'use_best_expert_rho1', False)
            use_best_expert_seq = getattr(args.data, 'use_best_expert_seq', False)
            use_best_expert_seq_rho1 = getattr(args.data, 'use_best_expert_seq_rho1', False)
            best_expert_seq_rho1_select_ratio = getattr(args.data, 'best_expert_seq_rho1_select_ratio', 0.6)
            use_best_expert_seq_kd = getattr(args.data, 'use_best_expert_seq_kd', False)
            online_rw_teacher_routing_only = bool(
                getattr(args.online_reweighting, 'teacher_routing_only', False)
            )
            use_kl_distillation = getattr(args.data, 'use_kl_distillation', False) and teacher_model is not None
            use_projected_kd = getattr(args.data, 'use_projected_kd', False) and teacher_model is not None
            use_reverse_kl_distillation = getattr(args.data, 'use_reverse_kl_distillation', False) and teacher_model is not None
            if use_reverse_kl_distillation and getattr(args.data, 'reverse_kl_stage_schedule_enabled', False):
                _rkl_switch_step = int(args.steps * float(getattr(args.data, 'reverse_kl_stage_switch_frac', 0.8)))
                if train_state.step >= _rkl_switch_step:
                    use_reverse_kl_distillation = False
                    # Bug fix (2026-06): also need to force use_weighted=False
                    # so the dispatch falls through to vanilla NTP. That's
                    # handled at the use_weighted assignment above. Without
                    # both flips, the chain falls through to the legacy
                    # delta-weighted CE path (compute_weighted_loss_with_teacher)
                    # instead of vanilla NTP for the tail.
                    if train_state.step == _rkl_switch_step and train_state.acc_step == 1 and get_is_master():
                        logger.info(
                            f"[RKL stage schedule] step={train_state.step} >= switch_step={_rkl_switch_step} "
                            f"(switch_frac={float(args.data.reverse_kl_stage_switch_frac)}, total_steps={args.steps}); "
                            f"switching from reverse-KL distillation to vanilla NTP for the tail."
                        )
            use_rkl_fkl_mix_distillation = getattr(args.data, 'use_rkl_fkl_mix_distillation', False) and teacher_model is not None
            use_rkl_with_gated_ce_distillation = getattr(args.data, 'use_rkl_with_gated_ce_distillation', False) and teacher_model is not None
            use_rkl_with_lowent_ce_distillation = getattr(args.data, 'use_rkl_with_lowent_ce_distillation', False) and teacher_model is not None
            use_rkl_with_source_ce_distillation = getattr(args.data, 'use_rkl_with_source_ce_distillation', False) and teacher_model is not None
            use_rkl_with_teacher_disagree_ce_distillation = getattr(args.data, 'use_rkl_with_teacher_disagree_ce_distillation', False) and teacher_model is not None
            use_rkl_with_teacher_fail_ce_distillation = getattr(args.data, 'use_rkl_with_teacher_fail_ce_distillation', False) and teacher_model is not None
            use_rkl_with_teacher_success_ce_distillation = getattr(args.data, 'use_rkl_with_teacher_success_ce_distillation', False) and teacher_model is not None
            use_rkl_with_topk_gap_ce_distillation = getattr(args.data, 'use_rkl_with_topk_gap_ce_distillation', False) and teacher_model is not None
            use_rkl_with_topkgap_schedule_ce_distillation = getattr(args.data, 'use_rkl_with_topkgap_schedule_ce_distillation', False) and teacher_model is not None
            use_rkl_with_topkgap_replace_ce_distillation = getattr(args.data, 'use_rkl_with_topkgap_replace_ce_distillation', False) and teacher_model is not None
            use_rkl_with_uniform_ce_distillation = getattr(args.data, 'use_rkl_with_uniform_ce_distillation', False) and teacher_model is not None
            use_rkl_entropy_gated_distillation = getattr(args.data, 'use_rkl_entropy_gated_distillation', False) and teacher_model is not None
            use_fkl_entropy_gated_distillation = getattr(args.data, 'use_fkl_entropy_gated_distillation', False) and teacher_model is not None
            use_entropy_band_kl_distillation = getattr(args.data, 'use_entropy_band_kl_distillation', False) and teacher_model is not None
            use_rkl_random_gated_distillation = getattr(args.data, 'use_rkl_random_gated_distillation', False) and teacher_model is not None
            use_rkl_student_entropy_gated_distillation = getattr(args.data, 'use_rkl_student_entropy_gated_distillation', False) and teacher_model is not None
            use_rkl_entropy_switched_distillation = getattr(args.data, 'use_rkl_entropy_switched_distillation', False) and teacher_model is not None
            use_rkl_with_gradagree_gap_ce_distillation = getattr(args.data, 'use_rkl_with_gradagree_gap_ce_distillation', False) and teacher_model is not None
            use_rho1_kd_distillation = getattr(args.data, 'use_rho1_kd_distillation', False) and teacher_model is not None
            use_akl_distillation = getattr(args.data, 'use_akl_distillation', False) and teacher_model is not None
            use_geometric_kd = getattr(args.data, 'use_geometric_kd', False) and teacher_model is not None
            use_entropy_gated_kd = getattr(args.data, 'use_entropy_gated_kd', False) and teacher_model is not None
            use_bucket_aware_kd = (
                getattr(args.data, 'use_bucket_aware_kd', False)
                and bucket_kd_main_logits is not None
                and bucket_kd_gate_logits is not None
            )
            use_hybrid_residual_kd = (
                getattr(args.data, 'use_hybrid_residual_kd', False)
                and bucket_kd_main_logits is not None
                and bucket_kd_gate_logits is not None
            )
            use_two_teacher_geometric_kd = (
                getattr(args.data, 'use_two_teacher_geometric_kd', False)
                and bucket_kd_main_logits is not None
                and bucket_kd_gate_logits is not None
            )
            use_projected_two_teacher_kd = (
                getattr(args.data, 'use_projected_two_teacher_kd', False)
                and bucket_kd_main_logits is not None
                and bucket_kd_gate_logits is not None
            )
            use_agreement_gated_two_teacher_kd = (
                getattr(args.data, 'use_agreement_gated_two_teacher_kd', False)
                and bucket_kd_main_logits is not None
                and bucket_kd_gate_logits is not None
            )
            use_competence_routed_two_teacher_kd = (
                getattr(args.data, 'use_competence_routed_two_teacher_kd', False)
                and bucket_kd_main_logits is not None
                and bucket_kd_gate_logits is not None
            )
            use_intersection_projected_kd = (
                getattr(args.data, 'use_intersection_projected_kd', False)
                and bucket_kd_main_logits is not None
                and bucket_kd_gate_logits is not None
            )
            use_remit = getattr(args.data, 'use_remit', False) and teacher_logprobs is not None
            use_teacher_critic = getattr(args.data, 'use_teacher_critic', False) and teacher_logprobs is not None
            distinctive_requested = bool(getattr(args.data, "rho1_distinctive_advantage", False))
            use_entropy_aware_margin = (
                getattr(args.data, 'use_entropy_aware_margin', False) and
                (teacher_logits_for_eam is not None or teacher_margin_precomputed is not None)
            )
            use_margin_constraint = (
                getattr(args.data, 'use_margin_constraint', False) and
                (teacher_logits_for_eam is not None or teacher_margin_precomputed is not None)
            )
            use_mile = getattr(args.data, 'use_mile', False)
            batch_delta_stats = None

            def _validate_distinctive_prereqs() -> None:
                if not distinctive_requested:
                    return
                if (
                    all_expert_nlls is None
                    or all_expert_nlls.dim() != 3
                    or all_expert_nlls.size(0) < 2
                ):
                    raise RuntimeError(
                        "rho1_distinctive_advantage=true requires per-expert NLLs with shape (E,B,T) and E>=2. "
                        "Distinctiveness would otherwise silently fall back to standard rho1."
                    )

            def _fmt_metric(v) -> str:
                """Safely format scalars/tensors for first-batch logging."""
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        return f"{float(v.item()):.4f}"
                    return str(v.detach().float().mean().item())
                try:
                    return f"{float(v):.4f}"
                except Exception:
                    return str(v)

            if not expert_routing_stats:
                expert_routing_stats = {}
            if use_entropy_aware_margin:
                logits = model(input_ids, target=None)
                if teacher_logits_for_eam.shape[-1] != logits.shape[-1]:
                    V_teacher = teacher_logits_for_eam.shape[-1]
                    V_student = logits.shape[-1]
                    if V_teacher < V_student:
                        padding = torch.full(
                            (*teacher_logits_for_eam.shape[:-1], V_student - V_teacher),
                            -1e10, device=teacher_logits_for_eam.device, dtype=teacher_logits_for_eam.dtype
                        )
                        teacher_logits_for_eam = torch.cat([teacher_logits_for_eam, padding], dim=-1)
                    else:
                        teacher_logits_for_eam = teacher_logits_for_eam[..., :V_student]

                loss, batch_delta_stats = compute_entropy_aware_margin_loss(
                    logits=logits,
                    labels=labels,
                    teacher_logits=teacher_logits_for_eam,
                    teacher_entropy=teacher_entropy_precomputed,
                    lambda_max=getattr(args.data, 'eam_lambda_max', 1.0),
                    gamma=getattr(args.data, 'eam_gamma', 2.0),
                    teacher_margin=teacher_margin_precomputed,
                )

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("ENTROPY-AWARE MARGIN MATCHING - First batch statistics")
                    logger.info(f"  Teacher model: {args.teacher_model_path}")
                    logger.info(f"  lambda_max: {getattr(args.data, 'eam_lambda_max', 1.0)}")
                    logger.info(f"  gamma: {getattr(args.data, 'eam_gamma', 2.0)}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif use_mile:
                logits = model(input_ids, target=None)

                loss, batch_delta_stats = compute_mile_loss(
                    logits=logits,
                    labels=labels,
                    gamma=getattr(args.data, 'mile_gamma', 1.0),
                )

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("MiLe LOSS - First batch statistics")
                    logger.info(f"  gamma: {getattr(args.data, 'mile_gamma', 1.0)}")
                    logger.info("  weighting: w = (H / log|V|)^gamma")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif use_remit:
                logits = model(input_ids, target=None)

                loss, batch_delta_stats = compute_remit_loss(
                    logits=logits,
                    labels=labels,
                    teacher_logprobs=teacher_logprobs,
                    clip_floor=getattr(args.data, 'remit_clip_floor', 0.2),
                )

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("ReMiT - RL-Guided Mid-Training - First batch statistics")
                    logger.info(f"  Reference model: {args.teacher_model_path}")
                    logger.info(f"  clip_floor: {getattr(args.data, 'remit_clip_floor', 0.2)}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif use_teacher_critic:
                logits = model(input_ids, target=None)

                loss, batch_delta_stats = compute_teacher_critic_loss(
                    logits=logits,
                    labels=labels,
                    teacher_logprobs=teacher_logprobs,
                    beta=getattr(args.data, 'critic_beta', 1.0),
                    clip_c=getattr(args.data, 'critic_clip', 3.0),
                    sigma=getattr(args.data, 'critic_sigma', 1.0),
                    mu=getattr(args.data, 'critic_mu', 0.0),
                    detach_weights=getattr(args.data, 'critic_detach_weights', True),
                )

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("TEACHER-AS-CRITIC - First batch statistics")
                    logger.info(f"  Reference model: {args.teacher_model_path}")
                    logger.info(f"  beta: {getattr(args.data, 'critic_beta', 1.0)}")
                    logger.info(f"  clip: {getattr(args.data, 'critic_clip', 3.0)}")
                    logger.info(f"  sigma: {getattr(args.data, 'critic_sigma', 1.0)}")
                    logger.info(f"  mu: {getattr(args.data, 'critic_mu', 0.0)}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif use_margin_constraint:
                logits = model(input_ids, target=None)
                if teacher_logits_for_eam.shape[-1] != logits.shape[-1]:
                    V_teacher = teacher_logits_for_eam.shape[-1]
                    V_student = logits.shape[-1]
                    if V_teacher < V_student:
                        padding = torch.full(
                            (*teacher_logits_for_eam.shape[:-1], V_student - V_teacher),
                            -1e10, device=teacher_logits_for_eam.device, dtype=teacher_logits_for_eam.dtype
                        )
                        teacher_logits_for_eam = torch.cat([teacher_logits_for_eam, padding], dim=-1)
                    else:
                        teacher_logits_for_eam = teacher_logits_for_eam[..., :V_student]

                loss, batch_delta_stats = compute_margin_constraint_loss(
                    logits=logits,
                    labels=labels,
                    teacher_logprobs=teacher_logprobs,
                    teacher_logits=teacher_logits_for_eam,
                    lam=getattr(args.data, 'mc_lambda', 1.0),
                    tau=getattr(args.data, 'mc_tau', 1.0),
                    teacher_margin=teacher_margin_precomputed,
                )

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("CONFIDENCE-GATED MARGIN HINGE - First batch statistics")
                    logger.info(f"  Teacher model: {args.teacher_model_path}")
                    logger.info(f"  lambda: {getattr(args.data, 'mc_lambda', 1.0)}")
                    logger.info(f"  tau: {getattr(args.data, 'mc_tau', 1.0)}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif use_bucket_aware_kd:
                logits = model(input_ids, target=None)

                _bakd_main_idx = int(getattr(args.data, "bucket_aware_kd_main_idx", 1))
                _bakd_gate_idx = int(getattr(args.data, "bucket_aware_kd_gate_idx", 0))
                if all_expert_nlls is None or all_expert_nlls.dim() != 3:
                    raise RuntimeError(
                        "use_bucket_aware_kd requires multi-teacher all_expert_nlls "
                        "(use_best_expert_rho1=true with teacher_model_paths=[gate, main]); "
                        "got all_expert_nlls=None."
                    )
                if all_expert_nlls.size(0) <= max(_bakd_main_idx, _bakd_gate_idx):
                    raise RuntimeError(
                        "use_bucket_aware_kd: teacher_model_paths must contain at least "
                        f"max(main_idx={_bakd_main_idx}, gate_idx={_bakd_gate_idx})+1 entries."
                    )
                if _bakd_main_idx == _bakd_gate_idx:
                    raise RuntimeError(
                        "use_bucket_aware_kd: main_idx and gate_idx must differ "
                        "(main = strong-capacity ref, gate = same-capacity ref)."
                    )
                _teacher_main_nll = all_expert_nlls[_bakd_main_idx]
                _teacher_gate_nll = all_expert_nlls[_bakd_gate_idx]

                _bakd_main_logits = bucket_kd_main_logits
                _bakd_gate_logits = bucket_kd_gate_logits
                if _bakd_main_logits.shape[-1] != logits.shape[-1]:
                    V_t = _bakd_main_logits.shape[-1]
                    V_s = logits.shape[-1]
                    if V_t < V_s:
                        padding = torch.full(
                            (*_bakd_main_logits.shape[:-1], V_s - V_t),
                            -1e10, device=_bakd_main_logits.device, dtype=_bakd_main_logits.dtype,
                        )
                        _bakd_main_logits = torch.cat([_bakd_main_logits, padding], dim=-1)
                    else:
                        _bakd_main_logits = _bakd_main_logits[..., :V_s]
                if _bakd_gate_logits.shape[-1] != logits.shape[-1]:
                    V_t = _bakd_gate_logits.shape[-1]
                    V_s = logits.shape[-1]
                    if V_t < V_s:
                        padding = torch.full(
                            (*_bakd_gate_logits.shape[:-1], V_s - V_t),
                            -1e10, device=_bakd_gate_logits.device, dtype=_bakd_gate_logits.dtype,
                        )
                        _bakd_gate_logits = torch.cat([_bakd_gate_logits, padding], dim=-1)
                    else:
                        _bakd_gate_logits = _bakd_gate_logits[..., :V_s]

                # Optional γ-annealing schedule for target_mode='soft'. When the
                # schedule is 'constant' (default) the resolved γ is exactly the
                # configured `bucket_aware_kd_soft_gamma`, preserving the existing
                # behavior of every in-flight soft-BAKD run (idx 37/38). For
                # 'linear' / 'cosine' the resolved γ interpolates between
                # `..._soft_gamma_start` and `..._soft_gamma_end` over the first
                # `..._soft_gamma_anneal_steps` optimizer steps (saturating at
                # the end value afterwards). The resolved γ is what gets logged
                # to wandb as `bakd/soft_gamma` (loss-fn-side), so wandb already
                # shows the schedule trajectory without any additional plumbing.
                _bakd_gamma_schedule = str(getattr(args.data, 'bucket_aware_kd_soft_gamma_schedule', 'constant')).lower()
                _bakd_gamma_base = float(getattr(args.data, 'bucket_aware_kd_soft_gamma', 1.0))
                if _bakd_gamma_schedule == 'constant':
                    _bakd_gamma_now = _bakd_gamma_base
                else:
                    _bakd_gamma_start = float(getattr(args.data, 'bucket_aware_kd_soft_gamma_start', _bakd_gamma_base))
                    _bakd_gamma_end = float(getattr(args.data, 'bucket_aware_kd_soft_gamma_end', _bakd_gamma_base))
                    _bakd_anneal_steps = int(getattr(args.data, 'bucket_aware_kd_soft_gamma_anneal_steps', 9600))
                    _bakd_step_norm = min(max(float(train_state.step) / max(_bakd_anneal_steps, 1), 0.0), 1.0)
                    if _bakd_gamma_schedule == 'linear':
                        _bakd_gamma_now = _bakd_gamma_start + (_bakd_gamma_end - _bakd_gamma_start) * _bakd_step_norm
                    elif _bakd_gamma_schedule == 'cosine':
                        # Smooth ease, starts at gamma_start and ends at gamma_end (cosine half-period).
                        _bakd_cos_frac = 0.5 * (1.0 + math.cos(math.pi * _bakd_step_norm))
                        _bakd_gamma_now = _bakd_gamma_end + (_bakd_gamma_start - _bakd_gamma_end) * _bakd_cos_frac
                    else:
                        raise RuntimeError(
                            f"bucket_aware_kd_soft_gamma_schedule must be one of "
                            f"'constant'|'linear'|'cosine'; got {_bakd_gamma_schedule!r}."
                        )

                loss, batch_delta_stats = compute_bucket_aware_kd_loss(
                    student_logits=logits,
                    teacher_main_logits=_bakd_main_logits,
                    teacher_gate_logits=_bakd_gate_logits,
                    teacher_main_nll=_teacher_main_nll,
                    teacher_gate_nll=_teacher_gate_nll,
                    labels=labels,
                    temperature=getattr(args.data, 'bucket_aware_kd_temperature', 2.0),
                    alpha=getattr(args.data, 'bucket_aware_kd_alpha', 0.5),
                    x_margin=getattr(args.data, 'bucket_aware_kd_x_margin', 0.0),
                    y_margin=getattr(args.data, 'bucket_aware_kd_y_margin', 0.0),
                    b_teacher=str(getattr(args.data, 'bucket_aware_kd_b_teacher', 'main')),
                    other_teacher=str(getattr(args.data, 'bucket_aware_kd_other_teacher', 'gate')),
                    chunk_size=int(getattr(args.data, 'bucket_aware_kd_chunk_size', 128)),
                    routing=str(getattr(args.data, 'bucket_aware_kd_routing', 'rectangular')),
                    k=float(getattr(args.data, 'bucket_aware_kd_k', 0.4)),
                    main_headroom_margin=float(getattr(args.data, 'bucket_aware_kd_main_headroom_margin', 0.0)),
                    target_mode=str(getattr(args.data, 'bucket_aware_kd_target_mode', 'hard')),
                    soft_gamma=_bakd_gamma_now,
                    soft_gamma_disagree=float(getattr(args.data, 'bucket_aware_kd_soft_gamma_disagree', -1.0)),
                    top1_guard=bool(getattr(args.data, 'bucket_aware_kd_top1_guard', False)),
                )

                if bool(getattr(args.data, 'kd_diagnostics_enabled', False)):
                    # Diagnostic uses the *gate* teacher as the 1B anchor, since
                    # the comparison-of-interest is "is kd-rl (1B-teacher uniform KD)
                    # signal-concentrated?". This is the apples-to-apples view to
                    # decide whether selective KD with the 1B teacher is worth running.
                    _kd_mask = (labels != -100).to(dtype=logits.dtype)
                    _kd_domain_ids, _kd_id_to_name = _build_token_domain_ids(
                        labels=labels,
                        cu_seqlens=batch_cu_seqlens,
                        doc_sources=batch_doc_sources,
                        domain_labels=_source_labels,
                    )
                    _kd_diag_stats = _compute_kd_concentration_diagnostics(
                        student_logits=logits,
                        teacher_logits=_bakd_gate_logits,
                        mask=_kd_mask,
                        temperature=float(getattr(args.data, 'bucket_aware_kd_temperature', 2.0)),
                        domain_ids=_kd_domain_ids,
                        id_to_name=_kd_id_to_name,
                        chunk_size=int(getattr(args.data, 'kd_diagnostics_chunk_size', 128)),
                    )
                    batch_delta_stats.update(_kd_diag_stats)

                del _bakd_main_logits, _bakd_gate_logits
                bucket_kd_main_logits = None
                bucket_kd_gate_logits = None

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("BUCKET-AWARE KD - First batch statistics")
                    logger.info(
                        f"  Teachers: gate=teacher[{_bakd_gate_idx}], main=teacher[{_bakd_main_idx}]"
                    )
                    logger.info(
                        f"  Routing: bucket B → {getattr(args.data, 'bucket_aware_kd_b_teacher', 'main')}; "
                        f"others → {getattr(args.data, 'bucket_aware_kd_other_teacher', 'gate')}"
                    )
                    logger.info(
                        f"  Margins: x_margin={getattr(args.data, 'bucket_aware_kd_x_margin', 0.0)}  "
                        f"y_margin={getattr(args.data, 'bucket_aware_kd_y_margin', 0.0)}"
                    )
                    logger.info(
                        f"  Temperature: {getattr(args.data, 'bucket_aware_kd_temperature', 2.0)}  "
                        f"Alpha: {getattr(args.data, 'bucket_aware_kd_alpha', 0.5)}"
                    )
                    _bakd_target_mode = str(getattr(args.data, 'bucket_aware_kd_target_mode', 'hard')).lower()
                    if _bakd_target_mode == 'soft':
                        _bakd_log_sched = str(getattr(args.data, 'bucket_aware_kd_soft_gamma_schedule', 'constant')).lower()
                        if _bakd_log_sched == 'constant':
                            logger.info(
                                f"  Target mode: soft (geometric/PoE mixture)  "
                                f"γ={getattr(args.data, 'bucket_aware_kd_soft_gamma', 1.0)} (constant)"
                            )
                        else:
                            logger.info(
                                f"  Target mode: soft (geometric/PoE mixture)  "
                                f"γ schedule={_bakd_log_sched}  "
                                f"start={getattr(args.data, 'bucket_aware_kd_soft_gamma_start', None)} → "
                                f"end={getattr(args.data, 'bucket_aware_kd_soft_gamma_end', None)} "
                                f"over {getattr(args.data, 'bucket_aware_kd_soft_gamma_anneal_steps', 9600)} steps  "
                                f"(γ@step0={_bakd_gamma_now:.4f})"
                            )
                    else:
                        logger.info("  Target mode: hard (in_B → main, else → gate)")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_hybrid_residual_kd:
                # Hybrid-Residual KD: 1B-RLVR forward-KL distillation (= kd-rl)
                # plus a small compatibility-gated auxiliary divergence against
                # the strong 7B teacher. Reuses BAKD's multi-teacher logits
                # captured upstream into bucket_kd_{main,gate}_logits and the
                # per-teacher NLLs in all_expert_nlls.
                logits = model(input_ids, target=None)

                _hkd_main_idx = int(getattr(args.data, "hybrid_residual_kd_main_idx", 1))
                _hkd_gate_idx = int(getattr(args.data, "hybrid_residual_kd_gate_idx", 0))
                if all_expert_nlls is None or all_expert_nlls.dim() != 3:
                    raise RuntimeError(
                        "use_hybrid_residual_kd requires multi-teacher all_expert_nlls "
                        "(use_best_expert_rho1=true with teacher_model_paths=[gate, main]); "
                        "got all_expert_nlls=None."
                    )
                if all_expert_nlls.size(0) <= max(_hkd_main_idx, _hkd_gate_idx):
                    raise RuntimeError(
                        "use_hybrid_residual_kd: teacher_model_paths must contain at least "
                        f"max(main_idx={_hkd_main_idx}, gate_idx={_hkd_gate_idx})+1 entries."
                    )
                if _hkd_main_idx == _hkd_gate_idx:
                    raise RuntimeError(
                        "use_hybrid_residual_kd: main_idx and gate_idx must differ "
                        "(main = strong-capacity ref, gate = same-capacity ref)."
                    )
                _hkd_main_nll = all_expert_nlls[_hkd_main_idx]
                _hkd_gate_nll = all_expert_nlls[_hkd_gate_idx]

                _hkd_main_logits = bucket_kd_main_logits
                _hkd_gate_logits = bucket_kd_gate_logits
                if _hkd_main_logits.shape[-1] != logits.shape[-1]:
                    V_t = _hkd_main_logits.shape[-1]
                    V_s = logits.shape[-1]
                    if V_t < V_s:
                        padding = torch.full(
                            (*_hkd_main_logits.shape[:-1], V_s - V_t),
                            -1e10, device=_hkd_main_logits.device, dtype=_hkd_main_logits.dtype,
                        )
                        _hkd_main_logits = torch.cat([_hkd_main_logits, padding], dim=-1)
                    else:
                        _hkd_main_logits = _hkd_main_logits[..., :V_s]
                if _hkd_gate_logits.shape[-1] != logits.shape[-1]:
                    V_t = _hkd_gate_logits.shape[-1]
                    V_s = logits.shape[-1]
                    if V_t < V_s:
                        padding = torch.full(
                            (*_hkd_gate_logits.shape[:-1], V_s - V_t),
                            -1e10, device=_hkd_gate_logits.device, dtype=_hkd_gate_logits.dtype,
                        )
                        _hkd_gate_logits = torch.cat([_hkd_gate_logits, padding], dim=-1)
                    else:
                        _hkd_gate_logits = _hkd_gate_logits[..., :V_s]

                loss, batch_delta_stats = compute_hybrid_residual_kd_loss(
                    student_logits=logits,
                    teacher_main_logits=_hkd_main_logits,
                    teacher_gate_logits=_hkd_gate_logits,
                    teacher_main_nll=_hkd_main_nll,
                    teacher_gate_nll=_hkd_gate_nll,
                    labels=labels,
                    temperature=getattr(args.data, 'hybrid_residual_kd_temperature', 2.0),
                    alpha=getattr(args.data, 'hybrid_residual_kd_alpha', 0.5),
                    aux_lambda=getattr(args.data, 'hybrid_residual_kd_lambda', 0.1),
                    aux_divergence=str(getattr(args.data, 'hybrid_residual_kd_divergence', 'jsd')),
                    gate=str(getattr(args.data, 'hybrid_residual_kd_gate', 'rectangular')),
                    x_margin=float(getattr(args.data, 'hybrid_residual_kd_x_margin', 0.0)),
                    y_margin=float(getattr(args.data, 'hybrid_residual_kd_y_margin', 0.0)),
                    main_headroom_margin=float(getattr(args.data, 'hybrid_residual_kd_main_headroom_margin', 0.0)),
                    weight_mode=str(getattr(args.data, 'hybrid_residual_kd_weight_mode', 'binary')),
                    weight_clip_max=float(getattr(args.data, 'hybrid_residual_kd_weight_clip_max', 3.0)),
                    main_better_gate=bool(getattr(args.data, 'hybrid_residual_kd_main_better_gate', False)),
                    selfnorm_compat=str(getattr(args.data, 'hybrid_residual_kd_selfnorm_compat', 'top1')),
                    chunk_size=int(getattr(args.data, 'hybrid_residual_kd_chunk_size', 64)),
                    # Opt-in fast path: skip activation checkpointing of the per-chunk JSD body.
                    # Only safe when HBM headroom permits (e.g. with FP8 teachers freeing weight
                    # memory). Driven by env var to avoid touching configs of in-flight runs.
                    disable_activation_checkpoint=(
                        os.environ.get("LINGUA_HYBRID_KD_NO_AC", "0") == "1"
                    ),
                )

                del _hkd_main_logits, _hkd_gate_logits
                bucket_kd_main_logits = None
                bucket_kd_gate_logits = None

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("HYBRID-RESIDUAL KD - First batch statistics")
                    logger.info(
                        f"  Teachers: gate=teacher[{_hkd_gate_idx}] (main KD target), "
                        f"main=teacher[{_hkd_main_idx}] (auxiliary)"
                    )
                    logger.info(
                        f"  Auxiliary divergence: {getattr(args.data, 'hybrid_residual_kd_divergence', 'jsd')}; "
                        f"gate rule: {getattr(args.data, 'hybrid_residual_kd_gate', 'rectangular')}"
                    )
                    logger.info(
                        f"  Margins: x_margin={getattr(args.data, 'hybrid_residual_kd_x_margin', 0.0)}  "
                        f"y_margin={getattr(args.data, 'hybrid_residual_kd_y_margin', 0.0)}  "
                        f"main_headroom_margin={getattr(args.data, 'hybrid_residual_kd_main_headroom_margin', 0.0)}"
                    )
                    logger.info(
                        f"  Temperature: {getattr(args.data, 'hybrid_residual_kd_temperature', 2.0)}  "
                        f"Alpha: {getattr(args.data, 'hybrid_residual_kd_alpha', 0.5)}  "
                        f"Lambda: {getattr(args.data, 'hybrid_residual_kd_lambda', 0.1)}"
                    )
                    logger.info(
                        f"  Weight mode: {getattr(args.data, 'hybrid_residual_kd_weight_mode', 'binary')}  "
                        f"Selfnorm compat: {getattr(args.data, 'hybrid_residual_kd_selfnorm_compat', 'top1')}  "
                        f"Main-better-gate: {getattr(args.data, 'hybrid_residual_kd_main_better_gate', False)}"
                    )
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_two_teacher_geometric_kd:
                # Two-Teacher Student-Anchored Geometric KD.
                # Target: log q_λ = (1-λ)·log p_anchor + λ·log p_broad   (renormalised)
                # Loss (Option-A gate, default-on):
                #   L = CE + α·T²·m_t·KL(q_λ || p_S),   m_t = 1[L_S > min(L_1, L_2)].
                # Reuses BAKD's bucket_kd_{main,gate}_logits — captured upstream
                # via the _bakd_enabled multi-teacher fwd path. main_idx selects
                # the broad teacher (default 1 = 7B-Instruct); gate_idx selects
                # the anchor teacher (default 0 = 1B-RLVR1).
                logits = model(input_ids, target=None)

                _2tgkd_main_idx = int(getattr(args.data, "two_teacher_geometric_kd_main_idx", 1))
                _2tgkd_gate_idx = int(getattr(args.data, "two_teacher_geometric_kd_gate_idx", 0))
                if _2tgkd_main_idx == _2tgkd_gate_idx:
                    raise RuntimeError(
                        "use_two_teacher_geometric_kd: main_idx and gate_idx must differ "
                        "(main = broad teacher, gate = anchor teacher)."
                    )
                # Convention: t1 = anchor teacher (= bucket_kd_gate_logits), t2 = broad teacher.
                _t1_logits = bucket_kd_gate_logits
                _t2_logits = bucket_kd_main_logits

                # Pad/truncate each teacher's vocab to match student.
                for _name in ("t1", "t2"):
                    _t = _t1_logits if _name == "t1" else _t2_logits
                    if _t.shape[-1] != logits.shape[-1]:
                        V_t = _t.shape[-1]
                        V_s = logits.shape[-1]
                        if V_t < V_s:
                            padding = torch.full(
                                (*_t.shape[:-1], V_s - V_t),
                                -1e10,
                                device=_t.device,
                                dtype=_t.dtype,
                            )
                            _t = torch.cat([_t, padding], dim=-1)
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[2t-geom-kd] Padded teacher {_name} logits {V_t} → {V_s}"
                                )
                        else:
                            _t = _t[..., :V_s]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[2t-geom-kd] Truncated teacher {_name} logits {V_t} → {V_s}"
                                )
                        if _name == "t1":
                            _t1_logits = _t
                        else:
                            _t2_logits = _t

                loss, batch_delta_stats = compute_two_teacher_geometric_kd_loss(
                    student_logits=logits,
                    teacher_1_logits=_t1_logits,
                    teacher_2_logits=_t2_logits,
                    labels=labels,
                    lambda_mix=float(getattr(args.data, "two_teacher_geometric_kd_lambda", 0.5)),
                    temperature=float(getattr(args.data, "two_teacher_geometric_kd_temperature", 2.0)),
                    alpha=float(getattr(args.data, "two_teacher_geometric_kd_alpha", 1.0)),
                    chunk_size=int(getattr(args.data, "two_teacher_geometric_kd_chunk_size", 128)),
                    gold_gate=bool(getattr(args.data, "two_teacher_geometric_kd_gold_gate", True)),
                    teacher_1_nll=None,  # recompute inside the loss fn; cheap and avoids fragile coupling
                    teacher_2_nll=None,
                    use_canonical_loss=bool(getattr(args.data, "two_teacher_geometric_kd_use_canonical_loss", False)),
                    # Per-document adaptive λ: cu_seqlens carries doc boundaries from FA2 varlen
                    # path; the remaining flags select fixed vs bucketed_gold_nll vs noise_control.
                    cu_seqlens=batch_cu_seqlens,
                    adaptive_lambda_mode=str(getattr(args.data, "two_teacher_geometric_kd_adaptive_lambda_mode", "fixed")),
                    lambda_mid=float(getattr(args.data, "two_teacher_geometric_kd_lambda_mid", 0.25)),
                    lambda_spread=float(getattr(args.data, "two_teacher_geometric_kd_lambda_spread", 0.20)),
                    lambda_threshold_low=float(getattr(args.data, "two_teacher_geometric_kd_lambda_threshold_low", -10.0)),
                    lambda_threshold_high=float(getattr(args.data, "two_teacher_geometric_kd_lambda_threshold_high", +10.0)),
                    global_step=int(train_state.step),
                    # Temporal λ schedule: orthogonal to per-doc routing. When
                    # lambda_schedule="constant" (default) this is byte-identical
                    # to the legacy path; "cosine" overrides lambda_mix with a
                    # per-step value resolved from (global_step / total_steps).
                    lambda_schedule=str(getattr(args.data, "two_teacher_geometric_kd_lambda_schedule", "constant")),
                    lambda_start=float(getattr(args.data, "two_teacher_geometric_kd_lambda_start", 0.10)),
                    lambda_end=float(getattr(args.data, "two_teacher_geometric_kd_lambda_end", 0.55)),
                    lambda_warmup_frac=float(getattr(args.data, "two_teacher_geometric_kd_lambda_warmup_frac", 0.20)),
                    lambda_plateau_frac=float(getattr(args.data, "two_teacher_geometric_kd_lambda_plateau_frac", 0.20)),
                    total_steps=int(args.steps),
                    use_reverse_kl=bool(getattr(args.data, "two_teacher_geometric_kd_use_reverse_kl", False)),
                )

                del _t1_logits, _t2_logits
                bucket_kd_main_logits = None
                bucket_kd_gate_logits = None

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("TWO-TEACHER STUDENT-ANCHORED GEOMETRIC KD - First batch statistics")
                    logger.info(f"  Anchor teacher (t1, gate_idx={_2tgkd_gate_idx}): args.teacher_model_paths[{_2tgkd_gate_idx}]")
                    logger.info(f"  Broad  teacher (t2, main_idx={_2tgkd_main_idx}): args.teacher_model_paths[{_2tgkd_main_idx}]")
                    logger.info(f"  Lambda (mix exponent on broad teacher): {getattr(args.data, 'two_teacher_geometric_kd_lambda', 0.5)}")
                    logger.info(f"  Temperature: {getattr(args.data, 'two_teacher_geometric_kd_temperature', 2.0)}")
                    logger.info(f"  Alpha (KL weight; CE coef = 1): {getattr(args.data, 'two_teacher_geometric_kd_alpha', 1.0)}")
                    logger.info(f"  Gold-advantage gate (Option A: any teacher beats S): {getattr(args.data, 'two_teacher_geometric_kd_gold_gate', True)}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_projected_two_teacher_kd:
                # Projected Two-Teacher KD: arithmetic mixture of two teachers
                # (anchor + broad) in *probability* space; standard forward-KL
                # against the student. Reuses BAKD's bucket_kd_{main,gate}_logits
                # captured upstream via the _bakd_enabled multi-teacher fwd path.
                #
                #   p_T'  = (1-rho)·p_anchor  +  rho·p_broad        (both at T)
                #   L     = (1-alpha)·CE(z_S, y) + alpha·T²·KL(p_T' || p_S)
                #
                # Convention (matches two_teacher_geometric_kd):
                #   gate_idx -> anchor teacher (e.g. 1B-RLVR1) -> t1
                #   main_idx -> broad  teacher (e.g. 7B-Instruct) -> t2
                # so rho=0 reduces exactly to kd-rl (anchor-only) and rho=1
                # reduces exactly to kd-rl7b (broad-only).
                logits = model(input_ids, target=None)

                _p2tkd_main_idx = int(getattr(args.data, "projected_two_teacher_kd_main_idx", 1))
                _p2tkd_gate_idx = int(getattr(args.data, "projected_two_teacher_kd_gate_idx", 0))
                if _p2tkd_main_idx == _p2tkd_gate_idx:
                    raise RuntimeError(
                        "use_projected_two_teacher_kd: main_idx and gate_idx must differ "
                        "(main = broad teacher, gate = anchor teacher)."
                    )
                _t1_logits = bucket_kd_gate_logits   # anchor (1B-RLVR1)
                _t2_logits = bucket_kd_main_logits   # broad  (7B-Instruct)

                # Pad/truncate each teacher's vocab to match student.
                for _name in ("t1", "t2"):
                    _t = _t1_logits if _name == "t1" else _t2_logits
                    if _t.shape[-1] != logits.shape[-1]:
                        V_t = _t.shape[-1]
                        V_s = logits.shape[-1]
                        if V_t < V_s:
                            padding = torch.full(
                                (*_t.shape[:-1], V_s - V_t),
                                -1e10,
                                device=_t.device,
                                dtype=_t.dtype,
                            )
                            _t = torch.cat([_t, padding], dim=-1)
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[proj-2t-kd] Padded teacher {_name} logits {V_t} → {V_s}"
                                )
                        else:
                            _t = _t[..., :V_s]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[proj-2t-kd] Truncated teacher {_name} logits {V_t} → {V_s}"
                                )
                        if _name == "t1":
                            _t1_logits = _t
                        else:
                            _t2_logits = _t

                loss, batch_delta_stats = compute_projected_two_teacher_kd_loss(
                    student_logits=logits,
                    teacher_1_logits=_t1_logits,
                    teacher_2_logits=_t2_logits,
                    labels=labels,
                    rho=float(getattr(args.data, "projected_two_teacher_kd_rho", 0.5)),
                    temperature=float(getattr(args.data, "projected_two_teacher_kd_temperature", 2.0)),
                    alpha=float(getattr(args.data, "projected_two_teacher_kd_alpha", 0.5)),
                    chunk_size=int(getattr(args.data, "projected_two_teacher_kd_chunk_size", 128)),
                )

                del _t1_logits, _t2_logits
                bucket_kd_main_logits = None
                bucket_kd_gate_logits = None

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("PROJECTED TWO-TEACHER KD - First batch statistics")
                    logger.info(f"  Anchor teacher (t1, gate_idx={_p2tkd_gate_idx}): args.teacher_model_paths[{_p2tkd_gate_idx}]")
                    logger.info(f"  Broad  teacher (t2, main_idx={_p2tkd_main_idx}): args.teacher_model_paths[{_p2tkd_main_idx}]")
                    logger.info(f"  rho (mix weight on broad teacher in prob space): {getattr(args.data, 'projected_two_teacher_kd_rho', 0.5)}")
                    logger.info(f"  Temperature: {getattr(args.data, 'projected_two_teacher_kd_temperature', 2.0)}")
                    logger.info(f"  Alpha (KL vs CE weight): {getattr(args.data, 'projected_two_teacher_kd_alpha', 0.5)}")
                    logger.info("  Endpoints: rho=0 ≡ kd-rl(anchor),  rho=1 ≡ kd-rl(broad)")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_agreement_gated_two_teacher_kd:
                # Agreement-Gated Two-Teacher KD: same arithmetic-in-prob-space
                # mixture as the projected two-teacher path, but the broad
                # teacher's weight rho_t is binary and conditioned on
                # per-token teacher agreement (low JSD == high agreement):
                #   d_t   = JSD(p_anchor || p_broad)
                #   q     = per-batch JSD quantile at level jsd_quantile
                #   rho_t = rho_max  if d_t <= q  else 0
                #   p_T'  = (1 - rho_t) p_anchor + rho_t p_broad
                #   L     = (1-alpha) CE + alpha T^2 KL(p_T' || p_S)
                # Reuses the BAKD multi-teacher capture path
                # (bucket_kd_{main,gate}_logits).
                logits = model(input_ids, target=None)

                _a2tkd_main_idx = int(getattr(args.data, "agreement_gated_two_teacher_kd_main_idx", 1))
                _a2tkd_gate_idx = int(getattr(args.data, "agreement_gated_two_teacher_kd_gate_idx", 0))
                if _a2tkd_main_idx == _a2tkd_gate_idx:
                    raise RuntimeError(
                        "use_agreement_gated_two_teacher_kd: main_idx and gate_idx must differ "
                        "(main = broad teacher, gate = anchor teacher)."
                    )
                _t1_logits = bucket_kd_gate_logits   # anchor (1B-RLVR1)
                _t2_logits = bucket_kd_main_logits   # broad  (7B-Instruct)

                for _name in ("t1", "t2"):
                    _t = _t1_logits if _name == "t1" else _t2_logits
                    if _t.shape[-1] != logits.shape[-1]:
                        V_t = _t.shape[-1]
                        V_s = logits.shape[-1]
                        if V_t < V_s:
                            padding = torch.full(
                                (*_t.shape[:-1], V_s - V_t),
                                -1e10,
                                device=_t.device,
                                dtype=_t.dtype,
                            )
                            _t = torch.cat([_t, padding], dim=-1)
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[agree-2t-kd] Padded teacher {_name} logits {V_t} → {V_s}"
                                )
                        else:
                            _t = _t[..., :V_s]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[agree-2t-kd] Truncated teacher {_name} logits {V_t} → {V_s}"
                                )
                        if _name == "t1":
                            _t1_logits = _t
                        else:
                            _t2_logits = _t

                loss, batch_delta_stats = compute_agreement_gated_two_teacher_kd_loss(
                    student_logits=logits,
                    teacher_1_logits=_t1_logits,
                    teacher_2_logits=_t2_logits,
                    labels=labels,
                    rho_max=float(getattr(args.data, "agreement_gated_two_teacher_kd_rho_max", 0.25)),
                    jsd_quantile=float(getattr(args.data, "agreement_gated_two_teacher_kd_jsd_quantile", 0.25)),
                    temperature=float(getattr(args.data, "agreement_gated_two_teacher_kd_temperature", 2.0)),
                    alpha=float(getattr(args.data, "agreement_gated_two_teacher_kd_alpha", 0.5)),
                    chunk_size=int(getattr(args.data, "agreement_gated_two_teacher_kd_chunk_size", 128)),
                )

                del _t1_logits, _t2_logits
                bucket_kd_main_logits = None
                bucket_kd_gate_logits = None

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("AGREEMENT-GATED TWO-TEACHER KD - First batch statistics")
                    logger.info(f"  Anchor teacher (t1, gate_idx={_a2tkd_gate_idx}): args.teacher_model_paths[{_a2tkd_gate_idx}]")
                    logger.info(f"  Broad  teacher (t2, main_idx={_a2tkd_main_idx}): args.teacher_model_paths[{_a2tkd_main_idx}]")
                    logger.info(f"  rho_max:      {getattr(args.data, 'agreement_gated_two_teacher_kd_rho_max', 0.25)}")
                    logger.info(f"  jsd_quantile: {getattr(args.data, 'agreement_gated_two_teacher_kd_jsd_quantile', 0.25)}")
                    logger.info(f"  Temperature:  {getattr(args.data, 'agreement_gated_two_teacher_kd_temperature', 2.0)}")
                    logger.info(f"  Alpha (KL vs CE weight): {getattr(args.data, 'agreement_gated_two_teacher_kd_alpha', 0.5)}")
                    logger.info("  Gate: rho_t = rho_max on tokens with JSD(p_anchor,p_broad) <= per-batch q-quantile, else 0")
                    logger.info("  Endpoints: q=1 ≡ projected-2t(ρ=ρ_max);  q=0 or ρ_max=0 ≡ kd-rl(anchor)")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_competence_routed_two_teacher_kd:
                # Competence-Routed Two-Teacher KD: same arithmetic-in-prob-
                # space mixture as projected / agreement-gated 2t, but the
                # per-token mixing weight ρ_t on the broad teacher is gated
                # on **teacher-likelihood advantage on the observed gold
                # token** (untempered) rather than on per-token teacher
                # agreement (JSD).
                #   D1  ('token_advantage'):
                #     ρ_t = rho_max  if log p_2(y_t) > log p_1(y_t) else 0
                #   D3b ('sequence_advantage'):
                #     A_s = mean_t [ log p_2(y_t) - log p_1(y_t) ] over seq s
                #     ρ_s = rho_max  if A_s in top-(sequence_top_frac) of batch else 0
                # Reuses the BAKD multi-teacher capture path.
                logits = model(input_ids, target=None)

                _crt2tkd_main_idx = int(getattr(args.data, "competence_routed_two_teacher_kd_main_idx", 1))
                _crt2tkd_gate_idx = int(getattr(args.data, "competence_routed_two_teacher_kd_gate_idx", 0))
                if _crt2tkd_main_idx == _crt2tkd_gate_idx:
                    raise RuntimeError(
                        "use_competence_routed_two_teacher_kd: main_idx and gate_idx must differ "
                        "(main = broad teacher, gate = anchor teacher)."
                    )
                _t1_logits = bucket_kd_gate_logits   # anchor (1B-RLVR1)
                _t2_logits = bucket_kd_main_logits   # broad  (7B-Instruct)

                for _name in ("t1", "t2"):
                    _t = _t1_logits if _name == "t1" else _t2_logits
                    if _t.shape[-1] != logits.shape[-1]:
                        V_t = _t.shape[-1]
                        V_s = logits.shape[-1]
                        if V_t < V_s:
                            padding = torch.full(
                                (*_t.shape[:-1], V_s - V_t),
                                -1e10,
                                device=_t.device,
                                dtype=_t.dtype,
                            )
                            _t = torch.cat([_t, padding], dim=-1)
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[crt-2t-kd] Padded teacher {_name} logits {V_t} → {V_s}"
                                )
                        else:
                            _t = _t[..., :V_s]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[crt-2t-kd] Truncated teacher {_name} logits {V_t} → {V_s}"
                                )
                        if _name == "t1":
                            _t1_logits = _t
                        else:
                            _t2_logits = _t

                _crt2tkd_gate_type = str(getattr(args.data, "competence_routed_two_teacher_kd_gate_type", "token_advantage"))
                loss, batch_delta_stats = compute_competence_routed_two_teacher_kd_loss(
                    student_logits=logits,
                    teacher_1_logits=_t1_logits,
                    teacher_2_logits=_t2_logits,
                    labels=labels,
                    gate_type=_crt2tkd_gate_type,
                    rho_max=float(getattr(args.data, "competence_routed_two_teacher_kd_rho_max", 0.125)),
                    sequence_top_frac=float(getattr(args.data, "competence_routed_two_teacher_kd_sequence_top_frac", 0.25)),
                    temperature=float(getattr(args.data, "competence_routed_two_teacher_kd_temperature", 2.0)),
                    alpha=float(getattr(args.data, "competence_routed_two_teacher_kd_alpha", 0.5)),
                    chunk_size=int(getattr(args.data, "competence_routed_two_teacher_kd_chunk_size", 128)),
                )

                del _t1_logits, _t2_logits
                bucket_kd_main_logits = None
                bucket_kd_gate_logits = None

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("COMPETENCE-ROUTED TWO-TEACHER KD - First batch statistics")
                    logger.info(f"  Anchor teacher (t1, gate_idx={_crt2tkd_gate_idx}): args.teacher_model_paths[{_crt2tkd_gate_idx}]")
                    logger.info(f"  Broad  teacher (t2, main_idx={_crt2tkd_main_idx}): args.teacher_model_paths[{_crt2tkd_main_idx}]")
                    logger.info(f"  gate_type:         {_crt2tkd_gate_type}")
                    logger.info(f"  rho_max:           {getattr(args.data, 'competence_routed_two_teacher_kd_rho_max', 0.125)}")
                    logger.info(f"  sequence_top_frac: {getattr(args.data, 'competence_routed_two_teacher_kd_sequence_top_frac', 0.25)}  (only used when gate_type='sequence_advantage')")
                    logger.info(f"  Temperature:       {getattr(args.data, 'competence_routed_two_teacher_kd_temperature', 2.0)}")
                    logger.info(f"  Alpha (KL vs CE):  {getattr(args.data, 'competence_routed_two_teacher_kd_alpha', 0.5)}")
                    logger.info("  Gate signal: untempered log p_broad(y_t) vs log p_anchor(y_t)")
                    logger.info("  Target:      p_T' = (1-ρ_t)·p_anchor + ρ_t·p_broad  (at T)")
                    logger.info("  Loss:        L = (1-α)·CE + α·T²·KL(p_T' || p_S)")
                    if _crt2tkd_gate_type == "token_advantage":
                        logger.info("  D1 mode: ρ_t = rho_max on tokens where p_broad(y*) > p_anchor(y*), else 0")
                    else:
                        logger.info(f"  D3b mode: ρ_s = rho_max on top {getattr(args.data, 'competence_routed_two_teacher_kd_sequence_top_frac', 0.25):.2f} of sequences by A_s, else 0")
                        logger.info("    Endpoints: sequence_top_frac=1.0 ⇒ projected-2t(ρ=rho_max); sequence_top_frac=0.0 or rho_max=0 ⇒ kd-rl(anchor)")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_intersection_projected_kd:
                # Rank-Compatible Intersection-Projected KD. Per token:
                #   A_t = topK(p_anchor,t),  B_t = topK(p_broad,t),  C_t = A_t ∩ B_t
                #   if |C_t| < min_intersection_size: kd-rl(anchor) over full vocab at T
                #   else: KL( renormalized p_broad,T over C_t  ||  renormalized p_S,T over C_t )
                # Reuses the BAKD multi-teacher capture path; t1=anchor, t2=broad.
                logits = model(input_ids, target=None)

                _ipkd_main_idx = int(getattr(args.data, "intersection_projected_kd_main_idx", 1))
                _ipkd_gate_idx = int(getattr(args.data, "intersection_projected_kd_gate_idx", 0))
                if _ipkd_main_idx == _ipkd_gate_idx:
                    raise RuntimeError(
                        "use_intersection_projected_kd: main_idx and gate_idx must differ "
                        "(main = broad teacher, gate = anchor teacher)."
                    )
                _t1_logits = bucket_kd_gate_logits   # anchor (e.g. 1B-RLVR1) — compatibility filter
                _t2_logits = bucket_kd_main_logits   # broad  (e.g. 7B-Instruct) — ranking source

                for _name in ("t1", "t2"):
                    _t = _t1_logits if _name == "t1" else _t2_logits
                    if _t.shape[-1] != logits.shape[-1]:
                        V_t = _t.shape[-1]
                        V_s = logits.shape[-1]
                        if V_t < V_s:
                            padding = torch.full(
                                (*_t.shape[:-1], V_s - V_t),
                                -1e10,
                                device=_t.device,
                                dtype=_t.dtype,
                            )
                            _t = torch.cat([_t, padding], dim=-1)
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[intersection-proj-kd] Padded teacher {_name} logits {V_t} → {V_s}"
                                )
                        else:
                            _t = _t[..., :V_s]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[intersection-proj-kd] Truncated teacher {_name} logits {V_t} → {V_s}"
                                )
                        if _name == "t1":
                            _t1_logits = _t
                        else:
                            _t2_logits = _t

                loss, batch_delta_stats = compute_intersection_projected_kd_loss(
                    student_logits=logits,
                    teacher_anchor_logits=_t1_logits,
                    teacher_broad_logits=_t2_logits,
                    labels=labels,
                    topk=int(getattr(args.data, "intersection_projected_kd_topk", 20)),
                    min_intersection_size=int(getattr(args.data, "intersection_projected_kd_min_intersection_size", 5)),
                    temperature=float(getattr(args.data, "intersection_projected_kd_temperature", 1.25)),
                    alpha=float(getattr(args.data, "intersection_projected_kd_alpha", 0.5)),
                    chunk_size=int(getattr(args.data, "intersection_projected_kd_chunk_size", 128)),
                )

                del _t1_logits, _t2_logits
                bucket_kd_main_logits = None
                bucket_kd_gate_logits = None

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("INTERSECTION-PROJECTED KD - First batch statistics")
                    logger.info(f"  Anchor teacher (t1, gate_idx={_ipkd_gate_idx}): args.teacher_model_paths[{_ipkd_gate_idx}]   [compatibility filter]")
                    logger.info(f"  Broad  teacher (t2, main_idx={_ipkd_main_idx}): args.teacher_model_paths[{_ipkd_main_idx}]   [ranking source]")
                    logger.info(f"  topK:                  {getattr(args.data, 'intersection_projected_kd_topk', 20)}")
                    logger.info(f"  min_intersection_size: {getattr(args.data, 'intersection_projected_kd_min_intersection_size', 5)}")
                    logger.info(f"  Temperature:           {getattr(args.data, 'intersection_projected_kd_temperature', 1.25)}")
                    logger.info(f"  Alpha (KL vs CE):      {getattr(args.data, 'intersection_projected_kd_alpha', 0.5)}")
                    logger.info("  Per-token branch:")
                    logger.info("    |C_t| >= min  →  KL(renorm(p_broad,T)[C_t] || renorm(p_S,T)[C_t])  (intersection branch)")
                    logger.info("    |C_t| < min   →  KL(p_anchor,T || p_S,T)  over full vocab        (1B-RLVR fallback)")
                    logger.info("  Loss:  L = (1-α)·CE + α·T²·mean_t[token_kl]")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_entropy_gated_kd:
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10, device=teacher_logits_for_kd.device, dtype=teacher_logits_for_kd.dtype
                            )
                            teacher_logits_for_kd = torch.cat([teacher_logits_for_kd, padding], dim=-1)
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]

                loss, batch_delta_stats = compute_entropy_gated_kd_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=getattr(args.data, 'egkd_temperature', 2.0),
                    gate_tau=getattr(args.data, 'egkd_gate_tau', 0.0),
                )

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("ENTROPY-GATED KD - First batch statistics")
                    logger.info(f"  Teacher model: {args.teacher_model_path}")
                    logger.info(f"  Temperature: {getattr(args.data, 'egkd_temperature', 2.0)}")
                    logger.info(f"  Gate tau: {batch_delta_stats.get('egkd/gate_tau', 'auto')}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif getattr(args.data, 'use_selective_kd', False) and teacher_model is not None:
                # Selective KD: same compute + total KL gradient norm as kd-rl, but
                # the per-token KL is weighted by a score that focuses gradient on
                # tokens where the student gains most from the teacher signal.
                # See compute_selective_kd_loss for the exact loss form.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out
                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_t = teacher_logits_for_kd.shape[-1]
                        V_s = logits.shape[-1]
                        if V_t < V_s:
                            pad = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_s - V_t),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat([teacher_logits_for_kd, pad], dim=-1)
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_s]

                score_type = getattr(args.data, 'selective_kd_score', 'student_entropy')
                weight_mode = getattr(args.data, 'selective_kd_weight_mode', 'soft')
                # `reference_excess_loss` does NOT require the teacher's full
                # distribution to compute the score (only gold-NLL of teacher
                # vs student, both already cheap in the chunked path).
                use_teacher_signal = score_type in ('js_div', 'max_norm', 'teacher_confidence')

                # Domain ids for per-domain normalization (and per-domain
                # telemetry). Cheap to build — same helper used by the
                # kd_diagnostics path.
                _sel_domain_normalize = bool(
                    getattr(args.data, 'selective_kd_domain_normalize', False)
                )
                _sel_emit_domain_stats = bool(
                    getattr(args.data, 'selective_kd_emit_domain_stats', False)
                )
                _sel_domain_ids = None
                _sel_id_to_name: Dict[int, str] = {}
                if _sel_domain_normalize or _sel_emit_domain_stats:
                    _sel_domain_ids, _sel_id_to_name = _build_token_domain_ids(
                        labels=labels,
                        cu_seqlens=batch_cu_seqlens,
                        doc_sources=batch_doc_sources,
                        domain_labels=_source_labels,
                    )
                    # FAIL-LOUD GUARD: if the user requested dom-norm /
                    # per-domain telemetry but `_build_token_domain_ids`
                    # could not resolve any domain attribution (neither
                    # `doc_sources` nor `_source_labels` reached us), the
                    # selective-KD recipe will silently degrade to the
                    # global-median path — exactly the failure mode that
                    # took down 7223992_46 (s3-js-guarded-domnorm):
                    # `sel/domain_norm_enabled` logged as 0 in wandb
                    # despite `selective_kd_domain_normalize=true` in the
                    # launch EXTRA_ARGS, because this branch was a
                    # silent warning + fallback at the time. Refuse to
                    # train rather than mislabel another run.
                    if _sel_domain_ids is None:
                        raise RuntimeError(
                            "Selective KD: selective_kd_domain_normalize/"
                            "selective_kd_emit_domain_stats was set, but "
                            "no per-token domain attribution is available "
                            "(batch has no `doc_sources` and the loader "
                            "produced no `_source_labels`). Ensure the "
                            "source-labeled (online) loader is in use — "
                            "see `_use_source_labeled_loader` in train.py."
                        )
                    # Belt-and-braces: even if `_sel_domain_ids` is
                    # non-None, an empty `id_to_name` means
                    # `_build_token_domain_ids` produced a zeroed
                    # tensor without any actual source attribution.
                    # That would silently lump everything into a single
                    # "dom0" bucket and make per-domain stats useless.
                    if not _sel_id_to_name:
                        raise RuntimeError(
                            "Selective KD: per-token domain_ids tensor "
                            "was built but `id_to_name` is empty. This "
                            "indicates the batch has no resolvable "
                            "domain sources; per-domain normalization "
                            "and telemetry would be meaningless. Refusing "
                            "to train; check the data loader configuration."
                        )

                _sel_gold_guard_delta = getattr(args.data, 'selective_kd_gold_guard_delta', None)
                if isinstance(_sel_gold_guard_delta, str) and _sel_gold_guard_delta.lower() in ("", "none", "null"):
                    _sel_gold_guard_delta = None

                loss, batch_delta_stats = compute_selective_kd_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=getattr(args.data, 'kl_temperature', 2.0),
                    alpha=getattr(args.data, 'kl_alpha', 0.5),
                    score=score_type,
                    weight_mode=weight_mode,
                    top_k_frac=getattr(args.data, 'selective_kd_top_k_frac', 0.5),
                    soft_w_max=getattr(args.data, 'selective_kd_soft_w_max', 3.0),
                    rescale_to_uniform=getattr(args.data, 'selective_kd_rescale_to_uniform', True),
                    use_teacher_signal=use_teacher_signal,
                    gold_guard_delta=_sel_gold_guard_delta,
                    gold_guard_w_min=float(getattr(args.data, 'selective_kd_gold_guard_w_min', 0.0)),
                    domain_ids=_sel_domain_ids,
                    domain_normalize=_sel_domain_normalize,
                    domain_norm_min_tokens=int(getattr(args.data, 'selective_kd_domain_norm_min_tokens', 32)),
                    id_to_name=(_sel_id_to_name if _sel_id_to_name else None),
                    full_ce_weight=bool(getattr(args.data, 'selective_kd_full_ce_weight', False)),
                )
                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("SELECTIVE KD - First batch statistics")
                    logger.info(f"  Teacher model: {args.teacher_model_path}")
                    logger.info(f"  Score: {score_type}")
                    logger.info(f"  Weight mode: {weight_mode}")
                    logger.info(f"  Temperature: {getattr(args.data, 'kl_temperature', 2.0)}")
                    logger.info(f"  Alpha: {getattr(args.data, 'kl_alpha', 0.5)}")
                    logger.info(f"  Rescale-to-uniform: {getattr(args.data, 'selective_kd_rescale_to_uniform', True)}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)

                    # ── Step-0 sanity assertions ──────────────────────
                    # Each assertion targets one of the failure modes we
                    # diagnosed from s3-js-guarded-domnorm / s3-rho1-soft.
                    # If any assertion fires, fail-fast with a clear
                    # error so the run does not consume ~9h of GPU only
                    # to produce another GSM8K collapse.
                    _gp_frac = float(batch_delta_stats.get("sel/guard_pass_frac", 1.0))
                    _kd_nz = float(batch_delta_stats.get("sel/kd_nonzero_frac", 1.0))
                    _w_max = float(batch_delta_stats.get("sel/weight_max", 0.0))
                    _w_min_cfg = float(
                        getattr(args.data, "selective_kd_gold_guard_w_min", 0.0)
                    )
                    _guard_enabled = bool(batch_delta_stats.get("sel/guard_enabled", 0.0))
                    _rescale = bool(
                        getattr(args.data, "selective_kd_rescale_to_uniform", True)
                    )
                    _w_max_cfg = float(
                        getattr(args.data, "selective_kd_soft_w_max", 3.0)
                    )

                    # (1) If gold-guard is on, the guard should pass the
                    # vast majority of tokens at step 0. A delta=1.0
                    # guard should pass ~0.99 since the teacher should
                    # beat the student by ≥1 nat on essentially every
                    # gold token at init. A delta=0.0 strict guard
                    # already passed only 0.65 in the failed run, which
                    # is itself an out-of-distribution data point — any
                    # config that comes in below 0.85 is breaking the
                    # dense-anchor rule from token 1.
                    if _guard_enabled and _gp_frac < 0.85:
                        raise RuntimeError(
                            f"Selective KD step-0 sanity: "
                            f"sel/guard_pass_frac={_gp_frac:.4f} < 0.85 "
                            f"on the first batch. With "
                            f"gold_guard_delta="
                            f"{getattr(args.data, 'selective_kd_gold_guard_delta', None)!r}"
                            f", more than 15% of tokens are excluded "
                            f"from KD at initialization — this is the "
                            f"failure mode that caused the s3-js-"
                            f"guarded-domnorm GSM8K collapse. Use a "
                            f"larger guard_delta (e.g. 1.0) or disable "
                            f"the guard."
                        )

                    # (2) If a non-zero floor is configured, every token
                    # MUST retain at least floor-weight KD. Anything
                    # else means the floor logic is broken upstream.
                    if _w_min_cfg > 0.0 and _kd_nz < 0.999:
                        raise RuntimeError(
                            f"Selective KD step-0 sanity: "
                            f"selective_kd_gold_guard_w_min={_w_min_cfg:.3f} > 0 "
                            f"but sel/kd_nonzero_frac={_kd_nz:.4f} < 1.0. "
                            f"With a positive floor every valid token "
                            f"must carry KD gradient; some tokens have "
                            f"w=0 anyway. The floor logic in "
                            f"compute_selective_kd_loss is broken — do "
                            f"not train."
                        )

                    # (3) When rescale_to_uniform is disabled, weights
                    # must respect the soft_w_max clamp. The 9.31 vs
                    # soft_w_max=3.0 gap we saw in the failed run was
                    # entirely from post-clamp rescaling; this
                    # assertion guarantees weights stay bounded when the
                    # user opts out of rescaling.
                    _eps = 1e-3
                    if (not _rescale) and _w_max > _w_max_cfg + _eps:
                        raise RuntimeError(
                            f"Selective KD step-0 sanity: "
                            f"rescale_to_uniform=false but "
                            f"sel/weight_max={_w_max:.4f} > "
                            f"soft_w_max={_w_max_cfg:.4f} (+1e-3 eps). "
                            f"Weights are exceeding the clamp without "
                            f"a rescaling pass — likely a clamp ordering "
                            f"bug in compute_selective_kd_loss. Do not "
                            f"train."
                        )

                    logger.info(
                        f"  ✓ step-0 sanity OK: "
                        f"guard_pass={_gp_frac:.3f}, "
                        f"kd_nonzero={_kd_nz:.3f}, "
                        f"weight_max={_w_max:.3f}"
                    )
                    logger.info("=" * 60)
            elif use_kl_distillation:
                # Classic Knowledge Distillation: KL divergence between student and teacher
                logits = model(input_ids, target=None)

                # Recompute teacher logits for KL distillation (need full logits, not just NLL)
                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    # Handle vocab size mismatch between student and teacher
                    # Student vocab may be padded to multiple_of (e.g., 100278 → 100352)
                    # Teacher from HuggingFace has actual vocab size (100278)
                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]

                        if V_teacher < V_student:
                            # Pad teacher logits with very negative values (will have ~0 probability after softmax)
                            padding_size = V_student - V_teacher
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], padding_size),
                                -1e10,  # Very negative logits → 0 probability
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype
                            )
                            teacher_logits_for_kd = torch.cat([teacher_logits_for_kd, padding], dim=-1)
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(f"Padded teacher logits from {V_teacher} to {V_student} (student uses multiple_of padding)")
                        else:
                            # Truncate teacher logits if teacher is larger (shouldn't happen, but handle it)
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(f"Truncated teacher logits from {V_teacher} to {V_student}")

                loss, batch_delta_stats = compute_kl_distillation_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=getattr(args.data, 'kl_temperature', 2.0),
                    alpha=getattr(args.data, 'kl_alpha', 0.5),
                )

                if bool(getattr(args.data, 'kd_diagnostics_enabled', False)):
                    _kd_mask = (labels != -100).to(dtype=logits.dtype)
                    _kd_domain_ids, _kd_id_to_name = _build_token_domain_ids(
                        labels=labels,
                        cu_seqlens=batch_cu_seqlens,
                        doc_sources=batch_doc_sources,
                        domain_labels=_source_labels,
                    )
                    _kd_diag_stats = _compute_kd_concentration_diagnostics(
                        student_logits=logits,
                        teacher_logits=teacher_logits_for_kd,
                        mask=_kd_mask,
                        temperature=float(getattr(args.data, 'kl_temperature', 2.0)),
                        domain_ids=_kd_domain_ids,
                        id_to_name=_kd_id_to_name,
                        chunk_size=int(getattr(args.data, 'kd_diagnostics_chunk_size', 128)),
                    )
                    batch_delta_stats.update(_kd_diag_stats)

                del teacher_logits_for_kd  # Free memory

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("CLASSIC KNOWLEDGE DISTILLATION - First batch statistics")
                    logger.info(f"  Teacher model: {args.teacher_model_path}")
                    logger.info(f"  Temperature: {getattr(args.data, 'kl_temperature', 2.0)}")
                    logger.info(f"  Alpha (KL weight): {getattr(args.data, 'kl_alpha', 0.5)}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif use_projected_kd:
                # Projected-Teacher KD: distill from p_T' = (1-rho)*p_S_sg + rho*p_T
                # i.e. an arithmetic mixture in probability space (NOT log-space).
                # rho=1.0 is byte-identical to compute_kl_distillation_loss
                # (a self-check of the projected-KD pipeline). rho<1 moves the KD
                # target part of the way toward the (stop-grad) student, which is
                # the capacity-gap remedy this branch is here to test.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[projected-kd] Padded teacher logits {V_teacher} → {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[projected-kd] Truncated teacher logits {V_teacher} → {V_student}"
                                )

                loss, batch_delta_stats = compute_projected_kd_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    rho=float(getattr(args.data, 'projected_kd_rho', 0.5)),
                    temperature=float(getattr(args.data, 'projected_kd_temperature', 2.0)),
                    alpha=float(getattr(args.data, 'projected_kd_alpha', 0.5)),
                    chunk_size=int(getattr(args.data, 'projected_kd_chunk_size', 128)),
                )

                if bool(getattr(args.data, 'kd_diagnostics_enabled', False)):
                    # Reuse the kd_concentration diagnostics on the *raw* teacher
                    # vs. student — gives us the per-domain KL/JS/Hs concentration
                    # picture that the rho=1 (kd-rl7b) run would have seen.
                    _kd_mask = (labels != -100).to(dtype=logits.dtype)
                    _kd_domain_ids, _kd_id_to_name = _build_token_domain_ids(
                        labels=labels,
                        cu_seqlens=batch_cu_seqlens,
                        doc_sources=batch_doc_sources,
                        domain_labels=_source_labels,
                    )
                    _kd_diag_stats = _compute_kd_concentration_diagnostics(
                        student_logits=logits,
                        teacher_logits=teacher_logits_for_kd,
                        mask=_kd_mask,
                        temperature=float(getattr(args.data, 'projected_kd_temperature', 2.0)),
                        domain_ids=_kd_domain_ids,
                        id_to_name=_kd_id_to_name,
                        chunk_size=int(getattr(args.data, 'kd_diagnostics_chunk_size', 128)),
                    )
                    batch_delta_stats.update(_kd_diag_stats)

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("PROJECTED-TEACHER KD - First batch statistics")
                    logger.info(f"  Teacher model: {args.teacher_model_path}")
                    logger.info(f"  rho:         {getattr(args.data, 'projected_kd_rho', 0.5)}  "
                                f"(p_T' = (1-rho)*p_S_sg + rho*p_T, arithmetic)")
                    logger.info(f"  temperature: {getattr(args.data, 'projected_kd_temperature', 2.0)}")
                    logger.info(f"  alpha:       {getattr(args.data, 'projected_kd_alpha', 0.5)}  "
                                "(KL weight; CE weight = 1 - alpha)")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif use_reverse_kl_distillation:
                # Reverse-KL distillation (MiniLLM): minimize KL(p_S || p_T) instead
                # of KL(p_T || p_S). Mode-seeking divergence -- the principled
                # capacity-gap remedy when teacher >> student, because it lets the
                # student concentrate on the dominant teacher mode and ignore the
                # long tail of teacher mass that the student cannot represent.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[reverse-kl] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[reverse-kl] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_reverse_kl_distillation_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'reverse_kl_temperature', 2.0)),
                    alpha=float(getattr(args.data, 'reverse_kl_alpha', 0.5)),
                    chunk_size=int(getattr(args.data, 'reverse_kl_chunk_size', 128)),
                )

                if bool(getattr(args.data, 'kd_diagnostics_enabled', False)):
                    # Same per-domain KL/JS/Hs concentration diagnostics as
                    # kd-rl7b (computed on forward KL for interpretability;
                    # the loss optimizes the reverse direction).
                    _kd_mask = (labels != -100).to(dtype=logits.dtype)
                    _kd_domain_ids, _kd_id_to_name = _build_token_domain_ids(
                        labels=labels,
                        cu_seqlens=batch_cu_seqlens,
                        doc_sources=batch_doc_sources,
                        domain_labels=_source_labels,
                    )
                    _kd_diag_stats = _compute_kd_concentration_diagnostics(
                        student_logits=logits,
                        teacher_logits=teacher_logits_for_kd,
                        mask=_kd_mask,
                        temperature=float(getattr(args.data, 'reverse_kl_temperature', 2.0)),
                        domain_ids=_kd_domain_ids,
                        id_to_name=_kd_id_to_name,
                        chunk_size=int(getattr(args.data, 'kd_diagnostics_chunk_size', 128)),
                    )
                    batch_delta_stats.update(_kd_diag_stats)

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("REVERSE-KL DISTILLATION (MiniLLM-style) - First batch statistics")
                    logger.info(f"  Teacher model: {args.teacher_model_path}")
                    logger.info(f"  Loss:         L = (1-alpha)*CE + alpha*T^2*KL(p_S || p_T)")
                    logger.info(f"  temperature:  {getattr(args.data, 'reverse_kl_temperature', 2.0)}")
                    logger.info(f"  alpha:        {getattr(args.data, 'reverse_kl_alpha', 0.5)}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif use_rkl_fkl_mix_distillation:
                # Reverse-KL + forward-KL mixture (pure-KD, no CE term):
                #     L = T^2 * [ alpha * KL(p_S || p_T) + (1-alpha) * KL(p_T || p_S) ]
                # Motivation: revKL is mode-seeking (wins on MC / reasoning but
                # under-allocates mass to rare gold tokens, hurting open-ended
                # factual recall); a small fwdKL addback restores mass coverage
                # without re-introducing the CE term (which would pin the
                # student to potentially-noisy gold tokens). The two KL terms
                # share student_log_soft per chunk, so this is computed in a
                # single fused activation-checkpointed kernel — ~2x cheaper
                # than running revKL and fwdKL distillation passes separately.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-fkl-mix] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-fkl-mix] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_rkl_fkl_mix_distillation_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'rkl_fkl_mix_temperature', 2.0)),
                    alpha=float(getattr(args.data, 'rkl_fkl_mix_alpha', 0.5)),
                    chunk_size=int(getattr(args.data, 'rkl_fkl_mix_chunk_size', 128)),
                )

                if bool(getattr(args.data, 'kd_diagnostics_enabled', False)):
                    _kd_mask = (labels != -100).to(dtype=logits.dtype)
                    _kd_domain_ids, _kd_id_to_name = _build_token_domain_ids(
                        labels=labels,
                        cu_seqlens=batch_cu_seqlens,
                        doc_sources=batch_doc_sources,
                        domain_labels=_source_labels,
                    )
                    _kd_diag_stats = _compute_kd_concentration_diagnostics(
                        student_logits=logits,
                        teacher_logits=teacher_logits_for_kd,
                        mask=_kd_mask,
                        temperature=float(getattr(args.data, 'rkl_fkl_mix_temperature', 2.0)),
                        domain_ids=_kd_domain_ids,
                        id_to_name=_kd_id_to_name,
                        chunk_size=int(getattr(args.data, 'kd_diagnostics_chunk_size', 128)),
                    )
                    batch_delta_stats.update(_kd_diag_stats)

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("RKL+FKL PURE-KD MIX - First batch statistics")
                    logger.info(f"  Teacher model: {args.teacher_model_path}")
                    logger.info(f"  Loss:         L = T^2 * [alpha*KL(p_S||p_T) + (1-alpha)*KL(p_T||p_S)]")
                    logger.info(f"  temperature:  {getattr(args.data, 'rkl_fkl_mix_temperature', 2.0)}")
                    logger.info(f"  alpha:        {getattr(args.data, 'rkl_fkl_mix_alpha', 0.5)}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif use_rkl_with_gated_ce_distillation:
                # Reverse-KL purekd with sparse CE correction gated on teacher NLL:
                #     L = T^2 * KL(p_S || p_T) + lambda_ce * w_CE(n_t) * CE(y_t)
                #     n_t = -log p_T(y_t)
                #     w_CE(n_t) = clip((n_t - a) / (b - a), 0, 1)
                # Built off the diagnostic showing pure RKL (idx 100) inherits ~0
                # teacher factual knowledge on C1 (teacher confident on gold,
                # student wrong) and CE-on recipes recover ~+2-3 pt NQ mainly via
                # bucket-A preservation + rare-corpus-token exposure. The gate
                # fires CE on exactly the tokens where the teacher would
                # underweight the corpus token, leaving 80% of tokens (mostly
                # stylistic / well-predicted) on pure RKL.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-gce] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-gce] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_rkl_with_gated_ce_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'rkl_gce_temperature', 2.0)),
                    chunk_size=int(getattr(args.data, 'rkl_gce_chunk_size', 128)),
                    lambda_ce=float(getattr(args.data, 'rkl_gce_lambda_ce', 0.1)),
                    nll_threshold_a=float(getattr(args.data, 'rkl_gce_nll_threshold_a', 2.0)),
                    nll_threshold_b=float(getattr(args.data, 'rkl_gce_nll_threshold_b', 6.0)),
                )

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("RKL + GATED-CE - First batch statistics")
                    logger.info(f"  Teacher model: {args.teacher_model_path}")
                    logger.info(f"  Loss:          L = T^2 * KL(p_S||p_T) + lambda_ce * w_CE(n_t) * CE")
                    logger.info(f"  temperature:   {getattr(args.data, 'rkl_gce_temperature', 2.0)}")
                    logger.info(f"  lambda_ce:     {getattr(args.data, 'rkl_gce_lambda_ce', 0.1)}")
                    logger.info(f"  gate a (Q80):  {getattr(args.data, 'rkl_gce_nll_threshold_a', 2.0)}")
                    logger.info(f"  gate b (Q95):  {getattr(args.data, 'rkl_gce_nll_threshold_b', 6.0)}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif use_rkl_with_lowent_ce_distillation:
                # Reverse-KL purekd with binary CE correction gated on teacher entropy:
                #     L = T^2 * KL(p_S || p_T) + lambda_ce * 1[H(p_T,t) <= tau] * CE(y_t)
                # Tests the "fire CE only at one-true-answer positions" hypothesis.
                # Low teacher entropy at T=1 includes some factual continuations
                # (NASA, 1969) but also a lot of tokenizer/syntactic determinism
                # (mid-BPE continuations, list numbering, sentence-end punct).
                # At lambda_ce=1.0 the effective CE pressure across the corpus
                # ~= gate_on_frac, typically ~0.2 — comparable to a small uniform
                # alpha but concentrated on the determined-position tail.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-lent] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-lent] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_rkl_with_lowent_ce_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'rkl_lent_temperature', 2.0)),
                    chunk_size=int(getattr(args.data, 'rkl_lent_chunk_size', 128)),
                    lambda_ce=float(getattr(args.data, 'rkl_lent_lambda_ce', 1.0)),
                    entropy_threshold=float(getattr(args.data, 'rkl_lent_entropy_threshold', 0.029)),
                )

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("RKL + LOW-ENTROPY GATED-CE - First batch statistics")
                    logger.info(f"  Teacher model: {args.teacher_model_path}")
                    logger.info(f"  Loss:          L = T^2 * KL(p_S||p_T) + lambda_ce * 1[H(p_T) <= tau] * CE")
                    logger.info(f"  temperature:        {getattr(args.data, 'rkl_lent_temperature', 2.0)}")
                    logger.info(f"  lambda_ce:          {getattr(args.data, 'rkl_lent_lambda_ce', 1.0)}")
                    logger.info(f"  entropy_threshold:  {getattr(args.data, 'rkl_lent_entropy_threshold', 0.029)}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif use_rkl_with_source_ce_distillation:
                # Reverse-KL purekd with CE only on tokens from configured sources:
                #     L = T^2 * KL(p_S || p_T) + lambda_ce * 1[source(t) in target_sources] * CE(y_t)
                # RKL runs on EVERY token; CE runs ONLY on tokens from sources in the
                # target_sources list (e.g. ["wiki_shuffled"]). The teacher-student delta
                # diagnostic motivates this: NTP's NQ-F1 advantage (19.8 vs idx 100's 15.6)
                # plausibly comes from direct corpus exposure to factual content in wiki.
                # Wiki is 7.11% of corpus → with lambda_ce=2.0, effective CE pressure ~14%
                # (vs canonloss's 50%), concentrated on the most NQ-shaped tokens.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-src] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-src] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                _rkl_src_target = getattr(args.data, 'rkl_src_target_sources', ['wiki_shuffled'])
                if isinstance(_rkl_src_target, str):
                    _rkl_src_target = [_rkl_src_target]
                loss, batch_delta_stats = compute_rkl_with_source_ce_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    target_sources=_rkl_src_target,
                    cu_seqlens=batch_cu_seqlens,
                    doc_sources=batch_doc_sources,
                    temperature=float(getattr(args.data, 'rkl_src_temperature', 2.0)),
                    chunk_size=int(getattr(args.data, 'rkl_src_chunk_size', 128)),
                    lambda_ce=float(getattr(args.data, 'rkl_src_lambda_ce', 2.0)),
                )

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("RKL + SOURCE-GATED-CE - First batch statistics")
                    logger.info(f"  Teacher model:    {args.teacher_model_path}")
                    logger.info(f"  Loss:             L = T^2 * KL(p_S||p_T) + lambda_ce * 1[src in target] * CE")
                    logger.info(f"  temperature:      {getattr(args.data, 'rkl_src_temperature', 2.0)}")
                    logger.info(f"  lambda_ce:        {getattr(args.data, 'rkl_src_lambda_ce', 2.0)}")
                    logger.info(f"  target_sources:   {_rkl_src_target}")
                    logger.info(f"  batch_cu_seqlens type:    {type(batch_cu_seqlens).__name__}  len: {len(batch_cu_seqlens) if batch_cu_seqlens is not None else 'None'}")
                    logger.info(f"  batch_doc_sources type:   {type(batch_doc_sources).__name__}  len: {len(batch_doc_sources) if batch_doc_sources is not None else 'None'}")
                    if batch_doc_sources is not None and len(batch_doc_sources) > 0:
                        logger.info(f"  batch_doc_sources[0]:     {batch_doc_sources[0]!r}")
                        if batch_doc_sources[0] is not None and len(batch_doc_sources[0]) > 0:
                            logger.info(f"  sample src key (b=0,d=0): {batch_doc_sources[0][0]!r}  type: {type(batch_doc_sources[0][0]).__name__}")
                    for k, v in batch_delta_stats.items():
                        if isinstance(v, (int, float)):
                            logger.info(f"  {k}: {v:.4f}")
                        else:
                            logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_rkl_with_teacher_disagree_ce_distillation:
                # Reverse-KL purekd with CE on the (low RKL × high teacher-NLL) cell:
                #     L = T^2 * KL(p_S || p_T) + lambda_ce * 1[RKL<=rkl_thr ∧ t_nll>tnll_thr] * CE
                # CE fires ONLY where the teacher confidently rejects the corpus
                # token AND the student has already matched that rejection. In
                # this regime RKL has no remaining corrective signal for the
                # corpus token; CE supplies the only direct anchoring signal.
                # See diag_three_gates-7935809.out for cell composition.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-td] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-td] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_rkl_with_teacher_disagree_ce_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'rkl_td_temperature', 2.0)),
                    chunk_size=int(getattr(args.data, 'rkl_td_chunk_size', 128)),
                    lambda_ce=float(getattr(args.data, 'rkl_td_lambda_ce', 0.25)),
                    rkl_threshold=float(getattr(args.data, 'rkl_td_rkl_threshold', 0.1368)),
                    tnll_threshold=float(getattr(args.data, 'rkl_td_tnll_threshold', 1.8601)),
                )

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("RKL + TEACHER-DISAGREE-GATED-CE - First batch statistics")
                    logger.info(f"  Teacher model:    {args.teacher_model_path}")
                    logger.info(f"  Loss:             L = T^2 * KL(p_S||p_T) + lambda_ce * 1[RKL<=rkl_thr ∧ t_nll>tnll_thr] * CE")
                    logger.info(f"  temperature:      {getattr(args.data, 'rkl_td_temperature', 2.0)}")
                    logger.info(f"  lambda_ce:        {getattr(args.data, 'rkl_td_lambda_ce', 0.25)}")
                    logger.info(f"  rkl_threshold:    {getattr(args.data, 'rkl_td_rkl_threshold', 0.1368)}  (corpus p33)")
                    logger.info(f"  tnll_threshold:   {getattr(args.data, 'rkl_td_tnll_threshold', 1.8601)}  (corpus p66)")
                    for k, v in batch_delta_stats.items():
                        if isinstance(v, (int, float)):
                            logger.info(f"  {k}: {v:.4f}")
                        else:
                            logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_rkl_with_teacher_fail_ce_distillation:
                # Reverse-KL purekd + CE only where teacher itself fails the gold:
                #     L = T^2 * KL(p_S||p_T) + lambda_ce * 1[t_nll>tnll_thr] * CE
                # Single-axis variant of rkl_td: drops the RKL<=thr AND clause
                # and gates on teacher gold NLL alone. Targets the teacher's
                # closed-book recall ceiling — on tokens where the teacher can't
                # predict the gold, RKL has no useful signal toward the corpus
                # token; CE supplies the only direct anchoring signal.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-tfail] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-tfail] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_rkl_with_teacher_fail_ce_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'rkl_tfail_temperature', 2.0)),
                    chunk_size=int(getattr(args.data, 'rkl_tfail_chunk_size', 128)),
                    lambda_ce=float(getattr(args.data, 'rkl_tfail_lambda_ce', 0.5)),
                    tnll_threshold=float(getattr(args.data, 'rkl_tfail_tnll_threshold', 2.5)),
                )

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("RKL + TEACHER-FAIL-GATED-CE - First batch statistics")
                    logger.info(f"  Teacher model:    {args.teacher_model_path}")
                    logger.info(f"  Loss:             L = T^2 * KL(p_S||p_T) + lambda_ce * 1[t_nll>tnll_thr] * CE")
                    logger.info(f"  temperature:      {getattr(args.data, 'rkl_tfail_temperature', 2.0)}")
                    logger.info(f"  lambda_ce:        {getattr(args.data, 'rkl_tfail_lambda_ce', 0.5)}")
                    logger.info(f"  tnll_threshold:   {getattr(args.data, 'rkl_tfail_tnll_threshold', 2.5)}  (~corpus p80)")
                    for k, v in batch_delta_stats.items():
                        if isinstance(v, (int, float)):
                            logger.info(f"  {k}: {v:.4f}")
                        else:
                            logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_rkl_with_teacher_success_ce_distillation:
                # Inverse of tfail: CE only where teacher SUCCEEDS on the gold
                # token (idx 137). Complementary gate to idx 126 at same
                # threshold — tests whether "teacher-gold compatibility" is
                # the right selector vs 126's "fix teacher mistakes."
                #     L = T^2 * KL(p_S||p_T) + lambda_ce * 1[t_nll<tnll_thr] * CE
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-tsuccess] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-tsuccess] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_rkl_with_teacher_success_ce_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'rkl_tsuccess_temperature', 2.0)),
                    chunk_size=int(getattr(args.data, 'rkl_tsuccess_chunk_size', 128)),
                    lambda_ce=float(getattr(args.data, 'rkl_tsuccess_lambda_ce', 0.5)),
                    tnll_threshold=float(getattr(args.data, 'rkl_tsuccess_tnll_threshold', 2.5)),
                )

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("RKL + TEACHER-SUCCESS-GATED-CE - First batch statistics")
                    logger.info(f"  Teacher model:    {args.teacher_model_path}")
                    logger.info(f"  Loss:             L = T^2 * KL(p_S||p_T) + lambda_ce * 1[t_nll<tnll_thr] * CE")
                    logger.info(f"  temperature:      {getattr(args.data, 'rkl_tsuccess_temperature', 2.0)}")
                    logger.info(f"  lambda_ce:        {getattr(args.data, 'rkl_tsuccess_lambda_ce', 0.5)}")
                    logger.info(f"  tnll_threshold:   {getattr(args.data, 'rkl_tsuccess_tnll_threshold', 2.5)}  (inverse of idx 126; ~corpus 0..p80 = ~80% of tokens)")
                    for k, v in batch_delta_stats.items():
                        if isinstance(v, (int, float)):
                            logger.info(f"  {k}: {v:.4f}")
                        else:
                            logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_rkl_with_topk_gap_ce_distillation:
                # Reverse-KL purekd everywhere + CE on top-K% by acquisition gap:
                #     score_t = s_nll_t - t_nll_t = log p_T(y_t) - log p_S(y_t)
                #     m_t     = 1[ t in top-K% by score within batch ]
                #     L = T^2 * KL(p_S||p_T) + lambda_ce * m_t * CE(y_t)
                # RKL gradient stays on EVERY token (preserving i100's macro
                # ceiling); CE adds selective NQ-targeted pressure on the
                # top-K cell where the student lags the teacher on the gold
                # token. Unlike Rho-1 (which masks 40% of all gradient),
                # selection here only affects the extra CE term.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-topkgap] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-topkgap] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_rkl_with_topk_gap_ce_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'rkl_topkgap_temperature', 2.0)),
                    chunk_size=int(getattr(args.data, 'rkl_topkgap_chunk_size', 128)),
                    lambda_ce=float(getattr(args.data, 'rkl_topkgap_lambda_ce', 0.5)),
                    topk_frac=float(getattr(args.data, 'rkl_topkgap_topk_frac', 0.20)),
                )

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("RKL + TOP-K ACQUISITION-GAP CE - First batch statistics")
                    logger.info(f"  Teacher model:    {args.teacher_model_path}")
                    logger.info(f"  Loss:             L = T^2 * KL(p_S||p_T) + lambda_ce * 1[top-K gap] * CE")
                    logger.info(f"  temperature:      {getattr(args.data, 'rkl_topkgap_temperature', 2.0)}")
                    logger.info(f"  lambda_ce:        {getattr(args.data, 'rkl_topkgap_lambda_ce', 0.5)}")
                    logger.info(f"  topk_frac:        {getattr(args.data, 'rkl_topkgap_topk_frac', 0.20)}")
                    for k, v in batch_delta_stats.items():
                        if isinstance(v, (int, float)):
                            logger.info(f"  {k}: {v:.4f}")
                        else:
                            logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_rkl_with_topkgap_replace_ce_distillation:
                # Reverse-KL OR CE per token, mutually exclusive, partitioned
                # by gap (idx 131). Restatement of idx 125 where CE REPLACES
                # rather than augments RKL on top-K tokens.
                #     score_t = s_nll_t - t_nll_t = log p_T(y_t) - log p_S(y_t)
                #     m_t     = 1[ t in top-K% by score within batch ]
                #     L = (1 - m_t) * T^2 * KL(p_S||p_T) + m_t * CE(y_t)
                # Tests whether committing per-token to one supervision
                # (gold or teacher) yields a better Pareto than 125's additive
                # mix, which fires both losses on selected tokens.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-topkgap-replace] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-topkgap-replace] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_rkl_with_topk_gap_ce_replace_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'rkl_topkgap_replace_temperature', 2.0)),
                    chunk_size=int(getattr(args.data, 'rkl_topkgap_replace_chunk_size', 128)),
                    topk_frac=float(getattr(args.data, 'rkl_topkgap_replace_topk_frac', 0.20)),
                )

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("RKL OR CE (top-K gap partition) - First batch statistics")
                    logger.info(f"  Teacher model:    {args.teacher_model_path}")
                    logger.info(f"  Loss:             L = (1-m_t) * T^2 * KL(p_S||p_T) + m_t * CE")
                    logger.info(f"  temperature:      {getattr(args.data, 'rkl_topkgap_replace_temperature', 2.0)}")
                    logger.info(f"  topk_frac:        {getattr(args.data, 'rkl_topkgap_replace_topk_frac', 0.20)}")
                    for k, v in batch_delta_stats.items():
                        if isinstance(v, (int, float)):
                            logger.info(f"  {k}: {v:.4f}")
                        else:
                            logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_rkl_with_uniform_ce_distillation:
                # Reverse-KL purekd on every token + uniform additive CE on
                # every token (idx 132/133/134/135). "No selection" control
                # for the selective-CE family.
                #     L = T^2 * KL(p_S||p_T) + lambda_ce * CE(y)
                # Sweep lambda_ce ∈ {0.1, 0.3, 0.5, 1.0} to map the uniform-CE
                # curve directly against the selective-CE Pareto points.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-uniform] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-uniform] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_rkl_with_uniform_ce_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'rkl_uniform_temperature', 2.0)),
                    chunk_size=int(getattr(args.data, 'rkl_uniform_chunk_size', 128)),
                    lambda_ce=float(getattr(args.data, 'rkl_uniform_lambda_ce', 0.3)),
                )

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("RKL + UNIFORM CE - First batch statistics")
                    logger.info(f"  Teacher model:    {args.teacher_model_path}")
                    logger.info(f"  Loss:             L = T^2 * KL(p_S||p_T) + lambda_ce * CE(y)")
                    logger.info(f"  temperature:      {getattr(args.data, 'rkl_uniform_temperature', 2.0)}")
                    logger.info(f"  lambda_ce:        {getattr(args.data, 'rkl_uniform_lambda_ce', 0.3)}")
                    for k, v in batch_delta_stats.items():
                        if isinstance(v, (int, float)):
                            logger.info(f"  {k}: {v:.4f}")
                        else:
                            logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_rkl_entropy_gated_distillation:
                # Entropy-gated RKL (idx 138). CE on every token; RKL fires only
                # on the lowest-entropy gate_quantile of teacher tokens.
                #     tau = per-batch quantile(H(p_T), gate_quantile)
                #     L = lambda_ce * CE(y) + 1[H_t <= tau] * T^2 * KL(p_S||p_T)
                # First recipe to shape the RKL term (every other recipe gates
                # CE). Motivated by the teacher-entropy-by-source diagnostic
                # (math median H = 0.79; factual median H = 7.3-7.6 at T=2.0):
                # RKL on factual tokens drags the student toward an uninformed
                # teacher distribution. Gating RKL by teacher entropy preserves
                # the math/structural signal while letting CE drive NQ.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-entgate] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-entgate] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_entropy_gated_rkl_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'rkl_entgate_temperature', 2.0)),
                    chunk_size=int(getattr(args.data, 'rkl_entgate_chunk_size', 128)),
                    lambda_ce=float(getattr(args.data, 'rkl_entgate_lambda_ce', 1.0)),
                    gate_quantile=float(getattr(args.data, 'rkl_entgate_quantile', 0.30)),
                )

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("RKL + ENTROPY-GATED RKL (idx 138) - First batch statistics")
                    logger.info(f"  Teacher model:    {args.teacher_model_path}")
                    logger.info(f"  Loss:             L = lambda_ce * CE(y) + 1[H_t<=tau] * T^2 * KL(p_S||p_T)")
                    logger.info(f"  temperature:      {getattr(args.data, 'rkl_entgate_temperature', 2.0)}")
                    logger.info(f"  lambda_ce:        {getattr(args.data, 'rkl_entgate_lambda_ce', 1.0)}")
                    logger.info(f"  gate_quantile:    {getattr(args.data, 'rkl_entgate_quantile', 0.30)}")
                    for k, v in batch_delta_stats.items():
                        if isinstance(v, (int, float)):
                            logger.info(f"  {k}: {v:.4f}")
                        else:
                            logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_fkl_entropy_gated_distillation:
                # FKL-entgate ablation (idx 140). Identical to idx 138 but FKL
                # = KL(p_T || p_S) instead of RKL. Tests whether the entgate
                # win is specific to RKL or generalizes to FKL under the same
                # masking. Matches kd-rl baseline FKL form (T^2 * KL(p_T||p_S),
                # both softened at T).
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[fkl-entgate] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[fkl-entgate] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_entropy_gated_fkl_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'fkl_entgate_temperature', 2.0)),
                    chunk_size=int(getattr(args.data, 'fkl_entgate_chunk_size', 128)),
                    lambda_ce=float(getattr(args.data, 'fkl_entgate_lambda_ce', 1.0)),
                    gate_quantile=float(getattr(args.data, 'fkl_entgate_quantile', 0.30)),
                )

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("FKL + ENTROPY-GATED FKL (idx 140) - First batch statistics")
                    logger.info(f"  Teacher model:    {args.teacher_model_path}")
                    logger.info(f"  Loss:             L = lambda_ce * CE(y) + 1[H_t<=tau] * T^2 * KL(p_T||p_S)")
                    logger.info(f"  temperature:      {getattr(args.data, 'fkl_entgate_temperature', 2.0)}")
                    logger.info(f"  lambda_ce:        {getattr(args.data, 'fkl_entgate_lambda_ce', 1.0)}")
                    logger.info(f"  gate_quantile:    {getattr(args.data, 'fkl_entgate_quantile', 0.30)}")
                    for k, v in batch_delta_stats.items():
                        if isinstance(v, (int, float)):
                            logger.info(f"  {k}: {v:.4f}")
                        else:
                            logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_entropy_band_kl_distillation:
                # Complement-band KD (idx 151). CE on every token; RKL on
                # bottom low_quantile of teacher entropy; FKL on top high_quantile;
                # middle band gets CE only. Tests whether high-entropy teacher
                # distributions contain useful "soft target" info that CE alone
                # misses, while keeping RKL pressure on the low-entropy/math
                # tokens where idx 139's win is located.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[entband] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[entband] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_entropy_band_kl_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'entband_temperature', 2.0)),
                    chunk_size=int(getattr(args.data, 'entband_chunk_size', 128)),
                    lambda_ce=float(getattr(args.data, 'entband_lambda_ce', 1.0)),
                    low_band_quantile=float(getattr(args.data, 'entband_low_quantile', 0.30)),
                    high_band_quantile=float(getattr(args.data, 'entband_high_quantile', 0.30)),
                )

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("ENTROPY-BAND KD (idx 151) - First batch statistics")
                    logger.info(f"  Teacher model:    {args.teacher_model_path}")
                    logger.info(f"  Loss: L = lambda_ce*CE(y) + T^2*[mean_low(RKL) + mean_high(FKL)]")
                    logger.info(f"  temperature:      {getattr(args.data, 'entband_temperature', 2.0)}")
                    logger.info(f"  lambda_ce:        {getattr(args.data, 'entband_lambda_ce', 1.0)}")
                    logger.info(f"  low_band_q:       {getattr(args.data, 'entband_low_quantile', 0.30)}  (RKL fires here)")
                    logger.info(f"  high_band_q:      {getattr(args.data, 'entband_high_quantile', 0.30)}  (FKL fires here)")
                    for k, v in batch_delta_stats.items():
                        if isinstance(v, (int, float)):
                            logger.info(f"  {k}: {v:.4f}")
                        else:
                            logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_rkl_random_gated_distillation:
                # Random-mask RKL ablation (idx 141). Identical to idx 138 but
                # the binary mask is a per-batch uniform-random subset of valid
                # tokens (top-k over uniform scores), matched in size to the
                # entgate fire count. Controls for "is the win from entropy
                # specifically, or any 30% sparsity?" Mask seed = base_seed +
                # global_step + rank, reproducible across restarts.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-randmask] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-randmask] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_random_gated_rkl_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'rkl_randmask_temperature', 2.0)),
                    chunk_size=int(getattr(args.data, 'rkl_randmask_chunk_size', 128)),
                    lambda_ce=float(getattr(args.data, 'rkl_randmask_lambda_ce', 1.0)),
                    gate_quantile=float(getattr(args.data, 'rkl_randmask_quantile', 0.30)),
                    base_seed=int(getattr(args.data, 'rkl_randmask_base_seed', 0)),
                    global_step=int(train_state.step),
                    rank=int(get_global_rank()),
                )

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("RKL + RANDOM-MASK-GATED RKL (idx 141) - First batch statistics")
                    logger.info(f"  Teacher model:    {args.teacher_model_path}")
                    logger.info(f"  Loss:             L = lambda_ce * CE(y) + m_t * T^2 * KL(p_S||p_T), m_t ~ random top-k")
                    logger.info(f"  temperature:      {getattr(args.data, 'rkl_randmask_temperature', 2.0)}")
                    logger.info(f"  lambda_ce:        {getattr(args.data, 'rkl_randmask_lambda_ce', 1.0)}")
                    logger.info(f"  gate_quantile:    {getattr(args.data, 'rkl_randmask_quantile', 0.30)}")
                    logger.info(f"  base_seed:        {getattr(args.data, 'rkl_randmask_base_seed', 0)}")
                    for k, v in batch_delta_stats.items():
                        if isinstance(v, (int, float)):
                            logger.info(f"  {k}: {v:.4f}")
                        else:
                            logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_rkl_student_entropy_gated_distillation:
                # Student-entropy-gated RKL (idx 143). Mirror of idx 139 but
                # the gate selects the top-K most uncertain *student* tokens
                # instead of the bottom-K most confident *teacher* tokens.
                #     tau = quantile(H(p_S), 1 - gate_quantile)
                #     L = lambda_ce * CE(y) + 1[H_S_t >= tau] * T^2 * KL(p_S||p_T)
                # Active-learning framing: distill where student is uncertain.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-stentgate] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-stentgate] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_student_entropy_gated_rkl_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'rkl_stentgate_temperature', 2.0)),
                    chunk_size=int(getattr(args.data, 'rkl_stentgate_chunk_size', 128)),
                    lambda_ce=float(getattr(args.data, 'rkl_stentgate_lambda_ce', 1.0)),
                    gate_quantile=float(getattr(args.data, 'rkl_stentgate_quantile', 0.30)),
                )

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("RKL + STUDENT-ENTROPY-GATED RKL (idx 143) - First batch statistics")
                    logger.info(f"  Teacher model:    {args.teacher_model_path}")
                    logger.info(f"  Loss:             L = lambda_ce * CE(y) + 1[H_S_t>=tau] * T^2 * KL(p_S||p_T)")
                    logger.info(f"  temperature:      {getattr(args.data, 'rkl_stentgate_temperature', 2.0)}")
                    logger.info(f"  lambda_ce:        {getattr(args.data, 'rkl_stentgate_lambda_ce', 1.0)}")
                    logger.info(f"  gate_quantile:    {getattr(args.data, 'rkl_stentgate_quantile', 0.30)}")
                    for k, v in batch_delta_stats.items():
                        if isinstance(v, (int, float)):
                            logger.info(f"  {k}: {v:.4f}")
                        else:
                            logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_rkl_entropy_switched_distillation:
                # Switched entropy-gated RKL (idx 148). Partitions CE and RKL
                # by the same teacher-entropy gate (no overlap):
                #     m_t = 1[H(p_T)_t <= tau]
                #     L = (1 - m_t) * lambda_ce * CE + m_t * T^2 * KL(p_S||p_T)
                # Tests whether the gold-token CE on math-flavored tokens is
                # helping (idx 139) or just diluting the RKL signal there
                # (idx 142 partially supports the dilution hypothesis at the
                # global level). Per-token switch isolates the effect.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-entswitch] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-entswitch] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_entropy_switched_rkl_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'rkl_entswitch_temperature', 2.0)),
                    chunk_size=int(getattr(args.data, 'rkl_entswitch_chunk_size', 128)),
                    lambda_ce=float(getattr(args.data, 'rkl_entswitch_lambda_ce', 1.0)),
                    gate_quantile=float(getattr(args.data, 'rkl_entswitch_quantile', 0.30)),
                )

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("RKL + ENTROPY-SWITCHED RKL/CE (idx 148) - First batch statistics")
                    logger.info(f"  Teacher model:    {args.teacher_model_path}")
                    logger.info(f"  Loss:             L = (1-m_t)*lambda_ce*CE + m_t*T^2*KL(p_S||p_T)")
                    logger.info(f"  temperature:      {getattr(args.data, 'rkl_entswitch_temperature', 2.0)}")
                    logger.info(f"  lambda_ce:        {getattr(args.data, 'rkl_entswitch_lambda_ce', 1.0)}")
                    logger.info(f"  gate_quantile:    {getattr(args.data, 'rkl_entswitch_quantile', 0.30)}")
                    for k, v in batch_delta_stats.items():
                        if isinstance(v, (int, float)):
                            logger.info(f"  {k}: {v:.4f}")
                        else:
                            logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_rkl_with_topkgap_schedule_ce_distillation:
                # Reverse-KL purekd everywhere + CE on top-K% by acquisition
                # gap, where K is a TIME-VARYING fraction (idx 130).
                #     K(step) = K_min + (K_max - K_min) * (step / total_steps)^α
                # With defaults (min=0, max=1, α=2), the gate starts at 0% CE
                # and ramps quadratically to 100% by end of training.
                # Mirrors idx 112's "RKL early → canon-late" finding with a
                # smooth schedule instead of a discrete switch. Reuses the
                # bit-identical topkgap kernel (compute_rkl_with_topk_gap_ce_loss);
                # only the topk_frac argument is replaced with a step-dependent value.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-topkgap-schedule] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-topkgap-schedule] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                # Compute step-dependent topk_frac
                _k_min = float(getattr(args.data, 'rkl_topkgap_schedule_min', 0.0))
                _k_max = float(getattr(args.data, 'rkl_topkgap_schedule_max', 1.0))
                _k_exp = float(getattr(args.data, 'rkl_topkgap_schedule_exponent', 2.0))
                _total_steps = max(int(args.steps), 1)
                _progress = min(max(train_state.step / _total_steps, 0.0), 1.0)
                _topk_frac_step = _k_min + (_k_max - _k_min) * (_progress ** _k_exp)

                loss, batch_delta_stats = compute_rkl_with_topk_gap_ce_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'rkl_topkgap_temperature', 2.0)),
                    chunk_size=int(getattr(args.data, 'rkl_topkgap_chunk_size', 128)),
                    lambda_ce=float(getattr(args.data, 'rkl_topkgap_lambda_ce', 0.5)),
                    topk_frac=_topk_frac_step,
                )

                # Surface the scheduled topk_frac in metrics so the trajectory
                # is visible in the training log/wandb.
                batch_delta_stats["kd_rkl_topkgap_schedule/topk_frac_step"] = _topk_frac_step
                batch_delta_stats["kd_rkl_topkgap_schedule/progress"] = _progress

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("RKL + TOP-K ACQUISITION-GAP CE (SCHEDULED) - First batch statistics")
                    logger.info(f"  Teacher model:    {args.teacher_model_path}")
                    logger.info(f"  Loss:             L = T^2 * KL(p_S||p_T) + lambda_ce * 1[top-K gap] * CE")
                    logger.info(f"  Schedule:         K(t) = {_k_min} + ({_k_max}-{_k_min})*(t/{args.steps})^{_k_exp}")
                    logger.info(f"  K @ start:        {_k_min:.4f}")
                    logger.info(f"  K @ midpoint:     {_k_min + (_k_max-_k_min)*(0.5**_k_exp):.4f}")
                    logger.info(f"  K @ end:          {_k_max:.4f}")
                    logger.info(f"  temperature:      {getattr(args.data, 'rkl_topkgap_temperature', 2.0)}")
                    logger.info(f"  lambda_ce:        {getattr(args.data, 'rkl_topkgap_lambda_ce', 0.5)}")
                    for k, v in batch_delta_stats.items():
                        if isinstance(v, (int, float)):
                            logger.info(f"  {k}: {v:.4f}")
                        else:
                            logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_rkl_with_gradagree_gap_ce_distillation:
                # Reverse-KL purekd everywhere + CE on top-K% by gradient
                # agreement × acquisition gap (idx 129).
                #     score_t = max(cos(g_CE,t, g_RKL,t), 0) * max(gap_t, 0)
                #     m_t     = 1[ t in top-K% by score AND cos > 0 AND gap > 0 ]
                #     L = T^2 * KL(p_S||p_T) + lambda_ce * m_t * CE(y_t)
                # Same RKL kernel and CE normalization as idx 125; only the
                # gate criterion changes from gap-only to (cos > 0) × gap.
                # Tests whether suppressing CE on gradient-conflict tokens
                # (the §2.5 low h_t × high t_nll toxic cell) improves the
                # macro/NQ Pareto vs gap-only selection.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-gradagree] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rkl-gradagree] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_rkl_with_gradagree_gap_ce_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'rkl_gradagree_gap_temperature', 2.0)),
                    chunk_size=int(getattr(args.data, 'rkl_gradagree_gap_chunk_size', 128)),
                    lambda_ce=float(getattr(args.data, 'rkl_gradagree_gap_lambda_ce', 0.5)),
                    topk_frac=float(getattr(args.data, 'rkl_gradagree_gap_topk_frac', 0.20)),
                )

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("RKL + GRADIENT-AGREEMENT × ACQUISITION-GAP CE - First batch")
                    logger.info(f"  Teacher model:    {args.teacher_model_path}")
                    logger.info(f"  Loss:             L = T^2 * KL(p_S||p_T) + lambda_ce * m * CE,")
                    logger.info(f"                        m = top-K[ relu(cos)*relu(gap) ], cos>0 ∩ gap>0")
                    logger.info(f"  temperature:      {getattr(args.data, 'rkl_gradagree_gap_temperature', 2.0)}")
                    logger.info(f"  lambda_ce:        {getattr(args.data, 'rkl_gradagree_gap_lambda_ce', 0.5)}")
                    logger.info(f"  topk_frac:        {getattr(args.data, 'rkl_gradagree_gap_topk_frac', 0.20)}")
                    for k, v in batch_delta_stats.items():
                        if isinstance(v, (int, float)):
                            logger.info(f"  {k}: {v:.4f}")
                        else:
                            logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            elif use_rho1_kd_distillation:
                # Rho-1-style top-k selection applied to canonloss (FKL + CE):
                #     score(t) = n_S(y_t) - n_T(y_t) = log p_T - log p_S
                #     mask(t)  = 1 if t in top-topk_frac% by score within batch
                #     L = mean_t mask(t) * [alpha*T^2*KL(p_T||p_S) + (1-alpha)*CE]
                # Reference model = teacher (single extra model). Selects
                # "learnable but unlearned" tokens by Rho-1's logic; tokens
                # outside the top-k contribute zero gradient. See
                # compute_rho1_kd_loss.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rho1-kd] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[rho1-kd] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_rho1_kd_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'rho1_kd_temperature', 2.0)),
                    chunk_size=int(getattr(args.data, 'rho1_kd_chunk_size', 128)),
                    alpha=float(getattr(args.data, 'rho1_kd_alpha', 0.5)),
                    topk_frac=float(getattr(args.data, 'rho1_kd_topk_frac', 0.6)),
                )

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("RHO-1 KD - First batch statistics")
                    logger.info(f"  Teacher model: {args.teacher_model_path}")
                    logger.info(f"  Loss:          mean_sel [alpha*T^2*KL(p_T||p_S) + (1-alpha)*CE]")
                    logger.info(f"  Selection:     top {getattr(args.data, 'rho1_kd_topk_frac', 0.6)*100:.0f}% by n_S - n_T")
                    logger.info(f"  temperature:   {getattr(args.data, 'rho1_kd_temperature', 2.0)}")
                    logger.info(f"  alpha:         {getattr(args.data, 'rho1_kd_alpha', 0.5)}")
                    logger.info(f"  topk_frac:     {getattr(args.data, 'rho1_kd_topk_frac', 0.6)}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif use_akl_distillation:
                # Adaptive-KL (AKL) pure-KD distillation:
                #     L = alpha * T^2 * mean_t [w_FKL_t * KL(p_T||p_S)_t + w_RKL_t * KL(p_S||p_T)_t]
                # where w_FKL_t and w_RKL_t are per-token weights derived from
                # teacher head/tail disagreement (head = smallest teacher
                # top-prob set with cum mass >= mu). Per-token generalisation
                # of the rkl_fkl_mix family (idx 103/104) with adaptive,
                # data-dependent alpha. Same fused activation-checkpointed
                # kernel as rkl_fkl_mix; sort over vocab to build the head
                # mask adds modest extra cost. No CE term.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[akl] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[akl] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_akl_distillation_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(getattr(args.data, 'akl_temperature', 2.0)),
                    mu=float(getattr(args.data, 'akl_mu', 0.5)),
                    alpha=float(getattr(args.data, 'akl_alpha', 1.0)),
                    chunk_size=int(getattr(args.data, 'akl_chunk_size', 128)),
                )

                if bool(getattr(args.data, 'kd_diagnostics_enabled', False)):
                    _kd_mask = (labels != -100).to(dtype=logits.dtype)
                    _kd_domain_ids, _kd_id_to_name = _build_token_domain_ids(
                        labels=labels,
                        cu_seqlens=batch_cu_seqlens,
                        doc_sources=batch_doc_sources,
                        domain_labels=_source_labels,
                    )
                    _kd_diag_stats = _compute_kd_concentration_diagnostics(
                        student_logits=logits,
                        teacher_logits=teacher_logits_for_kd,
                        mask=_kd_mask,
                        temperature=float(getattr(args.data, 'akl_temperature', 2.0)),
                        domain_ids=_kd_domain_ids,
                        id_to_name=_kd_id_to_name,
                        chunk_size=int(getattr(args.data, 'kd_diagnostics_chunk_size', 128)),
                    )
                    batch_delta_stats.update(_kd_diag_stats)

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("ADAPTIVE-KL (AKL) PURE-KD - First batch statistics")
                    logger.info(f"  Teacher model: {args.teacher_model_path}")
                    logger.info(f"  Loss:         L = alpha * T^2 * E[w_FKL*KL(T||S) + w_RKL*KL(S||T)]")
                    logger.info(f"  temperature:  {getattr(args.data, 'akl_temperature', 2.0)}")
                    logger.info(f"  mu:           {getattr(args.data, 'akl_mu', 0.5)}")
                    logger.info(f"  alpha:        {getattr(args.data, 'akl_alpha', 1.0)}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif use_geometric_kd:
                # Student-Anchored Geometric KD: target = renorm(p_S_sg^(1-λ) · p_T^λ)
                # The teacher forward is rerun here for full logits (the gold-NLL
                # block above only kept per-token NLL); same pattern as use_kl_distillation.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], V_student - V_teacher),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[geometric-kd] Padded teacher logits {V_teacher} → {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[..., :V_student]
                            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                                logger.info(
                                    f"[geometric-kd] Truncated teacher logits {V_teacher} → {V_student}"
                                )

                loss, batch_delta_stats = compute_geometric_kd_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    lambda_mix=float(getattr(args.data, 'geometric_kd_lambda', 0.5)),
                    temperature=float(getattr(args.data, 'geometric_kd_temperature', 2.0)),
                    alpha=float(getattr(args.data, 'geometric_kd_alpha', 1.0)),
                    chunk_size=int(getattr(args.data, 'geometric_kd_chunk_size', 128)),
                    gold_gate=bool(getattr(args.data, 'geometric_kd_gold_gate', False)),
                )

                del teacher_logits_for_kd

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("STUDENT-ANCHORED GEOMETRIC KD - First batch statistics")
                    logger.info(f"  Teacher model: {args.teacher_model_path}")
                    logger.info(f"  Lambda: {getattr(args.data, 'geometric_kd_lambda', 0.5)}")
                    logger.info(f"  Temperature: {getattr(args.data, 'geometric_kd_temperature', 2.0)}")
                    logger.info(f"  Alpha (KL weight; CE coef = 1): {getattr(args.data, 'geometric_kd_alpha', 1.0)}")
                    logger.info(f"  Gold-advantage gate: {getattr(args.data, 'geometric_kd_gold_gate', False)}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif use_rho1 and teacher_logprobs is not None:
                logits = model(input_ids, target=None)
                _validate_distinctive_prereqs()

                loss, batch_delta_stats = compute_rho1_loss(
                    logits=logits,
                    labels=labels,
                    teacher_logprobs=teacher_logprobs,
                    select_ratio=getattr(args.data, 'rho1_select_ratio', 0.6),
                    domain_labels=_source_labels,
                    domain_normalize=getattr(args.data, "rho1_domain_normalize", False),
                    domain_norm_min_tokens=getattr(args.data, "rho1_domain_norm_min_tokens", 32),
                    domain_norm_eps=getattr(args.data, "rho1_domain_norm_eps", 1e-6),
                    domain_norm_clip=getattr(args.data, "rho1_domain_norm_clip", 0.0),
                    stratified_by_domain=getattr(args.data, "rho1_stratified_by_domain", False),
                    stratified_exact_budget=getattr(args.data, "rho1_stratified_exact_budget", True),
                    cu_seqlens=batch_cu_seqlens,
                    doc_sources=batch_doc_sources,
                    all_expert_nlls=all_expert_nlls,
                    distinctive_advantage=getattr(args.data, "rho1_distinctive_advantage", False),
                    distinctive_alpha=getattr(args.data, "rho1_distinctive_alpha", 3.0),
                    distinctive_margin=getattr(args.data, "rho1_distinctive_margin", 0.1),
                    distinctive_normalize=getattr(args.data, "rho1_distinctive_normalize", True),
                    distinctive_eps=getattr(args.data, "rho1_distinctive_eps", 1e-6),
                    entropy_gate_frac=getattr(args.data, "rho1_entropy_gate_frac", 0.0),
                    entropy_chunk_size=getattr(args.data, "rho1_entropy_chunk_size", 256),
                    excess_mass_select=getattr(args.data, "rho1_excess_mass_select", False),
                    excess_mass_p=getattr(args.data, "rho1_excess_mass_p", 0.9),
                    excess_mass_min_ratio=getattr(args.data, "rho1_excess_mass_min_ratio", 0.1),
                    excess_mass_max_ratio=getattr(args.data, "rho1_excess_mass_max_ratio", 0.8),
                    excess_mass_cap_quantile=getattr(args.data, "rho1_excess_mass_cap_quantile", 0.99),
                    excess_mass_min_positive_mass=getattr(args.data, "rho1_excess_mass_min_positive_mass", 1e-8),
                    excess_mass_rising_floor=getattr(args.data, "rho1_excess_mass_rising_floor", False),
                    excess_mass_floor_start_ratio=getattr(args.data, "rho1_excess_mass_floor_start_ratio", 0.2),
                    excess_mass_floor_end_ratio=getattr(args.data, "rho1_excess_mass_floor_end_ratio", 0.8),
                    current_step=train_state.step,
                    total_steps=args.steps,
                    capgap_enable=getattr(args.data, "rho1_capgap_enable", False),
                    capgap_lambda=getattr(args.data, "rho1_capgap_lambda", 0.5),
                    capgap_main_idx=getattr(args.data, "rho1_capgap_main_expert_idx", 1),
                    capgap_gate_idx=getattr(args.data, "rho1_capgap_gate_expert_idx", 0),
                    capgap_hard_x_gate=getattr(args.data, "rho1_capgap_hard_x_gate", False),
                    capgap_hard_x_gate_margin=getattr(args.data, "rho1_capgap_hard_x_gate_margin", 0.0),
                )

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("RHO-1 SELECTIVE LANGUAGE MODELING - First batch statistics")
                    logger.info(f"  Reference model: {'on-the-fly' if teacher_model is not None else 'cached'}")
                    logger.info(f"  select_ratio: {getattr(args.data, 'rho1_select_ratio', 0.6)}")
                    _attr_mode = "per-token (cu_seqlens)" if batch_doc_sources is not None else "per-sequence (legacy)"
                    logger.info(f"  domain_attribution: {_attr_mode}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {_fmt_metric(v)}")
                    logger.info("=" * 60)
            elif use_best_expert_seq_kd and teacher_logprobs is not None and best_expert_logits is not None:
                logits = model(input_ids, target=None,
                               cu_seqlens=cu_seqlens_tensor, max_seqlen=max_seqlen)

                loss, batch_delta_stats = compute_best_expert_seq_kd_loss(
                    student_logits=logits,
                    best_expert_logits=best_expert_logits,
                    labels=labels,
                    temperature=getattr(args.data, 'best_expert_kd_temperature', 2.0),
                    alpha=getattr(args.data, 'best_expert_kd_alpha', 0.5),
                )
                # Free expert logits as soon as the loss is computed.
                best_expert_logits = None

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("BEST-EXPERT-SEQ KD - First batch statistics")
                    logger.info(f"  Routing:     sequence-level (per-seq argmin mean NLL)")
                    logger.info(f"  Loss:        alpha*KL(expert||student)*tau^2 + (1-alpha)*CE")
                    logger.info(f"  temperature: {getattr(args.data, 'best_expert_kd_temperature', 2.0)}")
                    logger.info(f"  alpha:       {getattr(args.data, 'best_expert_kd_alpha', 0.5)}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif use_best_expert_seq_rho1 and teacher_logprobs is not None:
                logits = model(input_ids, target=None,
                               cu_seqlens=cu_seqlens_tensor, max_seqlen=max_seqlen)
                _validate_distinctive_prereqs()
                loss, batch_delta_stats = compute_rho1_loss(
                    logits=logits,
                    labels=labels,
                    teacher_logprobs=teacher_logprobs,   # per-token NLL from sequence-routed expert
                    select_ratio=best_expert_seq_rho1_select_ratio,
                    domain_labels=_source_labels,
                    domain_normalize=getattr(args.data, "rho1_domain_normalize", False),
                    domain_norm_min_tokens=getattr(args.data, "rho1_domain_norm_min_tokens", 32),
                    domain_norm_eps=getattr(args.data, "rho1_domain_norm_eps", 1e-6),
                    domain_norm_clip=getattr(args.data, "rho1_domain_norm_clip", 0.0),
                    stratified_by_domain=getattr(args.data, "rho1_stratified_by_domain", False),
                    stratified_exact_budget=getattr(args.data, "rho1_stratified_exact_budget", True),
                    cu_seqlens=batch_cu_seqlens,
                    doc_sources=batch_doc_sources,
                    all_expert_nlls=all_expert_nlls,
                    distinctive_advantage=getattr(args.data, "rho1_distinctive_advantage", False),
                    distinctive_alpha=getattr(args.data, "rho1_distinctive_alpha", 3.0),
                    distinctive_margin=getattr(args.data, "rho1_distinctive_margin", 0.1),
                    distinctive_normalize=getattr(args.data, "rho1_distinctive_normalize", True),
                    distinctive_eps=getattr(args.data, "rho1_distinctive_eps", 1e-6),
                    entropy_gate_frac=getattr(args.data, "rho1_entropy_gate_frac", 0.0),
                    entropy_chunk_size=getattr(args.data, "rho1_entropy_chunk_size", 256),
                    excess_mass_select=getattr(args.data, "rho1_excess_mass_select", False),
                    excess_mass_p=getattr(args.data, "rho1_excess_mass_p", 0.9),
                    excess_mass_min_ratio=getattr(args.data, "rho1_excess_mass_min_ratio", 0.1),
                    excess_mass_max_ratio=getattr(args.data, "rho1_excess_mass_max_ratio", 0.8),
                    excess_mass_cap_quantile=getattr(args.data, "rho1_excess_mass_cap_quantile", 0.99),
                    excess_mass_min_positive_mass=getattr(args.data, "rho1_excess_mass_min_positive_mass", 1e-8),
                    excess_mass_rising_floor=getattr(args.data, "rho1_excess_mass_rising_floor", False),
                    excess_mass_floor_start_ratio=getattr(args.data, "rho1_excess_mass_floor_start_ratio", 0.2),
                    excess_mass_floor_end_ratio=getattr(args.data, "rho1_excess_mass_floor_end_ratio", 0.8),
                    current_step=train_state.step,
                    total_steps=args.steps,
                    capgap_enable=getattr(args.data, "rho1_capgap_enable", False),
                    capgap_lambda=getattr(args.data, "rho1_capgap_lambda", 0.5),
                    capgap_main_idx=getattr(args.data, "rho1_capgap_main_expert_idx", 1),
                    capgap_gate_idx=getattr(args.data, "rho1_capgap_gate_expert_idx", 0),
                    capgap_hard_x_gate=getattr(args.data, "rho1_capgap_hard_x_gate", False),
                    capgap_hard_x_gate_margin=getattr(args.data, "rho1_capgap_hard_x_gate_margin", 0.0),
                )
                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("BEST-EXPERT-SEQ RHO-1 - First batch statistics")
                    logger.info(f"  Routing: sequence-level (per-seq argmin mean NLL on prefix)")
                    logger.info(f"  Loss:    Rho-1 top-{best_expert_seq_rho1_select_ratio:.0%} token selection")
                    _attr_mode = "per-token (cu_seqlens)" if batch_doc_sources is not None else "per-sequence (legacy)"
                    logger.info(f"  domain_attribution: {_attr_mode}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif use_best_expert_rho1 and getattr(args.data, 'rho1_expert_stratified', False) and all_expert_nlls is not None:
                logits = model(input_ids, target=None,
                               cu_seqlens=cu_seqlens_tensor, max_seqlen=max_seqlen)
                _full_budget = getattr(args.data, 'rho1_expert_stratified_full_budget', False)
                loss, batch_delta_stats = compute_rho1_expert_stratified_loss(
                    logits=logits,
                    labels=labels,
                    all_expert_nlls=all_expert_nlls,
                    select_ratio=getattr(args.data, 'rho1_select_ratio', 0.6),
                    full_budget=_full_budget,
                    cu_seqlens=batch_cu_seqlens,
                    doc_sources=batch_doc_sources,
                    domain_labels=_source_labels,
                )
                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    _sr = getattr(args.data, 'rho1_select_ratio', 0.6)
                    _E = all_expert_nlls.size(0)
                    logger.info("=" * 60)
                    logger.info("EXPERT-STRATIFIED RHO-1 - First batch statistics")
                    logger.info(f"  Mode: each expert independently selects top-k tokens")
                    logger.info(f"  Experts: {_E}")
                    logger.info(f"  select_ratio: {_sr}")
                    logger.info(f"  full_budget: {_full_budget}")
                    logger.info(f"  budget_per_expert: {_sr if _full_budget else _sr / _E:.3f}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif use_best_expert_rho1 and teacher_logprobs is not None:
                logits = model(input_ids, target=None,
                               cu_seqlens=cu_seqlens_tensor, max_seqlen=max_seqlen)
                _validate_distinctive_prereqs()
                _is_round_robin = getattr(args.data, 'rho1_round_robin_experts', False)
                _is_staged = bool(getattr(args.data, 'rho1_stage_boundaries', None))
                loss, batch_delta_stats = compute_rho1_loss(
                    logits=logits,
                    labels=labels,
                    teacher_logprobs=teacher_logprobs,
                    select_ratio=getattr(args.data, 'rho1_select_ratio', 0.6),
                    domain_labels=_source_labels,
                    domain_normalize=getattr(args.data, "rho1_domain_normalize", False),
                    domain_norm_min_tokens=getattr(args.data, "rho1_domain_norm_min_tokens", 32),
                    domain_norm_eps=getattr(args.data, "rho1_domain_norm_eps", 1e-6),
                    domain_norm_clip=getattr(args.data, "rho1_domain_norm_clip", 0.0),
                    stratified_by_domain=getattr(args.data, "rho1_stratified_by_domain", False),
                    stratified_exact_budget=getattr(args.data, "rho1_stratified_exact_budget", True),
                    cu_seqlens=batch_cu_seqlens,
                    doc_sources=batch_doc_sources,
                    all_expert_nlls=all_expert_nlls,
                    distinctive_advantage=getattr(args.data, "rho1_distinctive_advantage", False),
                    distinctive_alpha=getattr(args.data, "rho1_distinctive_alpha", 3.0),
                    distinctive_margin=getattr(args.data, "rho1_distinctive_margin", 0.1),
                    distinctive_normalize=getattr(args.data, "rho1_distinctive_normalize", True),
                    distinctive_eps=getattr(args.data, "rho1_distinctive_eps", 1e-6),
                    entropy_gate_frac=getattr(args.data, "rho1_entropy_gate_frac", 0.0),
                    entropy_chunk_size=getattr(args.data, "rho1_entropy_chunk_size", 256),
                    excess_mass_select=getattr(args.data, "rho1_excess_mass_select", False),
                    excess_mass_p=getattr(args.data, "rho1_excess_mass_p", 0.9),
                    excess_mass_min_ratio=getattr(args.data, "rho1_excess_mass_min_ratio", 0.1),
                    excess_mass_max_ratio=getattr(args.data, "rho1_excess_mass_max_ratio", 0.8),
                    excess_mass_cap_quantile=getattr(args.data, "rho1_excess_mass_cap_quantile", 0.99),
                    excess_mass_min_positive_mass=getattr(args.data, "rho1_excess_mass_min_positive_mass", 1e-8),
                    excess_mass_rising_floor=getattr(args.data, "rho1_excess_mass_rising_floor", False),
                    excess_mass_floor_start_ratio=getattr(args.data, "rho1_excess_mass_floor_start_ratio", 0.2),
                    excess_mass_floor_end_ratio=getattr(args.data, "rho1_excess_mass_floor_end_ratio", 0.8),
                    current_step=train_state.step,
                    total_steps=args.steps,
                    capgap_enable=getattr(args.data, "rho1_capgap_enable", False),
                    capgap_lambda=getattr(args.data, "rho1_capgap_lambda", 0.5),
                    capgap_main_idx=getattr(args.data, "rho1_capgap_main_expert_idx", 1),
                    capgap_gate_idx=getattr(args.data, "rho1_capgap_gate_expert_idx", 0),
                    capgap_hard_x_gate=getattr(args.data, "rho1_capgap_hard_x_gate", False),
                    capgap_hard_x_gate_margin=getattr(args.data, "rho1_capgap_hard_x_gate_margin", 0.0),
                )
                _is_capgap = getattr(args.data, 'rho1_capgap_enable', False)
                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    if _is_round_robin:
                        logger.info("ROUND-ROBIN EXPERT RHO-1 - First batch statistics")
                        logger.info(f"  Mode: cycle sole-reference expert each step")
                    elif _is_staged:
                        _boundaries = list(args.data.rho1_stage_boundaries)
                        logger.info("STAGED EXPERT RHO-1 - First batch statistics")
                        logger.info(
                            "  Mode: stage-based sole-reference expert "
                            f"(step boundaries={_boundaries}, n_experts={len(teacher_models)})"
                        )
                    elif _is_capgap:
                        _main_idx = int(getattr(args.data, 'rho1_capgap_main_expert_idx', 1))
                        _gate_idx = int(getattr(args.data, 'rho1_capgap_gate_expert_idx', 0))
                        _lambda = float(getattr(args.data, 'rho1_capgap_lambda', 0.5))
                        _hard_gate = bool(getattr(args.data, 'rho1_capgap_hard_x_gate', False))
                        _tau_x = float(getattr(args.data, 'rho1_capgap_hard_x_gate_margin', 0.0))
                        logger.info("CAPACITY-GAP-PENALTY RHO-1 - First batch statistics")
                        logger.info(
                            "  Mode: rank by main-expert excess, penalize tokens where "
                            "the gate expert is also far behind the main expert"
                        )
                        logger.info(
                            f"  score = (L_student - L_main[idx={_main_idx}]) "
                            f"- {_lambda:.3f} * relu(L_gate[idx={_gate_idx}] - L_main[idx={_main_idx}])"
                        )
                        if _hard_gate:
                            logger.info(
                                f"  HARD 1B-LEARNABILITY GATE: restrict top-K to tokens with "
                                f"L_student - L_gate[idx={_gate_idx}] > {_tau_x:.3f} "
                                "(Variant B: pre-filter capacity-only tokens)"
                            )
                    else:
                        _reduce_mode = str(getattr(args.data, "rho1_expert_reduce", "min")).lower()
                        logger.info("BEST-EXPERT RHO-1 - First batch statistics")
                        logger.info(
                            "  Routing: token-level "
                            f"(per-token {_reduce_mode} NLL across online experts)"
                        )
                    logger.info(f"  Loss:    Rho-1 top-{getattr(args.data, 'rho1_select_ratio', 0.6):.0%} token selection")
                    if getattr(args.data, "rho1_distinctive_advantage", False):
                        logger.info(
                            "  Distinctiveness: enabled "
                            f"(alpha={getattr(args.data, 'rho1_distinctive_alpha', 3.0)}, "
                            f"margin={getattr(args.data, 'rho1_distinctive_margin', 0.1)}, "
                            f"normalize={getattr(args.data, 'rho1_distinctive_normalize', True)})"
                        )
                    _attr_mode = "per-token (cu_seqlens)" if batch_doc_sources is not None else "per-sequence (legacy)"
                    logger.info(f"  domain_attribution: {_attr_mode}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif (
                (use_best_expert or use_best_expert_seq)
                and teacher_logprobs is not None
                and not online_rw_teacher_routing_only
            ):
                logits = model(input_ids, target=None,
                               cu_seqlens=cu_seqlens_tensor, max_seqlen=max_seqlen)

                loss, batch_delta_stats = compute_best_expert_loss(
                    logits=logits,
                    labels=labels,
                    teacher_logprobs=teacher_logprobs,
                    ce_anchor_alpha=getattr(args.data, "best_expert_ce_anchor_alpha", 0.0),
                )

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    routing_mode = 'sequence-level (per-seq argmin mean NLL)' if use_best_expert_seq else 'token-level (per-token argmin NLL)'
                    logger.info("=" * 60)
                    logger.info("BEST-EXPERT PROPORTIONAL REWEIGHTING - First batch statistics")
                    logger.info(f"  Reference: per-token min NLL across {'online experts' if teacher_models else 'precomputed expert signal'}")
                    logger.info(f"  Routing:   {routing_mode}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {_fmt_metric(v)}")
                    logger.info("=" * 60)
            elif use_weighted:
                logits = model(input_ids, target=None)

                loss, batch_delta_stats = compute_weighted_loss_with_teacher(
                    logits=logits,
                    labels=labels,
                    teacher_logprobs=teacher_logprobs,
                    delta_beta=args.data.delta_beta,
                    delta_max=args.data.delta_max,
                    use_entropy_delta=getattr(args.data, 'use_entropy_delta', False),
                    teacher_entropy_precomputed=teacher_entropy_precomputed if teacher_model is not None else None,
                )

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("CONTINUOUS DELTA WEIGHTING - First batch statistics")
                    logger.info(f"  Teacher source: {'on-the-fly' if teacher_model is not None else 'cached'}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info(f"  delta_beta: {args.data.delta_beta}")
                    logger.info(f"  delta_max: {args.data.delta_max}")
                    logger.info("=" * 60)
            elif getattr(args.data, 'use_ema_ref', False) and ema_ref_nll is not None:
                logits = model(input_ids, target=None)

                loss, batch_delta_stats = compute_ema_ref_loss(
                    logits=logits,
                    labels=labels,
                    ema_ref_nll=ema_ref_nll,
                    select_ratio=getattr(args.data, 'ema_select_ratio', 0.6),
                )

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("EMA SELF-REFERENCE TOKEN SELECTION - First batch statistics")
                    logger.info(f"  Reference: capacity-matched 1B (same init checkpoint)")
                    logger.info(f"  select_ratio: {getattr(args.data, 'ema_select_ratio', 0.6)}")
                    logger.info(f"  ema_decay: {getattr(args.data, 'ema_decay', 0.999)}")
                    logger.info(f"  ema_update_freq: {getattr(args.data, 'ema_update_freq', 10)}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif getattr(args.data, 'use_lwt', False):
                logits = model(input_ids, target=None)

                loss, batch_delta_stats = compute_lwt_loss(
                    logits=logits,
                    labels=labels,
                    use_entropy=getattr(args.data, 'lwt_use_entropy', True),
                )

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    variant = "confident-loss" if getattr(args.data, 'lwt_use_entropy', True) else "loss-only"
                    logger.info(f"LWT ({variant}) - First batch statistics")
                    logger.info(f"  use_entropy: {getattr(args.data, 'lwt_use_entropy', True)}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif getattr(args.data, 'use_frontier_band', False):
                logits = model(input_ids, target=None)

                loss, batch_delta_stats = compute_frontier_band_loss(
                    logits=logits,
                    labels=labels,
                    mu=getattr(args.data, 'frontier_mu', 0.0),
                    sigma=getattr(args.data, 'frontier_sigma', 0.0),
                    weight_clip_min=getattr(args.data, 'frontier_clip_min', 0.25),
                    weight_clip_max=getattr(args.data, 'frontier_clip_max', 4.0),
                )

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("FRONTIER-BAND MARGIN WEIGHTING - First batch statistics")
                    logger.info(f"  mu: {batch_delta_stats.get('frontier/mu', 'auto')}")
                    logger.info(f"  sigma: {batch_delta_stats.get('frontier/sigma', 'auto')}")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif getattr(args.data, 'use_frontierv2', False):
                logits = model(input_ids, target=None)

                loss, batch_delta_stats = compute_frontierv2_loss(
                    logits=logits,
                    labels=labels,
                    mu=getattr(args.data, 'frontierv2_mu', 0.0),
                    sigma=getattr(args.data, 'frontierv2_sigma', 0.0),
                    weight_clip_min=getattr(args.data, 'frontierv2_clip_min', 0.25),
                    weight_clip_max=getattr(args.data, 'frontierv2_clip_max', 4.0),
                )

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("FRONTIER-BAND v2 MARGIN WEIGHTING - First batch statistics")
                    logger.info(f"  mu: {batch_delta_stats.get('frontierv2/mu', 0.0)} (FIXED)")
                    logger.info(f"  sigma: {batch_delta_stats.get('frontierv2/sigma', 'auto')}")
                    logger.info(f"  frac_gold_not_top2: {batch_delta_stats.get('frontierv2/frac_gold_not_top2', 0.0)} (should be near 0%)")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif getattr(args.data, 'use_frontierv3', False):
                logits = model(input_ids, target=None)

                loss, batch_delta_stats = compute_frontierv3_loss(
                    logits=logits,
                    labels=labels,
                    sigma_margin=getattr(args.data, 'frontierv3_sigma_margin', 1.0),
                    alpha=getattr(args.data, 'frontierv3_alpha', 1.0),
                    weight_min=getattr(args.data, 'frontierv3_w_min', 1.0),
                    weight_max=getattr(args.data, 'frontierv3_w_max', 2.0),
                    detach_weights=getattr(args.data, 'frontierv3_detach_weights', True),
                )

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("FRONTIER-v3 (Sequence-centered margin) - First batch statistics")
                    logger.info(
                        f"  sigma_margin: {getattr(args.data, 'frontierv3_sigma_margin', 1.0)}"
                    )
                    logger.info(
                        f"  alpha: {getattr(args.data, 'frontierv3_alpha', 1.0)}"
                    )
                    logger.info(
                        f"  clip: [{getattr(args.data, 'frontierv3_w_min', 1.0)}, "
                        f"{getattr(args.data, 'frontierv3_w_max', 2.0)}]"
                    )
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif getattr(args.data, 'use_frontierv4', False):
                logits = model(input_ids, target=None)

                loss, batch_delta_stats = compute_frontierv4_loss(
                    logits=logits,
                    labels=labels,
                    alpha=getattr(args.data, 'frontierv4_alpha', 1.0),
                    weight_min=getattr(args.data, 'frontierv4_w_min', 1.0),
                    weight_max=getattr(args.data, 'frontierv4_w_max', 2.0),
                    detach_weights=getattr(args.data, 'frontierv4_detach_weights', True),
                )

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("FRONTIER-v4 (Adaptive centering) - First batch statistics")
                    logger.info(
                        f"  alpha: {getattr(args.data, 'frontierv4_alpha', 1.0)}"
                    )
                    logger.info(
                        f"  clip: [{getattr(args.data, 'frontierv4_w_min', 1.0)}, "
                        f"{getattr(args.data, 'frontierv4_w_max', 2.0)}]"
                    )
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif getattr(args.data, 'use_ema_frontier', False) and ema_ref_nll is not None:
                logits = model(input_ids, target=None)

                loss, batch_delta_stats = compute_ema_frontier_loss(
                    logits=logits,
                    labels=labels,
                    ema_ref_nll=ema_ref_nll,
                    beta=getattr(args.data, 'ema_frontier_beta', 1.0),
                    weight_clip_min=getattr(args.data, 'ema_frontier_clip_min', 0.1),
                    weight_clip_max=getattr(args.data, 'ema_frontier_clip_max', 10.0),
                )

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("EMA-FRONTIER (Progress-Aware) - First batch statistics")
                    logger.info(f"  Pseudo-EMA reference: {getattr(args.checkpoint, 'init_ckpt_path', 'unknown')}")
                    logger.info(f"  beta: {getattr(args.data, 'ema_frontier_beta', 1.0)}")
                    logger.info(f"  clip: [{getattr(args.data, 'ema_frontier_clip_min', 0.1)}, {getattr(args.data, 'ema_frontier_clip_max', 10.0)}]")
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif getattr(args.data, 'use_siw', False):
                logits = model(input_ids, target=None)

                loss, batch_delta_stats = compute_siw_loss(
                    logits=logits,
                    labels=labels,
                    k=getattr(args.data, 'siw_k', 8),
                    beta=getattr(args.data, 'siw_beta', 1.0),
                )

                if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                    logger.info("=" * 60)
                    logger.info("SIW - Sequential Influence Weighting - First batch statistics")
                    logger.info(f"  Reference: none (reference-free)")
                    logger.info(f"  k (window): {getattr(args.data, 'siw_k', 8)}")
                    logger.info(f"  beta: {getattr(args.data, 'siw_beta', 1.0)}")
                    for key, v in batch_delta_stats.items():
                        logger.info(f"  {key}: {v:.4f}")
                    logger.info("=" * 60)
            else:
                loss = model(input_ids, labels)
            if args.grad_acc_steps > 1:
                model.set_requires_gradient_sync(train_state.acc_step == 0)

            # We scale loss with grad_acc_steps so the gradient is the same
            # regardless of grad_acc_steps
            loss = loss / args.grad_acc_steps
            # backward on scaled loss to create scaled gradients
            loss.backward()
            # For logging we undo that scaling
            loss = loss.detach() * args.grad_acc_steps

            # ── online excess-loss controller: accumulate then maybe update ──
            if online_ctrl is not None and _source_labels is not None:
                # Per-DOCUMENT NLL attribution using cu_seqlens and doc_sources.
                # Each packed block contains multiple short documents; we split
                # student and expert NLL per document and attribute each to its
                # actual source rather than using the block-level average.
                _doc_src_list: List[str] = []
                _doc_student_nll_list: List[float] = []
                _doc_expert_nll_list: Optional[List[float]] = None if teacher_logprobs is None else []
                _doc_tokens_list: List[float] = []

                _student_token_losses = None
                try:
                    _has_logits = logits is not None
                except NameError:
                    _has_logits = False
                if _has_logits and batch_doc_sources is not None and batch_cu_seqlens is not None:
                    with torch.no_grad():
                        _student_token_losses = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            labels.reshape(-1),
                            reduction="none",
                            ignore_index=-100,
                        ).reshape(labels.shape)  # (B, T)

                if _student_token_losses is not None and batch_doc_sources is not None and batch_cu_seqlens is not None:
                    for b in range(labels.shape[0]):
                        cu = batch_cu_seqlens[b] if batch_cu_seqlens[b] is not None else [0, labels.shape[1]]
                        dsrcs = batch_doc_sources[b] if batch_doc_sources is not None and batch_doc_sources[b] is not None else None
                        for d in range(len(cu) - 1):
                            doc_start = cu[d]
                            doc_end = cu[d + 1]
                            doc_labels = labels[b, doc_start:doc_end]
                            valid_mask = (doc_labels != -100)
                            n_valid = valid_mask.sum().item()
                            if n_valid == 0:
                                continue
                            doc_student_nll = _student_token_losses[b, doc_start:doc_end][valid_mask].mean().item()
                            _doc_student_nll_list.append(doc_student_nll)
                            _doc_tokens_list.append(float(n_valid))
                            if dsrcs is not None and d < len(dsrcs):
                                _doc_src_list.append(dsrcs[d])
                            else:
                                _doc_src_list.append(_source_labels[b] if b < len(_source_labels) else "unknown")
                            if teacher_logprobs is not None:
                                doc_expert_nll = (teacher_logprobs[b, doc_start:doc_end] * valid_mask.float()).sum().item() / max(n_valid, 1)
                                _doc_expert_nll_list.append(doc_expert_nll)
                else:
                    # Fallback: no cu_seqlens/doc_sources available, use block-level
                    _ce_per_seq = batch_delta_stats.get("_ce_per_seq") if batch_delta_stats else None
                    _tokens_per_seq_fb = batch_delta_stats.get("_tokens_per_seq") if batch_delta_stats else None
                    _expert_per_seq_fb: Optional[list] = None
                    if teacher_logprobs is not None:
                        _valid_mask_f = (labels != -100).float()
                        _tok_per_seq = _valid_mask_f.sum(dim=-1).clamp(min=1)
                        _expert_per_seq_fb = (
                            (teacher_logprobs * _valid_mask_f).sum(dim=-1) / _tok_per_seq
                        ).tolist()
                        if _tokens_per_seq_fb is None:
                            _tokens_per_seq_fb = _tok_per_seq
                    if _expert_per_seq_fb is None and not teacher_models:
                        _expert_per_seq_fb = [0.0] * len(_source_labels)
                    _doc_src_list = list(_source_labels)
                    _doc_student_nll_list = _ce_per_seq.tolist() if _ce_per_seq is not None else None
                    _doc_expert_nll_list = _expert_per_seq_fb
                    _doc_tokens_list = _tokens_per_seq_fb.tolist() if _tokens_per_seq_fb is not None else None
                    # Keep controller on the per-seq accumulation path when possible.
                    # In plain CE mode batch_delta_stats may be absent, so we provide
                    # a per-sequence student proxy (shared batch CE) and unit tokens.
                    if _doc_student_nll_list is None and _source_labels is not None:
                        _doc_student_nll_list = [float(loss.item())] * len(_source_labels)
                    if _doc_tokens_list is None and _source_labels is not None:
                        _doc_tokens_list = [1.0] * len(_source_labels)

                # Raw-loss mode (Run B): treat expert NLL as 0 per document.
                if _doc_expert_nll_list is None and not teacher_models:
                    _doc_expert_nll_list = [0.0] * len(_doc_src_list)

                online_ctrl.accumulate(
                    source_labels=_doc_src_list,
                    per_seq_student_nll=_doc_student_nll_list,
                    per_seq_expert_nll=_doc_expert_nll_list,
                    per_seq_tokens=_doc_tokens_list,
                    student_nll=loss.item(),
                    n_valid_tokens=int((labels != -100).sum().item()),
                )

                if train_state.acc_step == 0:  # only update on optimizer steps
                    new_weights = online_ctrl.maybe_update(train_state.step)
                    if new_weights is not None:
                        online_loader.set_source_weights(new_weights)
                        if get_is_master():
                            norm_str = " ".join(
                                f"{k.replace('_shuffled','')}={v:.4f}"
                                for k, v in new_weights.items()
                            )
                            logger.info(
                                f"[OnlineReweighting] step={train_state.step} "
                                f"new weights: {norm_str}"
                            )
            # ─────────────────────────────────────────────────────────────────

            # optimizer step
            grad_norm = -1.0
            skipped_nonfinite_step = False
            if train_state.acc_step == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.optim.clip, foreach=True
                )

                grad_norm = (
                    grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
                ).item()

                # NaN/inf guard. A single bad batch (FP8 teacher overflow,
                # transient NCCL bit-flip, etc.) can produce a non-finite grad
                # that, if let through to optimizer.step(), poisons the
                # parameters and the AdamW moment estimates — at which point
                # every subsequent step is also NaN and the rest of the run is
                # wasted compute. Concrete failure mode this guards against:
                # task 65 (kd-rl7b-28800) was perfectly healthy at step 13660
                # (loss ~1.5-1.8, grad 1.86e-1) and all 32 ranks simultaneously
                # went NaN at step 13670, propagating through the rest of the
                # run. We catch that here by skipping the optimizer step (and
                # leaving params/moments untouched) while still ticking the
                # scheduler and step counter so the LR schedule stays aligned
                # with the configured horizon. Threshold for "give up and
                # raise" defaults to 10 consecutive skipped steps and can be
                # overridden via LINGUA_MAX_CONSECUTIVE_NAN_SKIPS.
                if not math.isfinite(grad_norm):
                    skipped_nonfinite_step = True
                    optimizer.zero_grad()
                    scheduler.step()
                    train_state.step += 1
                    if not hasattr(train_state, "consecutive_nonfinite_steps"):
                        train_state.consecutive_nonfinite_steps = 0
                    train_state.consecutive_nonfinite_steps += 1
                    if get_is_master():
                        logger.warning(
                            f"[NaN-guard] step={train_state.step}: "
                            f"non-finite grad_norm ({grad_norm}); skipped "
                            f"optimizer.step(). "
                            f"consecutive_skipped={train_state.consecutive_nonfinite_steps}"
                        )
                    _max_consec = int(
                        os.environ.get("LINGUA_MAX_CONSECUTIVE_NAN_SKIPS", "10")
                    )
                    if train_state.consecutive_nonfinite_steps >= _max_consec:
                        raise RuntimeError(
                            f"[NaN-guard] {train_state.consecutive_nonfinite_steps} "
                            f"consecutive non-finite-grad steps at step "
                            f"{train_state.step}; aborting to avoid silently "
                            f"burning compute. Override threshold with "
                            f"LINGUA_MAX_CONSECUTIVE_NAN_SKIPS=<N>."
                        )
                else:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    train_state.step += 1
                    if hasattr(train_state, "consecutive_nonfinite_steps"):
                        train_state.consecutive_nonfinite_steps = 0

            # updates the scale for next iteration
            # training iteration complete
            end_timer.record()

            torch.cuda.synchronize()

            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            # if profiler is active
            if torch_profiler:
                xformers.profiler.step()

            # log metrics
            if every_n_steps(
                train_state,
                args.logging.freq,
                acc_step=None if args.logging.acc_freq else 0,
                acc_freq=args.logging.acc_freq,
            ):
                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (time_delta * args.distributed.tp_size)

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                total_acc_steps = (
                    args.grad_acc_steps * train_state.step + train_state.acc_step
                )
                tokens_per_gpu = (
                    total_acc_steps * args.data.batch_size * args.data.seq_len
                )
                total_tokens = dp_degree * tokens_per_gpu
                # This is an estimate and the correct values may change
                # if you change the architecture
                # Use xformer's analyze profile trace to get actual measurement
                FLOPS = (
                    get_num_flop_per_token(
                        model_param_count - args.model.vocab_size * args.model.dim,
                        args.model.n_layers,
                        args.model.dim,
                        args.data.seq_len,
                    )
                    * wps
                )
                metrics = flatten_dict(
                    {
                        "global_step": train_state.step,
                        "acc_step": train_state.acc_step,
                        "speed": {
                            "wps": wps,
                            "FLOPS": FLOPS,
                            "curr_iter_time": curr_iter_time,
                            "data_load_time": data_load_time,
                        },
                        "optim": {
                            "grad_norm": grad_norm,
                            "lr": curr_lr,
                            "total_tokens": total_tokens,
                        },
                        "memory": gpu_mem_stats._asdict(),
                    },
                    sep="/",
                )

                to_sync = {}
                to_sync["loss/out"] = loss.item()
                if batch_delta_stats is not None:
                    to_sync.update({k: v for k, v in batch_delta_stats.items() if not k.startswith("_")})
                if expert_routing_stats:
                    to_sync.update(expert_routing_stats)
                if pending_grad_probe_stats:
                    to_sync.update(pending_grad_probe_stats)
                if online_ctrl is not None:
                    to_sync.update(online_ctrl.metrics_dict())
                metrics.update(dist_mean_dict(to_sync))
                pending_grad_probe_stats = {}

                if get_is_master():
                    if batch_delta_stats is not None:
                        for k, v in batch_delta_stats.items():
                            if k.startswith("_"):
                                # Debug-only stats may be vectors (e.g. per-seq arrays).
                                # Log scalar tensors directly and reduce non-scalar tensors
                                # to their mean to keep metrics logging robust.
                                if isinstance(v, torch.Tensor):
                                    if v.numel() == 1:
                                        metrics[k[1:]] = v.item()
                                    else:
                                        metrics[f"{k[1:]}/mean"] = v.float().mean().item()
                                else:
                                    metrics[k[1:]] = v
                    metric_logger.log(metrics)

                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()
                delta_info = ""
                if batch_delta_stats is not None:
                    if "eam/total_loss" in batch_delta_stats:
                        # Entropy-aware margin matching stats
                        delta_info = (
                            f"  m_δ: {batch_delta_stats['eam/margin_delta_mean']:.3f}±{batch_delta_stats['eam/margin_delta_std']:.3f}"
                            f"  λ: {batch_delta_stats['eam/lambda_mean']:.3f}±{batch_delta_stats['eam/lambda_std']:.3f}"
                            f"  H_norm: {batch_delta_stats['eam/entropy_norm_mean']:.3f}"
                            f"  m_loss: {batch_delta_stats['eam/margin_loss_mean']:.4f}"
                        )
                    elif "kd/kl_loss" in batch_delta_stats:
                        # KL distillation stats
                        delta_info = (
                            f"  KL: {batch_delta_stats['kd/kl_loss']:.3f}"
                            f"  CE: {batch_delta_stats['kd/ce_loss']:.3f}"
                        )
                    elif "rho1/excess_loss_mean" in batch_delta_stats:
                        delta_info = (
                            f"  excess: {batch_delta_stats['rho1/excess_loss_mean']:.3f}"
                            f"  thresh: {batch_delta_stats['rho1/threshold']:.3f}"
                            f"  sel: {batch_delta_stats['rho1/percent_selected']:.1f}%"
                            f"  sel_loss: {batch_delta_stats['rho1/selected_loss_mean']:.3f}"
                        )
                    elif "delta_H/mean" in batch_delta_stats:
                        delta_info = (
                            f"  ΔH: {batch_delta_stats['delta_H/mean']:.3f}±{batch_delta_stats['delta_H/std']:.3f}"
                            f"  w_avg: {batch_delta_stats['delta_H/weight_mean']:.3f}"
                            f"  H_s: {batch_delta_stats['student_entropy/mean']:.3f}±{batch_delta_stats['student_entropy/std']:.3f}"
                            f"  H_t: {batch_delta_stats['teacher_entropy/mean']:.3f}±{batch_delta_stats['teacher_entropy/std']:.3f}"
                        )
                    elif "lwt/weight_mean" in batch_delta_stats:
                        delta_info = (
                            f"  w_avg: {batch_delta_stats['lwt/weight_mean']:.3f}±{batch_delta_stats['lwt/weight_std']:.3f}"
                            f"  ul: {batch_delta_stats['lwt/loss_mean']:.3f}"
                        )
                        if "lwt/entropy_mean" in batch_delta_stats:
                            delta_info += f"  H: {batch_delta_stats['lwt/entropy_mean']:.3f}"
                    elif "mile/weight_mean" in batch_delta_stats:
                        delta_info = (
                            f"  w_avg: {batch_delta_stats['mile/weight_mean']:.3f}±{batch_delta_stats['mile/weight_std']:.3f}"
                            f"  H_norm: {batch_delta_stats['mile/entropy_norm_mean']:.3f}±{batch_delta_stats['mile/entropy_norm_std']:.3f}"
                            f"  γ: {batch_delta_stats['mile/gamma']:.2f}"
                        )
                    elif "ema_ref/excess_loss_mean" in batch_delta_stats:
                        delta_info = (
                            f"  excess: {batch_delta_stats['ema_ref/excess_loss_mean']:.3f}"
                            f"  sel: {batch_delta_stats['ema_ref/frac_selected']:.2f}"
                            f"  sel_loss: {batch_delta_stats['ema_ref/selected_loss_mean']:.3f}"
                            f"  drop_loss: {batch_delta_stats['ema_ref/dropped_loss_mean']:.3f}"
                        )
                    elif "egkd/total_loss" in batch_delta_stats:
                        delta_info = (
                            f"  α: {batch_delta_stats['egkd/alpha_mean']:.3f}±{batch_delta_stats['egkd/alpha_std']:.3f}"
                            f"  α>.5: {batch_delta_stats['egkd/alpha_gt_0.5_frac']:.2f}"
                            f"  ΔH: {batch_delta_stats['egkd/entropy_gap_mean']:.3f}"
                            f"  τ: {batch_delta_stats['egkd/gate_tau']:.3f}"
                        )
                    elif "frontier/total_loss" in batch_delta_stats:
                        delta_info = (
                            f"  margin: {batch_delta_stats['frontier/margin_mean']:.3f}±{batch_delta_stats['frontier/margin_std']:.3f}"
                            f"  w: {batch_delta_stats['frontier/weight_mean']:.3f}±{batch_delta_stats['frontier/weight_std']:.3f}"
                            f"  μ: {batch_delta_stats['frontier/mu']:.3f}"
                            f"  σ: {batch_delta_stats['frontier/sigma']:.3f}"
                        )
                    elif "remit/total_loss" in batch_delta_stats:
                        delta_info = (
                            f"  w: {batch_delta_stats['remit/weight_mean']:.3f}±{batch_delta_stats['remit/weight_std']:.3f}"
                            f"  δ: {batch_delta_stats['remit/delta_mean']:.3f}"
                            f"  δc: {batch_delta_stats['remit/delta_centered_mean']:.3f}±{batch_delta_stats['remit/delta_centered_std']:.3f}"
                            f"  clip%: {batch_delta_stats['remit/frac_at_clip_floor']:.2f}"
                        )
                    elif "critic/loss_weighted" in batch_delta_stats:
                        delta_info = (
                            f"  w: {batch_delta_stats['critic/w_combined_mean']:.3f}±{batch_delta_stats['critic/w_combined_std']:.3f}"
                            f"  w_seq: {batch_delta_stats['critic/w_seq_mean']:.3f}[{batch_delta_stats['critic/w_seq_min']:.2f},{batch_delta_stats['critic/w_seq_max']:.2f}]"
                            f"  gap: {batch_delta_stats['critic/gap_mean']:.3f}±{batch_delta_stats['critic/gap_std']:.3f}"
                            f"  gap+%: {batch_delta_stats['critic/gap_frac_positive']:.2f}"
                            f"  w_tok: {batch_delta_stats['critic/w_tok_mean']:.3f}"
                        )
                    elif "mc/total_loss" in batch_delta_stats:
                        delta_info = (
                            f"  gap: {batch_delta_stats['mc/gap_mean']:.3f}±{batch_delta_stats['mc/gap_std']:.3f}"
                            f"  act%: {batch_delta_stats['mc/frac_active']:.2f}"
                            f"  c_loss: {batch_delta_stats['mc/constraint_loss']:.4f}"
                            f"  m_s: {batch_delta_stats['mc/margin_student_mean']:.3f}"
                            f"  m_t: {batch_delta_stats['mc/margin_teacher_mean']:.3f}"
                        )
                    elif "siw/total_loss" in batch_delta_stats:
                        delta_info = (
                            f"  w: {batch_delta_stats['siw/weight_mean']:.3f}±{batch_delta_stats['siw/weight_std']:.3f}"
                            f"  infl: {batch_delta_stats['siw/influence_mean']:.3f}"
                            f"  nll: {batch_delta_stats['siw/nll_mean']:.3f}"
                        )
                    elif "frontierv2/total_loss" in batch_delta_stats:
                        delta_info = (
                            f"  margin: {batch_delta_stats['frontierv2/margin_mean']:.3f}±{batch_delta_stats['frontierv2/margin_std']:.3f}"
                            f"  w: {batch_delta_stats['frontierv2/weight_mean']:.3f}±{batch_delta_stats['frontierv2/weight_std']:.3f}"
                            f"  μ: {batch_delta_stats['frontierv2/mu']:.3f}"
                            f"  σ: {batch_delta_stats['frontierv2/sigma']:.3f}"
                        )
                    elif "frontierv3/loss_weighted" in batch_delta_stats:
                        delta_info = (
                            f"  m_raw: {batch_delta_stats['frontierv3/margin_raw_mean']:.3f}"
                            f"  m_ctr: {batch_delta_stats['frontierv3/margin_centered_mean']:.3f}±{batch_delta_stats['frontierv3/margin_centered_std']:.3f}"
                            f"  w: {batch_delta_stats['frontierv3/weight_mean']:.3f}±{batch_delta_stats['frontierv3/weight_std']:.3f}"
                            f"  corr(v2): {batch_delta_stats['frontierv3/corr_seq_margin_seq_weight_v2']:.3f}"
                            f"  corr(v3): {batch_delta_stats['frontierv3/corr_seq_margin_seq_weight_v3']:.3f}"
                        )
                    elif "frontierv4/loss_weighted" in batch_delta_stats:
                        delta_info = (
                            f"  m_raw: {batch_delta_stats['frontierv4/margin_raw_mean']:.3f}"
                            f"  m_sh: {batch_delta_stats['frontierv4/margin_shifted_mean']:.3f}±{batch_delta_stats['frontierv4/margin_shifted_std']:.3f}"
                            f"  w: {batch_delta_stats['frontierv4/weight_mean']:.3f}±{batch_delta_stats['frontierv4/weight_std']:.3f}"
                            f"  α_blend: {batch_delta_stats['frontierv4/blend_alpha_mean']:.3f}±{batch_delta_stats['frontierv4/blend_alpha_std']:.3f}"
                            f"  σ: {batch_delta_stats['frontierv4/sigma']:.3f}"
                        )
                    elif "ema_frontier/total_loss" in batch_delta_stats:
                        delta_info = (
                            f"  margin: {batch_delta_stats['ema_frontier/margin_mean']:.3f}"
                            f"  σ: {batch_delta_stats['ema_frontier/sigma']:.3f}"
                            f"  Δ: {batch_delta_stats['ema_frontier/delta_mean']:.4f}(p95={batch_delta_stats['ema_frontier/delta_p95']:.4f})"
                            f"  w: {batch_delta_stats['ema_frontier/weight_mean']:.3f}±{batch_delta_stats['ema_frontier/weight_std']:.3f}"
                        )
                    elif "delta/mean" in batch_delta_stats:
                        # Legacy delta weighting (only if delta/mean exists)
                        delta_info = (
                            f"  delta: {batch_delta_stats['delta/mean']:.3f}"
                            f"  w_avg: {batch_delta_stats['delta/weight_mean']:.3f}"
                        )
                    else:
                        # No specific loss type matched - no delta info to display
                        delta_info = ""
                routing_info = ""
                if expert_routing_stats:
                    # Only include global best-expert routing keys. Per-source keys
                    # (routing_fraction/{source}/expert_i) should not inflate this list.
                    expert_ids = sorted(
                        int(k.rsplit("_", 1)[-1])
                        for k in expert_routing_stats
                        if k.startswith("best_expert/routing_frac_expert_")
                        and k.rsplit("_", 1)[-1].isdigit()
                    )
                    if expert_ids:
                        fracs = [
                            expert_routing_stats.get(
                                f"best_expert/routing_frac_expert_{i}", 0.0
                            )
                            for i in expert_ids
                        ]
                        routing_info = "  route:[" + "/".join(f"{f:.2f}" for f in fracs) + "]"
                logger.info(
                    f"step: {train_state.step}"
                    f"  total_tokens: {total_tokens}"
                    f"  loss: {round(loss.item(),4):>7}"
                    f"  grad: {grad_norm:.2e}"
                    f"{delta_info}"
                    f"{routing_info}"
                    f"  flops: {FLOPS:.2e}"
                    f"  wps: {wps:.2e}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {data_load_time:>5}"
                    f"  lr: {curr_lr:.2e}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                    f"  pow: {gpu_mem_stats.power_draw/1000} W"
                )

            saved = False
            is_dump_step = every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0
            )
            is_eval_step = every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0)
            if is_dump_step or is_eval_step:
                saved = checkpoint.save(
                    model,
                    optimizer,
                    train_state,
                    args,
                    device_mesh=world_mesh,
                )

                if saved and online_ctrl is not None and get_is_master():
                    import json as _json
                    _ctrl_path = os.path.join(args.dump_dir, "online_ctrl_state.json")
                    with open(_ctrl_path, "w") as _f:
                        _json.dump({"step": train_state.step, **online_ctrl.state_dict()}, _f)

                # EMA-Frontier lagged swap: triggered inside the checkpoint save
                # block so the lagged checkpoint is guaranteed to exist on disk
                # (it was saved at the previous checkpoint interval).
                # NOTE: runs on ALL ranks — ema_ref_model is replicated per-GPU.
                if (
                    getattr(args.data, 'use_ema_frontier', False)
                    and getattr(args.data, 'ema_frontier_lagged', False)
                    and ema_ref_model is not None
                    and is_dump_step
                ):
                    interval = args.checkpoint.dump.every
                    lagged_step = train_state.step - interval
                    if lagged_step > 0:
                        ok = swap_ema_ref_checkpoint(
                            ema_ref_model,
                            lagged_step,
                            args.dump_dir,
                            "cuda",
                            current_step=train_state.step,
                        )
                        if get_is_master():
                            if ok:
                                logger.info(
                                    f"EMA-Frontier: swapped reference to step {lagged_step} "
                                    f"(lag={interval} steps)"
                                )
                            else:
                                logger.warning(
                                    f"EMA-Frontier: could not load step {lagged_step}, "
                                    f"keeping previous reference"
                                )

            if args.eval is not None and every_n_steps(
                train_state, args.checkpoint.eval.every, acc_step=0
            ):
                eval_backend = getattr(args, "eval_backend", "harness")

                if eval_backend == "olmes":
                    from apps.main.eval_olmes import (
                        launch_olmes_eval as _launch_eval,
                        EVAL_FOLDER_NAME,
                        OlmesEvalArgs as _EvalArgs,
                    )
                    _script = "apps.main.eval_olmes"
                else:
                    from apps.main.eval import (
                        launch_eval as _launch_eval,
                        EVAL_FOLDER_NAME,
                        EvalArgs as _EvalArgs,
                    )
                    _script = "apps.main.eval"

                eval_dict = dict(args.eval)
                if eval_backend == "olmes":
                    eval_dict.pop("harness", None)
                else:
                    eval_dict.pop("olmes", None)
                eval_args = dataclass_from_dict(_EvalArgs, eval_dict)

                eval_args.global_step = train_state.step
                eval_args.ckpt_dir = str(checkpoint.existing_saves[-1])
                eval_args.dump_dir = str(
                    os.path.join(
                        args.dump_dir,
                        "evals",
                        EVAL_FOLDER_NAME.format(train_state.step),
                    )
                )
                eval_args.metric_log_dir = args.dump_dir
                if args.async_eval_gpus is None:
                    _launch_eval(eval_args)
                elif get_is_master():
                    if wandb.run is not None and args.logging.wandb is not None:
                        eval_args.wandb = deepcopy(args.logging.wandb)
                        eval_args.wandb.id = wandb.run.id
                        eval_args.wandb.entity = wandb.run.entity
                    assert args.async_eval_gpus > 0
                    logger.info(f"Launching {eval_backend} evals on {args.async_eval_gpus} gpus")
                    with clean_env():
                        launch_job(
                            StoolArgs(
                                asdict(eval_args),
                                script=_script,
                                copy_code=False,
                                nodes=max(1, args.async_eval_gpus // 8),
                                ngpu=args.async_eval_gpus,
                                time=480,
                                mem="200GB",
                                account=os.environ.get("SLURM_ACCOUNT", "comem"),
                                qos=os.environ.get("SLURM_QOS", "h200_dev"),
                                override=False,
                                dirs_exists_ok=True,
                                anaconda=os.environ.get("LINGUA_VENV", "default"),
                            )
                        )

            if preemption_flag["flag"]:
                if not saved:
                    checkpoint.save(
                        model,
                        optimizer,
                        train_state,
                        args,
                        device_mesh=world_mesh,
                    )
                requeue_slurm_job()
                sys.exit(0)

    if not saved:
        checkpoint.save(
            model,
            optimizer,
            train_state,
            args,
            device_mesh=world_mesh,
        )
    gc.collect()


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgsgs

    @dataclass
    class LMTransformerArgsgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate TrainArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call train.py with train.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in TrainArgs dataclass.
    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(TrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
