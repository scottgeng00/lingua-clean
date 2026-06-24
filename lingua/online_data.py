"""
lingua/online_data.py

Online data mixing for excess-loss reweighting experiments.

Extends lingua's data pipeline with three additions:
  1. Tags each sequence with its source domain at sample time.
  2. Threads that label through tokenization and packing (source of
     whichever document opened the packed block is used).
  3. Exposes per-sequence source labels alongside each batch so the
     training loop can accumulate per-domain student/expert losses.
  4. Allows source weights to be updated mid-training without
     restarting the full training process.

Public API
----------
build_online_dataloader(state, ...)
    Context manager; drop-in for lingua.data.build_dataloader.
    Yields an iterator of (batch_dict, state, source_labels).

ReweightableDataLoader(args, state)
    Wraps build_online_dataloader and exposes set_source_weights().
    Internally rebuilds the async subprocess when weights change;
    this takes ~1-2 s and should only be called every K steps.

Usage in train.py
-----------------
    # Replace:
    data_loader = context_stack.enter_context(
        build_dataloader_from_args(args.data, state=train_state.data_loader_state)
    )
    batch, train_state.data_loader_state = next(data_loader)

    # With:
    data_loader = ReweightableDataLoader(args.data, train_state.data_loader_state)
    context_stack.callback(data_loader.close)
    batch, train_state.data_loader_state, source_labels = next(data_loader)
    # source_labels: List[str] of length batch_size, e.g. ["math_shuffled", "dclm_shuffled", ...]

    # Every K steps:
    data_loader.set_source_weights(new_weights)  # Dict[str, float]
"""

import contextlib
import logging
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from lingua.data import (
    DataArgs,
    MultiChoiceState,
    PackTokensState,
    PrefetchState,
    TokenizerState,
    async_iterator,
    choose_source,
    filter_by_delta,
    get_empty_buffer_state,
    init_dataloader_state_from_args,
    setup_sources,
    tokenize,
)

logger = logging.getLogger()


def _is_rank0() -> bool:
    """Return True on rank 0 (or single-process mode)."""
    try:
        import torch
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except Exception:
        pass
    return True


# ── Step 1: source-tagged sequence sampler ────────────────────────────────────

def choose_source_labeled(
    source_to_iterator: Dict[str, Iterator],
    source_to_state: Dict[str, Any],
    root_dir: str,
    sources: Dict[str, float],
    rng_state: Dict[str, Any],
):
    """
    Identical to lingua.data.choose_source, but stamps the key
    ``"__source__"`` on every yielded sequence dict so downstream
    stages can track which domain the document came from.

    Yields
    ------
    (seq, MultiChoiceState)
        seq is the raw JSON dict with an extra "__source__" field.
    """
    n_sources = len(sources)
    possible_sources = list(sources.keys())
    weights = list(sources.values())

    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state

    while True:
        norm_weights = np.array(weights) / np.array(weights).sum()
        source_choice = possible_sources[rng.choice(n_sources, p=norm_weights)]
        seq, state = next(source_to_iterator[source_choice])

        # Stamp source name; spread so we never mutate the original dict.
        seq = {**seq, "__source__": source_choice}

        source_to_state = {**source_to_state, source_choice: state}
        multi_choice_state = MultiChoiceState(
            root_dir=root_dir,
            sources=sources,
            source_to_state=source_to_state,
            rng_state=rng.bit_generator.state,
        )
        yield seq, multi_choice_state


# ── Step 2: tokenizer wrapper that threads source through ─────────────────────

def tokenize_labeled(
    iterator: Iterator,
    add_bos: bool,
    add_eos: bool,
    tokenizer_type: str,
    tokenizer_path: Optional[str] = None,
    synth_context_mode: Optional[str] = None,
    synth_context_separator: str = "\n\n",
    mask_synth_context_loss: bool = False,
    synth_context_prob: float = 1.0,
    synth_rng_state: Optional[Dict[str, Any]] = None,
    max_length: Optional[int] = None,
    nll_field: Optional[str] = None,
    entropy_field: Optional[str] = None,
    margin_field: Optional[str] = None,
):
    """
    Wraps lingua.data.tokenize to thread the ``"__source__"`` tag
    through tokenization without touching the tokenizer internals.

    The trick: a deque collects source names as documents enter, and
    popleft() retrieves them in the same order as tokenize() yields.
    This is safe because tokenize() yields exactly one item per input
    document (1:1 correspondence).

    Yields
    ------
    ((tokens, teacher_logprobs, teacher_entropy, teacher_margin, source), TokenizerState)
        Same as tokenize() but with source appended to the tuple.
    """
    source_queue: deque = deque()

    def _strip_source(it):
        """Peel off __source__ before passing content to tokenize."""
        for content, state in it:
            source_queue.append(content.get("__source__", "unknown"))
            # Build a new dict without the __source__ key so tokenize
            # doesn't encounter unexpected fields.
            yield {k: v for k, v in content.items() if k != "__source__"}, state

    for (tokens, lp, ent, mar), state in tokenize(
        _strip_source(iterator),
        add_bos,
        add_eos,
        tokenizer_type,
        tokenizer_path,
        synth_context_mode,
        synth_context_separator,
        mask_synth_context_loss,
        synth_context_prob,
        synth_rng_state,
        max_length,
        nll_field=nll_field,
        entropy_field=entropy_field,
        margin_field=margin_field,
    ):
        source = source_queue.popleft()
        yield (tokens, lp, ent, mar, source), state


# ── Step 3: packer that records block source ──────────────────────────────────

def pack_tokens_labeled(
    iterator: Iterator,
    empty_buffer_state: PackTokensState,
):
    """
    Mirrors lingua.data.pack_tokens exactly, with two additions:

    * Accepts a 5-tuple ``(tokens, teacher_logprobs, teacher_entropy,
      teacher_margin, source)`` from tokenize_labeled.
    * Tracks which source opened each packed block (i.e. the source of
      the first document to contribute tokens after a yield boundary)
      and includes it as ``"__source__"`` in the output dict.

    Yields
    ------
    ({'tokens': ndarray, 'cu_seqlens': list, '__source__': str,
      '__doc_sources__': list}, PackTokensState)

    ``__doc_sources__`` is a list of source names, one per document segment
    in the packed block (aligned with the intervals defined by ``cu_seqlens``).
    ``__source__`` is kept for backward compatibility (first document's source).
    """
    buffer = []
    teacher_logprobs_buffer = []
    teacher_entropy_buffer = []
    teacher_margin_buffer = []
    pos_buffer = []
    docid_buffer = []
    doc_boundaries = []
    states = []

    output_seq_len = empty_buffer_state["output_seq_len"]
    n_views = empty_buffer_state["n_views"]
    start_token = empty_buffer_state["start_token"]
    previous_state = empty_buffer_state["it_state"]
    buffer_size = output_seq_len + n_views - 1

    mask_synth_context_loss = bool(empty_buffer_state.get("mask_synth_context_loss", False))
    next_doc_id = 0
    doc_synth_len: Dict[int, int] = {}
    docid_to_source: Dict[int, str] = {}
    has_teacher_logprobs = None
    has_teacher_entropy = False
    has_teacher_margin = False

    block_source: Optional[str] = None

    for i, ((tokens, teacher_logprobs, teacher_entropy, teacher_margin, source), state) in enumerate(iterator):

        # Detect teacher signal presence on first documents.
        if has_teacher_logprobs is None and i < 10:
            if teacher_logprobs is not None:
                has_teacher_logprobs = True
                has_teacher_entropy = teacher_entropy is not None
                has_teacher_margin = teacher_margin is not None
                logger.info(
                    f"[online_data] Teacher signals detected at doc {i}: "
                    f"nll=True entropy={has_teacher_entropy} margin={has_teacher_margin}"
                )
        elif has_teacher_logprobs is None and i >= 10:
            has_teacher_logprobs = False
            logger.info("[online_data] No teacher logprobs in first 10 docs.")

        end_token = start_token
        sample_is_read = False

        # Defensive recovery: if a resumed state carries an out-of-range token
        # offset for the current document, skip this doc instead of crashing the
        # async loader process.
        if start_token >= len(tokens):
            logger.warning(
                f"[online_data] skipping doc with invalid start_token={start_token} "
                f"(len(tokens)={len(tokens)})"
            )
            start_token = 0
            previous_state = state
            continue

        curr_doc_id = next_doc_id
        next_doc_id += 1
        synth_token_len = state.get("synth_token_len", 0) if isinstance(state, dict) else 0
        doc_synth_len[curr_doc_id] = synth_token_len
        pos_in_doc = start_token

        docid_to_source[curr_doc_id] = source

        if len(buffer) > 0 or start_token == 0:
            if start_token == 0:
                doc_boundaries.append(len(buffer))

        while not sample_is_read:
            if block_source is None:
                block_source = source

            assert start_token < len(tokens), (
                f"start_token {start_token} >= len(tokens) {len(tokens)}"
            )
            free_space = buffer_size - len(buffer)
            seq_len = min(free_space, len(tokens) - start_token)
            end_token = start_token + seq_len
            buffer.extend(tokens[start_token:end_token])

            if has_teacher_logprobs:
                if teacher_logprobs is not None:
                    lp_start = max(0, start_token - 1)
                    lp_end = end_token - 1
                    if start_token == 0:
                        teacher_logprobs_buffer.append(0.0)
                        teacher_entropy_buffer.append(0.0)
                        teacher_margin_buffer.append(0.0)
                        if lp_end > lp_start:
                            teacher_logprobs_buffer.extend(teacher_logprobs[lp_start:lp_end])
                            teacher_entropy_buffer.extend(
                                teacher_entropy[lp_start:lp_end]
                                if has_teacher_entropy and teacher_entropy is not None
                                else [0.0] * (lp_end - lp_start)
                            )
                            teacher_margin_buffer.extend(
                                teacher_margin[lp_start:lp_end]
                                if has_teacher_margin and teacher_margin is not None
                                else [0.0] * (lp_end - lp_start)
                            )
                    else:
                        teacher_logprobs_buffer.extend(teacher_logprobs[lp_start:lp_end])
                        teacher_entropy_buffer.extend(
                            teacher_entropy[lp_start:lp_end]
                            if has_teacher_entropy and teacher_entropy is not None
                            else [0.0] * (lp_end - lp_start)
                        )
                        teacher_margin_buffer.extend(
                            teacher_margin[lp_start:lp_end]
                            if has_teacher_margin and teacher_margin is not None
                            else [0.0] * (lp_end - lp_start)
                        )
                else:
                    teacher_logprobs_buffer.extend([0.0] * seq_len)
                    teacher_entropy_buffer.extend([0.0] * seq_len)
                    teacher_margin_buffer.extend([0.0] * seq_len)

            docid_buffer.extend([curr_doc_id] * seq_len)
            pos_buffer.extend(range(pos_in_doc, pos_in_doc + seq_len))
            pos_in_doc += seq_len
            start_token = end_token

            states.append(
                PackTokensState(
                    start_token=start_token,
                    seq_len=seq_len,
                    it_state=previous_state,
                    output_seq_len=output_seq_len,
                    n_views=n_views,
                    mask_synth_context_loss=mask_synth_context_loss,
                )
            )
            assert len(buffer) <= buffer_size, "Buffer overflow"

            if len(buffer) == buffer_size:
                out = np.array(buffer)
                assert out.ndim == 1
                out = np.lib.stride_tricks.sliding_window_view(out, n_views, axis=0)

                if has_teacher_logprobs:
                    def _extra_view(buf):
                        arr = np.array(buf, dtype=np.float32)
                        return np.lib.stride_tricks.sliding_window_view(arr, n_views, axis=0)[:, 1:2]

                    out = np.concatenate([out, _extra_view(teacher_logprobs_buffer)], axis=1)
                    if has_teacher_entropy:
                        out = np.concatenate([out, _extra_view(teacher_entropy_buffer)], axis=1)
                    if has_teacher_margin:
                        out = np.concatenate([out, _extra_view(teacher_margin_buffer)], axis=1)

                if mask_synth_context_loss and n_views >= 2:
                    pos_out = np.lib.stride_tricks.sliding_window_view(
                        np.array(pos_buffer, dtype=np.int32), n_views, axis=0
                    )
                    docid_out = np.lib.stride_tricks.sliding_window_view(
                        np.array(docid_buffer, dtype=np.int32), n_views, axis=0
                    )
                    label_doc_ids = docid_out[:, 1]
                    label_positions = pos_out[:, 1]
                    synth_len_for_label = np.array(
                        [doc_synth_len.get(int(d), 0) for d in label_doc_ids], dtype=np.int32
                    )
                    synth_mask = label_positions < synth_len_for_label
                    if synth_mask.any():
                        out = out.copy()
                        out[synth_mask, 1] = -100

                cu_seqlens = [0]
                for boundary in doc_boundaries:
                    if 0 < boundary < output_seq_len:
                        cu_seqlens.append(boundary)
                cu_seqlens.append(output_seq_len)
                cu_seqlens = sorted(set(cu_seqlens))

                per_doc_sources: List[str] = []
                for seg_idx in range(len(cu_seqlens) - 1):
                    seg_start = cu_seqlens[seg_idx]
                    did = docid_buffer[seg_start]
                    per_doc_sources.append(docid_to_source.get(did, block_source or "unknown"))

                rewinded_idx = start_token - (n_views - 1)
                empty_buffer_state = get_empty_buffer_state(rewinded_idx, states)
                buffer = buffer[output_seq_len:]
                pos_buffer = pos_buffer[output_seq_len:]
                docid_buffer = docid_buffer[output_seq_len:]
                if has_teacher_logprobs:
                    teacher_logprobs_buffer = teacher_logprobs_buffer[output_seq_len:]
                    teacher_entropy_buffer = teacher_entropy_buffer[output_seq_len:]
                    teacher_margin_buffer = teacher_margin_buffer[output_seq_len:]
                assert len(buffer) == (n_views - 1)

                new_doc_boundaries = []
                for boundary in doc_boundaries:
                    shifted = boundary - output_seq_len
                    if shifted >= 0:
                        new_doc_boundaries.append(shifted)
                doc_boundaries = new_doc_boundaries

                live_docids = set(docid_buffer[:n_views - 1]) if (n_views - 1) > 0 else set()
                for old_id in list(docid_to_source):
                    if old_id not in live_docids:
                        del docid_to_source[old_id]

                yield (
                    {"tokens": out, "cu_seqlens": cu_seqlens,
                     "__source__": block_source,
                     "__doc_sources__": per_doc_sources},
                    empty_buffer_state,
                )
                block_source = None

            if start_token == len(tokens):
                start_token = 0
                sample_is_read = True
                previous_state = state


# ── Step 4: batcher that keeps source labels in sync with the shuffle ─────────

def batch_and_shuffle_with_source(
    data_loader: Iterator,
    batch_size: int,
    prefetch_size: int,
    seq_len: int,
    n_views: int,
    state: PrefetchState,
):
    """
    Mirrors lingua.data.batch_and_shuffle_prefetched_sequences exactly,
    but also maintains a ``source_labels_buffer`` that is shuffled with
    the same permutation as ``prefetch_buffer`` so each batch item stays
    paired with its source label.

    Yields
    ------
    ({'tokens': ndarray, 'cu_seqlens': list, 'doc_sources': list}, PrefetchState, List[str])
        ``doc_sources`` in the batch dict is a list-of-lists: one list of
        per-document source names per batch item (aligned with ``cu_seqlens``).
        Third element is a list of source names of length batch_size (first
        document's source per block, kept for backward compatibility).
    """
    # Initialized lazily once we see the first item (actual n_views may
    # differ from n_views if teacher logprobs are present).
    prefetch_buffer = None
    cu_seqlens_buffer: List[Any] = []
    source_labels_buffer: List[Optional[str]] = []
    doc_sources_buffer: List[Optional[List[str]]] = []
    actual_n_views = n_views

    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng_state"]

    seq_idx = state["seq_idx"]
    assert 0 <= seq_idx < prefetch_size, (
        f"seq_idx {seq_idx} out of range [0, {prefetch_size})"
    )

    _rng_state = state["rng_state"]
    _it_state = state["it_state"]

    def _extract_item(item, seq_len_default):
        tokens = item["tokens"]
        cu = item.get("cu_seqlens", [0, seq_len_default])
        src = item.get("__source__", "unknown")
        dsrc = item.get("__doc_sources__", [src] * max(1, len(cu) - 1))
        return tokens, cu, src, dsrc

    # ── fill initial buffer ───────────────────────────────────────────────────
    for i in range(prefetch_size * batch_size):
        item, next_it_state = next(data_loader)
        tokens, cu_seqlens, source, dsources = _extract_item(item, seq_len)

        if prefetch_buffer is None:
            actual_n_views = tokens.shape[-1]
            prefetch_buffer = -1 * np.ones(
                (prefetch_size * batch_size, seq_len, actual_n_views),
                dtype=tokens.dtype,
            )
            cu_seqlens_buffer = [None] * (prefetch_size * batch_size)
            source_labels_buffer = [None] * (prefetch_size * batch_size)
            doc_sources_buffer = [None] * (prefetch_size * batch_size)

        prefetch_buffer[i] = tokens
        cu_seqlens_buffer[i] = cu_seqlens
        source_labels_buffer[i] = source
        doc_sources_buffer[i] = dsources

    # ── shuffle all buffers with a single permutation ─────────────────────────
    perm = rng.permutation(prefetch_size * batch_size)
    prefetch_buffer = prefetch_buffer[perm]
    cu_seqlens_buffer = [cu_seqlens_buffer[p] for p in perm]
    source_labels_buffer = [source_labels_buffer[p] for p in perm]
    doc_sources_buffer = [doc_sources_buffer[p] for p in perm]

    # ── overwrite already-consumed slots for checkpoint resume ────────────────
    for i in range(seq_idx * batch_size):
        item, _ = next(data_loader)
        tokens, cu_seqlens, source, dsources = _extract_item(item, seq_len)
        prefetch_buffer[i] = tokens
        cu_seqlens_buffer[i] = cu_seqlens
        source_labels_buffer[i] = source
        doc_sources_buffer[i] = dsources

    # ── main yield loop ───────────────────────────────────────────────────────
    idx = seq_idx
    while True:
        if idx == prefetch_size - 1:
            _it_state = next_it_state
            _rng_state = rng.bit_generator.state

        state = PrefetchState(
            it_state=_it_state,
            seq_idx=(idx + 1) % prefetch_size,
            rng_state=_rng_state,
            batch_size=batch_size,
            prefetch_size=prefetch_size,
        )

        lo, hi = idx * batch_size, (idx + 1) * batch_size
        batch_tokens = prefetch_buffer[lo:hi].copy()
        batch_cu_seqlens = cu_seqlens_buffer[lo:hi]
        batch_source_labels = source_labels_buffer[lo:hi]
        batch_doc_sources = doc_sources_buffer[lo:hi]

        yield (
            {"tokens": batch_tokens, "cu_seqlens": batch_cu_seqlens,
             "doc_sources": batch_doc_sources},
            state,
            batch_source_labels,
        )

        for i in range(batch_size):
            item, pack_state = next(data_loader)
            tokens, cu_seqlens, source, dsources = _extract_item(item, seq_len)
            prefetch_buffer[lo + i] = tokens
            cu_seqlens_buffer[lo + i] = cu_seqlens
            source_labels_buffer[lo + i] = source
            doc_sources_buffer[lo + i] = dsources

        if idx == prefetch_size - 1:
            next_it_state = pack_state
            perm = rng.permutation(prefetch_size * batch_size)
            prefetch_buffer = prefetch_buffer[perm]
            cu_seqlens_buffer = [cu_seqlens_buffer[p] for p in perm]
            source_labels_buffer = [source_labels_buffer[p] for p in perm]

        idx = (idx + 1) % prefetch_size


# ── Step 5: context-manager that assembles the labeled pipeline ───────────────

@contextlib.contextmanager
def build_online_dataloader(
    state: PrefetchState,
    max_large_ppl: Optional[float] = None,
    min_delta: Optional[float] = None,
):
    """
    Drop-in for lingua.data.build_dataloader, but assembles the
    labeled pipeline (choose_source_labeled → tokenize_labeled →
    pack_tokens_labeled → batch_and_shuffle_with_source).

    Yields an iterator of:
        ({'tokens': ndarray, 'cu_seqlens': list}, PrefetchState, List[str])
    """
    pack_state = state["it_state"]
    tokenizer_state = pack_state["it_state"]
    multi_state = tokenizer_state["it_state"]

    path_to_iter = setup_sources(multi_state)
    try:
        # 1. source-tagged sampler
        data_it = choose_source_labeled(
            source_to_iterator=path_to_iter,
            source_to_state=multi_state["source_to_state"],
            root_dir=multi_state["root_dir"],
            sources=multi_state["sources"],
            rng_state=multi_state["rng_state"],
        )

        # 2. optional quality filter (passes __source__ through unchanged)
        if max_large_ppl is not None or min_delta is not None:
            data_it = filter_by_delta(
                data_it,
                max_large_ppl=max_large_ppl,
                min_delta=min_delta,
            )

        # 3. tokenizer (threads source through via deque)
        data_it = tokenize_labeled(
            data_it,
            tokenizer_state["add_bos"],
            tokenizer_state["add_eos"],
            tokenizer_state["name"],
            tokenizer_state["path"],
            tokenizer_state.get("synth_context_mode"),
            tokenizer_state.get("synth_context_separator", "\n\n"),
            pack_state.get("mask_synth_context_loss", False),
            tokenizer_state.get("synth_context_prob", 1.0),
            tokenizer_state.get("synth_rng_state"),
            tokenizer_state.get("max_length"),
            nll_field=tokenizer_state.get("nll_field"),
            entropy_field=tokenizer_state.get("entropy_field"),
            margin_field=tokenizer_state.get("margin_field"),
        )

        # 4. packer (records block source)
        data_it = pack_tokens_labeled(data_it, pack_state)

        # 5. batcher with synchronized shuffle
        data_it = batch_and_shuffle_with_source(
            data_loader=data_it,
            seq_len=pack_state["output_seq_len"],
            n_views=pack_state["n_views"],
            batch_size=state["batch_size"],
            prefetch_size=state["prefetch_size"],
            state=state,
        )

        yield data_it

    finally:
        for it in path_to_iter.values():
            it.close()


# ── Step 6: the reweightable wrapper ─────────────────────────────────────────

class ReweightableDataLoader:
    """
    Wraps build_online_dataloader to allow online source-weight updates.

    On each call to set_source_weights(), the weights are patched into
    the nested PrefetchState and the underlying async subprocess is
    restarted from the current data position.  The restart takes ~1-2 s
    and should only be called every K steps (e.g. K=1000).

    Parameters
    ----------
    args : DataArgs
    state : PrefetchState
        Initial state from init_dataloader_state_from_args().

    Usage
    -----
        loader = ReweightableDataLoader(args.data, train_state.data_loader_state)
        context_stack.callback(loader.close)

        # in the training loop:
        batch, train_state.data_loader_state, source_labels = next(loader)

        # every K steps:
        loader.set_source_weights({"dclm_shuffled": 0.5, "math_shuffled": 0.3, ...})
    """

    def __init__(self, args: DataArgs, state: PrefetchState):
        self.args = args
        self._state = state
        self._ctx = None
        self._iter = None
        self._build()

    # ── internal ──────────────────────────────────────────────────────────────

    def _build(self):
        """(Re)construct the dataloader from the current state."""
        self._teardown()
        builder = partial(
            build_online_dataloader,
            self._state,
            max_large_ppl=self.args.max_large_ppl,
            min_delta=self.args.min_delta,
        )
        if self.args.load_async:
            self._ctx = async_iterator(self.args.prefetch_size, builder)
        else:
            self._ctx = builder()
        self._iter = self._ctx.__enter__()

    def _teardown(self):
        if self._ctx is not None:
            try:
                self._ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._ctx = None
            self._iter = None

    # ── public ────────────────────────────────────────────────────────────────

    def __next__(self):
        """
        Returns
        -------
        (batch_dict, PrefetchState, List[str])
            batch_dict   : {'tokens': ndarray, 'cu_seqlens': list}
            PrefetchState: updated state (assign to train_state.data_loader_state)
            source_labels: list of source names, one per item in the batch
        """
        batch_dict, self._state, source_labels = next(self._iter)
        return batch_dict, self._state, source_labels

    def set_source_weights(self, new_weights: Dict[str, float]):
        """
        Update source sampling weights and restart the data pipeline.

        Patches the weights into the nested PrefetchState, then rebuilds
        the underlying async subprocess from the current data position.

        State nesting (all plain dicts at runtime):
            PrefetchState
              ["it_state"]  PackTokensState
                ["it_state"]  TokenizerState
                  ["it_state"]  MultiChoiceState
                    ["sources"]  ← patched here

        Parameters
        ----------
        new_weights : Dict[str, float]
            Must contain the same keys as the original sources dict.
            Values need not sum to 1; choose_source_labeled normalises them.
        """
        # Validate keys match to catch mismatches early.
        existing_sources = self._state["it_state"]["it_state"]["it_state"]["sources"]
        unknown = set(new_weights) - set(existing_sources)
        if unknown:
            raise ValueError(
                f"set_source_weights: unknown source keys {unknown}. "
                f"Valid keys: {set(existing_sources)}"
            )

        self._state["it_state"]["it_state"]["it_state"]["sources"] = dict(new_weights)
        if _is_rank0():
            logger.info(
                f"[ReweightableDataLoader] Rebuilding dataloader with new weights: "
                + ", ".join(f"{k}={v:.4f}" for k, v in new_weights.items())
            )
        self._build()

    def close(self):
        """Release the async subprocess. Called automatically via context_stack.callback."""
        self._teardown()


# ── Config and controller for excess-loss reweighting ────────────────────────

@dataclass
class OnlineReweightingArgs:
    """
    Config for the online excess-loss reweighting controller.
    Add as a field to TrainArgs and set enabled=True to activate.
    """
    enabled: bool = False

    # Log full per-source stats (raw_excess, EMA, proposed weights) just before
    # each weight update fires.  Useful for debugging controller behaviour.
    debug: bool = False

    # How many optimizer steps between weight updates.
    update_interval: int = 1000

    # EMA smoothing factor for excess loss (higher = smoother / slower).
    ema_beta: float = 0.8

    # Reweighting temperature: higher = more aggressive tilting.
    eta: float = 0.25

    # Per-source weight bounds as multiples of the base (Dolmino) weight.
    floor_mul: float = 0.5
    ceil_mul: float = 2.0

    # If True, use relative excess:
    #   (student_nll - expert_nll) / max(|expert_nll|, relative_excess_eps)
    # This removes global loss-scale drift and compares headroom proportionally.
    relative_excess: bool = False
    relative_excess_eps: float = 1e-6

    # If True, z-score the centered controller signal across sources before exp().
    # This makes eta comparable even when per-source excess magnitudes are tiny.
    zscore_signal: bool = False
    zscore_eps: float = 1e-6

    # If True, allow teacher/expert routing to build online reweighting signals
    # while keeping the main training loss as plain CE (no best-expert token loss).
    teacher_routing_only: bool = False


class ExcessLossController:
    """
    Accumulates per-source student NLL and expert NLL over a window of
    K steps, then computes new sampling weights via:

        w_s  ∝  w_s^base · exp(η · ema_excess_s)
        clipped to [floor_mul · w_s^base, ceil_mul · w_s^base]
        then renormalised to sum to 1.

    Intended use
    ------------
        ctrl = ExcessLossController(base_weights, args.online_reweighting)

        # inside the training loop, after each batch:
        ctrl.accumulate(source_labels, student_nll, expert_nll, n_valid_tokens)

        # every K steps:
        new_weights = ctrl.maybe_update(step)
        if new_weights is not None:
            online_loader.set_source_weights(new_weights)
            # log new_weights …

    Parameters
    ----------
    base_weights : Dict[str, float]
        The fixed reference mix (Dolmino weights).  Clip bounds and the
        exponential update are defined relative to these values.
    args : OnlineReweightingArgs
    """

    def __init__(self, base_weights: Dict[str, float], args: OnlineReweightingArgs):
        self.base_weights = dict(base_weights)
        self.args = args
        self.sources = list(base_weights.keys())

        # EMA state — persists across update windows.
        self.ema_excess: Dict[str, float] = {s: 0.0 for s in self.sources}
        self._ema_init: Dict[str, bool] = {s: False for s in self.sources}

        # Current live weights (start at base).
        self.current_weights: Dict[str, float] = dict(base_weights)

        self._reset()

    # ── state persistence (survives job restarts) ─────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        return {
            "ema_excess": dict(self.ema_excess),
            "ema_init": dict(self._ema_init),
            "current_weights": dict(self.current_weights),
            "token_count": dict(self._token_count),
            "student_nll_sum": dict(self._student_nll_sum),
            "expert_nll_sum": dict(self._expert_nll_sum),
        }

    def load_state_dict(self, state: Dict[str, Any]):
        if state is None:
            return
        for s in self.sources:
            if s in state.get("ema_excess", {}):
                self.ema_excess[s] = state["ema_excess"][s]
            if s in state.get("ema_init", {}):
                self._ema_init[s] = state["ema_init"][s]
            if s in state.get("current_weights", {}):
                self.current_weights[s] = state["current_weights"][s]
            if s in state.get("token_count", {}):
                self._token_count[s] = state["token_count"][s]
            if s in state.get("student_nll_sum", {}):
                self._student_nll_sum[s] = state["student_nll_sum"][s]
            if s in state.get("expert_nll_sum", {}):
                self._expert_nll_sum[s] = state["expert_nll_sum"][s]

    # ── accumulation ──────────────────────────────────────────────────────────

    def _reset(self):
        self._token_count: Dict[str, int] = {s: 0 for s in self.sources}
        self._student_nll_sum: Dict[str, float] = {s: 0.0 for s in self.sources}
        self._expert_nll_sum: Dict[str, float] = {s: 0.0 for s in self.sources}

    def accumulate(
        self,
        source_labels: List[str],
        per_seq_student_nll: Optional[List[float]] = None,
        per_seq_expert_nll: Optional[List[float]] = None,
        per_seq_tokens: Optional[List[float]] = None,
        # Fallbacks used when per-seq data is unavailable.
        student_nll: float = 0.0,
        n_valid_tokens: int = 1,
    ):
        """
        Record one batch's contribution, preferring per-sequence NLL arrays
        (one value per batch item) for accurate per-source attribution.

        If ``per_seq_student_nll`` is provided, each sequence's NLL is
        attributed only to its own source rather than spreading the batch mean
        across all sources.  This fixes the uniform-excess problem that occurs
        when all sources in a mixed batch receive the same batch-average NLL.

        Falls back to proportional batch-mean splitting when per-seq data is
        unavailable (legacy behaviour).
        """
        batch_size = len(source_labels)
        if batch_size == 0:
            return

        if per_seq_student_nll is not None and len(per_seq_student_nll) == batch_size:
            # Per-sequence path: accurate per-source attribution.
            for i, src in enumerate(source_labels):
                if src not in self._token_count:
                    continue
                tok = int(per_seq_tokens[i]) if per_seq_tokens is not None else 1
                tok = max(tok, 1)
                self._token_count[src] += tok
                self._student_nll_sum[src] += per_seq_student_nll[i] * tok
                if per_seq_expert_nll is not None:
                    self._expert_nll_sum[src] += per_seq_expert_nll[i] * tok
                else:
                    # Raw-loss mode (no expert): use expert_nll=0 so
                    # excess = student_nll - 0 = student_nll.
                    self._expert_nll_sum[src] += 0.0
        else:
            # Fallback: split batch-level stats proportionally across sources.
            from collections import Counter
            counts = Counter(s for s in source_labels if s in self._token_count)
            if not counts:
                return

            total_items = sum(counts.values())
            # If per-seq expert values are available but per-seq student values are
            # not, use the batch-average expert NLL in fallback instead of copying
            # student NLL into expert NLL (which would force excess=0 everywhere).
            expert_mean = 0.0
            if per_seq_expert_nll is not None and len(per_seq_expert_nll) > 0:
                expert_mean = float(sum(per_seq_expert_nll) / len(per_seq_expert_nll))
            for source, n_items in counts.items():
                tok = int(n_valid_tokens * n_items / total_items)
                self._token_count[source] += tok
                self._student_nll_sum[source] += student_nll * tok
                if per_seq_expert_nll is not None:
                    self._expert_nll_sum[source] += expert_mean * tok
                else:
                    # Raw-loss mode (no expert): expert_nll=0, so excess is
                    # driven directly by student NLL.
                    self._expert_nll_sum[source] += 0.0

    # ── update ────────────────────────────────────────────────────────────────

    def maybe_update(self, step: int) -> Optional[Dict[str, float]]:
        """
        If ``step`` is a multiple of ``update_interval``, compute new
        weights and return them; otherwise return None.
        """
        if step == 0 or (step % self.args.update_interval) != 0:
            return None
        return self._compute_weights(step)

    def _compute_weights(self, step: int) -> Dict[str, float]:
        # ── sync accumulation buffers across ranks ────────────────────────────
        # All ranks call maybe_update() together at optimizer-step boundaries,
        # so this all_reduce is always called collectively — no hang risk.
        # We batch all sources into a single tensor (shape: [n_sources, 3]) to
        # minimise collective overhead: columns are [token_count, student_nll_sum,
        # expert_nll_sum].
        import torch
        is_dist = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        rank = torch.distributed.get_rank() if is_dist else 0

        if is_dist:
            n = len(self.sources)
            buf = torch.zeros(n, 3, dtype=torch.float64, device="cuda")
            for i, s in enumerate(self.sources):
                buf[i, 0] = self._token_count[s]
                buf[i, 1] = self._student_nll_sum[s]
                buf[i, 2] = self._expert_nll_sum[s]
            torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
            for i, s in enumerate(self.sources):
                self._token_count[s] = int(buf[i, 0].item())
                self._student_nll_sum[s] = buf[i, 1].item()
                self._expert_nll_sum[s] = buf[i, 2].item()

        raw_excess: Dict[str, float] = {}

        for s in self.sources:
            n = self._token_count[s]
            if n == 0:
                # No data seen; hold EMA value (neutral if uninitialised).
                raw_excess[s] = self.ema_excess[s] if self._ema_init[s] else 0.0
            else:
                avg_student = self._student_nll_sum[s] / n
                avg_expert = self._expert_nll_sum[s] / n
                if self.args.relative_excess:
                    denom = max(abs(avg_expert), self.args.relative_excess_eps)
                    raw_excess[s] = (avg_student - avg_expert) / denom
                else:
                    raw_excess[s] = avg_student - avg_expert

            # Update EMA.
            if not self._ema_init[s]:
                self.ema_excess[s] = raw_excess[s]
                self._ema_init[s] = True
            else:
                self.ema_excess[s] = (
                    self.args.ema_beta * self.ema_excess[s]
                    + (1.0 - self.args.ema_beta) * raw_excess[s]
                )

        # Multiplicative update anchored to base weights.
        # Mean-center the excess signal so only relative differences drive reweighting.
        # Without this, all sources share the same absolute excess (~0.10 nats at midtraining)
        # and exp(eta * excess) ≈ constant for all sources, cancelling in renormalisation.
        mean_excess = sum(self.base_weights[s] * self.ema_excess[s] for s in self.sources)
        centered_excess = {s: self.ema_excess[s] - mean_excess for s in self.sources}
        if self.args.zscore_signal:
            var = sum(self.base_weights[s] * (centered_excess[s] ** 2) for s in self.sources)
            std = float(np.sqrt(max(var, self.args.zscore_eps ** 2)))
            centered_excess = {s: centered_excess[s] / std for s in self.sources}
        raw_new = {s: 0.0 for s in self.sources}
        clipped = {s: 0.0 for s in self.sources}
        if (not is_dist) or rank == 0:
            raw_new = {
                s: self.base_weights[s] * float(np.exp(self.args.eta * centered_excess[s]))
                for s in self.sources
            }

            # Clip to [floor, ceiling] relative to base.
            clipped = {
                s: float(np.clip(
                    raw_new[s],
                    self.args.floor_mul * self.base_weights[s],
                    self.args.ceil_mul * self.base_weights[s],
                ))
                for s in self.sources
            }

            # Renormalise.
            total = sum(clipped.values())
            self.current_weights = {s: clipped[s] / total for s in self.sources}

        if is_dist:
            weights_buf = torch.zeros(len(self.sources), dtype=torch.float64, device="cuda")
            ema_buf = torch.zeros(len(self.sources), dtype=torch.float64, device="cuda")
            if rank == 0:
                for i, s in enumerate(self.sources):
                    weights_buf[i] = self.current_weights[s]
                    ema_buf[i] = self.ema_excess[s]
            torch.distributed.broadcast(weights_buf, src=0)
            torch.distributed.broadcast(ema_buf, src=0)
            for i, s in enumerate(self.sources):
                self.current_weights[s] = float(weights_buf[i].item())
                self.ema_excess[s] = float(ema_buf[i].item())

        if self.args.debug and ((not is_dist) or rank == 0):
            logger = logging.getLogger(__name__)
            lines = [f"[OnlineReweighting DEBUG] pre-update dump at step={step}"]
            lines.append(f"  {'source':<24} {'tokens':>10} {'student_nll':>12} {'expert_nll':>11} {'raw_excess':>11} {'ema_excess':>11} {'base_w':>8} {'raw_new':>8} {'clipped':>8} {'final_w':>8}")
            for s in self.sources:
                n = self._token_count[s]  # already reset? no — _reset() called after
                tag = s.replace("_shuffled", "")
                lines.append(
                    f"  {tag:<24} {n:>10} "
                    f"{(self._student_nll_sum[s]/n if n else 0):>12.4f} "
                    f"{(self._expert_nll_sum[s]/n if n else 0):>11.4f} "
                    f"{raw_excess[s]:>11.4f} "
                    f"{self.ema_excess[s]:>11.4f} "
                    f"{self.base_weights[s]:>8.4f} "
                    f"{raw_new[s]:>8.4f} "
                    f"{clipped[s]:>8.4f} "
                    f"{self.current_weights[s]:>8.4f}"
                )
            logger.info("\n".join(lines))

        self._reset()
        return dict(self.current_weights)

    # ── logging helpers ───────────────────────────────────────────────────────

    def metrics_dict(self) -> Dict[str, float]:
        """
        Returns a flat dict of current controller state suitable for
        passing to metric_logger.  Keys use the prefix
        ``online_ctrl/``.

        Reported every logging step (not just at update boundaries):
          online_ctrl/weight/{src}          current sampling weight
          online_ctrl/ema_excess/{src}      EMA-smoothed excess loss
          online_ctrl/delta_from_base/{src} drift from Dolmino base weight
          online_ctrl/student_nll/{src}     window-avg student NLL so far
          online_ctrl/expert_nll/{src}      window-avg expert NLL so far
          online_ctrl/raw_excess/{src}      window-avg raw excess (student - expert)
          online_ctrl/tokens_seen/{src}     tokens accumulated this window
        """
        out: Dict[str, float] = {}
        for s in self.sources:
            tag = s.replace("_shuffled", "")
            n = self._token_count[s]
            out[f"online_ctrl/weight/{tag}"] = self.current_weights[s]
            out[f"online_ctrl/ema_excess/{tag}"] = self.ema_excess[s]
            out[f"online_ctrl/delta_from_base/{tag}"] = (
                self.current_weights[s] - self.base_weights[s]
            )
            out[f"online_ctrl/tokens_seen/{tag}"] = float(n)
            # IMPORTANT: Always emit the same metric keys on every rank.
            # train.py calls dist_mean_dict() over this dict, which performs
            # one all_reduce per key. Rank-dependent key presence causes
            # collective sequence mismatch and watchdog timeouts.
            if n > 0:
                avg_student = self._student_nll_sum[s] / n
                avg_expert = self._expert_nll_sum[s] / n
            else:
                avg_student = 0.0
                avg_expert = 0.0
            out[f"online_ctrl/student_nll/{tag}"] = avg_student
            out[f"online_ctrl/expert_nll/{tag}"] = avg_expert
            out[f"online_ctrl/raw_excess/{tag}"] = avg_student - avg_expert
        return out
