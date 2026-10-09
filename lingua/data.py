# Copyright (c) Meta Platforms, Inc. and affiliates.

import contextlib
from copy import deepcopy
from functools import lru_cache, partial
import json
import zlib
from dataclasses import dataclass, field
from multiprocessing import Process, Queue, Event
from queue import Full, Empty
from multiprocessing.synchronize import Event as EventClass
import os
from pathlib import Path
from typing import Dict, Any, Iterator, Optional, TypedDict, Union, Tuple, List
from lingua.tokenizer import build_tokenizer, TokenizerArgs
import numpy as np
import logging
logger = logging.getLogger()

"""
This file contains all code necessary for text data loading from preshuffled jsonl chunks.
For example if given the following files with a world size of 8

/path/to/arxiv:
arxiv.chunk.00.jsonl (Contains many lines of {"text":...} or {"content":...})
arxiv.chunk.01.jsonl
arxiv.chunk.02.jsonl
arxiv.chunk.03.jsonl

/path/to/wikipedia:
wikipedia.chunk.00.jsonl
wikipedia.chunk.01.jsonl
wikipedia.chunk.02.jsonl
wikipedia.chunk.03.jsonl

Step (1) => infinite_block_jsonl_iterator
2 workers will read each jsonl chunk (world_size = 8 distributed over 4 workers) from each source.
Each worker will read 1 line and skip the next, therefore workers on the same file read in an interleaved manner.

Step (2) => multi_choice_iterator
At every iteration, a source is sampled randomly given some weights

Step (3) => tokenizer and pack_tokens
Reads sequences until reaching seq_len tokens and yields a numpy array of shape (seq_len, n_views)

Step (4) => prefetch_data_loader
Prefetches batches in advance and shuffles them to reduce correlation, yields a numpy array of shape (batch_size, seq_len, n_views)

This create a nested iterator structure where each iterator is responsible for a specific task:
    [ [ [ [ [ (1) read document ] -> (2) sample source ] -> (3) tokenize ] -> (4) tokenize and build sequence of fixed seq_len ] -> (5) prefetch batches ]

Each iterator returns a tuple (output, state) where state contains all the info necessary to resume from the last output.

build_mixed_token_packing_dataloader creates the states and return an iterator that does everything above

build_seperate_token_packing_dataloader does the same thing but swaps step 2 and 3

Both can be called with a resume_state to resume from any given position deterministically
"""

TRAIN_DATA_FILE_PATTERN = "*.chunk.*.jsonl"
TRAIN_DATA_FILE_PATTERN_SHARD = "*shard*.jsonl"  # Alternative pattern for shard-based naming

class JSONLState(TypedDict):
    """Represents the current state of a JSON line reader.

    Attributes:
        content (Dict): The JSON content of the line.
        file_path (str): The path to the JSONL file.
        position (int): The file position after reading the line (in bytes).
        window (int): The window size used for iteration.
        offset (int): The offset used for iteration.
        current_iter (Optional[int]): Number of iterations over the jsonl file (for infinite iteration).
        line_idx (int): Sequential epochs: absolute index of the next line of the file.
            Shuffled epochs: index into this epoch's permutation of the reader's lines.
        max_lines (Optional[int]): Only the first max_lines lines of the file are used (an
            epoch ends there); None = whole file. Set from DataArgs.source_max_docs.
        shuffle (bool): Read epochs >= 1 in a fresh seeded permutation (DataArgs.shuffle_each_epoch).
        shuffle_seed (int): Seed for those permutations.
    """

    file_path: str
    position: int
    block_size: int
    offset: int
    current_iter: int
    line_idx: int
    max_lines: Optional[int]
    shuffle: bool
    shuffle_seed: int


class MultiChoiceState(TypedDict):
    """Represents the current state of a Multi choice iterator.

    Attributes:
        root_dir: path to dataset root directory
        sources Dict[str, float]: Dict from subdirectory to the weight used for sampling
        source_states: Dict[str, Any] Dict from source to iterator state
        rng_state: dict numpy bit generator state used to resume rng
    """

    root_dir: str
    sources: Dict[str, float]
    source_to_state: Dict[str, Any]
    rng_state: Dict[str, Any]


class TokenizerState(TypedDict):
    it_state: Any
    name: str
    add_bos: bool
    add_eos: bool
    path: Optional[str]
    synth_context_mode: Optional[str]
    synth_context_separator: str
    synth_token_len: int  # Number of tokens in synthetic prefix (for loss masking)
    synth_context_prob: float  # Probability of augmenting with synthetic context
    synth_rng_state: Dict[str, Any]  # RNG state for reproducible synth context sampling
    max_length: Optional[int]  # Maximum sequence length for tokenization truncation
    # Precomputed teacher signal field names (from precompute_teacher_logprobs.py)
    nll_field: Optional[str]      # e.g. "domain_expert_nll"
    entropy_field: Optional[str]  # e.g. "domain_expert_entropy"
    margin_field: Optional[str]   # e.g. "domain_expert_margin"


class PackTokensState(TypedDict):
    """Represents the current state of a packing iterator.

    Attributes:
        start_token: int index to start reading from in the current sequence
        output_seq_len: int Length of sequences to output
        n_views: dict int Number of views to output. Each view is the same sequence but shifted by 1 from the previous
        mask_synth_context_loss: bool whether to mask loss on synthetic context tokens
    """

    start_token: int
    it_state: Any
    output_seq_len: int
    n_views: int
    seq_len: int
    mask_synth_context_loss: bool


class PrefetchState(TypedDict):
    """Represents the current state of a prefetching iterator.

    Attributes:
        prefetch_buffer: numpy array to store prefetched data
        seq_idx: int index of the current sequence to resume from
        rng_state: dict numpy bit generator state used to resume rng
    """

    it_state: Any
    seq_idx: int
    rng_state: Dict[str, Any]
    prefetch_size: int
    batch_size: int


def _jsonl_state(file_path, position, block_size, offset, current_iter, line_idx, max_lines, shuffle, shuffle_seed):
    return JSONLState(
        file_path=file_path,
        position=position,
        block_size=block_size,
        offset=offset,
        current_iter=current_iter,
        line_idx=line_idx,
        max_lines=max_lines,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
    )


@lru_cache(maxsize=None)
def _reader_line_offsets(file_path: str, block_size: int, offset: int, max_lines: Optional[int]) -> np.ndarray:
    """Byte offsets of the lines this reader owns (line i with i % block_size == offset),
    among the first `max_lines` complete lines of the file. Built once per process."""
    starts = [np.zeros(1, dtype=np.int64)]
    n_found, pos = 0, 0
    with open(file_path, "rb", buffering=0) as f:
        while max_lines is None or n_found < max_lines:
            block = f.read(64 << 20)
            if not block:
                break
            nl = np.flatnonzero(np.frombuffer(block, dtype=np.uint8) == 10).astype(np.int64) + pos + 1
            starts.append(nl)
            n_found += len(nl)
            pos += len(block)
    line_starts = np.concatenate(starts)[:-1]  # start of every complete line (last entry = EOF)
    if max_lines is not None:
        line_starts = line_starts[:max_lines]
    return line_starts[offset::block_size]


def read_jsonl(
    file_path: str,
    position: int,
    block_size: int,
    offset: int,
    current_iter: int,
    line_idx: Optional[int] = None,
    max_lines: Optional[int] = None,
    shuffle: bool = False,
    shuffle_seed: int = 0,
):
    """Iterates over a JSON Lines file, yielding a line every `block_size` lines with an offset

    Example : If block_size = 3, offset = 1, iterator will yield lines 1 4 7 10 ...
    Example : If block_size = 2, offset = 0, iterator will yield lines 0 2 4 6 ...

    Only the first `max_lines` lines are used when it is set. With `shuffle`, epochs
    after the first (current_iter >= 1) visit this reader's lines in a permutation
    seeded by (shuffle_seed, file name, offset, epoch); the first epoch is read in file
    order (prepared data is already shuffled).

    Args:
        file_path (str): Path to the JSONL file.
        position (int): The file position (in bytes) from which to start reading (sequential epochs).
        block_size (int): The number of lines to skip between yields
        offset (int): The initial number of lines skiped
        line_idx (int): See JSONLState; None = legacy state without it.

    Yields:
        JSONLState: Represents the state of each line read according to window and offset.
    """
    if (offset < 0) or (offset >= block_size):
        raise RuntimeError(f"JSONL iterator offset value is invalid")

    def state(position, line_idx):
        return _jsonl_state(file_path, position, block_size, offset, current_iter,
                            line_idx, max_lines, shuffle, shuffle_seed)

    if shuffle and current_iter >= 1:
        offsets = _reader_line_offsets(file_path, block_size, offset, max_lines)
        seed = (shuffle_seed, zlib.crc32(os.path.basename(file_path).encode()), offset, current_iter)
        perm = np.random.default_rng(seed).permutation(len(offsets))
        with open(file_path, "rb") as file:
            for k in range(line_idx or 0, len(perm)):
                file.seek(int(offsets[perm[k]]))
                line = file.readline()
                try:
                    content = json.loads(line.decode("utf-8", errors="ignore"))
                except json.JSONDecodeError:
                    continue
                yield content, state(0, k + 1)
        return

    # Sequential epoch. `idx` counts the lines read from the start of the file.
    if line_idx is None:
        # Legacy state: only the line number modulo block_size is known, which is
        # all that matters without max_lines.
        line_idx = offset + 1 if position > 0 else 0
    idx = line_idx
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        file.seek(position)
        while max_lines is None or idx < max_lines:
            line = file.readline()
            if not line:
                if max_lines is not None:
                    logger.warning(f"[data] {file_path} has only {idx} lines, fewer than max_lines={max_lines}")
                break
            idx += 1
            if (idx - 1) % block_size == offset:
                try:
                    content = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed lines (can happen with shuffled data)
                    continue
                # We return state that will allow resuming from this position
                yield content, state(file.tell(), idx)


def loop_on_jsonl(
    file_path: str,
    position: int,
    block_size: int,
    offset: int,
    current_iter: int,
    line_idx: Optional[int] = None,
    max_lines: Optional[int] = None,
    shuffle: bool = False,
    shuffle_seed: int = 0,
):
    """Makes the block jsonl iterator infinite and updates n_iter counter"""
    it = None
    try:
        while True:
            it = read_jsonl(file_path, position, block_size, offset, current_iter,
                            line_idx, max_lines, shuffle, shuffle_seed)
            n = 0
            for content, jsonl_state in it:
                n += 1
                yield content, jsonl_state
            if n == 0 and (line_idx in (None, 0)) and position == 0:
                raise RuntimeError(f"{file_path} (offset {offset}/{block_size}, max_lines={max_lines}) yields no lines")
            logger.info(f"[data] finished epoch {current_iter} of {os.path.basename(file_path)} "
                        f"(reader {offset}/{block_size}, max_lines={max_lines}, shuffle={shuffle})")
            current_iter += 1
            position = 0
            line_idx = 0
    finally:
        if it is not None:
            it.close()


def filter_by_delta(
    iterator: Iterator,
    max_large_ppl: Optional[float] = None,
    min_delta: Optional[float] = None,
):
    """
    Filter documents by delta (small_ppl - large_ppl) and quality (large_ppl).

    Args:
        iterator: Iterator yielding (content, state) pairs where content has
                  'large_ppl', 'small_ppl', 'delta' fields from merged scoring
        max_large_ppl: Quality filter - only keep docs where large_ppl <= this
                       (lower large_ppl = large model understands it = quality content)
        min_delta: Only keep docs where delta >= this threshold
                   (higher delta = small model struggles more = more learnable)

    Yields:
        (content, state) pairs that pass the filter criteria
    """
    if max_large_ppl is None and min_delta is None:
        # No filtering, pass through
        yield from iterator
        return

    filtered_count = 0
    passed_count = 0

    for content, state in iterator:
        # Check if this doc has valid delta scoring fields (keys exist AND values are not None)
        large_ppl = content.get('large_ppl')
        small_ppl = content.get('small_ppl')
        delta = content.get('delta')

        has_valid_scoring = (large_ppl is not None and
                            small_ppl is not None and
                            delta is not None)

        if not has_valid_scoring:
            # No valid scoring data - pass through (allows mixing scored/unscored data)
            yield content, state
            passed_count += 1
            continue

        # Apply quality filter (max_large_ppl)
        if max_large_ppl is not None and large_ppl > max_large_ppl:
            filtered_count += 1
            continue

        # Apply delta filter (min_delta)
        if min_delta is not None and delta < min_delta:
            filtered_count += 1
            continue

        # Passed all filters
        yield content, state
        passed_count += 1

        # Log progress periodically
        total = filtered_count + passed_count
        if total % 10000 == 0:
            logger.info(f"Delta filter: {passed_count}/{total} docs passed "
                       f"({100*passed_count/total:.1f}%)")


def extract_text_and_teacher_logprobs_from_content(
    content: Dict[str, Any],
    synth_context_mode: Optional[str] = None,
    synth_context_separator: str = "\n\n",
    return_synth_boundary: bool = False,
    nll_field: Optional[str] = None,
    entropy_field: Optional[str] = None,
    margin_field: Optional[str] = None,
) -> Tuple[Union[str, Tuple[str, str]], Optional[List[int]], Optional[List[float]], Optional[List[float]], Optional[List[float]]]:
    """
    Extracts text and optionally teacher logprobs from content dict.

    Standard format: {"text": "..."} or {"content": "..."}
    Synthetic format: {"original_text": "...", "contexts": {"style1": {"synthetic_context": "..."}, ...}}
    Legacy teacher format: {"teacher_token_ids": [...], "teacher_token_logprobs": [...]}
    Precomputed expert format (nll_field/entropy_field/margin_field): {"domain_expert_nll": [...], ...}

    Args:
        content: The JSON content dict
        synth_context_mode: How to handle synthetic contexts
        synth_context_separator: Separator between text chunks
        return_synth_boundary: If True, return (full_text, synth_prefix_text) for boundary tracking
        nll_field: JSONL field name for precomputed NLL (e.g. "domain_expert_nll").
            If None, falls back to legacy "teacher_token_logprobs".
        entropy_field: JSONL field name for precomputed entropy (e.g. "domain_expert_entropy").
        margin_field: JSONL field name for precomputed margin (e.g. "domain_expert_margin").

    Returns:
        Tuple of (text, teacher_token_ids, teacher_token_logprobs, teacher_entropy, teacher_margin)
        where the last four may be None if not present in the document.
    """
    # Precomputed expert signals (new format)
    if nll_field is not None:
        teacher_token_logprobs = content.get(nll_field, None)
        teacher_token_ids = None  # Not stored in precomputed format; length checked by caller
    else:
        # Legacy format with explicit token IDs for alignment verification
        teacher_token_ids = content.get("teacher_token_ids", None)
        teacher_token_logprobs = content.get("teacher_token_logprobs", None)

    teacher_entropy = content.get(entropy_field, None) if entropy_field else None
    teacher_margin = content.get(margin_field, None) if margin_field else None

    return (
        _extract_text_only(content, synth_context_mode, synth_context_separator, return_synth_boundary),
        teacher_token_ids,
        teacher_token_logprobs,
        teacher_entropy,
        teacher_margin,
    )


def _extract_text_only(
    content: Dict[str, Any],
    synth_context_mode: Optional[str] = None,
    synth_context_separator: str = "\n\n",
    return_synth_boundary: bool = False,
) -> Union[str, Tuple[str, str]]:
    """
    Extracts text from content dict, handling both standard and synthetic context formats.

    Standard format: {"text": "..."} or {"content": "..."}
    Synthetic format: {"original_text": "...", "contexts": {"style1": {"synthetic_context": "..."}, ...}}

    Args:
        content: The JSON content dict
        synth_context_mode: How to handle synthetic contexts:
            - None: Use standard text/content field only
            - "prepend": Prepend synthetic contexts before original text
            - "append": Append synthetic contexts after original text
        synth_context_separator: Separator between text chunks
        return_synth_boundary: If True, return (full_text, synth_prefix_text) for boundary tracking

    Returns:
        If return_synth_boundary is False: the combined text string
        If return_synth_boundary is True: (full_text, synth_prefix_text) where synth_prefix_text
            is the part before original text (for computing token boundary)
    """
    # Check if this is synthetic context format
    # Support two formats:
    #   1. "contexts" format: {"original_text": "...", "contexts": {"style1": {"synthetic_context": "..."}, ...}}
    #   2. "generations" format (blind mode): {"original_text": "...", "generations": [{"synthetic_context": "...", "ok": true}, ...]}
    has_contexts = "original_text" in content and ("contexts" in content or "generations" in content)

    if has_contexts:
        original = content["original_text"]

        if synth_context_mode is None:
            # Just return original text (baseline behavior)
            if return_synth_boundary:
                return original, ""  # No synthetic prefix
            return original

        # Extract all synthetic contexts, excluding infeasible ones
        synthetic_texts = []

        if "contexts" in content:
            # Format 1: contexts dict with style_name -> style_data
            for style_name, style_data in content["contexts"].items():
                if isinstance(style_data, dict) and "synthetic_context" in style_data:
                    ctx = style_data["synthetic_context"]
                    # Skip contexts marked as infeasible/unfeasible
                    if ctx.strip() in ("INFEASIBLE", "UNFEASIBLE"):
                        continue
                    synthetic_texts.append(ctx)
        elif "generations" in content:
            # Format 2: generations list from blind mode
            for gen in content["generations"]:
                if isinstance(gen, dict) and gen.get("ok", False) and "synthetic_context" in gen:
                    ctx = gen["synthetic_context"]
                    # Skip contexts marked as infeasible/unfeasible
                    if ctx.strip() in ("INFEASIBLE", "UNFEASIBLE"):
                        continue
                    synthetic_texts.append(ctx)

        if not synthetic_texts:
            if return_synth_boundary:
                return original, ""
            return original

        combined_synthetic = synth_context_separator.join(synthetic_texts)

        if synth_context_mode == "prepend":
            synth_prefix = combined_synthetic + synth_context_separator
            full_text = synth_prefix + original
            if return_synth_boundary:
                return full_text, synth_prefix
            return full_text
        elif synth_context_mode == "append":
            full_text = original + synth_context_separator + combined_synthetic
            if return_synth_boundary:
                return full_text, ""  # No prefix to mask when appending
            return full_text
        else:
            raise ValueError(f"Unknown synth_context_mode: {synth_context_mode}")

    # Standard format
    if "text" in content:
        text = content["text"]
    elif "content" in content:
        text = content["content"]
    elif "original_text" in content:
        # Fallback for data with original_text but no contexts/generations
        text = content["original_text"]
    else:
        # Skip malformed lines (can happen with shuffled data that got cut mid-line)
        # Return empty string to signal this should be skipped
        if return_synth_boundary:
            return "", ""
        return ""

    if return_synth_boundary:
        return text, ""  # No synthetic prefix for standard format
    return text


def tokenize(
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
    Tokenizes text from an iterator of content-state pairs using a specified tokenizer.

    Parameters:
    - iterator: An iterable of (content, state) pairs where content is a dict with a 'text' or 'content' key.
    - tokenizer: Tokenizer object with an `encode` method to convert text to tokens, supporting `add_bos` and `add_eos`.
    - add_bos (bool): Flag to add a beginning-of-sequence token.
    - add_eos (bool): Flag to add an end-of-sequence token.
    - synth_context_mode: How to handle synthetic contexts (None, "prepend", "append")
    - synth_context_separator: Separator between contexts and original text
    - mask_synth_context_loss: If True, track synthetic token boundaries for loss masking
    - synth_context_prob: Probability of augmenting with synthetic context (0.0 to 1.0)
    - synth_rng_state: RNG state for reproducible synth context sampling
    - nll_field: JSONL field for precomputed NLL (e.g. "domain_expert_nll"). Falls back to
        legacy "teacher_token_logprobs" if None.
    - entropy_field: JSONL field for precomputed entropy (e.g. "domain_expert_entropy").
    - margin_field: JSONL field for precomputed margin (e.g. "domain_expert_margin").

    Yields:
    - ((tokens, teacher_logprobs, teacher_entropy, teacher_margin), state) pairs, where:
      - `tokens` is a list of tokenized text
      - `teacher_logprobs` is a list of per-token NLL values (or None if not available)
      - `teacher_entropy` is a list of per-token entropy values (or None)
      - `teacher_margin` is a list of per-token margin values (or None)
      - `state` is the TokenizerState (includes synth_token_len for loss masking).
    """
    tokenizer = build_tokenizer(name=tokenizer_type, path=tokenizer_path)

    # Setup RNG for synth context probability sampling
    synth_rng = np.random.default_rng()
    if synth_rng_state is not None:
        synth_rng.bit_generator.state = synth_rng_state

    # Track statistics for error handling
    if not hasattr(tokenize, '_error_count'):
        tokenize._error_count = 0
        tokenize._processed_count = 0

    for content, state in iterator:
        try:
            tokenize._processed_count += 1

            # Decide whether to use synthetic context for this document
            use_synth = synth_context_mode is not None and synth_rng.random() < synth_context_prob
            effective_mode = synth_context_mode if use_synth else None

            # Extract text and teacher signals (NLL, entropy, margin)
            text_result, teacher_token_ids, teacher_token_logprobs, teacher_entropy, teacher_margin = (
                extract_text_and_teacher_logprobs_from_content(
                    content,
                    synth_context_mode=effective_mode,
                    synth_context_separator=synth_context_separator,
                    return_synth_boundary=True,
                    nll_field=nll_field,
                    entropy_field=entropy_field,
                    margin_field=margin_field,
                )
            )

            # text_result is either (full_text, synth_prefix) or just full_text
            if isinstance(text_result, tuple):
                full_text, synth_prefix = text_result
            else:
                full_text = text_result
                synth_prefix = ""

            # Validate that full_text is not None or empty
            if full_text is None:
                tokenize._error_count += 1
                if tokenize._error_count <= 1 or tokenize._error_count % 10000 == 0:
                    logger.warning(f"Skipping document with None text at position {tokenize._processed_count}")
                continue

            if not isinstance(full_text, str):
                tokenize._error_count += 1
                if tokenize._error_count <= 1 or tokenize._error_count % 10000 == 0:
                    logger.warning(f"Skipping document with non-string text (type: {type(full_text)}) at position {tokenize._processed_count}")
                continue

            if len(full_text.strip()) == 0:
                tokenize._error_count += 1
                # This runs in many loader workers/ranks; keep logs very sparse.
                if tokenize._error_count <= 1 or tokenize._error_count % 10000 == 0:
                    logger.warning(f"Skipping document with empty text at position {tokenize._processed_count} (total empty: {tokenize._error_count})")
                continue

            if not full_text.endswith('\n'):
                full_text = full_text + '\n'

            # Tokenize full text with error handling
            try:
                tokens = tokenizer.encode(full_text, add_bos=add_bos, add_eos=add_eos)
            except Exception as tokenize_error:
                logger.error(
                    f"Tokenization failed at position {tokenize._processed_count}: {str(tokenize_error)}. "
                    f"Text preview (first 200 chars): {full_text[:200]!r}"
                )
                tokenize._error_count += 1
                if tokenize._error_count <= 10:
                    logger.error(f"Full error details: {tokenize_error}", exc_info=True)
                continue

        except Exception as e:
            # Catch any other unexpected errors in the data processing pipeline
            logger.error(
                f"Unexpected error processing document at position {tokenize._processed_count}: {str(e)}"
            )
            tokenize._error_count += 1
            if tokenize._error_count <= 10:
                logger.error(f"Full error details: {e}", exc_info=True)

            # Log periodic error statistics
            if tokenize._error_count % 100 == 1 or tokenize._error_count <= 10:
                error_rate = (tokenize._error_count / tokenize._processed_count) * 100
                logger.warning(
                    f"Data processing error statistics: {tokenize._error_count} errors out of "
                    f"{tokenize._processed_count} documents ({error_rate:.2f}% error rate)"
                )
            continue

        # Apply max_length truncation if specified (MUST match score_dclm.py truncation)
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]
            # Log truncation occasionally
            if hasattr(tokenize, '_truncation_count'):
                tokenize._truncation_count += 1
            else:
                tokenize._truncation_count = 1
                logger.info(f"Tokenization truncation enabled: max_length={max_length}")
            if tokenize._truncation_count <= 5 or tokenize._truncation_count % 1000 == 0:
                logger.info(f"Truncated document from {len(tokens)} to {max_length} tokens (count: {tokenize._truncation_count})")

        aligned_teacher_logprobs = None
        aligned_teacher_entropy = None
        aligned_teacher_margin = None
        expected_len = len(tokens) - 1

        # Handle precomputed NLL (new format: nll_field set, no token_ids alignment check)
        if nll_field is not None and teacher_token_logprobs is not None:
            if len(teacher_token_logprobs) == expected_len:
                aligned_teacher_logprobs = teacher_token_logprobs
            else:
                logger.error(
                    f"Precomputed NLL length mismatch: expected {expected_len}, "
                    f"got {len(teacher_token_logprobs)}. Skipping teacher signals for this doc."
                )
        elif teacher_token_ids is not None and teacher_token_logprobs is not None:
            # Legacy format: verify via token IDs
            actual_len = len(teacher_token_logprobs)
            if actual_len == expected_len:
                aligned_teacher_logprobs = teacher_token_logprobs
            else:
                logger.error(
                    f"Teacher logprobs length mismatch: expected {expected_len}, "
                    f"got {actual_len} (diff: {abs(actual_len - expected_len)}). "
                    f"This indicates preprocessing mismatch between score_dclm.py and training pipeline. "
                    f"Tokens length: {len(tokens)}, add_bos: {add_bos}. "
                    f"Fix: Ensure score_dclm.py uses same text preprocessing as training (add trailing newline)."
                )

        # Align entropy and margin only when NLL also aligned (same-length guarantee from precompute)
        if aligned_teacher_logprobs is not None:
            if teacher_entropy is not None and len(teacher_entropy) == expected_len:
                aligned_teacher_entropy = teacher_entropy
            if teacher_margin is not None and len(teacher_margin) == expected_len:
                aligned_teacher_margin = teacher_margin

        # Compute synth token length (for loss masking)
        # Only non-zero when mask_synth_context_loss is True and there's a synth prefix
        synth_token_len = 0
        if mask_synth_context_loss and synth_prefix:
            # Tokenize prefix WITHOUT BOS to get exact token count of synthetic content
            # The BOS token (if add_bos=True) will be at position 0 of the full sequence
            # and we want to mask positions [0, synth_content_len + (1 if add_bos else 0))
            synth_content_tokens = tokenizer.encode(synth_prefix, add_bos=False, add_eos=False)
            synth_token_len = len(synth_content_tokens)
            # Add 1 for BOS if it's included, since BOS is part of the synthetic prefix
            if add_bos:
                synth_token_len += 1

        yield (tokens, aligned_teacher_logprobs, aligned_teacher_entropy, aligned_teacher_margin), TokenizerState(
            it_state=state,
            add_bos=add_bos,
            add_eos=add_eos,
            name=tokenizer_type,
            path=tokenizer_path,
            synth_context_mode=synth_context_mode,
            synth_context_separator=synth_context_separator,
            synth_token_len=synth_token_len,
            synth_context_prob=synth_context_prob,
            synth_rng_state=synth_rng.bit_generator.state,
            max_length=max_length,
            nll_field=nll_field,
            entropy_field=entropy_field,
            margin_field=margin_field,
        )


def choose_source(
    source_to_iterator: Dict[str, Iterator],
    source_to_state: Dict[str, Any],
    root_dir: str,
    sources: Dict[str, float],
    rng_state: Dict[str, Any],
):
    """
    Iterates over multiple data sources, selecting sequences based on weighted random choice.

    Parameters:
    - source_to_iterator (Dict[str, Iterator]): Dict from source paths to their iterators.
    - source_to_state (Dict[str, State]): Initial state for each source, allowing state tracking.
    - root_dir str: Root dir of data sources
    - sources Dict[str, float]: Dict from subdirectory to the weight used for sampling
    - rng_state (dict): State of the random number generator for reproducibility.

    Yields:
    - Tuple of (seq, multi_choice_state) where `seq` is the next sequence from the chosen source,
    and `multi_choice_state` includes the current state of all sources and the RNG.

    This function ensures that sequences are chosen from the provided sources based on the specified weights,
    maintaining state information for each source and the RNG to allow for reproducible iteration.
    """
    # We create the rng and set its state
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state
    while True:
        # We save the rng state before sampling to be able to yield the same sequence on reload
        # Read sources/weights live so callers can update domain weights during training
        # by mutating the same `sources` dict in loader state.
        possible_sources = list(sources.keys())
        weights = np.array([float(v) for v in sources.values()], dtype=np.float64)
        n_sources = len(possible_sources)
        if n_sources == 0:
            raise RuntimeError("No data sources configured for choose_source")
        if np.any(weights < 0):
            raise RuntimeError(f"Negative source weights are not allowed: {sources}")
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            raise RuntimeError(f"Sum of source weights must be > 0, got {weight_sum}")
        norm_weights = weights / weight_sum
        source_choice = possible_sources[rng.choice(n_sources, p=norm_weights)]
        seq, state = next(source_to_iterator[source_choice])
        source_to_state = {**source_to_state, source_choice: state}
        # We update the corresponding source state
        multi_choice_state = MultiChoiceState(
            root_dir=root_dir,
            sources=sources,
            source_to_state=source_to_state,
            rng_state=rng.bit_generator.state,
        )
        yield seq, multi_choice_state


def get_empty_buffer_state(
    start_token,
    states,
):
    """
    Calculates the state to resume iteration after the buffer is cleared.

    This function determines the starting point for resuming iteration by rewinding `n_views` from the `end_token`.
    It handles cases where the rewind goes beyond the current sequence, adjusting the starting sequence and token index accordingly.
    """
    # We rewind n_views
    # This index can be negative if we go beyond the current sample
    # In that case we go back to find which sequence to start from
    # And the correct token index to start from
    seq_to_resume_from = -1
    while start_token < 0:
        seq_to_resume_from -= 1
        start_token += states[seq_to_resume_from]["seq_len"]
    resume_state = deepcopy(states[seq_to_resume_from])
    resume_state["start_token"] = start_token
    # When resuming, the iterator will then correctly fill the buffer
    del states[:seq_to_resume_from]
    if "seq_len" in resume_state:
        del resume_state["seq_len"]

    return resume_state


def pack_tokens(
    iterator: Iterator,
    empty_buffer_state: PackTokensState,
):
    """
    Iterates over tokens, packing them into chunks.

    This function aggregates tokens into a buffer and yields fixed-size chunks with dimensions `(output_seq_len, n_views)`,
    where each column represents shifted sequences of tokens. It ensures continuity in token sequences across chunks,
    preventing boundary effects and maintaining consistency regardless of `n_views`.

    When teacher logprobs are available, they are packed as an additional view (n_views + 1).

    Also tracks document boundaries (cu_seqlens) for cross-document attention masking.

    Parameters:
    - iterator: An iterator that yields pairs of ((tokens, teacher_logprobs), state), where tokens is a 1D sequence of tokens,
                teacher_logprobs is either None or a 1D sequence of per-token logprobs, and state contains all necessary
                information to resume iterator from current position.
    - empty_buffer_state: Initial PackTokensState with parameters

    Yields:
    - dict containing:
      - 'tokens': numpy.ndarray of shape `(output_seq_len, n_views)` or `(output_seq_len, n_views+1)`
      - 'cu_seqlens': list of cumulative sequence lengths marking document boundaries
    - PackTokensState: The state required to resume packing tokens from where the last returned chunk.

    The function handles the complexity of determining the correct state for resuming iteration after the buffer is cleared, ensuring seamless continuation of token sequences.
    """
    buffer = []
    teacher_logprobs_buffer = []  # Track teacher NLL for each token
    teacher_entropy_buffer = []   # Track teacher entropy for each token
    teacher_margin_buffer = []    # Track teacher margin for each token
    pos_buffer = []  # Track position within each document (for synth context masking)
    docid_buffer = []  # Track which doc each token belongs to
    doc_boundaries = []  # Track document start positions within the current buffer
    states = []
    output_seq_len = empty_buffer_state["output_seq_len"]
    n_views = empty_buffer_state["n_views"]
    start_token = empty_buffer_state["start_token"]
    previous_state = empty_buffer_state["it_state"]
    buffer_size = output_seq_len + n_views - 1

    mask_synth_context_loss = bool(empty_buffer_state.get("mask_synth_context_loss", False))
    next_doc_id = 0
    doc_synth_len = {}  # Map doc_id -> synth_token_len for that doc
    has_teacher_logprobs = None  # Will be set on first iteration
    has_teacher_entropy = False
    has_teacher_margin = False
    current_doc_start = 0  # Track where current document starts in buffer

    for i, ((tokens, teacher_logprobs, teacher_entropy, teacher_margin), state) in enumerate(iterator):
        # Check if this dataset has teacher signals (check first few documents)
        if has_teacher_logprobs is None and i < 10:
            if teacher_logprobs is not None:
                has_teacher_logprobs = True
                has_teacher_entropy = teacher_entropy is not None
                has_teacher_margin = teacher_margin is not None
                logger.info(
                    f"Teacher signals detected at doc {i}: "
                    f"nll=True entropy={has_teacher_entropy} margin={has_teacher_margin}"
                )
        elif has_teacher_logprobs is None and i >= 10:
            # After checking 10 documents, if we still haven't found teacher logprobs, assume none
            has_teacher_logprobs = False
            logger.info("No teacher logprobs detected in first 10 documents - proceeding without teacher logprobs")

        # Log statistics periodically
        if i > 0 and i % 1000 == 0 and has_teacher_logprobs:
            # Note: Accurate per-doc counting would require tracking across iterations.
            # For now, just log that teacher logprobs are being packed.
            logger.info(f"Teacher logprobs packing in progress at doc {i}")

        end_token = start_token
        sample_is_read = False

        # Defensive recovery: if a resumed state carries an out-of-range token
        # offset for the current document, skip this doc instead of crashing the
        # async loader process.
        if start_token >= len(tokens):
            logger.warning(
                f"[data] skipping doc with invalid start_token={start_token} "
                f"(len(tokens)={len(tokens)})"
            )
            start_token = 0
            previous_state = state
            continue

        # Assign doc_id and get synth_token_len from TokenizerState
        curr_doc_id = next_doc_id
        next_doc_id += 1
        synth_token_len = state.get("synth_token_len", 0) if isinstance(state, dict) else 0
        doc_synth_len[curr_doc_id] = synth_token_len
        pos_in_doc = start_token  # Position within this document

        # Track document boundary (start of new document in buffer)
        if len(buffer) > 0 or start_token == 0:
            # Only add boundary at start of a new document
            if start_token == 0:
                doc_boundaries.append(len(buffer))

        while not sample_is_read:
            assert start_token < len(
                tokens
            ), f"Start token index {start_token} bigger than sequence {len(tokens)}"
            free_space = buffer_size - len(buffer)
            seq_len = min(free_space, len(tokens) - start_token)
            end_token = start_token + seq_len
            buffer.extend(tokens[start_token:end_token])

            # Pack teacher signals if available
            # NOTE: teacher arrays have length len(tokens)-1 because there's no logprob for BOS.
            # We need to handle the offset: signals[i] predicts tokens[i+1].
            if has_teacher_logprobs:
                if teacher_logprobs is not None:
                    # Adjust indices: teacher arrays are shifted by 1 relative to tokens.
                    # For tokens[start_token:end_token], logprobs are at indices [start_token-1:end_token-1]
                    # But we need to handle start_token=0 (BOS has no logprob)
                    lp_start = max(0, start_token - 1)
                    lp_end = end_token - 1
                    if start_token == 0:
                        # First position is BOS — use dummy 0.0 for it
                        teacher_logprobs_buffer.append(0.0)
                        teacher_entropy_buffer.append(0.0)
                        teacher_margin_buffer.append(0.0)
                        if lp_end > lp_start:
                            teacher_logprobs_buffer.extend(teacher_logprobs[lp_start:lp_end])
                            if has_teacher_entropy and teacher_entropy is not None:
                                teacher_entropy_buffer.extend(teacher_entropy[lp_start:lp_end])
                            else:
                                teacher_entropy_buffer.extend([0.0] * (lp_end - lp_start))
                            if has_teacher_margin and teacher_margin is not None:
                                teacher_margin_buffer.extend(teacher_margin[lp_start:lp_end])
                            else:
                                teacher_margin_buffer.extend([0.0] * (lp_end - lp_start))
                    else:
                        teacher_logprobs_buffer.extend(teacher_logprobs[lp_start:lp_end])
                        if has_teacher_entropy and teacher_entropy is not None:
                            teacher_entropy_buffer.extend(teacher_entropy[lp_start:lp_end])
                        else:
                            teacher_entropy_buffer.extend([0.0] * (lp_end - lp_start))
                        if has_teacher_margin and teacher_margin is not None:
                            teacher_margin_buffer.extend(teacher_margin[lp_start:lp_end])
                        else:
                            teacher_margin_buffer.extend([0.0] * (lp_end - lp_start))
                else:
                    # This doc doesn't have teacher signals — use dummy 0.0 values
                    teacher_logprobs_buffer.extend([0.0] * seq_len)
                    teacher_entropy_buffer.extend([0.0] * seq_len)
                    teacher_margin_buffer.extend([0.0] * seq_len)

            # Track doc_id and position for each token
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
                assert out.ndim == 1, "Iterator should return 1D sequences"
                out = np.lib.stride_tricks.sliding_window_view(
                    out, n_views, axis=0
                )  # (output_seq_len, n_views)

                # Pack teacher signals as additional views if available.
                # View 2: NLL, View 3: entropy (if present), View 4: margin (if present).
                if has_teacher_logprobs:
                    def _extra_view(buf):
                        arr = np.array(buf, dtype=np.float32)
                        return np.lib.stride_tricks.sliding_window_view(arr, n_views, axis=0)[:, 1:2]

                    out = np.concatenate([out, _extra_view(teacher_logprobs_buffer)], axis=1)
                    if has_teacher_entropy:
                        out = np.concatenate([out, _extra_view(teacher_entropy_buffer)], axis=1)
                    if has_teacher_margin:
                        out = np.concatenate([out, _extra_view(teacher_margin_buffer)], axis=1)

                # --- Mask synthetic context labels (following custom_data.py pattern) ---
                if mask_synth_context_loss and n_views >= 2:
                    # Build position windows to check label positions
                    pos_out = np.array(pos_buffer, dtype=np.int32)
                    pos_out = np.lib.stride_tricks.sliding_window_view(pos_out, n_views, axis=0)

                    docid_out = np.array(docid_buffer, dtype=np.int32)
                    docid_out = np.lib.stride_tricks.sliding_window_view(docid_out, n_views, axis=0)

                    # Labels are at view index 1
                    label_doc_ids = docid_out[:, 1]
                    label_positions = pos_out[:, 1]

                    # Get synth_token_len for each label's doc
                    synth_len_for_label = np.array(
                        [doc_synth_len.get(int(d), 0) for d in label_doc_ids],
                        dtype=np.int32,
                    )

                    # Mask where label position < synth_token_len (i.e., in synthetic prefix)
                    synth_mask = label_positions < synth_len_for_label
                    if synth_mask.any():
                        out = out.copy()
                        out[synth_mask, 1] = -100  # Ignore index for cross-entropy

                # Build cu_seqlens from document boundaries
                # cu_seqlens marks cumulative positions: [0, doc1_end, doc1_end+doc2_end, ..., output_seq_len]
                # We need to adjust boundaries to be within [0, output_seq_len]
                cu_seqlens = [0]
                for boundary in doc_boundaries:
                    if boundary > 0 and boundary < output_seq_len:
                        cu_seqlens.append(boundary)
                cu_seqlens.append(output_seq_len)
                # Ensure cu_seqlens is sorted and unique
                cu_seqlens = sorted(set(cu_seqlens))

                # We rewind by n_views to account for the last tokens not having their targets
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

                # Adjust doc_boundaries for the remaining buffer
                # Any boundaries that were in the yielded part should be removed
                # Boundaries after output_seq_len should be shifted
                new_doc_boundaries = []
                for boundary in doc_boundaries:
                    shifted = boundary - output_seq_len
                    if shifted >= 0:
                        new_doc_boundaries.append(shifted)
                doc_boundaries = new_doc_boundaries

                yield {'tokens': out, 'cu_seqlens': cu_seqlens}, empty_buffer_state

            if start_token == len(tokens):
                start_token = 0
                sample_is_read = True
                previous_state = state


def batch_and_shuffle_prefetched_sequences(
    data_loader: Iterator,
    batch_size: int,
    prefetch_size: int,
    seq_len: int,
    n_views: int,
    state: PrefetchState,
):
    """
    Prepare batch in advance and shuffle them to reduce correlation inside batches (for ex when very long document is encountered).

    This function aggregates batches into a buffer and yields fixed-size batch size and seqlen with dimensions `(batch_size, seqlen, n_views)`,

    It uses a prefetch buffer to store batches in advance and shuffles them, the prefetch buffer is similar to `reservoir sampling`,
    but by block to preserve a smooth, easy and deterministic reloading. To ensure more uniform sequence sampling -> prefetch_size * batch_size * seq_len >> max_document_seqlength.

    Parameters:
    - iterator: An iterator that yields pairs of (sequence_dict, state), where sequence_dict contains 'tokens' and 'cu_seqlens'.
    - batch_size: The desired batch size.
    - prefetch_size: The number of batches to prefetch in advance.
    - seq_len (int): The length of the output sequences to be generated.
    - n_views (int): The number of shifted views to include in each output chunk.

    Yields:
    - dict with 'tokens' array of shape `(batch_size, seq_len, n_views)` and 'cu_seqlens' list per batch item.
    - PrefetchState: The state required to resume prefetched batch. Contains also the internal of iterator.
    """
    # NOTE: When teacher logprobs are present, pack_tokens adds an extra view (n_views+1)
    # We initialize the buffer lazily after seeing the first item to get the actual shape
    prefetch_buffer = None
    cu_seqlens_buffer = []  # Store cu_seqlens for each sequence
    actual_n_views = n_views  # Will be updated after seeing first item
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng_state"]

    # Rewind the iterator to the correct position by skipping seq_idx sequences to roll the buffer accordingly
    seq_idx = state["seq_idx"]
    assert (
        seq_idx >= 0 and seq_idx < prefetch_size
    ), "Prefetch state seq_idx should be in 0 <= seq_idx < prefetch_size."

    _rng_state = state["rng_state"]
    _it_state = state["it_state"]

    for i in range(prefetch_size * batch_size):
        item, next_it_state = next(data_loader)
        # Handle both dict format (new) and array format (old)
        if isinstance(item, dict):
            tokens = item['tokens']
            cu_seqlens = item.get('cu_seqlens', [0, seq_len])
        else:
            tokens = item
            cu_seqlens = [0, seq_len]  # Default: single document spanning full sequence

        # Lazy initialization of prefetch buffer on first item
        if prefetch_buffer is None:
            actual_n_views = tokens.shape[-1]  # Get actual number of views (may include teacher logprobs)
            prefetch_buffer = -1 * np.ones(
                (prefetch_size * batch_size, seq_len, actual_n_views), dtype=tokens.dtype
            )
        prefetch_buffer[i] = tokens
        cu_seqlens_buffer.append(cu_seqlens)

    # Shuffle both buffers together
    shuffle_indices = rng.permutation(prefetch_size * batch_size)
    prefetch_buffer = prefetch_buffer[shuffle_indices]
    cu_seqlens_buffer = [cu_seqlens_buffer[i] for i in shuffle_indices]

    for i in range(seq_idx * batch_size):
        item, _ = next(data_loader)
        if isinstance(item, dict):
            prefetch_buffer[i] = item['tokens']
            cu_seqlens_buffer[i] = item.get('cu_seqlens', [0, seq_len])
        else:
            prefetch_buffer[i] = item
            cu_seqlens_buffer[i] = [0, seq_len]

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

        # Extract batch tokens and cu_seqlens
        batch_tokens = prefetch_buffer[idx * batch_size : (idx + 1) * batch_size].copy()
        batch_cu_seqlens = cu_seqlens_buffer[idx * batch_size : (idx + 1) * batch_size]

        yield {'tokens': batch_tokens, 'cu_seqlens': batch_cu_seqlens}, state

        for i in range(batch_size):
            item, pack_state = next(data_loader)
            if isinstance(item, dict):
                prefetch_buffer[idx * batch_size + i] = item['tokens']
                cu_seqlens_buffer[idx * batch_size + i] = item.get('cu_seqlens', [0, seq_len])
            else:
                prefetch_buffer[idx * batch_size + i] = item
                cu_seqlens_buffer[idx * batch_size + i] = [0, seq_len]

        if idx == prefetch_size - 1:
            next_it_state = pack_state
            shuffle_indices = rng.permutation(prefetch_size * batch_size)
            prefetch_buffer = prefetch_buffer[shuffle_indices]
            cu_seqlens_buffer = [cu_seqlens_buffer[i] for i in shuffle_indices]

        idx = (idx + 1) % prefetch_size


def find_and_sanitize_chunks(dataset_path: str, world_size: int, file_pattern: str = TRAIN_DATA_FILE_PATTERN):
    dataset_chunks = [str(p) for p in Path(dataset_path).glob(file_pattern)]
    n_chunks = len(dataset_chunks)

    # If no chunks found with default pattern, try shard pattern as fallback
    if n_chunks == 0:
        logger.info(f"No chunks found with pattern '{file_pattern}', trying shard pattern '{TRAIN_DATA_FILE_PATTERN_SHARD}'")
        dataset_chunks = [str(p) for p in Path(dataset_path).glob(TRAIN_DATA_FILE_PATTERN_SHARD)]
        n_chunks = len(dataset_chunks)
        if n_chunks > 0:
            logger.info(f"Found {n_chunks} chunks with shard pattern")

    # Check n_chunks > 0 first before any modulo operations
    if n_chunks == 0:
        raise ValueError(f"No valid chunks found in {dataset_path} matching patterns '{file_pattern}' or '{TRAIN_DATA_FILE_PATTERN_SHARD}'. Check that data files exist.")

    if n_chunks > world_size:
        n_discard = n_chunks - world_size
        dataset_chunks = dataset_chunks[:world_size]
    else:
        assert (
            world_size % n_chunks == 0
        ), "World size should be a multiple of number of chunks"

    return dataset_chunks


def distribute_data_to_rank(
    dataset_path: str,
    rank: int,
    world_size: int,
    file_pattern: str,
    max_docs: Optional[int] = None,
    shuffle: bool = False,
    shuffle_seed: int = 0,
):
    """
    Distributes the chunk files in a dataset path to each worker.
    If world_size is smaller than the number of chunks, the extra chunks are discarded.
    Otherwise, world_size is assumed to be a multiple of number of chunks.
    In that case there are world_size//nb_chunks workers on each chunk file, reading with different offsets.

    With `max_docs`, only the first max_docs documents of the source are used: each
    chunk contributes a prefix of max_docs // n_chunks lines (the remainder goes to the
    first chunks in sorted order). The subset is the same for every run and nested
    across max_docs values; for globally shuffled chunks it is a uniform random subset,
    and for rows round-robined over chunks it is exactly the first max_docs rows.
    """
    dataset_chunks = find_and_sanitize_chunks(dataset_path, world_size, file_pattern)
    n_ranks_per_chunk = world_size // len(dataset_chunks)
    sorted_chunks = sorted(dataset_chunks)
    if max_docs is not None:
        if max_docs < len(dataset_chunks):
            raise ValueError(f"source_max_docs={max_docs} for {dataset_path} is below the number of chunks ({len(dataset_chunks)})")
        base, extra = divmod(max_docs, len(dataset_chunks))
    rank_to_jsonl_iterator_params = []
    for chunk_path in dataset_chunks:
        max_lines = None
        if max_docs is not None:
            max_lines = base + (1 if sorted_chunks.index(chunk_path) < extra else 0)
        for i in range(n_ranks_per_chunk):
            rank_to_jsonl_iterator_params.append(
                _jsonl_state(chunk_path, 0, n_ranks_per_chunk, i, 0, 0, max_lines, shuffle, shuffle_seed)
            )

    return rank_to_jsonl_iterator_params[rank]


def init_choice_state(
    root_dir: str,
    sources: Dict[str, float],
    seed: int,
    rank: int,
    world_size: int,
    file_pattern: str,
    source_max_docs: Optional[Dict[str, int]] = None,
    shuffle_each_epoch: bool = False,
):
    source_max_docs = dict(source_max_docs or {})
    unknown = set(source_max_docs) - set(sources)
    if unknown:
        raise ValueError(f"source_max_docs has sources not in data.sources: {sorted(unknown)}")
    data_path_to_jsonl_state = dict()
    for dataset_path in sources:
        jsonl_state = distribute_data_to_rank(
            os.path.join(root_dir, dataset_path), rank, world_size, file_pattern,
            max_docs=source_max_docs.get(dataset_path),
            shuffle=shuffle_each_epoch,
            shuffle_seed=seed,
        )
        data_path_to_jsonl_state[dataset_path] = jsonl_state

    multi_rng_state = np.random.default_rng(
        (seed, rank, world_size)
    ).bit_generator.state

    multi_choice_state = MultiChoiceState(
        root_dir=root_dir,
        sources=sources,
        source_to_state=data_path_to_jsonl_state,
        rng_state=multi_rng_state,
    )
    return multi_choice_state


def init_state(
    root_dir: str,
    sources: Dict[str, float],
    batch_size: int,
    prefetch_size: int,
    seq_len: int,
    n_views: int,
    seed: int,
    rank: int,
    world_size: int,
    add_bos: bool,
    add_eos: bool,
    tokenizer_name: str,
    tokenizer_path: Optional[str] = None,
    file_pattern: str = TRAIN_DATA_FILE_PATTERN,
    synth_context_mode: Optional[str] = None,
    synth_context_separator: str = "\n\n",
    mask_synth_context_loss: bool = False,
    synth_context_prob: float = 1.0,
    max_length: Optional[int] = None,
    nll_field: Optional[str] = None,
    entropy_field: Optional[str] = None,
    margin_field: Optional[str] = None,
    source_max_docs: Optional[Dict[str, int]] = None,
    shuffle_each_epoch: bool = False,
):
    multi_choice_state = init_choice_state(
        root_dir=root_dir, sources=sources, seed=seed, rank=rank, world_size=world_size, file_pattern=file_pattern,
        source_max_docs=source_max_docs, shuffle_each_epoch=shuffle_each_epoch,
    )

    # RNG for synth context probability sampling (use different seed offset)
    synth_rng_state = np.random.default_rng(
        (seed + 2, rank, world_size)
    ).bit_generator.state

    tokenizer_state = TokenizerState(
        it_state=multi_choice_state,
        add_bos=add_bos,
        add_eos=add_eos,
        name=tokenizer_name,
        path=tokenizer_path,
        synth_context_mode=synth_context_mode,
        synth_context_separator=synth_context_separator,
        synth_token_len=0,  # Will be set per-document in tokenize()
        synth_context_prob=synth_context_prob,
        synth_rng_state=synth_rng_state,
        max_length=max_length,
        nll_field=nll_field,
        entropy_field=entropy_field,
        margin_field=margin_field,
    )
    pack_state = PackTokensState(
        start_token=0,
        it_state=tokenizer_state,
        output_seq_len=seq_len,
        n_views=n_views,
        seq_len=0,
        mask_synth_context_loss=mask_synth_context_loss,
    )

    prefetch_rng_state = np.random.default_rng(
        (seed + 1, rank, world_size)
    ).bit_generator.state

    return PrefetchState(
        it_state=pack_state,
        seq_idx=0,
        rng_state=prefetch_rng_state,
        batch_size=batch_size,
        prefetch_size=prefetch_size,
    )


def setup_sources(multi_state):
    path_to_iter = dict()
    for source in multi_state["sources"]:
        jsonl_state = multi_state["source_to_state"][source]
        path_to_iter[source] = loop_on_jsonl(
            jsonl_state["file_path"],
            jsonl_state["position"],
            jsonl_state["block_size"],
            jsonl_state["offset"],
            jsonl_state["current_iter"],
            # absent in states saved before these fields existed
            jsonl_state.get("line_idx"),
            jsonl_state.get("max_lines"),
            jsonl_state.get("shuffle", False),
            jsonl_state.get("shuffle_seed", 0),
        )

    return path_to_iter


@contextlib.contextmanager
def build_dataloader(
    state: PrefetchState,
    max_large_ppl: Optional[float] = None,
    min_delta: Optional[float] = None,
):
    pack_state = state["it_state"]
    tokenizer_state = pack_state["it_state"]
    multi_state = tokenizer_state["it_state"]

    path_to_iter = setup_sources(multi_state)
    data_it = choose_source(
        source_to_iterator=path_to_iter,
        source_to_state=multi_state["source_to_state"],
        root_dir=multi_state["root_dir"],
        sources=multi_state["sources"],
        rng_state=multi_state["rng_state"],
    )

    # Apply delta filtering if configured
    if max_large_ppl is not None or min_delta is not None:
        data_it = filter_by_delta(
            data_it,
            max_large_ppl=max_large_ppl,
            min_delta=min_delta,
        )

    data_it = tokenize(
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

    data_it = pack_tokens(
        data_it,
        pack_state,
    )

    data_it = batch_and_shuffle_prefetched_sequences(
        data_loader=data_it,
        seq_len=pack_state["output_seq_len"],
        n_views=pack_state["n_views"],
        batch_size=state["batch_size"],
        prefetch_size=state["prefetch_size"],
        state=state,
    )
    yield data_it
    for it in path_to_iter.values():
        it.close()
    data_it.close()


def feed_buffer(queue: Queue, stop_event: EventClass, iterator_builder):
    """
    Producer function to fetch data from an iterable dataset and put it into a queue.
    Incorporates timeout management to avoid hanging on queue.put() when the queue is full.
    """
    with iterator_builder() as iterator:
        for item in iterator:
            while not stop_event.is_set():
                try:
                    queue.put(
                        item, timeout=0.1
                    )  # Attempts to put item into the queue with a timeout
                    break  # On successful put, breaks out of the while loop
                except Full:
                    pass
            if stop_event.is_set():
                break


def consume_buffer(producer: Process, queue: Queue):
    """
    Consumer function to process items from the queue.
    Handles cases where the queue might be empty by implementing timeouts on queue.get().
    """
    while producer.exitcode is None:
        try:
            item = queue.get(
                timeout=0.1
            )  # Tries to get an item from the queue with a timeout
            yield item
        except Empty:
            pass

    raise RuntimeError(
        "Data loader quit unexpectedly, real error has been raised previously"
    )


@contextlib.contextmanager
def async_iterator(buffer_size: int, iterator_builder):
    """
    Context manager to setup and manage asynchronous iteration with producer-consumer model.
    """
    queue = Queue(maxsize=buffer_size)
    stop_event = Event()
    producer = Process(target=feed_buffer, args=(queue, stop_event, iterator_builder))
    logger.info("Async dataloader started")
    producer.start()

    consumer = consume_buffer(producer, queue)
    try:
        yield consumer
    finally:
        stop_event.set()  # Ensures the stop event is signaled
        consumer.close()
        producer.join(timeout=0.2)  # Waits for the producer to finish
        if producer.exitcode is None:
            logger.info(f"Killing async data process {producer.pid} ...")
            producer.kill()
        else:
            logger.info(
                f"Async data process {producer.pid} exited with code {producer.exitcode}"
            )
        logger.info("Async dataloader cleaned up")


@dataclass
class DataArgs:
    """Minimal data + KD configuration for the lingua-clean recipes.

    Only the fields needed by the 13 supported losses are declared here.
    All other ``use_*`` knobs from upstream lingua/ are intentionally
    omitted; train.py reads optional flags via ``getattr(args.data, X,
    default)`` so unset families simply fall through to their defaults.
    """

    # ---------------- Data loading ----------------
    root_dir: Optional[str] = None
    sources: Dict[str, float] = field(default_factory=dict)
    batch_size: int = 2
    seq_len: int = 2048
    n_views: int = 2
    seed: int = 42
    add_bos: bool = True
    add_eos: bool = True
    load_async: bool = True
    prefetch_size: int = 64
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)
    max_length: Optional[int] = None  # max sequence length for tokenization
    # Use only the first N documents of a source (uniform random subset of a prepared,
    # shuffled source; nested across N). Sources past their N docs repeat (epochs).
    source_max_docs: Dict[str, int] = field(default_factory=dict)
    # Read every epoch after the first in a fresh seeded permutation instead of file order.
    shuffle_each_epoch: bool = False

    # Synthetic context (kept for orig_data compatibility)
    synth_context_mode: Optional[str] = None
    synth_context_separator: str = "\n\n"
    mask_synth_context_loss: bool = False
    synth_context_prob: float = 1.0

    # Delta filtering (kept for dataloader compatibility; defaults disable)
    max_large_ppl: Optional[float] = None
    min_delta: Optional[float] = None
    delta_beta: float = 1.0
    delta_max: float = 10.0

    # Misc. scaffolding referenced directly by train.py (defaults are no-ops)
    add_special_tokens: bool = False
    mask_cross_doc_loss: bool = False
    disable_cross_doc_attn: bool = False
    rho1_stage_boundaries: List[int] = field(default_factory=list)
    reverse_kl_stage_switch_frac: float = 0.8
    source_weight_schedule: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Precomputed teacher signal field names (consumed by init_state; safe defaults)
    teacher_logprob_field: Optional[str] = None
    teacher_entropy_field: Optional[str] = None
    teacher_margin_field: Optional[str] = None

    # Read by train.py but not used by our 13 losses (default = disabled paths)
    use_teacher_logprobs: bool = False
    use_entropy_delta: bool = False

    # ============================================================
    # Distillation flags + hyperparameters for the 13 supported losses
    # ============================================================

    # ----- Forward KL (FKD baseline: kd_rl7b, kd_rl_1binstruct) -----
    use_kl_distillation: bool = False
    kl_temperature: float = 2.0
    kl_alpha: float = 0.5
    kl_chunk_size: int = 128

    # ----- Reverse KL pure / mode-seeking (MiniLLM-style) -----
    use_reverse_kl_distillation: bool = False
    reverse_kl_temperature: float = 2.0
    reverse_kl_alpha: float = 0.5
    reverse_kl_chunk_size: int = 128

    # ----- RKL + uniform CE (the lam_CE sweep family) -----
    use_rkl_with_uniform_ce_distillation: bool = False
    rkl_uniform_temperature: float = 2.0
    rkl_uniform_lambda_ce: float = 0.3
    rkl_uniform_chunk_size: int = 128

    # ----- RKL + entropy-gated CE (idx 139 == ours / entgate) -----
    use_rkl_entropy_gated_distillation: bool = False
    rkl_entgate_temperature: float = 2.0
    rkl_entgate_lambda_ce: float = 1.0
    rkl_entgate_quantile: float = 0.30
    rkl_entgate_chunk_size: int = 128

    # ----- FKL on the same low-entropy gate (idx 140 ablation: fkl_entgate) -----
    use_fkl_entropy_gated_distillation: bool = False
    fkl_entgate_temperature: float = 2.0
    fkl_entgate_lambda_ce: float = 1.0
    fkl_entgate_quantile: float = 0.30
    fkl_entgate_chunk_size: int = 128

    # ----- Entropy-band: RKL low / FKL high / CE middle (idx 151, idx 152) -----
    use_entropy_band_kl_distillation: bool = False
    entband_temperature: float = 2.0
    entband_lambda_ce: float = 1.0
    entband_low_quantile: float = 0.30   # set 0.0 to disable the RKL low-band (idx 152)
    entband_high_quantile: float = 0.30
    entband_chunk_size: int = 128

    # ----- Random-mask control matched to entgate fire rate -----
    use_rkl_random_gated_distillation: bool = False
    rkl_randmask_temperature: float = 2.0
    rkl_randmask_lambda_ce: float = 1.0
    rkl_randmask_quantile: float = 0.30
    rkl_randmask_chunk_size: int = 128
    rkl_randmask_base_seed: int = 0

    # ----- Student-entropy gated (negative-result ablation) -----
    use_rkl_student_entropy_gated_distillation: bool = False
    rkl_stentgate_temperature: float = 2.0
    rkl_stentgate_lambda_ce: float = 1.0
    rkl_stentgate_quantile: float = 0.30
    rkl_stentgate_chunk_size: int = 128

    # ----- RKL/CE entropy-switched partition (idx 148, the headline winner) -----
    use_rkl_entropy_switched_distillation: bool = False
    rkl_entswitch_temperature: float = 2.0
    rkl_entswitch_lambda_ce: float = 1.0
    rkl_entswitch_quantile: float = 0.30
    rkl_entswitch_chunk_size: int = 128

    # ----- RKL + FKL convex blend (kept for the alpha=0.5/0.8 sweep) -----
    use_rkl_fkl_mix_distillation: bool = False
    rkl_fkl_mix_temperature: float = 2.0
    rkl_fkl_mix_alpha: float = 0.5         # alpha=1.0 ⇒ pure revKL
    rkl_fkl_mix_chunk_size: int = 128

    # ----- KD diagnostics (per-domain JS/KL/H concentration; default off) -----
    kd_diagnostics_enabled: bool = False
    kd_diagnostics_chunk_size: int = 128

def init_dataloader_state_from_args(
    args: DataArgs,
    rank: int,
    world_size: int,
):
    return init_state(
        root_dir=args.root_dir,
        sources=args.sources,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        prefetch_size=args.prefetch_size,
        n_views=args.n_views,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
        tokenizer_name=args.tokenizer.name,
        tokenizer_path=args.tokenizer.path,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
        synth_context_mode=args.synth_context_mode,
        synth_context_separator=args.synth_context_separator,
        mask_synth_context_loss=args.mask_synth_context_loss,
        synth_context_prob=args.synth_context_prob,
        max_length=args.max_length,
        nll_field=args.teacher_logprob_field,
        entropy_field=args.teacher_entropy_field,
        margin_field=args.teacher_margin_field,
        source_max_docs=args.source_max_docs,
        shuffle_each_epoch=args.shuffle_each_epoch,
    )


def build_dataloader_from_args(
    args: DataArgs,
    state: Optional[PrefetchState] = None,
):
    data_builder = partial(
        build_dataloader,
        state,
        max_large_ppl=args.max_large_ppl,
        min_delta=args.min_delta,
    )
    if args.load_async:
        return async_iterator(args.prefetch_size, data_builder)
    else:
        return data_builder()
