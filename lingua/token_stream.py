"""Deterministic packed-token loader over a pre-tokenized stream (setup/pretokenize_dclm_qwen3.py).

The batch at optimizer step s is a pure function of s, so a run resumed from any checkpoint, or a decay branch started
from another run's checkpoint, reads exactly the sequences the original run would have read, independent of the
number of GPUs. The loader state saved in checkpoints is just the next global sequence index.

Two modes over a prefix of the global document order:
  unique  Documents 0, 1, ... back to back, read once. Asking for a sequence past `n_tokens` is an error.
  repeat  A pool made of exactly `n_tokens` tokens: the first documents in order, the last one cut short so the pool
          is a whole number of sequences. Pass p (0-based) lays the pool's documents out in the order
          permutation(seed, p), so every pass gets new document neighbours and sequence boundaries. Passes are back
          to back, so pass p ends exactly at global sequence (p + 1) * n_tokens / seq_len.

Sequence i is stream[i * L : i * L + L + 1] (inputs = first L tokens, labels = last L), the same one-token overlap as
lingua's pack_tokens, with cu_seqlens = [0, <document starts inside the input window>, L].
"""

import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger()


class TokenStream:
    def __init__(self, path: str, mode: str, n_tokens: int, seq_len: int, seed: int = 0):
        assert mode in ("unique", "repeat"), mode
        self.path, self.mode, self.L, self.seed = path, mode, seq_len, seed
        self.tokens = np.memmap(os.path.join(path, "tokens.bin"), dtype=np.uint32, mode="r")
        offsets = np.load(os.path.join(path, "doc_offsets.npy"), mmap_mode="r")
        if n_tokens + 1 > len(self.tokens):
            raise ValueError(f"{path} has {len(self.tokens):,} tokens; {n_tokens + 1:,} requested")
        self.n_tokens = n_tokens
        # Documents needed to cover n_tokens; the last one is cut at n_tokens in repeat mode.
        n_docs = int(np.searchsorted(offsets, n_tokens, side="left"))
        self.starts = np.asarray(offsets[:n_docs], dtype=np.int64)
        self.lengths = np.diff(np.append(self.starts, n_tokens))
        if mode == "repeat":
            if n_tokens % seq_len:
                raise ValueError(f"repeat pool of {n_tokens:,} tokens is not a whole number of {seq_len}-token sequences")
            self.seqs_per_pass = n_tokens // seq_len
        else:
            self.seqs_per_pass = None
        logger.info(f"TokenStream {mode}: {path}, {n_tokens:,} tokens, {len(self.starts):,} documents"
                    + (f", {self.seqs_per_pass:,} sequences per pass, seed {seed}" if mode == "repeat" else ""))

    @lru_cache(maxsize=4)
    def _pass_layout(self, p: int) -> Tuple[np.ndarray, np.ndarray]:
        """(document order, start of each document within the pass) for pass p."""
        order = np.random.default_rng([self.seed, p]).permutation(len(self.starts))
        pos = np.concatenate([[0], np.cumsum(self.lengths[order])[:-1]])
        if p < 3 or p in (7, 15):
            logger.info(f"TokenStream pass {p + 1}: first documents {order[:5].tolist()}")
        return order, pos

    def _read(self, a: int, b: int) -> Tuple[np.ndarray, List[int]]:
        """Tokens [a, b) of the stream and the document starts in [a, b) (relative to a)."""
        if self.mode == "unique":
            if b > self.n_tokens + 1:
                raise IndexError(f"unique stream exhausted: tokens up to {b:,} requested, {self.n_tokens:,} allowed")
            i = int(np.searchsorted(self.starts, a, side="right"))
            j = int(np.searchsorted(self.starts, b, side="left"))
            return np.asarray(self.tokens[a:b]), (self.starts[i:j] - a).tolist()
        out, doc_starts, pos = [], [], a
        while pos < b:
            p, off = divmod(pos, self.n_tokens)
            order, pstart = self._pass_layout(p)
            k = int(np.searchsorted(pstart, off, side="right")) - 1
            while pos < b and k < len(order):
                d = order[k]
                in_doc = off - pstart[k]
                if in_doc == 0:
                    doc_starts.append(pos - a)
                take = min(self.lengths[d] - in_doc, b - pos)
                s = self.starts[d] + in_doc
                out.append(np.asarray(self.tokens[s:s + take]))
                pos += take
                off += take
                k += 1
        return np.concatenate(out), doc_starts

    def sequence(self, i: int) -> Tuple[np.ndarray, List[int]]:
        """(tokens [L, 2] = inputs/labels, cu_seqlens) for global sequence i."""
        toks, doc_starts = self._read(i * self.L, i * self.L + self.L + 1)
        views = np.stack([toks[:-1], toks[1:]], axis=1).astype(np.int64)
        # Python ints: numpy scalars here would reach flash-attn's max_seqlen as traced tensors under torch.compile.
        cu = [0] + [int(s) for s in doc_starts if 0 < s < self.L] + [self.L]
        return views, cu


def token_stream_batches(stream: TokenStream, batch_size: int, rank: int, world_size: int,
                         state: Dict[str, Any]) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Yields ({'tokens': [b, L, 2], 'cu_seqlens': [...]}, state) for this rank; global batch = batch_size * world_size.

    state["next_seq"] is the first global sequence of the next global batch; it advances by the global batch."""
    global_bs = batch_size * world_size
    nxt = int(state["next_seq"])
    while True:
        first = nxt + rank * batch_size
        seqs = [stream.sequence(first + k) for k in range(batch_size)]
        nxt += global_bs
        yield ({"tokens": np.stack([s[0] for s in seqs]), "cu_seqlens": [s[1] for s in seqs]},
               {"token_stream": True, "next_seq": nxt})


def init_token_stream_state() -> Dict[str, Any]:
    return {"token_stream": True, "next_seq": 0}
