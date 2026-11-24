"""Turn a finished multi-sample simplescale run (simplescale.harnesses.multisample) into a Qwen3 token stream for
lingua/token_stream.py, shuffled at the sample level.

Keeps every sample that finished (finish_reason "stop") with a non-empty answer. The samples of the last
--heldout-sources sources form a validation stream (<out>/<name>_heldout, with doc_bytes for bits per byte). All
other samples are tokenized (no special tokens, one <|endoftext|> after each sample, as in
setup/pretokenize_dclm_qwen3.py) and ordered by a block shuffle (--seed): sources in random order, then the samples
shuffled within consecutive blocks of --block samples. Any prefix of the stream (a run reads only what its horizon
needs) is therefore all samples of a random subset of sources, and the samples of one source are spread over a block
(~100k samples, ~90M tokens) instead of packed into the same sequences.

Output (<out>/):
    tokens.bin       uint32, samples back to back in shuffled order
    doc_offsets.npy  int64 [n + 1]
    doc_source.npz   per stream document: subset (index into meta["subsets"]), source_row (row in that DCLM subset),
                     sample (index among the source's samples), kind (index into meta["kinds"])
    meta.json        counts, run, seed, tokenizer
    <name>_heldout/  tokens.bin, doc_offsets.npy, doc_bytes.npy, doc_source.npz, meta.json (dataset order)

Data-constrained runs (--pool-mult): only the samples of the documents in a pool of setup/build_pool_stream.py's
stream (its first pools_w1152[mult]["docs"] documents, the same filtered sources in the same order), no held-out
set, and --order sample-major: every source's samples in a random order, round k = the k-th sample of every source,
rounds back to back and each shuffled, so any prefix is about the same number of views of every pool document.

    .venv-kd/bin/python setup/export_multisample.py --run .../simplescale/runs/f200w-qwen35-9b-mix \
        --out .../assets/data/ms_mix_pool025_qwen3_tokens --pool-mult 0.25 --order sample-major
    .venv-kd/bin/python setup/export_multisample.py --run .../simplescale/runs/f200w-qwen35-9b-mix \
        --out /home/sgeng/scottg/pretrain/assets/data/ms_mix_qwen3_tokens
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import orjson

TOKENIZER = "/home/sgeng/scottg/pretrain/assets/models/Qwen3-0.6B-Base-tokenizer"
EOS = 151643
_tok = None
STRIP_THINK = False  # --strip-think (set before the worker pool forks)


def _tokenizer():
    global _tok
    if _tok is None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(TOKENIZER)
    return _tok


def read_shard(uri, run_idx=0):
    """[(source key, sample index, kind, text)] for the usable samples of one shard, plus dropped-sample counts.
    Sample indices of the run_idx-th --run are offset by 100 * run_idx (a top-up run adds views of the same sources)."""
    out, dropped = [], {"length": 0, "empty": 0}
    if STRIP_THINK:
        dropped["think_stripped"] = 0
    for line in open(uri, "rb"):
        r = orjson.loads(line)["result"]
        src = r["source"]
        key = (src["source_subset"], int(src["source_row"]))
        for i, s in enumerate(r["samples"]):
            text = s["text"]
            if STRIP_THINK and s["finish_reason"] == "stop" and "</think>" in text:
                # Thinking-off samples that still reason and close with </think> (Qwen3.5-2B, mostly the math
                # prompt): keep only the answer after the last </think>, as the harness does with thinking on.
                text = text.rsplit("</think>", 1)[1].strip()
                dropped["think_stripped"] += 1
            if s["finish_reason"] != "stop":
                dropped["length"] += 1
            elif not text:
                dropped["empty"] += 1
            else:
                out.append((key, i + 100 * run_idx, s["kind"], text))
    return out, dropped


_SKIP, _KEEP = set(), None  # held-out sources / pool sources (None: all), set per worker by _init


def _init(skip, keep):
    global _SKIP, _KEEP
    _SKIP, _KEEP = skip, keep


def tokenize(task):
    """Tokenize one shard's kept samples -> tmp file of tokens (EOS-terminated) and per-sample lengths. A shard whose
    tmp files already exist (an interrupted export) is not tokenized again."""
    k, uri, tmp, run_idx = task
    samples, dropped = read_shard(uri, run_idx)
    samples = [s for s in samples if s[0] not in _SKIP and (_KEEP is None or s[0] in _KEEP)]
    meta = [(s[0][0], s[0][1], s[1], s[2]) for s in samples]
    if os.path.exists(f"{tmp}/{k}.len.npy") and len(np.load(f"{tmp}/{k}.len.npy")) == len(samples):
        return k, meta, dropped
    ids = _tokenizer()([s[3] for s in samples], add_special_tokens=False)["input_ids"] if samples else []
    arrs = [np.asarray(x + [EOS], dtype=np.uint32) for x in ids]
    np.concatenate(arrs or [np.zeros(0, np.uint32)]).tofile(f"{tmp}/{k}.bin")
    lens = np.asarray([len(a) for a in arrs], dtype=np.int64)
    np.save(f"{tmp}/{k}.len.npy", lens)
    return k, meta, dropped


def block_shuffle(keys, block, seed):
    """Stream order for samples with source keys `keys`: sources in random order (samples of a source together),
    then samples shuffled within consecutive blocks of `block`. Returns perm with perm[stream index] = sample index."""
    rng = np.random.default_rng(seed)
    ids = {}
    src = np.asarray([ids.setdefault(k, len(ids)) for k in keys], dtype=np.int64)
    rank = rng.permutation(len(ids))  # rank[source id] = position of the source
    perm = np.argsort(rank[src], kind="stable")
    for lo in range(0, len(perm), block):
        perm[lo:lo + block] = rng.permutation(perm[lo:lo + block])
    return perm


def sample_major(keys, seed):
    """Stream order for samples with source keys `keys`: each source's samples in a random order, round k = the k-th
    sample of every source, rounds back to back and each shuffled. Returns perm with perm[stream index] = index."""
    rng = np.random.default_rng(seed)
    ids = {}
    src = np.asarray([ids.setdefault(k, len(ids)) for k in keys], dtype=np.int64)
    by_src = np.lexsort((rng.random(len(src)), src))  # samples grouped by source, random order within a source
    starts = np.searchsorted(src[by_src], src[by_src], side="left")
    rnd = np.empty(len(src), dtype=np.int64)
    rnd[by_src] = np.arange(len(src)) - starts  # rank of each sample within its source = its round
    return np.lexsort((rng.random(len(src)), rnd))


def write_heldout(samples, out, subsets, kinds):
    os.makedirs(out, exist_ok=True)
    tok = _tokenizer()
    ids = tok([s[3] for s in samples], add_special_tokens=False)["input_ids"]
    arrs = [np.asarray(x + [EOS], dtype=np.uint32) for x in ids]
    np.concatenate(arrs).tofile(f"{out}/tokens.bin")
    np.save(f"{out}/doc_offsets.npy", np.concatenate([[0], np.cumsum([len(a) for a in arrs])]).astype(np.int64))
    n_bytes = np.asarray([len(s[3].encode("utf-8")) for s in samples], dtype=np.int64)
    np.save(f"{out}/doc_bytes.npy", n_bytes)
    np.savez(f"{out}/doc_source.npz", subset=np.asarray([subsets.index(s[0][0]) for s in samples], np.int8),
             source_row=np.asarray([s[0][1] for s in samples], np.int64),
             sample=np.asarray([s[1] for s in samples], np.int16),
             kind=np.asarray([kinds.index(s[2]) for s in samples], np.int8))
    meta = {"docs": len(samples), "tokens": int(sum(len(a) for a in arrs)), "bytes": int(n_bytes.sum()),
            "sources": len({s[0] for s in samples}), "tokenizer": TOKENIZER, "eos_id": EOS, "subsets": subsets,
            "kinds": kinds}
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True,
                    help="simplescale run dir(s) (results/dataset.json must exist); extra runs add views of the same sources")
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout-sources", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block", type=int, default=100_000, help="samples per shuffle block")
    ap.add_argument("--order", choices=["block", "sample-major"], default="block")
    ap.add_argument("--pool-mult", type=float, help="keep only the documents of this pool (data-constrained runs)")
    ap.add_argument("--pool-stream", default="/home/sgeng/scottg/pretrain/assets/data/dclm_f200w_qwen3_tokens")
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--strip-think", action="store_true",
                    help="cut finished samples containing </think> to the text after the last </think> (default: keep as is)")
    a = ap.parse_args()
    global STRIP_THINK
    STRIP_THINK = a.strip_think
    dss = [json.load(open(f"{run}/results/dataset.json")) for run in a.run]
    shards = [{**s, "run_idx": j, "chunk_size": d["chunk_size"]}
              for j, d in enumerate(dss) for s in sorted(d["shards"], key=lambda s: s["chunk"])]
    assert len(dss) == 1 or a.heldout_sources == 0 or a.pool_mult is not None, "held-out sets need a single --run"
    keep, pool_meta = None, None
    if a.pool_mult is not None:
        pm = json.load(open(f"{a.pool_stream}/meta.json"))
        pool_meta = {"stream": a.pool_stream, "mult": a.pool_mult, **pm["pools_w1152"][str(a.pool_mult)]}
        src = np.load(f"{a.pool_stream}/doc_source.npz")
        n = pool_meta["docs"]
        keep = {(pm["subsets"][i], int(r)) for i, r in zip(src["subset"][:n].tolist(), src["source_row"][:n].tolist())}
        # Manifests share the pool stream's order, so only the first ceil(n / chunk_size) chunks hold pool sources.
        shards = [s for s in shards if s["chunk"] * s["chunk_size"] < n]
        a.heldout_sources = 0
    name = os.path.basename(os.path.normpath(a.out)).replace("_qwen3_tokens", "")
    os.makedirs(a.out, exist_ok=True)
    tmp = f"{a.out}/tmp"
    os.makedirs(tmp, exist_ok=True)

    # Held-out sources: the last --heldout-sources sources of the run (chunks are in manifest order).
    held, held_keys = [], set()
    for s in reversed(shards if a.heldout_sources else []):
        samples, _ = read_shard(s["uri"])
        keys = list(dict.fromkeys(x[0] for x in samples))
        for key in reversed(keys):
            if len(held_keys) < a.heldout_sources:
                held_keys.add(key)
        held = [x for x in samples if x[0] in held_keys] + held
        if len(held_keys) >= a.heldout_sources:
            break

    with Pool(a.workers, initializer=_init, initargs=(held_keys, keep)) as pool:
        results = sorted(pool.imap_unordered(tokenize, [(k, s["uri"], tmp, s["run_idx"]) for k, s in enumerate(shards)]))
    dropped = {"length": 0, "empty": 0, **({"think_stripped": 0} if STRIP_THINK else {})}
    for _, _, d in results:
        for x in d:
            dropped[x] += d[x]
    lens = [np.load(f"{tmp}/{k}.len.npy") for k, _, _ in results]
    meta_rows = [m for _, ms, _ in results for m in ms]
    all_lens = np.concatenate(lens)
    subsets = sorted({m[0] for m in meta_rows})
    kinds = sorted({m[3] for m in meta_rows})

    # Stream position of every sample (perm[stream index] = sample index), then its token offset.
    keys = [(r[0], r[1]) for r in meta_rows]
    perm = block_shuffle(keys, a.block, a.seed) if a.order == "block" else sample_major(keys, a.seed)
    offsets = np.concatenate([[0], np.cumsum(all_lens[perm])]).astype(np.int64)
    # Gather in memory and write once, sequentially: scattering samples into a memmap on NFS costs a page
    # read-modify-write per sample and stalls for hours.
    src_starts = np.concatenate([[0], np.cumsum(all_lens)]).astype(np.int64)
    src = np.empty(int(src_starts[-1]), dtype=np.uint32)
    first = np.concatenate([[0], np.cumsum([len(x) for x in lens])])
    for k, _, _ in results:
        src[src_starts[first[k]]:src_starts[first[k + 1]]] = np.fromfile(f"{tmp}/{k}.bin", dtype=np.uint32)
    out = np.empty_like(src)
    for d, i in enumerate(perm.tolist()):
        out[offsets[d]:offsets[d + 1]] = src[src_starts[i]:src_starts[i + 1]]
    del src
    out.tofile(f"{a.out}/tokens.bin")
    del out
    np.save(f"{a.out}/doc_offsets.npy", offsets)
    m = np.asarray([(subsets.index(r[0]), r[1], r[2], kinds.index(r[3])) for r in meta_rows], dtype=np.int64)[perm]
    np.savez(f"{a.out}/doc_source.npz", subset=m[:, 0].astype(np.int8), source_row=m[:, 1],
             sample=m[:, 2].astype(np.int16), kind=m[:, 3].astype(np.int8))
    heldout = write_heldout(held, f"{a.out}/{name}_heldout", subsets, kinds) if held else None
    meta = {"docs": len(all_lens), "tokens": int(offsets[-1]), "sources": len({(r[0], r[1]) for r in meta_rows}),
            "dropped_samples": dropped, "pool": pool_meta,
            "heldout": heldout and {"sources": heldout["sources"], "docs": heldout["docs"], "tokens": heldout["tokens"]},
            "kinds": kinds, "subsets": subsets, "kind_docs": {k: int((m[:, 3] == i).sum()) for i, k in enumerate(kinds)},
            "order": (f"block shuffle (seed {a.seed}, {a.block} samples per block): sources in random order, samples "
                      f"shuffled within blocks") if a.order == "block" else
                     f"sample-major (seed {a.seed}): round k = k-th sample (random order) of every source, rounds shuffled",
            "run": a.run if len(a.run) > 1 else a.run[0],
            "manifest": [d["manifest"] for d in dss] if len(dss) > 1 else dss[0]["manifest"], "tokenizer": TOKENIZER, "eos_id": EOS, "dtype": "uint32",
            "script": "setup/export_multisample.py"}
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
    for f in os.listdir(tmp):
        os.remove(f"{tmp}/{f}")
    os.rmdir(tmp)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
