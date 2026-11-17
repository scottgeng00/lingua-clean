"""DCLM quality score (mlfoundations/fasttext-oh-eli5: P(__label__hq), text joined into one line as in DCLM) for the
documents of one shard of a pre-tokenized stream (same doc ranges as embed_tokens.py). CPU only; runs in the
fastText venv (/checkpoint/mnm/sgeng/pretrain/venvs/quality: fasttext, numpy<2, tokenizers). Output:
OUT/shard_XXXX.dclm.npz with doc, dclm (float32).

    venvs/quality/bin/python scripts/embed/quality_dclm.py --stream DIR --out OUT --shard 3 --num-shards 40 [--docs N]
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

QWEN_TOK = "/home/sgeng/scottg/pretrain/assets/models/Qwen3-0.6B-Base-tokenizer/tokenizer.json"
FT = "/home/sgeng/scottg/pretrain/assets/models/fasttext-oh-eli5/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
EOS = 151643
_state = {}


def _init(stream):
    import fasttext
    from tokenizers import Tokenizer
    _state["ft"] = fasttext.load_model(FT)
    _state["tok"] = Tokenizer.from_file(QWEN_TOK)
    _state["tokens"] = np.memmap(f"{stream}/tokens.bin", dtype=np.uint32, mode="r")
    _state["offsets"] = np.load(f"{stream}/doc_offsets.npy", mmap_mode="r")


def _score(docs):
    tokens, offsets = _state["tokens"], _state["offsets"]
    ids = []
    for d in docs:
        a, b = int(offsets[d]), int(offsets[d + 1])
        if b > a and tokens[b - 1] == EOS:
            b -= 1
        ids.append(tokens[a:b].tolist())
    texts = [" ".join(t.strip().splitlines()) for t in _state["tok"].decode_batch(ids, skip_special_tokens=False)]
    labels, probs = _state["ft"].predict(texts, k=2)
    return np.array([p[list(l).index("__label__hq")] for l, p in zip(labels, probs)], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--docs", type=int, default=None)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    out = f"{a.out}/shard_{a.shard:04d}.dclm.npz"
    if os.path.exists(out):
        return print(f"{out} exists; skipping")
    os.makedirs(a.out, exist_ok=True)
    offsets = np.load(f"{a.stream}/doc_offsets.npy", mmap_mode="r")
    n = a.docs or len(offsets) - 1
    docs = np.arange(a.shard * n // a.num_shards, (a.shard + 1) * n // a.num_shards)
    t0 = time.time()
    with Pool(a.procs, initializer=_init, initargs=(a.stream,)) as pool:
        parts = pool.map(_score, np.array_split(docs, max(1, len(docs) // 2000)), chunksize=1)
    np.savez(out + ".tmp.npz", doc=docs, dclm=np.concatenate(parts))
    os.replace(out + ".tmp.npz", out)
    print(f"done: {len(docs):,} docs in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
