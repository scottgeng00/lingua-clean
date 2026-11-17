"""FineWeb-Edu classifier score (HuggingFaceFW/fineweb-edu-classifier, regression 0-5, first 512 BERT tokens) for the
documents of one shard of a pre-tokenized stream (same doc ranges as embed_tokens.py). Texts are decoded from the
Qwen3 token ids (trailing <|endoftext|> dropped). Output: OUT/shard_XXXX.edu.npz with doc, edu (float32).

    .venv-kd/bin/python scripts/embed/quality_edu.py --stream DIR --out OUT --shard 3 --num-shards 40 [--docs N]
"""
import argparse
import os
import time

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

QWEN_TOK = "/home/sgeng/scottg/pretrain/assets/models/Qwen3-0.6B-Base-tokenizer/tokenizer.json"
EDU = "/home/sgeng/scottg/pretrain/assets/models/fineweb-edu-classifier"
EOS = 151643


def doc_range(offsets, shard, num_shards, n_docs):
    n = n_docs or len(offsets) - 1
    return shard * n // num_shards, (shard + 1) * n // num_shards


def decode(qtok, tokens, offsets, docs):
    ids = []
    for d in docs:
        a, b = int(offsets[d]), int(offsets[d + 1])
        if b > a and tokens[b - 1] == EOS:
            b -= 1
        ids.append(tokens[a:b].tolist())
    return qtok.decode_batch(ids, skip_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--docs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    out = f"{a.out}/shard_{a.shard:04d}.edu.npz"
    if os.path.exists(out):
        return print(f"{out} exists; skipping")
    os.makedirs(a.out, exist_ok=True)
    tokens = np.memmap(f"{a.stream}/tokens.bin", dtype=np.uint32, mode="r")
    offsets = np.load(f"{a.stream}/doc_offsets.npy", mmap_mode="r")
    d0, d1 = doc_range(offsets, a.shard, a.num_shards, a.docs)
    qtok = Tokenizer.from_file(QWEN_TOK)
    btok = AutoTokenizer.from_pretrained(EDU)
    model = AutoModelForSequenceClassification.from_pretrained(EDU, torch_dtype=torch.bfloat16).cuda().eval()
    docs = np.arange(d0, d1)
    scores = np.zeros(len(docs), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(docs), a.batch):
            texts = decode(qtok, tokens, offsets, docs[i:i + a.batch])
            enc = btok(texts, truncation=True, max_length=512, padding="longest", return_tensors="pt").to("cuda")
            scores[i:i + len(texts)] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
            if (i // a.batch) % 500 == 0:
                print(f"  {i + len(texts):,}/{len(docs):,} docs, {(i + len(texts)) / (time.time() - t0):,.0f} docs/s",
                      flush=True)
    np.savez(out + ".tmp.npz", doc=docs, edu=scores)
    os.replace(out + ".tmp.npz", out)
    print(f"done: {len(docs):,} docs in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
