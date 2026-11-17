"""WebOrganizer topic + format classification (TopicClassifier-NoURL, FormatClassifier-NoURL; 24 classes each) of the
documents of one shard of a pre-tokenized stream (same doc ranges as scripts/embed/embed_tokens.py). Texts are decoded
from the Qwen3 token ids (trailing <|endoftext|> dropped), tokenized once with the shared gte tokenizer (first
--max-len tokens), sorted by length and batched by padded-token budget; both models see the same batches.
Output OUT/shard_XXXX.weborg.npz: doc, topic_probs / format_probs (float16 [n, 24]), n_tokens.

    .venv-kd/bin/python scripts/weborg/classify.py --stream DIR --out OUT --shard 3 --num-shards 40 [--docs N]
"""
import argparse
import os
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from scripts.weborg.load import load

M = "/home/sgeng/scottg/pretrain/assets/models"
QWEN_TOK = f"{M}/Qwen3-0.6B-Base-tokenizer/tokenizer.json"
EOS = 151643


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--docs", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch-tokens", type=int, default=131072)
    a = ap.parse_args()
    out = f"{a.out}/shard_{a.shard:04d}.weborg.npz"
    if os.path.exists(out):
        return print(f"{out} exists; skipping")
    os.makedirs(a.out, exist_ok=True)
    tokens = np.memmap(f"{a.stream}/tokens.bin", dtype=np.uint32, mode="r")
    offsets = np.load(f"{a.stream}/doc_offsets.npy", mmap_mode="r")
    n = a.docs or len(offsets) - 1
    docs = np.arange(a.shard * n // a.num_shards, (a.shard + 1) * n // a.num_shards)
    t0 = time.time()
    qtok = Tokenizer.from_file(QWEN_TOK)
    ids = []
    for d in docs:
        x, y = int(offsets[d]), int(offsets[d + 1])
        if y > x and tokens[y - 1] == EOS:
            y -= 1
        ids.append(tokens[x:min(y, x + 4 * a.max_len)].tolist())  # enough text for max_len gte tokens
    texts = qtok.decode_batch(ids, skip_special_tokens=False)
    tok, topic = load(f"{M}/TopicClassifier-NoURL")
    _, fmt = load(f"{M}/FormatClassifier-NoURL")
    btok = tok._tokenizer
    btok.enable_truncation(a.max_len)
    btok.no_padding()
    enc = [e.ids for e in btok.encode_batch(texts)]
    lens = np.array([len(e) for e in enc])
    print(f"shard {a.shard}: {len(docs):,} docs, {lens.sum():,} gte tokens, prep {time.time() - t0:.0f}s", flush=True)
    tp = np.zeros((len(docs), 24), np.float16)
    fp = np.zeros((len(docs), 24), np.float16)
    order = np.argsort(lens)[::-1]
    i, t1 = 0, time.time()
    with torch.inference_mode():
        while i < len(order):
            L = int(lens[order[i]])
            sel = order[i:i + max(1, a.batch_tokens // L)]
            x = np.zeros((len(sel), L), np.int64)
            msk = np.zeros((len(sel), L), np.int64)
            for r, c in enumerate(sel):
                x[r, :lens[c]] = enc[c]
                msk[r, :lens[c]] = 1
            x, msk = torch.from_numpy(x).cuda(), torch.from_numpy(msk).cuda()
            tp[sel] = topic(input_ids=x, attention_mask=msk).logits.float().softmax(-1).cpu().numpy()
            fp[sel] = fmt(input_ids=x, attention_mask=msk).logits.float().softmax(-1).cpu().numpy()
            i += len(sel)
    np.savez(out + ".tmp.npz", doc=docs, topic_probs=tp, format_probs=fp, n_tokens=lens)
    os.replace(out + ".tmp.npz", out)
    print(f"done: {len(docs):,} docs, models {time.time() - t1:.0f}s, total {time.time() - t0:.0f}s "
          f"({lens.sum() / (time.time() - t1):,.0f} gte tok/s for both models)")


if __name__ == "__main__":
    main()
