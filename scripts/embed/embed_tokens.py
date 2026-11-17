"""Embed documents of a pre-tokenized stream (setup/pretokenize_dclm_qwen3.py / pretokenize_paloma_qwen3.py layout:
tokens.bin uint32 + doc_offsets.npy) with Qwen3-Embedding-0.6B, one shard per call (one GPU).

The streams use the Qwen3 tokenizer, which is also Qwen3-Embedding's, so token ids are fed directly. Qwen3-Embedding
pools the hidden state of a final <|endoftext|> (id 151643), as its tokenizer appends for documents (no instruction
prompt for documents). Each document (without its trailing <|endoftext|>, if the stream has one) is split into
windows of at most --max-len - 1 tokens, each followed by <|endoftext|>, so no tokens are dropped. Output per shard:
    shard_XXXX.npy        float16 [n_chunks, 1024], L2-normalized
    shard_XXXX.meta.npz   doc (global document index), chunk (window index within the document), n_tokens

    python scripts/embed/embed_tokens.py --stream DIR --out OUT --shard 3 --num-shards 40 [--docs N]
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

MODEL = "/home/sgeng/scottg/pretrain/assets/models/Qwen3-Embedding-0.6B"
EOS = 151643


def chunks_of(tokens, offsets, d0, d1, max_len):
    docs, idx, starts, lens = [], [], [], []
    for d in range(d0, d1):
        a, b = int(offsets[d]), int(offsets[d + 1])
        if b > a and tokens[b - 1] == EOS:
            b -= 1
        w = max_len - 1
        for k, s in enumerate(range(a, max(b, a + 1), w)):
            docs.append(d); idx.append(k); starts.append(s); lens.append(min(w, b - s))
    return np.array(docs), np.array(idx), np.array(starts, dtype=np.int64), np.array(lens, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--docs", type=int, default=None, help="use only the first N documents")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch-tokens", type=int, default=262144, help="padded tokens per forward")
    a = ap.parse_args()
    out = f"{a.out}/shard_{a.shard:04d}"
    if os.path.exists(out + ".npy"):
        print(f"{out}.npy exists; skipping")
        return
    os.makedirs(a.out, exist_ok=True)
    tokens = np.memmap(f"{a.stream}/tokens.bin", dtype=np.uint32, mode="r")
    offsets = np.load(f"{a.stream}/doc_offsets.npy", mmap_mode="r")
    n = a.docs or len(offsets) - 1
    d0, d1 = a.shard * n // a.num_shards, (a.shard + 1) * n // a.num_shards
    docs, idx, starts, lens = chunks_of(tokens, offsets, d0, d1, a.max_len)
    print(f"shard {a.shard}/{a.num_shards}: docs {d0}..{d1 - 1}, {len(docs):,} chunks, {int(lens.sum()):,} tokens",
          flush=True)

    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model = model.cuda().eval()
    emb = np.zeros((len(docs), model.config.hidden_size), dtype=np.float16)
    order = np.argsort(lens + 1, kind="stable")[::-1]  # longest first: the first batch shows peak memory
    t0, done, i = time.time(), 0, 0
    with torch.inference_mode():
        while i < len(order):
            L = int(lens[order[i]]) + 1
            j = min(len(order), i + max(1, a.batch_tokens // L))
            sel = order[i:j]
            ids = np.full((len(sel), L), EOS, dtype=np.int64)
            mask = np.zeros((len(sel), L), dtype=np.int64)
            for r, c in enumerate(sel):
                m = int(lens[c])
                ids[r, :m] = tokens[starts[c]:starts[c] + m]
                ids[r, m] = EOS
                mask[r, :m + 1] = 1
            ids_t = torch.from_numpy(ids).cuda(non_blocking=True)
            mask_t = torch.from_numpy(mask).cuda(non_blocking=True)
            h = model(input_ids=ids_t, attention_mask=mask_t).last_hidden_state
            last = mask_t.sum(1) - 1  # right padding: the appended <|endoftext|>
            e = F.normalize(h[torch.arange(len(sel), device=h.device), last].float(), dim=-1)
            emb[sel] = e.half().cpu().numpy()
            done += int(lens[sel].sum()) + len(sel)
            i = j
            if (i // len(sel)) % 200 == 0 or i == len(order):
                el = time.time() - t0
                print(f"  {i:,}/{len(order):,} chunks, {done / el:,.0f} tok/s, {el / 60:.1f} min", flush=True)
    np.savez(out + ".meta.npz", doc=docs, chunk=idx, n_tokens=lens)
    np.save(out + ".tmp.npy", emb)
    os.replace(out + ".tmp.npy", out + ".npy")
    print(f"done: {len(docs):,} chunks in {(time.time() - t0) / 60:.1f} min ({done / (time.time() - t0):,.0f} tok/s)")


if __name__ == "__main__":
    main()
