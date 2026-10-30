"""Pre-tokenize DCLM subsets with the Qwen3 tokenizer into one flat token stream in global document order.

Global order = dclm_8m_0 rows 0..8M-1, then dclm_8m_1 rows (the order of the scottgeng00/dclm-8m upload, which is
already shuffled). In each lingua subset row j sits in chunk j % 16, line j // 16. Every document is encoded without
special tokens and followed by one <|endoftext|> (Qwen3's end-of-document token, id 151643).

Output (--out):
    tokens.bin       uint32, all documents back to back
    doc_offsets.npy  int64 [n_docs + 1]; document d is tokens[doc_offsets[d]:doc_offsets[d + 1]]
    meta.json        tokenizer, subsets, per-subset doc / token counts

Used by the data-constrained ladder (lingua/token_stream.py): stage 1 reads a prefix of this stream once, stage 2
repeats a prefix (the pool).

    .venv-kd/bin/python -m setup.pretokenize_dclm_qwen3
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import orjson

DATA = "/home/sgeng/scottg/pretrain/assets/data/dclm_subsets"
TOKENIZER = "/home/sgeng/scottg/pretrain/assets/models/Qwen3-0.6B-Base-tokenizer"
OUT = "/home/sgeng/scottg/pretrain/assets/data/dclm_qwen3_tokens"
N_CHUNKS = 16
PARTS = 8  # tasks per chunk file (each reads the file, tokenizes every PARTS-th line)
BATCH = 512

_tok = None


def _tokenizer():
    global _tok
    if _tok is None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(TOKENIZER)
    return _tok


def chunk_path(subset, c):
    return f"{DATA}/{subset}/{subset}.chunk.{c:02d}.jsonl"


def tokenize_part(task):
    """Tokenize lines part, part + PARTS, ... of one chunk file -> tmp tokens file + per-line lengths."""
    subset, c, part, tmp = task
    tok = _tokenizer()
    eos = tok.eos_token_id
    lengths, out = [], open(f"{tmp}/{subset}.{c:02d}.{part}.u32", "wb")
    texts = []

    def flush():
        for ids in tok(texts, add_special_tokens=False)["input_ids"]:
            a = np.asarray(ids + [eos], dtype=np.uint32)
            out.write(a.tobytes())
            lengths.append(len(a))
        texts.clear()

    with open(chunk_path(subset, c), "rb", buffering=1 << 24) as f:
        for i, line in enumerate(f):
            if i % PARTS == part:
                texts.append(orjson.loads(line)["text"])
                if len(texts) == BATCH:
                    flush()
    if texts:
        flush()
    out.close()
    np.save(f"{tmp}/{subset}.{c:02d}.{part}.len.npy", np.asarray(lengths, dtype=np.int64))
    return subset, c, part, len(lengths)


def place_part(task):
    """Copy one part's documents into their global positions in tokens.bin."""
    subset, s_index, rows, c, part, tmp, out = task
    lens = np.load(f"{tmp}/{subset}.{c:02d}.{part}.len.npy")
    offsets = np.load(f"{out}/doc_offsets.npy", mmap_mode="r")
    dst = np.memmap(f"{out}/tokens.bin", dtype=np.uint32, mode="r+")
    src = np.fromfile(f"{tmp}/{subset}.{c:02d}.{part}.u32", dtype=np.uint32)
    lines = np.arange(part, part + PARTS * len(lens), PARTS)  # line indices within the chunk
    docs = s_index + c + N_CHUNKS * lines                       # global document ids
    pos = 0
    for d, n in zip(docs.tolist(), lens.tolist()):
        o = int(offsets[d])
        dst[o:o + n] = src[pos:pos + n]
        pos += n
    dst.flush()
    return subset, c, part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["dclm_8m_0", "dclm_8m_1"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=140)
    a = ap.parse_args()
    tmp = f"{a.out}/tmp"
    os.makedirs(tmp, exist_ok=True)
    rows = {s: json.load(open(f"{DATA}/{s}/dataset.json"))["rows"] for s in a.subsets}

    tasks = [(s, c, p, tmp) for s in a.subsets for c in range(N_CHUNKS) for p in range(PARTS)]
    with Pool(a.workers) as pool:
        for k, (s, c, p, n) in enumerate(pool.imap_unordered(tokenize_part, tasks)):
            print(f"tokenized {k + 1}/{len(tasks)}: {s} chunk {c} part {p} ({n:,} docs)", flush=True)

    # Global per-document lengths: subset s row j = chunk j % 16, line j // 16 = part (line % PARTS).
    lengths, s_starts, start = [], {}, 0
    for s in a.subsets:
        L = np.zeros(rows[s], dtype=np.int64)
        for c in range(N_CHUNKS):
            for p in range(PARTS):
                lens = np.load(f"{tmp}/{s}.{c:02d}.{p}.len.npy")
                lines = np.arange(p, p + PARTS * len(lens), PARTS)
                L[c + N_CHUNKS * lines] = lens
        assert (L > 0).all(), f"{s}: missing documents"
        s_starts[s] = start
        start += rows[s]
        lengths.append(L)
    lengths = np.concatenate(lengths)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    np.save(f"{a.out}/doc_offsets.npy", offsets)
    np.memmap(f"{a.out}/tokens.bin", dtype=np.uint32, mode="w+", shape=(int(offsets[-1]),)).flush()
    print(f"{len(lengths):,} docs, {int(offsets[-1]):,} tokens; placing", flush=True)

    place = [(s, s_starts[s], rows[s], c, p, tmp, a.out) for s in a.subsets for c in range(N_CHUNKS) for p in range(PARTS)]
    with Pool(min(a.workers, 64)) as pool:
        for k, _ in enumerate(pool.imap_unordered(place_part, place)):
            if k % 16 == 0:
                print(f"placed {k + 1}/{len(place)}", flush=True)

    meta = {"tokenizer": TOKENIZER, "eos_id": 151643, "dtype": "uint32", "order": "subsets in order, rows in order",
            "subsets": {s: {"first_doc": s_starts[s], "docs": rows[s],
                            "tokens": int(offsets[s_starts[s] + rows[s]] - offsets[s_starts[s]])} for s in a.subsets},
            "docs": len(lengths), "tokens": int(offsets[-1]), "script": "setup/pretokenize_dclm_qwen3.py"}
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
    for f in os.listdir(tmp):
        os.remove(f"{tmp}/{f}")
    os.rmdir(tmp)
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()


def pretokenize_heldout(src="/home/sgeng/scottg/pretrain/assets/data/dclm_subsets/dclm_heldout/dclm_heldout.val.jsonl",
                        out="/home/sgeng/scottg/pretrain/assets/data/dclm_qwen3_tokens/dclm_heldout"):
    """dclm_heldout in the same format, plus the UTF-8 byte count of every document (for bits per byte)."""
    os.makedirs(out, exist_ok=True)
    tok = _tokenizer()
    texts = [orjson.loads(l)["text"] for l in open(src, "rb")]
    ids = tok(texts, add_special_tokens=False)["input_ids"]
    arrs = [np.asarray(x + [tok.eos_token_id], dtype=np.uint32) for x in ids]
    np.concatenate(arrs).tofile(f"{out}/tokens.bin")
    np.save(f"{out}/doc_offsets.npy", np.concatenate([[0], np.cumsum([len(a) for a in arrs])]).astype(np.int64))
    n_bytes = np.asarray([len(t.encode("utf-8")) for t in texts], dtype=np.int64)
    np.save(f"{out}/doc_bytes.npy", n_bytes)
    meta = {"source": src, "docs": len(texts), "tokens": int(sum(len(a) for a in arrs)), "bytes": int(n_bytes.sum()),
            "tokenizer": TOKENIZER, "eos_id": tok.eos_token_id}
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    print(meta)
