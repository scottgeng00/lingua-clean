"""Qwen3 token stream of the multi-sample study's source documents, in manifest order, for data-constrained runs
(lingua/token_stream.py "repeat" mode reads a pool of its first n tokens).

The sources are the filtered, deduplicated DCLM docs that simplescale rewrote (setup/build_multisample_manifests.py;
every arm's manifest is a prefix of the same order), so a pool of the first P tokens is exactly the real data whose
synthetic views setup/export_multisample.py --pool-sources exports. Documents are tokenized like
setup/pretokenize_dclm_qwen3.py (raw text, no special tokens, one <|endoftext|> each).

Output (--out): tokens.bin (uint32), doc_offsets.npy (int64 [n + 1]), doc_source.npz (subset index, source_row per
document), meta.json (counts and, per --pool-mult, the pool's token count and number of documents it touches).

    .venv-kd/bin/python setup/build_pool_stream.py --tokens 3000000000
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import orjson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_dataconstrained_configs import pool_tokens_v2  # noqa: E402

MANIFEST = "/home/sgeng/scottg/pretrain/simplescale/runs/manifests/dclm.f200w.ideas.manifest.jsonl"  # 4.5M sources
OUT = "/home/sgeng/scottg/pretrain/assets/data/dclm_f200w_qwen3_tokens"
TOKENIZER = "/home/sgeng/scottg/pretrain/assets/models/Qwen3-0.6B-Base-tokenizer"
EOS = 151643
BLOCK = 5_000  # documents per task
_tok = None


def _tokenizer():
    global _tok
    if _tok is None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(TOKENIZER)
    return _tok


def tokenize(task):
    k, refs = task
    texts, files = [], {}
    for ref in refs:  # one handle per chunk file: opening a file per document is slow on NFS
        f = files.get(ref["path"]) or files.setdefault(ref["path"], open(ref["path"], "rb"))
        texts.append(orjson.loads(os.pread(f.fileno(), ref["length"], ref["offset"]))["text"])
    for f in files.values():
        f.close()
    ids = _tokenizer()(texts, add_special_tokens=False)["input_ids"]
    arrs = [np.asarray(x + [EOS], dtype=np.uint32) for x in ids]
    return k, np.concatenate(arrs), np.asarray([len(a) for a in arrs], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--tokens", type=int, required=True, help="stop after the document that reaches this many tokens")
    ap.add_argument("--pool-mult", type=float, nargs="+", default=[0.125, 0.25, 0.5])
    ap.add_argument("--workers", type=int, default=60)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lines = open(a.manifest, "rb")
    rows, lens, total = [], [], 0
    with open(f"{a.out}/tokens.bin.tmp", "wb") as tokens, Pool(a.workers) as pool:
        while total < a.tokens:  # a few million documents per round, written in manifest order
            batch = []
            for _ in range(a.workers * 4):
                block = [orjson.loads(line) for _, line in zip(range(BLOCK), lines)]
                if not block:
                    break
                batch.append(block)
            if not batch:
                break
            for k, toks, ls in pool.imap(tokenize, enumerate([[r["_ref"] for r in b] for b in batch])):
                tokens.write(toks.tobytes())
                rows += [(r["source_subset"], r["source_row"]) for r in batch[k]]
                lens += ls.tolist()
                total += int(ls.sum())
            print(f"{len(lens):,} docs, {total:,} tokens", flush=True)
    # Keep documents up to the one that reaches --tokens.
    n = min(len(lens), int(np.searchsorted(np.cumsum(lens), a.tokens, side="left")) + 1)
    rows, lens = rows[:n], lens[:n]
    os.truncate(f"{a.out}/tokens.bin.tmp", int(sum(lens)) * 4)
    os.replace(f"{a.out}/tokens.bin.tmp", f"{a.out}/tokens.bin")
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    assert os.path.getsize(f"{a.out}/tokens.bin") == offsets[-1] * 4
    np.save(f"{a.out}/doc_offsets.npy", offsets)
    subsets = sorted({s for s, _ in rows})
    np.savez(f"{a.out}/doc_source.npz", subset=np.asarray([subsets.index(s) for s, _ in rows], np.int8),
             source_row=np.asarray([r for _, r in rows], np.int64))
    pools = {}
    for m in a.pool_mult:
        p = pool_tokens_v2(1152, m)
        if p < offsets[-1]:  # documents with any token in the pool (the last one is cut)
            pools[str(m)] = {"tokens": int(p), "docs": int(np.searchsorted(offsets, p, side="left"))}
    meta = {"docs": len(lens), "tokens": int(offsets[-1]), "manifest": a.manifest, "subsets": subsets,
            "order": "manifest order (filtered, deduplicated sources)", "pools_w1152": pools, "tokenizer": TOKENIZER,
            "eos_id": EOS, "dtype": "uint32", "script": "setup/build_pool_stream.py"}
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
