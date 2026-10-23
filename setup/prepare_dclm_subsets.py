"""Cut disjoint, equal-size DCLM subsets for the generator-scaling experiments.

Rows [k*N, (k+1)*N) of dclm_shuffled/dclm.chunk.00.jsonl become dclm_subsets/dclm_<n>_<k>
(default N = 8M docs ~ 10B tokens, 4 subsets). That file is a uniform shuffle of DCLM, and
its first 15M lines are byte-identical to simplescale's dclm.first-15m (all earlier runs), so
dclm_8m_0 contains every doc used so far. Lines are copied byte-for-byte.

Layout matches the other prepared splits: local row j of a subset goes to chunk j % 16 at
line j // 16, so data.source_max_docs = M selects exactly the first M rows, and a rewrite
split written in the same layout (setup/prepare_dclm_rewrites.py) is row-aligned. Global row
of (subset k, chunk c, line i) = k*N + 16*i + c.

Every row is checked against the held-out DCLM validation set (dclm_shuffled/dclm.val.jsonl,
the first 1000 lines of the shuffled stream, so disjoint by line) by id (URL) and by exact
text; matches are counted, listed in dataset.json and, with --drop-val-overlap, not written
(the subset then has fewer than N rows). Token counts come from a 1-in-200 sample.

    python setup/prepare_dclm_subsets.py --out /home/sgeng/scottg/pretrain/assets/data/dclm_subsets
"""

import argparse
import hashlib
import json
import os
import shutil
import time

import orjson

DCLM = "/home/sgeng/scottg/pretrain/assets/data/dolmino_splits/dclm_shuffled"
TOKENIZER = "/home/sgeng/scottg/pretrain/assets/models/Qwen3.5-2B-tokenizer-eot"
N_CHUNKS = 16
SAMPLE_EVERY = 200


def text_key(text):
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).digest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default=f"{DCLM}/dclm.chunk.00.jsonl")
    ap.add_argument("--val", default=f"{DCLM}/dclm.val.jsonl")
    ap.add_argument("--docs-per-subset", type=int, default=8_000_000)
    ap.add_argument("--n-subsets", type=int, default=4)
    ap.add_argument("--name", default="dclm_8m")
    ap.add_argument("--drop-val-overlap", action="store_true")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    args = ap.parse_args()
    t0 = time.time()
    N, K = args.docs_per_subset, args.n_subsets

    val = [orjson.loads(l) for l in open(args.val, "rb")]
    val_ids = {v["id"] for v in val}
    val_texts = {text_key(v["text"]) for v in val}
    print(f"{len(val)} val docs ({len(val_ids)} ids)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    names = [f"{args.name}_{k}" for k in range(K)]
    tmp = {n: os.path.join(args.out, f".{n}.tmp") for n in names}
    for n in names:
        if os.path.exists(os.path.join(args.out, n)):
            raise SystemExit(f"{n} already exists in {args.out}")
        shutil.rmtree(tmp[n], ignore_errors=True)
        os.makedirs(tmp[n])

    stats = {n: {"rows": 0, "bytes": 0, "val_id_overlap": [], "val_text_overlap": [], "dropped": 0,
                 "sample_texts": []} for n in names}
    with open(args.source, "rb", buffering=1 << 24) as src:
        for k, n in enumerate(names):
            outs = [open(os.path.join(tmp[n], f"{n}.chunk.{c:02d}.jsonl"), "wb", buffering=1 << 22)
                    for c in range(N_CHUNKS)]
            st, j = stats[n], 0
            for r in range(N):
                line = src.readline()
                if not line:
                    raise SystemExit(f"{args.source} ended at row {k * N + r}")
                rec = orjson.loads(line)
                id_hit, text_hit = rec["id"] in val_ids, text_key(rec["text"]) in val_texts
                if id_hit:
                    st["val_id_overlap"].append({"row": k * N + r, "id": rec["id"]})
                if text_hit:
                    st["val_text_overlap"].append({"row": k * N + r, "id": rec["id"]})
                if args.drop_val_overlap and (id_hit or text_hit):
                    st["dropped"] += 1
                    continue
                outs[j % N_CHUNKS].write(line)
                if j % SAMPLE_EVERY == 0:
                    st["sample_texts"].append(rec["text"])
                st["rows"] += 1
                st["bytes"] += len(line)
                j += 1
                if r % 1_000_000 == 0:
                    print(f"{n}: {r:,} rows, {time.time() - t0:.0f}s", flush=True)
            for f in outs:
                f.close()
            print(f"{n}: done, {st['rows']:,} rows, val overlap id={len(st['val_id_overlap'])} "
                  f"text={len(st['val_text_overlap'])}, {time.time() - t0:.0f}s", flush=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    for n in names:
        st = stats[n]
        sample = st.pop("sample_texts")
        per_doc = sum(len(x) + 1 for x in tok(sample)["input_ids"]) / len(sample)  # +1 for EOS
        st["qwen35_tokens_per_doc_incl_eos"] = round(per_doc, 1)
        st["qwen35_tokens_est"] = round(per_doc * st["rows"])
        st["token_sample_docs"] = len(sample)
        k = names.index(n)
        meta = {
            "description": f"Rows [{k * N}, {(k + 1) * N}) of {args.source}, copied byte-for-byte. Local row j "
                           f"-> chunk j % {N_CHUNKS}, line j // {N_CHUNKS}; global row = {k * N} + 16*line + chunk"
                           + (" (rows overlapping the val set dropped, so later rows shift)" if st["dropped"] else ""),
            "source": args.source, "global_rows": [k * N, (k + 1) * N], "chunks": N_CHUNKS,
            "val_file": args.val, "tokenizer": args.tokenizer, **st,
        }
        json.dump(meta, open(os.path.join(tmp[n], "dataset.json"), "w"), indent=2)
        os.replace(tmp[n], os.path.join(args.out, n))
        print(f"{n}: {st['qwen35_tokens_est'] / 1e9:.2f}B Qwen3.5 tokens ({per_doc:.0f}/doc)", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
