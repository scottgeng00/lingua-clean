"""Diversity of the multi-sample synthetic arms, measured with Vendi scores of Qwen3-Embedding-4B embeddings.

  within:  per source document, the Vendi score of its 8 views (max 8), averaged over --groups documents.
  across:  per batch of 256 random documents (one random view each), the Vendi score (max 256), averaged over
           --batches batches; "real" = the source documents themselves. Every arm uses the same documents.

Vendi score = exp(entropy of the eigenvalues of K / n), K the cosine-similarity matrix. Also reported: mean pairwise
cosine similarity (off-diagonal) and mean view length in tokens. Documents are pool documents (first 0.25x C
sources, setup/build_pool_stream.py) with 8 non-empty views in every arm; texts truncated to --max-tokens.

    .venv-kd/bin/python analysis/multisample_vendi.py --out analysis/multisample_vendi.json
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

RUNS = "/home/sgeng/scottg/pretrain/simplescale/runs"
ARMS = ["direct_nothink", "direct_think", "mix", "ideas"]
MANIFEST = f"{RUNS}/manifests/dclm.f200w.ideas.manifest.jsonl"
MODEL = "/home/sgeng/scottg/pretrain/assets/models/Qwen3-Embedding-4B"
POOL_DOCS = 1_503_775  # documents touched by the 0.25x C pool
CHUNK = 64


def load_chunks(arm, chunks):
    shards = {s["chunk"]: s["uri"] for s in json.load(open(f"{RUNS}/f200w-qwen35-9b-{arm}/results/dataset.json"))["shards"]}
    out = {}
    for c in chunks:
        for line in open(shards[c]):
            r = json.loads(line)["result"]
            out[r["source"]["src_row"]] = [s["text"].strip() for s in r["samples"]]
    return out


def source_texts(rows):
    """Source documents by src_row (= DCLM source_row; every pool document is in dclm_8m_0)."""
    want, out = set(rows), {}
    with open(MANIFEST) as f:
        for line in f:
            task = json.loads(line)
            i = task["source_row"]
            if i in want and task["source_subset"] == "dclm_8m_0":
                ref = task["_ref"]
                with open(ref["path"], "rb") as g:
                    out[i] = json.loads(os.pread(g.fileno(), ref["length"], ref["offset"]))["text"]
                if len(out) == len(want):
                    break
    return out


@torch.no_grad()
def embed(texts, tok, model, max_tokens, batch_tokens=65536):
    ids = [x[:max_tokens - 1] + [tok.pad_token_id] for x in tok(texts, add_special_tokens=False)["input_ids"]]
    order = np.argsort([-len(x) for x in ids])
    out = np.zeros((len(texts), model.config.hidden_size), np.float32)
    i = 0
    while i < len(order):
        n = max(1, batch_tokens // len(ids[order[i]]))
        idx = order[i:i + n]
        L = len(ids[idx[0]])
        batch = torch.full((len(idx), L), tok.pad_token_id)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for j, k in enumerate(idx):  # left padding: the last position is every text's final <|endoftext|>
            batch[j, L - len(ids[k]):] = torch.tensor(ids[k])
            mask[j, L - len(ids[k]):] = 1
        h = model(input_ids=batch.cuda(), attention_mask=mask.cuda()).last_hidden_state[:, -1]
        out[idx] = torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()
        i += n
        print(f"embedded {i}/{len(texts)}", flush=True)
    return out, [len(x) for x in ids]


def vendi(X):
    lam = np.linalg.eigvalsh(X @ X.T / len(X))
    lam = lam[lam > 1e-12]
    return float(np.exp(-(lam * np.log(lam)).sum()))


def mean_cos(X):
    K = X @ X.T
    n = len(X)
    return float((K.sum() - np.trace(K)) / (n * (n - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=48, help="random generation chunks (64 documents each) to read")
    ap.add_argument("--groups", type=int, default=256)
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    chunks = rng.sample(range(POOL_DOCS // CHUNK), a.chunks)
    views = {arm: load_chunks(arm, chunks) for arm in ARMS}
    docs = sorted(set.intersection(*[{r for r, v in views[arm].items() if len(v) == 8 and all(v)} for arm in ARMS]))
    print(f"{len(docs)} documents with 8 non-empty views in every arm", flush=True)
    groups = rng.sample(docs, a.groups)
    batches = [rng.sample(docs, a.batch_size) for _ in range(a.batches)]
    pick = {(b, r): rng.randrange(8) for b, rows in enumerate(batches) for r in rows}  # same view index across arms
    real = source_texts({r for rows in batches for r in rows})

    # Unique texts to embed: (arm, row, view) keys, view -1 = the source document.
    keys = sorted({(arm, r, v) for arm in ARMS for r in groups for v in range(8)}
                  | {(arm, r, pick[b, r]) for arm in ARMS for b, rows in enumerate(batches) for r in rows}
                  | {("real", r, -1) for rows in batches for r in rows}
                  | {("mix", r, v) for rows in batches for r in rows for v in range(8)})  # per-format diagnostic
    texts = [real[r] if arm == "real" else views[arm][r][v] for arm, r, v in keys]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    E, lens = embed(texts, tok, model, a.max_tokens)
    at = {k: i for i, k in enumerate(keys)}

    res = {}
    for arm in ARMS + ["real"]:
        r = {}
        if arm != "real":
            w = [np.stack([E[at[arm, g, v]] for v in range(8)]) for g in groups]
            r["within_vendi"] = float(np.mean([vendi(x) for x in w]))
            r["within_vendi_sem"] = float(np.std([vendi(x) for x in w]) / np.sqrt(len(w)))
            r["within_cos"] = float(np.mean([mean_cos(x) for x in w]))
        acr = [np.stack([E[at[arm, x, -1 if arm == "real" else pick[b, x]]] for x in rows]) for b, rows in enumerate(batches)]
        r["across_vendi"] = float(np.mean([vendi(x) for x in acr]))
        r["across_vendi_sem"] = float(np.std([vendi(x) for x in acr]) / np.sqrt(len(acr)))
        r["across_cos"] = float(np.mean([mean_cos(x) for x in acr]))
        r["mean_tokens"] = float(np.mean([lens[i] for i, k in enumerate(keys) if k[0] == arm]))
        res[arm] = r
        print(arm, json.dumps(r), flush=True)
    # mix: across-batch Vendi with every document in one format (view index = format, see harness MIX order), and
    # cosine between views of different documents in the same vs different formats.
    kinds = ["article", "discussion", "faq", "math", "narrative", "table", "tutorial", "wiki"]
    res["mix_by_format"] = {k: float(np.mean([vendi(np.stack([E[at["mix", x, v]] for x in rows])) for rows in batches]))
                            for v, k in enumerate(kinds)}
    X = np.stack([np.stack([E[at["mix", x, v]] for v in range(8)]) for x in batches[0]])  # [docs, formats, dim]
    K = np.einsum("afd,bgd->afbg", X, X)
    other = ~np.eye(len(X), dtype=bool)[:, None, :, None]
    same_fmt = np.eye(8, dtype=bool)[None, :, None, :]
    res["mix_cross_doc_cos"] = {"same_format": float(K[other & same_fmt].mean()),
                                "different_format": float(K[other & ~same_fmt].mean())}
    print(json.dumps({k: res[k] for k in ("mix_by_format", "mix_cross_doc_cos")}), flush=True)
    # Faithfulness: cosine between each view and its own source document (batch documents, every view embedded).
    fmt = {"mix": lambda v: f"mix:{kinds[v]}"}
    cos = {}
    for (arm, x, v), i in at.items():
        if arm != "real" and ("real", x, -1) in at:
            cos.setdefault(fmt.get(arm, lambda v: arm)(v), []).append(float(E[i] @ E[at["real", x, -1]]))
    res["view_source_cos"] = {k: float(np.mean(c)) for k, c in sorted(cos.items())}
    print(json.dumps(res["view_source_cos"]), flush=True)
    np.savez(a.out.replace(".json", ".npz"), E=E.astype(np.float16), keys=np.array([f"{k[0]}|{k[1]}|{k[2]}" for k in keys]))
    json.dump({"args": vars(a), "documents": len(docs), "results": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
