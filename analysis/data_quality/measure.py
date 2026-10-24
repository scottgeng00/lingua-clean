"""Quality and diversity of real vs rewritten training splits.

Sets: S1 (dclm_8m_0), S2 (dclm_8m_1) and every dclm_rewrites/dclm_8m_1_* split (rewrites of S2).
The splits are uniform shuffles, so the first N lines of each of the 16 chunks are a uniform random
sample; rewrite splits are row-aligned with S2, so S2 and its rewrites are sampled at the *same*
rows (paired: differences come from the rewriting, not from which docs were drawn).

Steps (each writes to --out):
  sample   first 3125 lines/chunk = 50,000 docs per set -> <set>.texts.jsonl (CPU)
  quality  DCLM fastText classifier (mlfoundations/fasttext-oh-eli5, the DCLM-baseline filter):
           P(__label__hq) with newlines replaced by spaces, as in DCLM -> <set>.quality.npy (CPU,
           needs the fasttext venv)
  embed    (GPU, sglang; simplescale venv) on the first 5,120 docs of each sample (first 320
           lines/chunk): Qwen3-Embedding-8B (last-token pooling, L2-normalized, no instruction,
           max 4096 tokens) -> <set>.emb.npy
  weborg   (CPU by default) same 5,120 docs: WebOrganizer Topic/Format classifiers (NoURL variants,
           so rewrites and real docs are judged on text alone; softmax probs, max 4096 tokens)
           -> <set>.{topic,format}.npy
  report   Vendi score (exp entropy of eigenvalues of K/n, K = cosine kernel) on 5 disjoint
           1,024-doc subsets -> mean +- std; quality mean / median, with the std of the mean over the same 1,024-doc
           subsets; WebOrganizer distributions
           (mean probs), their entropy and total-variation distance to S2.

    python analysis/data_quality/measure.py sample --out OUT
    <fasttext venv>/bin/python analysis/data_quality/measure.py quality --out OUT
    <simplescale venv>/bin/python analysis/data_quality/measure.py embed --out OUT --set <name>  # 1 GPU
    python analysis/data_quality/measure.py weborg --out OUT --set <name>   # CPU
    python analysis/data_quality/measure.py report --out OUT
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np

DATA = "/home/sgeng/scottg/pretrain/assets/data"
FASTTEXT = "/checkpoint/dream-sc-3/scottg/pretrain/assets/models/dclm-fasttext/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
EMBED = "Qwen/Qwen3-Embedding-8B"
TOPIC, FORMAT = "WebOrganizer/TopicClassifier-NoURL", "WebOrganizer/FormatClassifier-NoURL"
N_CHUNKS, Q_PER_CHUNK, G_PER_CHUNK, SUB = 16, 3125, 320, 1024
MAX_TOKENS = 4096


def sets():
    out = {"S1": f"{DATA}/dclm_subsets/dclm_8m_0", "S2": f"{DATA}/dclm_subsets/dclm_8m_1"}
    for d in sorted(glob.glob(f"{DATA}/dclm_rewrites/dclm_8m_1_*")):
        name = os.path.basename(d).replace("dclm_8m_1_qwen35_", "Rw(S2) ").replace("_g1_max4096", "")
        out[name] = d
    return out


def fname(out, name, ext):
    return os.path.join(out, name.replace(" ", "_").replace("(", "").replace(")", "") + ext)


def cmd_sample(a):
    os.makedirs(a.out, exist_ok=True)
    ids = {}
    for name, d in sets().items():
        base = os.path.basename(d)
        rows = [[] for _ in range(N_CHUNKS)]
        for c in range(N_CHUNKS):
            with open(f"{d}/{base}.chunk.{c:02d}.jsonl") as f:
                rows[c] = [json.loads(l) for l in itertools.islice(f, Q_PER_CHUNK)]
        # interleave chunks so any prefix of the file is a round-robin (uniform) sample
        docs = [rows[c][i] for i in range(Q_PER_CHUNK) for c in range(N_CHUNKS)]
        ids[name] = [x["id"] for x in docs]
        with open(fname(a.out, name, ".texts.jsonl"), "w") as f:
            for x in docs:
                f.write(json.dumps({"id": x["id"], "text": x["text"]}) + "\n")
        print(f"{name}: {len(docs)} docs", flush=True)
    for name in ids:  # pairing check: every rewrite set is drawn at S2's rows
        if name.startswith("Rw(S2)"):
            assert ids[name] == ids["S2"], name
    json.dump(list(sets()), open(os.path.join(a.out, "sets.json"), "w"))


def load_texts(out, name, n=None):
    with open(fname(out, name, ".texts.jsonl")) as f:
        return [json.loads(l)["text"] for l in itertools.islice(f, n)]


def cmd_quality(a):
    import fasttext
    model = fasttext.load_model(FASTTEXT)
    for name in json.load(open(os.path.join(a.out, "sets.json"))):
        texts = load_texts(a.out, name)
        labels, probs = model.predict([t.replace("\n", " ") for t in texts], k=2)
        hq = np.array([p[list(l).index("__label__hq")] for l, p in zip(labels, probs)], dtype=np.float32)
        np.save(fname(a.out, name, ".quality.npy"), hq)
        print(f"{name}: quality mean {hq.mean():.4f}", flush=True)


def _batches(texts, tk, bs, max_len, dev):
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))  # length-sorted for padding
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        enc = tk([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len,
                 return_tensors="pt").to(dev)
        yield idx, enc


def cmd_embed(a):
    """Qwen3-Embedding-8B served by an in-process sglang Engine (run with the simplescale venv).

    sglang's Qwen3 embedding path pools the last token and L2-normalizes, matching the model's
    sentence-transformers config; the tokenizer's post-processor appends <|endoftext|>. Texts are
    truncated to MAX_TOKENS - 1 tokens first so every input fits the context.
    """
    import sglang as sgl
    from transformers import AutoTokenizer

    names = json.load(open(os.path.join(a.out, "sets.json")))
    n = G_PER_CHUNK * N_CHUNKS
    tok = AutoTokenizer.from_pretrained(EMBED)
    engine = sgl.Engine(model_path=EMBED, is_embedding=True, context_length=MAX_TOKENS + 64,
                        mem_fraction_static=0.85, log_level="error")
    try:
        for name in [a.set] if a.set else names:
            texts = load_texts(a.out, name, n)
            ids = tok(texts, add_special_tokens=False)["input_ids"]
            texts = [tok.decode(x[:MAX_TOKENS - 1]) if len(x) >= MAX_TOKENS else t for t, x in zip(texts, ids)]
            E = np.zeros((n, 4096), dtype=np.float32)
            for k in range(0, n, 512):
                out = engine.encode(texts[k:k + 512])
                E[k:k + len(out)] = np.array([o["embedding"] for o in out], dtype=np.float32)
            norms = np.linalg.norm(E, axis=1)
            assert np.allclose(norms, 1, atol=1e-2), (norms.min(), norms.max())
            np.save(fname(a.out, name, ".emb.npy"), E)
            print(f"{name}: embeddings done", flush=True)
    finally:
        engine.shutdown()


def cmd_weborg(a):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = a.device
    if dev == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
    names = json.load(open(os.path.join(a.out, "sets.json")))
    n = G_PER_CHUNK * N_CHUNKS
    for key, repo in (("topic", TOPIC), ("format", FORMAT)):
        tk = AutoTokenizer.from_pretrained(repo)
        m = AutoModelForSequenceClassification.from_pretrained(
            repo, trust_remote_code=True, use_memory_efficient_attention=False,
            torch_dtype=torch.float32 if dev == "cpu" else torch.bfloat16).to(dev).eval()
        for name in [a.set] if a.set else names:
            texts = load_texts(a.out, name, n)
            P = np.zeros((n, m.config.num_labels), dtype=np.float32)
            with torch.no_grad():
                for idx, enc in _batches(texts, tk, 16, MAX_TOKENS, dev):
                    P[idx] = m(**enc).logits.float().softmax(-1).cpu().numpy()
            np.save(fname(a.out, name, f".{key}.npy"), P)
            print(f"{name}: {key} done", flush=True)


def vendi(X):
    K = X @ X.T / len(X)
    w = np.linalg.eigvalsh(K)
    w = w[w > 1e-12]
    return float(np.exp(-(w * np.log(w)).sum()))


def cmd_report(a):
    from transformers import AutoConfig
    names = json.load(open(os.path.join(a.out, "sets.json")))
    labels = {k: [AutoConfig.from_pretrained(r, trust_remote_code=True).id2label[i]
                  for i in range(AutoConfig.from_pretrained(r, trust_remote_code=True).num_labels)]
              for k, r in (("topic", TOPIC), ("format", FORMAT))}
    n_sub = (G_PER_CHUNK * N_CHUNKS) // SUB
    res = {}
    for name in names:
        q = np.load(fname(a.out, name, ".quality.npy"))
        E = np.load(fname(a.out, name, ".emb.npy"))
        subs = [slice(i * SUB, (i + 1) * SUB) for i in range(n_sub)]
        v = [vendi(E[s]) for s in subs]
        r = {"quality_mean": float(q.mean()), "quality_sem": float(q.std() / np.sqrt(len(q))),
             "quality_median": float(np.median(q)),
             "quality_mean_std_over_1024": float(np.std([q[s].mean() for s in subs])),
             "vendi_mean": float(np.mean(v)), "vendi_std": float(np.std(v)), "vendi_subsets": v,
             "n_quality": int(len(q)), "n_embed": int(len(E))}
        for key in ("topic", "format"):
            P = np.load(fname(a.out, name, f".{key}.npy"))
            dist = P.mean(0)
            r[key] = dict(zip(labels[key], map(float, dist)))
            r[f"{key}_entropy"] = float(-(dist * np.log(dist)).sum())
            r[f"{key}_sub_dists"] = [P[s].mean(0).tolist() for s in subs]
        res[name] = r
    s2 = res["S2"]
    for name, r in res.items():
        for key in ("topic", "format"):
            ref = np.array(list(s2[key].values()))
            tv = [0.5 * np.abs(np.array(d) - np.array(s2[f"{key}_sub_dists"][i])).sum()
                  for i, d in enumerate(r[f"{key}_sub_dists"])]
            r[f"{key}_tv_to_S2"] = float(0.5 * np.abs(np.array(list(r[key].values())) - ref).sum())
            r[f"{key}_tv_to_S2_std"] = float(np.std(tv))
    json.dump(res, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    for name, r in res.items():
        print(f"{name:22s} quality {r['quality_mean']:.4f}+-{r['quality_mean_std_over_1024']:.4f}  "
              f"vendi {r['vendi_mean']:.1f}+-{r['vendi_std']:.1f}  topicH {r['topic_entropy']:.3f}  "
              f"formatH {r['format_entropy']:.3f}  TV(topic) {r['topic_tv_to_S2']:.3f}  TV(format) {r['format_tv_to_S2']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sample", "quality", "embed", "weborg", "report"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--set")
    ap.add_argument("--device", default="cpu", help="weborg only: cpu or cuda")
    a = ap.parse_args()
    {"sample": cmd_sample, "quality": cmd_quality, "embed": cmd_embed, "weborg": cmd_weborg, "report": cmd_report}[a.step](a)


if __name__ == "__main__":
    main()
