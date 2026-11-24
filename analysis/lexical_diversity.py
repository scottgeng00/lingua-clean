"""Lexical diversity of synthetic arms vs real pool text on the same source documents (48 random 64-doc pool chunks).
  batch:  one random kept view per document, concatenated in random order, first N words -> distinct word types,
          hapax share, and type count of the real sources over the same N words.
  corpus: all kept views of those documents pooled in random order, first N words (repetition across views included).
Words = lowercase \\w+ tokens. Kept = finish_reason not rejected/filtered/length; text after the last </think>."""
import json, os, random, re, sys
from collections import Counter
RUNS = "/home/sgeng/scottg/pretrain/simplescale/runs"
MAN = f"{RUNS}/manifests/dclm.f200w.ideas.manifest.jsonl"
ARMS = {"direct_nothink (9B)": ["f200w-qwen35-9b-direct_nothink"], "direct_think (9B)": ["f200w-qwen35-9b-direct_think"],
        "mix (9B)": ["f200w-qwen35-9b-mix"], "ideas (9B)": ["f200w-qwen35-9b-ideas"], "arv2 (9B)": ["pool025-qwen35-9b-arv2"],
        "basegen (9B base)": ["pool025-g8-qwen35-9b-base-basegen"],
        "direct_nothink (2B)": ["pool025-g14-qwen35-2b-direct_nothink"], "direct_nothink (27B)": ["pool025-g12-qwen35-27b-direct_nothink"],
        "mix (2B)": ["pool025-r3-qwen35-2b-mix"], "mix (27B)": ["pool025-r2-qwen35-27b-mix"]}
WORD = re.compile(r"\w+")
def kept(s):
    f = str(s.get("finish_reason") or "")
    return s.get("text") and not f.startswith(("rejected", "filtered")) and f != "length"
def load(run, chunks64):
    """Views keyed by DCLM source_row for the pool documents in the given 64-document manifest chunks."""
    d = json.load(open(f"{RUNS}/{run}/results/dataset.json")); cs = d["chunk_size"]
    shards = {s["chunk"]: s["uri"] for s in d["shards"]}
    out = {}
    for c64 in chunks64:
        for c in range(c64 * 64 // cs, (c64 + 1) * 64 // cs):
            for line in open(shards[c]):
                r = json.loads(line)["result"]
                out[r["source"]["source_row"]] = [s["text"].split("</think>")[-1].strip() for s in r["samples"] if kept(s)]
    return out
def sources(chunks64):
    """Source texts keyed by source_row: manifest lines 64c .. 64c+63 (all arms' manifests share this order)."""
    want, out = {64 * c + i for c in chunks64 for i in range(64)}, {}
    with open(MAN) as f:
        for i, line in enumerate(f):
            if i in want:
                t = json.loads(line); ref = t["_ref"]
                with open(ref["path"], "rb") as g:
                    out[t["source_row"]] = json.loads(os.pread(g.fileno(), ref["length"], ref["offset"]))["text"]
                if len(out) == len(want): break
    return out
def stats(texts, n):
    words = [w for t in texts for w in WORD.findall(t.lower())][:n]
    c = Counter(words)
    return {"words": len(words), "types": len(c), "hapax_share": sum(v == 1 for v in c.values()) / max(1, len(c))}
def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
    rng = random.Random(0)
    chunks64 = rng.sample(range(1_503_775 // 64), 48)
    src = sources(chunks64)
    order = sorted(src); rng.shuffle(order)
    res = {"real (sources)": {"batch": stats([src[r] for r in order], n)}}
    for name, runs in ARMS.items():
        views = {}
        for run in runs:
            for r, v in load(run, chunks64).items(): views.setdefault(r, []).extend(v)
        docs = [r for r in order if views.get(r)]
        one = [rng.choice(views[r]) for r in docs]
        allv = [v for r in docs for v in views[r]]; rng.shuffle(allv)
        res[name] = {"docs": len(docs), "views_per_doc": len(allv) / len(docs), "batch": stats(one, n), "corpus": stats(allv, n)}
        print(name, json.dumps(res[name]), flush=True)
    print(f"\nN = {n:,} words. real types {res['real (sources)']['batch']['types']:,}, hapax {res['real (sources)']['batch']['hapax_share']:.3f}")
    print(f"{'arm':24s} {'batch types':>12s} {'vs real':>8s} {'hapax':>6s} {'corpus types':>13s} {'vs real':>8s} views/doc")
    R = res["real (sources)"]["batch"]["types"]
    for k, v in res.items():
        if k.startswith("real"): continue
        print(f"{k:24s} {v['batch']['types']:12,d} {v['batch']['types'] / R - 1:+8.0%} {v['batch']['hapax_share']:6.3f} {v['corpus']['types']:13,d} {v['corpus']['types'] / R - 1:+8.0%} {v['views_per_doc']:.1f}")
    json.dump(res, open("/home/sgeng/scottg/pretrain/lingua-clean/analysis/lexical_diversity.json", "w"), indent=1)
if __name__ == "__main__":
    main()
