"""Teacher-entropy by dolmino source — go/no-go diagnostic for entropy-gated RKL.

Loads OLMo-2-1124-7B-Instruct, samples ~5k tokens per dolmino source from local
shards, runs the teacher forward pass at T=1.0 and T=2.0, computes per-token
entropy, and reports per-source distributions.

Decision rule: if `H(p_T | factual) - H(p_T | reasoning) >= 0.5 nats` at T=2.0,
entropy-gated RKL has a chance of breaking the (NQ, gsm8k) Pareto curve. If
the separation is below noise, the mechanism is dead on arrival.

Run:
    srun -A comem --qos h200_dev --gres=gpu:1 --mem=200G --time=1:00:00 \
        python teacher_entropy_by_source.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DOLMINO_ROOT = Path(os.environ.get("DATA_ROOT", "./dolmino_splits"))
TEACHER_PATH = os.environ.get("TEACHER_7B_PATH", "./OLMo-2-1124-7B-Instruct")
ANALYSIS_OUT_DIR = Path(os.environ.get("ANALYSIS_OUT_DIR", "./analysis_outputs")) / "idx139"

# Match the dolmino mixin weights used in idx 132-135 (and 125)
SOURCES = {
    "dclm":           {"dir": "dclm_shuffled",          "category": "factual_web"},
    "wiki":           {"dir": "wiki_shuffled",          "category": "factual_curated"},
    "stackexchange":  {"dir": "stackexchange_shuffled", "category": "factual_curated"},
    "pes2o":          {"dir": "pes2o_shuffled",         "category": "factual_curated"},
    "math":           {"dir": "math_shuffled",          "category": "reasoning"},
    "flan":           {"dir": "flan_shuffled",          "category": "instruction"},
}


def sample_documents(shard_dir: Path, n_docs: int, seed: int = 0) -> list[str]:
    """Read up to n_docs text fields from the first jsonl shard in this dir."""
    files = sorted(shard_dir.glob("*.chunk.*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no chunk files in {shard_dir}")
    rng = random.Random(seed)
    docs: list[str] = []
    # Stream through the first few shards until we have enough docs (uniform random
    # over the first chunk file is good enough — shards are already shuffled).
    with files[0].open(encoding="utf-8", errors="replace") as f:
        all_lines = []
        for i, line in enumerate(f):
            if i > 20000:  # cap memory
                break
            all_lines.append(line)
    rng.shuffle(all_lines)
    for line in all_lines:
        if len(docs) >= n_docs:
            break
        try:
            d = json.loads(line)
            t = d.get("text", "")
            if isinstance(t, str) and len(t) > 200:  # skip empty / tiny
                docs.append(t)
        except json.JSONDecodeError:
            continue
    return docs


@torch.no_grad()
def teacher_entropies_for_text(
    model, tokenizer, text: str, max_tokens: int = 1024, temperatures=(1.0, 2.0)
) -> dict[float, torch.Tensor]:
    """Return per-token entropies under each temperature for one document."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = enc["input_ids"].cuda()
    out = model(input_ids, use_cache=False)
    logits = out.logits.float()  # (1, S, V)
    result = {}
    for T in temperatures:
        log_p = F.log_softmax(logits / T, dim=-1)
        p = log_p.exp()
        H = -(p * log_p).sum(dim=-1).squeeze(0)  # (S,)
        result[T] = H.cpu()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-per-source", type=int, default=40,
                    help="N docs to sample per source. ~40 docs × 1024 tokens = ~40k tokens/source.")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out-parquet", default=str(ANALYSIS_OUT_DIR / "teacher_entropy_by_source.parquet"))
    ap.add_argument("--out-summary", default=str(ANALYSIS_OUT_DIR / "teacher_entropy_summary.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)

    print(f"loading tokenizer + 7B-Instruct teacher from {TEACHER_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH, torch_dtype=torch.bfloat16
    ).cuda().eval()

    rows = []  # (source, category, temperature, entropy_nats)
    for src_name, meta in SOURCES.items():
        shard_dir = DOLMINO_ROOT / meta["dir"]
        print(f"\n=== {src_name} ({meta['category']}) — sampling from {shard_dir} ===")
        docs = sample_documents(shard_dir, args.docs_per_source, seed=args.seed)
        print(f"  got {len(docs)} docs, processing ...")
        n_tokens = 0
        for doc in docs:
            ents = teacher_entropies_for_text(model, tokenizer, doc, max_tokens=args.max_tokens)
            for T, H in ents.items():
                for h in H.tolist():
                    rows.append((src_name, meta["category"], float(T), float(h)))
                n_tokens += H.numel()
        print(f"  ~{n_tokens // 2} tokens processed (counted once per T)")

    # Dump rows as parquet
    try:
        import pandas as pd
        df = pd.DataFrame(rows, columns=["source", "category", "temperature", "entropy"])
        df.to_parquet(args.out_parquet)
        print(f"\nwrote {len(df):,} rows -> {args.out_parquet}")
    except ImportError:
        # Fallback to jsonl if pandas isn't around
        with open(args.out_parquet.replace(".parquet", ".jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(dict(zip(["source", "category", "temperature", "entropy"], r))) + "\n")
        print(f"\nwrote {len(rows):,} rows -> {args.out_parquet.replace('.parquet', '.jsonl')}")

    # Compute summary
    import statistics
    summary = {}
    for T in (1.0, 2.0):
        per_src = {}
        per_cat: dict[str, list[float]] = {}
        for src_name, meta in SOURCES.items():
            vals = [h for s, c, t, h in rows if s == src_name and t == T]
            if not vals:
                continue
            per_src[src_name] = {
                "n_tokens": len(vals),
                "mean":   statistics.mean(vals),
                "median": statistics.median(vals),
                "p10":    sorted(vals)[len(vals) // 10],
                "p90":    sorted(vals)[9 * len(vals) // 10],
            }
            per_cat.setdefault(meta["category"], []).extend(vals)
        per_cat_summary = {
            cat: {
                "n_tokens": len(vs),
                "mean":     statistics.mean(vs),
                "median":   statistics.median(vs),
            }
            for cat, vs in per_cat.items()
        }
        summary[f"T={T}"] = {
            "per_source":   per_src,
            "per_category": per_cat_summary,
        }
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote summary -> {args.out_summary}")

    # Decision printout: at T=2.0, factual vs reasoning gap
    print("\n=== GO/NO-GO at T=2.0 ===")
    fac_means, reason_means = [], []
    for src_name, meta in SOURCES.items():
        s = summary["T=2.0"]["per_source"].get(src_name)
        if not s:
            continue
        line = f"  {src_name:15s} cat={meta['category']:18s} mean={s['mean']:.3f}  median={s['median']:.3f}  p10={s['p10']:.3f}  p90={s['p90']:.3f}"
        print(line)
        if meta["category"].startswith("factual"):
            fac_means.append(s["mean"])
        elif meta["category"] == "reasoning":
            reason_means.append(s["mean"])

    if fac_means and reason_means:
        gap = statistics.mean(fac_means) - statistics.mean(reason_means)
        print(f"\n  mean(factual) - mean(reasoning) = {gap:+.3f} nats")
        if gap >= 0.5:
            print(f"  >>> GO: gap >= 0.5 nats. Entropy-gated RKL is worth running.")
        elif gap <= -0.5:
            print(f"  >>> INVERTED: reasoning has higher entropy than factual. Gate logic flips.")
        else:
            print(f"  >>> NO-GO: |gap| < 0.5 nats. Entropy doesn't separate the regimes.")


if __name__ == "__main__":
    main()
