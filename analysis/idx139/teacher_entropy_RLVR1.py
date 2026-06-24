"""Pre-run diagnostic for idx 145: per-token teacher entropy with the 1B-RLVR1
teacher (instead of idx 139's 7B-Instruct teacher).

Question this answers: does the entropy-gate intuition (low-teacher-entropy
tokens are math/structural, high-teacher-entropy tokens are factual/web)
generalize to a *small same-family* teacher? Or is the 7B-Instruct sharpness
on math a function of teacher *scale* rather than the recipe?

Outputs:
  - per_token_RLVR1_teacher_summary.json  (mean H per source, gate fire rate)
  - per_token_RLVR1_teacher.parquet       (full per-token table)
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
TEACHER_PATH = os.environ.get("TEACHER_1B_RLVR1_PATH", "./OLMo-2-0425-1B-RLVR1")
ANALYSIS_OUT_DIR = Path(os.environ.get("ANALYSIS_OUT_DIR", "./analysis_outputs")) / "idx139"

SOURCES = {
    "dclm":           {"dir": "dclm_shuffled",          "category": "factual_web"},
    "wiki":           {"dir": "wiki_shuffled",          "category": "factual_curated"},
    "stackexchange":  {"dir": "stackexchange_shuffled", "category": "factual_curated"},
    "pes2o":          {"dir": "pes2o_shuffled",         "category": "factual_curated"},
    "math":           {"dir": "math_shuffled",          "category": "reasoning"},
    "flan":           {"dir": "flan_shuffled",          "category": "instruction"},
}


def sample_documents(shard_dir: Path, n_docs: int, seed: int = 0) -> list[str]:
    files = sorted(shard_dir.glob("*.chunk.*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no chunk files in {shard_dir}")
    rng = random.Random(seed)
    with files[0].open(encoding="utf-8", errors="replace") as f:
        all_lines = []
        for i, line in enumerate(f):
            if i > 20000:
                break
            all_lines.append(line)
    rng.shuffle(all_lines)
    docs: list[str] = []
    for line in all_lines:
        if len(docs) >= n_docs:
            break
        try:
            d = json.loads(line)
            t = d.get("text", "")
            if isinstance(t, str) and len(t) > 200:
                docs.append(t)
        except json.JSONDecodeError:
            continue
    return docs


@torch.no_grad()
def teacher_entropies_for_doc(model, tokenizer, text, max_tokens, temperature):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = enc["input_ids"].cuda()
    out = model(input_ids, use_cache=False)
    logits = out.logits.float().squeeze(0)
    log_p = F.log_softmax(logits / temperature, dim=-1)
    H = -(log_p.exp() * log_p).sum(dim=-1)
    ids = input_ids.squeeze(0)
    H_a = H[:-1].cpu()
    target_ids = ids[1:].cpu()
    prev_ids = ids[:-1].cpu()
    target_strs = [tokenizer.decode([int(t)]) for t in target_ids]
    prev_strs = [tokenizer.decode([int(t)]) for t in prev_ids]
    return H_a, target_ids, prev_ids, target_strs, prev_strs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-per-source", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--gate-quantile", type=float, default=0.30)
    ap.add_argument("--out-parquet", default=str(ANALYSIS_OUT_DIR / "per_token_RLVR1_teacher.parquet"))
    ap.add_argument("--out-summary", default=str(ANALYSIS_OUT_DIR / "per_token_RLVR1_teacher_summary.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)

    print(f"loading teacher {TEACHER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH, torch_dtype=torch.bfloat16
    ).cuda().eval()
    print(f"  vocab: {model.config.vocab_size}")

    all_rows = []
    for src_name, meta in SOURCES.items():
        shard_dir = DOLMINO_ROOT / meta["dir"]
        print(f"\n=== {src_name} ({meta['category']}) ===")
        docs = sample_documents(shard_dir, args.docs_per_source, seed=args.seed)
        print(f"  {len(docs)} docs")
        for doc_idx, doc in enumerate(docs):
            H, tid, pid, tstr, pstr = teacher_entropies_for_doc(
                model, tokenizer, doc, max_tokens=args.max_tokens, temperature=args.temperature
            )
            for j in range(len(H)):
                all_rows.append((
                    src_name, meta["category"], doc_idx,
                    int(tid[j]), int(pid[j]),
                    tstr[j], pstr[j],
                    float(H[j]),
                ))
        print(f"  cumulative rows: {len(all_rows):,}")

    import pandas as pd
    df = pd.DataFrame(all_rows, columns=[
        "source", "category", "doc_idx",
        "target_token_id", "prev_token_id",
        "target_token_str", "prev_token_str",
        "entropy",
    ])
    tau = df.entropy.quantile(args.gate_quantile)
    df["fired"] = df.entropy <= tau
    df.to_parquet(args.out_parquet)
    print(f"\nwrote {len(df):,} rows -> {args.out_parquet}")
    print(f"global tau @ q={args.gate_quantile}: {tau:.3f} nats")
    print(f"global fire rate: {100*df.fired.mean():.2f}%")

    print("\n=== Per-source fire rate ===")
    print(f"{'source':16s}{'n_tok':>9s}{'mean H':>9s}{'fire %':>9s}")
    per_source = {}
    for src in df.source.unique():
        sub = df[df.source == src]
        mean_h = float(sub.entropy.mean())
        frate = float(sub.fired.mean())
        print(f"{src:16s}{len(sub):>9d}{mean_h:>9.3f}{100*frate:>8.1f}%")
        per_source[src] = {
            "n": int(len(sub)),
            "mean_H": round(mean_h, 4),
            "fire_rate": round(frate, 4),
            "median_H": round(float(sub.entropy.median()), 4),
        }

    summary = {
        "teacher": TEACHER_PATH,
        "temperature": args.temperature,
        "gate_quantile": args.gate_quantile,
        "n_total_tokens": int(len(df)),
        "tau_global": round(float(tau), 4),
        "mean_H_overall": round(float(df.entropy.mean()), 4),
        "per_source": per_source,
    }
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote -> {args.out_summary}")


if __name__ == "__main__":
    main()
