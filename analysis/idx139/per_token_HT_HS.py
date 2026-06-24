"""Predict idx 143 behavior by computing per-token (H_T, H_S) correlation.

We compare:
  - 139's gate: top-30% lowest teacher entropy (most-confident teacher)
  - 143's gate: top-30% highest STUDENT entropy (most-uncertain student)

The key question: how much do these two sets overlap? If high overlap, 143 ≈ 139.
If low overlap, 143 fires on a very different population — and we can read off
*what kinds of tokens* it fires on (factual? math? boilerplate?).

Uses the SAME sources as analysis/idx139/teacher_entropy_per_token.py.
Student = 1B init (OLMo-2-0425-1B-stage1-4001B), which the 139 telemetry shows
is a good proxy for the trained student's entropy pattern (mean H_S barely drifts
across training: 7.10 -> 7.10 across all quartiles).

Outputs:
  - per_token_HT_HS.parquet   (source, category, doc_idx, target_token_str,
                              prev_token_str, H_T, H_S)
  - HT_HS_summary.json        (correlation, gate overlap, per-source breakdown)
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
STUDENT_PATH = os.environ.get("STUDENT_HF_PATH", "./OLMo-2-0425-1B-stage1-4001B/hf")
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
def entropies_for_doc(teacher, student, tokenizer, text, max_tokens, temperature):
    """Return (H_T[S], H_S[S], input_ids, token_strs, prev_strs) — same alignment as 139."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = enc["input_ids"].cuda()

    out_t = teacher(input_ids, use_cache=False)
    lT = out_t.logits.float().squeeze(0)
    log_pT = F.log_softmax(lT / temperature, dim=-1)
    H_T = -(log_pT.exp() * log_pT).sum(dim=-1)

    out_s = student(input_ids, use_cache=False)
    lS = out_s.logits.float().squeeze(0)
    log_pS = F.log_softmax(lS / temperature, dim=-1)
    H_S = -(log_pS.exp() * log_pS).sum(dim=-1)

    ids = input_ids.squeeze(0)
    H_T_a = H_T[:-1].cpu()
    H_S_a = H_S[:-1].cpu()
    target_ids = ids[1:].cpu()
    prev_ids = ids[:-1].cpu()
    target_strs = [tokenizer.decode([int(t)]) for t in target_ids]
    prev_strs   = [tokenizer.decode([int(t)]) for t in prev_ids]
    return H_T_a, H_S_a, target_ids, prev_ids, target_strs, prev_strs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-per-source", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--gate-quantile", type=float, default=0.30)
    ap.add_argument("--out-parquet", default=str(ANALYSIS_OUT_DIR / "per_token_HT_HS.parquet"))
    ap.add_argument("--out-summary", default=str(ANALYSIS_OUT_DIR / "HT_HS_summary.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)

    print(f"loading teacher {TEACHER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH)
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH, torch_dtype=torch.bfloat16
    ).cuda().eval()

    print(f"loading student {STUDENT_PATH}")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_PATH, torch_dtype=torch.bfloat16
    ).cuda().eval()

    # Sanity: vocab sizes
    print(f"teacher vocab: {teacher.config.vocab_size}, student vocab: {student.config.vocab_size}")

    all_rows = []
    for src_name, meta in SOURCES.items():
        shard_dir = DOLMINO_ROOT / meta["dir"]
        print(f"\n=== {src_name} ({meta['category']}) — {shard_dir} ===")
        docs = sample_documents(shard_dir, args.docs_per_source, seed=args.seed)
        print(f"  {len(docs)} docs")
        for doc_idx, doc in enumerate(docs):
            HT, HS, tid, pid, tstr, pstr = entropies_for_doc(
                teacher, student, tokenizer, doc,
                max_tokens=args.max_tokens, temperature=args.temperature
            )
            for j in range(len(HT)):
                all_rows.append((
                    src_name, meta["category"], doc_idx,
                    int(tid[j]), int(pid[j]),
                    tstr[j], pstr[j],
                    float(HT[j]), float(HS[j]),
                ))
        print(f"  cumulative rows: {len(all_rows):,}")

    import pandas as pd
    df = pd.DataFrame(all_rows, columns=[
        "source", "category", "doc_idx",
        "target_token_id", "prev_token_id",
        "target_token_str", "prev_token_str",
        "H_T", "H_S",
    ])
    df.to_parquet(args.out_parquet)
    print(f"\nwrote {len(df):,} rows -> {args.out_parquet}")

    # === Compute the gate-overlap analysis ===
    q = args.gate_quantile
    tau_T = df.H_T.quantile(q)          # bottom-q teacher (139 gate)
    tau_S = df.H_S.quantile(1 - q)      # top-q student   (143 gate)
    df["fires_139"] = df.H_T <= tau_T
    df["fires_143"] = df.H_S >= tau_S
    df["fires_both"] = df.fires_139 & df.fires_143

    n_total = len(df)
    n_139 = int(df.fires_139.sum())
    n_143 = int(df.fires_143.sum())
    n_both = int(df.fires_both.sum())
    overlap = n_both / max(n_139, 1)   # fraction of 139's fired set that 143 also fires on
    jaccard = n_both / max(n_139 + n_143 - n_both, 1)

    pearson = float(df[["H_T", "H_S"]].corr().iloc[0, 1])
    try:
        from scipy.stats import spearmanr
        spear = float(spearmanr(df.H_T, df.H_S).statistic)
    except Exception:
        spear = float("nan")

    print(f"\n=== Gate overlap @ q={q} ===")
    print(f"  tau_T (139's bottom-{int(100*q)}% H_T): {tau_T:.3f} nats")
    print(f"  tau_S (143's top-{int(100*q)}% H_S):    {tau_S:.3f} nats")
    print(f"  fires_139:   {n_139:>8,} tokens  ({100*n_139/n_total:.1f}%)")
    print(f"  fires_143:   {n_143:>8,} tokens  ({100*n_143/n_total:.1f}%)")
    print(f"  fires_both:  {n_both:>8,} tokens  ({100*n_both/n_total:.1f}%)")
    print(f"  overlap (143-fires within 139-fires set): {100*overlap:.1f}%")
    print(f"  Jaccard:     {jaccard:.3f}")
    print(f"  per-token Pearson r(H_T, H_S): {pearson:.3f}")
    print(f"  per-token Spearman r(H_T, H_S): {spear:.3f}")

    print(f"\n=== Per-source fire profiles ===")
    print(f"{'source':16s}{'tokens':>8s}{'%fired139':>11s}{'%fired143':>11s}{'%both':>8s}")
    per_source = {}
    for src in df.source.unique():
        sub = df[df.source == src]
        f139 = float(sub.fires_139.mean())
        f143 = float(sub.fires_143.mean())
        fb = float(sub.fires_both.mean())
        print(f"{src:16s}{len(sub):>8d}{100*f139:>10.1f}%{100*f143:>10.1f}%{100*fb:>7.1f}%")
        per_source[src] = {
            "n": int(len(sub)),
            "mean_H_T": round(float(sub.H_T.mean()), 4),
            "mean_H_S": round(float(sub.H_S.mean()), 4),
            "fire_rate_139": round(f139, 4),
            "fire_rate_143": round(f143, 4),
            "fire_rate_both": round(fb, 4),
        }

    summary = {
        "temperature": args.temperature,
        "gate_quantile": q,
        "n_total_tokens": int(n_total),
        "tau_T": round(float(tau_T), 4),
        "tau_S": round(float(tau_S), 4),
        "fire_rate_139": round(n_139 / n_total, 4),
        "fire_rate_143": round(n_143 / n_total, 4),
        "fire_rate_both": round(n_both / n_total, 4),
        "overlap_143_within_139": round(overlap, 4),
        "jaccard_139_143": round(jaccard, 4),
        "pearson_HT_HS": round(pearson, 4),
        "spearman_HT_HS": round(spear, 4),
        "per_source": per_source,
        "mean_H_T_overall": round(float(df.H_T.mean()), 4),
        "mean_H_S_overall": round(float(df.H_S.mean()), 4),
    }
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote -> {args.out_summary}")


if __name__ == "__main__":
    main()
