"""Per-token RKL-gate inspection — what tokens does the idx 139 gate fire / skip on?

Same teacher (OLMo-2-1124-7B-Instruct), same Dolmino sources, same T=2.0 as the
idx 139 launch. But now we keep the decoded token string at every position so we
can sort by entropy and look at concrete examples of fired vs. unfired tokens.

The idx 139 gate is per-batch adaptive: tau = quantile(H, 0.30) over the batch.
We approximate that here with a single global tau computed over the full corpus
of sampled tokens (equal-source mix). The per-batch tau in training drifts with
batch composition, but the *which kinds of tokens land below tau* picture is what
this script answers.

Outputs:
  - per_token_with_strings.parquet  (source, category, token_str, prev_token_str, H, fired)
  - fired_vs_unfired_examples.json  (top/bottom-H exemplars per source)

Run:
    sbatch teacher_entropy_per_token.sbatch
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
    docs: list[str] = []
    with files[0].open(encoding="utf-8", errors="replace") as f:
        all_lines = []
        for i, line in enumerate(f):
            if i > 20000:
                break
            all_lines.append(line)
    rng.shuffle(all_lines)
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
def teacher_entropies_and_tokens(model, tokenizer, text: str, max_tokens: int, temperature: float):
    """Return (H[S], input_ids[S], token_strs[S]) for one document at temperature T.

    Per-position semantics: at position i, H[i] is the teacher's entropy over its
    distribution for predicting token i+1 given tokens [0..i]. The 'target' token
    string we tag at position i is the *next* token (input_ids[i+1]) — i.e., the
    token the gate would or wouldn't apply RKL to.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = enc["input_ids"].cuda()
    out = model(input_ids, use_cache=False)
    logits = out.logits.float().squeeze(0)              # (S, V)
    log_p = F.log_softmax(logits / temperature, dim=-1)
    p = log_p.exp()
    H = -(p * log_p).sum(dim=-1)                        # (S,)
    # Drop the last position (no next-token target) and align H[i] with token i+1.
    ids = input_ids.squeeze(0)
    H_aligned = H[:-1].cpu()
    target_ids = ids[1:].cpu()
    prev_ids = ids[:-1].cpu()
    # Decode each token individually so we keep BPE leading-space markers etc.
    target_strs = [tokenizer.decode([int(t)]) for t in target_ids]
    prev_strs   = [tokenizer.decode([int(t)]) for t in prev_ids]
    return H_aligned, target_ids, prev_ids, target_strs, prev_strs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-per-source", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=2.0,
                    help="Gate uses T=2.0 in idx 139.")
    ap.add_argument("--gate-quantile", type=float, default=0.30)
    ap.add_argument("--out-parquet", default=str(ANALYSIS_OUT_DIR / "per_token_with_strings.parquet"))
    ap.add_argument("--out-examples", default=str(ANALYSIS_OUT_DIR / "fired_vs_unfired_examples.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples-per-bucket", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)

    print(f"loading teacher {TEACHER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH, torch_dtype=torch.bfloat16
    ).cuda().eval()

    all_rows = []
    for src_name, meta in SOURCES.items():
        shard_dir = DOLMINO_ROOT / meta["dir"]
        print(f"\n=== {src_name} ({meta['category']}) — {shard_dir} ===")
        docs = sample_documents(shard_dir, args.docs_per_source, seed=args.seed)
        print(f"  {len(docs)} docs")
        for doc_idx, doc in enumerate(docs):
            H, tid, pid, tstr, pstr = teacher_entropies_and_tokens(
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

    # Per-source fire rate
    print("\n=== Per-source fire rate ===")
    print(f"{'source':16s}{'n_tok':>9s}{'mean H':>9s}{'fire %':>9s}")
    for src in df.source.unique():
        sub = df[df.source == src]
        print(f"{src:16s}{len(sub):>9d}{sub.entropy.mean():>9.3f}{100*sub.fired.mean():>8.1f}%")

    # Build exemplar JSON: lowest-H and highest-H tokens per source, plus a
    # short window of preceding context for readability.
    print("\nbuilding exemplars JSON ...")
    examples = {}
    K = args.examples_per_bucket
    for src in df.source.unique():
        sub = df[df.source == src].sort_values("entropy")
        # Lowest-H tokens: the ones the gate FIRES on (RKL applied)
        # Highest-H tokens: the ones the gate SKIPS (CE only)
        lowest = sub.head(K)
        highest = sub.tail(K)

        def to_record(row):
            return {
                "H": round(float(row.entropy), 4),
                "prev_token": row.prev_token_str,
                "target_token": row.target_token_str,
            }

        examples[src] = {
            "n_tokens": int(len(sub)),
            "mean_H": round(float(sub.entropy.mean()), 4),
            "fire_rate": round(float(sub.fired.mean()), 4),
            "fired_examples":    [to_record(r) for _, r in lowest.iterrows()],
            "unfired_examples":  [to_record(r) for _, r in highest.iterrows()],
        }
    payload = {
        "tau_global": round(float(tau), 4),
        "temperature": args.temperature,
        "gate_quantile": args.gate_quantile,
        "examples_per_bucket": K,
        "per_source": examples,
    }
    with open(args.out_examples, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote -> {args.out_examples}")


if __name__ == "__main__":
    main()
