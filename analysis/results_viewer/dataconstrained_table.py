"""Results table for the data-constrained ladder: one row per (run, evaluation point).

Columns follow the sweep doc: width, stage, learning_rate, batch_sequences, weight_decay, embedding_decay, eval_point,
tokens_seen, steps, val_ce (dclm_heldout, nats/token, token-weighted), val_bpb, train_loss (mean over the last 1% of
steps before the evaluation point), skipped_steps (bad-step guard), seed, plus paloma_bpb (mean of 16 sources, no
EOT prefix). Rows come from the manifest (apps/main/configs/experiments/qwen3_ladder_dataconstrained/manifest.json);
an evaluation point without results yet has empty metric columns.

    .venv-kd/bin/python analysis/results_viewer/dataconstrained_table.py [--select '^w512_s1_'] [--csv out.csv]
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GROUP = "qwen3_ladder_dataconstrained"
MANIFEST = f"{ROOT}/apps/main/configs/experiments/{GROUP}/manifest.json"
RUNS = f"/checkpoint/mnm/sgeng/pretrain/lingua-runs/runs/midtrain/experiments/{GROUP}"
LOGS = "/checkpoint/mnm/sgeng/pretrain/lingua-runs/runs/slurm_logs"
COLS = ["run", "width", "stage", "learning_rate", "batch_sequences", "weight_decay", "embedding_decay", "eval_point",
        "tokens_seen", "steps", "val_ce", "val_bpb", "paloma_bpb", "train_loss", "skipped_steps", "seed"]


def record(path, step):
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r.get("global_step") == step:
                return r
    return None


def train_loss(run_dir, step, main_dir=None):
    """Mean training loss over the last 1% of steps up to `step` (from metrics.jsonl; a branch also reads its main
    run's log for steps before the branch point)."""
    lo = step - max(1, round(0.01 * step))
    vals = {}
    for d in [main_dir, run_dir]:
        if d and os.path.exists(f"{d}/metrics.jsonl"):
            for line in open(f"{d}/metrics.jsonl"):
                r = json.loads(line)
                s = r.get("global_step")
                if s is not None and lo < s <= step and "loss/out" in r:
                    vals[s] = r["loss/out"]
    return sum(vals.values()) / len(vals) if vals else None


def skipped(name):
    n = 0
    for f in glob.glob(f"{LOGS}/lingua_clean_pt_experiments_{GROUP}_{name}-*.out"):
        n += sum("[NaN-guard]" in line for line in open(f, errors="ignore"))
    return n


def rows(select):
    manifest = json.load(open(MANIFEST))
    out = []
    for name, e in sorted(manifest.items()):
        if not re.search(select, name):
            continue
        d = f"{RUNS}/{name}"
        v = record(f"{d}/metrics.validation.jsonl", e["steps"]) or {}
        p = record(f"{d}/metrics.paloma_np.jsonl", e["steps"]) or {}
        pal = [x for k, x in p.items() if k.endswith("::paloma/primary_score") and "/paloma::" not in k]
        h = v.get("dclm_heldout", {})
        out.append({
            "run": name, "width": e["width"], "stage": e["stage"], "learning_rate": e["lr"],
            "batch_sequences": e["bs"], "weight_decay": e["wd"], "embedding_decay": "off",
            "eval_point": e["eval_point"], "tokens_seen": e["tokens"], "steps": e["steps"],
            "val_ce": h.get("ce_per_token"), "val_bpb": h.get("bpb"),
            "paloma_bpb": sum(pal) / len(pal) if len(pal) == 16 else None,
            "train_loss": train_loss(d, e["steps"], f"{RUNS}/{e['main']}" if e["main"] != name else None),
            "skipped_steps": skipped(name), "seed": e["seed"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", default=".")
    ap.add_argument("--csv")
    a = ap.parse_args()
    rs = rows(a.select)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, COLS)
            w.writeheader()
            w.writerows(rs)
    fmt = lambda x: "" if x is None else (f"{x:.4f}" if isinstance(x, float) and x < 100 else str(x))
    print("| " + " | ".join(COLS) + " |")
    print("|" + "---|" * len(COLS))
    for r in rs:
        print("| " + " | ".join(fmt(r[c]) for c in COLS) + " |")


if __name__ == "__main__":
    main()
