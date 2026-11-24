"""Final-step markdown table for the w1152 multi-sample study (setup/make_multisample_configs.py).

    python3 analysis/results_viewer/multisample_table.py
"""
import json
import os

ROOT = "/home/sgeng/scottg/pretrain/lingua-runs/runs/midtrain/experiments/multisample_w1152"
RUNS = ["dclm_2x", "syn_direct_nothink", "syn_direct_think", "syn_mix", "syn_ideas"]
STEP = 16398
OLMES = [  # (column, metric key, higher is better)
    ("ARC-E", "arc_easy:rc::olmes:full", True), ("ARC-C", "arc_challenge:rc::olmes:full", True),
    ("HSwag", "hellaswag:rc::olmes:full", True), ("PIQA", "piqa:rc::olmes:full", True),
    ("WinoG", "winogrande:rc::olmes:full", True), ("SciQ", "sciq:rc::olmo3", True),
    ("MMLU", "mmlu:rc::olmes", True), ("MMLU bpb", "mmlu:rc:bpb::olmes", False),
    ("GSM8K p@32", "gsm8k::olmo3:n32:pass32", True), ("Math bpb", "olmo3:base_easy:math_bpb", False),
    ("HumanEval bpb", "codex_humaneval:3shot:bpb::none", False), ("MBPP bpb", "mbpp:3shot:bpb::none", False),
]


def row(run):
    d = f"{ROOT}/{run}"
    out = {}
    v = f"{d}/evals/{STEP:010d}/validation.json"
    if os.path.exists(v):
        val = json.load(open(v))
        out["DCLM bpb"] = val.get("dclm_heldout", {}).get("bpb")
        out["Paloma bpb"] = val.get("paloma_qwen3/mean", {}).get("bpb")
        own = [k for k in val if k.startswith("ms_") and k.endswith("_heldout")]
        out["own syn bpb"] = val[own[0]]["bpb"] if own else None
    m = f"{d}/metrics.olmes_eval.jsonl"
    if os.path.exists(m):
        rows = [json.loads(line) for line in open(m)]
        last = [r for r in rows if r.get("global_step") == STEP]
        if last:
            for col, key, _ in OLMES:
                out[col] = last[-1].get(f"olmes_eval/{key}/primary_score")
    return out


def main():
    cols = ["DCLM bpb", "Paloma bpb", "own syn bpb"] + [c for c, _, _ in OLMES]
    print("| run | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for run in RUNS:
        r = row(run)
        print(f"| {run} | " + " | ".join("" if r.get(c) is None else f"{r[c]:.4f}" for c in cols) + " |")
    print("\nbpb: lower is better; other OLMES columns: primary_score, higher is better.")


if __name__ == "__main__":
    main()
