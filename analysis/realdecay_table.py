"""Anneal-with-real results (multisample_dc_w1152, 3C = step 24588): each base run, its same-trunk re-decay control,
and the real-only decay branch. avg7 = mean of the 7 OLMES accuracies (ARC-E/C, HellaSwag, PIQA, Winogrande, SciQ,
MMLU); bpb columns: DCLM heldout, Paloma mean, mix heldout; GSM8K pass@32, MMLU / math bpb (OLMES).

    python3 analysis/realdecay_table.py
"""
import json
import os

R = "/home/sgeng/scottg/pretrain/lingua-runs/runs/midtrain/experiments/multisample_dc_w1152"
ACC = ["arc_easy:rc::olmes:full", "arc_challenge:rc::olmes:full", "hellaswag:rc::olmes:full", "piqa:rc::olmes:full",
       "winogrande:rc::olmes:full", "sciq:rc::olmo3", "mmlu:rc::olmes"]
GROUPS = [["real_p12", "fresh_3x"]] + [[b, f"{b}_redecay", f"{b}_realdecay"] + ([f"{b}_realdecay_kd"] if "kd" in b else []) for b in
          ["r4s_mix_3x", "r4s_arv2_3x", "r4s_direct_nothink_3x", "r4s_mix_3x_kd8b_a1", "r4s_direct_nothink_3x_kd8b_a1"]] + \
         [[b, f"{b}_realdecay"] for b in ["real_p12_kd8b_a1", "real_p12_kd4b_a0p9"]]


def last(path, step=24588):
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get("step", step) == step] or rows
    return rows[-1]


def row(run):
    o, v = last(f"{R}/{run}/metrics.olmes_eval.jsonl"), last(f"{R}/{run}/metrics.validation.jsonl")
    if not o or not v:
        return None
    acc = [o[f"olmes_eval/{t}/primary_score"] for t in ACC]
    val = lambda k: next((x["bpb"] for kk, x in v.items() if isinstance(x, dict) and kk.startswith(k)), float("nan"))
    return [100 * sum(acc) / 7] + [100 * a for a in acc] + [100 * o["olmes_eval/gsm8k::olmo3:n32:pass32/primary_score"],
            o["olmes_eval/mmlu:rc:bpb::olmes/primary_score"], o["olmes_eval/olmo3:base_easy:math_bpb/primary_score"],
            val("dclm_heldout"), val("paloma_qwen3/mean"), val("ms_mix_heldout")]


print(f"{'run':38s} {'avg7':>5s} {'ARC-E':>5s} {'ARC-C':>5s} {'HSwag':>5s} {'PIQA':>5s} {'Wino':>5s} {'SciQ':>5s} {'MMLU':>5s} "
      f"{'GSM32':>5s} {'MMLUb':>6s} {'mathb':>6s} {'DCLMb':>6s} {'Palom':>6s} {'mixb':>6s}")
for g in GROUPS:
    for run in g:
        r = row(run)
        print(f"{run:38s} " + ("  (not finished)" if r is None else
              " ".join(f"{x:5.1f}" for x in r[:9]) + " " + " ".join(f"{x:6.3f}" for x in r[9:])))
    print()
