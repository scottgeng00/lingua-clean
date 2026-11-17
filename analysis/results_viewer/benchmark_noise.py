"""Sampling noise of each OLMES benchmark from per-item predictions (bootstrap over items), vs the size of real effects.
For each benchmark: number of items, standard error of one model's score, paired standard error of the difference
between two models scored on the same items, and signal-to-noise = |effect| / paired SE for two reference effects:
doubling data (w1152 2x -> 4x Chinchilla) and KD (alpha 0.5 vs CE, w1152 2x). Item-sampling noise only (no seed noise).
MMLU and math are macro averages over subjects, bootstrapped within subject.

    .venv-kd/bin/python analysis/results_viewer/benchmark_noise.py
"""
import glob
import json
from collections import defaultdict

import numpy as np

X = "/checkpoint/mnm/sgeng/pretrain/lingua-runs/runs/midtrain/experiments"
MODELS = {"2x": f"{X}/data_constrainedv2/w1152_u_bs256_lr2e-3_wd0p1_d2x",
          "4x": f"{X}/data_constrainedv2/w1152_u_bs256_lr2e-3_wd0p1",
          "ce": f"{X}/kd_w1152/b_ce", "kd": f"{X}/kd_w1152/b_kd0p5",
          "w512": f"{X}/data_constrainedv2/w512_u_bs128_lr8e-3_wd0p01_d1x",
          "w1536": f"{X}/data_constrainedv2/w1536_u_bs512_lr2e-3_wd0p1"}
BENCH = [("ARC-E", "arc_easy:rc::olmes:full", None, 100), ("ARC-C", "arc_challenge:rc::olmes:full", None, 100),
         ("HellaSwag", "hellaswag:rc::olmes:full", None, 100), ("PIQA", "piqa:rc::olmes:full", None, 100),
         ("Winogrande", "winogrande:rc::olmes:full", None, 100), ("SciQ", "sciq:rc::olmo3", None, 100),
         ("MMLU acc", ":rc::olmes", "mmlu_", 100), ("MMLU BPB", ":rc:bpb::olmes", "mmlu_", 1),
         ("math BPB", ":bpb::olmes", "minerva_math_", 1), ("HumanEval BPB", "codex_humaneval:3shot:bpb::none", None, 1),
         ("MBPP BPB", "mbpp:3shot:bpb::none", None, 1), ("GSM8K pass@1", "gsm8k::olmo3:n32:pass32", "p1", 100),
         ("GSM8K pass@32", "gsm8k::olmo3:n32:pass32", None, 100)]


def items(model_dir):
    """alias -> {doc_id: primary value}"""
    out = defaultdict(dict)
    for m in glob.glob(f"{model_dir}/evals/*/olmes_results/shard_*/task-*-metrics.json"):
        j = json.load(open(m))
        alias, pm = j["task_config"]["metadata"]["alias"], j["task_config"]["primary_metric"]
        for line in open(m.replace("-metrics.json", "-predictions.jsonl")):
            p = json.loads(line)
            out[alias][p["doc_id"]] = p["metrics"]
        out[alias]["__pm__"] = pm
    return out


def groups(it, alias, prefix, key=None):
    """list of (subject, {doc: value}) making up the benchmark"""
    if prefix in (None, "p1"):
        sel = [alias]
    else:
        sel = [a for a in it if a.startswith(prefix) and a.endswith(alias) and a.count(":") == alias.count(":")]
    res = []
    for a in sorted(sel):
        pm = "pass_at_1" if prefix == "p1" else it[a]["__pm__"]
        res.append((a, {d: v[pm] for d, v in it[a].items() if d != "__pm__"}))
    return res


def boot(ga, gb=None, n=2000, seed=0):
    """bootstrap SE of the macro mean (or of the paired difference if gb is given)"""
    rng = np.random.default_rng(seed)
    subj = []
    for (_, a), b in zip(ga, gb or [None] * len(ga)):
        docs = sorted(a)
        va = np.array([a[d] for d in docs], dtype=float)
        vb = np.array([b[1][d] for d in docs], dtype=float) if b else 0.0
        subj.append(va - vb)
    est = np.mean([s.mean() for s in subj])
    reps = [np.mean([s[rng.integers(0, len(s), len(s))].mean() for s in subj]) for _ in range(n)]
    return est, np.std(reps)


if __name__ == "__main__":
    it = {k: items(v) for k, v in MODELS.items()}
    print("| benchmark | items | score (w1152 2×) | SE one model | paired SE of a difference | Δ 2×→4× data | SNR | "
          "Δ KD α0.5−CE | SNR | range w512 1× → w1536 4× | range / paired SE |\n|" + "---|" * 11)
    for name, alias, prefix, scale in BENCH:
        g = {k: groups(it[k], alias, prefix) for k in it}
        nitems = sum(len(x[1]) for x in g["2x"])
        s, se = boot(g["2x"])
        d_data, pse = boot(g["4x"], g["2x"])
        d_kd, pse_kd = boot(g["kd"], g["ce"])
        rng_, pse_r = boot(g["w1536"], g["w512"])
        f = lambda v: f"{v * scale:+.2f}" if scale == 100 else f"{v:+.4f}"
        f0 = lambda v: f"{v * scale:.2f}" if scale == 100 else f"{v:.4f}"
        print(f"| {name} | {nitems:,} | {f0(s)} | {f0(se)} | {f0(pse)} | {f(d_data)} | {abs(d_data) / pse:.1f} | "
              f"{f(d_kd)} | {abs(d_kd) / pse_kd:.1f} | {f(rng_)} | {abs(rng_) / pse_r:.0f} |", flush=True)
