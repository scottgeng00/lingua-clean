"""Final-checkpoint markdown table of the OLMES lingua:v2 suite for runs under a common parent dir.

    .venv-kd/bin/python analysis/results_viewer/v2_table.py pt_dclm8m0_lr1p3e3 pt_dclm8m0_lr2p65e3 ...
    .venv-kd/bin/python analysis/results_viewer/v2_table.py --root <parent dir> run_a run_b

Reads <root>/<run>/evals/<last step>*/olmes_results/metrics.json (the shard merges written by
apps/main/eval_olmes.py). Columns: mean primary score of the QA rank-classification tasks (accuracy),
mean gold-answer BPB of the QA / Minerva-math / code BPB tasks (lower is better), mean Paloma BPB
(when present) and gsm8k pass@1 / pass@32. Suite aggregates (tasks whose alias has no subtasks of
its own, e.g. arc:rc::olmes:full) are used as reported by OLMES; their subtasks are skipped.
"""
import argparse
import glob
import json
import os

ROOT = "/checkpoint/mnm/sgeng/pretrain/lingua-runs/runs/midtrain/experiments/gb300"


def load(run_dir):
    # evals/<step>/ plus any reruns of a task subset for the same step (e.g. evals/<step>_paloma/).
    evals = glob.glob(f"{run_dir}/evals/[0-9]*/olmes_results/metrics.json")
    if not evals:
        return None, {}
    step_of = lambda p: int(p.split("/evals/")[1].split("/")[0].split("_")[0])
    step = max(step_of(p) for p in evals)
    tasks = {}
    for path in sorted(p for p in evals if step_of(p) == step):
        tasks.update({t["alias"]: t["metrics"] for t in json.load(open(path))["tasks"]})
    return step, tasks


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    print("| run | step | QA RC acc (n) | QA BPB | math BPB | code BPB | Paloma BPB (n) | GSM8K pass@1 | GSM8K pass@32 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for run in a.runs:
        step, t = load(os.path.join(a.root, run))
        if step is None:
            print(f"| {run} | — | no eval yet |||||||")
            continue
        # Top-level tasks only: drop subtasks whose parent suite is also present (e.g. mmlu_*:rc).
        # mt_mbpp_v2fix names its per-language subtasks mt_mbpp_v2fix:<lang>.
        top = {k: v for k, v in t.items()
               if not k.startswith("mt_mbpp_v2fix:")
               and not any(k != p and k.split(":")[0].startswith(p.split(":")[0] + "_") and k.split("::")[-1] == p.split("::")[-1]
                           for p in t)}
        # arc:rc::olmes:full is reported only as arc_easy + arc_challenge; count it as one task (their mean).
        for suffix in (":rc::olmes:full", ":rc:bpb::olmes:full"):
            parts = [top.pop(f"arc_{s}{suffix}", None) for s in ("easy", "challenge")]
            if None not in parts:
                top[f"arc{suffix}"] = {"primary_score": mean([p.get("primary_score") for p in parts])}
        rc = [v.get("primary_score") for k, v in top.items() if ":rc" in k and ":bpb" not in k] + \
             [v.get("primary_score") for k, v in top.items() if k.split(":")[0] in ("lab_bench_dbqa", "lab_bench_protocolqa", "lambada") and ":bpb" not in k]
        qa_bpb = [v.get("primary_score") for k, v in top.items() if ":bpb" in k and not k.startswith(("minerva", "codex", "mbpp", "mt_mbpp"))]
        math_bpb = [v.get("primary_score") for k, v in top.items() if k.startswith("minerva_math")]
        code_bpb = [v.get("primary_score") for k, v in top.items() if k.startswith(("codex_humaneval", "mbpp", "mt_mbpp"))]
        paloma = [v.get("primary_score") for k, v in top.items() if k.endswith("::paloma")]
        g = t.get("gsm8k::olmo3:n32:pass32", {})
        f = lambda x, s=1, n=3: "—" if x is None else f"{x * s:.{n}f}"
        print(f"| {run} | {step} | {f(mean(rc), 100, 1)} ({len(rc)}) | {f(mean(qa_bpb))} | {f(mean(math_bpb))} | {f(mean(code_bpb))} | "
              f"{f(mean(paloma))} ({len(paloma)}) | {f(g.get('pass_at_1'), 100, 1)} | {f(g.get('pass_at_32'), 100, 1)} |")


if __name__ == "__main__":
    main()
