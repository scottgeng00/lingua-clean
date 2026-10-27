"""Eval v2 for existing checkpoints: Paloma + OLMo-3 base_easy (OLMES), in a separate output tree.

Writes <run_dir>/evals_v2/<step>/{base_config.yaml, submit.slurm} from the run's existing
evals/<step>/base_config.yaml, with
  - olmes.tasks = the members of the OLMES suite lingua:v2 (olmes fork, branch lm-eval-0.4.13): OLMo-3
    base_easy qa_rc / qa_bpb (minus SocialIQA) / math_bpb / code_bpb, the 16 Paloma sources, and
    gsm8k::olmo3:n32:pass32, listed individually so eval_olmes can spread them over the GPUs;
  - validation off (Paloma replaces the lingua dclm_heldout / Dolmino val sets);
  - metric_log_dir = <run_dir>/evals_v2, so metrics land in evals_v2/metrics.olmes_eval.jsonl and
    never mix with the v1 metrics; wandb off.
Datasets must be in the HF cache (evals run offline); see lingua-runs/analysis/eval_v2/download.py.
Dry run by default.

    .venv/bin/python setup/submit_eval_v2.py mr_iid_k2_lr5p3e3_wd01 mr_nemo_k2_lr5p3e3_wd01 --submit
"""

import argparse
import os
import re
import subprocess

import yaml

GENSCALE = "/checkpoint/dream-sc-3/scottg/pretrain/lingua-runs/runs/midtrain/experiments/genscale"
OLMES_ROOT = "/home/sgeng/scottg/pretrain/olmes"
SUITE = "lingua:v2"


def suite_members():
    """Members of SUITE from the OLMES fork's task_suites.py, with paloma::paloma expanded to its sources."""
    out = subprocess.run(
        [f"{OLMES_ROOT}/.venv-lmeval/bin/python", "-W", "ignore", "-c",
         "import json,sys; from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS as S; "
         "t = S[sys.argv[1]]['tasks']; "
         "print(json.dumps([x for m in t for x in (S[m]['tasks'] if m == 'paloma::paloma' else [m])]))", SUITE],
        capture_output=True, text=True, check=True, cwd=OLMES_ROOT)
    return yaml.safe_load(out.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run names under genscale/")
    ap.add_argument("--step", type=int, default=19074)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--qos", default="h200_comm_shared")
    ap.add_argument("--hours", type=int, default=12)
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()
    tasks = suite_members()
    print(f"{len(tasks)} tasks: {tasks}")
    for run in args.runs:
        run_dir = f"{GENSCALE}/{run}"
        src = f"{run_dir}/evals/{args.step:010d}"
        dst = f"{run_dir}/evals_v2/{args.step:010d}"
        cfg = yaml.safe_load(open(f"{src}/base_config.yaml"))
        cfg["dump_dir"] = dst
        cfg["metric_log_dir"] = f"{run_dir}/evals_v2"
        cfg["olmes"]["tasks"] = tasks
        cfg["olmes"]["timeout_hours"] = float(args.hours - 1)
        cfg["validation"] = None
        cfg["wandb"] = None
        os.makedirs(f"{dst}/logs", exist_ok=True)
        yaml.safe_dump(cfg, open(f"{dst}/base_config.yaml", "w"), sort_keys=False)
        sub = open(f"{src}/submit.slurm").read().replace(src, dst)
        sub = re.sub(r"#SBATCH --qos=\S+", f"#SBATCH --qos={args.qos}", sub)
        sub = re.sub(r"#SBATCH --gres=gpu:\d+", f"#SBATCH --gres=gpu:{args.gpus}", sub)
        sub = re.sub(r"#SBATCH --time=\d+", f"#SBATCH --time={args.hours * 60}", sub)
        sub = re.sub(r"#SBATCH --job-name=\S+", "#SBATCH --job-name=olmes_evals_v2", sub)
        open(f"{dst}/submit.slurm", "w").write(sub)
        if args.submit:
            env = {k: v for k, v in os.environ.items() if k.lower() not in ("http_proxy", "https_proxy", "no_proxy")}
            r = subprocess.run(["sbatch", f"{dst}/submit.slurm"], capture_output=True, text=True, env=env,
                               cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            print(run, r.stdout.strip() or r.stderr.strip())
        else:
            print(f"{run}: wrote {dst}/submit.slurm (dry run)")


if __name__ == "__main__":
    main()
