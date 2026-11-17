"""Launch an OLMES downstream-eval job for the final checkpoint of finished data-constrained runs.

Uses the run's own eval block but overrides the task list (DOWNSTREAM below), runs OLMES through vLLM on 1 GPU, skips
the token-stream validation (already done at training time) and writes to evals/<step>_olmes/ so the original
evals/<step>/ (validation + packed Paloma) is untouched. Results: evals/<step>_olmes/olmes_results/metrics.json
(read by analysis/results_viewer/v2_table.py) and <run>/metrics.olmes_eval.jsonl.

    source scripts/env.sh && .venv-kd/bin/python scripts/launch_olmes_suite.py --group data_constrainedv2 RUN [RUN ...]

External checkpoints (a lingua checkpoint dir with a readable consolidated/, e.g. another user's runs):
pass NAME=CKPT_DIR pairs with --external OUT_ROOT; results go to OUT_ROOT/NAME/evals/<step>_olmes/, and the eval block
comes from --template (any data_constrainedv2 config: same Qwen3 tokenizer).
    ... launch_olmes_suite.py --external /path/out_root real=/path/runs/real/checkpoints/0000016398
"""
import argparse
import json
import os
from dataclasses import asdict

from omegaconf import OmegaConf

from apps.main.eval_olmes import OlmesEvalArgs
from lingua.args import dataclass_from_dict
from lingua.stool import StoolArgs, launch_job

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNSTREAM = [
    "arc:rc::olmes:full",  # ARC-Challenge + ARC-Easy
    "hellaswag:rc::olmes:full",
    "piqa:rc::olmes:full",
    "winogrande:rc::olmes:full",
    "sciq:rc::olmo3",
    "mmlu:rc::olmes",
    "mmlu:rc:bpb::olmes",
    *[f"minerva_math_{s}:bpb::olmes" for s in ("algebra", "counting_and_probability", "geometry",
                                               "intermediate_algebra", "number_theory", "prealgebra", "precalculus")],
    "codex_humaneval:3shot:bpb::none",
    "mbpp:3shot:bpb::none",
    "gsm8k::olmo3:n32:pass32",  # 8-shot, pass@1 and pass@32 over 32 samples
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="data_constrainedv2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--external", help="OUT_ROOT; runs are NAME=CKPT_DIR pairs")
    ap.add_argument("--template", default=f"{ROOT}/apps/main/configs/experiments/data_constrainedv2/"
                                          "w1152_u_bs256_lr2e-3_wd0p1.yaml")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    manifest = {} if a.external else json.load(open(f"{ROOT}/apps/main/configs/experiments/{a.group}/manifest.json"))
    for run in a.runs:
        if a.external:
            run, ckpt = run.split("=", 1)
            step = int(os.path.basename(ckpt.rstrip("/")))
            cfg = OmegaConf.to_container(OmegaConf.load(a.template), resolve=True)
            cfg["dump_dir"] = os.path.join(a.external, run)
            # Their hf/ export may not be readable by us: symlink the consolidated weights into
            # OUT_ROOT/NAME/checkpoints/<step>/consolidated/ and let eval_olmes write its own hf/ next to them.
            local = os.path.join(cfg["dump_dir"], "checkpoints", f"{step:010d}")
            os.makedirs(os.path.join(local, "consolidated"), exist_ok=True)
            for f in ("consolidated.pth", "params.json"):
                src, dst = os.path.join(ckpt, "consolidated", f), os.path.join(local, "consolidated", f)
                assert os.access(src, os.R_OK), f"cannot read {src}"
                if not os.path.lexists(dst):
                    os.symlink(src, dst)
            ckpt = local
        else:
            step = manifest[run]["steps"]
            cfg = OmegaConf.to_container(OmegaConf.load(f"{ROOT}/apps/main/configs/experiments/{a.group}/{run}.yaml"),
                                         resolve=True)
        eval_dict = dict(cfg["eval"])
        eval_dict.pop("harness", None)
        eval_dict["validation"] = None
        eval_dict["olmes"] = dict(eval_dict["olmes"], tasks=DOWNSTREAM, model_type="vllm", timeout_hours=7.0)
        args = dataclass_from_dict(OlmesEvalArgs, eval_dict)
        if not a.external:
            ckpt = os.path.join(cfg["dump_dir"], "checkpoints", f"{step:010d}")
        assert os.path.isfile(os.path.join(ckpt, "consolidated", "consolidated.pth")), f"no final weights at {ckpt}"
        args.global_step = step
        args.ckpt_dir = ckpt
        args.dump_dir = os.path.join(cfg["dump_dir"], "evals", f"{step:010d}_olmes")
        args.metric_log_dir = cfg["dump_dir"]
        if a.dry_run:
            print(f"[dry-run] {run} step {step} -> {args.dump_dir}")
            continue
        launch_job(StoolArgs(asdict(args), script="apps.main.eval_olmes", copy_code=False, nodes=1, ngpu=1,
                             time=480, mem="200GB", account=os.environ["SLURM_ACCOUNT"], qos=os.environ["SLURM_QOS"],
                             partition=os.environ.get("SLURM_PARTITION", ""), tasks_per_node=1, override=False, dirs_exists_ok=True,
                             anaconda=os.environ.get("LINGUA_EVAL_VENV") or os.environ.get("LINGUA_VENV", "default")))
        print(f"launched OLMES suite for {run} step {step} -> {args.dump_dir}")


if __name__ == "__main__":
    main()
