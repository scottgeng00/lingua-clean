"""Launch the async eval job that train.py launches at an evaluation step, for an existing checkpoint (for runs whose
eval launch failed or was skipped, e.g. preempted right after the final save). Same eval config (the run's `eval`
block), GPUs, account, QoS and partition as the training-time launch. Needs scripts/env.sh in the environment.

    source scripts/env.sh && .venv-kd/bin/python scripts/launch_eval_for_ckpt.py \
        apps/main/configs/experiments/qwen3_ladder_dataconstrained/<run>.yaml <step>
"""

import os
import sys
from dataclasses import asdict

from omegaconf import OmegaConf

from apps.main.eval_olmes import EVAL_FOLDER_NAME, OlmesEvalArgs
from lingua.args import dataclass_from_dict
from lingua.stool import StoolArgs, launch_job


def main():
    cfg_path, step = sys.argv[1], int(sys.argv[2])
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    assert cfg.get("eval_backend") == "olmes", "only the olmes eval backend is supported"
    eval_dict = dict(cfg["eval"])
    eval_dict.pop("harness", None)
    eval_args = dataclass_from_dict(OlmesEvalArgs, eval_dict)
    ckpt = os.path.join(cfg["dump_dir"], "checkpoints", f"{step:010d}")
    assert os.path.isfile(os.path.join(ckpt, ".metadata")), f"no complete checkpoint at {ckpt}"
    eval_args.global_step = step
    eval_args.ckpt_dir = ckpt
    eval_args.dump_dir = os.path.join(cfg["dump_dir"], "evals", EVAL_FOLDER_NAME.format(step))
    eval_args.metric_log_dir = cfg["dump_dir"]
    gpus = cfg["async_eval_gpus"]
    launch_job(StoolArgs(asdict(eval_args), script="apps.main.eval_olmes", copy_code=False,
                         nodes=max(1, gpus // 8), ngpu=gpus, time=480, mem="200GB",
                         account=os.environ["SLURM_ACCOUNT"], qos=os.environ["SLURM_QOS"], partition="g3",
                         tasks_per_node=1, override=False, dirs_exists_ok=True,
                         anaconda=os.environ.get("LINGUA_EVAL_VENV") or os.environ.get("LINGUA_VENV", "default")))
    print(f"launched eval for {cfg['dump_dir']} step {step} -> {eval_args.dump_dir}")


if __name__ == "__main__":
    main()
