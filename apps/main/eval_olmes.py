"""
OLMES-based evaluation for Lingua training checkpoints.

Replaces lm-eval-harness with OLMES by:
  1. Consolidating the distributed checkpoint (subprocess — frees GPU)
  2. Converting to HuggingFace format          (subprocess — frees GPU)
  3. Running validation perplexity             (subprocess — frees GPU)
  4. Running OLMES via subprocess (in its own conda env, gets a clean GPU)
  5. Parsing results and logging to wandb

GPU-heavy work (steps 1-3) runs in a child process so that ALL GPU memory
is released before OLMES/vLLM starts. This avoids OOM from lingering CUDA
contexts, NCCL buffers, and compiled kernel caches.
"""

import gc
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import wandb
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from apps.main.eval import ValidationArgs, eval_on_val
from apps.main.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer,
)
from apps.main.transformer import LMTransformer, LMTransformerArgs
from lingua.args import dump_config
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from lingua.distributed import (
    DistributedArgs,
    get_global_rank,
    setup_torch_distributed,
)
from lingua.metrics import WandbArgs
from setup.convert_consolidated_lingua_ckpt_to_hf import write_model

EVAL_FOLDER_NAME = "{:010d}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()


@dataclass
class OlmesArgs:
    tasks: List[str] = field(default_factory=lambda: [
        "arc_challenge::olmes",
        "arc_easy::olmes",
        "hellaswag::olmes",
        "mmlu::olmes",
        "winogrande::olmes",
        "boolq::olmes",
        "piqa::olmes",
        "openbookqa::olmes",
        "naturalqs::olmes",
        "drop::olmes",
        "gsm8k::olmes",
    ])
    model_type: str = "vllm"
    tokenizer_path: str = "allenai/OLMo-2-0425-1B"
    tokenizer_revision: str = "main"
    max_length: int = 4096
    gpu_memory_utilization: float = 0.8
    trust_remote_code: bool = True
    olmes_env: str = os.environ.get(
        "OLMES_CONDA_ENV",
        os.path.join(os.environ.get("HOME", ""), "miniconda3", "envs", "olmes"),
    )
    # Kill OLMES if it runs longer than this: a crashed vLLM run can leave the
    # process alive (worker processes don't exit) and hold the GPU until the
    # job's time limit.
    timeout_hours: float = 3.0


@dataclass
class OlmesEvalArgs:
    name: str = "olmes_evals"
    dump_dir: Optional[str] = None
    metric_log_dir: Optional[str] = None
    ckpt_dir: str = ""
    global_step: Optional[int] = None

    generator: PackedCausalTransformerGeneratorArgs = field(
        default_factory=PackedCausalTransformerGeneratorArgs
    )
    olmes: OlmesArgs = field(default_factory=OlmesArgs)
    validation: Optional[ValidationArgs] = field(default_factory=ValidationArgs)

    wandb: Optional[WandbArgs] = None


# ---------------------------------------------------------------------------
# GPU work — runs in a subprocess so memory is fully freed on exit
# ---------------------------------------------------------------------------

def do_gpu_work(cfg: OlmesEvalArgs):
    """Consolidate checkpoint, run validation perplexity, convert to HF.

    This function is designed to run in a *separate process* so that when it
    returns (and the process exits), all GPU memory — CUDA context, NCCL
    buffers, compiled kernels — is completely freed for the OLMES/vLLM step.
    """
    if not torch.distributed.is_initialized():
        setup_torch_distributed(DistributedArgs())

    # --- Consolidate ---
    consolidate_path = Path(cfg.ckpt_dir) / CONSOLIDATE_FOLDER
    if not consolidate_path.exists():
        if get_global_rank() == 0:
            logger.info(f"Consolidating checkpoint at {cfg.ckpt_dir}")
            consolidate_path = consolidate_checkpoints(cfg.ckpt_dir)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    else:
        logger.info(f"Consolidated checkpoint already exists at {consolidate_path}")

    # --- Validation perplexity ---
    if cfg.validation and get_global_rank() == 0:
        logger.info("Loading Lingua model for validation perplexity...")
        model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
            str(consolidate_path),
            model_cls=LMTransformer,
            model_args_cls=LMTransformerArgs,
        )
        model.eval()
        generator = PackedCausalTransformerGenerator(cfg.generator, model, tokenizer)
        val_results = eval_on_val(generator, cfg.validation, train_cfg)

        Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(cfg.dump_dir) / "validation.json", "w") as f:
            json.dump(val_results, f)
        logger.info(f"Validation results: {val_results}")

        del generator, model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Freed Lingua model from GPU memory")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # --- Convert to HF ---
    hf_path = Path(cfg.ckpt_dir) / "hf"
    if hf_path.exists() and (hf_path / "success.txt").exists():
        logger.info(f"HF checkpoint already exists at {hf_path}, skipping conversion")
    elif get_global_rank() == 0:
        logger.info(f"Converting consolidated checkpoint to HF format -> {hf_path}")
        write_model(
            model_path=str(hf_path),
            input_base_path=str(consolidate_path),
            tokenizer_path=cfg.olmes.tokenizer_path,
            safe_serialization=True,
        )
        tok = AutoTokenizer.from_pretrained(cfg.olmes.tokenizer_path)
        tok.save_pretrained(str(hf_path))
        logger.info(f"HF checkpoint saved to {hf_path}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    logger.info("GPU work subprocess finished")


# ---------------------------------------------------------------------------
# OLMES invocation
# ---------------------------------------------------------------------------

def _clean_env_for_olmes() -> dict:
    """Build a clean environment for OLMES/vLLM by stripping every
    distributed / SLURM / MPI / PMIX / NCCL / torchelastic variable that could
    trick PyTorch or vLLM into attempting multi-node / multi-GPU init."""
    env = os.environ.copy()

    _REMOVE_PREFIXES = (
        "MASTER_",
        "TORCHELASTIC_",
        "TORCH_NCCL_",
        "TORCH_FR_",
        "NCCL_",
        "SLURM_",
        "PMIX_",
        "OMPI_",
        "PMI_",
        "I_MPI_",
        "ROLE_",
        "GROUP_",
        "HYDRA_",
    )
    _REMOVE_EXACT = {
        "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
        "DORA_FORCE_DISTRIB",
        "ENABLE_INTRA_NODE_COMM",
        "LAUNCH_WITH",
        "ENVIRONMENT",
        "NODE_IP",
        "OPEN_MPI_PATH",
        "EFA_PATH",
        "FI_PROVIDER",
        "FI_EFA_FORK_SAFE",
        "RAY_CLIENT_MODE",
    }

    for key in list(env.keys()):
        if any(key.startswith(p) for p in _REMOVE_PREFIXES) or key in _REMOVE_EXACT:
            del env[key]

    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    return env


# Rough relative cost of OLMES tasks, for balancing tasks across GPUs.
_OLMES_TASK_COST = {"mmlu": 6, "mmlu_pro": 6, "gsm8k": 4, "triviaqa": 4, "naturalqs": 3, "drop": 3, "minerva_math": 3}


def _task_cost(task: str) -> int:
    base = task.split(":")[0]
    if ":bpb" in task:  # gold-likelihood pass only, no generation
        return 1
    return _OLMES_TASK_COST.get(base, 1)


def _visible_gpus() -> List[str]:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return [g for g in vis.split(",") if g.strip()]
    return [str(i) for i in range(torch.cuda.device_count())]


def _shard_tasks(tasks: List[str], n: int) -> List[List[str]]:
    """Longest-processing-time-first assignment of tasks to n shards."""
    shards: List[List[str]] = [[] for _ in range(n)]
    load = [0] * n
    for t in sorted(tasks, key=_task_cost, reverse=True):
        i = load.index(min(load))
        shards[i].append(t)
        load[i] += _task_cost(t)
    return [sh for sh in shards if sh]


def _olmes_cmd(hf_path: str, olmes_args: OlmesArgs, tasks: List[str], output_dir: str) -> str:
    model_args = {
        "trust_remote_code": olmes_args.trust_remote_code,
        "gpu_memory_utilization": olmes_args.gpu_memory_utilization,
        "max_length": olmes_args.max_length,
        "tokenizer": olmes_args.tokenizer_path,
    }
    # Newer OLMES rejects unknown model args ("Hashing error! Key tokenizer_revision
    # not in defaults"), so only pass a revision when one is actually requested.
    if olmes_args.tokenizer_revision and olmes_args.tokenizer_revision != "main":
        model_args["tokenizer_revision"] = olmes_args.tokenizer_revision
    model_args = json.dumps(model_args)

    conda_exe = os.environ.get("CONDA_EXE", "conda")
    olmes_env = olmes_args.olmes_env

    # Belt-and-suspenders: also unset in bash in case conda activate re-introduces anything
    cmd = f"""set -e
if [ -f {olmes_env}/pyvenv.cfg ]; then
    source {olmes_env}/bin/activate
else
    eval "$({conda_exe} shell.bash hook)"
    conda activate {olmes_env}
fi

# Nuke every distributed / SLURM / MPI / PMIX / torchelastic var
unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
unset GROUP_RANK GROUP_WORLD_SIZE ROLE_RANK ROLE_WORLD_SIZE
unset TORCHELASTIC_RUN_ID TORCHELASTIC_USE_AGENT_STORE TORCHELASTIC_ERROR_FILE
unset TORCHELASTIC_MAX_RESTARTS TORCHELASTIC_RESTART_COUNT
unset PMIX_RANK PMIX_NAMESPACE PMIX_SERVER_URI2 PMIX_SERVER_URI3 PMIX_SERVER_URI4 PMIX_SERVER_URI21 PMIX_SERVER_URI41
unset SLURM_PROCID SLURM_LOCALID SLURM_NTASKS SLURM_NODEID SLURM_GTIDS
unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_SIZE OMPI_COMM_WORLD_LOCAL_RANK
unset PMI_RANK PMI_SIZE PMI_FD
unset DORA_FORCE_DISTRIB NODE_IP ENABLE_INTRA_NODE_COMM

export TORCHDYNAMO_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

olmes \
    --model {hf_path} \
    --model-type {olmes_args.model_type} \
    --model-args '{model_args}' \
    --task {' '.join(tasks)} \
    --output-dir {output_dir}
"""

    return cmd


def _wait_olmes(proc: subprocess.Popen, deadline: float, label: str) -> int:
    try:
        proc.wait(timeout=max(1.0, deadline - time.time()))
    except subprocess.TimeoutExpired:
        logger.error(f"OLMES {label} still running at the time limit; killing it")
        import signal
        os.killpg(proc.pid, signal.SIGKILL)  # whole group, incl. vLLM workers
        proc.wait()
    return proc.returncode


def run_olmes(hf_path: str, olmes_args: OlmesArgs, output_dir: str) -> Optional[dict]:
    """Run OLMES in the OLMES env; with N visible GPUs, split the tasks into N
    shards, run one OLMES process per GPU, and merge the shards' metrics.json."""
    os.makedirs(output_dir, exist_ok=True)
    gpus = _visible_gpus() or [None]
    shards = _shard_tasks(list(olmes_args.tasks), len(gpus))
    clean_env = _clean_env_for_olmes()
    logger.info("Clean env — removed all SLURM/NCCL/MPI/PMIX/TORCHELASTIC/distributed vars")

    procs = []
    for i, tasks in enumerate(shards):
        out = output_dir if len(shards) == 1 else str(Path(output_dir) / f"shard_{i}")
        cmd = _olmes_cmd(hf_path, olmes_args, tasks, out)
        env = dict(clean_env)
        if gpus[i] is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpus[i]
        logger.info(f"Running OLMES shard {i} on GPU {gpus[i]}: {tasks}\n{cmd}")
        procs.append((i, out, subprocess.Popen(["bash", "-c", cmd], text=True, env=env, start_new_session=True)))

    deadline = time.time() + olmes_args.timeout_hours * 3600
    shard_metrics = []
    for i, out, proc in procs:
        rc = _wait_olmes(proc, deadline, f"shard {i}")
        metrics_file = Path(out) / "metrics.json"
        if rc != 0 and metrics_file.exists():
            # OLMES writes metrics.json last; if it then hung on exit and was killed,
            # the results are complete.
            logger.warning(f"OLMES shard {i} exited with code {rc} after writing metrics.json; using the results")
        elif rc != 0 or not metrics_file.exists():
            logger.error(f"OLMES shard {i} failed (return code {rc}, metrics.json present: {metrics_file.exists()})")
            continue
        with open(metrics_file) as f:
            shard_metrics.append(json.load(f))

    if not shard_metrics:
        return None
    if len(shards) == 1:
        metrics = shard_metrics[0]
    else:
        metrics = {
            "all_primary_scores": [x for m in shard_metrics for x in m.get("all_primary_scores", [])],
            "tasks": [x for m in shard_metrics for x in m.get("tasks", [])],
            "model_config": shard_metrics[0].get("model_config"),
        }
        with open(Path(output_dir) / "metrics.json", "w") as f:
            json.dump(metrics, f)
        if len(shard_metrics) < len(shards):
            logger.error(f"Only {len(shard_metrics)}/{len(shards)} OLMES shards produced results")

    logger.info(f"OLMES evaluation complete. Tasks evaluated: {len(metrics.get('tasks', []))}")
    return metrics


def parse_olmes_results(olmes_metrics: dict) -> dict:
    """Extract per-task primary scores from OLMES metrics.json into a flat dict
    suitable for wandb logging and metric files.
    """
    results = {}
    for task in olmes_metrics.get("tasks", []):
        alias = task.get("alias", "unknown")
        task_metrics = task.get("metrics", {})
        primary_score = task_metrics.get("primary_score")
        if primary_score is not None:
            results[f"olmes_eval/{alias}/primary_score"] = primary_score
        acc_raw = task_metrics.get("acc_raw")
        if acc_raw is not None:
            results[f"olmes_eval/{alias}/acc_raw"] = acc_raw
    return results


# ---------------------------------------------------------------------------
# Orchestrator — the parent process never touches the GPU
# ---------------------------------------------------------------------------

def launch_olmes_eval(cfg: OlmesEvalArgs):
    Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)
    dump_config(cfg, Path(cfg.dump_dir) / "config.yaml", log_config=False)

    # --- Wandb (no GPU needed) ---
    wandb_run = None
    if cfg.wandb is not None:
        wandb_kwargs = {k: v for k, v in asdict(cfg.wandb).items() if v is not None}
        wandb_kwargs["resume"] = "allow"
        if cfg.wandb.id:
            logger.info(f"Resuming wandb run with id: {cfg.wandb.id}")
        wandb_run = wandb.init(**wandb_kwargs)
        logger.info(f"Wandb run initialized: {wandb_run.id} (name: {wandb_run.name})")

    # --- Step 1: GPU work in a subprocess (consolidate + validate + convert) ---
    # The subprocess exits after completion, fully freeing all GPU memory.
    config_path = Path(cfg.dump_dir) / "config.yaml"
    logger.info("Launching GPU work subprocess (consolidate + validate + convert)...")
    gpu_result = subprocess.run(
        [sys.executable, "-m", "apps.main.eval_olmes",
         f"config={config_path}", "mode=gpu_work"],
        text=True,
        capture_output=False,
    )
    if gpu_result.returncode != 0:
        logger.error(f"GPU work subprocess failed with return code {gpu_result.returncode}")
        return

    logger.info("GPU work subprocess exited — GPU memory fully released")

    # Read validation results written by the subprocess
    val_results = None
    val_result_path = Path(cfg.dump_dir) / "validation.json"
    if val_result_path.exists():
        with open(val_result_path) as f:
            val_results = json.load(f)

    # --- Step 2: Run OLMES (gets a completely clean GPU) ---
    olmes_results = None
    hf_path = str(Path(cfg.ckpt_dir) / "hf")
    olmes_output_dir = str(Path(cfg.dump_dir) / "olmes_results")
    olmes_metrics = run_olmes(hf_path, cfg.olmes, olmes_output_dir)
    if olmes_metrics is not None:
        olmes_results = parse_olmes_results(olmes_metrics)
        with open(Path(cfg.dump_dir) / "results.json", "w") as f:
            json.dump(olmes_metrics, f)
        logger.info(f"OLMES results: {olmes_results}")

    # --- Step 3: Write metric logs ---
    if cfg.metric_log_dir:
        timestamp = {"created_at": datetime.utcnow().isoformat()}
        if cfg.global_step is not None:
            timestamp["global_step"] = cfg.global_step

        if olmes_results is not None:
            metric_log_path = Path(cfg.metric_log_dir) / "metrics.olmes_eval.jsonl"
            logger.info(f"Writing OLMES metric logs to {metric_log_path}")
            with open(metric_log_path, mode="a") as f:
                print(json.dumps(timestamp | olmes_results), file=f, flush=True)

        if val_results is not None:
            val_log_path = Path(cfg.metric_log_dir) / "metrics.validation.jsonl"
            with open(val_log_path, mode="a") as f:
                print(json.dumps(timestamp | val_results), file=f, flush=True)

    # --- Step 4: Log to wandb ---
    if wandb_run is not None:
        wandb_metrics = {}

        if olmes_results is not None:
            wandb_metrics.update(olmes_results)

        if val_results is not None:
            for src_name, src_metrics in val_results.items():
                for metric_name, value in src_metrics.items():
                    if isinstance(value, (int, float)):
                        wandb_metrics[f"validation/{src_name}/{metric_name}"] = value

        if wandb_metrics and cfg.global_step is not None:
            wandb.log(wandb_metrics, step=cfg.global_step)
            logger.info(f"Logged {len(wandb_metrics)} metrics to wandb at step {cfg.global_step}")

        if cfg.wandb and cfg.wandb.id:
            logger.info(f"Not finishing wandb run {wandb_run.id} - will remain open for training")
        else:
            wandb.finish()
            logger.info("Wandb run finished")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config

    mode = cli_args.get("mode", "full")
    if "mode" in cli_args:
        del cli_args.mode

    default_cfg = OmegaConf.structured(OlmesEvalArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    if mode == "gpu_work":
        do_gpu_work(cfg)
    else:
        print(cfg)
        launch_olmes_eval(cfg)


if __name__ == "__main__":
    main()
