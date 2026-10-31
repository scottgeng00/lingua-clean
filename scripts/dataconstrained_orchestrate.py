"""One pass of the data-constrained ladder orchestrator (setup/make_dataconstrained_configs.py manifest).

For every manifest run whose name matches --select: skip it if finished (final checkpoint saved) or queued/running;
launch a main run right away and a decay branch once its source checkpoint is complete. A run is (re)launched at
most --max-launches times (counted in logs/dataconstrained/launches/), so a deterministic failure is reported instead
of retried forever. Prints one line per action; silent when there is nothing to do.

    python3 scripts/dataconstrained_orchestrate.py --select '^w512_s1_' [--group data_constrainedv2] [--dry-run]
"""

import argparse
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUP = "qwen3_ladder_dataconstrained"  # or data_constrainedv2 (--group)
COUNTS = "/checkpoint/mnm/sgeng/pretrain/logs/dataconstrained/launches"
EXCLUDE = "g3-144-119"


def ckpt_done(path):
    return os.path.isfile(f"{path}/.metadata") and os.path.isfile(f"{path}/train_state_00000.json")


def resolve(p):
    return p.replace("${oc.env:MIDTRAIN_ROOT}", "/checkpoint/mnm/sgeng/pretrain/lingua-runs/runs/midtrain")


def launch(entry, dry, group):
    gpus = entry["gpus"]
    per_node = min(gpus, 4)
    env = {k: v for k, v in os.environ.items() if k.lower() not in ("http_proxy", "https_proxy")}
    env.update(CONFIG=f"experiments/{group}/{entry['name']}", NNODES=str(-(-gpus // 4)), GPUS_PER_NODE=str(per_node),
               CPUS_PER_NODE=str(32 * per_node), MEM_PER_NODE=f"{180 * per_node}G", STEPS_OVERRIDE=str(entry["steps"]),
               TIME_LIMIT="3-00:00:00", SLURM_EXCLUDE=EXCLUDE, SLURM_QOS="g3_comm_shared")
    if dry:
        return "dry-run"
    cmd = f"cd {ROOT} && source scripts/env.sh >/dev/null && bash scripts/launch_pretrain.sh"
    out = subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True).stdout
    m = re.search(r"Submitted batch job (\d+)", out)
    return m.group(1) if m else f"FAILED: {out[-300:]}"


def launch_eval(entry, group):
    env = {k: v for k, v in os.environ.items() if k.lower() not in ("http_proxy", "https_proxy", "hf_token")}
    env["PYTHONPATH"] = ROOT
    cmd = (f"cd {ROOT} && source scripts/env.sh >/dev/null && .venv-kd/bin/python scripts/launch_eval_for_ckpt.py "
           f"apps/main/configs/experiments/{group}/{entry['name']}.yaml {entry['steps']}")
    out = subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True)
    return "ok" if "launched eval" in out.stdout else f"FAILED: {(out.stdout + out.stderr)[-300:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", default="^$", help="regex on run names to manage")
    ap.add_argument("--max-launches", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--group", default=GROUP)
    a = ap.parse_args()
    manifest = json.load(open(f"{ROOT}/apps/main/configs/experiments/{a.group}/manifest.json"))
    runs = f"/checkpoint/mnm/sgeng/pretrain/lingua-runs/runs/midtrain/experiments/{a.group}"
    active = set(subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%j"], capture_output=True,
                                text=True).stdout.split())
    os.makedirs(COUNTS, exist_ok=True)
    sel = re.compile(a.select)
    for name, e in sorted(manifest.items()):
        if not sel.search(name):
            continue
        if f"lingua_clean_pt_experiments_{a.group}_{name}" in active:
            continue  # queued or running (or finishing: the eval launch follows the final save)
        if ckpt_done(f"{runs}/{name}/checkpoints/{e['steps']:010d}"):
            # Trained; make sure its eval was launched (a failed launch or a preemption right after the final save
            # leaves no evals/<step> folder).
            if not os.path.isdir(f"{runs}/{name}/evals/{e['steps']:010d}") and not a.dry_run:
                print(f"launching missing eval for {name}: {launch_eval(e, a.group)}")
            continue
        if "branch_from" in e and not ckpt_done(resolve(e["branch_from"])):
            continue  # its main run hasn't reached the branch point yet
        cpath = f"{COUNTS}/{name}"
        n = int(open(cpath).read()) if os.path.exists(cpath) else 0
        if n >= a.max_launches:
            print(f"GIVING UP on {name} (launched {n} times)")
            continue
        job = launch(e, a.dry_run, a.group)
        if not a.dry_run:
            with open(cpath, "w") as f:
                f.write(str(n + 1))
        print(f"{'re' if n else ''}launched {name} ({e['gpus']} GPU, {e['steps']} steps): {job}")


if __name__ == "__main__":
    main()
