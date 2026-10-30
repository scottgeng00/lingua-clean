"""Compact status of the data-constrained ladder runs: slurm state, last logged step / total, validation and
Paloma results present, plus counts by state.

    python3 scripts/dataconstrained_status.py [--select '^w512_s1_'] [--all]   (--all also lists finished runs)
"""

import argparse
import collections
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUP = "qwen3_ladder_dataconstrained"
MANIFEST = f"{ROOT}/apps/main/configs/experiments/{GROUP}/manifest.json"
RUNS = f"/checkpoint/mnm/sgeng/pretrain/lingua-runs/runs/midtrain/experiments/{GROUP}"


def last_step(d):
    path = f"{d}/metrics.jsonl"
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        f.seek(max(0, os.path.getsize(path) - 4096))
        lines = f.read().decode(errors="ignore").strip().split("\n")
    for line in reversed(lines):
        try:
            return json.loads(line)["global_step"]
        except (ValueError, KeyError):
            continue
    return 0


def has_step(path, step):
    return os.path.exists(path) and any(json.loads(l).get("global_step") == step for l in open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", default=".")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    manifest = json.load(open(MANIFEST))
    q = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%j %T"], capture_output=True, text=True).stdout
    state = dict(l.split() for l in q.strip().split("\n") if l)
    counts = collections.Counter()
    print(f"{'run':44s} {'state':9s} {'step':>13s} {'val':>4s} {'pal':>4s}")
    for name, e in sorted(manifest.items()):
        if not re.search(a.select, name):
            continue
        d = f"{RUNS}/{name}"
        st = state.get(f"lingua_clean_pt_experiments_{GROUP}_{name}", "")
        done = os.path.isfile(f"{d}/checkpoints/{e['steps']:010d}/.metadata")
        val = has_step(f"{d}/metrics.validation.jsonl", e["steps"])
        pal = has_step(f"{d}/metrics.paloma_np.jsonl", e["steps"])
        label = st or ("evaluated" if val and pal else "trained" if done else "waiting")
        counts[label] += 1
        if a.all or label != "evaluated":
            print(f"{name:44s} {label:9s} {last_step(d):>6}/{e['steps']:<6} {'y' if val else '-':>4s} {'y' if pal else '-':>4s}")
    print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
