"""Prune checkpoints of finished data-constrained / CompleteP runs.

A run is finished when its final eval (evals/<steps>/validation.json) exists. For finished runs:
  - delete every checkpoint step dir except the final one, unless an unfinished run branches from it;
  - in the final step dir delete the FSDP shards (*.distcp), train_state_*.json and hf/ (OLMES conversion);
    keep consolidated/ (model weights) and params.json.
Unfinished runs are not touched. Usage: prune_ckpts.py [--apply]
"""
import json, os, shutil, sys
RUNS = "/home/sgeng/scottg/pretrain/lingua-runs/runs/midtrain/experiments"
GROUPS = {"data_constrainedv2": "/home/sgeng/scottg/pretrain/lingua-clean",
          "qwen3_ladder_dataconstrained": "/home/sgeng/scottg/pretrain/lingua-clean",
          "completep_ladder": "/home/sgeng/scottg/pretrain/lingua-clean-completep"}
apply = "--apply" in sys.argv

def size(p):
    if os.path.isfile(p): return os.path.getsize(p)
    t = 0
    for r, _, fs in os.walk(p):
        for f in fs:
            try: t += os.path.getsize(os.path.join(r, f))
            except OSError: pass
    return t

def expand(p):
    return p.replace("${oc.env:MIDTRAIN_ROOT}", "/home/sgeng/scottg/pretrain/lingua-runs/runs/midtrain")

tot = 0; nrun = 0; skipped = 0
for g, root in GROUPS.items():
    M = json.load(open(f"{root}/apps/main/configs/experiments/{g}/manifest.json"))
    fin = {n: os.path.exists(f"{RUNS}/{g}/{n}/evals/{e['steps']:010d}/validation.json") for n, e in M.items()}
    keep = set()  # checkpoint dirs some unfinished run still branches from
    for n, e in M.items():
        if not fin[n] and e.get("branch_from"):
            keep.add(os.path.normpath(expand(e["branch_from"])))
    gt = 0
    for n, e in M.items():
        if not fin[n]:
            skipped += 1; continue
        cdir = f"{RUNS}/{g}/{n}/checkpoints"
        if not os.path.isdir(cdir): continue
        final = f"{e['steps']:010d}"
        for d in sorted(os.listdir(cdir)):
            p = os.path.normpath(f"{cdir}/{d}")
            if d != final:
                if p in keep: continue
                gt += size(p)
                if apply: shutil.rmtree(p, ignore_errors=True)
            else:
                for f in os.listdir(p):
                    if f.endswith(".distcp") or f.startswith("train_state_") or f == "hf":
                        q = f"{p}/{f}"; gt += size(q)
                        if apply:
                            shutil.rmtree(q, ignore_errors=True) if os.path.isdir(q) else os.remove(q)
        nrun += 1
    print(f"{g}: {'freed' if apply else 'would free'} {gt/1e12:.2f} TB", flush=True)
    tot += gt
print(f"total {tot/1e12:.2f} TB over {nrun} finished runs; {skipped} unfinished runs untouched")
