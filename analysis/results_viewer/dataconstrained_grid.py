"""LR x batch grids of validation cross-entropy (dclm_heldout, nats/token) for the data-constrained ladder.

    .venv-kd/bin/python analysis/results_viewer/dataconstrained_grid.py --width 512 --stage 1 [--metric val_ce|paloma]
Stage 1 prints the 2x (main) and 1x (branch) grids per weight decay; stage 2 prints each run's best loss over the
evaluated passes (and which pass), plus the grid at each pass.
"""
import argparse
import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GROUP = "qwen3_ladder_dataconstrained"
M = json.load(open(f"{ROOT}/apps/main/configs/experiments/{GROUP}/manifest.json"))
RUNS = f"/checkpoint/mnm/sgeng/pretrain/lingua-runs/runs/midtrain/experiments/{GROUP}"


def metric(name, which):
    e = M[name]
    if which == "val_ce":
        f = f"{RUNS}/{name}/evals/{e['steps']:010d}/validation.json"
        return json.load(open(f))["dclm_heldout"]["ce_per_token"] if os.path.exists(f) else None
    f = f"{RUNS}/{name}/metrics.paloma_np.jsonl"
    if not os.path.exists(f):
        return None
    for line in open(f):
        r = json.loads(line)
        if r["global_step"] == e["steps"]:
            p = [v for k, v in r.items() if k.endswith("primary_score") and "/paloma::" not in k]
            return sum(p) / len(p) if len(p) == 16 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--metric", default="val_ce", choices=["val_ce", "paloma"])
    a = ap.parse_args()
    runs = {n: e for n, e in M.items() if e["width"] == a.width and e["stage"] == a.stage
            and re.match(rf"w{a.width}_s{a.stage}_", n)}
    lrs = sorted({e["lr"] for e in runs.values()})
    bss = sorted({e["bs"] for e in runs.values()})
    wds = sorted({e["wd"] for e in runs.values()})
    points = sorted({e["eval_point"] for e in runs.values()}, key=lambda p: (len(p), p))
    vals = {(e["lr"], e["bs"], e["wd"], e["eval_point"]): metric(n, a.metric) for n, e in runs.items() if e["seed"] == 0}
    ranked = []
    for wd in wds:
        for pt in points:
            if not any(vals.get((l, b, wd, pt)) for l in lrs for b in bss):
                continue
            print(f"\nwd {wd}, eval {pt} ({a.metric})")
            print("bs \\ lr " + "".join(f"{l:>9.2g}" for l in lrs))
            for b in bss:
                print(f"{b:<8d}" + "".join(f"{vals[(l, b, wd, pt)]:9.4f}" if vals.get((l, b, wd, pt)) else "        -"
                                           for l in lrs))
            best = min(((v, l, b) for (l, b, w, p), v in vals.items() if w == wd and p == pt and v), default=None)
            if best:
                print(f"best: lr {best[1]:g}, bs {best[2]} -> {best[0]:.4f}")
    if a.stage == 2:
        best_over = defaultdict(lambda: (9, None))
        for (l, b, w, p), v in vals.items():
            if v and v < best_over[(l, b, w)][0]:
                best_over[(l, b, w)] = (v, p)
        print("\nrank by best loss over evaluated passes:")
        for (l, b, w), (v, p) in sorted(best_over.items(), key=lambda x: x[1][0])[:12]:
            print(f"  lr {l:g}  bs {b:<3d} wd {w:<4g} best {v:.4f} at {p}")


if __name__ == "__main__":
    main()
