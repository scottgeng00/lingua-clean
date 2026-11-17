"""Strip the AdamW state from final consolidated checkpoints of finished runs (user-approved 2026-11-03).

consolidated.pth = {"model": fp32 weights, "optim": Adam state (2/3 of the bytes)}. For every finished run (final
evals/<steps>/validation.json exists) we rewrite it as {"model": ...} (all non-"optim" keys kept, still fp32):
write consolidated.pth.tmp, reload it and check every tensor is bit-identical to the original, then os.replace.
Kept untouched (full weights + optimizer): the best run (lowest own-heldout val CE) of every
(group, width, family, eval point), e.g. (data_constrainedv2, 512, "u", "2x") — a few dozen runs for Adam-state analysis.

    strip_optim.py            # dry run: counts, bytes, list of kept-best runs
    strip_optim.py --apply
"""
import json, os, re, sys
from collections import defaultdict
from multiprocessing import Pool

import torch

RUNS = "/home/sgeng/scottg/pretrain/lingua-runs/runs/midtrain/experiments"
GROUPS = {"data_constrainedv2": "/home/sgeng/scottg/pretrain/lingua-clean",
          "completep_ladder": "/home/sgeng/scottg/pretrain/lingua-clean-completep"}


def val_ce(path):
    d = json.load(open(path))
    for k in ("fwedu_heldout", "dclm_heldout"):  # own heldout first (FineWeb-Edu runs list it first)
        if k in d and isinstance(d[k], dict):
            return d[k]["ce_per_token"]


def candidates():
    best = {}
    runs = []
    for g, root in GROUPS.items():
        M = json.load(open(f"{root}/apps/main/configs/experiments/{g}/manifest.json"))
        for n, e in M.items():
            v = f"{RUNS}/{g}/{n}/evals/{e['steps']:010d}/validation.json"
            ck = f"{RUNS}/{g}/{n}/checkpoints/{e['steps']:010d}/consolidated/consolidated.pth"
            if not (os.path.exists(v) and os.path.exists(ck)):
                continue
            fam = re.match(r"w\d+_([a-z0-9]+)_", n).group(1)
            ce = val_ce(v)
            if "fwedu_heldout" in json.load(open(v)):
                fam += "@fwedu"
            key = (g, e["width"], fam, e.get("eval_point"))
            runs.append((key, ce, n, ck))
            if ce is not None and (key not in best or ce < best[key][0]):
                best[key] = (ce, n)
    keep = {n for _, n in best.values()}
    return runs, keep, best


def strip(ck):
    sd = torch.load(ck, map_location="cpu", mmap=True, weights_only=False)
    if "optim" not in sd:
        return ck, 0, "no optim"
    before = os.path.getsize(ck)
    out = {k: v for k, v in sd.items() if k != "optim"}
    tmp = ck + ".tmp"
    torch.save(out, tmp)
    chk = torch.load(tmp, map_location="cpu", mmap=True, weights_only=False)
    assert chk.keys() == out.keys()
    for k in out:
        a, b = out[k], chk[k]
        if isinstance(a, dict):
            assert a.keys() == b.keys(), k
            for t in a:
                assert (torch.equal(a[t], b[t]) and a[t].dtype == b[t].dtype) if torch.is_tensor(a[t]) else a[t] == b[t], t
        else:
            assert a == b, k
    os.replace(tmp, ck)
    return ck, before - os.path.getsize(ck), "ok"


if __name__ == "__main__":
    runs, keep, best = candidates()
    todo = [ck for _, _, n, ck in runs if n not in keep]
    print(f"{len(runs)} finished finals; keeping full state for {len(keep)} best runs; stripping {len(todo)}", flush=True)
    for key, (ce, n) in sorted(best.items(), key=str):
        print(f"  keep {key}: {n} ({ce:.4f})")
    if "--apply" not in sys.argv:
        sys.exit()
    freed = 0
    with Pool(12) as pool:
        for i, (ck, b, msg) in enumerate(pool.imap_unordered(strip, todo), 1):
            freed += b
            if msg != "ok" or i % 100 == 0:
                print(f"[{i}/{len(todo)}] {msg} {ck} freed so far {freed / 1e12:.2f} TB", flush=True)
    print(f"done: freed {freed / 1e12:.2f} TB", flush=True)
