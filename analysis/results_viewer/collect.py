"""Collect training / validation / OLMES results for the runs in runs.yaml into one dict.

Sources per run (all optional; missing ones are skipped):
  <dir>/metrics.jsonl                    training log (loss/out, optim/lr, optim/total_tokens)
  <dir>/<val>                            validation NLL per Dolmino source, one line per step
  <dir>/{evals,olmes_base,olmes}/<step>/olmes_results/[shard_*/]task-*-metrics.json
                                         OLMES per-task metrics (primary score + gold BPB)

Subtask families (mmlu_*, minerva_math_*, ...) are macro-averaged into one task per
format, e.g. 57 x mmlu_<subject>:rc::olmes -> mmlu:rc::olmes, as OLMES itself reports.

    python analysis/results_viewer/collect.py --out /tmp/results.json
"""

import argparse
import glob
import json
import math
import os
import re
import time
from collections import defaultdict

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
OLMES_SUBDIRS = ["evals", "olmes_base", "olmes"]
# family prefix -> expected number of subtasks (averages over fewer are flagged partial,
# e.g. an eval that is still running). Longer prefixes first: mmlu_pro_* is MMLU-Pro.
FAMILIES = {"mmlu_pro": 14, "mmlu": 57, "minerva_math": 7, "bbh": 27}
TRAIN_BIN = 50  # training-log points are averaged over windows of this many steps

_cache = {}  # path -> (mtime, parsed)


def _cached(path, parse):
    try:
        mtime = os.stat(path).st_mtime
    except FileNotFoundError:
        return None
    hit = _cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        value = parse(path)
    except (OSError, ValueError):
        return None
    _cache[path] = (mtime, value)
    return value


def _read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # partially written last line
    return rows


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def load_registry(path=None):
    with open(path or os.path.join(HERE, "runs.yaml")) as f:
        reg = yaml.safe_load(f)
    root = os.environ.get("MIDTRAIN_ROOT") or reg["root"]
    for g in reg["groups"]:
        for r in g["runs"]:
            r["group"] = g["id"]
            r["path"] = r["dir"] if os.path.isabs(r["dir"]) else os.path.join(root, r["dir"])
    return reg


def collect_train(run_dir):
    rows = _cached(os.path.join(run_dir, "metrics.jsonl"), _read_jsonl) or []
    bins = defaultdict(lambda: [0.0, 0, None, None])  # bin -> [loss sum, n, lr, tokens]
    tokens_per_step, max_step = None, 0
    for d in rows:
        step, loss = d.get("global_step"), d.get("loss/out")
        if step is None:
            continue
        max_step = max(max_step, step)
        if d.get("optim/total_tokens") and step > 0:
            tokens_per_step = d["optim/total_tokens"] / step
        if loss is None or not math.isfinite(loss):
            continue
        b = bins[(step - 1) // TRAIN_BIN]
        b[0] += loss
        b[1] += 1
        b[2] = d.get("optim/lr")
        b[3] = step
    keys = sorted(bins)
    return {
        "step": [bins[k][3] for k in keys],
        "loss": [round(bins[k][0] / bins[k][1], 5) for k in keys],
        "lr": [bins[k][2] for k in keys],
    }, tokens_per_step, max_step


def collect_val(path):
    by_step = {}
    for d in _cached(path, _read_jsonl) or []:
        step = d.get("global_step")
        if step is None:
            continue
        doms = {}
        for k, v in d.items():
            if isinstance(v, dict) and "nll_per_token" in v:
                doms[k.removesuffix("_shuffled")] = {
                    "tok": round(-v["nll_per_token"], 5),
                    # nats/char -> bits/char (bytes for ASCII)
                    "char": round(-v["nll_per_char"] / math.log(2), 5) if "nll_per_char" in v else None,
                }
        prev = by_step.get(step)
        if doms and (prev is None or d.get("created_at", "") >= prev[0]):
            by_step[step] = (d.get("created_at", ""), doms)
    return [{"step": s, "domains": by_step[s][1]} for s in sorted(by_step)]


def _task_alias(path, m):
    alias = (m.get("task_config") or {}).get("metadata", {}).get("alias")
    if alias:
        return alias
    # older runs without metadata: name from the file, e.g. task-003-mmlu_anatomy:mc-metrics.json
    return re.sub(r"^task-\d+-|-metrics\.json$", "", os.path.basename(path))


def _family(alias):
    for fam in FAMILIES:
        mt = re.match(rf"^{fam}_(?P<sub>[a-z0-9_]+?)(?P<rest>(:.*)?)$", alias)
        if mt and not (fam == "mmlu" and mt.group("sub").startswith("pro_")):
            return fam, fam + mt.group("rest")
    return None, None


def collect_olmes(run_dir):
    # (alias, step) -> (mtime, score, bpb, primary_metric)
    found = {}
    for sub in OLMES_SUBDIRS:
        for step_dir in glob.glob(os.path.join(run_dir, sub, "[0-9]" * 10)):
            step = int(os.path.basename(step_dir))
            res = os.path.join(step_dir, "olmes_results")
            for f in glob.glob(os.path.join(res, "task-*-metrics.json")) + glob.glob(
                os.path.join(res, "shard_*", "task-*-metrics.json")
            ):
                m = _cached(f, _read_json)
                if not m or "metrics" not in m:
                    continue
                alias = _task_alias(f, m)
                met = m["metrics"]
                score = met.get("primary_score")
                bpb = met.get("bits_per_byte_corr")
                pm = (m.get("task_config") or {}).get("primary_metric", "primary_score")
                mtime = _cache[f][0]
                prev = found.get((alias, step))
                if prev is None or mtime > prev[0]:
                    found[(alias, step)] = (mtime, score, bpb, pm)

    tasks = defaultdict(lambda: {"metric": None, "points": {}})
    fam_acc = defaultdict(list)  # (family alias, step) -> [(score, bpb, metric)]
    for (alias, step), (_, score, bpb, pm) in found.items():
        _, fam = _family(alias)
        if fam:
            fam_acc[(fam, step)].append((score, bpb, pm))
            continue
        t = tasks[alias]
        t["metric"] = pm
        t["points"][step] = [score, bpb, None]
    for (fam, step), vals in fam_acc.items():
        scores = [v[0] for v in vals if v[0] is not None]
        bpbs = [v[1] for v in vals if v[1] is not None]
        t = tasks[fam]
        t["metric"] = f"{vals[0][2]} (macro avg)"
        t["points"][step] = [
            sum(scores) / len(scores) if scores else None,
            sum(bpbs) / len(bpbs) if bpbs else None,
            len(vals),  # number of subtasks averaged
        ]
    out = {}
    for alias, t in tasks.items():
        fam_n = FAMILIES.get(alias.split(":")[0]) if alias.split(":")[0] in FAMILIES else None
        out[alias] = {
            "metric": t["metric"],
            "expected_subtasks": fam_n,
            "points": [[s] + t["points"][s] for s in sorted(t["points"])],
        }
    return out


def _finite(x):
    """NaN / inf -> None so the output is valid JSON."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite(v) for v in x]
    return x


def to_json(data):
    return json.dumps(_finite(data), separators=(",", ":"), allow_nan=False)


def collect(registry_path=None):
    t0 = time.time()
    reg = load_registry(registry_path)
    runs, val_domains, task_names = {}, [], set()
    for g in reg["groups"]:
        for r in g["runs"]:
            path = r["path"]
            train, tps, max_step = collect_train(path)
            val = collect_val(os.path.join(path, r.get("val", "metrics.validation.jsonl")))
            olmes = collect_olmes(path)
            for rec in val:
                for d in rec["domains"]:
                    if d not in val_domains:
                        val_domains.append(d)
            task_names.update(olmes)
            runs[r["id"]] = {
                "id": r["id"], "label": r["label"], "group": g["id"], "color": r.get("color"),
                "path": path, "exists": os.path.isdir(path),
                "tokens_per_step": tps, "max_step": max_step,
                "train": train, "val": val, "olmes": olmes,
            }
    preferred = ["dclm", "math", "flan", "wiki", "pes2o", "stackexchange"]
    val_domains.sort(key=lambda d: (preferred.index(d) if d in preferred else len(preferred), d))
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "collect_seconds": round(time.time() - t0, 2),
        "groups": [
            {"id": g["id"], "title": g["title"], "description": (g.get("description") or "").strip(),
             "runs": [r["id"] for r in g["runs"]]}
            for g in reg["groups"]
        ],
        "runs": runs,
        "val_domains": val_domains,
        "tasks": sorted(task_names),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data = collect(args.registry)
    s = to_json(data)
    if args.out:
        with open(args.out, "w") as f:
            f.write(s)
    n_pts = sum(len(t["points"]) for r in data["runs"].values() for t in r["olmes"].values())
    print(f"{len(data['runs'])} runs, {len(data['tasks'])} tasks, {n_pts} OLMES points, "
          f"{len(s) / 1e6:.2f} MB, {data['collect_seconds']}s")
