"""Markdown table of the batch-size x LR sweep (setup/make_lrsweep_configs.py), one row per run and eval step.

Columns: Paloma BPB (mean over the 16 sources; n = sources present), and lingua validation loss
(nats/token) on dclm_heldout and fwedu_heldout. Reads <root>/<run>/metrics.{olmes_eval,validation}.jsonl.

    .venv-kd/bin/python analysis/results_viewer/lrsweep_table.py [--root ...] [--dataset dclm|fwedu]
"""

import argparse
import json
import os
import re

ROOT = "/checkpoint/mnm/sgeng/pretrain/lingua-runs/runs/midtrain/experiments/lrsweep"
NAME = re.compile(r"(dclm|fwedu)_bs(\d+)_lr([0-9p]+)e(\d+)(?:_seed(\d+))?$")


def records(path):
    """{global_step: record}; later lines for the same step win."""
    out = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            out[r["global_step"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--dataset", choices=["dclm", "fwedu"])
    a = ap.parse_args()
    runs = []
    for n in os.listdir(a.root):
        m = NAME.match(n)
        if m and (a.dataset is None or m[1] == a.dataset):
            lr = float(m[3].replace("p", ".")) * 10 ** -int(m[4])
            runs.append(((m[1], int(m[2]), lr, int(m[5] or 0)), n))
    print("| run | bs (seqs) | lr | seed | step | Paloma BPB (n) | dclm_heldout nll/tok | fwedu_heldout nll/tok |")
    print("|---|---|---|---|---|---|---|---|")
    for (ds, bs, lr, seed), n in sorted(runs):
        ol = records(f"{a.root}/{n}/metrics.olmes_eval.jsonl")
        va = records(f"{a.root}/{n}/metrics.validation.jsonl")
        steps = sorted(set(ol) | set(va))
        if not steps:
            print(f"| {n} | {bs} | {lr:.3g} | {seed} | — | no eval yet | | |")
        for s in steps:
            p = [v for k, v in ol.get(s, {}).items() if k.endswith("::paloma/primary_score")]
            pal = f"{sum(p) / len(p):.4f} ({len(p)})" if p else "—"
            v = va.get(s, {})
            nll = [f"{-v[k]['nll_per_token']:.4f}" if k in v else "—" for k in ("dclm_heldout", "fwedu_heldout")]
            print(f"| {n} | {bs} | {lr:.3g} | {seed} | {s} | {pal} | {nll[0]} | {nll[1]} |")


if __name__ == "__main__":
    main()
