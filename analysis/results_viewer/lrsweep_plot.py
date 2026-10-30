"""Plot the batch-size x LR sweep (setup/make_lrsweep_configs.py) at 20B tokens: rows = training data, columns =
in-distribution held-out loss and Paloma BPB; one curve per batch size vs LR (log x). Extra seeds are hollow markers;
the curve goes through their mean.

    .venv-kd/bin/python analysis/results_viewer/lrsweep_plot.py --out lrsweep.png [--exclude gab twitterAAE_HELM_fixed]

--exclude drops Paloma sources (task names without the paloma_ prefix) from the mean. Paloma comes from
metrics.paloma_np.jsonl (apps/main/rescore_paloma.py, no EOT prefix) unless --paloma-file says otherwise.
"""

import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/checkpoint/mnm/sgeng/pretrain/lingua-runs/runs/midtrain/experiments/lrsweep"
FINAL = {64: 76294, 128: 38147, 256: 19073}
NAME = re.compile(r"(dclm|fwedu)_bs(\d+)_lr([0-9p]+)e(\d+)(?:_seed(\d+))?$")
COLORS = {64: "tab:blue", 128: "tab:orange", 256: "tab:green"}


def final_record(path, step):
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r["global_step"] == step:
                return r
    return None


def collect(root, exclude=(), paloma_file="metrics.paloma_np.jsonl"):
    """{(dataset, metric): {bs: {lr: [values over seeds]}}} at the final step."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for n in os.listdir(root):
        m = NAME.match(n)
        if not m:
            continue
        ds, bs = m[1], int(m[2])
        lr = float(m[3].replace("p", ".")) * 10 ** -int(m[4])
        v = final_record(f"{root}/{n}/metrics.validation.jsonl", FINAL[bs])
        if v:
            out[(ds, "heldout")][bs][lr].append(-v[f"{'dclm' if ds == 'dclm' else 'fwedu'}_heldout"]["nll_per_token"])
        o = final_record(f"{root}/{n}/{paloma_file}", FINAL[bs])
        p = {k.split("/")[1].split("::")[0].removeprefix("paloma_"): x for k, x in (o or {}).items()
             if k.endswith("::paloma/primary_score")}
        if len(p) == 16:
            kept = [x for k, x in p.items() if k not in exclude]
            out[(ds, "paloma")][bs][lr].append(sum(kept) / len(kept))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default="lrsweep.png")
    ap.add_argument("--exclude", nargs="*", default=[], help="Paloma sources to leave out of the mean")
    ap.add_argument("--paloma-file", default="metrics.paloma_np.jsonl",
                    help="metrics.olmes_eval.jsonl for the original EOT-prefixed Paloma scores")
    a = ap.parse_args()
    data = collect(a.root, set(a.exclude), a.paloma_file)
    n_paloma = 16 - len(a.exclude)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for i, ds in enumerate(("dclm", "fwedu")):
        for j, metric in enumerate(("heldout", "paloma")):
            ax = axes[i, j]
            for bs in (64, 128, 256):
                pts = sorted(data[(ds, metric)][bs].items())
                if not pts:
                    continue
                ax.plot([lr for lr, _ in pts], [sum(v) / len(v) for _, v in pts], "o-", color=COLORS[bs],
                        label=f"batch {bs} ({bs * 4096 / 1e6:.2f}M tok)")
                for lr, v in pts:
                    if len(v) > 1:
                        ax.scatter([lr] * len(v), v, facecolors="none", edgecolors=COLORS[bs], s=60, zorder=3)
            ax.set_xscale("log")
            ax.set_xlabel("peak LR")
            ylabel = f"{ds}_heldout loss (nats/token)" if metric == "heldout" else f"Paloma BPB (mean of {n_paloma})"
            ax.set_ylabel(ylabel)
            ax.set_title(f"trained on {'DCLM' if ds == 'dclm' else 'FineWeb-Edu'}: {ylabel.split(' (')[0]}")
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=8)
    excl = ("; Paloma: no EOT prefix" if "paloma_np" in a.paloma_file else "; Paloma: EOT-prefixed")
    excl += f", excludes {', '.join(a.exclude)}" if a.exclude else ""
    fig.suptitle(f"600M Qwen3, 20B tokens, WSD (1-sqrt decay over last 20%). Hollow = extra seeds; missing = pending{excl}", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(a.out)


if __name__ == "__main__":
    main()
