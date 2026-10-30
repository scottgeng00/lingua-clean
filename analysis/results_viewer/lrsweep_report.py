"""Markdown report of the 600M LR x batch sweep (setup/make_lrsweep_configs.py) plus the Qwen3.5-9B KD alpha sweep.

Paloma = mean BPB over 16 sources scored without the EOT prefix (metrics.paloma_np.jsonl); held-out = lingua
validation loss (nats/token) at 20B tokens. Best LR per batch size by each metric.

    .venv-kd/bin/python analysis/results_viewer/lrsweep_report.py > report.md
"""
import datetime
import json
import os
import re

R = "/checkpoint/mnm/sgeng/pretrain/lingua-runs/runs/midtrain/experiments"
FINAL = {64: 76294, 128: 38147, 256: 19073}


def rec(path, step):
    if os.path.exists(path):
        for line in open(path):
            d = json.loads(line)
            if d["global_step"] == step:
                return d
    return {}


def paloma(d):
    p = [v for k, v in d.items() if k.endswith("::paloma/primary_score")]
    return sum(p) / 16 if len(p) == 16 else None


def main():
    f = lambda x: f"{x:.4f}" if x is not None else "pending"
    print("# 600M Qwen3: LR x batch sweep (20B tokens, DCLM / FineWeb-Edu) and Qwen3.5-9B KD alpha sweep\n")
    print(f"Generated {datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC. Paloma = mean BPB over 16 sources, scored WITHOUT "
          "the EOT prefix (seed spread ~0.001-0.002). Held-out = nats/token at 20B tokens, 10k docs.")
    print("Plots: https://meta-fair.wandb.io/sgeng/lingua-clean-lrsweep/runs/sweep-summary-plots\n")
    rows = []
    for n in sorted(os.listdir(f"{R}/lrsweep")):
        m = re.match(r"(dclm|fwedu)_bs(\d+)_lr([0-9p]+)e(\d+)(?:_seed(\d))?$", n)
        if not m:
            continue
        bs = int(m[2]); lr = float(m[3].replace("p", ".")) * 10 ** -int(m[4])
        v = rec(f"{R}/lrsweep/{n}/metrics.validation.jsonl", FINAL[bs])
        rows.append(dict(ds=m[1], bs=bs, lr=lr, seed=int(m[5] or 0), pal=paloma(rec(f"{R}/lrsweep/{n}/metrics.paloma_np.jsonl", FINAL[bs])),
                         dh=-v["dclm_heldout"]["nll_per_token"] if v else None,
                         fh=-v["fwedu_heldout"]["nll_per_token"] if v else None))
    for ds in ("dclm", "fwedu"):
        print(f"## trained on {ds}\n")
        print("| batch (seqs) | LR | seed | Paloma BPB | dclm_heldout | fwedu_heldout |")
        print("|---|---|---|---|---|---|")
        for r in sorted((r for r in rows if r["ds"] == ds), key=lambda r: (r["bs"], r["lr"], r["seed"])):
            print(f"| {r['bs']} | {r['lr']:.3g} | {r['seed']} | {f(r['pal'])} | {f(r['dh'])} | {f(r['fh'])} |")
        print("\n| batch | best LR by Paloma | Paloma | best LR by in-domain held-out | held-out |")
        print("|---|---|---|---|---|")
        key = "dh" if ds == "dclm" else "fh"
        for bs in (64, 128, 256):
            sub = [r for r in rows if r["ds"] == ds and r["bs"] == bs and r["seed"] == 0]
            p = min((r for r in sub if r["pal"]), key=lambda r: r["pal"], default=None)
            h = min((r for r in sub if r[key]), key=lambda r: r[key], default=None)
            if p and h:
                print(f"| {bs} | {p['lr']:.3g} | {p['pal']:.4f} | {h['lr']:.3g} | {h[key]:.4f} |")
        print()
    print("## KD from Qwen3.5-9B-Base (alpha * KL + (1 - alpha) * CE), dclm_8m_0+1, 20B tokens, LR 2.65e-3\n")
    print("| alpha | Paloma BPB (no prefix) |")
    print("|---|---|")
    for a in ["0p1", "0p3", "0p5", "0p7", "0p9"]:
        print(f"| {a.replace('p', '.')} | {f(paloma(rec(f'{R}/gb300/kd9bbase_a{a}/metrics.paloma_np.jsonl', 19074)))} |")
    best = min((r for r in rows if r["ds"] == "dclm" and r["pal"]), key=lambda r: r["pal"])
    print(f"\nBest 20B DCLM pretraining run by Paloma: batch {best['bs']}, LR {best['lr']:.3g} -> {best['pal']:.4f}.")
    print("lingua:v2 task results for the KD runs: analysis/results_viewer/v2_table.py kd9bbase_a0p{1,3,5,7,9}")


if __name__ == "__main__":
    main()
