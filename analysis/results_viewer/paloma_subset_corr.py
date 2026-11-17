"""Which Paloma sources track which downstream evals, beyond overall loss?

Models: every run with an OLMES suite eval and per-source Paloma BPB: data_constrainedv2 (DCLM, FineWeb-Edu,
repetition; Paloma from metrics.paloma_np.jsonl), kd_w1152 (validation.json) and jhe_pt17b (token_stream_scores.jsonl).
1. Partial correlation over all models: per-source BPB vs downstream metric, both residualized on the mean BPB of the
   other 15 sources (leave-one-out, so the control does not contain the source itself).
2. Fixed-compute differences: (FineWeb-Edu - DCLM twin) pairs, (mix - real) for the pt17b models, (KD - CE) for
   kd_w1152; same partial correlation on the differences, controlling for the leave-one-out Paloma-mean difference.
All metrics are oriented lower = better (errors, BPB), so positive r = the two improve together.

    .venv-kd/bin/python analysis/results_viewer/paloma_subset_corr.py [--out temp/paloma_subset_corr.png] [--wandb]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataconstrained_lr_v2 import ROOT, RUNS
from olmes_vs_val import QA

X = "/home/sgeng/scottg/pretrain/lingua-runs/runs/midtrain/experiments"
SOURCES = ["4chan_meta_sep", "c4_100_domains", "c4_en", "dolma-v1_5", "dolma_100_programing_languages",
           "dolma_100_subreddits", "falcon-refinedweb", "gab", "m2d2_s2orc_unsplit", "m2d2_wikipedia_unsplit",
           "manosphere_meta_sep", "mc4", "ptb", "redpajama", "twitterAAE_HELM_fixed", "wikitext_103"]
EVALS = [("qa", "QA err"), ("arc_e", "ARC-E err"), ("arc_c", "ARC-C err"), ("hs", "HSwag err"), ("piqa", "PIQA err"),
         ("wino", "Wino err"), ("sciq", "SciQ err"), ("mmlu", "MMLU err"), ("mmlu_bpb", "MMLU BPB"),
         ("math", "math BPB"), ("he", "HumanEval BPB"), ("mbpp", "MBPP BPB")]


def downstream(paths):
    t = {}
    for p in paths:
        t.update({x["alias"]: x["metrics"]["primary_score"] for x in json.load(open(p))["tasks"]})
    err = lambda a: 100 - 100 * t[a]
    return dict(qa=np.mean([err(a) for a in QA]), arc_e=err(QA[0]), arc_c=err(QA[1]), hs=err(QA[2]), piqa=err(QA[3]),
                wino=err(QA[4]), sciq=err(QA[5]), mmlu=err("mmlu:rc::olmes"), mmlu_bpb=t["mmlu:rc:bpb::olmes"],
                math=t["olmo3:base_easy:math_bpb"], he=t["codex_humaneval:3shot:bpb::none"],
                mbpp=t["mbpp:3shot:bpb::none"])


def load():
    rows = {}
    M = json.load(open(f"{ROOT}/apps/main/configs/experiments/data_constrainedv2/manifest.json"))
    for n, e in M.items():
        o = glob.glob(f"{RUNS}/{n}/evals/*_olmes/olmes_results/metrics.json")
        f = f"{RUNS}/{n}/metrics.paloma_np.jsonl"
        if not o or not os.path.exists(f):
            continue
        pal = None
        for line in open(f):
            r = json.loads(line)
            if r["global_step"] == e["steps"]:
                pal = {s: r.get(f"olmes_eval/paloma_{s}::paloma/primary_score") for s in SOURCES}
        if pal and None not in pal.values():
            rows[n] = dict(pal=pal, **downstream(o))
    for r in ("b_ce", "b_kd0p1", "b_kd0p3", "b_kd0p5", "b_kd0p7", "b_kd0p9"):
        v = json.load(open(f"{X}/kd_w1152/{r}/evals/0000016398/validation.json"))
        rows[f"kd/{r}"] = dict(pal={s: v[f"paloma_qwen3/{s}"]["bpb"] for s in SOURCES},
                               **downstream(glob.glob(f"{X}/kd_w1152/{r}/evals/*/olmes_results/metrics.json")))
    for line in open(f"{X}/jhe_pt17b/token_stream_scores.jsonl"):
        r = json.loads(line)
        name = r["ckpt"].split("/jhe_pt17b/")[1].split("/")[0]
        rows[f"jhe/{name}"] = dict(pal={s: r["results"][f"paloma_qwen3/{s}"]["bpb"] for s in SOURCES},
                                   **downstream(glob.glob(f"{X}/jhe_pt17b/{name}/evals/*_olmes/olmes_results/metrics.json")))
    return rows


def diffs(rows):
    pairs = [(n.replace("_fe_", "_u_"), n) for n in rows if "_fe_" in n and n.replace("_fe_", "_u_") in rows]
    pairs += [("jhe/real", f"jhe/{m}") for m in ("mix_fs_gacha", "mix_fs_direct", "mix_rw_gacha", "mix_rw_direct")]
    pairs += [("kd/b_ce", f"kd/{r}") for r in ("b_kd0p1", "b_kd0p3", "b_kd0p5", "b_kd0p7", "b_kd0p9")]
    out = []
    for a, b in pairs:
        d = dict(pal={s: rows[b]["pal"][s] - rows[a]["pal"][s] for s in SOURCES})
        d.update({k: rows[b][k] - rows[a][k] for k, _ in EVALS})
        out.append(d)
    return out, len(pairs)


def partial(rows):
    """[source, eval] partial r controlling for the leave-one-out Paloma mean, plus raw r."""
    P = np.array([[r["pal"][s] for s in SOURCES] for r in rows])
    Y = np.array([[r[k] for k, _ in EVALS] for r in rows])
    pr, raw = np.zeros((len(SOURCES), len(EVALS))), np.zeros((len(SOURCES), len(EVALS)))
    for i in range(len(SOURCES)):
        c = np.delete(P, i, 1).mean(1)
        A = np.stack([np.ones_like(c), c], 1)
        res = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
        rx = res(P[:, i])
        for j in range(len(EVALS)):
            pr[i, j] = np.corrcoef(rx, res(Y[:, j]))[0, 1]
            raw[i, j] = np.corrcoef(P[:, i], Y[:, j])[0, 1]
    return pr, raw


def heat(ax, m, title, ylabels=True):
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(EVALS)), [l for _, l in EVALS], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(SOURCES)), SOURCES if ylabels else [], fontsize=9)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(m[i, j]) > 0.6 else "#222")
    ax.set_title(title, fontsize=11)
    return im


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/temp/paloma_subset_corr.png")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    rows = load()
    pr_all, raw_all = partial(list(rows.values()))
    D, npairs = diffs(rows)
    pr_d, raw_d = partial(D)
    fig, axes = plt.subplots(1, 3, figsize=(28, 9), gridspec_kw={"wspace": 0.05})
    heat(axes[0], raw_all, f"(a) raw r, all {len(rows)} models\n(everything scales together)")
    heat(axes[1], pr_all, f"(b) partial r, all models, controlling for\nPaloma mean of the other 15 sources", False)
    im = heat(axes[2], pr_d, f"(c) partial r over {npairs} fixed-compute differences\n(93 FW-Edu−DCLM, 4 mix−real, 5 KD−CE)", False)
    fig.colorbar(im, ax=axes, shrink=0.6, label="r (positive = improve together; all metrics lower = better)")
    fig.suptitle("Paloma source BPB vs downstream (errors / BPB)", fontsize=14)
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print(a.out, len(rows), "models,", npairs, "differences")
    for name, m in (("all models", pr_all), ("differences", pr_d)):
        print(f"\nTop partial r ({name}):")
        flat = sorted(((m[i, j], SOURCES[i], EVALS[j][1]) for i in range(len(SOURCES)) for j in range(len(EVALS))),
                      reverse=True)
        for v, s, e in flat[:12]:
            print(f"  {v:+.2f}  {s:32s} {e}")
        print("  ... most negative:", ", ".join(f"{s}/{e} {v:+.2f}" for v, s, e in flat[-4:]))
    if a.wandb:
        import wandb
        run = wandb.init(project="lingua-clean-data_constrainedv2", name="paloma-subset-corr", id="paloma-subset-corr",
                         resume="allow")
        run.log({"paloma_subset_corr": wandb.Image(a.out)})
        run.finish()
