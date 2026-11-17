"""Coverage of each Paloma source by each pretraining corpus, from reverse_nn.py outputs (each corpus chunk -> its
nearest Paloma chunk). A Paloma chunk is "hit" if it is the nearest neighbour of at least one corpus chunk (optionally
with cosine >= t). Variants: full corpora; size-matched (random 17.17M chunks = DCLM 2x size); and mixes mirroring the
pt17b models (random 80% of DCLM + rewrite chunks equal to 20% of the DCLM size). Then: per source, does the coverage
gain of a mix over DCLM predict its Paloma BPB gain (pt17b mix - real)?

    .venv-kd/bin/python scripts/embed/reverse_nn_report.py --rnn .../embeddings/reverse_nn > report.md
"""
import argparse
import json

import numpy as np

CORPORA = ["dclm_2x_w1152", "rw_gacha", "rw_direct"]
LABEL = {"dclm_2x_w1152": "DCLM 2×", "rw_gacha": "rw_gacha", "rw_direct": "rw_direct"}
SCORES = "/home/sgeng/scottg/pretrain/lingua-runs/runs/midtrain/experiments/jhe_pt17b/token_stream_scores.jsonl"


def stats(idx, sim, n_keys, ts=(0.7, 0.8)):
    hits = np.bincount(idx, minlength=n_keys)
    out = {"cov": (hits > 0).mean()}
    for t in ts:
        out[f"cov{t}"] = (np.bincount(idx[sim >= t], minlength=n_keys) > 0).mean()
    s = np.sort(hits)[::-1]
    out["top1pct_share"] = s[:max(1, n_keys // 100)].sum() / max(1, s.sum())
    out["median_hits"] = float(np.median(hits))
    out["sim_mean"], out["sim_p90"] = float(sim.mean()), float(np.quantile(sim, 0.9))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rnn", required=True)
    a = ap.parse_args()
    sizes = json.load(open(f"{a.rnn}/{CORPORA[0]}/paloma_sizes.json"))
    srcs = sorted(sizes) + ["pooled"]
    sizes["pooled"] = sum(sizes.values())
    data = {c: {s: (np.load(f"{a.rnn}/{c}/{s}.idx.npy", mmap_mode="r"), np.load(f"{a.rnn}/{c}/{s}.sim.npy", mmap_mode="r"))
                for s in srcs} for c in CORPORA}
    n = {c: len(data[c]["pooled"][0]) for c in CORPORA}
    m = n["dclm_2x_w1152"]
    rng = np.random.default_rng(0)
    sel = {c: np.sort(rng.choice(n[c], m, replace=False)) if n[c] > m else None for c in CORPORA}
    mix_d = np.sort(rng.choice(m, int(0.8 * m), replace=False))
    mix_r = {c: np.sort(rng.choice(n[c], int(0.2 * m), replace=False)) for c in CORPORA[1:]}
    variants = {}
    for c in CORPORA:
        variants[f"{LABEL[c]} (full, {n[c] / 1e6:.1f}M)"] = {s: (data[c][s][0], data[c][s][1]) for s in srcs}
    for c in CORPORA[1:]:
        variants[f"{LABEL[c]} (matched, {m / 1e6:.1f}M)"] = {s: (data[c][s][0][sel[c]], data[c][s][1][sel[c]]) for s in srcs}
    for c in CORPORA[1:]:
        variants[f"mix 80% DCLM + 20% {LABEL[c]}"] = {
            s: (np.concatenate([data["dclm_2x_w1152"][s][0][mix_d], data[c][s][0][mix_r[c]]]),
                np.concatenate([data["dclm_2x_w1152"][s][1][mix_d], data[c][s][1][mix_r[c]]])) for s in srcs}
    R = {v: {s: stats(np.asarray(i), np.asarray(x, dtype=np.float32), sizes[s]) for s, (i, x) in d.items()}
         for v, d in variants.items()}

    print("## All of Paloma pooled (170,559 chunks)\n")
    print("| corpus | Paloma chunks hit | hit with cos ≥ 0.7 | hit with cos ≥ 0.8 | share of hits on top 1% of Paloma chunks | "
          "mean cos to nearest Paloma chunk | p90 cos |\n|---|---|---|---|---|---|---|")
    for v in R:
        r = R[v]["pooled"]
        print(f"| {v} | {r['cov']:.3f} | {r['cov0.7']:.3f} | {r['cov0.8']:.3f} | {r['top1pct_share']:.3f} | "
              f"{r['sim_mean']:.3f} | {r['sim_p90']:.3f} |")
    for key, title in (("cov", "fraction of chunks hit (any distance)"), ("cov0.7", "fraction hit with cos ≥ 0.7")):
        cols = [v for v in R if "full" in v or "matched" in v]
        print(f"\n## Per Paloma source: {title}\n")
        print("| source | chunks | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 2))
        for s in srcs[:-1]:
            vals = [R[v][s][key] for v in cols]
            b = int(np.argmax(vals))
            print(f"| {s} | {sizes[s]:,} | " + " | ".join(f"**{x:.3f}**" if i == b else f"{x:.3f}"
                                                         for i, x in enumerate(vals)) + " |")
    # coverage gain of a mix vs BPB gain of the matching pt17b model
    S = {}
    for line in open(SCORES):
        r = json.loads(line)
        S[r["ckpt"].split("/jhe_pt17b/")[1].split("/")[0]] = r["results"]
    base = f"{LABEL['dclm_2x_w1152']} (full, {m / 1e6:.1f}M)"
    print("\n## Hypothesis: more coverage → lower BPB (per source; pt17b mix − real)\n")
    print("| source | " + " | ".join(f"Δcov@0.7 {a} | ΔBPB mix_{a}" for a in ("rw_gacha", "rw_direct")) + " |\n|---|---|---|---|---|")
    xs, ys = {k: [] for k in ("cov", "cov0.7", "cov0.8")}, []
    for s in srcs[:-1]:
        cells = []
        for c in CORPORA[1:]:
            mv = f"mix 80% DCLM + 20% {LABEL[c]}"
            db = S[f"mix_{c}"][f"paloma_qwen3/{s}"]["bpb"] - S["real"][f"paloma_qwen3/{s}"]["bpb"]
            for k in xs:
                xs[k].append(R[mv][s][k] - R[base][s][k])
            ys.append(db)
            cells += [f"{R[mv][s]['cov0.7'] - R[base][s]['cov0.7']:+.3f}", f"{db:+.3f}"]
        print(f"| {s} | " + " | ".join(cells) + " |")
    ys = np.array(ys)
    print("\nCorrelation over the 32 (source, arm) points between coverage gain and BPB change "
          "(negative = more coverage, lower BPB): " + ", ".join(f"{k}: r = {np.corrcoef(xs[k], ys)[0, 1]:+.2f}" for k in xs))


if __name__ == "__main__":
    main()
