"""Markdown report of the data-constrained Qwen3 ladder (setup/make_dataconstrained_configs.py runs).

    .venv-kd/bin/python analysis/results_viewer/dataconstrained_report.py > report.md
Metrics: val CE = dclm_heldout cross-entropy (nats/token, token-weighted, packed + document-masked as in training);
Paloma = mean BPB over 16 sources, no EOT prefix. Selection follows the sweep doc (val CE).
"""
import datetime
import os
import sys
from collections import defaultdict

sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataconstrained_grid as G  # noqa: E402

M = G.M
f4 = lambda x: f"{x:.4f}" if x is not None else "-"


def runs(width, stage, final=False):
    return {n: e for n, e in M.items() if e["width"] == width and e["stage"] == stage and ("final" in n) == final
            and not n.startswith(f"w{width}_check")}


def grid(width, stage, wd, point, metric="val_ce"):
    rs = [e for n, e in runs(width, stage).items() if e["wd"] == wd and e["eval_point"] == point and e["seed"] == 0]
    if not rs:
        return
    lrs = sorted({e["lr"] for e in rs}); bss = sorted({e["bs"] for e in rs})
    v = {(e["lr"], e["bs"]): G.metric(e["name"], metric) for e in rs}
    best = min((x, k) for k, x in v.items() if x)
    print(f"\n**weight decay {wd:g}, {point}** ({'val CE' if metric == 'val_ce' else 'Paloma BPB'}; best in bold)\n")
    print("| batch \\ LR | " + " | ".join(f"{l:g}" for l in lrs) + " |")
    print("|---|" + "---|" * len(lrs))
    for b in bss:
        cells = []
        for l in lrs:
            x = v.get((l, b))
            cells.append("" if x is None else (f"**{x:.4f}**" if (l, b) == best[1] else f"{x:.4f}"))
        print(f"| {b} | " + " | ".join(cells) + " |")


def stage1(width):
    print(f"\n### Width {width}, stage 1 (unique data; 2x = main run, 1x = decay branch)")
    for wd in sorted({e["wd"] for e in runs(width, 1).values()}):
        for pt in ("2x", "d1x"):
            grid(width, 1, wd, pt)
    for pt in ("d1x", "2x"):
        c = [(G.metric(n, "val_ce"), G.metric(n, "paloma"), e) for n, e in runs(width, 1).items() if e["eval_point"] == pt]
        v, p, e = min((x for x in c if x[0]), key=lambda x: x[0])
        print(f"\nBest at {'1x' if pt == 'd1x' else '2x'}: LR {e['lr']:g}, batch {e['bs']}, wd {e['wd']:g} -> val CE {v:.4f}, Paloma {f4(p)}")


def stage2(width):
    rs = runs(width, 2)
    print(f"\n### Width {width}, stage 2 (16 passes over the {'125.8M' if width == 512 else '1.274B'}-token pool)")
    best = defaultdict(lambda: (9, None, None))
    per_pass = defaultdict(list)
    for n, e in rs.items():
        if e["seed"]:
            continue
        v = G.metric(n, "val_ce")
        if v is None:
            continue
        k = (e["lr"], e["bs"], e["wd"])
        per_pass[e["eval_point"]].append((v, k))
        if v < best[k][0]:
            best[k] = (v, e["eval_point"], G.metric(n, "paloma"))
    print("\nTop configurations by best val CE over evaluated passes (passes 1, 2, 3, 4, 6, 8, 12, 16):\n")
    print("| rank | LR | batch | wd | best val CE | at pass | Paloma there |")
    print("|---|---|---|---|---|---|---|")
    for i, (k, (v, p, pal)) in enumerate(sorted(best.items(), key=lambda x: x[1][0])[:10], 1):
        print(f"| {i} | {k[0]:g} | {k[1]} | {k[2]:g} | {v:.4f} | {p[1:]} | {f4(pal)} |")
    print(f"\n({len(best)} configurations.) Best configuration at each pass count:\n")
    print("| pass | LR | batch | wd | val CE |")
    print("|---|---|---|---|---|")
    for p in sorted(per_pass, key=lambda s: int(s[1:])):
        v, k = min(per_pass[p])
        print(f"| {p[1:]} | {k[0]:g} | {k[1]} | {k[2]:g} | {v:.4f} |")


def final(width, base):
    print(f"\n### Width {width}, final run ({base.split('final_')[1].replace('_', ', ')}), decay branch after every pass\n")
    print("| pass | " + " | ".join(str(k) for k in range(1, 17)) + " |")
    print("|---|" + "---|" * 16)
    for s in (0, 1):
        for metric, label in (("val_ce", "val CE"), ("paloma", "Paloma")):
            vals = []
            for k in range(1, 17):
                n = (base + ("_seed1" if s else "")) if k == 16 else f"{base}{'_seed1' if s else ''}_p{k}"
                vals.append(G.metric(n, metric))
            print(f"| {label} seed {s} | " + " | ".join(f"{x:.3f}" if x else "-" for x in vals) + " |")


def main():
    print("# Data-constrained pretraining, first sweep: Qwen3 ladder (widths 512 and 1152) on DCLM\n")
    print(f"Generated {datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC from {len(M)} runs (main runs + decay branches), "
          "all trained and evaluated.\n")
    print("- Models: Qwen3 dense, MHA, head 128, SwiGLU 4x, untied embeddings, Qwen3 tokenizer (vocab 151,936), ctx 4096, "
          "document masking. Width 512: 6 layers, 25.17M body params; width 1152: 12 layers, 254.8M.")
    print("- Data: DCLM (dclm_8m_0, then dclm_8m_1), Qwen3-tokenized in the upload's row order. Stage 1: unique stream to "
          "2x Chinchilla. Stage 2: pool = first 0.25x Chinchilla tokens (125.8M / 1.274B, whole sequences), reshuffled + "
          "repacked every pass, 16 passes.")
    print("- AdamW (0.9, 0.95), eps 1e-8, clip 1.0, wd on all matrices + output head (not norms / input embedding). WSD: "
          "warmup 1 token per body param, 1-sqrt decay over the last 10% of tokens to each eval point (earlier points = "
          "decay branches from the main run's checkpoint).")
    print("- val CE = dclm_heldout (10k docs, 11.2M tokens) cross-entropy, nats/token, token-weighted, packed with document "
          "masking as in training. Paloma = mean BPB over 16 sources, no EOT prefix. Selection metric: val CE.\n")
    print("## Summary\n")
    print("| width | stage | best config (val CE) | val CE | Paloma |")
    print("|---|---|---|---|---|")
    for w in (512, 1152):
        for st, pts in ((1, ("d1x", "2x")), (2, None)):
            if st == 1:
                for pt in pts:
                    c = [(G.metric(n, "val_ce"), n, e) for n, e in runs(w, 1).items() if e["eval_point"] == pt]
                    v, n, e = min((x for x in c if x[0]), key=lambda x: x[0])
                    print(f"| {w} | 1 @ {'1x' if pt == 'd1x' else '2x'} | LR {e['lr']:g}, batch {e['bs']}, wd {e['wd']:g} | {v:.4f} | {f4(G.metric(n, 'paloma'))} |")
            else:
                c = [(G.metric(n, "val_ce"), n, e) for n, e in runs(w, 2).items() if e["seed"] == 0]
                v, n, e = min((x for x in c if x[0]), key=lambda x: x[0])
                print(f"| {w} | 2 (best over passes, pass {e['eval_point'][1:]}) | LR {e['lr']:g}, batch {e['bs']}, wd {e['wd']:g} | {v:.4f} | {f4(G.metric(n, 'paloma'))} |")
    print()
    for w in (512, 1152):
        a = min(G.metric(n, "val_ce") for n, e in runs(w, 1).items() if e["eval_point"] == "d1x")
        b = min(G.metric(n, "val_ce") for n, e in runs(w, 2).items() if e["eval_point"] == "p4" and e["seed"] == 0)
        print(f"- Cross-stage, width {w}: best stage-1 val CE at 1x Chinchilla {a:.4f} vs best stage-2 at 4 passes (same "
              f"token count) {b:.4f}: four passes cost {b - a:+.4f} nats/token.")
    print("- Noise: rerunning the width-512 stage-2 winner with the same seed gave 3.585 vs 3.576 (nondeterminism amplified "
          "at high LR early in training); seed 1 gave 3.577. Treat width-512 differences below ~0.01 as ties. Width 1152 "
          "final seeds agree to 0.002.")
    print("- Branch equivalence check (width 512): 2-pass branch val CE 3.9041 vs independent 2-pass run 3.9029 (doc "
          "tolerance 0.002).")
    print("\n## Stage 1")
    stage1(512); stage1(1152)
    print("\n## Stage 2")
    stage2(512); stage2(1152)
    print("\n## Final runs (repetition curves)")
    final(512, "w512_s2final_bs64_lr6e-3_wd0p3"); final(1152, "w1152_s2final_bs128_lr1p5e-3_wd0p3")
    print("\nPer-(run, eval point) table with every column of the doc: "
          "`analysis/results_viewer/dataconstrained_table.py --csv out.csv` (lingua-clean).")


if __name__ == "__main__":
    main()
