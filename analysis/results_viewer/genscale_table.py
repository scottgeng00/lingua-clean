"""Final-step markdown table for genscale runs: dclm_heldout / OOD-5 PPL, OLMES accuracies, gold BPB.

    .venv/bin/python analysis/results_viewer/genscale_table.py r_lr5p3e3_wd01 mr_repeat_k2_lr5p3e3_wd01 ...

OOD-5 PPL = exp(mean per-token NLL over math / flan / wiki / pes2o / stackexchange). Gold BPB(5) = mean
bits_per_byte_corr over SciQ / ARC-E / PIQA / MMLU plus the TriviaQA :bpb task. Uses the latest
validation step of each run (19074 for finished runs).
"""
import json, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect import collect_olmes
G = "/checkpoint/dream-sc-3/scottg/pretrain/lingua-runs/runs/midtrain/experiments/genscale"
OOD = ["math_shuffled", "flan_shuffled", "wiki_shuffled", "pes2o_shuffled", "stackexchange_shuffled"]
ACC = [("sciq:rc::olmo3", "SciQ"), ("arc_easy:rc::olmes:full", "ARC-E"), ("piqa:rc::olmes:full", "PIQA"), ("mmlu:rc::olmes", "MMLU"), ("triviaqa::olmes", "TQA")]
runs = sys.argv[1:]
print("| run | step | heldout PPL | OOD-5 PPL | SciQ | ARC-E | PIQA | MMLU | TQA | avg acc | gold BPB(5) | GSM8K BPB | MBPP BPB |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in runs:
    d = os.path.join(G, r)
    vals = [json.loads(l) for l in open(f"{d}/metrics.validation.jsonl")]
    v = max(vals, key=lambda x: x["global_step"])
    step = v["global_step"]
    ppl = lambda k: math.exp(-v[k]["nll_per_token"])
    held = ppl("dclm_heldout")
    ood = math.exp(-sum(v[k]["nll_per_token"] for k in OOD) / len(OOD))
    o = collect_olmes(d)
    def at(alias, i):
        pts = {p[0]: p for p in o.get(alias, {}).get("points", [])}
        p = pts.get(step)
        return None if p is None else p[i]
    accs = [at(a, 1) for a, _ in ACC]
    bpbs = [at(a, 2) for a, _ in ACC[:4]] + [at("triviaqa:bpb::olmes", 1)]
    f = lambda x, s=100, n=1: "—" if x is None else f"{x*s:.{n}f}"
    avg = None if None in accs else sum(accs) / 5
    gb = None if None in bpbs else sum(bpbs) / 5
    print(f"| {r.replace('_lr5p3e3_wd01','')} | {step} | {held:.2f} | {ood:.2f} | " + " | ".join(f(a) for a in accs)
          + f" | {f(avg)} | {f(gb,1,3)} | {f(at('gsm8k:bpb::olmes',1),1,3)} | {f(at('mbpp:3shot:bpb::none',1),1,3)} |")
