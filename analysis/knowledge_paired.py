"""Paired per-item comparison on knowledge tasks (ARC-E, ARC-C, SciQ, MMLU minus math subjects): accuracy (acc_per_char)
and bits-per-byte of the correct answer, with bootstrap 95% CIs over items. Run from the experiment group dir."""
import glob, json, re, sys
import numpy as np
MATH = {"abstract_algebra", "college_mathematics", "elementary_mathematics", "high_school_mathematics", "high_school_statistics"}
def items(r, s):
    out = {}
    for f in glob.glob(f"{r}/evals/{s:010d}/olmes_results/shard_*/task-*-predictions.jsonl"):
        t = re.search(r"task-\d+-(.+)-predictions", f).group(1)
        if t in ("gsm8k", "hellaswag", "piqa", "winogrande") or (t.startswith("mmlu_") and t[5:] in MATH): continue
        grp = "MMLU (non-math)" if t.startswith("mmlu_") else t
        for l in open(f):
            x = json.loads(l); m = x["metrics"]
            out[(t, x["doc_id"])] = (grp, m["acc_per_char"], m["bits_per_byte_corr"])
    return out
rng = np.random.default_rng(0)
def ci(d):
    return np.quantile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)], [.025, .975])
def compare(a, b, s):
    A, B = items(a, s), items(b, s); keys = sorted(set(A) & set(B)); res = {}
    for grp in ["arc_easy", "arc_challenge", "sciq", "MMLU (non-math)", "ALL"]:
        ks = [k for k in keys if grp == "ALL" or A[k][0] == grp]
        da = np.array([A[k][1] - B[k][1] for k in ks]); db = np.array([A[k][2] - B[k][2] for k in ks])
        res[grp] = (len(ks), da.mean(), *ci(da), db.mean(), *ci(db))
    return res
if __name__ == "__main__":
    pairs = [l.split() for l in sys.stdin if l.strip()]
    for s, a, b in pairs:
        s = int(s); r = compare(a, b, s)
        print(f"\n{'3C' if s == 24588 else '2C'} {a} − {b}")
        for g, (n, acc, alo, ahi, bpb, blo, bhi) in r.items():
            sig = lambda lo, hi: "*" if lo > 0 or hi < 0 else " "
            print(f"  {g:16s} n={n:5d}  acc {acc:+.3f} [{alo:+.3f},{ahi:+.3f}]{sig(alo, ahi)}  ans-bpb {bpb:+.3f} [{blo:+.3f},{bhi:+.3f}]{sig(blo, bhi)}")
