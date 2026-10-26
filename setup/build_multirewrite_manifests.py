"""Simplescale manifests for the multi-rewrite study: k rewrites per doc of the first 4M rows of dclm_8m_1.

S1 = dclm_8m_0 (seen once); the S2 slot is either 2 views of rows [0, 4M) or 4 views of rows
[0, 2M). So rows < 2M need views 0..3 and rows in [2M, 4M) need views 0..1. View 0 of iid / nemo /
wofmt is the existing full-8M Qwen3.5-9B wiki rewrite (dclm_8m_1_qwen35_9b_wiki_g1_max4096), so those
methods only generate views 1..3; t12 and gacha generate all four. One manifest per method, task j
-> dclm_8m_1 row j (same _ref layout as setup/build_dclm_rewrite_manifest.py), with per-task:

  iid    rewrite:wiki,   generations 3 / 1, T=1                      -> views 1..3 / 1
  t12    rewrite:wiki,   generations 4 / 2, T=1.2 (no top_p / min_p)  -> views 0..3 / 0..1
  gacha  gacha:rewrite_wiki, generations 4 / 2, T=1 (dice seeded by doc id, so view i is the same
         dice roll in both row ranges)                               -> views 0..3 / 0..1
  nemo   rewrite:nemotron, views [diverse_qa, distill, knowledge_list] / [diverse_qa] -> views 1..3 / 1
  wofmt  rewrite:wofmt, 3 / 1 WebOrganizer formats                   -> views 1..3 / 1

wofmt formats: drawn per doc without replacement, rng seeded by the doc id, with probability
proportional to S2's WebOrganizer format distribution (NoURL classifier, mean probs over 5,120 S2
docs; analysis/data_quality/measure.py report.json), excluding Truncated, Spam / Ads, Legal
Notices and Knowledge Article (the wiki view) and renormalized. Rows >= 2M take the first of the
same three draws, so the k=2 view is the same format whichever row range a doc is in.

    python setup/build_multirewrite_manifests.py
"""

import argparse
import hashlib
import json
import os

import numpy as np
import orjson

SUBSET = "/home/sgeng/scottg/pretrain/assets/data/dclm_subsets/dclm_8m_1"
OUT = "/checkpoint/dream-sc-3/scottg/pretrain/simplescale/runs/manifests"
REPORT = "/checkpoint/dream-sc-3/scottg/pretrain/lingua-runs/analysis/rewrite_quality/report.json"
EXCLUDE = {"Truncated", "Spam / Ads", "Legal Notices", "Knowledge Article"}
N_CHUNKS, ROWS, ROWS_K4 = 16, 4_000_000, 2_000_000
METHODS = ("iid", "t12", "gacha", "nemo", "wofmt")


def format_rates():
    dist = json.load(open(REPORT))["S2"]["format"]
    keep = {k: v for k, v in dist.items() if k not in EXCLUDE}
    z = sum(keep.values())
    return {k: v / z for k, v in sorted(keep.items(), key=lambda x: -x[1])}


def task_fields(method, doc_id, k4, rates, max_new, max_doc):
    base = {"max_document_tokens": max_doc}
    sp = {"temperature": 1.0, "max_new_tokens": max_new}
    if method == "iid":
        return {**base, "generations": 3 if k4 else 1, "sampling_params": sp}
    if method == "t12":
        return {**base, "generations": 4 if k4 else 2, "sampling_params": {**sp, "temperature": 1.2}}
    if method == "gacha":
        return {**base, "generations": 4 if k4 else 2, "sampling_params": sp, "planning_max_new_tokens": 2048}
    if method == "nemo":
        return {**base, "views": ["diverse_qa", "distill", "knowledge_list"] if k4 else ["diverse_qa"],
                "sampling_params": sp}
    if method == "wofmt":
        seed = int.from_bytes(hashlib.blake2b(doc_id.encode(), digest_size=8).digest(), "little")
        labels, p = list(rates), np.array(list(rates.values()))
        views = [labels[i] for i in np.random.default_rng(seed).choice(len(labels), 3, replace=False, p=p)]
        return {**base, "views": views if k4 else views[:1], "sampling_params": sp}
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--rows", type=int, default=ROWS)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--max-document-tokens", type=int, default=11000)
    ap.add_argument("--tag", default="rewrite_first4m", help="manifest name: dclm_8m_1_<tag>.<method>.manifest.jsonl")
    args = ap.parse_args()

    name = os.path.basename(SUBSET)
    files = [os.path.join(SUBSET, f"{name}.chunk.{c:02d}.jsonl") for c in range(N_CHUNKS)]
    rates = format_rates()
    os.makedirs(OUT, exist_ok=True)
    json.dump(rates, open(os.path.join(OUT, f"{name}_{args.tag}.wofmt_rates.json"), "w"), indent=1)
    paths = {m: os.path.join(OUT, f"{name}_{args.tag}.{m}.manifest.jsonl") for m in args.methods}
    outs = {m: open(p + ".tmp", "w") for m, p in paths.items()}
    readers = [open(f, "rb") for f in files]
    offsets = [0] * N_CHUNKS
    for j in range(args.rows):
        c = j % N_CHUNKS
        line = readers[c].readline()
        assert line, f"ran out of rows at {j}"
        ref = {"path": files[c], "offset": offsets[c], "length": len(line)}
        offsets[c] += len(line)
        doc_id = orjson.loads(line)["id"]
        for m, f in outs.items():
            fields = task_fields(m, doc_id, j < ROWS_K4, rates, args.max_new_tokens, args.max_document_tokens)
            f.write(json.dumps({"_ref": ref, **fields}, separators=(",", ":")) + "\n")
    for m, f in outs.items():
        f.close()
        os.replace(paths[m] + ".tmp", paths[m])
        print(f"{args.rows} tasks -> {paths[m]}")


if __name__ == "__main__":
    main()
