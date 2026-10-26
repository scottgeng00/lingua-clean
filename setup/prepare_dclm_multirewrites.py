"""Export a multi-rewrite simplescale run (setup/build_multirewrite_manifests.py) as one row-aligned
training split per view.

The run covers the first 4M rows of dclm_8m_1, with views 0..3 for rows < 2M and views 0..1 for
rows in [2M, 4M) (iid / nemo / wofmt start at view 1: view 0 is the existing full-8M wiki rewrite).
View g is written to

    <out_root>/dclm_8m_1_rewrite_first{4m,2m}_qwen35_9b_<method>_g<g>_max4096/

first4m for g < 2 (rows [0, 4M)), first2m for g >= 2 (rows [0, 2M)): these splits do NOT have
all 8M rows of dclm_8m_1, the name says how many. Row j is the rewrite of dclm_8m_1 row j (chunk
j % 16, line j // 16), as in setup/prepare_dclm_rewrites.py, so `source_max_docs: 2000000` on a
first4m split gives exactly the rows of the first2m splits. task_ids are checked against the
subset; any missing row or wrong rewrite count aborts before anything is published.

Rows: {"id", "text", "src_row", "finish_reason", "source_truncated"} plus "view" (nemo: prompt,
wofmt: format label) or "gacha" (decisions + idea) when the run has them.

    python setup/prepare_dclm_multirewrites.py --method t12 \
        --run /checkpoint/dream-sc-3/scottg/pretrain/simplescale/runs/dclm_8m_1_rewrite_first4m-qwen35-9b-t12
"""

import argparse
import json
import os
import shutil
import time

import orjson

from prepare_dclm_rewrites import N_CHUNKS, SAMPLE_EVERY, TOKENIZER, iter_run_rows, iter_subset_rows

SUBSET = "/home/sgeng/scottg/pretrain/assets/data/dclm_subsets/dclm_8m_1"
OUT_ROOT = "/home/sgeng/scottg/pretrain/assets/data/dclm_rewrites"
ROWS, ROWS_K4 = 4_000_000, 2_000_000
FIRST_VIEW = {"iid": 1, "nemo": 1, "wofmt": 1, "t12": 0, "gacha": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--method", required=True, choices=sorted(FIRST_VIEW))
    ap.add_argument("--out-root", default=OUT_ROOT)
    args = ap.parse_args()
    t0 = time.time()
    first = FIRST_VIEW[args.method]
    views = list(range(first, 4))
    rows_of = {g: ROWS if g < 2 else ROWS_K4 for g in views}
    names = {g: f"dclm_8m_1_rewrite_first{rows_of[g] // 1_000_000}m_qwen35_9b_{args.method}_g{g}_max4096" for g in views}
    for g in views:
        if os.path.exists(os.path.join(args.out_root, names[g])):
            raise SystemExit(f"{names[g]} exists")
    tmps = {g: os.path.join(args.out_root, f".{names[g]}.tmp") for g in views}
    outs = {}
    for g in views:
        shutil.rmtree(tmps[g], ignore_errors=True)
        os.makedirs(tmps[g])
        outs[g] = [open(os.path.join(tmps[g], f"{names[g]}.chunk.{c:02d}.jsonl"), "wb", buffering=1 << 22)
                   for c in range(N_CHUNKS)]
    st = {g: {"rows": 0, "chars": 0, "finish_length": 0, "finish_unknown": 0, "empty": 0, "source_truncated": 0}
          for g in views}
    sample = {g: [] for g in views}
    view_counts = {g: {} for g in views}
    start = json.load(open(os.path.join(SUBSET, "dataset.json")))["global_rows"][0]
    subset_ids = iter_subset_rows(SUBSET)
    try:
        j = -1
        for j, rec in enumerate(iter_run_rows(args.run)):
            sid = next(subset_ids)
            if rec["task_id"] != sid:
                raise SystemExit(f"row {j}: task_id {rec['task_id']!r} != subset id {sid!r}")
            res = rec["result"]
            gs = [g for g in views if j < rows_of[g]]
            if len(res["rephrases"]) != len(gs):
                raise SystemExit(f"row {j}: expected {len(gs)} rewrites, got {len(res['rephrases'])}")
            frs = res.get("finish_reasons") or [None] * len(gs)
            for i, g in enumerate(gs):
                text = res["rephrases"][i] or ""
                row = {"id": rec["task_id"], "text": text, "src_row": start + j, "finish_reason": frs[i],
                       "source_truncated": bool(res.get("source_truncated"))}
                if "views" in res:
                    row["view"] = res["views"][i]
                    view_counts[g][row["view"]] = view_counts[g].get(row["view"], 0) + 1
                if "gacha" in res:
                    row["gacha"] = res["gacha"][i]
                outs[g][j % N_CHUNKS].write(orjson.dumps(row) + b"\n")
                s = st[g]
                s["rows"] += 1
                s["chars"] += len(text)
                s["finish_length"] += frs[i] == "length"
                s["finish_unknown"] += frs[i] is None
                s["empty"] += not text.strip()
                s["source_truncated"] += row["source_truncated"]
                if j % SAMPLE_EVERY == 0:
                    sample[g].append(text)
        if j + 1 != ROWS:
            raise SystemExit(f"run has {j + 1} rows, expected {ROWS}")
    except BaseException:
        for g in views:
            for f in outs[g]:
                f.close()
            shutil.rmtree(tmps[g], ignore_errors=True)
        raise

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    run_ds = json.load(open(os.path.join(args.run, "results", "dataset.json")))
    for g in views:
        for f in outs[g]:
            f.close()
        per_doc = sum(len(x) + 1 for x in tok(sample[g])["input_ids"]) / len(sample[g])
        s = st[g]
        json.dump({
            "description": f"View {g} of the {args.method} multi-rewrite of the FIRST {rows_of[g]:,} rows of dclm_8m_1 "
                           f"(not all 8M; row j = rewrite of dclm_8m_1 row j; chunk j % {N_CHUNKS}, line j // {N_CHUNKS}).",
            "subset": SUBSET, "partial_of": "dclm_8m_1", "src_rows": [0, rows_of[g]], "method": args.method, "view": g,
            "run_dir": os.path.abspath(args.run), "manifest": run_ds.get("manifest"),
            "manifest_sha256": run_ds.get("sha256"), "chunks": N_CHUNKS, **s,
            "frac_finish_length": s["finish_length"] / s["rows"], "view_counts": view_counts[g] or None,
            "qwen35_tokens_per_doc_incl_eos": round(per_doc, 1), "qwen35_tokens_est": round(per_doc * s["rows"]),
            "token_sample_docs": len(sample[g]), "tokenizer": TOKENIZER, "elapsed_seconds": round(time.time() - t0),
        }, open(os.path.join(tmps[g], "dataset.json"), "w"), indent=2)
        os.replace(tmps[g], os.path.join(args.out_root, names[g]))
        print(f"{names[g]}: {s['rows']:,} rows, {per_doc:.0f} tok/doc, {s['finish_length'] / s['rows']:.1%} hit cap, "
              f"{s['empty']} empty")
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
