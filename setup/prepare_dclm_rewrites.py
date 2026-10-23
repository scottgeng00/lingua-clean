"""Export a simplescale rewrite run over a DCLM subset as a row-aligned training split.

Output: <out_root>/<subset>_<details>/ (e.g. dclm_rewrites/dclm_8m_1_qwen35_2b_wiki_g1_max4096),
the guarantee being that the name starts with the source subset's name and that local row j
(chunk j % 16, line j // 16) is the rewrite of the subset's local row j, the same layout as
setup/prepare_dclm_subsets.py. Every row's task_id is checked against the subset's id at the
same position; any mismatch, missing or extra row aborts before anything is published.

Rows: {"id", "text", "src_row" (global DCLM row), "finish_reason", "source_truncated"}. Every
row is kept, so alignment holds; outputs that hit max_new_tokens or are empty are counted in
dataset.json. Token counts come from a 1-in-200 sample.

    python setup/prepare_dclm_rewrites.py \
        --run /checkpoint/dream-sc-3/scottg/pretrain/simplescale/runs/dclm_8m_1-qwen35-2b-wiki-g1 \
        --subset /home/sgeng/scottg/pretrain/assets/data/dclm_subsets/dclm_8m_1 \
        --details qwen35_2b_wiki_g1_max4096
"""

import argparse
import glob
import json
import os
import shutil
import time

import orjson

N_CHUNKS = 16
SAMPLE_EVERY = 200
OUT_ROOT = "/home/sgeng/scottg/pretrain/assets/data/dclm_rewrites"
TOKENIZER = "/home/sgeng/scottg/pretrain/assets/models/Qwen3.5-2B-tokenizer-eot"


def iter_run_rows(run_dir):
    ds = json.load(open(os.path.join(run_dir, "results", "dataset.json")))
    shards = sorted(ds["shards"], key=lambda s: s["chunk"])
    if [s["chunk"] for s in shards] != list(range(ds["chunks"])) or sum(s["rows"] for s in shards) != ds["rows"]:
        raise SystemExit(f"incomplete or inconsistent results in {run_dir}")
    for s in shards:
        n = 0
        with open(s["uri"], "rb") as f:
            for line in f:
                n += 1
                yield orjson.loads(line)
        if n != s["rows"]:
            raise SystemExit(f"{s['uri']}: expected {s['rows']} rows, found {n}")


def iter_subset_rows(subset_dir):
    name = os.path.basename(os.path.normpath(subset_dir))
    files = sorted(glob.glob(os.path.join(subset_dir, f"{name}.chunk.*.jsonl")))
    assert len(files) == N_CHUNKS, files
    readers = [open(f, "rb") for f in files]
    j = 0
    while True:
        line = readers[j % N_CHUNKS].readline()
        if not line:
            if any(r.readline() for r in readers):
                raise SystemExit(f"{subset_dir} is not in round-robin layout")
            return
        yield orjson.loads(line)["id"]
        j += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="simplescale run dir")
    ap.add_argument("--subset", required=True, help="dclm_subsets/<name> the run rewrote")
    ap.add_argument("--details", required=True, help="suffix after the subset name, e.g. qwen35_2b_wiki_g1_max4096")
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    args = ap.parse_args()
    t0 = time.time()

    subset = os.path.basename(os.path.normpath(args.subset))
    smeta = json.load(open(os.path.join(args.subset, "dataset.json")))
    if smeta.get("dropped"):
        raise SystemExit("subset dropped rows; global row numbers are not derivable from position")
    start = smeta["global_rows"][0]
    name = f"{subset}_{args.details}"
    final, tmp = os.path.join(args.out_root, name), os.path.join(args.out_root, f".{name}.tmp")
    if os.path.exists(final):
        raise SystemExit(f"{final} exists")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    outs = [open(os.path.join(tmp, f"{name}.chunk.{c:02d}.jsonl"), "wb", buffering=1 << 22) for c in range(N_CHUNKS)]
    st = {"rows": 0, "chars": 0, "finish_length": 0, "finish_unknown": 0, "empty": 0, "source_truncated": 0}
    sample = []
    subset_ids = iter_subset_rows(args.subset)
    try:
        for j, rec in enumerate(iter_run_rows(args.run)):
            sid = next(subset_ids, None)
            if rec["task_id"] != sid:
                raise SystemExit(f"row {j}: task_id {rec['task_id']!r} != subset id {sid!r}")
            res = rec["result"]
            if len(res["rephrases"]) != 1:
                raise SystemExit(f"row {j}: expected 1 rewrite, got {len(res['rephrases'])}")
            text = res["rephrases"][0] or ""
            fr = (res.get("finish_reasons") or [None])[0]
            outs[j % N_CHUNKS].write(orjson.dumps({
                "id": rec["task_id"], "text": text, "src_row": start + j,
                "finish_reason": fr, "source_truncated": bool(res.get("source_truncated")),
            }) + b"\n")
            st["rows"] += 1
            st["chars"] += len(text)
            st["finish_length"] += fr == "length"
            st["finish_unknown"] += fr is None
            st["empty"] += not text.strip()
            st["source_truncated"] += bool(res.get("source_truncated"))
            if j % SAMPLE_EVERY == 0:
                sample.append(text)
        if next(subset_ids, None) is not None or st["rows"] != smeta["rows"]:
            raise SystemExit(f"run has {st['rows']} rows, subset has {smeta['rows']}")
    except BaseException:
        for f in outs:
            f.close()
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    for f in outs:
        f.close()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    per_doc = sum(len(x) + 1 for x in tok(sample)["input_ids"]) / len(sample)
    run_ds = json.load(open(os.path.join(args.run, "results", "dataset.json")))
    json.dump({
        "description": f"Rewrites of {subset} (row j = rewrite of {subset} row j; chunk j % {N_CHUNKS}, line j // {N_CHUNKS}).",
        "subset": os.path.abspath(args.subset), "run_dir": os.path.abspath(args.run),
        "manifest": run_ds.get("manifest"), "manifest_sha256": run_ds.get("sha256"), "chunks": N_CHUNKS,
        **st, "frac_finish_length": st["finish_length"] / st["rows"],
        "qwen35_tokens_per_doc_incl_eos": round(per_doc, 1), "qwen35_tokens_est": round(per_doc * st["rows"]),
        "token_sample_docs": len(sample), "tokenizer": args.tokenizer, "elapsed_seconds": round(time.time() - t0),
    }, open(os.path.join(tmp, "dataset.json"), "w"), indent=2)
    os.replace(tmp, final)
    print(f"{name}: {st['rows']:,} rows, {per_doc:.0f} tok/doc, {st['finish_length'] / st['rows']:.1%} hit max_new_tokens, "
          f"{st['empty']} empty, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
