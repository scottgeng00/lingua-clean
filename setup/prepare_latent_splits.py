"""Build lingua training splits from simplescale latent-thought runs.

Each simplescale row is one DCLM doc split into g pieces with a model-written
<think>...</think> block inserted between pieces (result.megadoc). The output keeps the
row layout of rewrite_splits/dclm_15m exactly (row i -> chunk i % 16, line i // 16), so
data.source_max_docs = N selects the latent versions of exactly the first N real docs,
read in the same order as the real docs. Every row's task_id is checked against the id
of the same row in dclm_15m.

    python setup/prepare_latent_splits.py --out /home/sgeng/scottg/pretrain/assets/data/latent_splits \
        --run lat2b_g4=/checkpoint/.../qwen35-2b-latent-g4-2m ...
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

from prepare_rephrase_splits import N_CHUNKS, read_ids

IDS = None  # set before the worker pool forks


def scatter(args):
    """Rows of a contiguous shard range -> per-chunk partial files (rows stay in order)."""
    name, worker, shard_items, rows_per_shard, tmp = args
    files = [open(os.path.join(tmp, f"{name}.{c:02d}.w{worker:04d}"), "w") for c in range(N_CHUNKS)]
    stats = {"rows": 0, "think_blocks": 0, "unclosed": 0}
    for chunk_idx, uri in shard_items:
        with open(uri) as f:
            for k, line in enumerate(f):
                row = chunk_idx * rows_per_shard + k
                if row >= len(IDS):
                    break
                rec = json.loads(line)
                if rec["task_id"] != IDS[row]:
                    raise RuntimeError(f"row {row}: task_id {rec['task_id']!r} != dclm_15m id {IDS[row]!r} ({uri})")
                doc = rec["result"]["megadoc"]
                n_open, n_close = doc.count("<think>"), doc.count("</think>")
                stats["rows"] += 1
                stats["think_blocks"] += n_open
                stats["unclosed"] += n_open != n_close
                files[row % N_CHUNKS].write(json.dumps({"id": rec["task_id"], "text": doc, "src_row": row}, ensure_ascii=False) + "\n")
    for f in files:
        f.close()
    return name, worker, stats


def main():
    global IDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", required=True, help="name=simplescale run dir")
    ap.add_argument("--n-rows", type=int, default=2_000_000)
    ap.add_argument("--dclm", default="/home/sgeng/scottg/pretrain/assets/data/rewrite_splits/dclm_15m")
    ap.add_argument("--workers-per-run", type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()
    runs = dict(r.split("=", 1) for r in args.run)
    tmp = os.path.join(args.out, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    print(f"reading ids of the first {args.n_rows} rows of {args.dclm}", flush=True)
    IDS = read_ids(args.dclm, args.n_rows)

    jobs, meta = [], {}
    for name, run_dir in runs.items():
        os.makedirs(os.path.join(args.out, name), exist_ok=True)
        ds = json.load(open(os.path.join(run_dir, "results", "dataset.json")))
        meta[name] = {"run_dir": run_dir, "sha256": ds.get("sha256"), "manifest": ds.get("manifest"), "rows": ds.get("rows")}
        shards = sorted(ds["shards"], key=lambda s: s["chunk"])
        assert [s["chunk"] for s in shards] == list(range(len(shards)))
        rps = ds["chunk_size"]
        assert rps % N_CHUNKS == 0  # contiguous shard ranges keep per-chunk row order when concatenated
        items = [(s["chunk"], s["uri"]) for s in shards if s["chunk"] * rps < args.n_rows]
        per = (len(items) + args.workers_per_run - 1) // args.workers_per_run
        for w, i in enumerate(range(0, len(items), per)):
            jobs.append((name, w, items[i:i + per], rps, tmp))

    stats = {n: {"rows": 0, "think_blocks": 0, "unclosed": 0} for n in runs}
    with Pool(min(len(jobs), 96)) as p:
        for name, _, st in p.imap_unordered(scatter, jobs):
            for k in ("rows", "think_blocks", "unclosed"):
                stats[name][k] += st[k]
    print(f"pass 1 done in {time.time() - t0:.0f}s: {stats}", flush=True)

    # concatenate the workers' partials per chunk in worker order (= row order)
    for name in runs:
        n_workers = len([j for j in jobs if j[0] == name])
        for c in range(N_CHUNKS):
            path = os.path.join(args.out, name, f"{name}.chunk.{c:02d}.jsonl")
            with open(path + ".tmp", "w") as out:
                for w in range(n_workers):
                    part = os.path.join(tmp, f"{name}.{c:02d}.w{w:04d}")
                    with open(part) as f:
                        for line in f:
                            out.write(line)
                    os.remove(part)
            os.replace(path + ".tmp", path)
        assert stats[name]["rows"] == args.n_rows, (name, stats[name])
    os.rmdir(tmp)
    json.dump({
        "description": "Latent-thought versions (simplescale result.megadoc: the doc split into g pieces "
                       "with <think>...</think> blocks inserted) of the first n_rows docs of dclm.first-15m, "
                       "in the same row layout as rewrite_splits/dclm_15m (row i -> chunk i % 16, line i // 16).",
        "real_docs_split": args.dclm, "n_rows": args.n_rows, "inputs": meta, "stats": stats,
        "elapsed_seconds": round(time.time() - t0),
    }, open(os.path.join(args.out, "dataset.json"), "w"), indent=2)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
