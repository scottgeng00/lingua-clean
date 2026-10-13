"""Build lingua training splits from simplescale multi-rephrase runs (rephrases of the
first 2M DCLM docs, 16 per doc).

Each output is the first `n_docs` documents x rephrases [0, k) of one rewriter, as a source
of 16 chunk files that is globally shuffled (random chunk per record, then shuffled within
each chunk), so the k rephrases of a document land at unrelated points in training. The
real documents themselves are not copied: train on them from rewrite_splits/dclm_15m with
data.source_max_docs = n_docs, which selects exactly the first n_docs rows. Every rephrase
row's task_id is checked against the id of the same row in dclm_15m.

    python setup/prepare_rephrase_splits.py --out /home/sgeng/scottg/pretrain/assets/data/rephrase_g16_splits \
        --run 2b=/checkpoint/.../qwen35-2b-rephrase-g16-2m --run 27b=/checkpoint/.../qwen35-27b-rephrase-g16-2m \
        --split 1875000:7 --split 937500:15
"""

import argparse
import hashlib
import json
import os
import random
import time
from multiprocessing import Pool

N_CHUNKS = 16
SEED = 20261010


def _chunk_ids(args):
    path, need = args
    out = []
    with open(path) as f:
        for _ in range(need):
            out.append(json.loads(f.readline())["id"])
    return out


def read_ids(dclm_dir, n_rows):
    """ids of rows [0, n_rows) of a round-robin split (row i = chunk i % 16, line i // 16)."""
    chunks = sorted(f for f in os.listdir(dclm_dir) if f.endswith(".jsonl"))
    assert len(chunks) == N_CHUNKS, chunks
    jobs = [(os.path.join(dclm_dir, chunks[c]), (n_rows - c + N_CHUNKS - 1) // N_CHUNKS) for c in range(N_CHUNKS)]
    with Pool(N_CHUNKS) as p:
        per_chunk = p.map(_chunk_ids, jobs)
    return [per_chunk[i % N_CHUNKS][i // N_CHUNKS] for i in range(n_rows)]


def _chunk_of(row, rep, salt):
    h = hashlib.blake2b(f"{salt}:{row}:{rep}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") % N_CHUNKS


# set in main() before the worker pools fork, so workers inherit them without pickling
IDS = None
OUTPUTS = None


def scatter(args):
    """Pass 1 for a range of simplescale shards: route each kept rephrase to a temp chunk file."""
    worker, label, shard_items, tmp = args
    ids, outputs = IDS, [o for o in OUTPUTS if o["run"] == label]
    files = {}
    counts = {o["name"]: 0 for o in outputs}
    for chunk_idx, uri, rows_per_shard in shard_items:
        with open(uri) as f:
            for k, line in enumerate(f):
                row = chunk_idx * rows_per_shard + k
                wanted = [o for o in outputs if row < o["n_docs"]]
                if not wanted:
                    continue
                rec = json.loads(line)
                if rec["task_id"] != ids[row]:
                    raise RuntimeError(f"row {row}: task_id {rec['task_id']!r} != dclm_15m id {ids[row]!r} ({uri})")
                reps = rec["result"]["rephrases"]
                for o in wanted:
                    if len(reps) < o["k"]:
                        raise RuntimeError(f"row {row}: {len(reps)} rephrases < k={o['k']} ({uri})")
                    for j in range(o["k"]):
                        c = _chunk_of(row, j, o["name"])
                        key = (o["name"], c)
                        if key not in files:
                            files[key] = open(os.path.join(tmp, f"{o['name']}.{c:02d}.w{worker:03d}"), "w")
                        files[key].write(json.dumps({"id": f"{rec['task_id']}#r{j}", "text": reps[j],
                                                     "src_row": row, "rephrase_idx": j}, ensure_ascii=False) + "\n")
                        counts[o["name"]] += 1
    for fh in files.values():
        fh.close()
    return counts


def gather(args):
    """Pass 2: concatenate the workers' temp files of one output chunk, shuffle, write."""
    name, c, tmp, out_dir = args
    parts = sorted(f for f in os.listdir(tmp) if f.startswith(f"{name}.{c:02d}.w"))
    lines = []
    for p in parts:
        with open(os.path.join(tmp, p)) as f:
            lines.extend(f)
    random.Random(f"{SEED}:{name}:{c}").shuffle(lines)
    path = os.path.join(out_dir, name, f"{name}.chunk.{c:02d}.jsonl")
    with open(path + ".tmp", "w") as f:
        f.writelines(lines)
    os.replace(path + ".tmp", path)
    for p in parts:
        os.remove(os.path.join(tmp, p))
    return name, c, len(lines)


def main():
    global IDS, OUTPUTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", required=True, help="label=simplescale run dir")
    ap.add_argument("--split", action="append", required=True, help="n_docs:k (first n_docs docs, rephrases 0..k-1)")
    ap.add_argument("--dclm", default="/home/sgeng/scottg/pretrain/assets/data/rewrite_splits/dclm_15m")
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    t0 = time.time()

    runs = dict(r.split("=", 1) for r in args.run)
    splits = [tuple(int(x) for x in s.split(":")) for s in args.split]
    outputs = []
    for label, run_dir in runs.items():
        for n_docs, k in splits:
            outputs.append({"name": f"rw{label}_first{n_docs // 1000}k_k{k}", "run": label, "run_dir": run_dir,
                            "n_docs": n_docs, "k": k})
    max_rows = max(o["n_docs"] for o in outputs)
    os.makedirs(args.out, exist_ok=True)
    tmp = os.path.join(args.out, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    for o in outputs:
        os.makedirs(os.path.join(args.out, o["name"]), exist_ok=True)

    print(f"reading ids of the first {max_rows} rows of {args.dclm}", flush=True)
    IDS = read_ids(args.dclm, max_rows)
    OUTPUTS = outputs

    jobs, meta = [], {}
    worker = 0
    for label, run_dir in runs.items():
        ds = json.load(open(os.path.join(run_dir, "results", "dataset.json")))
        meta[label] = {"run_dir": run_dir, "sha256": ds.get("sha256"), "manifest": ds.get("manifest"), "rows": ds.get("rows")}
        shards = sorted(ds["shards"], key=lambda s: s["chunk"])
        assert [s["chunk"] for s in shards] == list(range(len(shards)))
        rows_per_shard = ds["chunk_size"]
        items = [(s["chunk"], s["uri"], rows_per_shard) for s in shards if s["chunk"] * rows_per_shard < max_rows]
        per = (len(items) + args.workers // len(runs) - 1) // (args.workers // len(runs))
        for i in range(0, len(items), per):
            jobs.append((worker, label, items[i:i + per], tmp))
            worker += 1

    print(f"pass 1: {len(jobs)} workers over {sum(len(j[2]) for j in jobs)} shards", flush=True)
    totals = {o["name"]: 0 for o in outputs}
    with Pool(len(jobs)) as p:
        for counts in p.imap_unordered(scatter, jobs):
            for n, c in counts.items():
                totals[n] += c
    print(f"pass 1 done in {time.time() - t0:.0f}s: {totals}", flush=True)

    with Pool(min(32, len(outputs) * N_CHUNKS)) as p:
        written = p.map(gather, [(o["name"], c, tmp, args.out) for o in outputs for c in range(N_CHUNKS)])
    os.rmdir(tmp)
    for o in outputs:
        n = sum(cnt for name, _, cnt in written if name == o["name"])
        assert n == o["n_docs"] * o["k"] == totals[o["name"]], (o["name"], n, totals[o["name"]])
        o["records"] = n
    # merge into an existing dataset.json so several invocations can share one output dir
    meta_path = os.path.join(args.out, "dataset.json")
    prev = json.load(open(meta_path)) if os.path.exists(meta_path) else {"inputs": {}, "outputs": []}
    new_names = {o["name"] for o in outputs}
    json.dump({
        "description": "Rephrases of the first n_docs docs of dclm.first-15m (rows of rewrite_splits/dclm_15m), "
                       "rephrase indices 0..k-1, globally shuffled into 16 chunks. Train the real docs from "
                       "dclm_15m with data.source_max_docs = n_docs.",
        "real_docs_split": args.dclm, "seed": SEED, "inputs": {**prev["inputs"], **meta},
        "outputs": [o for o in prev["outputs"] if o["name"] not in new_names]
                   + [{k: v for k, v in o.items() if k != "run_dir"} for o in outputs],
        "elapsed_seconds": round(time.time() - t0),
    }, open(meta_path, "w"), indent=2)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
