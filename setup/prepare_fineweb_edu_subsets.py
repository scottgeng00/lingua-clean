"""Uniform 32M-document sample of HuggingFaceTB/smollm-corpus fineweb-edu-dedup in our lingua layout.

    .venv-kd/bin/python setup/prepare_fineweb_edu_subsets.py --raw <staging dir> \
        --out /home/sgeng/scottg/pretrain/assets/data/fineweb_edu_subsets

1. Download all 234 parquet files (~550 GB, ~190.2M documents) to --raw (resumable).
2. Draw 4 x 8M + 1000 global row indices uniformly without replacement (seeded). The draw order is the
   shuffle: output position k < 32M goes to subset fwedu_8m_{k // 8M} as local row j = k % 8M; the last
   1000 form the held-out validation set fwedu_heldout/fwedu_heldout.val.jsonl.
3. Local row j is written to chunk j % 16, line j // 16 (the dclm_subsets layout), so `source_max_docs: N`
   is exactly the first N rows of a subset, itself a uniform random sample of the corpus.

Each line: {"id", "text", "url", "dump", "score", "token_count", "src_row"}, src_row being the global row
index (file order, row order within file). Every chunk file's line count is checked, and each output
directory is staged as <name>.tmp and renamed only when complete. --delete-raw removes the parquet files
after a successful run.
"""

import argparse
import glob
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq

REPO = "HuggingFaceTB/smollm-corpus"
SUBDIR = "fineweb-edu-dedup"
N_SUBSETS, SUBSET_ROWS, N_CHUNKS, N_HELDOUT = 4, 8_000_000, 16, 1000
SEED = 20261028
HELDOUT = "fwedu_heldout"


def subset_name(k):
    return f"fwedu_8m_{k}"


def bucket_dir(tmp, b):
    # b in [0, N_SUBSETS * N_CHUNKS) for training chunks, N_SUBSETS * N_CHUNKS for held-out
    return f"{tmp}/bucket_{b:03d}"


def extract(args):
    """Pass 1: write the selected rows of one parquet file into per-(subset, chunk) bucket parts."""
    fi, path, local_rows, positions, row_offset, tmp = args
    done = f"{tmp}/done/{fi:05d}"
    if os.path.exists(done):
        return fi, 0
    t = pq.read_table(path, columns=["id", "text", "metadata"])
    # A file's text exceeds 2 GB, which overflows `string`'s 32-bit offsets when take() concatenates.
    t = t.set_column(t.schema.get_field_index("text"), "text", t.column("text").cast(pa.large_string()))
    t = t.take(local_rows).flatten()
    cols = {c: t.column(c).to_pylist() for c in ("id", "text", "metadata.url", "metadata.dump",
                                                 "metadata.score", "metadata.token_count")}
    out = {}
    for i, (lr, k) in enumerate(zip(local_rows.tolist(), positions.tolist())):
        if k < N_SUBSETS * SUBSET_ROWS:
            s, j = divmod(k, SUBSET_ROWS)
            b, line = s * N_CHUNKS + j % N_CHUNKS, j // N_CHUNKS
        else:
            b, line = N_SUBSETS * N_CHUNKS, k - N_SUBSETS * SUBSET_ROWS
        rec = {"_line": line, "id": cols["id"][i], "text": cols["text"][i], "url": cols["metadata.url"][i],
               "dump": cols["metadata.dump"][i], "score": cols["metadata.score"][i],
               "token_count": cols["metadata.token_count"][i], "src_row": row_offset + lr}
        out.setdefault(b, []).append(orjson.dumps(rec))
    for b, lines in out.items():
        with open(f"{bucket_dir(tmp, b)}/part_{fi:05d}.jsonl", "wb") as f:
            f.write(b"\n".join(lines) + b"\n")
    open(done, "w").close()
    return fi, len(local_rows)


def assemble(args):
    """Pass 2: order one bucket's rows by line index and write the final chunk / val file."""
    b, tmp, out = args
    recs = []
    for p in glob.glob(f"{bucket_dir(tmp, b)}/part_*.jsonl"):
        with open(p, "rb") as f:
            recs += [orjson.loads(l) for l in f if l.strip()]
    recs.sort(key=lambda r: r["_line"])
    assert [r["_line"] for r in recs] == list(range(len(recs))), f"bucket {b}: lines not contiguous"
    if b < N_SUBSETS * N_CHUNKS:
        s, c = divmod(b, N_CHUNKS)
        path = f"{out}/{subset_name(s)}.tmp/{subset_name(s)}.chunk.{c:02d}.jsonl"
        assert len(recs) == SUBSET_ROWS // N_CHUNKS, (b, len(recs))
    else:
        path = f"{out}/{HELDOUT}.tmp/{HELDOUT}.val.jsonl"
        assert len(recs) == N_HELDOUT, len(recs)
    with open(path, "wb") as f:
        for r in recs:
            del r["_line"]
            f.write(orjson.dumps(r) + b"\n")
    return b, len(recs), sum(r["token_count"] for r in recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="staging dir for the parquet download")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--delete-raw", action="store_true")
    a = ap.parse_args()
    names = [subset_name(s) for s in range(N_SUBSETS)] + [HELDOUT]
    if all(os.path.exists(f"{a.out}/{n}") for n in names):
        print("all outputs exist; nothing to do")
        return

    from huggingface_hub import snapshot_download
    snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"{SUBDIR}/*.parquet"], local_dir=a.raw,
                      max_workers=32)
    files = sorted(glob.glob(f"{a.raw}/{SUBDIR}/train-*.parquet"))
    assert len(files) == 234, len(files)
    counts = np.array([pq.ParquetFile(p).metadata.num_rows for p in files])
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    total = int(counts.sum())
    n_draw = N_SUBSETS * SUBSET_ROWS + N_HELDOUT
    print(f"{total:,} documents in {len(files)} files; drawing {n_draw:,} uniformly (seed {SEED})", flush=True)

    sel = np.random.default_rng(SEED).choice(total, size=n_draw, replace=False)  # random order = shuffle
    order = np.argsort(sel, kind="stable")
    sel_sorted, pos_sorted = sel[order], order
    file_of = np.searchsorted(offsets, sel_sorted, side="right") - 1

    tmp = f"{a.out}/.prep_tmp"
    os.makedirs(f"{tmp}/done", exist_ok=True)
    for b in range(N_SUBSETS * N_CHUNKS + 1):
        os.makedirs(bucket_dir(tmp, b), exist_ok=True)
    jobs = []
    bounds = np.searchsorted(file_of, np.arange(len(files) + 1))
    for fi, p in enumerate(files):
        lo, hi = bounds[fi], bounds[fi + 1]
        jobs.append((fi, p, sel_sorted[lo:hi] - offsets[fi], pos_sorted[lo:hi], int(offsets[fi]), tmp))
    n = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for fi, k in ex.map(extract, jobs):
            n += k
            if fi % 20 == 0:
                print(f"pass 1: file {fi} done", flush=True)

    for name in names:
        shutil.rmtree(f"{a.out}/{name}.tmp", ignore_errors=True)
        os.makedirs(f"{a.out}/{name}.tmp")
    tokens = {name: 0 for name in names}
    with ProcessPoolExecutor(min(a.workers, 16)) as ex:
        for b, rows, tok in ex.map(assemble, [(b, tmp, a.out) for b in range(N_SUBSETS * N_CHUNKS + 1)]):
            tokens[names[b // N_CHUNKS] if b < N_SUBSETS * N_CHUNKS else HELDOUT] += tok

    for s, name in enumerate(names):
        rows = SUBSET_ROWS if name != HELDOUT else N_HELDOUT
        meta = {"source": f"{REPO}/{SUBDIR}", "corpus_rows": total, "rows": rows, "seed": SEED,
                "sample": "uniform without replacement over all rows; positions = draw order",
                "draw_positions": [s * SUBSET_ROWS, (s + 1) * SUBSET_ROWS] if name != HELDOUT
                else [N_SUBSETS * SUBSET_ROWS, n_draw],
                "token_count_sum (dataset's own token_count field)": tokens[name]}
        if name != HELDOUT:
            meta.update({"n_chunks": N_CHUNKS, "layout": "row j -> chunk j % 16, line j // 16"})
        json.dump(meta, open(f"{a.out}/{name}.tmp/dataset.json", "w"), indent=1)
        os.rename(f"{a.out}/{name}.tmp", f"{a.out}/{name}")
        print(f"{name}: {rows:,} rows, {tokens[name]:,} tokens (token_count field)", flush=True)
    shutil.rmtree(tmp)
    if a.delete_raw:
        shutil.rmtree(a.raw)
        print(f"deleted {a.raw}")


if __name__ == "__main__":
    main()
