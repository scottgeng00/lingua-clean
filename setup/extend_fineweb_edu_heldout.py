"""Grow fineweb_edu_subsets/fwedu_heldout (setup/prepare_fineweb_edu_subsets.py) to --total documents.

The extra documents are drawn uniformly without replacement from the corpus rows not used by the
original draw (regenerated from its seed: the 4 x 8M training rows plus the existing held-out rows),
so the result stays a uniform sample disjoint from every training subset. Only the parquet row groups
holding the new rows are read from the Hub (HTTP range reads), not the whole 550 GB. Existing held-out
lines keep their order; new ones are appended. The val file is replaced atomically.

    .venv-kd/bin/python setup/extend_fineweb_edu_heldout.py --total 10000
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import orjson
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from setup.prepare_fineweb_edu_subsets import HELDOUT, N_SUBSETS, REPO, SEED, SUBDIR, SUBSET_ROWS

OUT = "/home/sgeng/scottg/pretrain/assets/data/fineweb_edu_subsets"
EXTEND_SEED = SEED + 1
N_FILES = 234


def path(fi):
    return f"datasets/{REPO}/{SUBDIR}/train-{fi:05d}-of-{N_FILES:05d}.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--total", type=int, default=10_000)
    a = ap.parse_args()
    val = f"{a.out}/{HELDOUT}/{HELDOUT}.val.jsonl"
    old = [orjson.loads(l) for l in open(val, "rb")]
    n_new = a.total - len(old)
    assert n_new > 0, f"held-out already has {len(old)} docs"
    fs = HfFileSystem()

    def meta(fi):
        with fs.open(path(fi)) as f:
            m = pq.ParquetFile(f).metadata
            return m.num_rows, [m.row_group(g).num_rows for g in range(m.num_row_groups)]

    with ThreadPoolExecutor(32) as ex:
        metas = list(ex.map(meta, range(N_FILES)))
    counts = np.array([m[0] for m in metas])
    total = int(counts.sum())
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])

    # Rows of the original draw (training subsets + existing held-out), regenerated from its seed.
    n_orig = N_SUBSETS * SUBSET_ROWS + len(old)
    used = np.random.default_rng(SEED).choice(total, size=n_orig, replace=False)
    assert used[N_SUBSETS * SUBSET_ROWS:].tolist() == [r["src_row"] for r in old], "seed mismatch"
    used = np.sort(used)

    # Uniform draw from the unused rows: rejection-sample candidates until n_new distinct unused ones.
    rng = np.random.default_rng(EXTEND_SEED)
    new = []
    seen = set()
    while len(new) < n_new:
        for r in rng.integers(0, total, size=4 * n_new).tolist():
            i = np.searchsorted(used, r)
            if (i < len(used) and used[i] == r) or r in seen:
                continue
            seen.add(r)
            new.append(r)
            if len(new) == n_new:
                break
    print(f"{total:,} rows, {n_orig:,} used; drawing {n_new:,} new held-out docs (seed {EXTEND_SEED})", flush=True)

    # Group by (file, row group) and read only those row groups.
    by_file = {}
    for r in new:
        fi = int(np.searchsorted(offsets, r, side="right") - 1)
        by_file.setdefault(fi, []).append(r - int(offsets[fi]))

    def fetch(fi):
        starts = np.concatenate([[0], np.cumsum(metas[fi][1])])
        rows = sorted(by_file[fi])
        groups = sorted({int(np.searchsorted(starts, lr, side="right") - 1) for lr in rows})
        with fs.open(path(fi)) as f:
            t = pq.ParquetFile(f).read_row_groups(groups, columns=["id", "text", "metadata"]).flatten()
        base = {g: k for k, g in zip(np.cumsum([0] + [metas[fi][1][g] for g in groups[:-1]]), groups)}
        out = {}
        for lr in rows:
            g = int(np.searchsorted(starts, lr, side="right") - 1)
            i = int(base[g] + lr - starts[g])
            out[int(offsets[fi]) + lr] = {
                "id": t.column("id")[i].as_py(), "text": t.column("text")[i].as_py(),
                "url": t.column("metadata.url")[i].as_py(), "dump": t.column("metadata.dump")[i].as_py(),
                "score": t.column("metadata.score")[i].as_py(),
                "token_count": t.column("metadata.token_count")[i].as_py(), "src_row": int(offsets[fi]) + lr}
        return out

    recs = {}
    with ThreadPoolExecutor(32) as ex:
        for out in ex.map(fetch, sorted(by_file)):
            recs.update(out)
    assert len(recs) == n_new
    rows = old + [recs[r] for r in new]
    assert len({r["src_row"] for r in rows}) == a.total

    tmp = val + ".tmp"
    with open(tmp, "wb") as f:
        for r in rows:
            f.write(orjson.dumps(r) + b"\n")
    os.replace(tmp, val)
    mpath = f"{a.out}/{HELDOUT}/dataset.json"
    meta_d = json.load(open(mpath))
    meta_d.update({"rows": a.total, "token_count_sum (dataset's own token_count field)": sum(r["token_count"] for r in rows),
                   "extension": f"rows {len(old)}..{a.total - 1}: uniform without replacement over corpus rows not in the "
                                f"original draw (seed {EXTEND_SEED}); setup/extend_fineweb_edu_heldout.py"})
    json.dump(meta_d, open(mpath, "w"), indent=1)
    print(f"{HELDOUT}: {a.total:,} docs", flush=True)


if __name__ == "__main__":
    main()
