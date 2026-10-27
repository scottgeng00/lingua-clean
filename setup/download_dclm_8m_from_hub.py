"""Fetch scottgeng00/dclm-8m (setup/push_dclm_subsets_to_hub.py) and rebuild our lingua layout:
<out>/dclm_8m_k/dclm_8m_k.chunk.NN.jsonl with parquet row j at chunk j % 16, line j // 16, plus the
dataset.json the other setup scripts read. Output is written to <split>.tmp-* and renamed when complete.

    .venv-kd/bin/python setup/download_dclm_8m_from_hub.py --out /home/sgeng/scottg/pretrain/assets/data/dclm_subsets
"""

import argparse
import glob
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import orjson
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

REPO = "scottgeng00/dclm-8m"
SPLITS = ["dclm_8m_0", "dclm_8m_1"]
N_CHUNKS = 16


def export(split, snap, out):
    final = f"{out}/{split}"
    if os.path.exists(f"{final}/dataset.json"):
        return f"{split}: exists"
    tmp = f"{out}/{split}.tmp-{os.getpid()}"
    os.makedirs(tmp)
    outs = [open(f"{tmp}/{split}.chunk.{c:02d}.jsonl", "wb") for c in range(N_CHUNKS)]
    shards = sorted(glob.glob(f"{snap}/data/{split}/*.parquet"))
    j, first_src = 0, None
    for path in shards:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=10_000, columns=["id", "text", "src_row"]):
            cols = batch.to_pydict()
            for i, t, r in zip(cols["id"], cols["text"], cols["src_row"]):
                if first_src is None:
                    first_src = r
                assert r == first_src + j, (split, j, r)
                outs[j % N_CHUNKS].write(orjson.dumps({"id": i, "text": t, "src_row": r}) + b"\n")
                j += 1
    for f in outs:
        f.close()
    assert j == 8_000_000, (split, j)
    json.dump({"source": REPO, "config": split, "rows": j, "n_chunks": N_CHUNKS,
               "global_rows": [first_src, first_src + j], "layout": "row j -> chunk j % 16, line j // 16"},
              open(f"{tmp}/dataset.json", "w"), indent=1)
    os.rename(tmp, final)
    return f"{split}: {j:,} rows from {len(shards)} shards"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for d in glob.glob(f"{a.out}/*.tmp-*"):
        shutil.rmtree(d)
    snap = snapshot_download(REPO, repo_type="dataset", max_workers=16)
    with ProcessPoolExecutor(len(SPLITS)) as ex:
        for msg in ex.map(export, SPLITS, [snap] * len(SPLITS), [a.out] * len(SPLITS)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
