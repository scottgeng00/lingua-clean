"""Publish dclm_8m_0 and dclm_8m_1 (our data-constrained DCLM splits) to the Hugging Face Hub as parquet.

Each split becomes one dataset config with columns {id, text, src_row}, in the subset's local row
order: parquet row j == local row j of dclm_subsets/<split> (chunk j % 16, line j // 16), and src_row
is the global row in the shuffled Dolmino DCLM chunk.00 (dclm_8m_k covers rows [8M*k, 8M*(k+1))).
Writing parquet row j to chunk j % 16 reproduces our lingua jsonl layout exactly (see the card).

    .venv/bin/python setup/push_dclm_subsets_to_hub.py write   # parquet shards -> --staging
    .venv/bin/python setup/push_dclm_subsets_to_hub.py upload  # card + shards -> --repo (resumable)
"""

import argparse
import json
import os

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

DATA = "/home/sgeng/scottg/pretrain/assets/data/dclm_subsets"
STAGING = "/checkpoint/dream-sc-3/scottg/pretrain/assets/hf_upload/dclm-8m"
REPO = "scottgeng00/dclm-8m"
SPLITS = ["dclm_8m_0", "dclm_8m_1"]
N_CHUNKS, ROWS_PER_SHARD = 16, 125_000
SCHEMA = pa.schema([("id", pa.string()), ("text", pa.string()), ("src_row", pa.int64())])

CARD = """---
license: cc-by-4.0
language:
- en
task_categories:
- text-generation
pretty_name: DCLM 8M-doc splits (data-constrained pretraining)
size_categories:
- 1M<n<10M
configs:
- config_name: dclm_8m_0
  data_files:
  - split: train
    path: data/dclm_8m_0/*.parquet
- config_name: dclm_8m_1
  data_files:
  - split: train
    path: data/dclm_8m_1/*.parquet
---

# DCLM 8M-doc splits

Two disjoint 8,000,000-document slices of DCLM-baseline, used as a fixed, data-constrained pretraining
budget (~10.1B Qwen3.5 tokens each, ~1,265 tokens/doc) in our synthetic-data / rewriting experiments:

| config | documents | source rows | Qwen3.5 tokens (est.) |
|---|---|---|---|
| `dclm_8m_0` | 8,000,000 | [0, 8M) | 10.10B |
| `dclm_8m_1` | 8,000,000 | [8M, 16M) | 10.18B |

**Source.** The DCLM portion of [allenai/dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124)
(itself drawn from [DCLM-baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)), globally
shuffled; these are the first 16M rows of shuffled chunk 0, split in two. Documents are unmodified (`text` and
`id`, which is the page URL, are copied verbatim).

**Columns.** `id` (URL), `text`, `src_row` (row index in the shuffled source stream).

**Row order.** Row `j` of a config is local document `j` of the split. Any prefix of a config is a uniform
random subset (the source is shuffled), and prefixes are nested, so "the first N docs" is a well-defined
smaller budget.

**Reproducing our training layout.** Our dataloader reads a split as 16 `*.chunk.NN.jsonl` files with local
row `j` at chunk `j % 16`, line `j // 16`:

```python
import datasets, json
ds = datasets.load_dataset("scottgeng00/dclm-8m", "dclm_8m_0", split="train")
outs = [open(f"dclm_8m_0/dclm_8m_0.chunk.{c:02d}.jsonl", "w") for c in range(16)]
for j, row in enumerate(ds):
    outs[j % 16].write(json.dumps({"id": row["id"], "text": row["text"]}) + "\\n")
```

**Held-out validation.** Several hundred documents of the standard Dolmino DCLM val set share a URL or text
with these splits; we evaluate on a separately built held-out set with no URL / exact / normalized-text
overlap with either split.

**License.** DCLM-baseline is released under CC-BY-4.0 and Dolmino-mix-1124 under ODC-BY; please cite the
original works:

```
@article{li2024datacomplm,
  title={DataComp-LM: In search of the next generation of training sets for language models},
  author={Li, Jeffrey and Fang, Alex and Smyrnis, Georgios and others},
  journal={arXiv preprint arXiv:2406.11794}, year={2024}
}
@article{olmo20242olmo2furious,
  title={2 OLMo 2 Furious}, author={OLMo, Team and others},
  journal={arXiv preprint arXiv:2501.00656}, year={2024}
}
```
"""


def iter_rows(split):
    d = f"{DATA}/{split}"
    meta = json.load(open(f"{d}/dataset.json"))
    assert not meta.get("dropped"), split
    start = meta["global_rows"][0]
    readers = [open(f"{d}/{split}.chunk.{c:02d}.jsonl", "rb") for c in range(N_CHUNKS)]
    j = 0
    while True:
        line = readers[j % N_CHUNKS].readline()
        if not line:
            assert not any(r.readline() for r in readers), f"{split} not round-robin"
            assert j == meta["rows"], (j, meta["rows"])
            return
        rec = orjson.loads(line)
        yield rec["id"], rec["text"], start + j
        j += 1


def cmd_write(a):
    for split in SPLITS:
        out = f"{a.staging}/data/{split}"
        os.makedirs(out, exist_ok=True)
        n_shards = 8_000_000 // ROWS_PER_SHARD
        buf, shard, n = {"id": [], "text": [], "src_row": []}, 0, 0

        def flush():
            nonlocal buf, shard
            path = f"{out}/train-{shard:05d}-of-{n_shards:05d}.parquet"
            pq.write_table(pa.table(buf, schema=SCHEMA), path + ".tmp", compression="zstd", row_group_size=10_000)
            os.replace(path + ".tmp", path)
            buf, shard = {"id": [], "text": [], "src_row": []}, shard + 1

        for i, t, r in iter_rows(split):
            buf["id"].append(i)
            buf["text"].append(t)
            buf["src_row"].append(r)
            n += 1
            if len(buf["id"]) == ROWS_PER_SHARD:
                flush()
                print(f"{split}: {n:,} rows", flush=True)
        if buf["id"]:
            flush()
        assert shard == n_shards and n == 8_000_000, (shard, n)
    open(f"{a.staging}/README.md", "w").write(CARD)
    print("done")


def cmd_upload(a):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_large_folder(repo_id=a.repo, repo_type="dataset", folder_path=a.staging,
                            allow_patterns=["README.md", "data/*/*.parquet"])
    print(f"uploaded to https://huggingface.co/datasets/{a.repo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["write", "upload"])
    ap.add_argument("--staging", default=STAGING)
    ap.add_argument("--repo", default=REPO)
    a = ap.parse_args()
    {"write": cmd_write, "upload": cmd_upload}[a.step](a)


if __name__ == "__main__":
    main()
