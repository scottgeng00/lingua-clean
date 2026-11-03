"""Extend our DCLM subsets (dclm_8m_0..3 = global rows 0..32M of the shuffled Dolmino DCLM stream, see
setup/prepare_dclm_subsets.py) with new, disjoint 8M-doc splits dclm_8m_4.. drawn from the raw Dolmino DCLM shards.

The original shuffled stream is not available on this cluster, so the new splits are a fresh uniform draw from the
same corpus (allenai/dolmino-mix-1124, data/dclm, 1970 shards, ~612M docs):

    download  pick --n-shards shards uniformly at random (seeded) and download them
    exclude   uint64 hashes of id (URL) and of exact text for every doc in dclm_8m_0..3 and dclm_heldout
    filter    per shard: keep each doc with probability 1/--keep-every (hash of its id), drop docs whose id or text
              hash is excluded, write kept docs {id, text, src_row} to tmp/kept/<shard>.jsonl
    write     drop duplicates among the kept docs, shuffle them (seeded), take n_new x 8M and write dclm_8m_<k> in the
              lingua layout (row j -> chunk j % 16, line j // 16) with dataset.json
    check     row counts, schema, zero id/text overlap with the excluded docs and across new splits, tokens per doc

src_row of a new doc = shard index * 1_000_000 + line within the shard (shard index into dataset.json["shards"]).
Every phase is resumable; `all` runs them in order.

    .venv-kd/bin/python setup/extend_dclm_subsets.py all
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import orjson

DATA = "/home/sgeng/scottg/pretrain/assets/data/dclm_subsets"
WORK = "/checkpoint/mnm/sgeng/pretrain/tmp_dclm_extend"
REPO = "allenai/dolmino-mix-1124"
EXISTING = ["dclm_8m_0", "dclm_8m_1", "dclm_8m_2", "dclm_8m_3"]
HELDOUT = f"{DATA}/dclm_heldout/dclm_heldout.val.jsonl"
N_CHUNKS = 16
ROWS = 8_000_000
SALT = b"dclm-extend-2026-11:"


def h64(b):
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


def id_hash(doc_id):
    return h64(doc_id.encode("utf-8", "surrogatepass"))


def text_hash(text):
    return h64(text.encode("utf-8", "surrogatepass"))


def keep(doc_id, every):
    return h64(SALT + doc_id.encode("utf-8", "surrogatepass")) % every == 0


# ----------------------------------------------------------------------------- download
def shard_list(a):
    path = f"{WORK}/shards.json"
    if os.path.exists(path):
        return json.load(open(path))["shards"]
    from huggingface_hub import HfApi
    files = sorted(s.rfilename for s in HfApi().dataset_info(REPO).siblings if s.rfilename.startswith("data/dclm/"))
    rng = np.random.default_rng(a.seed)
    pick = sorted(rng.choice(len(files), size=a.n_shards, replace=False).tolist())
    shards = [files[i] for i in pick]
    os.makedirs(WORK, exist_ok=True)
    json.dump({"repo": REPO, "n_files": len(files), "seed": a.seed, "shards": shards}, open(path, "w"), indent=1)
    return shards


def download(a):
    from huggingface_hub import hf_hub_download
    shards = shard_list(a)

    def get(f):
        for attempt in range(10):
            try:
                return hf_hub_download(REPO, f, repo_type="dataset", local_dir=f"{WORK}/raw")
            except Exception as e:  # transient HTTP errors
                print(f"retry {f}: {type(e).__name__}", flush=True)
                time.sleep(30 * (attempt + 1))
        raise RuntimeError(f"failed to download {f}")

    t0 = time.time()
    with ThreadPoolExecutor(a.download_threads) as ex:
        for k, _ in enumerate(ex.map(get, shards)):
            if k % 20 == 0:
                print(f"downloaded {k + 1}/{len(shards)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"download done: {len(shards)} shards in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- exclude
def hash_file(path):
    ids, texts = [], []
    with open(path, "rb", buffering=1 << 24) as f:
        for line in f:
            d = orjson.loads(line)
            ids.append(id_hash(d["id"]))
            texts.append(text_hash(d["text"]))
    return np.asarray(ids, dtype=np.uint64), np.asarray(texts, dtype=np.uint64)


def exclude(a):
    if os.path.exists(f"{WORK}/excl_text.npy"):
        return
    files = [f"{DATA}/{s}/{s}.chunk.{c:02d}.jsonl" for s in EXISTING for c in range(N_CHUNKS)] + [HELDOUT]
    with Pool(min(a.workers, len(files))) as pool:
        res = pool.map(hash_file, files)
    ids = np.unique(np.concatenate([r[0] for r in res]))
    texts = np.unique(np.concatenate([r[1] for r in res]))
    n = sum(len(r[0]) for r in res)
    np.save(f"{WORK}/excl_id.npy", ids)
    np.save(f"{WORK}/excl_text.npy", texts)
    print(f"exclusion set: {n:,} docs, {len(ids):,} unique ids, {len(texts):,} unique texts", flush=True)


# ----------------------------------------------------------------------------- filter
_excl = None


def _load_excl():
    global _excl
    if _excl is None:
        _excl = (np.load(f"{WORK}/excl_id.npy"), np.load(f"{WORK}/excl_text.npy"))
    return _excl


def _in(sorted_arr, x):
    i = np.searchsorted(sorted_arr, x)
    return i < len(sorted_arr) and sorted_arr[i] == x


def filter_shard(task):
    k, shard, every = task
    out_npz = f"{WORK}/kept/{k:04d}.npz"
    if os.path.exists(out_npz):
        return k, None
    ex_id, ex_text = _load_excl()
    raw = f"{WORK}/raw/{shard}"
    tmp = f"{WORK}/kept/{k:04d}.jsonl.tmp"
    offs, lens, hid, htx = [], [], [], []
    n_read = n_sampled = n_excl = pos = 0
    proc = subprocess.Popen(["zstd", "-dc", raw], stdout=subprocess.PIPE, bufsize=1 << 24)
    with open(tmp, "wb", buffering=1 << 24) as out:
        for line_no, line in enumerate(proc.stdout):
            n_read += 1
            d = orjson.loads(line)
            if not keep(d["id"], every):
                continue
            n_sampled += 1
            i, t = id_hash(d["id"]), text_hash(d["text"])
            if _in(ex_id, np.uint64(i)) or _in(ex_text, np.uint64(t)):
                n_excl += 1
                continue
            b = orjson.dumps({"id": d["id"], "text": d["text"], "src_row": k * 1_000_000 + line_no}) + b"\n"
            out.write(b)
            offs.append(pos)
            lens.append(len(b))
            hid.append(i)
            htx.append(t)
            pos += len(b)
    assert proc.wait() == 0, f"zstd failed on {raw}"
    assert n_read < 1_000_000, f"{shard}: {n_read} lines does not fit the src_row encoding"
    os.rename(tmp, f"{WORK}/kept/{k:04d}.jsonl")
    np.savez(out_npz, off=np.asarray(offs, dtype=np.int64), len=np.asarray(lens, dtype=np.int64),
             hid=np.asarray(hid, dtype=np.uint64), htx=np.asarray(htx, dtype=np.uint64),
             stats=np.asarray([n_read, n_sampled, n_excl], dtype=np.int64))
    return k, (n_read, n_sampled, n_excl, len(offs))


def filter_all(a):
    shards = shard_list(a)
    os.makedirs(f"{WORK}/kept", exist_ok=True)
    tasks = [(k, s, a.keep_every) for k, s in enumerate(shards)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        for n, (k, st) in enumerate(pool.imap_unordered(filter_shard, tasks)):
            if n % 20 == 0:
                print(f"filtered {n + 1}/{len(tasks)} ({time.time() - t0:.0f}s) shard {k}: {st}", flush=True)


# ----------------------------------------------------------------------------- write
def write_chunk(task):
    """Write chunk c of new split `name`: rows j = c, c + 16, ... of that split, in order."""
    name, c, tmp, files, offs, lens = task
    handles = {}
    with open(f"{tmp}/{name}.chunk.{c:02d}.jsonl", "wb", buffering=1 << 24) as out:
        for f, o, n in zip(files.tolist(), offs.tolist(), lens.tolist()):
            h = handles.get(f)
            if h is None:
                h = handles[f] = open(f"{WORK}/kept/{f:04d}.jsonl", "rb")
            out.write(os.pread(h.fileno(), n, o))
    for h in handles.values():
        h.close()
    return name, c


def write(a):
    names = [f"dclm_8m_{len(EXISTING) + k}" for k in range(a.n_new)]
    if all(os.path.exists(f"{DATA}/{n}/dataset.json") for n in names):
        print("splits exist", flush=True)
        return
    shards = shard_list(a)
    parts = [np.load(f"{WORK}/kept/{k:04d}.npz") for k in range(len(shards))]
    file_idx = np.concatenate([np.full(len(p["off"]), k, dtype=np.int32) for k, p in enumerate(parts)])
    off = np.concatenate([p["off"] for p in parts])
    ln = np.concatenate([p["len"] for p in parts])
    hid = np.concatenate([p["hid"] for p in parts])
    htx = np.concatenate([p["htx"] for p in parts])
    stats = np.sum([p["stats"] for p in parts], axis=0)
    # Duplicates among the kept docs (same id or same text): keep the first occurrence.
    first = np.zeros(len(off), dtype=bool)
    _, i1 = np.unique(hid, return_index=True)
    _, i2 = np.unique(htx, return_index=True)
    first[np.intersect1d(i1, i2)] = True
    n_dup = int((~first).sum())
    cand = np.flatnonzero(first)
    rng = np.random.default_rng(a.seed + 1)
    order = cand[rng.permutation(len(cand))]
    need = a.n_new * ROWS
    assert len(order) >= need, f"only {len(order):,} candidates for {need:,} rows; raise --n-shards"
    order = order[:need]
    print(f"read {stats[0]:,}, sampled {stats[1]:,}, excluded {stats[2]:,}, duplicates {n_dup:,}, "
          f"candidates {len(cand):,}; writing {need:,}", flush=True)

    tasks = []
    for k, name in enumerate(names):
        tmp = f"{DATA}/{name}.tmp-{os.getpid()}"
        os.makedirs(tmp, exist_ok=True)
        rows = order[k * ROWS:(k + 1) * ROWS]
        for c in range(N_CHUNKS):
            sel = rows[c::N_CHUNKS]
            tasks.append((name, c, tmp, file_idx[sel], off[sel], ln[sel]))
    with Pool(min(a.workers, len(tasks))) as pool:
        for n, (name, c) in enumerate(pool.imap_unordered(write_chunk, tasks)):
            if n % 16 == 0:
                print(f"wrote {n + 1}/{len(tasks)}", flush=True)
    for k, name in enumerate(names):
        tmp = f"{DATA}/{name}.tmp-{os.getpid()}"
        json.dump({"source": f"{REPO} data/dclm (fresh draw; see setup/extend_dclm_subsets.py)", "config": name,
                   "rows": ROWS, "n_chunks": N_CHUNKS, "layout": "row j -> chunk j % 16, line j // 16",
                   "draw": {"shard_seed": a.seed, "shuffle_seed": a.seed + 1, "n_shards": len(shards),
                            "keep_every": a.keep_every, "split_index": k, "read": int(stats[0]),
                            "sampled": int(stats[1]), "excluded_overlap": int(stats[2]), "duplicates": n_dup},
                   "excluded": EXISTING + ["dclm_heldout"],
                   "src_row": "shard index * 1_000_000 + line within the shard; shard index into shards",
                   "shards": shards}, open(f"{tmp}/dataset.json", "w"), indent=1)
        os.rename(tmp, f"{DATA}/{name}")
    print(f"wrote {names}", flush=True)


# ----------------------------------------------------------------------------- check
def check(a):
    from transformers import AutoTokenizer
    names = [f"dclm_8m_{len(EXISTING) + k}" for k in range(a.n_new)]
    files = [f"{DATA}/{n}/{n}.chunk.{c:02d}.jsonl" for n in names for c in range(N_CHUNKS)]
    with Pool(min(a.workers, len(files))) as pool:
        res = pool.map(hash_file, files)
    per_split = [sum(len(r[0]) for r in res[k * N_CHUNKS:(k + 1) * N_CHUNKS]) for k in range(len(names))]
    ids = np.concatenate([r[0] for r in res])
    texts = np.concatenate([r[1] for r in res])
    ex_id, ex_text = np.load(f"{WORK}/excl_id.npy"), np.load(f"{WORK}/excl_text.npy")
    keys = set(orjson.loads(open(files[0], "rb").readline()).keys())
    tok = AutoTokenizer.from_pretrained("/home/sgeng/scottg/pretrain/assets/models/Qwen3-0.6B-Base-tokenizer")

    def tokens_per_doc(paths, every=1000):
        n = t = 0
        for p in paths:
            for i, line in enumerate(open(p, "rb")):
                if i % every == 0:
                    t += len(tok(orjson.loads(line)["text"], add_special_tokens=False)["input_ids"]) + 1
                    n += 1
        return t / n

    report = {
        "rows_per_split": dict(zip(names, per_split)),
        "schema": sorted(keys),
        "overlap_with_existing_ids": int(np.isin(ids, ex_id).sum()),
        "overlap_with_existing_texts": int(np.isin(texts, ex_text).sum()),
        "duplicate_ids_within_new": int(len(ids) - len(np.unique(ids))),
        "duplicate_texts_within_new": int(len(texts) - len(np.unique(texts))),
        "tokens_per_doc_new": round(tokens_per_doc(files[::8]), 1),
        "tokens_per_doc_existing": round(tokens_per_doc(
            [f"{DATA}/{s}/{s}.chunk.00.jsonl" for s in EXISTING]), 1),
    }
    json.dump(report, open(f"{WORK}/check.json", "w"), indent=1)
    print(json.dumps(report, indent=1), flush=True)
    assert all(n == ROWS for n in per_split) and keys == {"id", "text", "src_row"}
    assert report["overlap_with_existing_ids"] == report["overlap_with_existing_texts"] == 0
    assert report["duplicate_ids_within_new"] == report["duplicate_texts_within_new"] == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["download", "exclude", "filter", "write", "check", "all"])
    ap.add_argument("--n-shards", type=int, default=330)
    ap.add_argument("--keep-every", type=int, default=3)
    ap.add_argument("--n-new", type=int, default=4, help="number of new 8M-doc splits")
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--download-threads", type=int, default=16)
    a = ap.parse_args()
    phases = ["download", "exclude", "filter", "write", "check"] if a.phase == "all" else [a.phase]
    for p in phases:
        t0 = time.time()
        print(f"=== {p}", flush=True)
        {"download": download, "exclude": exclude, "filter": filter_all, "write": write, "check": check}[p](a)
        print(f"=== {p} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
