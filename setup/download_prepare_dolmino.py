"""Download `allenai/dolmino-mix-1124` and shuffle into per-domain chunks.

End layout, with `--data-dir ${DATA_ROOT}`:

    ${DATA_ROOT}/
      dolmino_mix_raw/                       # raw HF download (jsonl.gz)
        dclm/, flan/, math/, pes2o/, stackexchange/, wiki/
      dclm_shuffled/dclm.chunk.{00..NN}.jsonl
      flan_shuffled/flan.chunk.{00..NN}.jsonl
      math_shuffled/math.chunk.{00..NN}.jsonl
      pes2o_shuffled/pes2o.chunk.{00..NN}.jsonl
      stackexchange_shuffled/stackexchange.chunk.{00..NN}.jsonl
      wiki_shuffled/wiki.chunk.{00..NN}.jsonl
      <domain>_shuffled/<domain>.val.jsonl   # 1000 held-out val docs per domain

The `<domain>_shuffled/` directories are what the recipe YAMLs reference under
`data.sources:` (e.g. `dclm_shuffled: 0.472`). Skip the raw dir to save ~1.5 TB
after shuffling completes by passing `--cleanup-raw`.

Each domain is shuffled with one of two strategies, picked by the size of its raw
(compressed) shards:
  - large (> --in-memory-max-raw-gb, e.g. dclm, pes2o): decompress -> terashuf
    (external-memory shuffle), with terashuf's output dealt round-robin into the
    chunk files by a buffered writer in this process (GNU `split -n r/N` tops out
    at ~100 MB/s, which made the dclm merge take ~12h);
  - small (flan, math, wiki, stackexchange): all shards are decompressed in
    parallel, shuffled in memory, and written out the same way.
Both are uniform shuffles dealt round-robin into chunks; the first --val-docs
lines of the shuffled stream go to <domain>.val.jsonl. Every domain is checked
(line count vs. what was read, every line a JSON object) and gets a
`.prep_done.json` marker, so re-running skips finished domains.

Requires the `terashuf` binary for large domains; if missing, the script clones
+ builds it into `${data_dir}/terashuf/`.

Sample usage:

    # Full mix (~2 TB raw, ~2 TB shuffled; days on a single node).
    python setup/download_prepare_dolmino.py \\
        --data-dir ${DATA_ROOT} \\
        --memory 64

    # Single domain for testing:
    python setup/download_prepare_dolmino.py \\
        --data-dir ${DATA_ROOT} \\
        --memory 16 \\
        --domains math
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

HF_REPO = "allenai/dolmino-mix-1124"
DOMAINS = ["dclm", "flan", "math", "pes2o", "stackexchange", "wiki"]
# Raw shard formats vary by domain (dclm: .json.zst, flan: .json.gz, math: mostly
# plain .jsonl), all JSON-lines with a "text" field.
RAW_PATTERNS = ["*.json", "*.jsonl", "*.json.gz", "*.jsonl.gz", "*.json.zst", "*.jsonl.zst"]


def run(cmd: str):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")


def snapshot_with_retries(repo_id, local_dir, allow_patterns, max_retries=20, delay=300):
    # delay matches the Hub's 5-minute rate-limit window (HTTP 429); already
    # downloaded files are skipped on each retry.
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                resume_download=True,
                max_workers=16,
            )
            return
        except (requests.exceptions.ReadTimeout, HfHubHTTPError, LocalEntryNotFoundError) as e:
            if attempt == max_retries - 1:
                raise
            print(f"[retry {attempt + 1}/{max_retries}] {type(e).__name__}: {str(e)[:200]}; sleeping {delay}s")
            time.sleep(delay)


def setup_terashuf(parent_dir: Path) -> Path:
    terashuf_dir = parent_dir / "terashuf"
    terashuf_bin = terashuf_dir / "terashuf"
    if terashuf_bin.exists():
        return terashuf_bin
    print("[terashuf] building...")
    run(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    run(f"make -C {terashuf_dir}")
    return terashuf_bin


def raw_files(raw_dir: Path) -> list[Path]:
    return sorted({f for pat in RAW_PATTERNS for f in raw_dir.rglob(pat) if f.is_file()})


def decompress_cmd(path: Path) -> list[str]:
    return ["zstdcat", "--", str(path)] if path.suffix == ".zst" else ["zcat", "-f", "--", str(path)]


def is_json_object_line(line: bytes) -> bool:
    return line[:1] == b"{" and line.rstrip()[-1:] == b"}"


class RoundRobinWriter:
    """Deal lines into N chunk files (first `val_docs` lines to the val file) with big buffers."""

    def __init__(self, out_dir: Path, domain: str, nchunks: int, val_docs: int, buffer_bytes: int = 64 << 20):
        self.nchunks, self.val_docs = nchunks, val_docs
        self.val = open(out_dir / f"{domain}.val.jsonl", "wb")
        self.chunks = [open(out_dir / f"{domain}.chunk.{i:02d}.jsonl", "wb", buffering=buffer_bytes)
                       for i in range(nchunks)]
        self.n_val = self.n_chunk = self.bad = 0

    def write_block(self, lines: list[bytes]):
        """Write a block of lines (each without its trailing newline), in stream order."""
        self.bad += sum(1 for line in lines if not is_json_object_line(line))
        lines = [line + b"\n" for line in lines]
        if self.n_val < self.val_docs:
            k = min(self.val_docs - self.n_val, len(lines))
            self.val.writelines(lines[:k])
            self.n_val += k
            lines = lines[k:]
        # line j of this block goes to chunk (n_chunk + j) % nchunks
        for c in range(self.nchunks):
            self.chunks[(self.n_chunk + c) % self.nchunks].writelines(lines[c::self.nchunks])
        self.n_chunk += len(lines)

    def close(self):
        for f in [self.val, *self.chunks]:
            f.close()

    @property
    def total(self) -> int:
        return self.n_val + self.n_chunk


def shuffle_large(domain: str, files: list[Path], out_dir: Path, terashuf: Path,
                  memory_gb: float, seed: int, nchunks: int, val_docs: int) -> dict:
    """decompress -> terashuf -> round-robin writer (replaces `split -n r/N`)."""
    file_list = out_dir / ".raw_files.list"
    file_list.write_bytes(b"\0".join(str(f).encode() for f in files) + b"\0")
    terashuf_log = out_dir / ".terashuf.log"
    cmd = (
        "set -o pipefail && ulimit -n 100000 && "
        f"xargs -0 -n 1 -a {file_list} "
        # Decompress by extension. The trailing echo keeps files that lack a final
        # newline (e.g. math/*.jsonl) from gluing onto the next file's first record;
        # the resulting blank lines are dropped by grep.
        "sh -c 'case \"$0\" in *.zst) zstdcat -- \"$0\" ;; *) zcat -f -- \"$0\" ;; esac; echo' | "
        "LC_ALL=C grep -v '^$' | "
        f"MEMORY={memory_gb} SEED={seed} {terashuf} 2> {terashuf_log}"
    )
    print(f"$ {cmd}", flush=True)
    writer = RoundRobinWriter(out_dir, domain, nchunks, val_docs)
    proc = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, bufsize=0)
    carry = b""
    while block := proc.stdout.read(64 << 20):
        lines = (carry + block).split(b"\n")
        carry = lines.pop()
        writer.write_block(lines)
    if carry:
        writer.write_block([carry])
    rc = proc.wait()
    writer.close()
    if rc != 0:
        raise RuntimeError(f"{domain}: shuffle pipeline failed (exit {rc}); see {terashuf_log}")
    m = re.findall(r"Read (\d+) lines", terashuf_log.read_text(errors="replace"))
    read = int(m[-1]) if m else None
    file_list.unlink()
    return {"method": "terashuf+round-robin", "read": read, "written": writer.total, "val": writer.n_val, "bad": writer.bad}


def read_shard(path: Path) -> list[bytes]:
    out = subprocess.run(decompress_cmd(path), stdout=subprocess.PIPE, check=True).stdout
    return [line for line in out.split(b"\n") if line]


def shuffle_in_memory(domain: str, files: list[Path], out_dir: Path, seed: int,
                      nchunks: int, val_docs: int, threads: int) -> dict:
    """Parallel decompress, uniform in-memory shuffle, round-robin write."""
    with ThreadPoolExecutor(threads) as pool:
        per_file = list(pool.map(read_shard, files))  # keeps file order -> deterministic
    lines = [line for shard in per_file for line in shard]
    del per_file
    read = len(lines)
    random.Random(seed).shuffle(lines)
    writer = RoundRobinWriter(out_dir, domain, nchunks, val_docs)
    for i in range(0, len(lines), 1 << 20):
        writer.write_block(lines[i:i + (1 << 20)])
    writer.close()
    return {"method": "in-memory", "read": read, "written": writer.total, "val": writer.n_val, "bad": writer.bad}


def prepare_domain(domain: str, raw_dir: Path, out_dir: Path, data_dir: Path, args) -> None:
    done = out_dir / ".prep_done.json"
    chunk_glob = f"{domain}.chunk.*.jsonl"
    if done.exists() and len(list(out_dir.glob(chunk_glob))) == args.nchunks:
        print(f"[skip] {domain}: already prepared ({json.loads(done.read_text())})")
        return

    files = raw_files(raw_dir) if raw_dir.exists() else []
    if not files:
        raise RuntimeError(f"{domain}: no raw shards matching {RAW_PATTERNS} under {raw_dir}")
    n_other = sum(1 for f in raw_dir.rglob("*") if f.is_file()) - len(files)
    raw_gb = sum(f.stat().st_size for f in files) / 1e9
    large = raw_gb > args.in_memory_max_raw_gb
    print(f"[shuffle] {domain}: {len(files)} raw shards, {raw_gb:.1f} GB raw -> "
          f"{'terashuf + round-robin' if large else 'in-memory'} shuffle into {out_dir}"
          + (f" (WARNING: {n_other} other files ignored)" if n_other else ""), flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in [*out_dir.glob(chunk_glob), out_dir / f"{domain}.val.jsonl", done]:
        stale.unlink(missing_ok=True)
    t0 = time.time()
    if large:
        stats = shuffle_large(domain, files, out_dir, setup_terashuf(data_dir),
                              args.memory, args.seed, args.nchunks, args.val_docs)
    else:
        stats = shuffle_in_memory(domain, files, out_dir, args.seed, args.nchunks, args.val_docs, args.threads)
    stats.update(domain=domain, nchunks=args.nchunks, seed=args.seed, seconds=round(time.time() - t0))
    print(f"[shuffle] {domain}: {stats}", flush=True)
    if stats["read"] != stats["written"]:
        raise RuntimeError(f"{domain}: read {stats['read']} lines but wrote {stats['written']}")
    if stats["bad"]:
        raise RuntimeError(f"{domain}: {stats['bad']} lines are not JSON objects")
    done.write_text(json.dumps(stats))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True,
                    help="Root for raw + shuffled splits (set to ${DATA_ROOT}).")
    ap.add_argument("--memory", type=float, default=64,
                    help="terashuf RAM budget in GB (more = faster, less spill).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nchunks", type=int, default=32,
                    help="Per-domain output chunks. 32 matches the in-house layout.")
    ap.add_argument("--val-docs", type=int, default=1000,
                    help="Validation docs per domain (first lines of the shuffled stream).")
    ap.add_argument("--in-memory-max-raw-gb", type=float, default=50,
                    help="Domains whose raw shards exceed this (compressed GB) go through terashuf; "
                         "smaller ones are shuffled in memory (needs RAM for the decompressed text).")
    ap.add_argument("--threads", type=int, default=48,
                    help="Parallel decompression threads for in-memory domains.")
    ap.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS,
                    help="Subset of domains to process.")
    ap.add_argument("--skip-download", action="store_true",
                    help="Assume raw shards already at <data_dir>/dolmino_mix_raw/<domain>/.")
    ap.add_argument("--cleanup-raw", action="store_true",
                    help="rm -rf the raw dir after shuffling (saves ~1.5 TB).")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_root = data_dir / "dolmino_mix_raw"

    if not args.skip_download:
        # Pull only the domains we'll process. The repo's layout is `data/<domain>/...`.
        patterns = [f"data/{d}/*" for d in args.domains]
        print(f"[download] {HF_REPO} → {raw_root} (patterns={patterns})")
        snapshot_with_retries(HF_REPO, str(raw_root), patterns)
        # Move <raw>/data/<domain> up to <raw>/<domain> so the path matches DOMAINS.
        nested = raw_root / "data"
        if nested.exists():
            for d in args.domains:
                src = nested / d
                if src.exists():
                    dst = raw_root / d
                    if not dst.exists():
                        src.rename(dst)
            try:
                nested.rmdir()
            except OSError:
                pass

    for domain in args.domains:
        prepare_domain(domain, raw_root / domain, data_dir / f"{domain}_shuffled", data_dir, args)

    if args.cleanup_raw and not args.skip_download:
        print(f"[cleanup] rm -rf {raw_root}")
        run(f"rm -rf {raw_root}")

    print("[done] Dolmino splits prepared.")


if __name__ == "__main__":
    main()
