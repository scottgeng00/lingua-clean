"""Shuffle a Dolmino domain that fits in RAM into N chunks, in parallel.

Equivalent to download_prepare_dolmino.py's shuffle when terashuf would hold the
whole domain in one buffer (a uniform shuffle, dealt round-robin into chunks,
first --val-docs lines of chunk 00 moved to <prefix>.val.jsonl), without the
single-threaded decompress -> terashuf -> `split -n r/N` pipeline (~100 MB/s).

Raw shards (.json/.jsonl, plain/.gz/.zst) are decompressed by parallel
zcat/zstdcat processes; empty lines are dropped and every line is newline-
terminated, matching the prep script. Every line is parsed (orjson) and must
carry a "text" field; the run fails on any bad line or decompressor error.

Usage:
    python setup/shuffle_in_memory.py --raw-dir dolmino_mix_raw/pes2o \
        --out-dir pes2o_shuffled --prefix pes2o --nchunks 16
    python setup/shuffle_in_memory.py --raw-dir dolmino_mix_raw/math --count-only
"""
import argparse
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import orjson

RAW_PATTERNS = ["*.json", "*.jsonl", "*.json.gz", "*.jsonl.gz", "*.json.zst", "*.jsonl.zst"]


def raw_files(raw_dir: Path) -> list[Path]:
    return sorted({f for pat in RAW_PATTERNS for f in raw_dir.rglob(pat) if f.is_file()})


def read_lines(path: Path) -> list[bytes]:
    cmd = ["zstdcat", "--", str(path)] if path.suffix == ".zst" else ["zcat", "-f", "--", str(path)]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return [line + b"\n" for line in out.split(b"\n") if line]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir")
    ap.add_argument("--prefix")
    ap.add_argument("--nchunks", type=int, default=16)
    ap.add_argument("--val-docs", type=int, default=34)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--count-only", action="store_true", help="read + validate, print the line count, write nothing")
    args = ap.parse_args()
    if not args.count_only and not (args.out_dir and args.prefix):
        ap.error("--out-dir and --prefix are required unless --count-only")

    t0 = time.time()
    files = raw_files(Path(args.raw_dir))
    if not files:
        sys.exit(f"no raw shards under {args.raw_dir}")
    with ThreadPoolExecutor(args.threads) as pool:
        per_file = list(pool.map(read_lines, files))  # keeps file order -> deterministic
    lines = [line for chunk in per_file for line in chunk]
    del per_file
    nbytes = sum(map(len, lines))
    print(f"[mem-shuffle] {args.raw_dir}: {len(files)} shards, {len(lines)} lines, {nbytes / 1e9:.1f} GB "
          f"read in {time.time() - t0:.0f}s", flush=True)

    bad = 0
    for line in lines:
        try:
            if "text" not in orjson.loads(line):
                bad += 1
        except orjson.JSONDecodeError:
            bad += 1
    if bad:
        sys.exit(f"[mem-shuffle] FAILED: {bad} lines without a parseable 'text' field")
    if args.count_only:
        print(json.dumps({"raw_dir": args.raw_dir, "lines": len(lines), "bytes": nbytes, "bad": bad}), flush=True)
        return

    random.Random(args.seed).shuffle(lines)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"{args.prefix}.*.jsonl"):
        old.unlink()

    def write_chunk(c: int) -> int:
        part = lines[c::args.nchunks]
        if c == 0 and args.val_docs:
            with open(out / f"{args.prefix}.val.jsonl", "wb") as f:
                f.writelines(part[:args.val_docs])
            part = part[args.val_docs:]
        with open(out / f"{args.prefix}.chunk.{c:02d}.jsonl", "wb", buffering=64 << 20) as f:
            f.writelines(part)
        return len(part)

    with ThreadPoolExecutor(args.nchunks) as pool:
        written = sum(pool.map(write_chunk, range(args.nchunks)))
    total = written + min(args.val_docs, len(lines[0::args.nchunks]))
    print(json.dumps({"prefix": args.prefix, "lines": len(lines), "written": total, "bad": bad,
                      "seconds": round(time.time() - t0)}), flush=True)
    if total != len(lines):
        sys.exit(f"[mem-shuffle] FAILED: wrote {total} of {len(lines)} lines")
    print("[mem-shuffle] OK", flush=True)


if __name__ == "__main__":
    main()
