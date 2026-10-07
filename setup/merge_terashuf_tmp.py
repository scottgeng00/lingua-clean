"""Turn terashuf's temp files into N training chunks, in parallel.

terashuf writes each full memory buffer to a temp file as a uniformly shuffled
block of lines, then merges the blocks into stdout. Its merge feeds a single
`split -n r/N`, which tops out at ~100 MB/s; this script replaces that step
when the temp files already exist (e.g. after cancelling a slow prep job).

Worker c writes <prefix>.chunk.<c>.jsonl from the c-th 1/N byte range of every
temp file (aligned to line boundaries), interleaving the ranges at random with
probability proportional to their remaining bytes. Each range is a uniform
sample of its (shuffled) block, so every chunk gets 1/N of each block in random
order -- a stratified version of terashuf's global merge. Worker 0 diverts its
first --val-docs lines to <prefix>.val.jsonl, matching download_prepare_dolmino.py.

Every line is parsed (orjson) and must carry a "text" field. With
--expected-lines the run fails unless the outputs hold exactly that many lines,
and per-file line counts are reported so they can be checked against
terashuf's "lines read" log.

Usage:
    python setup/merge_terashuf_tmp.py --tmp-files tmp/terashuftmp* \
        --out-dir dclm_shuffled --prefix dclm --nchunks 16 --expected-lines 606266382
"""
import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import orjson

READ_BLOCK = 64 << 20
WRITE_BUF = 64 << 20


def aligned_offset(path: str, size: int, off: int) -> int:
    """First line start at or after byte `off` (lines end with b"\\n")."""
    if off <= 0:
        return 0
    if off >= size:
        return size
    with open(path, "rb") as f:
        f.seek(off - 1)
        f.readline()  # consume through the newline that ends the line containing off-1
        return f.tell()


def iter_lines(path: str, start: int, end: int):
    """Yield complete lines (with trailing newline) in the byte range [start, end)."""
    with open(path, "rb", buffering=0) as f:
        f.seek(start)
        remaining, carry = end - start, b""
        while remaining > 0:
            block = f.read(min(READ_BLOCK, remaining))
            if not block:
                raise RuntimeError(f"unexpected EOF in {path} at {end - remaining}")
            remaining -= len(block)
            lines = (carry + block).split(b"\n")
            carry = lines.pop()
            for line in lines:
                yield line + b"\n"
        if carry:
            raise RuntimeError(f"range [{start}, {end}) of {path} does not end on a line boundary")


def run_worker(job):
    c, files, nchunks, out_dir, prefix, val_docs, seed, limit_bytes = job
    rng = random.Random(seed * 100003 + c)
    ranges = []
    for path in files:
        size = os.path.getsize(path)
        start = aligned_offset(path, size, c * size // nchunks)
        end = aligned_offset(path, size, (c + 1) * size // nchunks)
        if limit_bytes:
            end = aligned_offset(path, size, min(end, start + limit_bytes))
        ranges.append((path, start, end))

    iters = [iter_lines(p, s, e) for p, s, e in ranges]
    remaining = [e - s for _, s, e in ranges]
    lines_per_file = [0] * len(files)
    bad = 0
    chunk_path = Path(out_dir) / f"{prefix}.chunk.{c:02d}.jsonl"
    out = open(chunk_path, "wb", buffering=WRITE_BUF)
    val = open(Path(out_dir) / f"{prefix}.val.jsonl", "wb") if (c == 0 and val_docs > 0) else None
    n_val = n_chunk = 0
    live = [i for i, r in enumerate(remaining) if r > 0]
    while live:
        # draw a batch of source indices with the current byte weights
        for i in rng.choices(live, weights=[remaining[j] for j in live], k=256):
            if remaining[i] <= 0:
                continue
            try:
                line = next(iters[i])
            except StopIteration:
                remaining[i] = 0
                continue
            remaining[i] -= len(line)
            lines_per_file[i] += 1
            try:
                if "text" not in orjson.loads(line):
                    bad += 1
            except orjson.JSONDecodeError:
                bad += 1
            if val is not None and n_val < val_docs:
                val.write(line)
                n_val += 1
            else:
                out.write(line)
                n_chunk += 1
        live = [i for i in live if remaining[i] > 0]
    out.close()
    if val is not None:
        val.close()
    for it in iters:  # every range must be fully consumed
        if next(it, None) is not None:
            raise RuntimeError(f"worker {c}: a range was not fully consumed")
    return dict(chunk=c, chunk_lines=n_chunk, val_lines=n_val, bad=bad,
                lines_per_file=lines_per_file, bytes=sum(e - s for _, s, e in ranges))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tmp-files", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True, help="e.g. dclm -> dclm.chunk.NN.jsonl")
    ap.add_argument("--nchunks", type=int, default=16)
    ap.add_argument("--val-docs", type=int, default=34)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=None, help="default: nchunks")
    ap.add_argument("--expected-lines", type=int, default=None,
                    help="fail unless chunks + val hold exactly this many lines")
    ap.add_argument("--limit-bytes", type=int, default=0,
                    help="testing only: read at most this many bytes per (file, chunk) range")
    args = ap.parse_args()

    files = sorted(args.tmp_files)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[merge] {len(files)} tmp files, {sum(os.path.getsize(f) for f in files) / 1e9:.0f} GB "
          f"-> {args.nchunks} chunks in {args.out_dir}", flush=True)
    t0 = time.time()
    jobs = [(c, files, args.nchunks, args.out_dir, args.prefix, args.val_docs, args.seed, args.limit_bytes)
            for c in range(args.nchunks)]
    results = []
    with Pool(args.workers or args.nchunks) as pool:
        for r in pool.imap_unordered(run_worker, jobs):
            results.append(r)
            print(f"[merge] chunk {r['chunk']:02d}: {r['chunk_lines']} lines (+{r['val_lines']} val), "
                  f"{r['bytes'] / 1e9:.1f} GB, {r['bad']} bad, {time.time() - t0:.0f}s", flush=True)

    total = sum(r["chunk_lines"] + r["val_lines"] for r in results)
    bad = sum(r["bad"] for r in results)
    per_file = [sum(r["lines_per_file"][i] for r in results) for i in range(len(files))]
    summary = dict(total_lines=total, bad_lines=bad, seconds=round(time.time() - t0),
                   lines_per_file={Path(f).name: n for f, n in zip(files, per_file)})
    print("[merge] summary: " + json.dumps(summary), flush=True)
    ok = bad == 0 and (args.expected_lines is None or total == args.expected_lines)
    if not ok:
        print(f"[merge] FAILED: bad={bad}, total={total}, expected={args.expected_lines}", flush=True)
        sys.exit(1)
    print("[merge] OK", flush=True)


if __name__ == "__main__":
    main()
