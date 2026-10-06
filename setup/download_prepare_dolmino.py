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
      <domain>_shuffled/<domain>.val.jsonl   # ~34 val docs per domain

The `<domain>_shuffled/` directories are what the recipe YAMLs reference under
`data.sources:` (e.g. `dclm_shuffled: 0.472`). Skip the raw dir to save ~1.5 TB
after shuffling completes by passing `--cleanup-raw`.

Requires the `terashuf` binary; if missing, the script clones + builds it into
`${data_dir}/terashuf/`.

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
import os
import subprocess
import time
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


def shuffle_domain(domain: str, raw_dir: Path, out_dir: Path, terashuf: Path,
                   memory_gb: float, seed: int, nchunks: int, val_docs: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{domain}.chunk."
    suffix = ".jsonl"
    name_expr = " -o ".join(f"-name '{pat}'" for pat in RAW_PATTERNS)
    run(
        "set -o pipefail && ulimit -n 100000 && "
        f"find {raw_dir} -type f \\( {name_expr} \\) -print0 | "
        # Decompress by extension. The trailing echo keeps files that lack a final
        # newline (e.g. math/*.jsonl) from gluing onto the next file's first record;
        # the resulting blank lines are dropped by grep.
        "xargs -0 -n 1 sh -c 'case \"$0\" in *.zst) zstdcat -- \"$0\" ;; *) zcat -f -- \"$0\" ;; esac; echo' | "
        "LC_ALL=C grep -v '^$' | "
        f"MEMORY={memory_gb} SEED={seed} {terashuf} | "
        f"split -n r/{nchunks} -d --suffix-length=2 --additional-suffix={suffix} "
        f"- {out_dir}/{prefix}"
    )

    # Pull a small validation slice off the head of each chunk.
    val_path = out_dir / f"{domain}.val.jsonl"
    remaining = val_docs
    for i in range(nchunks):
        if remaining <= 0:
            break
        chunk = out_dir / f"{prefix}{i:02d}{suffix}"
        if not chunk.exists():
            continue
        with chunk.open() as f:
            n_lines = sum(1 for _ in f)
        take = min(remaining, n_lines)
        if take > 0:
            run(f"head -n {take} {chunk} >> {val_path}")
            run(f"sed -i '1,{take}d' {chunk}")
            remaining -= take


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True,
                    help="Root for raw + shuffled splits (set to ${DATA_ROOT}).")
    ap.add_argument("--memory", type=float, default=64,
                    help="terashuf RAM budget in GB (more = faster, less spill).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nchunks", type=int, default=32,
                    help="Per-domain output chunks. 32 matches the in-house layout.")
    ap.add_argument("--val-docs", type=int, default=34,
                    help="Validation docs per domain (head-of-chunk; ~34*30=1020 total).")
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

    terashuf = setup_terashuf(data_dir)

    for domain in args.domains:
        raw_dir = raw_root / domain
        files = raw_files(raw_dir) if raw_dir.exists() else []
        if not files:
            raise RuntimeError(f"{domain}: no raw shards matching {RAW_PATTERNS} under {raw_dir}")
        n_other = sum(1 for f in raw_dir.rglob("*") if f.is_file()) - len(files)
        print(f"[shuffle] {domain}: {len(files)} raw shards"
              + (f" (WARNING: {n_other} other files ignored)" if n_other else ""))
        out_dir = data_dir / f"{domain}_shuffled"
        print(f"[shuffle] {domain} → {out_dir}")
        shuffle_domain(domain, raw_dir, out_dir, terashuf,
                       args.memory, args.seed, args.nchunks, args.val_docs)

    if args.cleanup_raw and not args.skip_download:
        print(f"[cleanup] rm -rf {raw_root}")
        run(f"rm -rf {raw_root}")

    print("[done] Dolmino splits prepared.")


if __name__ == "__main__":
    main()
