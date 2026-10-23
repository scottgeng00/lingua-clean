"""Write a simplescale pointer manifest over a DCLM subset, in the subset's row order.

Task j points at local row j of the subset (chunk j % 16, line j // 16), so simplescale's
output row j is the rewrite of subset row j and setup/prepare_dclm_rewrites.py can write it
back in the same layout. One manifest serves every generator/prompt (the handler and model
are chosen at submit time).

    python setup/build_dclm_rewrite_manifest.py \
        /home/sgeng/scottg/pretrain/assets/data/dclm_subsets/dclm_8m_1 \
        /checkpoint/dream-sc-3/scottg/pretrain/simplescale/runs/manifests/dclm_8m_1.g1.max4096.manifest.jsonl
"""

import argparse
import glob
import json
import os

N_CHUNKS = 16


def chunk_files(subset_dir):
    name = os.path.basename(os.path.normpath(subset_dir))
    files = sorted(glob.glob(os.path.join(subset_dir, f"{name}.chunk.*.jsonl")))
    if len(files) != N_CHUNKS:
        raise SystemExit(f"expected {N_CHUNKS} chunk files in {subset_dir}, found {len(files)}")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subset_dir")
    ap.add_argument("output")
    ap.add_argument("--generations", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--max-document-tokens", type=int, default=11000)  # + prompt + output < 16k context
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    files = chunk_files(args.subset_dir)
    meta = json.load(open(os.path.join(args.subset_dir, "dataset.json")))
    task_extra = {
        "generations": args.generations,
        "sampling_params": {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens},
        "max_document_tokens": args.max_document_tokens,
    }
    tmp = args.output + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    readers = [open(f, "rb") for f in files]
    offsets = [0] * N_CHUNKS
    rows = 0
    with open(tmp, "w") as out:
        while True:
            c = rows % N_CHUNKS
            line = readers[c].readline()
            if not line:
                break
            ref = {"path": files[c], "offset": offsets[c], "length": len(line)}
            out.write(json.dumps({"_ref": ref, **task_extra}, separators=(",", ":")) + "\n")
            offsets[c] += len(line)
            rows += 1
    # round-robin layout: every chunk must be exhausted exactly when the walk stops
    leftovers = [bool(r.readline()) for r in readers]
    if any(leftovers) or rows != meta["rows"]:
        os.remove(tmp)
        raise SystemExit(f"walked {rows} rows, dataset.json says {meta['rows']}; leftovers {leftovers}")
    os.replace(tmp, args.output)
    print(f"{rows} tasks -> {args.output}")


if __name__ == "__main__":
    main()
