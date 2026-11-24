"""simplescale manifests for the multi-sample generation study: 8 Qwen3.5-9B samples per DCLM document, one manifest
per arm. Sources are walked in row order (row j = chunk j % 16, line j // 16: the training stream's order, so for
dclm_8m_0 they come from the real data the mixed runs see), keeping docs with at least --min-words words (stripped
text; no upper limit) whose stripped text is not an exact duplicate of an earlier kept source (blake2b, as in
setup/build_gacha_rewrite_manifests.py). The first --sources survivors are written; every row records
source_subset / source_row (row within the subset), and the simplescale harness copies them plus the DCLM id into
its output.

Arms (handlers in simplescale.harnesses.multisample):
    direct_nothink  wiki x 8 (WRAP / Megadocs Wikipedia-style rephrase), thinking off
    direct_think    wiki x 8, thinking on
    mix             one sample each of 7 FinePhrase format prompts + wiki, thinking off
    ideas           Active Reading: 8 strategies (thinking on), each applied in its own call (thinking off)
    basegen         base model (Qwen3.5-9B-Base) writes "Document 2" after the source as "Document 1" (base_continue)
    arv2            Active Reading v2 (active_reading_v2): planned alternative web documents, 8-48 per source
                    (python3 setup/build_multisample_manifests.py --sources 1503775 --tag pool025 --arms arv2)
    mix16           8 of 16 pool prompts (prompt_pool), drawn uniformly; mix16plan: weighted by a planner call
                    (python3 setup/build_multisample_manifests.py --sources 1503775 --tag pool025 --arms mix16 mix16plan)

    python3 setup/build_multisample_manifests.py --sources 2000 --min-words 0 --no-dedup --tag pilot2k
    python3 setup/build_multisample_manifests.py --sources 3000000 --tag f200w_3m
    python3 setup/build_multisample_manifests.py --sources 1503775 --tag pool025_r2 --arms mix --mix-repeats 2  # 16 views
"""

import argparse
import hashlib
import json
import os

from build_dclm_rewrite_manifest import N_CHUNKS, chunk_files

DATA = "/home/sgeng/scottg/pretrain/assets/data"
OUT = "/home/sgeng/scottg/pretrain/simplescale/runs/manifests"
NOTHINK = {"max_new_tokens": 4096}
THINK = {"max_new_tokens": 16384}
MIX = ["article", "discussion", "faq", "math", "narrative", "table", "tutorial", "wiki"]  # multisample.MIX
ARMS = {
    # handler, task fields, context length
    "direct_nothink": ("wiki", {"thinking": False, "sampling_params": NOTHINK}, 16384),
    "direct_think": ("wiki", {"thinking": True, "sampling_params": THINK}, 32768),
    "mix": ("prompt_mix", {"thinking": False, "sampling_params": NOTHINK}, 16384),
    "ideas": ("active_reading", {"thinking": False, "strategy_thinking": True, "strategy_max_new_tokens": 16384,
                                 "sampling_params": NOTHINK}, 32768),
    # Qwen3.5-9B-BASE (--model .../Qwen3.5-9B-Base): raw "## Document 1 / ## Document 2" continuation, the harness's
    # BASE_PARAMS (T 1.0, top_p 0.95, presence 0.5, 8192 new tokens, stop "\n## Document") and degeneracy filters
    "basegen": ("base_continue", {"sampling_params": {"max_new_tokens": 8192}}, 16384),
    # Active Reading v2: planned alternative web documents, clamp(ceil(words / 23), 8, 48) views per source, planned
    # in calls of <= 24 prompts (thinking on), each written thinking off; ignores --generations. Not in the default
    # --arms (pass --arms arv2).
    "arv2": ("active_reading_v2", {"thinking": False, "strategy_thinking": True, "strategy_max_new_tokens": 16384,
                                   "words_per_view": 23, "min_views": 8, "max_views": 48, "prompts_per_call": 24,
                                   "sampling_params": NOTHINK}, 32768),
    # 8 of a 16-prompt pool per source (prompt_pool: the 8 mix prompts + 8 new ones; code / code_problem thinking on
    # with 12288 new tokens): mix16 draws the 8 types uniformly, mix16plan weights them by a planner call's 0/1/2
    # scores. Not in the default --arms (pass --arms mix16 mix16plan).
    "mix16": ("prompt_pool", {"thinking": False, "planned": False, "thinking_max_new_tokens": 12288,
                              "sampling_params": NOTHINK}, 32768),
    "mix16plan": ("prompt_pool", {"thinking": False, "planned": True, "thinking_max_new_tokens": 12288,
                                  "sampling_params": NOTHINK}, 32768),
}


def rows(subset_dir):
    """(row, _ref pointer, stripped text) for every row of a subset, in row order."""
    files = chunk_files(subset_dir)
    readers = [open(f, "rb") for f in files]
    offsets = [0] * N_CHUNKS
    j = 0
    while True:
        c = j % N_CHUNKS
        line = readers[c].readline()
        if not line:
            return
        yield j, {"path": files[c], "offset": offsets[c], "length": len(line)}, json.loads(line)["text"].strip()
        offsets[c] += len(line)
        j += 1


def sources(subsets, n, min_words, dedup, stats):
    """The first n (subset, row, ref) passing the filters, walking the subsets in order."""
    seen = set()
    for subset_dir in subsets:
        subset = os.path.basename(os.path.normpath(subset_dir))
        for row, ref, text in rows(subset_dir):
            stats["scanned"] += 1
            if len(text.split()) < min_words:
                stats["short"] += 1
                continue
            if dedup:
                key = hashlib.blake2b(text.encode(), digest_size=8).digest()
                if key in seen:
                    stats["duplicate"] += 1
                    continue
                seen.add(key)
            yield subset, row, ref
            stats["kept"] += 1
            if stats["kept"] == n:
                return
    raise SystemExit(f"only {stats['kept']} sources pass the filters in {subsets}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=[f"{DATA}/dclm_subsets/dclm_8m_{i}" for i in range(4)])
    ap.add_argument("--sources", type=int, required=True)
    ap.add_argument("--min-words", type=int, default=200)
    ap.add_argument("--no-dedup", dest="dedup", action="store_false")
    ap.add_argument("--tag", required=True, help="manifest name: dclm.<tag>.<arm>.manifest.jsonl")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--max-document-tokens", type=int, default=4096)
    ap.add_argument("--arms", nargs="+", default=[a for a in ARMS if a not in ("arv2", "mix16", "mix16plan")], choices=list(ARMS))
    ap.add_argument("--mix-repeats", type=int, default=1,
                    help="mix arm: ask for each of the 8 MIX prompts this many times (task field `prompts`; "
                         "prompt_mix ignores `generations`); 1 = the harness default, no `prompts` field")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    stats = {"scanned": 0, "short": 0, "duplicate": 0, "kept": 0}
    picked = list(sources(a.subsets, a.sources, a.min_words, a.dedup, stats))
    print(f"sources: {stats}; last = {picked[-1][0]} row {picked[-1][1]}")
    for arm in a.arms:
        handler, fields, context = ARMS[arm]
        extra = {"generations": a.generations, "max_document_tokens": a.max_document_tokens, **fields}
        if arm == "mix" and a.mix_repeats > 1:
            extra["prompts"] = MIX * a.mix_repeats
        path = f"{OUT}/dclm.{a.tag}.{arm}.manifest.jsonl"
        with open(path + ".tmp", "w") as out:
            for subset, row, ref in picked:  # source_subset / source_row: traced into every output row
                out.write(json.dumps({"_ref": ref, "source_subset": subset, "source_row": row, **extra},
                                     separators=(",", ":")) + "\n")
        os.replace(path + ".tmp", path)
        print(f"{arm}: {len(picked)} tasks -> {path}  (handler simplescale.harnesses.multisample:{handler}, "
              f"--context-length {context})")


if __name__ == "__main__":
    main()
