"""Pre-tokenize the 16 Paloma sources (test split, every document, as scored by OLMES' `paloma_<source>::paloma`
tasks) with the Qwen3 tokenizer, one token-stream directory per source, for packed scoring by
apps.main.eval.eval_on_token_streams.

What OLMES does (oe_eval/tasks/oe_eval_tasks/paloma.py, configs/tasks.py, models/eleuther_huggingface.py,
metrics/perplexity.py):
  - data: HF dataset allenai/paloma, config = source, split "test" (no limit / subsampling), text = doc["text"]
    verbatim (no stripping or other preprocessing).
  - tokens: tokenizer.encode(text, add_special_tokens=False); no BOS, no EOS. With rolling_no_prefix the first token
    is context only, every other token is scored.
  - windows: lm_eval get_rolling_token_windows(max_seq_len=4096, context_len=1) + make_disjoint_window: the first
    window scores tokens 1..4096 of the doc; each following window scores the next (up to) 4096 tokens with the
    model input being the 4096 tokens just before each window's last token (so a full window carries 1 token of
    context, the last partial window is topped up with earlier tokens to 4096 inputs).
  - primary_score = bits_per_byte = -(sum of logprobs over all docs of the source) / ln 2 / (sum of UTF-8 bytes of
    the docs' text): a micro average over the source; the mean over sources is taken outside OLMES.

Output per source (/home/sgeng/scottg/pretrain/assets/data/paloma_qwen3/<source>/), same format as dclm_heldout
(setup/pretokenize_dclm_qwen3.py:pretokenize_heldout) except that NO end-of-document token is appended:
    tokens.bin       uint32, documents back to back, dataset order
    doc_offsets.npy  int64 [n_docs + 1]
    doc_bytes.npy    int64 [n_docs], len(text.encode("utf-8"))
    meta.json        counts + "eval_mode": "rolling" (eval_on_token_streams then reproduces OLMES' windows instead of
                     packing the stream; see apps/main/eval.py) + "vocab_size" (= len(tokenizer); the HF export that
                     OLMES scores drops lingua's padded vocab rows, so logits are cut to this size)

    PYTHONPATH=. .venv-kd/bin/python -m setup.pretokenize_paloma_qwen3
"""

import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np

TOKENIZER = "/home/sgeng/scottg/pretrain/assets/models/Qwen3-0.6B-Base-tokenizer"
OUT = "/home/sgeng/scottg/pretrain/assets/data/paloma_qwen3"
HF_CACHE = os.path.join(os.environ.get("HF_DATASETS_CACHE") or os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "datasets"), "allenai___paloma")
# oe_eval/datasets_extra/paloma_sources.py
PALOMA_SOURCES = [
    "4chan_meta_sep", "c4_100_domains", "c4_en", "dolma_100_programing_languages", "dolma_100_subreddits",
    "dolma-v1_5", "falcon-refinedweb", "gab", "m2d2_s2orc_unsplit", "m2d2_wikipedia_unsplit", "manosphere_meta_sep",
    "mc4", "ptb", "redpajama", "twitterAAE_HELM_fixed", "wikitext_103",
]
MAX_LEN = 4096  # OLMES max_length; only used for the long-document statistics in meta.json


def load_texts(source, split="test"):
    """Texts of one Paloma source from the local HF datasets cache (what OLMES reads with HF_DATASETS_OFFLINE=1)."""
    files = sorted(glob.glob(os.path.join(HF_CACHE, source, "0.0.0", "*", f"paloma-{split}.arrow")))
    if not files:
        # Not cached: download (allenai/paloma is gated, needs HF_TOKEN, e.g. run under `bash -lic`).
        from datasets import load_dataset
        return list(load_dataset("allenai/paloma", source, split=split)["text"])
    from datasets import Dataset
    assert len(files) == 1, files
    return list(Dataset.from_file(files[0])["text"])


def pretokenize_source(task):
    source, out_root, split = task
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    texts = load_texts(source, split)
    ids = tok(texts, add_special_tokens=False)["input_ids"]
    lens = np.asarray([len(x) for x in ids], dtype=np.int64)
    out = os.path.join(out_root, source)
    os.makedirs(out, exist_ok=True)
    np.fromiter((t for x in ids for t in x), dtype=np.uint32, count=int(lens.sum())).tofile(f"{out}/tokens.bin")
    np.save(f"{out}/doc_offsets.npy", np.concatenate([[0], np.cumsum(lens)]).astype(np.int64))
    n_bytes = np.asarray([len(t.encode("utf-8")) for t in texts], dtype=np.int64)
    np.save(f"{out}/doc_bytes.npy", n_bytes)
    # A doc of n tokens is scored in one OLMES window iff n - 1 <= MAX_LEN.
    long = lens > MAX_LEN + 1
    meta = {"source": f"allenai/paloma/{source}/{split}", "docs": len(texts), "tokens": int(lens.sum()),
            "bytes": int(n_bytes.sum()), "tokenizer": TOKENIZER, "eos_appended": False, "eval_mode": "rolling", "rolling_max_len": MAX_LEN,
            "vocab_size": len(tok),
            "long_docs": int(long.sum()), "long_doc_tokens": int(lens[long].sum()), "max_doc_tokens": int(lens.max()),
            "empty_docs": int((lens == 0).sum()), "script": "setup/pretokenize_paloma_qwen3.py"}
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    return source, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=PALOMA_SOURCES)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    with Pool(len(a.sources)) as pool:
        metas = dict(pool.map(pretokenize_source, [(s, a.out, a.split) for s in a.sources]))
    print(f"{'source':32s} {'docs':>8s} {'tokens':>11s} {'bytes':>11s} {'long docs':>9s} {'%docs':>6s} {'%tokens':>7s}")
    for s in a.sources:
        m = metas[s]
        print(f"{s:32s} {m['docs']:8d} {m['tokens']:11d} {m['bytes']:11d} {m['long_docs']:9d} "
              f"{100 * m['long_docs'] / m['docs']:6.2f} {100 * m['long_doc_tokens'] / m['tokens']:7.2f}")


if __name__ == "__main__":
    main()
