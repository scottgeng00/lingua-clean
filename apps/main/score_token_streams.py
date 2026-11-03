"""Score lingua checkpoints on pre-tokenized token streams (apps.main.eval.eval_on_token_streams) and write JSON.

Defaults: the 16 Paloma sources (OLMES-equivalent rolling scoring, setup/pretokenize_paloma_qwen3.py) + dclm_heldout.
One GPU; several checkpoints are scored one after the other. Each checkpoint is a step directory
(<run>/checkpoints/<step:010d>, consolidated on the fly into its `consolidated/` subdir if needed, as the eval job does)
or a consolidated directory.

    PYTHONPATH=. .venv-kd/bin/python -m apps.main.score_token_streams --out scores.jsonl CKPT [CKPT ...]

Every checkpoint appends one JSON line to --out: {"ckpt", "seconds", "load_seconds", "results": {name: metrics}}.
Paloma sources are reported as "paloma_qwen3/<source>", their unweighted mean bpb as "paloma_qwen3/mean".
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch

from apps.main.eval import ValidationArgs, eval_on_token_streams
from apps.main.generate import load_consolidated_model_and_tokenizer
from apps.main.transformer import LMTransformer, LMTransformerArgs
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints, consolidation_complete

PALOMA = "/home/sgeng/scottg/pretrain/assets/data/paloma_qwen3"
HELDOUT = "/home/sgeng/scottg/pretrain/assets/data/dclm_qwen3_tokens/dclm_heldout"
logger = logging.getLogger()


def consolidated_dir(ckpt: str) -> str:
    p = Path(ckpt)
    if (p / "params.json").exists() and (p / "consolidated.pth").exists():
        return str(p)
    if not consolidation_complete(ckpt):
        consolidate_checkpoints(ckpt)
    return str(p / CONSOLIDATE_FOLDER)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--streams", nargs="+", default=[PALOMA, HELDOUT])
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--rolling-batch-tokens", type=int, default=32768)
    ap.add_argument("--rolling-dtype", default="mixed", choices=["mixed", "fp32", "bf16"])
    ap.add_argument("--rolling-fp32-below", type=int, default=128)
    a = ap.parse_args()
    val_args = ValidationArgs(token_streams=a.streams, token_stream_seq_len=a.seq_len,
                              token_stream_rolling_batch_tokens=a.rolling_batch_tokens,
                              token_stream_rolling_dtype=a.rolling_dtype,
                              token_stream_rolling_fp32_below=a.rolling_fp32_below)
    for ckpt in a.ckpts:
        t0 = time.time()
        model, _, _ = load_consolidated_model_and_tokenizer(consolidated_dir(ckpt), model_cls=LMTransformer,
                                                            model_args_cls=LMTransformerArgs)
        model.eval()
        torch.cuda.synchronize()
        t1 = time.time()
        results = eval_on_token_streams(model, val_args)
        torch.cuda.synchronize()
        t2 = time.time()
        rec = {"ckpt": os.path.abspath(ckpt), "seconds": t2 - t0, "load_seconds": t1 - t0, "results": results}
        with open(a.out, "a") as f:
            print(json.dumps(rec), file=f, flush=True)
        logger.info(f"{ckpt}: {t2 - t0:.1f}s (load {t1 - t0:.1f}s)")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
