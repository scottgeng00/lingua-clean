"""Re-score Paloma on already-converted checkpoints with rolling_no_prefix (OlmesArgs.rolling_no_prefix).

For each <run_dir>:<step>, runs the 16 Paloma tasks (OLMES hf backend, one shard per visible GPU) on
<run_dir>/checkpoints/<step>/hf, writes shards to <run_dir>/evals/<step>/olmes_results/paloma_np_shard_<i>/ and
appends {"global_step", "olmes_eval/<task>/primary_score", ...} to <run_dir>/metrics.paloma_np.jsonl. The original
EOT-prefixed results (paloma_shard_*, metrics.olmes_eval.jsonl) are left untouched. Checkpoints already in
metrics.paloma_np.jsonl are skipped, so the job can be requeued or re-run.

    python -m apps.main.rescore_paloma <run_dir>:<step> [<run_dir>:<step> ...]
"""

import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

from apps.main.eval_olmes import OlmesArgs, _run_shards, parse_olmes_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rescore_paloma")

TOKENIZER = "/home/sgeng/scottg/pretrain/assets/models/Qwen3.5-2B-tokenizer-eot"
OLMES_ENV = "/home/sgeng/scottg/pretrain/olmes/.venv-lmeval"
PALOMA = [
    "paloma_4chan_meta_sep::paloma", "paloma_c4_100_domains::paloma", "paloma_c4_en::paloma",
    "paloma_dolma_100_programing_languages::paloma", "paloma_dolma_100_subreddits::paloma",
    "paloma_dolma-v1_5::paloma", "paloma_falcon-refinedweb::paloma", "paloma_gab::paloma",
    "paloma_m2d2_s2orc_unsplit::paloma", "paloma_m2d2_wikipedia_unsplit::paloma",
    "paloma_manosphere_meta_sep::paloma", "paloma_mc4::paloma", "paloma_ptb::paloma",
    "paloma_redpajama::paloma", "paloma_twitterAAE_HELM_fixed::paloma", "paloma_wikitext_103::paloma",
]


def done_steps(path):
    return {json.loads(l)["global_step"] for l in open(path)} if path.exists() else set()


def main():
    args = replace(OlmesArgs(), tasks=PALOMA, model_type="hf", tokenizer_path=TOKENIZER, max_length=4096,
                   olmes_env=OLMES_ENV, timeout_hours=4.0, rolling_no_prefix=True)
    for spec in sys.argv[1:]:
        run_dir, step = spec.rsplit(":", 1)
        run_dir, step = Path(run_dir), int(step)
        out_file = run_dir / "metrics.paloma_np.jsonl"
        if step in done_steps(out_file):
            logger.info(f"{run_dir.name} step {step}: already scored, skipping")
            continue
        hf = run_dir / "checkpoints" / f"{step:010d}" / "hf"
        out_dir = run_dir / "evals" / f"{step:010d}" / "olmes_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        shard_metrics, n = _run_shards(str(hf), args, PALOMA, str(out_dir), "paloma_np_shard_",
                                       time.time() + args.timeout_hours * 3600)
        scores = parse_olmes_results({"tasks": [t for m in shard_metrics for t in m.get("tasks", [])]})
        n_tasks = sum(k.endswith("::paloma/primary_score") for k in scores)
        if n_tasks != len(PALOMA):
            logger.error(f"{run_dir.name} step {step}: only {n_tasks}/{len(PALOMA)} Paloma tasks scored ({len(shard_metrics)}/{n} shards); not recording")
            continue
        with open(out_file, "a") as f:
            f.write(json.dumps({"global_step": step, "rolling_no_prefix": True, **scores}) + "\n")
        logger.info(f"{run_dir.name} step {step}: mean Paloma BPB {sum(v for k, v in scores.items() if k.endswith('primary_score')) / n_tasks:.4f}")


if __name__ == "__main__":
    main()
