#!/usr/bin/env bash
# setup/fetch_assets.sh — download everything a fresh cluster needs (needs internet):
#   DCLM 8M splits from the Hub, the Qwen3.5 tokenizer (eos -> <|endoftext|>), the Qwen3.5-9B-Base
#   teacher, and the HF datasets behind the OLMES lingua:v2 suite.
#   sbatch -A mnm -q g3_comm_shared --gres=gpu:1 -c 32 --mem=200G -t 6:00:00 setup/fetch_assets.sh
set -uo pipefail
ROOT_DIR="${ROOT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
ASSETS="${ASSETS:-/home/sgeng/scottg/pretrain/assets}"
OLMES_ROOT="${OLMES_ROOT:-/home/sgeng/scottg/pretrain/olmes}"
PY="${ROOT_DIR}/.venv-kd/bin/python"
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "${ASSETS}/models" "${ASSETS}/data"
cd "${ROOT_DIR}"

"$PY" - "${ASSETS}" <<'PY' &
import json, os, sys
from huggingface_hub import snapshot_download
assets = sys.argv[1]
tok = f"{assets}/models/Qwen3.5-2B-tokenizer-eot"
if not os.path.exists(f"{tok}/tokenizer_config.json"):
    snapshot_download("Qwen/Qwen3.5-2B", local_dir=tok,
                      allow_patterns=["tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json", "*.jinja", "config.json"])
    cfg_path = f"{tok}/tokenizer_config.json"
    cfg = json.load(open(cfg_path))
    cfg["eos_token"] = "<|endoftext|>"
    cfg["pad_token"] = "<|endoftext|>"
    json.dump(cfg, open(cfg_path, "w"), indent=2, ensure_ascii=False)
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(tok)
print("tokenizer:", t.eos_token, t.eos_token_id, len(t), flush=True)
snapshot_download("Qwen/Qwen3.5-9B-Base", local_dir=f"{assets}/models/Qwen3.5-9B-Base", max_workers=8)
print("teacher: done", flush=True)
PY
MODELS_PID=$!

"$PY" setup/download_dclm_8m_from_hub.py --out "${ASSETS}/data/dclm_subsets" &
DATA_PID=$!

(cd "${OLMES_ROOT}" && .venv-lmeval/bin/python "${ROOT_DIR}/setup/download_eval_datasets.py" lingua:v2) &
EVAL_PID=$!

rc=0
wait $MODELS_PID || { echo "models FAILED"; rc=1; }
wait $DATA_PID || { echo "data FAILED"; rc=1; }
wait $EVAL_PID || { echo "eval datasets FAILED"; rc=1; }
grep -c '"ok' "${OLMES_ROOT}/download_status.json"; grep FAIL "${OLMES_ROOT}/download_status.json"
exit $rc
