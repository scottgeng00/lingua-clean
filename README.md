# lingua-clean

A build on top of [lingua](https://github.com/facebookresearch/lingua).

## Supported recipes

All recipes live under `apps/main/configs/recipes/`. They all share the same
canonical config (OLMo-2-0425-1B student, dolmino-splits dataset, 28800 steps,
4-node FSDP, bf16, FP8 7B teacher where applicable). They differ only in the
`teacher_model_path` and the `data.use_<X>_distillation` block.

### Headline recipes

| Recipe                          | Teacher                  | Loss form |
|---------------------------------|--------------------------|-----------|
| `ntp_baseline`                  | — (no teacher)           | Vanilla next-token-prediction CE. |
| `fkd_1b`                        | OLMo-2-0425-1B-Instruct  | Forward KL distillation, α=0.5, T=2. |
| `fkd_7b`                        | OLMo-2-1124-7B-Instruct  | Forward KL distillation, α=0.5, T=2. |
| `rkl_entgate_q30_lam1p0` (`idx139`) | OLMo-2-1124-7B-Instruct  | (idx 139) Entropy-gated reverse KL. L = λ·CE + 1[H(p_T)≤τ]·T²·KL(p_S‖p_T). q=0.30, λ=1, T=2. |
| `rkl_entswitch_q30_lam1p0` (`idx148`) | OLMo-2-1124-7B-Instruct | (idx 148) Entropy-switched RKL. L = (1−m_t)·λ·CE + m_t·T²·KL(p_S‖p_T). |
| `entband_q30_q30_lam1p0`        | OLMo-2-1124-7B-Instruct  | (idx 151) Entropy-band KD: CE + T²·RKL(low 30%) + T²·FKL(high 30%). |

### Ablation controls

| Recipe                          | Notes |
|---------------------------------|-------|
| `fkl_top30_lam1p0`              | (idx 152) Entropy-band scaffold with `entband_low_quantile=0`; FKL on top-30% teacher-H tokens only. |
| `fkl_entgate_q30_lam1p0`        | (idx 140) FKL counterpart of `idx139`. |
| `rkl_randmask_p30_lam1p0`       | Replaces the entgate predicate with a uniform random top-30% mask. |
| `rkl_stentgate_q30_lam1p0`      | Gate on **student** entropy (top-30%), not teacher entropy. |
| `rkl_uniform_lam{0p1,0p3,0p5,1p0}` | RKL purekd + uniform-CE-on-all-tokens sweep (no token selection). |
| `rkl_purekd`                    | (idx 100) Pure RKL (MiniLLM-style), α=1, no CE. |
| `rkl_fkl_mix_alpha{0p5,0p8}`    | RKL+FKL pure-KD mixture (idx 103/104). |

## Layout

```
lingua-clean/
├── README.md                                  ← this file
├── LICENSE                                    ← upstream Llama-2 license
├── pyproject.toml, uv.lock                    ← uv-managed deps (torch/xformers cu121, prebuilt flash-attn wheel)
├── requirements.txt                           ← plain-pip deps (no torch/xformers/flash-attn)
├── bin/
│   └── install_requirements.sh                ← env bootstrap (`uv sync` → ./.venv)
├── lingua/                                    ← lingua library (unchanged)
│   └── (args, data, distributed, optim, ...)
├── apps/main/
│   ├── train.py                               ← PRUNED (15791 → ~7600 lines)
│   ├── eval.py, eval_olmes.py, generate.py    ← unchanged
│   ├── transformer.py                         ← unchanged
│   └── configs/
│       ├── dolmino_midtrain_olmo2_v2.yaml     ← canonical 9600-step config
│       ├── dolmino_expert_cpt_baseline_28800.yaml  ← canonical 28800-step config
│       └── recipes/                           ← one YAML per supported recipe
├── setup/
│   ├── consolidate.py
│   ├── convert_consolidated_lingua_ckpt_to_hf.py
│   ├── download_hf_ckpt.py
│   └── download_prepare_hf_data.py
├── analysis/
│   └── idx139/                               ← per-token (H_T, H_S) gate-overlap analysis
└── scripts/
    ├── README.md
    ├── env.sh                                ← single source of truth for paths
    ├── launch_midtrain.sh                     ← sbatch wrapper (the entry point)
    └── _sbatch_inner.sh                       ← runs INSIDE sbatch / srun
```

## Setup

1. Build the venv with [uv](https://docs.astral.sh/uv/) (one-time; re-run after
   dependency changes):
   ```bash
   bash bin/install_requirements.sh
   source .venv/bin/activate
   ```
   This runs `uv sync --frozen` against `pyproject.toml` / `uv.lock` — Python
   3.11, `torch==2.5.0` + `xformers==0.0.28.post2` (cu121), the prebuilt
   `flash-attn==2.7.4.post1` wheel, and everything else pinned in the lock.
   Override `LINGUA_VENV=<path>` to put the venv somewhere other than
   `<repo>/.venv`. To add a dependency: `uv add <pkg>`, then commit
   `pyproject.toml` + `uv.lock`.

2. Edit `scripts/env.sh` (or export the variables in your shell beforehand)
   so the paths point at where you've actually downloaded the data and
   checkpoints. By default everything lives under one deletable sandbox
   dir, `${CACHE_DIR}` (default `~/lingua-clean-sandbox`):
   - `${CACHE_DIR}/data/dolmino_splits/`             (Dolmino mid-train splits)
   - `${CACHE_DIR}/teachers/OLMo-2-0425-1B-Instruct/`
   - `${CACHE_DIR}/teachers/OLMo-2-1124-7B-Instruct/`
   - `${CACHE_DIR}/students/OLMo-2-0425-1B-stage1-4001B/`     (Lingua DCP root, `STUDENT_INIT_PATH`)
   - `${CACHE_DIR}/students/OLMo-2-0425-1B-stage1-4001B/hf/`  (HF mirror, `STUDENT_HF_PATH`)
   - `${CACHE_DIR}/runs/{midtrain,evals,analysis,slurm_logs}/`

   To start over: `rm -rf ${CACHE_DIR}`. Override individual paths
   (e.g. point `DATA_ROOT` at a long-lived cache) to share artifacts across
   sandboxes.

   The student is split because Lingua's training driver loads the init via
   `lingua.checkpoint.load_from_checkpoint`, which expects a DCP directory
   (`.metadata` + `__0_0.distcp`), while the tokenizer / OLMES eval / idx139
   analysis path loads HF format via `AutoModelForCausalLM.from_pretrained`.
   `scripts/fetch_models.sh` populates both layouts in one shot.

   Optional cluster-specific knobs in `scripts/env.sh` (unset by default,
   slurm uses site defaults if you leave them empty):
   - `SLURM_ACCOUNT`, `SLURM_QOS`, `SLURM_PARTITION` — appended to the
     `launch_midtrain.sh` sbatch command if non-empty.
   - `WANDB_ENTITY` — wandb entity for all runs (leave empty to use your
     personal default).

3. Source the env script (must be done in every fresh shell):
   ```bash
   source scripts/env.sh
   ```

4. Download teachers (`OLMo-2-0425-1B-Instruct`, `OLMo-2-1124-7B-Instruct`) and
   the student init checkpoint (`OLMo-2-0425-1B-stage1-4001B`) into
   `${TEACHER_1B_PATH}`, `${TEACHER_7B_PATH}`, and `${STUDENT_INIT_PATH}`
   respectively. Use `setup/download_hf_ckpt.py` or download manually from HuggingFace.

5. Download and shard the Dolmino mid-training mix (DCLM / FLAN / Math / Wiki /
   pes2o / StackExchange splits) into `${DATA_ROOT}`. Use
   `setup/download_prepare_hf_data.py`.

6. Confirm the launch script can find the recipe you want and renders a sane
   sbatch invocation:
   ```bash
   RECIPE=ntp_baseline DRY_RUN=1 bash scripts/launch_midtrain.sh
   ```

## Launching a run

```bash
source scripts/env.sh

# Real submit (4-node, ~56h wall clock for the 7B-teacher recipes):
RECIPE=idx139 bash scripts/launch_midtrain.sh

# Single-node smoke test (override nodes; will still run for ~weeks at this size):
RECIPE=ntp_baseline NNODES=1 bash scripts/launch_midtrain.sh

# Just print the sbatch invocation, don't submit:
RECIPE=fkd_7b DRY_RUN=1 bash scripts/launch_midtrain.sh
```

Outputs:
- Checkpoints / wandb-offline dumps land at the `dump_dir` declared in the
  recipe YAML (default: `${MIDTRAIN_ROOT}/<recipe_name>`).
- Slurm logs land at `${SLURM_LOG_DIR}` (default `~/lingua-runs/slurm_logs`).
- OLMES evaluation runs automatically every 1200 steps via the
  `async_eval_gpus` path (see the `eval:` block of any recipe YAML).

## Analysis

Per-token (H_T, H_S) gate-overlap analysis backing the paper's "entropy-gating
is real" claim for the `idx139` recipe (originally `rkl_entgate_q30_lam1p0`) lives under
`analysis/idx139/`. See `analysis/idx139/README.md` for the workflow.


---

# Setting up on a new cluster (from-scratch pretraining + synthetic-data experiments)

This section covers everything except the training data itself: environments, assets, the on-disk
**data format the loader expects**, launching runs, evals and results. The generator-scaling and
multi-rewrite experiments (`apps/main/configs/experiments/genscale/`, `results/`) use this path;
synthetic data is generated with the sibling repo
[simplescale](../simplescale/README.md#setting-up-on-a-new-cluster).

## 1. Environments

There are three Python environments:

| Env | Built by | Used for |
|---|---|---|
| `.venv` | `bash bin/install_requirements.sh` (`uv sync --frozen`; Python 3.11, torch 2.5.0 cu121, flash-attn 2.7.4.post1, transformers 4.51) | data-prep / analysis scripts, the results viewer, async eval jobs (`LINGUA_EVAL_VENV`) |
| `.venv-kd` | `bash bin/install_kd_requirements.sh` (torch 2.8.0 cu128, flash-attn 2.8.3, transformers 5.19, flash-linear-attention 0.5.2) | **training** (`LINGUA_VENV=$PWD/.venv-kd`); needed for the Qwen3.5 tokenizer / teachers |
| OLMES env (`OLMES_CONDA_ENV`) | a checkout of our OLMES fork, branch `lm-eval-0.4.13` (OLMES ported to `lm_eval==0.4.13`; vLLM 0.11.0, transformers 4.57): `cd olmes && uv venv .venv-lmeval --python 3.11 && uv pip install --python .venv-lmeval/bin/python -e ".[gpu]"` | OLMES downstream evals, launched as a subprocess by `apps/main/eval_olmes.py` |

Both lingua venvs need `uv` on PATH (`curl -LsSf https://astral.sh/uv/install.sh | sh`); the
install scripts set `UV_LINK_MODE=copy` because the uv cache and `/checkpoint` are usually on
different filesystems.

**aarch64 / GB300 clusters** (e.g. the `g3` partition, account `mnm`). Only `.venv-kd` works there and it
serves for both training and evals (`scripts/env.sh` defaults `LINGUA_VENV` to it). The differences:
- torch 2.8.0's aarch64 CUDA build is `2.8.0+cu129`. flash-attn 2.8.3 and xformers 0.0.32.post2 have no
  aarch64 wheels, so `bin/build_aarch64_wheels.sh` builds them (sm_100 kernels, which also run on sm_103)
  into `../wheelhouse`. It needs a CUDA 12.9 toolkit at `../cuda-12.9`:
  `conda create --no-default-packages -p ../cuda-12.9 -c conda-forge --override-channels cuda-nvcc=12.9 cuda-cudart-dev=12.9 cuda-libraries-dev=12.9 cuda-nvtx-dev=12.9 cuda-profiler-api=12.9`.
  `bin/install_kd_requirements.sh` picks all of this up on aarch64.
- Triton 3.4 ships a CUDA 12.8 `ptxas` that rejects `sm_103a`. Each venv's `sitecustomize.py` sets
  `TRITON_PTXAS_PATH` to `../cuda-12.9/bin/ptxas`.
- OLMES env: install `-e .` plus `vllm==0.11.0` without the `[gpu]` extra (its xformers won't build), then
  add `"pyarrow<20" "setuptools<81" accelerate`. Install torch with `--no-sources` from the cu129 index.
- Nodes have 4 GPUs. Data parallelism must be a multiple of 16 chunks, so use 4, 8 or 16 nodes.
  `setup/make_gb300_configs.py` writes the GB300 sweep configs (`experiments/gb300/`).

> The OLMES fork branch `lm-eval-0.4.13` currently exists only in the local checkout
> (`/home/sgeng/scottg/pretrain/olmes`, a clone of `allenai/olmes` plus a few commits). Push it to
> a remote you can reach from the new cluster before setting up there.

## 2. Per-cluster configuration (`scripts/env.sh`)

`scripts/env.sh` is sourced by every launcher. On a new cluster, edit (or export before sourcing):

| Variable | Meaning | Current value here |
|---|---|---|
| `CACHE_DIR` | root for all run outputs (hard-coded here; restore the commented `${HOME}/lingua-clean-sandbox` default or set your own) | `/checkpoint/dream-sc-3/scottg/pretrain/lingua-runs` |
| `MIDTRAIN_ROOT` | run dirs (`dump_dir: ${oc.env:MIDTRAIN_ROOT}/...` in configs) | `${CACHE_DIR}/runs/midtrain` |
| `EVAL_ROOT` | async-eval slurm logs (`eval.dump_dir`) | `${CACHE_DIR}/runs/evals` |
| `SLURM_LOG_DIR` | training slurm logs | `${CACHE_DIR}/runs/slurm_logs` |
| `DATA_ROOT` | root of the **validation** sets (`eval.validation.root_dir: ${oc.env:DATA_ROOT}`) | `.../assets/data/dolmino_splits` |
| `LINGUA_VENV` | training venv (override per launch with `.venv-kd`) | `<repo>/.venv` |
| `OLMES_CONDA_ENV` | OLMES env (venv or conda) | `/home/sgeng/scottg/pretrain/olmes/.venv-lmeval` |
| `SLURM_ACCOUNT`, `SLURM_QOS`, `SLURM_PARTITION` | appended to sbatch if non-empty | `dream`, `h200_dream_high`, unset |
| `WANDB_ENTITY` | wandb entity (empty = your default) | empty |
| `TEACHER_*`, `STUDENT_*`, `TOKENIZER_PATH` | KD mid-training only; unused by from-scratch pretraining | |

The launcher (`scripts/launch_pretrain.sh`) also hard-codes the node shape: `--gres=gpu:8
--cpus-per-task=64 --mem=456G --time=122:00:00`. Adjust for the new cluster's nodes.

**Absolute paths inside configs.** The genscale configs hard-code `data.root_dir`
(`/home/sgeng/scottg/pretrain/assets/data`), `data.tokenizer.path` and `eval.olmes.tokenizer_path`
(`/home/sgeng/scottg/pretrain/assets/models/Qwen3.5-2B-tokenizer-eot`). Either recreate that
`assets/` layout on the new cluster (simplest: a symlink) or `sed` the configs.

wandb credentials come from `~/.netrc` (the host in `~/.config/wandb/settings`); see
`scripts/_sbatch_inner.sh` for `WANDB_USE_ENV_KEY` / `WANDB_MODE`.

## 3. Assets other than data

- **Tokenizer** `assets/models/Qwen3.5-2B-tokenizer-eot`: the `Qwen/Qwen3.5-2B` tokenizer (snapshot
  `15852e8`) with only `tokenizer_config.json` changed so `eos_token` is `<|endoftext|>` (id 248044)
  instead of `<|im_end|>`. Documents are separated by `<|endoftext|>` (`add_eos: true`, `add_bos:
  false`). To recreate: download the tokenizer files from `Qwen/Qwen3.5-2B` and set `"eos_token":
  "<|endoftext|>"` (and `"pad_token"`) in `tokenizer_config.json`.
- **Generator models** for synthetic data (`assets/models/Qwen3.5-{2B,9B,27B}`), used by simplescale,
  not by training.
- **OLMES eval datasets.** Async evals run with `HF_HUB_OFFLINE=1`
  (`apps/main/eval_olmes.py` sets it unless you export `HF_HUB_OFFLINE=0`), so every dataset an eval
  task touches must already be in the HF cache of the eval nodes.
  - **Standard eval (v2, used from 2026-10-27 on):** the OLMES suite `lingua:v2` in our fork. It covers:
    - OLMo-3 `base_easy` QA rank-classification and gold-answer BPB, 20 tasks each (SocialIQA dropped: its source zip is gone);
    - Minerva math BPB and code BPB (HumanEval, MBPP, MT-MBPP);
    - the 16 Paloma sources;
    - GSM8K pass@1..32 (`gsm8k::olmo3:n32:pass32`).
  - **Old eval (genscale configs before that):** `sciq:rc::olmo3, mmlu:rc::olmes, arc_easy:rc::olmes:full, piqa:rc::olmes:full, gsm8k::olmes, triviaqa::olmes, gsm8k:bpb::olmes, triviaqa:bpb::olmes, mbpp:3shot:bpb::none`, plus lingua perplexity on the Dolmino val sets and `dclm_heldout`.
  - Pre-download once from a node with internet (rerun for any task that hits HTTP 429):
    ```
    cd $OLMES_ROOT && HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
        .venv-lmeval/bin/python $LINGUA/setup/download_eval_datasets.py lingua:v2
    ```
    `allenai/paloma` is gated: accept its terms on the Hub with the account behind `HF_TOKEN`.
  - `setup/submit_eval_v2.py` runs `lingua:v2` on existing checkpoints into `<run>/evals_v2/` (4 GPUs per job).

## 4. Data format the loader expects

Training data is read by `lingua/data.py` from **pre-shuffled JSONL chunk files**. Nothing is
tokenized ahead of time; tokenization happens on the fly.

**Layout.** `data.root_dir` plus each key of `data.sources` (a path relative to `root_dir`) names a
directory. Each source directory holds chunk files matching `*.chunk.*.jsonl`
(`TRAIN_DATA_FILE_PATTERN`; `*shard*.jsonl` is accepted as a fallback):

```
<data.root_dir>/
├── dclm_subsets/dclm_8m_0/
│   ├── dataset.json                 # optional metadata, never read by the loader
│   ├── dclm_8m_0.chunk.00.jsonl
│   ├── ...
│   └── dclm_8m_0.chunk.15.jsonl
└── dclm_rewrites/dclm_8m_1_qwen35_9b_wiki_g1_max4096/
    └── dclm_8m_1_qwen35_9b_wiki_g1_max4096.chunk.{00..15}.jsonl
```

**Lines.** One JSON object per line. The loader uses the `text` field (falling back to `content`,
then `original_text`); lines with none of these are skipped. Other fields (`id`, `src_row`,
`finish_reason`, ...) are ignored by training but are used by our setup scripts for alignment checks.

**Chunk count vs. world size.** Chunks are assigned to data-parallel ranks
(`find_and_sanitize_chunks` / `distribute_data_to_rank`; world size = DP degree = total GPUs with
FSDP `full_shard`):
- if `n_chunks <= world_size`, `world_size` must be a multiple of `n_chunks`; ranks sharing a chunk
  read interleaved lines (rank i of k reads lines i, i+k, ...);
- if `n_chunks > world_size`, **the extra chunks are silently dropped**.

All our sources use **16 chunks**, which works for 16, 32 or 64 GPUs (2-8 nodes of 8). With fewer
than 16 GPUs you would lose data, so re-chunk first.

**Order.** The loader does not shuffle within a file, so chunks must be pre-shuffled. Within an epoch
each rank reads its lines in file order. When a rank exhausts its lines it starts a new epoch in
file order, or in a fresh seeded permutation if `data.shuffle_each_epoch: true`.

**Round-robin row layout (our convention).** `setup/prepare_dclm_subsets.py` writes global row j of
a subset to chunk `j % 16`, line `j // 16`. Combined with `data.source_max_docs: {<source>: N}`,
which reads the first `N // n_chunks` lines of every chunk (remainder to the first chunks), this
means `source_max_docs: N` is **exactly the first N rows** of the source, nested across N. Rewrite
splits are row-aligned with their source (row j = rewrite of source row j), so a rewrite view
truncated at N docs covers the same documents as the source truncated at N.

**Mixing.** `data.sources` maps source -> weight. The loader picks a source per document with
probability proportional to its weight (`choose_source`), then packs documents into `seq_len`
sequences (`add_eos` inserts `<|endoftext|>` between documents; `disable_cross_doc_attn: true` masks
attention across documents). Weights are therefore **per document**, not per token. Our configs
pick weights so a source gets a target token share: e.g. S1 at weight 1 is one epoch of 8M docs, and
a rewrite view at weight `1262 / sum(view tokens per doc)` gets the same tokens as S1.

**Tokens per step** = `data.batch_size` (per GPU) x DP world size x `seq_len`. The genscale configs
use 8 x 32 x 4096 = 1.05M tokens/step, so 19,074 steps = 20B tokens on 4 nodes. On a different
node count, adjust `batch_size` (or `grad_acc_steps`) to keep the global batch.

**Validation sets.** `eval.validation.root_dir` (= `DATA_ROOT`) + each name in
`eval.validation.sources` is a directory holding `*.val.jsonl` files with `text` lines
(`apps/main/eval.py:eval_on_val`), e.g. `dolmino_splits/dclm_heldout/dclm_heldout.val.jsonl`.
Up to `max_steps` documents are scored. Results are reported as `nll_per_token` / `nll_per_char`
per source name in `metrics.validation.jsonl`.

**Converting any dataset to this layout:**

```python
import json, os, sys
from datasets import load_dataset  # or iterate any JSONL / parquet

def write_lingua_source(rows, out_dir, name, n_chunks=16, val_dir=None, val_rows=0):
    """rows: iterable of dicts with "text" (pre-shuffled). Writes <name>.chunk.NN.jsonl round-robin,
    optionally holding out the first val_rows docs as <val_dir>/<name>.val.jsonl."""
    os.makedirs(out_dir, exist_ok=True)
    outs = [open(f"{out_dir}/{name}.chunk.{c:02d}.jsonl", "w") for c in range(n_chunks)]
    if val_rows:
        os.makedirs(val_dir, exist_ok=True)
    val = open(f"{val_dir}/{name}.val.jsonl", "w") if val_rows else None
    for j, r in enumerate(rows):
        line = json.dumps({"id": r.get("id", j), "text": r["text"]}) + "\n"
        if j < val_rows:
            val.write(line)   # hold out the first val_rows docs as a validation file
        else:
            outs[(j - val_rows) % n_chunks].write(line)
    for f in outs + ([val] if val else []):
        f.close()

ds = load_dataset("mlfoundations/dclm-baseline-1.0", split="train", streaming=True).shuffle(seed=0, buffer_size=100_000)
write_lingua_source((x for _, x in zip(range(8_001_000), ds)), "/data/train/dclm_8m", "dclm_8m",
                    val_dir="/data/val/dclm_8m_val", val_rows=1000)
```

Keep validation files in their own directory under `DATA_ROOT` (`/data/val` above). Training
globs only `*.chunk.*.jsonl` and validation globs only `*.val.jsonl`, so a stray file of the other
kind is ignored either way.

## 5. Launching a pretraining run

```bash
cd lingua-clean
LINGUA_VENV=$PWD/.venv-kd LINGUA_EVAL_VENV=$PWD/.venv HF_HUB_OFFLINE=1 \
CONFIG=experiments/genscale/r_lr5p3e3_wd01 NNODES=4 STEPS_OVERRIDE=19074 bash scripts/launch_pretrain.sh
# DRY_RUN=1 prints the sbatch command instead of submitting.
```

`CONFIG` resolves to `apps/main/configs/<CONFIG>.yaml`. The job runs `torchrun -m apps.main.train
config=<yaml> steps=<STEPS_OVERRIDE>` on every node (`scripts/_sbatch_inner.sh`) and is `--requeue`able.
A requeued or resubmitted job resumes from the latest checkpoint in its `dump_dir`.

Each run writes to `dump_dir` (`${MIDTRAIN_ROOT}/experiments/genscale/<name>/`):
- `checkpoints/<step>/` (DCP shards). `checkpoint.dump.every` saves for resumption (`keep` latest);
  `checkpoint.eval.every` saves for evaluation (kept forever with `keep: -1`);
- `metrics.jsonl` (training), `metrics.validation.jsonl` (val NLL per source),
  `metrics.olmes_eval.jsonl` (OLMES scores);
- `evals/<step>/` (one async eval job per eval checkpoint: config, `submit.slurm`, logs, OLMES outputs).

**Async evals.** At every `checkpoint.eval.every` step, training submits an `olmes_evals` sbatch job
with `async_eval_gpus` GPUs (`apps/main/eval_olmes.py`). It:
1. consolidates the DCP checkpoint to `checkpoints/<step>/consolidated/`;
2. runs lingua validation NLL;
3. converts the checkpoint to HF (`checkpoints/<step>/hf/`);
4. runs OLMES with vLLM in the OLMES env, splitting tasks across the GPUs.

To re-run one eval, `sbatch <dump_dir>/evals/<step>/submit.slurm`.

**Results.**
- `analysis/results_viewer/` is a web app over the run dirs (`runs.yaml` lists the groups).
- `analysis/results_viewer/genscale_table.py <run> ...` prints a final-step markdown table.
- Write-ups go in `results/` (see `results/README.md` for the contract).

## 6. Synthetic-data pipeline (DCLM subsets and rewrites)

| Step | Script |
|---|---|
| Disjoint 8M-doc DCLM subsets `dclm_subsets/dclm_8m_{0..3}` (16 round-robin chunks, `dataset.json`) | `setup/prepare_dclm_subsets.py` |
| Decontaminated DCLM validation set `dolmino_splits/dclm_heldout` | `setup/build_dclm_heldout_val.py` |
| simplescale manifest (`_ref` pointers into the subset, in row order) | `setup/build_dclm_rewrite_manifest.py` |
| Submit 1-rewrite-per-doc generation (one run per generator x prompt) | `setup/submit_dclm_rewrites.sh` |
| Export a finished run as a row-aligned split `dclm_rewrites/<subset>_<details>` | `setup/prepare_dclm_rewrites.py` |
| Multi-rewrite study: manifests / submit / per-view export / configs | `setup/build_multirewrite_manifests.py`, `setup/submit_dclm_multirewrites.sh`, `setup/prepare_dclm_multirewrites.py`, `setup/make_multirewrite_configs.py` |

Every export checks each row's `task_id` against the subset's `id` at the same position, and
publishes the output directory atomically. A finished split is therefore guaranteed row-aligned
with its source.

## 7. Cluster gotchas

- **Proxies.** If the login shell sets `http_proxy` / `https_proxy` / `no_proxy`, strip them when
  submitting jobs (`env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u no_proxy -u NO_PROXY sbatch ...`).
- **Preemptible queues.** Jobs on preemptible QoS get requeued. Use a smaller
  `checkpoint.dump.every` (the multi-rewrite configs use `every: 500, keep: 2`) so a requeue loses
  little work.
- **Interrupted evals.** Consolidation is atomic (built in `.consolidated.tmp-*`, then renamed) and an
  eval only skips it when `consolidated/` holds both `consolidated.pth` and `params.json`. A
  killed eval job is safe to rerun. Checkpoints consolidated by older code may have an empty
  `consolidated/`; the new code rebuilds it.
- **Offline evals.** A missing HF dataset makes an OLMES task fail at download time; pre-download
  (section 3).
