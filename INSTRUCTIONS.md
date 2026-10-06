# lingua-clean — quickstart

Some helpful scripts for (1) uv environment setup, (2) model downloading and conversion, (3) mid-training/pre-training, (4) trained model checkpoint conversion and test some sample generations.

```bash
cd /checkpoint/comem/jacquelinehe/lingua-clean
source scripts/env.sh   # required in every fresh shell; SET YOUR ENV VARIABLES HERE.
```

`DATA_ROOT` should already point at a prepared Dolmino splits dir
(`<domain>_shuffled/<domain>.chunk.*.jsonl` per `dclm`, `flan`, `math`,
`pes2o`, `stackexchange`, `wiki`).

## 1. Install the uv venv (ONE-TIME ONLY)

```bash
bash bin/install_requirements.sh   # `uv sync --frozen` → ./.venv (torch/xformers cu121 + prebuilt flash-attn + pinned deps)
source .venv/bin/activate
```

## 2. Fetch teachers + student, convert student to Lingua DCP (Knowledge distillation only!)

Must run on a compute node with internet access.
Downloads `OLMo-2-0425-1B-Instruct`, `OLMo-2-1124-7B-Instruct`, and
`OLMo-2-0425-1B @ stage1-step1907359-tokens4001B`, then runs
`setup/hf_to_lingua_dcp.py` so the student model (here, just Olmo 2 1B base) exists in both HF
(`${STUDENT_HF_PATH}`) and Lingua DCP (`${STUDENT_INIT_PATH}`) layouts.

```bash
sbatch --account=${SLURM_ACCOUNT} --qos=${SLURM_QOS} \
    --time=00:30:00 --cpus-per-task=8 --mem=64G --gres=gpu:1 \
    --output=${SLURM_LOG_DIR}/fetch-%j.out --error=${SLURM_LOG_DIR}/fetch-%j.err \
    --wrap "bash -c 'source scripts/env.sh && bash scripts/fetch_models.sh'"
```

~24 GB total. For student model download only: `ONLY=student bash scripts/fetch_models.sh`.

## 3. Train

### Mid-training (KD recipes)

```bash
RECIPE=idx139 DRY_RUN=1 bash scripts/launch_midtrain.sh   # inspect sbatch
RECIPE=idx139 bash scripts/launch_midtrain.sh             # submit (4-node, ~56 h)
```

`RECIPE` is any YAML basename under `apps/main/configs/recipes/` that JH was randomly experimenting with (e.g., each new idea is a separate yaml file). Headline:
`ntp_baseline` (you probably just want this as a default), `fkd_1b`, `fkd_7b`, `idx139` (entropy-gated RKL, 7B teacher),
`idx148` (entropy-switched RKL), `entband_q30_q30_lam1p0`. Full table in
`README.md`. Checkpoints land at `${MIDTRAIN_ROOT}/<recipe>/checkpoints/<step>/`.

### From-scratch pretrain

Uses the same Dolmino mid-train mix under `${DATA_ROOT}` as the KD recipes,
just with `init_ckpt_path: null` so it trains from scratch, e.g., with random init, instead of starting from
the pre-trained OLMo 1B student.

```bash
CONFIG=dolmino_pretrain bash scripts/launch_pretrain.sh   # 2-node, 9600 steps, ~26 B tokens
```

`CONFIG` resolves to `apps/main/configs/<CONFIG>.yaml`. dolmino_pretrain is just vanilla NTP.

### Auto-evals

KD recipes are configured to launch OLMES evaluation every 1200 steps. This requires a separate env (venv or conda) with the OLMES package installed — point `OLMES_CONDA_ENV` (in `scripts/env.sh`) at it. To disable, remove `eval_backend: olmes` from the recipe YAML (or switch it to `harness` to use lm-eval-harness instead, matching the `dolmino_pretrain` config).

## 4. Convert back to HF + sample generations

After training with lingua, you may want to convert your lingua checkpoint to an HF one for compatibility with your eval suite.

```bash
# Pick any per-step checkpoint dir (the one with .metadata + *.distcp):
CKPT=${MIDTRAIN_ROOT}/idx139/checkpoints/0000028800

# DCP → consolidated → HF (writes to ${CKPT}/hf):
bash scripts/lingua_to_hf.sh ${CKPT}

# Sample generations on one GPU:
sbatch --account=${SLURM_ACCOUNT} --qos=${SLURM_QOS} \
    --time=00:10:00 --cpus-per-task=4 --mem=32G --gres=gpu:1 \
    --output=${SLURM_LOG_DIR}/gen-%j.out --error=${SLURM_LOG_DIR}/gen-%j.err \
    --wrap "bash -c 'source scripts/env.sh && python scripts/hf_generate.py ${CKPT}/hf'"
```

`scripts/hf_generate.py <hf_dir>` prints completions for three fixed prompts (to make sure conversion goes through correctly);
`--prompt` overrides, `--temperature 0.7` enables sampling.
