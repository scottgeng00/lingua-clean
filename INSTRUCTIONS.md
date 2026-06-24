# lingua-clean — quickstart

Some helpful scripts for (1) conda environment setup, (2) model downloading and conversion, (3) mid-training/pre-training, (4) trained model checkpoint conversion and test some sample generations.

```bash
cd /checkpoint/comem/jacquelinehe/lingua-clean
source scripts/env.sh   # required in every fresh shell
```

`DATA_ROOT` should already point at a prepared Dolmino splits dir
(`<domain>_shuffled/<domain>.chunk.*.jsonl` per `dclm`, `flan`, `math`,
`pes2o`, `stackexchange`, `wiki`). If you need to (re)build it,
`setup/download_prepare_dolmino.py` is the script; it's a ~24 h job.

## 1. Install the conda env

```bash
conda create -n lingua python=3.11 -y
conda activate lingua
bash bin/install_requirements.sh   # pip deps + torch/xformers cu121 + flash-attn
```

## 2. Fetch teachers + student, convert student to Lingua DCP

Must run on a compute node with internet access.
Downloads `OLMo-2-0425-1B-Instruct`, `OLMo-2-1124-7B-Instruct`, and
`OLMo-2-0425-1B @ stage1-step1907359-tokens4001B`, then runs
`setup/hf_to_lingua_dcp.py` so the student exists in both HF
(`${STUDENT_HF_PATH}`) and Lingua DCP (`${STUDENT_INIT_PATH}`) layouts.

```bash
sbatch --account=${SLURM_ACCOUNT} --qos=${SLURM_QOS_TEST} \
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

`RECIPE` is any YAML basename under `apps/main/configs/recipes/`. Headline:
`ntp_baseline`, `fkd_1b`, `fkd_7b`, `idx139` (entropy-gated RKL, 7B teacher),
`idx148` (entropy-switched RKL), `entband_q30_q30_lam1p0`. Full table in
`README.md`. Checkpoints land at `${MIDTRAIN_ROOT}/<recipe>/checkpoints/<step>/`.

### From-scratch pretrain

Uses the same Dolmino data as mid-training (the `dclm_shuffled` shard under
`${DATA_ROOT}`), just with `init_ckpt_path: null` so it trains from random
init instead of the OLMo student.

```bash
CONFIG=dclm_pt bash scripts/launch_pretrain.sh   # 2-node, 9600 steps, ~26 B tokens
```

`CONFIG` resolves to `apps/main/configs/<CONFIG>.yaml`.

## 4. Convert back to HF + sample generations

```bash
# Pick any per-step checkpoint dir (the one with .metadata + *.distcp):
CKPT=${MIDTRAIN_ROOT}/idx139/checkpoints/0000028800

# DCP → consolidated → HF (writes to ${CKPT}/hf):
bash scripts/lingua_to_hf.sh ${CKPT}

# Sample generations on one GPU:
sbatch --account=${SLURM_ACCOUNT} --qos=${SLURM_QOS_TEST} \
    --time=00:10:00 --cpus-per-task=4 --mem=32G --gres=gpu:1 \
    --output=${SLURM_LOG_DIR}/gen-%j.out --error=${SLURM_LOG_DIR}/gen-%j.err \
    --wrap "bash -c 'source scripts/env.sh && python scripts/hf_generate.py ${CKPT}/hf'"
```

`scripts/hf_generate.py <hf_dir>` prints completions for three fixed prompts;
`--prompt` overrides, `--temperature 0.7` enables sampling.
