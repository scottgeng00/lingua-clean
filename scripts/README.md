# scripts/

Launch scripts for `lingua-clean`. The public entry point is
`launch_midtrain.sh`; the rest is plumbing.

## Files

- `launch_midtrain.sh` — small `sbatch` wrapper. Takes `RECIPE=<name>` as an
  env var, resolves the matching YAML under
  `../apps/main/configs/recipes/<name>.yaml`, and submits a 4-node H200 sbatch
  job. Supports `DRY_RUN=1` to print the sbatch command without submitting.
- `_sbatch_inner.sh` — the script that actually runs inside the sbatch
  allocation. Activates the uv venv (`$LINGUA_VENV`), sets the standard wandb /
  CUDA env, derives `NPROC_PER_NODE` / `MASTER_ADDR` / `MASTER_PORT` /
  `NODE_RANK` from SLURM, and `torchrun`s `apps.main.train` with the chosen
  recipe YAML.

## Common usage

```bash
# DRY_RUN: print sbatch invocation, don't submit.
RECIPE=ntp_baseline DRY_RUN=1 bash scripts/launch_midtrain.sh

# Submit a 4-node mid-training run for the headline RKL entgate recipe.
RECIPE=idx139 bash scripts/launch_midtrain.sh

# Override nodes (e.g. for a 1-node smoke test, though wallclock will balloon).
RECIPE=fkd_7b NNODES=1 bash scripts/launch_midtrain.sh

# Override total steps for a quick sanity check (default 28800).
RECIPE=ntp_baseline STEPS_OVERRIDE=200 bash scripts/launch_midtrain.sh
```

## Recipe discovery

```bash
ls ../apps/main/configs/recipes/ | sed 's/\.yaml$//'
```
