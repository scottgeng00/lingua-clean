# results/

Write-ups of experimental results, one markdown file per study or result snapshot. This directory is
the place to look for "what did we find", in contrast to configs (what we ran) and analysis/ (code that
computes numbers).

Contract for every file here:

- **Name:** `YYYY-MM-DD_<short_topic>.md`, dated by when the numbers were pulled. Add a new dated file
  for a substantially updated snapshot instead of silently rewriting old conclusions. Small edits that
  fill in pending runs are fine; record them in the file's status line.
- **Status line** at the top: complete or partial, and which runs are still pending.
- **Setup:** model, token budget, hyperparameters, data, and what is varied, with enough detail to read
  the table without opening configs.
- **Runs:** the config names under `apps/main/configs/` and any data dirs or generation runs they depend on.
- **Metrics:** the exact definitions used in the tables (which step, PPL aggregation, which tasks).
- **Results:** the tables, with noise estimates when known.
- **Takeaways and caveats:** what the numbers support, and the confounds.
- **Reproduce:** the command that regenerates the tables (e.g. `analysis/results_viewer/genscale_table.py`).

## Index

- [2026-10-26_multirewrite_study.md](2026-10-26_multirewrite_study.md): repeating real data vs multiple
  Qwen3.5-9B rewrites per doc (iid, T=1.2, Nemotron-CC prompt mixture, WebOrganizer formats, Gacha)
  in a 10B real + 10B variable setting. Partial.
