# Results viewer

A small web app for comparing runs across experiment groups: held-out validation NLL,
OLMES downstream scores and gold-answer BPB, and training curves, read straight from
the run directories.

```bash
# live: reads the run dirs on page load and on "Refresh"
.venv/bin/python analysis/results_viewer/serve.py --port 8765
# from your laptop:
ssh -L 8765:localhost:8765 <login-node>     # then open http://localhost:8765

# one self-contained HTML file with the data inlined (share / open offline)
.venv/bin/python analysis/results_viewer/serve.py --export /tmp/results.html
```

Uses only the repo venv (stdlib server + PyYAML); the page is plain HTML/JS with no
external assets. The server binds to localhost by default (`--host` to change).

## Adding runs and groups

Edit `runs.yaml`. Each group is a titled list of runs; each run needs an `id`, a
`label`, its `dir` (relative to `root`), and a fixed color slot `color: 1-8`, so a run
keeps its color in every view and selection. Slots must be unique across runs you want
to show together. `val:` points at a different validation file (the qwen3 baselines
use their `val1000/` re-evaluation so all runs share the same 1000-doc val sets).

## What it reads

| source | path under the run dir |
|---|---|
| training loss, LR, tokens/step | `metrics.jsonl` |
| validation NLL per Dolmino source | `metrics.validation.jsonl` (or `val:`) |
| OLMES per-task metrics | `{evals,olmes_base,olmes}/<step>/olmes_results/[shard_*/]task-*-metrics.json` |

Subtask families are macro-averaged like OLMES does (57 `mmlu_<subject>:rc::olmes`
-> `mmlu:rc::olmes`). An average over fewer subtasks than expected, e.g. an eval that
is still running, is left out of charts and tables. `collect.py` can also be run on its
own to dump everything as JSON (`--out results.json`).

## Views

- **Validation**: one chart per val source plus the mean; NLL/token or bits/char;
  absolute or Δ vs a reference run at the same step.
- **Downstream**: OLMES tasks (toggle chips); primary score or gold BPB; absolute or Δ.
- **Training**: training loss (own training data, so not comparable across data mixes)
  and learning rate.
- **Table**: every metric for the selected runs at the latest step they all share
  (default), each run's latest step, or a chosen step; best per row in bold, Δ vs the
  reference run.

Hover a chart (or focus it and use ←/→) for a tooltip listing every run at that step.
The view state is kept in the URL hash, so a view can be bookmarked or shared.
