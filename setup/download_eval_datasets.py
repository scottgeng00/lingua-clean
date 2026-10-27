"""Pre-download the HF datasets behind OLMES tasks/suites so offline eval jobs (HF_HUB_OFFLINE=1) work.

Run in the OLMES env from a node with internet, from the OLMES checkout:

    cd $OLMES_ROOT && HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
        .venv-lmeval/bin/python /path/to/lingua-clean/setup/download_eval_datasets.py lingua:v2

Suites are expanded recursively. allenai/paloma is gated: accept its terms on the Hub and set HF_TOKEN.
The Hub rate-limits bursts (HTTP 429); rerun for the tasks that failed. Prints one "<task> ok <n docs>" or
"<task> FAIL ..." line per task and writes download_status.json to the working directory.
"""
import copy, sys, traceback, json
from oe_eval.launch import resolve_task_suite
from oe_eval.configs.tasks import TASK_CONFIGS
from oe_eval.run_eval import load_task
SUITES = sys.argv[1:]
tasks = []
for s in SUITES:
    tasks += resolve_task_suite(s, {})
res = {}
for t in tasks:
    try:
        cfg = copy.deepcopy(TASK_CONFIGS[t]) if t in TASK_CONFIGS else {"task_name": t}
        cfg.pop("gantry_args", None)
        task = load_task(cfg)
        task.download()
        n = None
        try:
            n = len(task.get_eval_docs(limit=None)) if hasattr(task, "get_eval_docs") else None
        except Exception:
            pass
        res[t] = f"ok {n}"
    except Exception as e:
        res[t] = f"FAIL {type(e).__name__}: {str(e)[:200]}"
    print(t, res[t], flush=True)
json.dump(res, open("download_status.json", "w"), indent=1)
