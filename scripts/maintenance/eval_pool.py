"""Eval pool: steal pending single-GPU `olmes_evals` jobs and run them on this allocation's GPUs.

Runs once per node (srun --ntasks-per-node=1); one thread per local GPU. Each thread repeatedly:
  1. lists our PENDING olmes_evals jobs (squeue) and maps each to its DUMP_DIR (= its stdout path up to /logs/);
  2. claims one by creating DUMP_DIR/.pool_claim with O_EXCL (shared FS, so claims are unique across nodes), then
     scancels the pending job (only if scancel succeeds and the job was still pending do we run it);
  3. runs `python -m apps.main.eval_olmes config=DUMP_DIR/base_config.yaml` on its GPU, logging to
     DUMP_DIR/logs/pool_<slurm job>_<gpu>.log.
Exits when there has been nothing to steal for --idle-exit minutes, or near the time limit.
"""
import argparse, os, subprocess, threading, time

REPO = "/checkpoint/mnm/sgeng/pretrain/lingua-clean"
lock = threading.Lock()


def pending():
    out = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-n", "olmes_evals", "-t", "PD",
                          "-O", "JobID:20,STDOUT:400"], capture_output=True, text=True).stdout
    jobs = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and "/logs/" in parts[1]:
            jobs.append((parts[0], parts[1].split("/logs/")[0]))
    return jobs


def claim(jid, dump):
    try:
        fd = os.open(f"{dump}/.pool_claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.environ.get('SLURM_JOB_ID')} {jid}\n".encode()); os.close(fd)
    state = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%T"], capture_output=True, text=True).stdout.strip()
    if state != "PENDING" or subprocess.run(["scancel", jid]).returncode != 0:
        os.remove(f"{dump}/.pool_claim")  # it started on its own (or vanished): leave it
        return False
    return True


def worker(gpu, deadline, idle_exit):
    idle_since = time.time()
    while time.time() < deadline:
        job = None
        with lock:
            for jid, dump in pending():
                if os.path.exists(f"{dump}/validation.json"):
                    continue
                if claim(jid, dump):
                    job = (jid, dump)
                    break
        if job is None:
            if time.time() - idle_since > idle_exit * 60:
                return
            time.sleep(60)
            continue
        jid, dump = job
        os.makedirs(f"{dump}/logs", exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), DUMP_DIR=dump, OMP_NUM_THREADS="1",
                   LAUNCH_WITH="SBATCH", WANDB_MODE=os.environ.get("WANDB_MODE", "offline"), PYTHONPATH=REPO)
        for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                  "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS"):
            env.pop(k, None)
        t0 = time.time()
        with open(f"{dump}/logs/pool_{os.environ.get('SLURM_JOB_ID')}_{gpu}.log", "a") as log:
            rc = subprocess.run([f"{REPO}/.venv-kd/bin/python", "-u", "-m", "apps.main.eval_olmes",
                                 f"config={dump}/base_config.yaml"], cwd=REPO, env=env, stdout=log,
                                stderr=subprocess.STDOUT).returncode
        ok = os.path.exists(f"{dump}/validation.json")
        print(f"[{os.uname().nodename} gpu{gpu}] {dump} rc={rc} ok={ok} {time.time() - t0:.0f}s", flush=True)
        if not ok:
            os.remove(f"{dump}/.pool_claim")  # let a later pass (or a resubmitted job) retry
        idle_since = time.time()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=7.5)
    ap.add_argument("--idle-exit", type=float, default=30)
    ap.add_argument("--gpus", type=int, default=4)
    a = ap.parse_args()
    deadline = time.time() + a.hours * 3600 - 1800  # don't start an eval in the last 30 min
    ts = [threading.Thread(target=worker, args=(g, deadline, a.idle_exit)) for g in range(a.gpus)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
