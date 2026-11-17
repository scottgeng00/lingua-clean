"""Per-cluster settings for launching lingua-clean jobs, picked automatically (stdlib only: scripts/env.sh runs it
with the system python3 before any venv exists).

Detection: $LINGUA_CLUSTER (a profile name) if set, else Slurm's ClusterName ($SLURM_CLUSTER_NAME or `scontrol
show config`), else the GPU type in the partitions' GRES (`sinfo`). Kernel choices (FA4 vs flash-attn 2, ...) are
not made here: they follow the GPU the process actually runs on (torch.cuda.get_device_capability()).

Storage convention: every cluster has ~/scottg/pretrain pointing at that cluster's storage root (on Dalton
/home/sgeng/scottg -> /checkpoint/mnm/sgeng), so the /home/sgeng/scottg/pretrain paths in configs and scripts resolve
everywhere. PRETRAIN_ROOT is that path.

    python3 lingua/cluster.py            # print the detected profile
    eval "$(python3 lingua/cluster.py --shell)"   # what scripts/env.sh does: export defaults (existing values win)
"""

import argparse
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

PRETRAIN_ROOT = "/home/sgeng/scottg/pretrain"


@dataclass(frozen=True)
class Cluster:
    name: str
    account: str
    qos: str  # default (shared / preemptible) QoS
    high_qos: str  # higher-priority QoS, used only when asked for (e.g. SLURM_QOS=$SLURM_HIGH_QOS)
    gpus_per_node: int
    cpus_per_gpu: int  # what a job requests per GPU (not necessarily the whole node)
    mem_per_gpu_gb: int
    arch: str
    partition: str = ""  # empty: omit --partition (the site infers it)
    exclude: str = ""  # nodes to avoid (hardware errors)
    names: Tuple[str, ...] = ()  # Slurm ClusterName values
    gpu_types: Tuple[str, ...] = ()  # GRES GPU types identifying the cluster when its name is unknown
    env: Dict[str, str] = field(default_factory=dict)  # extra defaults exported by scripts/env.sh
    notes: str = ""


CLUSTERS = {c.name: c for c in (
    Cluster(
        name="gb300-dalton", names=("fair-cw-use2-3",), gpu_types=("gb300",),
        account="mnm", partition="g3", qos="g3_comm_shared", high_qos="g3_mnm_high",
        gpus_per_node=4, cpus_per_gpu=32, mem_per_gpu_gb=180, arch="aarch64", exclude="g3-144-119",
        env={"LINGUA_FA4_SITE": f"{PRETRAIN_ROOT}/speed-extras/site"},
        notes="FAIR CW USE2-3 (Dalton): 4x GB300 (sm_103) per node, 144 CPUs, 875 GB",
    ),
    Cluster(
        name="h100-dm1", names=("fair-cw-use1-1",), gpu_types=("h100",),
        account="dream", partition="h100_mdf", qos="h100_mdf_lowest", high_qos="h100_mdf_explore",
        gpus_per_node=8, cpus_per_gpu=16, mem_per_gpu_gb=220, arch="x86_64",
        notes="FAIR CW USE1-1 (DM1): 8x H100 per node, 128 CPUs, 1.86 TB; no /checkpoint/mnm (team dir /checkpoint/dream)",
    ),
    Cluster(
        name="h200", gpu_types=("h200",),
        account="dream", partition="gpu-h200", qos="h200_dream_high", high_qos="h200_dream_high",
        gpus_per_node=8, cpus_per_gpu=24, mem_per_gpu_gb=240, arch="x86_64",
        notes="original H200 cluster (pre-GB300 defaults; unverified)",
    ),
)}


def _cluster_name() -> Optional[str]:
    if os.environ.get("SLURM_CLUSTER_NAME"):
        return os.environ["SLURM_CLUSTER_NAME"]
    try:
        out = subprocess.run(["scontrol", "show", "config"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "ClusterName":
            return value.strip()
    return None


def _gpu_types() -> set:
    try:
        out = subprocess.run(["sinfo", "-h", "-o", "%G"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return set()
    types = set()
    for gres in out.split():
        for item in gres.split(","):
            parts = item.split("(")[0].split(":")
            if parts[0] == "gpu" and len(parts) >= 3:
                types.add(parts[1].lower())
    return types


def detect(name: Optional[str] = None) -> Cluster:
    """Profile `name`, else $LINGUA_CLUSTER, else by ClusterName, else by GPU type."""
    name = name or os.environ.get("LINGUA_CLUSTER")
    if name:
        if name not in CLUSTERS:
            raise SystemExit(f"unknown cluster profile {name!r}; known: {', '.join(CLUSTERS)}")
        return CLUSTERS[name]
    slurm_name = _cluster_name()
    for c in CLUSTERS.values():
        if slurm_name in c.names:
            return c
    types = _gpu_types()
    matches = [c for c in CLUSTERS.values() if types & set(c.gpu_types)]
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(f"cannot pick a cluster profile (ClusterName={slurm_name!r}, GPU types={sorted(types)}); "
                     f"set LINGUA_CLUSTER ({', '.join(CLUSTERS)}) or add this cluster to lingua/cluster.py")


def shell_exports(c: Cluster) -> str:
    """`export VAR="${VAR:-default}"` lines; values already in the environment win."""
    values = {
        "LINGUA_CLUSTER": c.name,
        "PRETRAIN_ROOT": PRETRAIN_ROOT,
        "SLURM_ACCOUNT": c.account,
        "SLURM_PARTITION": c.partition,
        "SLURM_QOS": c.qos,
        "SLURM_HIGH_QOS": c.high_qos,
        "SLURM_EXCLUDE": c.exclude,
        "GPUS_PER_NODE": str(c.gpus_per_node),
        "CPUS_PER_GPU": str(c.cpus_per_gpu),
        "MEM_PER_GPU_GB": str(c.mem_per_gpu_gb),
        "CPUS_PER_NODE": str(c.cpus_per_gpu * c.gpus_per_node),
        "MEM_PER_NODE": f"{c.mem_per_gpu_gb * c.gpus_per_node}G",
        **c.env,
    }
    lines = [f'export {k}="${{{k}-{v}}}"' for k, v in values.items()]
    # Defaults for any sbatch call (command-line flags still win), so standalone .sbatch files need no
    # cluster-specific #SBATCH lines.
    lines += ['export SBATCH_ACCOUNT="${SBATCH_ACCOUNT-${SLURM_ACCOUNT}}"',
              'export SBATCH_PARTITION="${SBATCH_PARTITION-${SLURM_PARTITION}}"',
              'export SBATCH_QOS="${SBATCH_QOS-${SLURM_QOS}}"',
              '[ -n "${SBATCH_PARTITION}" ] || unset SBATCH_PARTITION']
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell", action="store_true", help="print export lines for scripts/env.sh")
    ap.add_argument("--cluster", help="profile name (default: detect)")
    a = ap.parse_args()
    c = detect(a.cluster)
    print(shell_exports(c) if a.shell else "\n".join(f"{k}: {v}" for k, v in asdict(c).items()))
