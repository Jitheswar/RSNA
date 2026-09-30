#!/usr/bin/env bash
# One sequential, paired fold-0 Ticket 34 CPU pilot. Submit only after the
# exact Ticket 34 resource approval record has been approved.
#SBATCH --job-name=rsna-ticket34
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/ch.jitheswar/RSNA/artifacts/private/ticket34-resnet18-supervision/slurm-%j.out
#SBATCH --error=/home/ch.jitheswar/RSNA/artifacts/private/ticket34-resnet18-supervision/slurm-%j.err
#SBATCH --chdir=/home/ch.jitheswar/RSNA

set -euo pipefail
umask 077

project_dir=/home/ch.jitheswar/RSNA
run_root="$project_dir/artifacts/private/ticket34-resnet18-supervision"
python="$project_dir/.venv/bin/python"

if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "Ticket 34 requires the cluster1-cpu partition" >&2
  exit 1
fi
if [[ ! -x "$python" || ! -x /usr/bin/time ]]; then
  echo "Ticket 34 requires the project venv and /usr/bin/time" >&2
  exit 1
fi
if [[ ! -x /usr/bin/scontrol ]]; then
  echo "Ticket 34 requires scontrol to attest the actual Slurm allocation" >&2
  exit 1
fi
if [[ ! -d "$run_root" ]]; then
  echo "Ticket 34 private run root is missing" >&2
  exit 1
fi

allocation_record="$(/usr/bin/scontrol show job -o "${SLURM_JOB_ID:?}")"
"$python" - "$allocation_record" "$run_root/slurm-${SLURM_JOB_ID}-allocation.json" <<'PY'
import json
import sys
from pathlib import Path

fields = {
    key: value
    for token in sys.argv[1].split()
    if "=" in token
    for key, value in (token.split("=", 1),)
}
expected = {
    "Partition": "cluster1-cpu",
    "NumNodes": "1",
    "NumCPUs": "16",
    "NumTasks": "1",
    "CPUs/Task": "16",
    "MinMemoryNode": "64G",
    "TimeLimit": "06:00:00",
    "OverSubscribe": "OK",
}
observed = {key: fields.get(key) for key in expected}
tres = fields.get("TRES", "")
gres = fields.get("Gres", "")
if observed != expected or "gpu" in tres.lower() or "gpu" in gres.lower():
    print(json.dumps({"expected": expected, "observed": observed, "TRES": tres, "Gres": gres}, sort_keys=True), file=sys.stderr)
    raise SystemExit("Ticket 34 Slurm allocation differs from the approved CPU envelope")
Path(sys.argv[2]).write_text(
    json.dumps(
        {
            "schema_version": "rsna-ticket34-slurm-allocation-v1",
            "job_id": fields.get("JobId"),
            "partition": fields["Partition"],
            "nodes": int(fields["NumNodes"]),
            "cpus": int(fields["NumCPUs"]),
            "tasks": int(fields["NumTasks"]),
            "cpus_per_task": int(fields["CPUs/Task"]),
            "memory_per_node": fields["MinMemoryNode"],
            "time_limit": fields["TimeLimit"],
            "over_subscribe": fields["OverSubscribe"],
            "gpu_tres": False,
            "raw_scontrol_record": sys.argv[1],
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

# This cluster omits the memory and time-limit environment variables. Populate
# them only after scontrol attests the exact allocated values above.
export SLURM_JOB_PARTITION=cluster1-cpu
export SLURM_JOB_NUM_NODES=1
export SLURM_NTASKS=1
export SLURM_CPUS_PER_TASK=16
export SLURM_MEM_PER_NODE=64G
export SLURM_TIMELIMIT=06:00:00
export SLURM_JOB_EXCLUSIVE=0

export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
if [[ -z "${RSNA_CODE_REVISION:-}" ]]; then
  RSNA_CODE_REVISION="$(/usr/bin/git -C "$project_dir" rev-parse HEAD 2>/dev/null || echo unavailable)"
fi
export RSNA_CODE_REVISION

compute_evidence="$run_root/slurm-${SLURM_JOB_ID}-observed-compute.json"
/usr/bin/time \
  --format='{"schema_version":"rsna-ticket34-observed-compute-v1","elapsed_seconds":%e,"user_cpu_seconds":%U,"system_cpu_seconds":%S,"peak_rss_kb":%M}' \
  --output="$compute_evidence" \
  "$python" -m rsna_knee.ticket34_cli
