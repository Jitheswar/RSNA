#!/usr/bin/env bash
# Restartable feature extraction for the frozen Ticket 36 cohort.
# Submit only after the exact Ticket 36 resource approval record exists.
#SBATCH --job-name=rsna-ticket36-features
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-131%2
#SBATCH --output=/home/ch.jitheswar/RSNA/artifacts/private/ticket36-expanded-supervision/slurm-feature-%A_%a.out
#SBATCH --error=/home/ch.jitheswar/RSNA/artifacts/private/ticket36-expanded-supervision/slurm-feature-%A_%a.err
#SBATCH --chdir=/home/ch.jitheswar/RSNA

set -euo pipefail
umask 077

project_dir=/home/ch.jitheswar/RSNA
run_root="$project_dir/artifacts/private/ticket36-expanded-supervision"
python="$project_dir/.venv/bin/python"

if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "Ticket 36 requires the cluster1-cpu partition" >&2
  exit 1
fi
if [[ ! -x "$python" || ! -x /usr/bin/time || ! -x /usr/bin/scontrol ]]; then
  echo "Ticket 36 requires the project venv, /usr/bin/time and scontrol" >&2
  exit 1
fi
if [[ ! -f "$run_root/ticket36-run-plan.json" || ! -f "$run_root/private/code-revision.env" ]]; then
  echo "Ticket 36 frozen plan and source identity are required" >&2
  exit 1
fi
if [[ ! "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ || "$SLURM_ARRAY_TASK_ID" -gt 131 ]]; then
  echo "Ticket 36 feature task is outside the frozen 0-131 batch plan" >&2
  exit 1
fi

job_id="${SLURM_JOB_ID:?}"
array_job_id="${SLURM_ARRAY_JOB_ID:?}"
array_task_id="${SLURM_ARRAY_TASK_ID:?}"
restart_count="${SLURM_RESTART_COUNT:-0}"
if [[ ! "$restart_count" =~ ^[0-9]+$ ]]; then
  echo "Ticket 36 could not identify this Slurm restart attempt" >&2
  exit 1
fi
source "$run_root/private/code-revision.env"
export RSNA_PROJECT_ROOT="$project_dir"
if [[ ! -d "${RSNA_TICKET36_SOURCE_ROOT:-}/src/rsna_knee" ]]; then
  echo "Ticket 36 frozen source snapshot is unavailable" >&2
  exit 1
fi
export PYTHONPATH="$RSNA_TICKET36_SOURCE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
expected_script="$RSNA_TICKET36_SOURCE_ROOT/slurm/materialize_ticket36_features.sbatch"
allocation_lookup_id="${array_job_id}_${array_task_id}"
allocation_record="$(/usr/bin/scontrol show job -o "$allocation_lookup_id")"
"$python" - "$allocation_record" \
  "$run_root/allocations/feature-${array_task_id}-${job_id}-restart-${restart_count}.json" \
  "$job_id" "$array_job_id" "$array_task_id" "$expected_script" "$restart_count" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

raw_record = sys.argv[1]
expected_job_id, expected_array_job_id, expected_array_task_id = sys.argv[3:6]
expected_command = sys.argv[6]
restart_count = int(sys.argv[7])
if sum(token.startswith("JobId=") for token in raw_record.split()) != 1:
    raise SystemExit("scontrol did not return exactly one Ticket 36 array task record")
fields = {
    key: value
    for token in raw_record.split()
    if "=" in token
    for key, value in (token.split("=", 1),)
}
if (
    fields.get("ArrayJobId") != expected_array_job_id
    or fields.get("ArrayTaskId") != expected_array_task_id
    or fields.get("Command") != expected_command
    or (
        expected_job_id != expected_array_job_id
        and fields.get("JobId") != expected_job_id
    )
):
    raise SystemExit("scontrol returned a different Ticket 36 array task")
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
    raise SystemExit("Ticket 36 feature allocation differs from the approved CPU envelope")
destination = Path(sys.argv[2])
destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
if destination.is_symlink() or destination.exists() or destination.parent.is_symlink():
    raise SystemExit("Ticket 36 allocation attempt evidence already exists or is unsafe")
if destination.parent.stat().st_mode & 0o077:
    raise SystemExit("Ticket 36 allocation directory is not private")
payload = {
    "schema_version": "rsna-ticket36-slurm-allocation-v1",
    "job_id": expected_job_id,
    "scontrol_job_id": fields.get("JobId"),
    "array_task_id": expected_array_task_id,
    "restart_count": restart_count,
    "command": fields["Command"],
    "partition": fields["Partition"],
    "nodes": int(fields["NumNodes"]),
    "cpus": int(fields["NumCPUs"]),
    "tasks": int(fields["NumTasks"]),
    "cpus_per_task": int(fields["CPUs/Task"]),
    "memory_per_node": fields["MinMemoryNode"],
    "time_limit": fields["TimeLimit"],
    "over_subscribe": fields["OverSubscribe"],
    "gpu_tres": False,
    "raw_scontrol_record": raw_record,
}
descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
temporary = Path(temporary_name)
try:
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, destination, follow_symlinks=False)
finally:
    temporary.unlink(missing_ok=True)
PY

# This cluster omits some allocation fields from the job environment. Populate
# them only after scontrol has attested the exact allocation above.
export SLURM_JOB_PARTITION=cluster1-cpu
export SLURM_JOB_NUM_NODES=1
export SLURM_NTASKS=1
export SLURM_CPUS_PER_TASK=16
export SLURM_MEM_PER_NODE=64G
export SLURM_TIMELIMIT=06:00:00
export SLURM_JOB_EXCLUSIVE=0

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

compute_path="$run_root/compute/materialize-${SLURM_ARRAY_TASK_ID}-${SLURM_JOB_ID}-restart-${restart_count}.json"
compute_temporary="$run_root/compute/.materialize-${SLURM_ARRAY_TASK_ID}-${SLURM_JOB_ID}-restart-${restart_count}.tmp"
if [[ -e "$compute_path" || -L "$compute_path" || -e "$compute_temporary" || -L "$compute_temporary" ]]; then
  echo "Ticket 36 materialization compute attempt path already exists or is unsafe" >&2
  exit 1
fi
set +e
/usr/bin/time \
  --format='{"schema_version":"rsna-ticket36-materialization-compute-v1","elapsed_seconds":%e,"user_cpu_seconds":%U,"system_cpu_seconds":%S,"peak_rss_kb":%M}' \
  --output="$compute_temporary" \
  "$python" -m rsna_knee.ticket36 materialize --batch-index "$SLURM_ARRAY_TASK_ID"
materialization_status=$?
set -e
"$python" - "$compute_temporary" "$compute_path" "$materialization_status" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
status = int(sys.argv[3])
if source.is_symlink() or not source.is_file():
    raise SystemExit("Ticket 36 materialization timer did not publish attempt metrics")
payload = json.loads(source.read_text(encoding="utf-8"))
if payload.get("schema_version") != "rsna-ticket36-materialization-compute-v1":
    raise SystemExit("Ticket 36 materialization timer wrote an unsupported record")
payload["process_exit_status"] = status
destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
if destination.is_symlink() or destination.exists() or destination.parent.is_symlink():
    raise SystemExit("Ticket 36 materialization compute attempt already exists or is unsafe")
descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
temporary = Path(temporary_name)
try:
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, destination, follow_symlinks=False)
finally:
    temporary.unlink(missing_ok=True)
source.unlink()
PY
if (( materialization_status != 0 )); then
  exit "$materialization_status"
fi
