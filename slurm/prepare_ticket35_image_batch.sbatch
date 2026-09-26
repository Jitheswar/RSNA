#!/usr/bin/env bash
# Prepare one Ticket 35 image batch. Submit only after the matching CPU scope
# in .scratch/rsna-leaderboard/resource-approvals is approved.
#SBATCH --job-name=rsna-ticket35-image
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/ch.jitheswar/RSNA/artifacts/private/ticket35-expanded-image-preparation/logs/batch-%j.out
#SBATCH --error=/home/ch.jitheswar/RSNA/artifacts/private/ticket35-expanded-image-preparation/logs/batch-%j.err
#SBATCH --chdir=/home/ch.jitheswar/RSNA

set -euo pipefail
umask 077

project_dir=/home/ch.jitheswar/RSNA
python="$project_dir/.venv/bin/python"
run_root="$project_dir/artifacts/private/ticket35-expanded-image-preparation"
job_id="${SLURM_JOB_ID:?Slurm job ID is required}"
batch_index="${TICKET35_BATCH_INDEX:?set TICKET35_BATCH_INDEX to the approved batch number}"
approval="$project_dir/.scratch/rsna-leaderboard/resource-approvals/ticket35-image-preparation-cpu.json"
membership="$run_root/membership-v2.json"
inventory="$project_dir/artifacts/private/ticket-03-inventory-run/private/inventory.json"
reference_integration="$project_dir/artifacts/private/ticket-05-real-training-input/private/integration-manifest.json"
archive="$project_dir/rsna-knee-abnormality-detection.zip"
cache_root="$run_root/series-cache"
batch_root="$run_root/batches/batch-$batch_index"

if [[ "${SLURM_JOB_PARTITION:-}" != cluster1-cpu ]]; then
  echo "Ticket 35 image preparation requires cluster1-cpu" >&2
  exit 1
fi
if [[ ! -x "$python" || ! -f "$archive" || ! -f "$membership" || ! -f "$approval" ]]; then
  echo "Ticket 35 inputs, approval record, or project Python are missing" >&2
  exit 1
fi
if [[ ! -x /usr/bin/scontrol || ! -x /usr/bin/time ]]; then
  echo "Ticket 35 requires scontrol and /usr/bin/time for allocation evidence" >&2
  exit 1
fi
mkdir -p "$batch_root" "$run_root/logs"
chmod 700 "$batch_root" "$run_root/logs"

allocation_record="$(/usr/bin/scontrol show job -o "$job_id")"
"$python" - "$allocation_record" "$batch_root/job-$job_id-allocation.json" <<'PY'
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
    raise SystemExit("Ticket 35 Slurm allocation differs from the CPU-only proposal")
Path(sys.argv[2]).write_text(
    json.dumps(
        {
            "schema_version": "rsna-ticket35-slurm-allocation-v1",
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
        sort_keys=True,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

# Cluster nodes omit these environment variables; set them only after the
# allocation has passed the exact scontrol check above.
export SLURM_JOB_PARTITION=cluster1-cpu
export SLURM_CPUS_PER_TASK=16
export SLURM_MEM_PER_NODE=64G
export SLURM_TIMELIMIT=06:00:00
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

/usr/bin/time \
  --format='{"schema_version":"rsna-ticket35-observed-compute-v1","elapsed_seconds":%e,"user_cpu_seconds":%U,"system_cpu_seconds":%S,"peak_rss_kb":%M}' \
  --output="$batch_root/job-$job_id-compute.json" \
  "$python" -m rsna_knee.ticket35_cli prepare \
    --membership "$membership" \
    --batch-index "$batch_index" \
    --approval "$approval" \
    --inventory "$inventory" \
    --reference-integration-manifest "$reference_integration" \
    --source-archive "$archive" \
    --cache-root "$cache_root" \
    --storage-root canonical-repository=/home/ch.jitheswar/RSNA \
    --storage-root rsna-ticket28=/home/ch.jitheswar/RSNA-ticket-28 \
    --storage-root rsna-ticket29=/home/ch.jitheswar/RSNA-ticket-29 \
    --storage-root rsna-ticket30=/home/ch.jitheswar/RSNA-ticket-30 \
    --storage-root ticket15-primary=/home/ch.jitheswar/ticket-15-real-20260914 \
    --storage-root ticket15-repeat=/home/ch.jitheswar/ticket-15-real-20260914-repeat \
    --storage-root ticket16=/home/ch.jitheswar/ticket-16-real-20260914 \
    --storage-root ticket17=/home/ch.jitheswar/ticket-17-real-20260921 \
    --storage-root ticket18-additive=/home/ch.jitheswar/ticket-18-additive-v2 \
    --storage-root ticket18-real=/home/ch.jitheswar/ticket-18-real-20260923 \
    --storage-root ticket19=/home/ch.jitheswar/ticket-19-wider-encoder-v1 \
    --storage-root ticket21-primary=/home/ch.jitheswar/ticket-21-pseudo-20260923 \
    --storage-root ticket21-repeat=/home/ch.jitheswar/ticket-21-pseudo-20260923-attempt2 \
    --storage-root ticket22-v1=/home/ch.jitheswar/ticket-22-distillation-v1 \
    --storage-root ticket22-v2=/home/ch.jitheswar/ticket-22-distillation-v2 \
    --storage-root ticket22-v3=/home/ch.jitheswar/ticket-22-distillation-v3 \
    --storage-root ticket30-audit=/home/ch.jitheswar/ticket-30-report-audit-20260925 \
    --storage-root ticket13-gpu-environment=/home/ch.jitheswar/ticket13-gpu-env-20260913 \
    --storage-root ticket13-operational-environment=/home/ch.jitheswar/ticket13-operational-20260913 \
    --ledger-output "$batch_root/job-$job_id-storage-ledger.json" \
    --selection-output "$batch_root/job-$job_id-selection.json" \
    --batch-manifest-output "$batch_root/job-$job_id-integration-manifest.json" \
    --evidence-output "$batch_root/job-$job_id-evidence.json"
