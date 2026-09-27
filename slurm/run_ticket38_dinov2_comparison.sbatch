#!/usr/bin/env bash
# Frozen three-fold Ticket 38 DINOv2 CPU comparison and OOF analysis.
#SBATCH --job-name=rsna-ticket38
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:57:00
#SBATCH --output=/home/ch.jitheswar/RSNA/artifacts/private/ticket38-dinov2/slurm-%j.out
#SBATCH --error=/home/ch.jitheswar/RSNA/artifacts/private/ticket38-dinov2/slurm-%j.err
#SBATCH --chdir=/home/ch.jitheswar/RSNA

set -euo pipefail
umask 077

project_dir=/home/ch.jitheswar/RSNA
campaign_root="$project_dir/artifacts/private/ticket38-dinov2"
run_root="$campaign_root/attempt-2"
source_snapshot="$campaign_root/source-snapshot-2"
plan_path="$campaign_root/ticket38-run-plan-retry.json"
python="$project_dir/.venv/bin/python"
model_root=/home/ch.jitheswar/models/dinov2-small-ed25f3a31f01632728cabb09d1542f84ab7b0056
model_weights="$model_root/model.safetensors"
approval="$project_dir/.scratch/rsna-leaderboard/resource-approvals/ticket38-dinov2-vits14-cpu.json"
fold_manifest="$project_dir/artifacts/private/ticket-04-real-v2/private/fold-manifest.json"
integration_manifest="$project_dir/artifacts/private/ticket-05-real-training-input/private/integration-manifest.json"
source_archive="$project_dir/rsna-knee-abnormality-detection.zip"
job_id="${SLURM_JOB_ID:?}"
restart_count="${SLURM_RESTART_COUNT:-0}"

if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "Ticket 38 requires the cluster1-cpu partition" >&2
  exit 1
fi
if [[ ! -x "$python" || ! -x /usr/bin/time || ! -x /usr/bin/scontrol ]]; then
  echo "Ticket 38 requires the project venv, /usr/bin/time and scontrol" >&2
  exit 1
fi
if [[ ! -f "$plan_path" || ! -f "$approval" || ! -f "$model_weights" ]]; then
  echo "Ticket 38 frozen plan, approval and exact model snapshot are required" >&2
  exit 1
fi
if [[ -e "$run_root/fold-0" || -e "$run_root/fold-1" || -e "$run_root/fold-2" ]]; then
  echo "Ticket 38 fold output already exists; refusing to overwrite a prior attempt" >&2
  exit 1
fi

expected_command="$project_dir/slurm/run_ticket38_dinov2_comparison.sbatch"
allocation_record="$(/usr/bin/scontrol show job -o "$job_id")"
"$python" - "$allocation_record" "$job_id" "$expected_command" \
  "$run_root/allocations/allocation-${job_id}-restart-${restart_count}.json" \
  "$plan_path" "$approval" "$model_weights" \
  "$expected_command" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

raw_record, expected_job_id, expected_command, allocation_path, plan_path, approval_path, weights_path, script_path = sys.argv[1:]
if sum(token.startswith("JobId=") for token in raw_record.split()) != 1:
    raise SystemExit("scontrol did not return exactly one Ticket 38 job record")
fields = {
    key: value
    for token in raw_record.split()
    if "=" in token
    for key, value in (token.split("=", 1),)
}
if fields.get("JobId") != expected_job_id or fields.get("Command") != expected_command:
    raise SystemExit("scontrol returned a different Ticket 38 batch job")
plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
plan_digest = hashlib.sha256(
    json.dumps(
        {key: value for key, value in plan.items() if key != "plan_sha256"},
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
if plan_digest != plan.get("plan_sha256"):
    raise SystemExit("Ticket 38 frozen run plan digest does not match its contents")
expected = {
    "Partition": "cluster1-cpu",
    "NumNodes": "1",
    "NumCPUs": "16",
    "NumTasks": "1",
    "CPUs/Task": "16",
    "MinMemoryNode": "64G",
    "TimeLimit": plan["execution"]["wall_time_limit"],
    "OverSubscribe": "OK",
}
observed = {key: fields.get(key) for key in expected}
tres = fields.get("TRES", "")
gres = fields.get("Gres", "")
if observed != expected or "gpu" in tres.casefold() or "gpu" in gres.casefold():
    raise SystemExit(json.dumps({"expected": expected, "observed": observed, "TRES": tres, "Gres": gres}, sort_keys=True))

approval = Path(approval_path)
weights = Path(weights_path)
script = Path(script_path)
sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if sha256(approval) != plan["authorization"]["approval_sha256"]:
    raise SystemExit("Ticket 38 resource approval differs from the frozen run plan")
if sha256(weights) != plan["encoder"]["weights_sha256"]:
    raise SystemExit("Ticket 38 model weights differ from the frozen run plan")
if sha256(script) != plan["execution"]["script_sha256"]:
    raise SystemExit("Ticket 38 Slurm script differs from the frozen run plan")
manifest_path = Path(plan["execution"]["source_manifest_path"])
if sha256(manifest_path) != plan["execution"]["source_manifest_sha256"]:
    raise SystemExit("Ticket 38 source snapshot manifest differs from the frozen run plan")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
snapshot_root = manifest_path.parent
for item in manifest["files"]:
    path = snapshot_root / item["path"]
    if path.is_symlink() or not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
        raise SystemExit(f"Ticket 38 source snapshot entry failed verification: {item['path']}")
if manifest["snapshot_sha256"] != plan["execution"]["source_snapshot_sha256"]:
    raise SystemExit("Ticket 38 source snapshot identity differs from the frozen run plan")
if snapshot_root.resolve() != (Path.cwd() / plan["execution"]["source_snapshot_path"]).resolve():
    raise SystemExit("Ticket 38 source snapshot path differs from the frozen run plan")

destination = Path(allocation_path)
destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
if destination.is_symlink() or destination.exists() or destination.parent.is_symlink():
    raise SystemExit("Ticket 38 allocation evidence already exists or is unsafe")
if destination.parent.stat().st_mode & 0o077:
    raise SystemExit("Ticket 38 allocation evidence directory is not private")
payload = {
    "schema_version": "rsna-ticket38-slurm-allocation-v1",
    "job_id": fields["JobId"],
    "restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
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

export RSNA_PROJECT_ROOT="$project_dir"
export PYTHONPATH="$source_snapshot/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -m 700 -p "$run_root/compute"

run_timed() {
  local label="$1"
  shift
  local temporary="$run_root/compute/.${label}-${job_id}.tmp"
  local destination="$run_root/compute/${label}-${job_id}.json"
  if [[ -e "$temporary" || -L "$temporary" || -e "$destination" || -L "$destination" ]]; then
    echo "Ticket 38 compute evidence path already exists or is unsafe: $label" >&2
    return 1
  fi
  set +e
  /usr/bin/time \
    --format='{"schema_version":"rsna-ticket38-observed-compute-v1","elapsed_seconds":%e,"user_cpu_seconds":%U,"system_cpu_seconds":%S,"peak_rss_kb":%M}' \
    --output="$temporary" \
    "$@"
  local command_status=$?
  set -e
  "$python" - "$temporary" "$destination" "$command_status" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
status = int(sys.argv[3])
if source.is_symlink() or not source.is_file():
    raise SystemExit("Ticket 38 timer did not publish process counters")
payload = json.loads(source.read_text(encoding="utf-8"))
if payload.get("schema_version") != "rsna-ticket38-observed-compute-v1":
    raise SystemExit("Ticket 38 timer wrote an unsupported compute record")
payload["process_exit_status"] = status
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
  return "$command_status"
}

fold=0
while [[ "$fold" -le 2 ]]; do
  run_timed "fold-${fold}" \
    "$python" -m rsna_knee.training_cli \
    --integration-manifest "$integration_manifest" \
    --source "$source_archive" \
    --fold-manifest "$fold_manifest" \
    --fold "$fold" \
    --output "$run_root/fold-${fold}" \
    --image-variant ticket38-frozen-dinov2-small-vits14 \
    --pooling-id plane-stratified-series-mean-v1 \
    --pretrained-weights "$model_weights" \
    --resource-approval "$approval" \
    --feature-cache-root "$run_root/frozen-feature-cache" \
    --cache-root "$run_root/series-cache" \
    --source-version rsna-2026-08-04 \
    --source-fingerprint 25a70674e5af8808c4cf6d7e0da6c8b8f894e1856a11e064470ea0bcdab25648 \
    --preflight-evidence "$run_root/fold-${fold}.preflight.json" \
    --evidence "$run_root/fold-${fold}.public.json" \
    --execution-attempt-id "ticket38-${job_id}-fold-${fold}" \
    --require-slurm
  fold=$((fold + 1))
done

run_timed "analysis" \
  "$python" -m rsna_knee.ticket38 \
  --plan "$plan_path" \
  --fold-manifest "$fold_manifest" \
  --integration-manifest "$integration_manifest" \
  --source "$source_archive" \
  --dinov2-runs-root "$run_root" \
  --resnet-runs-root "$project_dir/artifacts/private/ticket32-resnet18" \
  --series-cache-root "$run_root/series-cache" \
  --feature-cache-root "$run_root/frozen-feature-cache" \
  --output-root "$run_root"
