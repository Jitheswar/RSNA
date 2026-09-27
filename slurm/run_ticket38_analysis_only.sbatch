#!/usr/bin/env bash
# One analysis-only Ticket 38 replay; folds, checkpoints and feature caches are inputs.
#SBATCH --job-name=rsna-t38-analysis
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/home/ch.jitheswar/RSNA/artifacts/private/ticket38-dinov2/analysis-only-1/slurm-%j.out
#SBATCH --error=/home/ch.jitheswar/RSNA/artifacts/private/ticket38-dinov2/analysis-only-1/slurm-%j.err
#SBATCH --chdir=/home/ch.jitheswar/RSNA

set -euo pipefail
umask 077

project_dir=/home/ch.jitheswar/RSNA
campaign_root="$project_dir/artifacts/private/ticket38-dinov2"
analysis_root="$campaign_root/analysis-only-1"
plan_path="$analysis_root/analysis-only-run-plan.json"
parent_plan="$campaign_root/ticket38-run-plan-retry.json"
source_snapshot="$campaign_root/source-snapshot-3"
approval="$project_dir/.scratch/rsna-leaderboard/resource-approvals/ticket38-dinov2-analysis-only-cpu-v1.json"
python="$project_dir/.venv/bin/python"
model_weights=/home/ch.jitheswar/models/dinov2-small-ed25f3a31f01632728cabb09d1542f84ab7b0056/model.safetensors
fold_manifest="$project_dir/artifacts/private/ticket-04-real-v2/private/fold-manifest.json"
integration_manifest="$project_dir/artifacts/private/ticket-05-real-training-input/private/integration-manifest.json"
source_archive="$project_dir/rsna-knee-abnormality-detection.zip"
job_id="${SLURM_JOB_ID:?}"

if [[ "${SLURM_JOB_PARTITION:-}" != "cluster1-cpu" ]]; then
  echo "Ticket 38 analysis requires the cluster1-cpu partition" >&2
  exit 1
fi
if [[ ! -x "$python" || ! -x /usr/bin/time || ! -x /usr/bin/scontrol ]]; then
  echo "Ticket 38 analysis requires the project venv, /usr/bin/time and scontrol" >&2
  exit 1
fi
if [[ ! -d "$analysis_root" || -e "$analysis_root/public" || -e "$analysis_root/private" ]]; then
  echo "Ticket 38 analysis output root is missing or already contains results" >&2
  exit 1
fi
if [[ ! -f "$plan_path" || ! -f "$parent_plan" || ! -f "$approval" || ! -f "$model_weights" ]]; then
  echo "Ticket 38 analysis plan, parent plan, approval or pinned weights are missing" >&2
  exit 1
fi

expected_command="$project_dir/slurm/run_ticket38_analysis_only.sbatch"
allocation_record="$(/usr/bin/scontrol show job -o "$job_id")"
allocation_path="$analysis_root/allocations/allocation-${job_id}.json"
mkdir -m 700 -p "$analysis_root/allocations" "$analysis_root/compute"
"$python" - "$allocation_record" "$job_id" "$expected_command" \
  "$allocation_path" "$plan_path" "$parent_plan" "$approval" \
  "$source_snapshot" "$model_weights" "$fold_manifest" "$integration_manifest" \
  "$source_archive" "$expected_command" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

(
    raw_record,
    expected_job_id,
    expected_command,
    allocation_path,
    plan_path,
    parent_plan_path,
    approval_path,
    snapshot_root_text,
    weights_path,
    fold_manifest_path,
    integration_manifest_path,
    source_archive_path,
    script_path,
) = sys.argv[1:]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def canonical_digest(payload: dict[str, object], checksum_key: str) -> str:
    body = {key: value for key, value in payload.items() if key != checksum_key}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

if sum(token.startswith("JobId=") for token in raw_record.split()) != 1:
    raise SystemExit("scontrol did not return exactly one Ticket 38 analysis job")
fields = {
    key: value
    for token in raw_record.split()
    if "=" in token
    for key, value in (token.split("=", 1),)
}
if fields.get("JobId") != expected_job_id or fields.get("Command") != expected_command:
    raise SystemExit("scontrol returned a different Ticket 38 analysis job")

plan_path = Path(plan_path)
parent_plan_path = Path(parent_plan_path)
approval_path = Path(approval_path)
plan = json.loads(plan_path.read_text(encoding="utf-8"))
parent_plan = json.loads(parent_plan_path.read_text(encoding="utf-8"))
approval = json.loads(approval_path.read_text(encoding="utf-8"))
if canonical_digest(plan, "plan_sha256") != plan.get("plan_sha256"):
    raise SystemExit("Ticket 38 analysis plan digest does not match its contents")
if canonical_digest(parent_plan, "plan_sha256") != parent_plan.get("plan_sha256"):
    raise SystemExit("parent Ticket 38 frozen plan digest does not match its contents")
if approval.get("approval_status") != "approved_by_project_owner" or approval.get("decision") != "approved_for_analysis_only":
    raise SystemExit("Ticket 38 analysis-only approval is not active")
if sha256(approval_path) != plan["authorization"]["approval_sha256"]:
    raise SystemExit("Ticket 38 analysis approval differs from the frozen plan")
if sha256(parent_plan_path) != plan["parent_plan_file_sha256"]:
    raise SystemExit("Ticket 38 parent plan file changed")
if parent_plan.get("plan_sha256") != plan["parent_plan_sha256"]:
    raise SystemExit("Ticket 38 parent plan identity changed")
if sha256(Path(script_path)) != plan["execution"]["script_sha256"]:
    raise SystemExit("Ticket 38 analysis Slurm script differs from the frozen plan")

resources = plan["execution"]["resources"]
expected = {
    "Partition": resources["partition"],
    "NumNodes": str(resources["nodes"]),
    "NumCPUs": str(resources["cpus_per_task"]),
    "NumTasks": str(resources["tasks"]),
    "CPUs/Task": str(resources["cpus_per_task"]),
    "MinMemoryNode": f"{resources['memory_gib']}G",
    "TimeLimit": resources["wall_time_limit"],
}
observed = {key: fields.get(key) for key in expected}
tres = fields.get("TRES", "")
gres = fields.get("Gres", "")
if observed != expected or "gpu" in tres.casefold() or "gpu" in gres.casefold():
    raise SystemExit(json.dumps({"expected": expected, "observed": observed, "TRES": tres, "Gres": gres}, sort_keys=True))
if fields.get("Requeue") not in {"0", "NO"}:
    raise SystemExit("Ticket 38 analysis job must not be requeued")

approval_resources = approval["resources"]
for key in ("partition", "nodes", "tasks", "cpus_per_task", "memory_gib", "walltime_seconds", "gpu_count", "external_spend_usd"):
    if approval_resources.get(key) != resources.get(key):
        raise SystemExit(f"Ticket 38 approval/plan resource mismatch for {key}")
if resources["gpu_count"] != 0 or resources["external_spend_usd"] != 0:
    raise SystemExit("Ticket 38 analysis scope must remain zero-GPU and zero-spend")

snapshot_root = Path(snapshot_root_text)
manifest_path = snapshot_root / "ticket38-source-manifest.json"
if sha256(manifest_path) != plan["execution"]["source_manifest_file_sha256"]:
    raise SystemExit("Ticket 38 analysis source manifest file changed")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("snapshot_sha256") != plan["execution"]["source_snapshot_sha256"]:
    raise SystemExit("Ticket 38 analysis source snapshot identity changed")
aggregate = hashlib.sha256(
    json.dumps(
        {"schema_version": manifest["schema_version"], "files": manifest["files"]},
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()
if aggregate != manifest["snapshot_sha256"]:
    raise SystemExit("Ticket 38 analysis source snapshot aggregate digest is invalid")
for item in manifest["files"]:
    path = snapshot_root / item["path"]
    if path.is_symlink() or not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
        raise SystemExit(f"Ticket 38 analysis source snapshot entry failed verification: {item['path']}")

expected_inputs = plan["inputs"]
for field, path_text in (
    ("model_weights_sha256", weights_path),
    ("fold_manifest_file_sha256", fold_manifest_path),
    ("integration_manifest_file_sha256", integration_manifest_path),
):
    if sha256(Path(path_text)) != expected_inputs[field]:
        raise SystemExit(f"Ticket 38 analysis input digest changed: {field}")
archive = Path(source_archive_path)
archive_stat = archive.stat()
if archive.is_symlink() or archive_stat.st_size != expected_inputs["source_archive_bytes"]:
    raise SystemExit("Ticket 38 source archive identity or size changed")
if parent_plan["cohort"]["source_fingerprint"] != expected_inputs["source_fingerprint"]:
    raise SystemExit("Ticket 38 source fingerprint differs from the approved parent plan")

for fold in range(3):
    entry = expected_inputs["dinov2_folds"][fold]
    run_path = Path(entry["run_path"])
    checkpoint_path = Path(entry["checkpoint_path"])
    if sha256(run_path) != entry["run_sha256"] or sha256(checkpoint_path) != entry["checkpoint_sha256"]:
        raise SystemExit(f"Ticket 38 fold {fold} run or checkpoint changed")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("run_id") != entry["run_id"]:
        raise SystemExit(f"Ticket 38 fold {fold} run ID changed")
for fold, entry in enumerate(expected_inputs["ticket32_folds"]):
    run_path = Path(entry["run_path"])
    if sha256(run_path) != entry["run_sha256"]:
        raise SystemExit(f"Ticket 32 fold {fold} comparator run changed")

destination = Path(allocation_path)
if destination.is_symlink() or destination.exists() or destination.parent.is_symlink():
    raise SystemExit("Ticket 38 analysis allocation evidence already exists or is unsafe")
payload = {
    "schema_version": "rsna-ticket38-analysis-slurm-allocation-v1",
    "job_id": fields["JobId"],
    "command": fields["Command"],
    "partition": fields["Partition"],
    "nodes": int(fields["NumNodes"]),
    "cpus_per_task": int(fields["CPUs/Task"]),
    "tasks": int(fields["NumTasks"]),
    "memory_per_node": fields["MinMemoryNode"],
    "time_limit": fields["TimeLimit"],
    "requeue": False,
    "gpu_tres": False,
    "raw_scontrol_record": raw_record,
    "approval_sha256": sha256(approval_path),
    "analysis_plan_sha256": plan["plan_sha256"],
}
destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
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

set +e
/usr/bin/time \
  --format='{"schema_version":"rsna-ticket38-analysis-compute-v1","elapsed_seconds":%e,"user_cpu_seconds":%U,"system_cpu_seconds":%S,"peak_rss_kb":%M}' \
  --output="$analysis_root/compute/.analysis-${job_id}.tmp" \
  "$python" -m rsna_knee.ticket38 \
  --plan "$parent_plan" \
  --fold-manifest "$fold_manifest" \
  --integration-manifest "$integration_manifest" \
  --source "$source_archive" \
  --dinov2-runs-root "$campaign_root/attempt-2" \
  --resnet-runs-root "$project_dir/artifacts/private/ticket32-resnet18" \
  --series-cache-root "$campaign_root/attempt-2/series-cache" \
  --feature-cache-root "$campaign_root/attempt-2/frozen-feature-cache" \
  --output-root "$analysis_root"
analysis_status=$?
set -e

"$python" - "$analysis_root/compute/.analysis-${job_id}.tmp" \
  "$analysis_root/compute/analysis-${job_id}.json" "$analysis_status" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
status = int(sys.argv[3])
payload = json.loads(source.read_text(encoding="utf-8"))
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
    source.unlink(missing_ok=True)
PY
exit "$analysis_status"
