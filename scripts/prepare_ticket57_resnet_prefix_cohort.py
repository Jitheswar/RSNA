"""Freeze the full activation cohort; this command never launches jobs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

from rsna_knee import ticket36 as t36
from rsna_knee.prefix_cache import identity_digest


PROJECT = t36.PROJECT_ROOT
ROOT = PROJECT / "artifacts/private/ticket57-resnet-prefix-cohort"
PILOT = PROJECT / "artifacts/private/ticket56-cpu-layer4-pilot"
COHORT = PROJECT / "artifacts/private/ticket49-convnext-feature-cohort/plan.json"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket57-resnet-prefix-extraction-20261001.json"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main():
    if ROOT.exists() or DECISION.exists():
        raise FileExistsError("Immutable prefix cohort request already exists")
    pilot = read(PILOT / "plan.json")
    result = read(PILOT / "pilot-result.json")
    verification = read(PILOT / "validation.json")
    if (verification["status"] != "COMPLETE" or verification["result_sha256"] != digest(PILOT / "pilot-result.json")
            or not verification["native_cached_parity_exact"]
            or result["cost_disposition"] != "eligible_for_separate_full_route_planning"):
        raise ValueError("Independently verified passing CPU adaptation pilot is required")
    cohort = read(COHORT)
    counts = cohort["valid_slice_counts"]
    reference = read(t36.FEATURE_ROOT / "feature-manifest.json")
    if len(counts) != 4228 or sum(counts.values()) != 187272 or set(counts) != {r["study_instance_uid"] for r in reference["features"]}:
        raise ValueError("Complete admitted MRI cohort identity differs")
    external_model = PROJECT.parent / "models/qwen3-8b-b968826d9c46dd6066d109eabc6255188de91218"
    usage = subprocess.check_output(["du", "--apparent-size", "-s", "-B1", str(PROJECT), str(external_model)], text=True)
    used = sum(int(line.split()[0]) for line in usage.splitlines())
    payload_bytes = sum(counts.values()) * 256 * 14 * 14 * 4
    overhead_allowance = 2_000_000_000
    temporary_peak = 2 * max(counts.values()) * 256 * 14 * 14 * 4
    additional_peak = payload_bytes + overhead_allowance + temporary_peak
    free = shutil.disk_usage(PROJECT).free
    if used + additional_peak > 2_000_000_000_000 or additional_peak > free:
        raise ValueError("Fresh storage headroom does not admit prefix extraction")
    ROOT.mkdir(mode=0o700, parents=True)
    feature_root = ROOT / "features"
    feature_root.mkdir(mode=0o700)
    snapshot = ROOT / "source-snapshot"
    shutil.copytree(PROJECT / "src/rsna_knee", snapshot / "src/rsna_knee",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    extract_source = PROJECT / "scripts/ticket57_resnet_prefix_extract.py"
    shutil.copyfile(extract_source, snapshot / "extract.py")
    partitions = [{"partition": index, "study_uids": [], "valid_slices": 0} for index in range(2)]
    for uid in sorted(counts, key=lambda key: (-counts[key], hashlib.sha256(key.encode()).hexdigest())):
        selected = min(partitions, key=lambda row: (row["valid_slices"], len(row["study_uids"]), row["partition"]))
        selected["study_uids"].append(uid)
        selected["valid_slices"] += counts[uid]
    for row in partitions:
        row["study_uids"].sort()
    resources = {"partition": "cluster1-cpu", "nodes_per_job": 1, "tasks_per_job": 1,
                 "cpus_per_task": 16, "memory_gib_per_job": 64, "walltime_seconds_per_job": 10800,
                 "maximum_concurrent_jobs": 2, "partition_count": 2, "maximum_launches_per_partition": 1,
                 "maximum_allocated_node_seconds": 21600, "code_watchdog_seconds_per_job": 10740,
                 "gpu_count": 0, "exclusive": False, "requeue": False, "external_spend_usd": 0}
    model = {**pilot["model"], "model_id": "resnet18-frozen-slice-features-v1"}
    runtime = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "pydicom")}
    identity = {"encoder": "torchvision-resnet18-imagenet1k-v1", "checkpoint_sha256": digest(t36.WEIGHTS_PATH),
                "stage": "output of frozen layer3 before adapted layer4", "activation_shape_per_slice": [256, 14, 14],
                "precision": "float32", "image_size": [224, 224], "max_series": 16, "max_slices_per_series": 8,
                "slice_mapping": "ascending flat indices of slice_mask AND series_mask",
                "encoding_batch_size": 16, "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225], "normalization_buffers": "fixed eval",
                "integration_manifest_checksum": cohort["integration_manifest_checksum"], "runtime_versions": runtime}
    fixed = [AUTH, PILOT / "plan.json", PILOT / "pilot-result.json", PILOT / "validation.json",
             COHORT, t36.INTEGRATION_PATH, t36.SERIES_INDEX_PATH, t36.FEATURE_ROOT / "feature-manifest.json",
             t36.WEIGHTS_PATH, Path(__file__), extract_source]
    fixed.extend(path for path in snapshot.rglob("*") if path.is_file())
    archive = t36.SOURCE_ARCHIVE
    plan = {"schema_version": "rsna-ticket57-prefix-cohort-plan-v1", "ticket": 57,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_root": str(ROOT),
            "feature_root": str(feature_root), "source_snapshot": str(snapshot), "decision_path": str(DECISION),
            "authorization_path": str(AUTH), "authorization_sha256": digest(AUTH),
            "integration_path": str(t36.INTEGRATION_PATH), "integration_manifest_checksum": cohort["integration_manifest_checksum"],
            "series_index_path": str(t36.SERIES_INDEX_PATH), "series_cache_root": str(t36.TICKET35_SERIES_CACHE),
            "reference_feature_manifest_path": str(t36.FEATURE_ROOT / "feature-manifest.json"),
            "reference_feature_root": str(t36.FEATURE_ROOT), "checkpoint_path": str(t36.WEIGHTS_PATH),
            "checkpoint_sha256": digest(t36.WEIGHTS_PATH), "model": model,
            "source_archive_path": str(archive), "source_archive_bytes": archive.stat().st_size,
            "source_archive_mtime_ns": archive.stat().st_mtime_ns,
            "feature_identity": identity, "feature_identity_sha256": identity_digest(identity),
            "native_feature_parity_tolerance": 1e-6, "runtime_versions": runtime,
            "partitions": partitions, "valid_slice_counts": counts, "cohort_study_count": 4228,
            "cohort_valid_slices": 187272, "resources": resources,
            "storage_observation": {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
                "du_output": usage, "tracked_existing_apparent_bytes": used, "filesystem_free_bytes": free,
                "prefix_payload_bytes": payload_bytes, "metadata_overhead_allowance_bytes": overhead_allowance,
                "temporary_peak_allowance_bytes": temporary_peak,
                "projected_campaign_peak_bytes": used + additional_peak, "storage_soft_cap_bytes": 2_000_000_000_000},
            "frozen_files": [{"path": str(path), "sha256": digest(path)} for path in fixed],
            "network_during_execution": False, "new_checkpoint_download": False,
            "labels_used": False, "training": False, "candidate_promotion": False,
            "full_adaptation_training_selected": False, "competition_submission": False, "final_selection": False}
    write(ROOT / "plan.json", plan)
    checksum = digest(ROOT / "plan.json")
    write(DECISION, {"schema_version": "rsna-campaign-autonomous-decision-v1", "decision": "approved",
          "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
          "decision_made_by": "campaign agent under standing owner authorization",
          "authorization_sha256": digest(AUTH), "plan_sha256": checksum, "resources": resources,
          "basis": "Verified exact native/cached pilot parity and measured cost; materialize only deterministic unfitted activations for a separately specified report-adaptation experiment.",
          "new_exact_owner_numeric_approval_claimed": False, "gpu_ledger_cap_changed": False,
          "new_gpu_request": False, "new_weight_download": False,
          "full_training_authorized": False, "candidate_promotion": False})
    launchers = []
    for partition in range(2):
        launcher = ROOT / f"extract-{partition}.sbatch"
        with launcher.open("x") as handle:
            handle.write(f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket57-p{partition}
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output={ROOT}/slurm-%j.out
#SBATCH --error={ROOT}/slurm-%j.err
#SBATCH --chdir={PROJECT}
set -euo pipefail
umask 077
export RSNA_PROJECT_ROOT={PROJECT}
export PYTHONPATH={snapshot}/src
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
{PROJECT}/.venv/bin/python {snapshot}/extract.py {ROOT}/plan.json {checksum} {partition}
''')
        launchers.append({"partition": partition, "path": str(launcher), "sha256": digest(launcher)})
    write(ROOT / "prepared-validation.json", {"plan_sha256": checksum, "launchers": launchers,
        "partition_study_counts": [len(row["study_uids"]) for row in partitions],
        "partition_slice_counts": [row["valid_slices"] for row in partitions], "launched": False})
    print(json.dumps({"plan_sha256": checksum, "partition_study_counts": [len(row["study_uids"]) for row in partitions],
                      "partition_slice_counts": [row["valid_slices"] for row in partitions]}))


if __name__ == "__main__":
    main()
