"""Freeze exact MRI pixel materialization and exhaustive native-source replay."""

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
ROOT = PROJECT / "artifacts/private/ticket62-mri-pixel-cohort-v1"
PILOT = PROJECT / "artifacts/private/ticket61-cpu-full-backbone-pilot-v1"
COHORT = PROJECT / "artifacts/private/ticket49-convnext-feature-cohort/plan.json"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket62-mri-pixel-materialization-20261002.json"


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
        raise FileExistsError("Immutable MRI pixel cohort request already exists")
    pilot = read(PILOT / "plan.json")
    result = read(PILOT / "pilot-result.json")
    verification = read(PILOT / "independent-validation.json")
    terminal = read(PILOT / "terminal-verification.json")
    if (verification["status"] != "COMPLETE" or verification["result_sha256"] != digest(PILOT / "pilot-result.json")
            or not terminal["passed"] or verification["prediction_max_abs_error"] != 0):
        raise ValueError("Independent real-MRI whole-backbone CPU admission required")
    cohort = read(COHORT)
    counts = cohort["valid_slice_counts"]
    reference = read(t36.FEATURE_ROOT / "feature-manifest.json")
    if len(counts) != 4228 or sum(counts.values()) != 187272 or set(counts) != {r["study_instance_uid"] for r in reference["features"]}:
        raise ValueError("Complete admitted MRI cohort identity differs")
    external_model = PROJECT.parent / "models/qwen3-8b-b968826d9c46dd6066d109eabc6255188de91218"
    usage = subprocess.check_output(["du", "--apparent-size", "-s", "-B1", str(PROJECT), str(external_model)], text=True)
    used = sum(int(line.split()[0]) for line in usage.splitlines())
    payload_bytes = sum(counts.values()) * 1 * 224 * 224 * 4
    overhead_allowance = 2_000_000_000
    temporary_peak = 3 * 128 * 224 * 224 * 4 + 4 * 128 * 224 * 224 * 4
    additional_peak = payload_bytes + overhead_allowance + temporary_peak
    free = shutil.disk_usage(PROJECT).free
    if used + additional_peak > 2_000_000_000_000 or additional_peak > free:
        raise ValueError("Fresh storage headroom does not admit pixel materialization")
    ROOT.mkdir(mode=0o700, parents=True)
    pixel_root = ROOT / "pixels"
    pixel_root.mkdir(mode=0o700)
    snapshot = ROOT / "source-snapshot"
    shutil.copytree(PROJECT / "src/rsna_knee", snapshot / "src/rsna_knee",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    extract_source = PROJECT / "scripts/ticket62_mri_pixel_materialize.py"
    shutil.copyfile(extract_source, snapshot / "ticket62_mri_pixel_materialize.py")
    verifier_source = PROJECT / "scripts/validate_ticket62_mri_pixels.py"
    shutil.copyfile(verifier_source, snapshot / verifier_source.name)
    partitions = [{"partition":0,"study_uids":sorted(counts),"valid_slices":sum(counts.values())}]
    prior_root = PROJECT / "artifacts/private/ticket57-resnet-prefix-cohort"
    prior_results = list(prior_root.glob("partition-*/partition-result.json"))
    if len(prior_results)!=2 or sum(read(p)["study_count"] for p in prior_results)!=4228:
        raise ValueError("Complete actual predecessor extraction cost required")
    prior_seconds = sum(read(p)["elapsed_seconds"] for p in prior_results)
    forecast = 2*prior_seconds+900
    if forecast>10740:raise ValueError("Complete materialization/replay projection exceeds bounded request")
    resources = {"partition":"cluster1-cpu","nodes_per_job":1,"tasks_per_job":1,
                 "cpus_per_task":16,"memory_gib_per_job":64,"walltime_seconds_per_job":10800,
                 "maximum_concurrent_jobs":1,"partition_count":1,"maximum_launches_per_partition":1,
                 "maximum_allocated_node_seconds":10800,"code_watchdog_seconds_per_job":10740,
                 "gpu_count":0,"exclusive":False,"requeue":False,"external_spend_usd":0}
    runtime = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "pydicom")}
    identity = {"representation":"exact selected packed MRI grayscale pixels; no encoder/fitted transform",
                "precision":"float32","shape_per_valid_slice":[1,224,224],"max_series":16,"max_slices_per_series":8,
                "slice_mapping":"ascending flat indices of slice_mask AND series_mask",
                "padding_reconstruction":"exact zeros", "quantization":False,
                "integration_manifest_checksum":cohort["integration_manifest_checksum"],"runtime_versions":runtime}
    fixed = [AUTH, PILOT / "plan.json", PILOT / "pilot-result.json", PILOT / "independent-validation.json", PILOT / "terminal-verification.json",
             COHORT, t36.INTEGRATION_PATH, t36.SERIES_INDEX_PATH, t36.FEATURE_ROOT / "feature-manifest.json",
             Path(__file__), extract_source, verifier_source, *prior_results]
    fixed.extend(path for path in snapshot.rglob("*") if path.is_file())
    archive = t36.SOURCE_ARCHIVE
    plan = {"schema_version": "rsna-ticket62-mri-pixel-cohort-plan-v1", "ticket": 62,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_root": str(ROOT),
            "pixel_root": str(pixel_root), "source_snapshot": str(snapshot), "decision_path": str(DECISION),
            "authorization_path": str(AUTH), "authorization_sha256": digest(AUTH),
            "integration_path": str(t36.INTEGRATION_PATH), "integration_manifest_checksum": cohort["integration_manifest_checksum"],
            "series_index_path": str(t36.SERIES_INDEX_PATH), "series_cache_root": str(t36.TICKET35_SERIES_CACHE),
            "reference_feature_manifest_path": str(t36.FEATURE_ROOT / "feature-manifest.json"),
            "reference_feature_root": str(t36.FEATURE_ROOT),
            "source_archive_path": str(archive), "source_archive_bytes": archive.stat().st_size,
            "source_archive_mtime_ns": archive.stat().st_mtime_ns,
            "pixel_identity": identity, "pixel_identity_sha256": identity_digest(identity),
            "admitted_pilot_plan":str(PILOT / "plan.json"),"admitted_pilot_result":str(PILOT / "pilot-result.json"),
            "forecast_node_seconds":forecast,"actual_predecessor_complete_cpu_seconds":prior_seconds,
            "runtime_versions": runtime,
            "partitions": partitions, "valid_slice_counts": counts, "cohort_study_count": 4228,
            "cohort_valid_slices": 187272, "resources": resources,
            "storage_observation": {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
                "du_output": usage, "tracked_existing_apparent_bytes": used, "filesystem_free_bytes": free,
                "pixel_payload_bytes": payload_bytes, "metadata_overhead_allowance_bytes": overhead_allowance,
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
          "basis": "Verified whole-backbone real-MRI pilot and complete predecessor source/cost; materialize exact unfitted MRI pixels and independently replay every native source before a separately admitted full comparison.",
          "new_exact_owner_numeric_approval_claimed": False, "gpu_ledger_cap_changed": False,
          "new_gpu_request": False, "new_weight_download": False,
          "full_training_authorized": False, "candidate_promotion": False})
    launchers = []
    for partition in range(1):
        launcher = ROOT / f"extract-{partition}.sbatch"
        with launcher.open("x") as handle:
            handle.write(f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket62-p{partition}
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
{PROJECT}/.venv/bin/python {snapshot}/ticket62_mri_pixel_materialize.py {ROOT}/plan.json {checksum} {partition}
{PROJECT}/.venv/bin/python {snapshot}/validate_ticket62_mri_pixels.py {ROOT}
''')
        launchers.append({"partition": partition, "path": str(launcher), "sha256": digest(launcher)})
    write(ROOT / "prepared-validation.json", {"plan_sha256": checksum, "launchers": launchers,
        "partition_study_counts": [len(row["study_uids"]) for row in partitions],
        "partition_slice_counts": [row["valid_slices"] for row in partitions], "launched": False})
    print(json.dumps({"plan_sha256": checksum, "partition_study_counts": [len(row["study_uids"]) for row in partitions],
                      "partition_slice_counts": [row["valid_slices"] for row in partitions]}))


if __name__ == "__main__":
    main()
