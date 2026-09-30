"""Freeze the separate 32-study CPU pilot; does not download weights or launch."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "artifacts/private/ticket48-convnext-cpu"
BASE = PROJECT / "artifacts/private/ticket36-expanded-supervision"
CHECKPOINT_SHA = "77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a"


def digest(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    if (ROOT / "plan.json").exists():
        raise FileExistsError("pilot plan is already frozen")
    snapshot = ROOT / "source-snapshot"
    shutil.copytree(PROJECT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(PROJECT / "scripts/ticket48_convnext_cpu_pilot.py", snapshot / "pilot.py")
    manifest_path = BASE / "features/feature-manifest.json"
    integration_path = BASE / "private/expanded-integration-manifest.json"
    source = json.loads(integration_path.read_text())
    features = json.loads(manifest_path.read_text())["features"]
    studies = {row["StudyInstanceUID"]: row for row in source["studies"]}
    counts = {
        uid: sum(min(8, len(series["entries"])) for series in row["series"])
        for uid, row in studies.items()
    }
    reports = sorted(
        [row["study_instance_uid"] for row in features
         if row["source_kind"] == "ticket35-prepared-series-cache-read-only"],
        key=lambda uid: (counts[uid], hashlib.sha256(uid.encode()).hexdigest()),
    )
    selected = [reports[round(index * (len(reports) - 1) / 31)] for index in range(32)]
    if len(set(selected)) != 32 or len(features) != 4228 or set(counts) != {
        row["study_instance_uid"] for row in features
    }:
        raise ValueError("pilot source cohort identity differs")
    evidence = PROJECT / "artifacts/private/external-notebook-audit/frontier-2026-09-30/complement-training"
    model_manifest = evidence / "convnext_tiny.fb_in22k_ft_in1k-api-manifest.json"
    response = json.loads(model_manifest.read_text())["response"]
    if (response["sha"] != "bc48a87f119bae1b06418f5340bfa97ef75d4609"
            or response["cardData"]["license"] != "apache-2.0"):
        raise ValueError("public checkpoint revision/license identity differs")
    wheels = sorted((ROOT / "wheels").glob("*.whl"))
    if len(wheels) != 3:
        raise ValueError("isolated runtime wheel inventory differs")
    runtime_site = ROOT / "runtime/lib/python3.11/site-packages"
    runtime_files = [runtime_site / "rsna-existing-cpu-dependencies.pth"]
    for package in ["timm", "torchvision", "PIL"]:
        runtime_files.extend(path for path in (runtime_site / package).rglob("*")
                             if path.is_file() and "__pycache__" not in path.parts)
    input_files = [integration_path, manifest_path, BASE / "ticket36-run-plan.json",
                   BASE / "private/ticket35-series-index.json", model_manifest,
                   evidence / "convnext_tiny.fb_in22k_ft_in1k-config.json"]
    source_files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    fixed = [*input_files, *wheels, *runtime_files, *source_files]
    checkpoint = ROOT / "weights/model.safetensors"
    approval_path = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket48-convnext-cpu-pilot.json"
    plan = {
        "schema_version": "rsna-ticket48-convnext-cpu-pilot-plan-v1", "ticket": 48,
        "run_root": str(ROOT), "source_snapshot": str(snapshot),
        "pilot_source_sha256": digest(snapshot / "pilot.py"),
        "frozen_files": [{"path": str(path), "sha256": digest(path)} for path in fixed]
            + [{"path": str(checkpoint), "sha256": CHECKPOINT_SHA}],
        "approval_path": str(approval_path), "checkpoint_path": str(checkpoint),
        "checkpoint_bytes": 114374272, "checkpoint_sha256": CHECKPOINT_SHA,
        "checkpoint_revision": response["sha"],
        "download_url": "https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k/resolve/"
                        + response["sha"] + "/model.safetensors",
        "integration_path": str(integration_path),
        "integration_manifest_checksum": source["manifest_checksum"],
        "reference_feature_manifest_path": str(manifest_path),
        "reference_feature_root": str(BASE / "features"),
        "series_index_path": str(BASE / "private/ticket35-series-index.json"),
        "series_cache_root": str(PROJECT / "artifacts/private/ticket35-expanded-image-preparation/series-cache"),
        "pilot_study_uids": selected,
        "pilot_valid_slice_counts": {uid: counts[uid] for uid in selected},
        "selection_policy": "32 equally spaced order statistics of report-only valid-slice count; SHA256(uid) breaks ties; no labels read",
        "cohort_study_count": len(studies), "cohort_valid_slices": sum(counts.values()),
        "maximum_full_extraction_node_hours": 24,
        "followup_completion_deadline": "2026-10-03T23:59:59+05:30",
        "runtime_versions": {name: importlib.metadata.version(name) for name in
                             ["torch", "torchvision", "timm", "pillow", "numpy", "safetensors"]},
        "representation": {"image_size": [224, 224], "max_slices_per_series": 8,
                           "max_series": 16, "grayscale_to_rgb": "repeat",
                           "normalization_mean": [0.485, 0.456, 0.406],
                           "normalization_std": [0.229, 0.224, 0.225],
                           "classifier": "removed after strict checkpoint load", "feature_dimension": 768,
                           "precision": "float32", "batch_size": 16, "backbone": "frozen eval"},
        "resources": {"partition": "cluster1-cpu", "nodes": 1, "tasks": 1,
                      "cpus_per_task": 16, "memory_gib": 64, "walltime_seconds": 600,
                      "code_watchdog_seconds": 590, "gpu_count": 0,
                      "maximum_concurrent_pilot_jobs": 1, "exclusive": False,
                      "maximum_launch_attempts": 3, "external_spend_usd": 0},
        "network_during_execution": False, "training": False,
        "candidate_promotion": False, "cohort_extraction_authorized": False,
    }
    save(ROOT / "plan.json", plan)
    plan_sha = digest(ROOT / "plan.json")
    save(approval_path, {
        "schema_version": "rsna-ticket48-resource-approval-v1",
        "approval_status": "approved_by_project_owner", "decision": "approved",
        "approved_by": "Project owner via existing conversation-wide authorization: approved for anything everything; no matter what 1st place",
        "authorization_interpretation": "Applies existing zero-spend campaign authorization to this exact separate CPU pilot; no Ticket 48-specific approval words claimed. Does not expand GPU capacity or alter Ticket 39.",
        "scope": "ticket48-convnext-cpu-pilot", "plan_path": str(ROOT / "plan.json"),
        "plan_sha256": plan_sha, "checkpoint_sha256": CHECKPOINT_SHA,
        "checkpoint_bytes": 114374272, "source_revision": response["sha"],
        "license": "apache-2.0", "new_weight_download": True,
        "rights_status": "eligible_for_private_experiment",
        "rights_basis": "Public ungated checkpoint, exact maintainer card declares Apache-2.0. This record authorizes private investigation and does not grant rights to underlying third-party training data or final redistribution.",
        "resources": plan["resources"], "publication": False,
        "final_candidate_selection": False, "followup_training_authorized": False,
    })
    launch = f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket48-pilot
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00
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
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
{ROOT}/runtime/bin/python {snapshot}/pilot.py {ROOT}/plan.json {plan_sha}
'''
    launch_path = ROOT / "run-pilot.sbatch"
    with launch_path.open("x") as handle:
        handle.write(launch)
    save(ROOT / "prepared-validation.json", {
        "plan_sha256": plan_sha, "launch_sha256": digest(launch_path),
        "pilot_studies": len(selected), "pilot_valid_slices": sum(counts[uid] for uid in selected),
        "cohort_valid_slices": sum(counts.values()), "weights_downloaded": False,
        "launched": False, "external_spend_usd": 0,
    })
    print(json.dumps({"plan_sha256": plan_sha, "pilot_studies": len(selected),
                      "pilot_valid_slices": sum(counts[uid] for uid in selected)}))


if __name__ == "__main__":
    main()
