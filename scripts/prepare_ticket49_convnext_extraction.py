"""Freeze the exact full-cohort extraction; preparation cannot launch jobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "artifacts/private/ticket49-convnext-feature-cohort"
PILOT = PROJECT / "artifacts/private/ticket48-convnext-cpu"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    pilot = json.loads((PILOT / "plan.json").read_text())
    result_path = PILOT / "job-7934/pilot-result.json"
    result = json.loads(result_path.read_text())
    validation_path = PILOT / "independent-output-validation.json"
    validation = json.loads(validation_path.read_text())
    terminal = PILOT / "controller-7934/terminal-controller.txt"
    if (result["throughput_gate"] != "pass" or validation["result_sha256"] != digest(result_path)
            or validation["feature_files_verified"] != 32
            or "JobState=COMPLETED" not in terminal.read_text()
            or "ExitCode=0:0" not in terminal.read_text()):
        raise ValueError("independently validated pilot did not pass")
    if (ROOT / "plan.json").exists():
        raise FileExistsError("extraction plan already frozen")
    snapshot = ROOT / "source-snapshot"
    shutil.copytree(Path(pilot["source_snapshot"]) / "src", snapshot / "src")
    shutil.copy(PROJECT / "scripts/ticket49_convnext_extract.py", snapshot / "extract.py")
    integration = json.loads(Path(pilot["integration_path"]).read_text())
    prior = json.loads(Path(pilot["reference_feature_manifest_path"]).read_text())["features"]
    counts = {row["StudyInstanceUID"]: sum(min(8, len(series["entries"])) for series in row["series"])
              for row in integration["studies"]}
    if len(prior) != 4228 or set(counts) != {row["study_instance_uid"] for row in prior}:
        raise ValueError("source cohort changed")
    partitions = [{"partition": index, "study_uids": [], "valid_slices": 0} for index in range(2)]
    for uid in sorted(counts, key=lambda item: (-counts[item], hashlib.sha256(item.encode()).hexdigest())):
        selected = min(partitions, key=lambda row: (row["valid_slices"], len(row["study_uids"]), row["partition"]))
        selected["study_uids"].append(uid)
        selected["valid_slices"] += counts[uid]
    for row in partitions:
        row["study_uids"].sort()
    (ROOT / "features").mkdir(mode=0o700)
    source_archive = PROJECT / "rsna-knee-abnormality-detection.zip"
    frozen = [row for row in pilot["frozen_files"] if not str(row["path"]).startswith(pilot["source_snapshot"])]
    for row in frozen:
        if digest(Path(row["path"])) != row["sha256"]:
            raise ValueError("admitted pilot dependency changed")
    more_files = [PILOT / "plan.json", result_path, validation_path, terminal,
                  PROJECT / "scripts/ticket49_convnext_extract.py",
                  PROJECT / "scripts/prepare_ticket49_convnext_extraction.py"]
    more_files.extend(path for path in snapshot.rglob("*") if path.is_file())
    frozen.extend({"path": str(path), "sha256": digest(path)} for path in more_files)
    approval = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket49-convnext-cohort-extraction.json"
    plan = {key: pilot[key] for key in ["checkpoint_path", "checkpoint_sha256", "checkpoint_bytes",
            "checkpoint_revision", "runtime_versions", "integration_path", "integration_manifest_checksum",
            "reference_feature_manifest_path", "reference_feature_root", "series_index_path", "series_cache_root"]}
    plan.update({
        "schema_version": "rsna-ticket49-convnext-extraction-plan-v1", "ticket": 49,
        "run_root": str(ROOT), "source_snapshot": str(snapshot),
        "feature_root": str(ROOT / "features"), "approval_path": str(approval),
        "frozen_files": frozen, "partitions": partitions, "valid_slice_counts": counts,
        "cohort_study_count": len(counts), "cohort_valid_slices": sum(counts.values()),
        "feature_identity": {"encoder": "timm.convnext_tiny.fb_in22k_ft_in1k",
                             "checkpoint_sha256": pilot["checkpoint_sha256"],
                             "runtime_versions": pilot["runtime_versions"],
                             "representation": pilot["representation"],
                             "integration_manifest_checksum": integration["manifest_checksum"]},
        "source_archive_path": str(source_archive), "source_archive_bytes": source_archive.stat().st_size,
        "source_archive_mtime_ns": source_archive.stat().st_mtime_ns,
        "pilot_plan_sha256": digest(PILOT / "plan.json"), "pilot_result_sha256": digest(result_path),
        "resources": {"partition": "cluster1-cpu", "nodes_per_job": 1, "tasks_per_job": 1,
                      "cpus_per_task": 16, "memory_gib_per_job": 64, "walltime_seconds_per_job": 10800,
                      "maximum_concurrent_jobs": 2, "partition_count": 2,
                      "maximum_launches_per_partition": 1, "maximum_allocated_node_seconds": 21600,
                      "code_watchdog_seconds_per_job": 10740, "gpu_count": 0,
                      "exclusive": False, "external_spend_usd": 0},
        "feature_bytes_uncompressed_estimate": len(counts) * (16 * 8 * 768 * 4 + 16 * 8 + 16 + 16 * 10 * 4),
        "storage_soft_cap_bytes": 2_000_000_000_000,
        "storage_basis": "Existing measured campaign storage approximately 465 GB; extraction adds under 2 GB plus source/metadata. Existing image caches are read-only.",
        "completion_deadline": "2026-10-03T23:59:59+05:30",
        "network_during_execution": False, "new_checkpoint_download": False,
        "training": False, "candidate_promotion": False, "head_training_authorized": False,
    })
    write(ROOT / "plan.json", plan)
    plan_sha = digest(ROOT / "plan.json")
    write(approval, {
        "schema_version": "rsna-ticket49-resource-approval-v1", "decision": "approved",
        "approval_status": "approved_by_project_owner",
        "approved_by": "Project owner via existing conversation-wide authorization: approved for anything everything; no matter what 1st place",
        "authorization_interpretation": "Applies existing zero-spend campaign authorization to this exact separate CPU extraction after the passing pilot. No Ticket 49-specific words claimed; no GPU expansion or Ticket 39 amendment.",
        "scope": "ticket49-convnext-cohort-extraction", "plan_path": str(ROOT / "plan.json"),
        "plan_sha256": plan_sha, "checkpoint_sha256": pilot["checkpoint_sha256"],
        "checkpoint_rights_basis": "Exact already admitted public checkpoint under Ticket 48 private experiment authorization; does not grant additional underlying dataset or final redistribution rights.",
        "resources": plan["resources"], "publication": False,
        "head_training_authorized": False, "final_candidate_selection": False,
    })
    scripts = []
    for partition in range(2):
        path = ROOT / f"extract-{partition}.sbatch"
        with path.open("x") as handle:
            handle.write(f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket49-p{partition}
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
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
{PILOT}/runtime/bin/python {snapshot}/extract.py {ROOT}/plan.json {plan_sha} {partition}
''')
        scripts.append({"partition": partition, "path": str(path), "sha256": digest(path)})
    write(ROOT / "prepared-validation.json", {
        "plan_sha256": plan_sha, "scripts": scripts,
        "source_count": len(counts), "valid_slices": sum(counts.values()),
        "partition_counts": [len(row["study_uids"]) for row in partitions],
        "partition_slice_counts": [row["valid_slices"] for row in partitions],
        "maximum_cpu_node_hours": 6, "gpu_count": 0,
        "external_spend_usd": 0, "launched": False,
    })
    print(json.dumps({"plan_sha256": plan_sha, "partition_counts": [len(row["study_uids"]) for row in partitions],
                      "partition_slice_counts": [row["valid_slices"] for row in partitions]}))


if __name__ == "__main__":
    main()
