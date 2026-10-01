"""Freeze two disjoint full-source MRI input probes after the measured pilot."""

import ast
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
PARENT = PROJECT / "artifacts/private/public-frontier-adoption/broader-cpu-input-probe-v1"
ROOT = PROJECT / "artifacts/private/public-frontier-adoption/full-cpu-input-probe-v2"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket59-full-cpu-input-probe-v2-20261001.json"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main():
    if ROOT.exists() or DECISION.exists():
        raise FileExistsError("Immutable full-source input probe already prepared")
    parent = read(PARENT / "plan.json")
    result = read(PARENT / "probe-result.json")
    validation = read(PARENT / "independent-summary-check.json")
    accepted = read(PARENT / "accepted-launch.json")
    terminal = PARENT / f"controller-{accepted['job_id']}/terminal-controller.txt"
    fields = dict(token.split("=", 1) for token in terminal.read_text().split() if "=" in token)
    if (result["status"] != "COMPLETE" or not result["cpu_input_guard_passed"]
            or result["study_count"] != 256 or result["job_id"] != str(accepted["job_id"])
            or fields.get("JobId") != result["job_id"] or fields.get("JobState") != "COMPLETED"
            or fields.get("ExitCode") != "0:0" or validation["status"] != "COMPLETE"
            or validation["result_sha256"] != digest(PARENT / "probe-result.json")
            or validation["allocated_cpu_node_seconds"] != 264 or result["model_inference"]
            or result["gpu_count"] != 0 or result["hidden_error_reproduced"]):
        raise ValueError("Successful exact measured CPU input pilot is required")
    for row in parent["frozen_files"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("Consumed input pilot source changed")
    storage_observed_at = datetime.now(timezone.utc).isoformat()
    storage_output = subprocess.check_output(["du", "-s", "-B1", str(PROJECT)], text=True)
    storage_bytes = int(storage_output.split()[0])
    # Bound both raw-data partitions and both temporary pixel banks prospectively.
    temporary_bound = 2 * (384 + 32) * 1024 ** 3
    if not 0 < storage_bytes < 2 * 1024 ** 4 - temporary_bound:
        raise ValueError("Actual project storage cannot fit the full-input ceiling")
    with zipfile.ZipFile(parent["archive_path"]) as archive:
        metadata_bytes = archive.read("train_series.csv")
    metadata = list(csv.DictReader(io.StringIO(metadata_bytes.decode("utf-8-sig"))))
    uids = sorted({row["StudyInstanceUID"] for row in metadata})
    if not 4000 <= len(uids) <= 5000 or not set(parent["selected_study_uids"]) <= set(uids):
        raise ValueError("Complete original training MRI membership differs")
    partitions = [uids[::2], uids[1::2]]
    ROOT.mkdir(mode=0o700)
    snapshot = ROOT / "source-snapshot"
    shutil.copytree(Path(parent["source_snapshot"]) / "upstream-cells", snapshot / "upstream-cells")
    shutil.copyfile(Path(parent["source_snapshot"]) / "public_frontier_validation.py", snapshot / "public_frontier_validation.py")
    shutil.copyfile(__file__, snapshot / Path(__file__).name)
    (ROOT / "original-train-series.csv").write_bytes(metadata_bytes)
    source_path = Path(parent["source_snapshot"]) / "ticket59_public_input_probe.py"
    source = source_path.read_text()
    nodes = ast.parse(source).body
    keep = [node for node in nodes if isinstance(node, (ast.Import, ast.ImportFrom))
        or isinstance(node, ast.FunctionDef) and node.name in {"read", "digest", "write", "require", "fragment_def", "run"}]
    worker = "\n\n".join(ast.get_source_segment(source, node) for node in keep)
    replacements = {'"MinMemoryNode": "32G"': '"MinMemoryNode": "64G"',
        '"TimeLimit": "01:00:00"': '"TimeLimit": "03:00:00"',
        'pixels.shape == (256, 6, 12, 336, 336)': 'pixels.shape == (len(ids), 6, 12, 336, 336)'}
    for old, new in replacements.items():
        if worker.count(old) != 1:
            raise ValueError("Measured worker resource/cohort substitution is ambiguous")
        worker = worker.replace(old, new)
    worker += '\n\nif __name__ == "__main__":\n    run(Path(sys.argv[1]), sys.argv[2])\n'
    compile(worker, "full-cpu-input-worker", "exec")
    (snapshot / "run_full_partition.py").write_text(worker)
    resources = {"partition": "cluster1-cpu", "nodes": 1, "tasks": 1, "cpus_per_task": 16,
        "memory_gib": 64, "walltime_seconds": 10800, "code_watchdog_seconds": 10560,
        "maximum_launch_attempts_per_partition": 1, "maximum_concurrent_jobs": 2,
        "partition_count": 2, "gpu_count": 0, "exclusive": False, "requeue": False,
        "external_spend_usd": 0, "maximum_raw_mri_bytes_per_partition": 384 * 1024 ** 3,
        "maximum_pixel_cache_bytes_per_partition": 32 * 1024 ** 3}
    common_inputs = [AUTH, PARENT / "plan.json", PARENT / "probe-result.json", PARENT / "independent-summary-check.json",
        terminal, terminal.with_name("terminal-observation.json"), source_path, ROOT / "original-train-series.csv"]
    common_inputs.extend(Path(row["path"]) for row in parent["frozen_files"])
    common_inputs.extend(path for path in snapshot.rglob("*") if path.is_file())
    common_inputs = sorted(set(common_inputs))
    plans = []
    for partition, members in enumerate(partitions):
        output = ROOT / f"partition-{partition}"
        output.mkdir(mode=0o700)
        estimate = 2 * validation["allocated_cpu_node_seconds"] / 256 * len(members) + 900
        if estimate >= resources["code_watchdog_seconds"]:
            raise ValueError("Measured conservative full-partition projection exceeds allocation")
        narrow = {**parent["resources"], "memory_gib": 64, "walltime_seconds": 10800,
            "code_watchdog_seconds": 10560, "maximum_raw_mri_bytes": 384 * 1024 ** 3,
            "maximum_pixel_cache_bytes": 32 * 1024 ** 3}
        plan = {**parent, "schema_version": "rsna-ticket59-full-input-partition-plan-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_root": str(output),
            "source_snapshot": str(snapshot), "decision_path": str(output / "partition-decision.json"), "resources": narrow,
            "partition": partition, "selected_study_uids": members, "study_count": len(members),
            "selection": "All original training-series metadata study UIDs, sorted and split alternating into two disjoint partitions; label values not read.",
            "expected_cache_bytes": len(members) * 6 * 12 * 336 * 336,
            "prospective_partition_cost_seconds_with_allowance": estimate,
            "measured_project_bytes_before_request": storage_bytes,
            "storage_measurement": {"observed_at_utc": storage_observed_at, "command": ["du", "-s", "-B1", str(PROJECT)], "output": storage_output},
            "frozen_files": [{"path": str(path), "sha256": digest(path)} for path in common_inputs]}
        plan.pop("studies", None)
        write(output / "plan.json", plan)
        plans.append({"partition": partition, "root": str(output), "study_count": len(members),
            "plan_sha256": digest(output / "plan.json")})
    campaign = {"schema_version": "rsna-ticket59-full-input-cohort-plan-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(ROOT), "parent_plan_sha256": digest(PARENT / "plan.json"),
        "resources": resources, "study_count": len(uids), "partitions": plans,
        "original_series_metadata_sha256": digest(ROOT / "original-train-series.csv"),
        "worker_sha256": digest(snapshot / "run_full_partition.py"),
        "worker_changes": replacements, "scientific_pixel_rules_unchanged": True,
        "measured_project_bytes": storage_bytes, "temporary_disk_bytes_bound": temporary_bound,
        "storage_measurement": {"observed_at_utc": storage_observed_at, "command": ["du", "-s", "-B1", str(PROJECT)], "output": storage_output},
        "labels_read": False, "model_inference": False, "candidate_promotion": False}
    write(ROOT / "plan.json", campaign)
    write(DECISION, {"schema_version": "rsna-campaign-autonomous-decision-v1", "decision": "approved",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": digest(ROOT / "plan.json"),
        "partition_plan_sha256s": {str(row["partition"]): row["plan_sha256"] for row in plans},
        "partition_resources": {str(row["partition"]): read(Path(row["root"]) / "plan.json")["resources"] for row in plans},
        "resources": resources, "authorization_sha256": digest(AUTH),
        "decision_made_by": "campaign agent under standing owner authorization",
        "basis": "Search the complete original training MRI input corpus for rare accessible preprocessing failures after the measured 256-study 4.4-minute probe passed. Two disjoint unchanged-preprocessing CPU partitions fit measured 2x cost plus 900s allowance and aggregate storage ceiling. No models or GPU requests.",
        "new_exact_owner_numeric_approval_claimed": False, "new_gpu_request": False,
        "gpu_ledger_cap_changed": False, "candidate_promotion": False})
    for row in plans:
        output = Path(row["root"])
        # The measured worker checks one plan/resource pair; bind a separate
        # immutable decision to each exact partition rather than weakening it.
        decision = {"schema_version": "rsna-campaign-autonomous-decision-v1", "decision": "approved",
            "plan_sha256": row["plan_sha256"], "resources": read(output / "plan.json")["resources"],
            "campaign_decision_sha256": digest(DECISION), "authorization_sha256": digest(AUTH)}
        write(output / "partition-decision.json", decision)
        launcher = f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket59-full-input-{row['partition']}
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output={output}/slurm-%j.out
#SBATCH --error={output}/slurm-%j.err
#SBATCH --chdir={PROJECT}
set -euo pipefail
umask 077
export RSNA_PROJECT_ROOT={PROJECT}
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
{PROJECT}/.venv/bin/python {snapshot}/run_full_partition.py {output}/plan.json {row['plan_sha256']}
'''
        (output / "run.sbatch").write_text(launcher)
        subprocess.run(["bash", "-n", str(output / "run.sbatch")], check=True)
    print(json.dumps({"plan_sha256": digest(ROOT / "plan.json"), "study_count": len(uids), "partitions": plans, "launched": False}))


if __name__ == "__main__":
    main()
