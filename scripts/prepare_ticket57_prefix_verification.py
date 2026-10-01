"""Prepare verification only after both exact extraction jobs completed."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "artifacts/private/ticket57-resnet-prefix-cohort"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket57-prefix-verification-20261001.json"


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
    output = ROOT / "verification"
    if output.exists() or DECISION.exists():
        raise FileExistsError("Immutable prefix verification already prepared")
    plan = read(ROOT / "plan.json")
    plan_sha = digest(ROOT / "plan.json")
    auth = Path(plan["authorization_path"])
    fixed = [ROOT / "plan.json", auth]
    for partition in range(2):
        accepted_path = ROOT / f"accepted-launch-{partition}.json"
        accepted = read(accepted_path)
        job = accepted["job_id"]
        terminal_path = ROOT / f"controller-{job}/terminal-controller.txt"
        fields = dict(token.split("=", 1) for token in terminal_path.read_text().split() if "=" in token)
        if (accepted["plan_sha256"] != plan_sha or accepted["partition"] != partition
                or fields.get("JobId") != str(job) or fields.get("JobState") != "COMPLETED"
                or fields.get("ExitCode") != "0:0"):
            raise ValueError("Both matching extraction requests must complete before verification")
        result_path = ROOT / f"partition-{partition}-job-{job}/partition-result.json"
        result = read(result_path)
        if (result["status"] != "COMPLETE" or result["plan_sha256"] != plan_sha
                or result["study_count"] != len(plan["partitions"][partition]["study_uids"])):
            raise ValueError("Complete matching partition results are required")
        fixed.extend((accepted_path, terminal_path, terminal_path.with_name("terminal-observation.json"), result_path))
    output.mkdir(mode=0o700)
    source = PROJECT / "scripts/validate_ticket57_resnet_prefix_cohort.py"
    compile(source.read_text(), str(source), "exec")
    snapshot = output / "verify.py"
    shutil.copyfile(source, snapshot)
    fixed.extend((source, snapshot, Path(__file__)))
    resources = {"partition": "cluster1-cpu", "nodes": 1, "tasks": 1, "cpus_per_task": 16,
                 "memory_gib": 64, "walltime_seconds": 1800, "code_watchdog_seconds": 1680,
                 "maximum_launch_attempts": 1, "gpu_count": 0, "exclusive": False,
                 "requeue": False, "external_spend_usd": 0}
    request = {"schema_version": "rsna-ticket57-prefix-verification-request-v1",
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "extraction_root": str(ROOT),
               "extraction_plan_sha256": plan_sha, "output_root": str(output), "decision_path": str(DECISION),
               "resources": resources, "runtime_versions": plan["runtime_versions"],
               "frozen_files": [{"path": str(path), "sha256": digest(path)} for path in fixed],
               "source_snapshot": plan["source_snapshot"], "verification_scope": "All4228entries and187272slices: hashes, masks, explicit indices, source identities and actual frozen layer4 reconstruction.",
               "training": False, "candidate_promotion": False, "competition_submission": False}
    write(output / "request.json", request)
    checksum = digest(output / "request.json")
    write(DECISION, {"schema_version": "rsna-campaign-autonomous-decision-v1", "decision": "approved",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_made_by": "campaign agent under standing owner authorization",
        "authorization_sha256": digest(auth), "request_sha256": checksum, "resources": resources,
        "basis": "Independently verify the completed activation cohort and actual frozen encoder parity before admitting it to a separately specified fitting experiment.",
        "new_exact_owner_numeric_approval_claimed": False, "new_gpu_request": False,
        "gpu_ledger_cap_changed": False, "candidate_promotion": False})
    launcher = output / "run.sbatch"
    with launcher.open("x") as handle:
        handle.write(f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-prefix57-verify
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output={output}/slurm-%j.out
#SBATCH --error={output}/slurm-%j.err
#SBATCH --chdir={PROJECT}
set -euo pipefail
umask 077
export RSNA_PROJECT_ROOT={PROJECT}
export PYTHONPATH={plan['source_snapshot']}/src
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
{PROJECT}/.venv/bin/python {snapshot} {output}/request.json {checksum}
''')
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    print(json.dumps({"request_sha256": checksum, "launcher": str(launcher), "launched": False}))


if __name__ == "__main__":
    main()
