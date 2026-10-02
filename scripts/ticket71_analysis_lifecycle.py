"""Admit and run the already frozen Ticket71 primary and independent replay.

This controller cannot fit a model, choose a vote weight, upload a candidate or
submit to the competition. Both CPU components consume one intent each.
"""

from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
sys.path.insert(0, str(PROJECT / "src"))
from rsna_knee.scored_anchor_comparison import digest, read, require, input_files_digest, load_verified_inputs, successful_cpu_terminal

ROOT = PROJECT / "artifacts/private/ticket71-scored-anchor-analysis-v2"
PLAN_SHA = "25fa04a170d1683cbc47c1bbaf573a64399b6302b08b3d9a2f4f6e44db84e560"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket71-fixed-analysis-20261002.json"
PARENT = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket72-checkpoint-audit-20261002.json"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def prepare():
    require(digest(ROOT / "plan.json") == PLAN_SHA and not DECISION.exists()
            and not (ROOT / "input-resource-admission.json").exists(), "Exact frozen plan and unused final admission required")
    plan = read(ROOT / "plan.json")
    medical, anchor = Path(plan["medical_root"]), Path(plan["anchor_root"])
    output = Path(plan["anchor_output_directory"])
    paths = dict(medical_fit_accepted=medical / "fit/accepted-launch.json",
                 medical_verification_accepted=medical / "verification/accepted-launch.json",
                 medical_independent=medical / "verification/independent-validation.json",
                 medical_terminal_verification=medical / "terminal-verification.json",
                 medical_analysis=medical / "analysis.json", anchor_accepted=anchor / "accepted-launch.json",
                 anchor_terminal=anchor / "kernel-terminal.json", anchor_independent=anchor / "independent-verification.json",
                 anchor_receipt=output / "exact-scored-anchor-receipt.json",
                 anchor_predictions=output / "development-exact-scored-anchor.csv")
    for role, directory in (("medical_fit", medical / "fit"), ("medical_verification", medical / "verification")):
        job = read(directory / "accepted-launch.json")["job_id"]
        paths[role + "_terminal"] = directory / f"controller-{job}/terminal-controller.txt"
        paths[role + "_observation"] = directory / f"controller-{job}/terminal-observation.json"
    independent = read(paths["anchor_independent"])
    for role in ("original_mri_evidence", "runtime_evidence"):
        paths["anchor_" + role] = Path(independent[role]["path"])
    files = {role: dict(path=str(path), sha256=digest(path)) for role, path in paths.items()}
    admission = dict(schema_version="rsna-ticket71-analysis-input-admission-v1", plan_sha256=PLAN_SHA,
                     all_inputs_verified=True, files=files, candidate_promotion=False, competition_submission=False)
    # Verify actual source/native/terminal chains and membership before admitting
    # these bytes. This performs no AUC, bootstrap, vote or candidate selection.
    load_verified_inputs(plan, admission, PLAN_SHA)
    previous = read(PARENT)
    protected = previous["protected_prior_calendar_seconds"] + previous["cpu_node_seconds"] + previous["queue_calendar_seconds"]
    finish = datetime.now(timezone.utc) + timedelta(seconds=protected + previous["retained_review_seconds"])
    require(finish < datetime.fromisoformat(previous["science_freeze_utc"]), "Complete protected scientific/delivery/review route exceeds freeze")
    usage = subprocess.check_output(["du", "-sb", str(PROJECT), "/home/ch.jitheswar/models/qwen3-8b-b968826d9c46dd6066d109eabc6255188de91218"], text=True)
    existing = sum(int(r.split()[0]) for r in usage.splitlines())
    protected_bytes = previous["storage"]["protected_allowance_bytes"] + previous["storage"]["additional_allowance_bytes"]
    require(existing + protected_bytes + 100_000_000 < 2_000_000_000_000
            and shutil.disk_usage(PROJECT).free > protected_bytes + 100_000_000, "Full protected storage plus analysis output exceeds bound")
    snapshot = ROOT / "lifecycle-source-snapshot"
    snapshot.mkdir(mode=0o700)
    shutil.copyfile(__file__, snapshot / Path(__file__).name)
    write(DECISION, dict(decision="approved", plan_sha256=PLAN_SHA, input_files_sha256=input_files_digest(files),
              authorization_sha256=digest(AUTH), external_spend_usd=0, gpu_seconds=0, tpu_seconds=0,
              decision_made_by="campaign agent under standing owner authorization", new_exact_owner_numeric_approval_claimed=False,
              maximum_launch_attempts=1, automatic_retry=False,
              components={name: dict(time_limit="01:00:00", watchdog_seconds=3540, maximum_launch_attempts=1)
                          for name in ("analysis", "verification")},
              previously_retained_cpu_node_seconds_reused=7200,
              reserve_source=str(PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket70-exact-anchor-20261002.json"),
              parent_resource_decision_sha256=digest(PARENT), complete_route_finish_utc=finish.isoformat(),
              science_freeze_utc=previous["science_freeze_utc"],
              storage=dict(existing_apparent_bytes=existing, protected_allowance_bytes=protected_bytes,
                           additional_allowance_bytes=100_000_000, projected_apparent_bytes=existing + protected_bytes + 100_000_000),
              lifecycle_source=dict(path=str(snapshot / Path(__file__).name), sha256=digest(__file__)),
              protocol_retuned=False, candidate_promotion=False, competition_submission=False))
    admission["resource_decision"] = dict(path=str(DECISION), sha256=digest(DECISION))
    write(ROOT / "input-resource-admission.json", admission)
    rows = []
    for kind in ("analysis", "verification"):
        directory = ROOT / kind
        directory.mkdir(mode=0o700)
        launcher = directory / "run.sbatch"
        source = Path(plan["source_snapshot"])
        launcher.write_text(f"""#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket71-{kind}
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output={directory}/slurm-%j.out
#SBATCH --error={directory}/slurm-%j.err
#SBATCH --chdir={PROJECT}
set -euo pipefail
umask 077
export RSNA_PROJECT_ROOT={PROJECT}
export PYTHONPATH={source}/src
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
{PROJECT}/.venv/bin/python {source}/scripts/ticket71_scored_anchor_analysis.py {ROOT}/plan.json {PLAN_SHA} {ROOT}/input-resource-admission.json {digest(ROOT / 'input-resource-admission.json')} --kind {kind}
""")
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        rows.append(dict(kind=kind, path=str(launcher), sha256=digest(launcher)))
    write(ROOT / "lifecycle-admission.json", dict(plan_sha256=PLAN_SHA, decision_sha256=digest(DECISION),
         input_admission_sha256=digest(ROOT / "input-resource-admission.json"), launchers=rows))
    print(json.dumps(dict(status="INPUT_RESOURCE_ADMITTED", plan_sha256=PLAN_SHA, complete_route_finish_utc=finish.isoformat())))


def admission():
    plan, frozen, decision = read(ROOT / "plan.json"), read(ROOT / "lifecycle-admission.json"), read(DECISION)
    require(digest(ROOT / "plan.json") == PLAN_SHA == frozen["plan_sha256"] == decision["plan_sha256"]
            and decision["decision"] == "approved" and digest(DECISION) == frozen["decision_sha256"]
            and digest(ROOT / "input-resource-admission.json") == frozen["input_admission_sha256"]
            and digest(AUTH) == decision["authorization_sha256"], "Immutable exact input/resource admission required")
    source = decision["lifecycle_source"]
    require(digest(source["path"]) == source["sha256"] == digest(__file__), "Exact admitted lifecycle source required")
    for row in plan["frozen_files"]:
        require(digest(row["path"]) == row["sha256"], "Original frozen analysis source changed")
    return plan, frozen


def terminal(kind):
    directory = ROOT / kind
    accepted = read(directory / "accepted-launch.json")
    require(accepted["plan_sha256"] == PLAN_SHA and accepted["launch_attempt"] == 1
            and accepted["admission_sha256"] == digest(ROOT / "input-resource-admission.json"), "Exact accepted component required")
    path = directory / f"controller-{accepted['job_id']}/terminal-controller.txt"
    observation = path.with_name("terminal-observation.json")
    bindings = dict(component_terminal=dict(path=str(path), sha256=digest(path)),
                    component_observation=dict(path=str(observation), sha256=digest(observation)))
    seconds, _ = successful_cpu_terminal(directory, accepted["job_id"], "01:00:00", bindings, "component")
    result_path = directory / ("fixed-paired-analysis.json" if kind == "analysis" else "independent-verification.json")
    result = read(result_path)
    require(result["status"] == "COMPLETE" and result["plan_sha256"] == PLAN_SHA
            and str(result["job_id"]) == str(accepted["job_id"])
            and result["input_admission_sha256"] == digest(ROOT / "input-resource-admission.json")
            and result["resource_decision_sha256"] == digest(DECISION), "Exact successful component output identity required")
    return result, seconds, path


def launch(kind):
    plan, frozen = admission()
    require(kind in ("analysis", "verification"), "Only the registered primary and independent replay may launch")
    directory = ROOT / kind
    require(not (directory / "launch-intent.json").exists(), "Consumed component launch; never retry")
    if kind == "verification":
        terminal("analysis")
    row = next(r for r in frozen["launchers"] if r["kind"] == kind)
    require(digest(row["path"]) == row["sha256"], "Exact frozen CPU launcher required")
    with (PROJECT / "artifacts/private/cpu-scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        jobs = subprocess.check_output(["squeue", "-u", "ch.jitheswar", "-h", "-o", "%i|%P"], text=True).splitlines()
        if sum(r.split("|")[1] == "cluster1-cpu" for r in jobs) >= 2:
            print(json.dumps(dict(status="WAIT_FOR_CPU_SLOT", kind=kind)), flush=True)
            return False
        write(directory / "launch-intent.json", dict(plan_sha256=PLAN_SHA, admission_sha256=frozen["input_admission_sha256"], launch_attempt=1))
        job = int(subprocess.check_output(["sbatch", "--parsable", row["path"]], text=True).strip().split(";")[0])
        write(directory / "accepted-launch.json", dict(job_id=job, plan_sha256=PLAN_SHA,
              admission_sha256=frozen["input_admission_sha256"], launch_attempt=1))
    subprocess.run(["systemd-run", "--user", f"--unit=rsna-ticket71-{kind}-{job}-observer", "--collect",
                    str(PROJECT / ".venv/bin/python"), str(PROJECT / "scripts/monitor_ticket48_cpu_pilot.py"), str(directory), str(job)], check=True)
    print(json.dumps(dict(status="ACCEPTED", kind=kind, job_id=job)), flush=True)


def watch():
    _, frozen = admission()
    for kind in ("analysis", "verification"):
        directory = ROOT / kind
        while True:
            if not (directory / "accepted-launch.json").exists():
                require(not (directory / "launch-intent.json").exists(), "Consumed component has no receipt; never retry")
                jobs = subprocess.check_output(["squeue", "-u", "ch.jitheswar", "-h", "-o", "%i|%P"], text=True).splitlines()
                if sum(r.split("|")[1] == "cluster1-cpu" for r in jobs) < 2:
                    launch(kind)
            else:
                accepted = read(directory / "accepted-launch.json")
                path = directory / f"controller-{accepted['job_id']}/terminal-controller.txt"
                if path.exists():
                    terminal(kind)
                    break
                raw = subprocess.run(["scontrol", "show", "job", str(accepted["job_id"]), "-o"], capture_output=True, text=True, timeout=15)
                require(not (raw.returncode != 0 and "Invalid job id" in raw.stderr), "Accepted component purged without terminal; never retry")
            time.sleep(15)
    result, seconds, path = terminal("verification")
    require(result["all_fixed_statistics_reproduced"] is True and result["full_input_native_source_terminal_chains_reverified"] is True
            and result["candidate_promotion"] is False and result["competition_submission"] is False, "Independent fixed replay required")
    write(ROOT / "terminal-verification.json", dict(status="COMPLETE", passed=True, plan_sha256=PLAN_SHA,
        input_admission_sha256=frozen["input_admission_sha256"], independent_validation_sha256=digest(ROOT / "verification/independent-verification.json"),
        verification_terminal_sha256=digest(path), allocated_verification_cpu_node_seconds=seconds,
        disposition=result["disposition"], gates=result["gates"], candidate_promotion=False, competition_submission=False))
    print(json.dumps(dict(status="COMPLETE", disposition=result["disposition"], gates=result["gates"])), flush=True)


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "launch":
        launch(sys.argv[2])
    else:
        {"prepare": prepare, "watch": watch}[command]()
