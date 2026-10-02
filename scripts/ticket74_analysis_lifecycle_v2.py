"""Account joint pilot reserves while running the unchanged frozen Ticket74 science.

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
from rsna_knee.full_backbone_anchor_comparison import digest, read, require, input_files_digest, load_verified_inputs, successful_cpu_terminal

ROOT = PROJECT / "artifacts/private/ticket74-full-backbone-anchor-analysis-v1"
PLAN_SHA = "187599d49ac25b90fd9fdd5c8dcf830c115481cbb59916771c94bc53edaa6470"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket74-fixed-analysis-20261002.json"
PARENT = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket73-configured-continuation-20261002.json"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def remaining_reserves(previous, *, complete_inputs_verified):
    require(complete_inputs_verified is True, "Complete actual native inputs required before releasing future reserves")
    # All three full fits and the original native replay were just verified as
    # completed. Only their future calendar reservations become zero; historical
    # charges and every remaining delivery/final-fit/review reserve stay intact.
    remaining_calendar = dict(previous["remaining_calendar"])
    remaining_calendar["remaining_fit_critical_path_seconds"] = 0
    remaining_calendar["independent_native_replay_seconds"] = 0
    remaining_calendar["new_fixed_analysis_and_independent_replay_seconds"] = 7200
    projection = previous["projection"]
    per_fit = (projection["history_count_upper_bound_per_fit"] + 4) * projection["maximum_checkpoint_bytes"]
    require(projection["checkpoint_storage_allowance_bytes"] == 7 * per_fit,
            "Original six experimental plus conditional final-fit storage envelope required")
    completed_storage_reservations = 6 * per_fit
    # Actual du retains every completed checkpoint and old copy. They no longer
    # require a second future allowance. Keep the seventh conditional-fit bound.
    protected_bytes = previous["storage"]["protected_allowance_bytes"] + previous["storage"]["additional_allowance_bytes"] - completed_storage_reservations
    require(protected_bytes >= per_fit, "Full conditional final-fit storage must remain protected")
    return remaining_calendar, protected_bytes, completed_storage_reservations


PILOT_ROUTE = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket75-joint-pilot-route-20261003.json"


def pilot_reserves():
    """Never release another route merely because its work may have finished."""
    if not PILOT_ROUTE.exists():
        return None
    row = read(PILOT_ROUTE)
    require(row["status"] == "ADMITTED" and row["fit_plan_sha256"] == read(ROOT / "plan.json")["fit_plan"]["sha256"]
            and row["T74_future_analysis_cpu_node_seconds"] == 7200
            and row["pilot_complete_calendar_seconds"] == 28800
            and row["pilot_retained_assets_bytes"] == 8_000_000_000
            and row["T74_future_admission_accounts_outstanding_pilot"] is True
            and row["T74_lifecycle_source_sha256"] == digest(__file__),
            "Exact joint route and matching lifecycle ownership required")
    return dict(path=str(PILOT_ROUTE), sha256=digest(PILOT_ROUTE),
                calendar_seconds=28800, storage_bytes=8_000_000_000)


def prepare():
    # The route creator uses this same lock and cannot admit a new pilot after
    # analysis resource admission. This prevents a check/create race.
    with (PROJECT / "artifacts/private/campaign-resource-admission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _prepare_joint_locked()


def _prepare_joint_locked():
    require(digest(ROOT / "plan.json") == PLAN_SHA and not DECISION.exists()
            and not (ROOT / "input-resource-admission.json").exists(), "Exact frozen plan and unused final admission required")
    plan = read(ROOT / "plan.json")
    fit, anchor = Path(plan["fit_root"]), Path(plan["anchor_root"])
    output = Path(plan["anchor_output_directory"])
    native = fit / "independent-original-mri-replay-v1"
    paths = dict(fit_native_request=native / "plan.json", fit_native_accepted=native / "accepted-launch.json",
        fit_native_validation=native / "validation.json", fit_terminal_verification=fit / "terminal-verification.json",
        fit_analysis=fit / "analysis.json", anchor_accepted=anchor / "accepted-launch.json",
        anchor_terminal=anchor / "kernel-terminal.json", anchor_independent=anchor / "independent-verification.json",
        anchor_receipt=output / "exact-scored-anchor-receipt.json", anchor_predictions=output / "development-exact-scored-anchor.csv")
    job = read(native / "accepted-launch.json")["job_id"]
    paths["fit_native_terminal"] = native / f"controller-{job}/terminal-controller.txt"
    paths["fit_native_observation"] = native / f"controller-{job}/terminal-observation.json"
    prior = fit.with_name("ticket64-full-backbone-comparison-v2")
    for fold in range(3):
        for arm in ("head-continuation", "full-backbone"):
            base = prior if arm == "head-continuation" else fit
            paths[f"fit_{arm}_{fold}"] = base / arm / f"fold-{fold}" / ("run.json" if arm == "head-continuation" else "stage-7/stage-result.json")
    independent = read(paths["anchor_independent"])
    for role in ("original_mri_evidence", "runtime_evidence"):
        paths["anchor_" + role] = Path(independent[role]["path"])
    files = {role: dict(path=str(path), sha256=digest(path)) for role, path in paths.items()}
    admission = dict(schema_version="rsna-ticket74-analysis-input-admission-v1", plan_sha256=PLAN_SHA,
                     all_inputs_verified=True, files=files, candidate_promotion=False, competition_submission=False)
    # Verify actual source/native/terminal chains and membership before admitting
    # these bytes. This performs no AUC, bootstrap, vote or candidate selection.
    load_verified_inputs(plan, admission, PLAN_SHA)
    previous = read(PARENT)
    require(previous["authorization_sha256"] == digest(AUTH), "Original standing authorization required")
    remaining_calendar, protected_bytes, completed_storage_reservations = remaining_reserves(previous, complete_inputs_verified=True)
    joint_pilot = pilot_reserves()
    if joint_pilot is not None:
        remaining_calendar["additional_joint_pilot_route_seconds"] = joint_pilot["calendar_seconds"]
        protected_bytes += joint_pilot["storage_bytes"]
    finish = datetime.now(timezone.utc) + timedelta(seconds=sum(remaining_calendar.values()))
    require(finish < datetime.fromisoformat(previous["science_freeze_utc"]), "Entire remaining final-fit/delivery/review route exceeds freeze")
    usage = subprocess.check_output(["du", "-sb", str(PROJECT), "/home/ch.jitheswar/models/qwen3-8b-b968826d9c46dd6066d109eabc6255188de91218"], text=True)
    existing = sum(int(r.split()[0]) for r in usage.splitlines())
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
              additional_fixed_analysis_cpu_node_seconds=7200, previously_retained_cpu_node_seconds_reused=0,
              remaining_calendar=remaining_calendar, completed_future_storage_reservations_bytes=completed_storage_reservations,
              conditional_final_fit_authorized=False, gpu_delivery_authorized=False,
              parent_resource_decision_sha256=digest(PARENT), complete_route_finish_utc=finish.isoformat(),
              science_freeze_utc=previous["science_freeze_utc"],
              storage=dict(existing_apparent_bytes=existing, protected_allowance_bytes=protected_bytes,
                           additional_allowance_bytes=100_000_000, projected_apparent_bytes=existing + protected_bytes + 100_000_000),
              lifecycle_source=dict(path=str(snapshot / Path(__file__).name), sha256=digest(__file__)),
              joint_pilot_reservation=joint_pilot,
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
#SBATCH --job-name=rsna-ticket74-{kind}
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
{PROJECT}/.venv/bin/python {source}/scripts/ticket74_full_backbone_anchor_analysis.py {ROOT}/plan.json {PLAN_SHA} {ROOT}/input-resource-admission.json {digest(ROOT / 'input-resource-admission.json')} --kind {kind}
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
    joint = decision["joint_pilot_reservation"]
    require(joint == pilot_reserves(), "Joint pilot reservation changed after final admission")
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
    subprocess.run(["systemd-run", "--user", f"--unit=rsna-ticket74-{kind}-{job}-observer", "--collect",
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


def await_inputs():
    """Wait for the live fitting controller; once ready, admit and run once."""
    require(digest(ROOT / "plan.json") == PLAN_SHA, "Exact prospective preparation required")
    with (ROOT / "await-inputs.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(ROOT / "await-inputs-v2-started.json", dict(plan_sha256=PLAN_SHA, pid=os.getpid(),
            source_sha256=digest(__file__), competition_submission=False))
        fit = Path(read(ROOT / "plan.json")["fit_root"])
        while not (fit / "terminal-verification.json").exists():
            try:
                raw = subprocess.check_output(["systemctl", "--user", "show", "rsna-ticket73-configured-continuation-v4.service",
                    "--property=ActiveState", "--property=MainPID"], text=True, timeout=15)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                # An observation failure is not proof of a stopped owner.
                time.sleep(30)
                continue
            fields = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
            require(fields.get("ActiveState") == "active" and int(fields.get("MainPID", "0")) > 0,
                "Fitting controller stopped before final native proof; preserve attempts and do not retry")
            # A PID must identify a live process, not just a retained unit field.
            os.kill(int(fields["MainPID"]), 0)
            time.sleep(30)
        prepare()
        watch()


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "launch":
        launch(sys.argv[2])
    else:
        {"prepare": prepare, "watch": watch, "await-inputs": await_inputs}[command]()
