"""Own the unused unchanged fitting stages and their distinct native replay."""

from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import ticket73_configured_full_continuation as fit


def next_unused_stage(states, occupied_slots):
    if occupied_slots >= 2:
        return None
    for fold, values in enumerate(states):
        if "active" in values:
            continue
        for offset, state in enumerate(values):
            if state == "unlaunched":
                fit.require(all(x == "complete" for x in values[:offset]), "New predecessor is incomplete")
                return fold, fit.START_STAGES[fold] + offset
    return None


def current_state(directory, checksum, target):
    if not (directory / "accepted-launch.json").exists():
        fit.require(not (directory / "launch-intent.json").exists(), "Consumed intent lacks accepted identity; never retry")
        return "unlaunched"
    accepted = fit.read(directory / "accepted-launch.json")
    fit.require(accepted["plan_sha256"] == checksum and accepted["launch_attempt"] == 1, "Exact sole new stage identity required")
    terminal = directory / f"controller-{accepted['job_id']}/terminal-controller.txt"
    if terminal.exists():
        fit.successful_terminal(directory, checksum, "05:30:00")
        row = fit.read(directory / "stage-result.json")
        fit.require(row["plan_sha256"] == checksum and row["job_id"] == str(accepted["job_id"])
                    and row["successful_steps"] == row["progress"]["successful_optimizer_steps"] == row["progress"]["attempted_batches"] == target
                    and row["intermediate_prediction"] is False and row["candidate_promotion"] is False,
                    "Exact successful new worker budget required")
        return "complete"
    raw = subprocess.run(["scontrol", "show", "job", str(accepted["job_id"]), "-o"], capture_output=True, text=True, timeout=15)
    fit.require(not (raw.returncode != 0 and "Invalid job id" in raw.stderr), "New accepted handle is purged without retained terminal; stop without retry")
    return "active"


def finish_replay(checksum, plan):
    import validate_ticket73_configured_full_comparison as replay
    fit.require(replay.FIT_SHA == checksum, "Native verifier must bind this exact continuation")
    if not replay.ROOT.exists():
        replay.prepare()
    request_sha = fit.digest(replay.ROOT / "plan.json")
    while True:
        if not (replay.ROOT / "accepted-launch.json").exists():
            fit.require(not (replay.ROOT / "launch-intent.json").exists(), "Consumed replay launch; never retry")
            jobs = subprocess.check_output(["squeue", "-u", "ch.jitheswar", "-h", "-o", "%i|%P"], text=True).splitlines()
            if sum(x.split("|")[1] == "cluster1-cpu" for x in jobs) < 2:
                with (fit.PROJECT / "artifacts/private/cpu-scheduler.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    fresh = subprocess.check_output(["squeue", "-u", "ch.jitheswar", "-h", "-o", "%i|%P"], text=True).splitlines()
                    if sum(x.split("|")[1] == "cluster1-cpu" for x in fresh) < 2:
                        replay.launch(request_sha)
        else:
            accepted = fit.read(replay.ROOT / "accepted-launch.json")
            terminal = replay.ROOT / f"controller-{accepted['job_id']}/terminal-controller.txt"
            if terminal.exists():
                seconds, terminal = fit.successful_terminal(replay.ROOT, request_sha, "01:30:00")
                result = fit.read(replay.ROOT / "validation.json")
                fit.require(result["status"] == "COMPLETE" and result["plan_sha256"] == request_sha
                            and result["fit_plan_sha256"] == checksum and result["job_id"] == str(accepted["job_id"])
                            and result["original_mri_studies"] == 58 and result["final_runs_verified"] == 6
                            and result["final_predictions_verified"] == 116
                            and result["original_pixels_targets_masks_exposures_verified"] is True
                            and result["all_registered_comparisons_recomputed"] is True
                            and result["maximum_prediction_error"] <= 1e-6
                            and result["historical_backend_terminals_verified"] is False
                            and result["original_strict_ancestry_passed"] is False
                            and result["original_missing_allocation_charge_seconds"] == 39600
                            and result["candidate_promotion"] is False and result["external_spend_usd"] == result["gpu_count"] == 0,
                            "Complete independent native/scientific proof with explicit recovery limits required")
                fit.write(fit.ROOT / "terminal-verification.json", dict(status="COMPLETE", passed=True, plan_sha256=checksum,
                    native_request_sha256=request_sha, verification_job_id=accepted["job_id"],
                    allocated_verification_cpu_node_seconds=seconds, native_validation_sha256=fit.digest(replay.ROOT / "validation.json"),
                    native_terminal_sha256=fit.digest(terminal), analysis_sha256=fit.digest(fit.ROOT / "analysis.json"),
                    disposition="provisional_delivery_review" if result["provisional_review_gate_passed"] else "inconclusive_no_promotion",
                    historical_backend_terminals_verified=False, original_strict_ancestry_passed=False,
                    original_missing_allocation_charge_seconds=39600, candidate_promotion=False, competition_submission=False))
                return
            raw = subprocess.run(["scontrol", "show", "job", str(accepted["job_id"]), "-o"], capture_output=True, text=True, timeout=15)
            fit.require(not (raw.returncode != 0 and "Invalid job id" in raw.stderr), "Native handle purged without terminal; never retry")
        time.sleep(30)


def run(checksum):
    plan = fit.admission(checksum)
    fit.require(fit.digest(__file__) == fit.digest(Path(plan["source_snapshot"]) / Path(__file__).name), "Exact frozen coordinator required")
    with ExitStack() as stack:
        # Hold the original scheduler locks so a restored old coordinator cannot
        # consume the stages whose future allocations this route reuses.
        for path in (fit.FIRST / "coordinator.lock", fit.PRIOR / "coordinator.lock", fit.V3 / "coordinator.lock", fit.ROOT / "coordinator.lock"):
            stream = stack.enter_context(path.open("a"))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fit.write(fit.ROOT / "coordinator-started.json", dict(plan_sha256=checksum, pid=os.getpid(), created_at_utc=datetime.now(timezone.utc).isoformat()))
        while True:
            live_ready = fit.close_live_input(checksum)
            states = [[current_state(fit.ROOT / f"full-backbone/fold-{f}/stage-{s}", checksum, min(7500, (s + 1) * 1000))
                       for s in range(fit.START_STAGES[f], 8)] for f in range(3)]
            if not live_ready:
                states[1][0] = "active"
            if all(all(x == "complete" for x in values) for values in states):
                finish_replay(checksum, plan)
                return
            jobs = subprocess.check_output(["squeue", "-u", "ch.jitheswar", "-h", "-o", "%i|%P"], text=True).splitlines()
            item = next_unused_stage(states, sum(x.split("|")[1] == "cluster1-cpu" for x in jobs))
            state = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=checksum,
                         states=states, owner_queue=jobs, next_stage=item, automatic_retry=False,
                         historical_backend_terminals_verified=False, original_strict_ancestry_passed=False)
            temporary = fit.ROOT / "coordinator-state.partial"
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(fit.ROOT / "coordinator-state.json")
            if item is not None:
                fit.launch(checksum, *item)
                continue
            time.sleep(30)


if __name__ == "__main__":
    run(sys.argv[1])
