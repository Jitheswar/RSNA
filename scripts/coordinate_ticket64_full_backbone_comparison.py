"""Advance only the frozen Ticket 64 stages; never retry a consumed launch.

This coordinator neither evaluates predictions nor selects a candidate. Every
submission still passes the original immutable plan's launch/resource guards.
"""

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "artifacts/private/ticket64-full-backbone-comparison-v1"
CHECKSUM = "557230fec672f6db9252b1895a4d174652fe217c0388a8c9598bd749183aa985"


def read(path):
    return json.loads(path.read_text())


def stage_state(directory, checksum, successful_steps, time_limit):
    receipt = directory / "accepted-launch.json"
    if not receipt.exists():
        if (directory / "launch-intent.json").exists():
            raise ValueError(f"Consumed intent has no accepted receipt: {directory}")
        return "unlaunched"
    accepted = read(receipt)
    if accepted["plan_sha256"] != checksum or accepted["launch_attempt"] != 1:
        raise ValueError("Accepted stage identity differs")
    job = str(accepted["job_id"])
    controller = directory / f"controller-{job}"
    observation_path = controller / "terminal-observation.json"
    if not observation_path.exists():
        return "active"
    observation = read(observation_path)
    fields = dict(t.split("=", 1) for t in
                  (controller / "terminal-controller.txt").read_text().split() if "=" in t)
    expected = dict(JobId=job, JobState="COMPLETED", ExitCode="0:0",
                    Partition="cluster1-cpu", NumNodes="1", NumCPUs="16", NumTasks="1",
                    MinMemoryNode="64G", TimeLimit=time_limit, OverSubscribe="OK",
                    Requeue="0", Restarts="0")
    if (observation["job_id"] != job or observation["state"] != "COMPLETED"
            or observation["returncode"] != 0
            or any(fields.get(k) != v for k, v in expected.items())
            or "gpu" in fields.get("TRES", "").lower()):
        raise ValueError(f"Stage lacks exact successful CPU terminal: {directory}")
    result = read(directory / "stage-result.json")
    if (result["status"] != "COMPLETE" or result["plan_sha256"] != checksum
            or str(result["job_id"]) != job or result["candidate_promotion"]):
        raise ValueError("Completed result identity differs")
    if successful_steps is not None and (
            result["successful_steps"] != successful_steps
            or result["progress"]["successful_optimizer_steps"] != successful_steps
            or result["progress"]["attempted_batches"] != successful_steps
            or result["intermediate_prediction"]):
        raise ValueError("Exact prespecified stage progress required")
    return "complete"


def next_stage(control, folds, occupied_slots):
    """Prefer extending an existing fold, then fill a free slot with a new fold."""
    if control != "complete" or occupied_slots >= 2:
        return None
    for fold, states in enumerate(folds):
        if "active" in states:
            continue
        for stage, state in enumerate(states):
            if state == "unlaunched":
                if any(prior != "complete" for prior in states[:stage]):
                    raise ValueError("Prior registered stages are incomplete")
                return fold, stage
    return None


def launch_environment(plan):
    return dict(os.environ, RSNA_PROJECT_ROOT=str(PROJECT),
                PYTHONPATH=str(Path(plan["source_snapshot"]) / "src"),
                PYTHONDONTWRITEBYTECODE="1")


def run(checksum):
    if checksum != CHECKSUM:
        raise ValueError("Only the exact registered comparison can advance")
    with (ROOT / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = ROOT / "plan.json"
        with plan_path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != checksum:
                raise ValueError("Frozen plan changed")
        plan = read(plan_path)
        projection = plan["projection"]
        full_limit = f'{projection["full_stage_wall_seconds"] // 3600:02}:30:00'
        # Admission is specific to this v1 plan, whose wall bounds are fixed.
        if projection["full_stage_wall_seconds"] != 19800 or projection["control_wall_seconds"] != 7200:
            raise ValueError("Unexpected registered wall bounds")
        missing_since = {}
        while True:
            control = stage_state(ROOT / "control", checksum, None, "02:00:00")
            folds = [[stage_state(ROOT / f"full-backbone/fold-{fold}/stage-{stage}", checksum,
                                  min(7500, (stage + 1) * projection["steps_per_stage"]), full_limit)
                      for stage in range(projection["full_stages_per_fold"])] for fold in range(3)]
            if control == "complete" and all(all(s == "complete" for s in f) for f in folds):
                print("All registered fits completed; independent original-MRI replay and analysis remain required.", flush=True)
                return
            rows = subprocess.check_output(["squeue", "-u", os.environ["USER"], "-h", "-o", "%i %P %t"], text=True).splitlines()
            cpu_jobs = [r.split()[0] for r in rows if r.split()[1] == "cluster1-cpu"]
            active = [(ROOT / "control", control)]
            active += [(ROOT / f"full-backbone/fold-{f}/stage-{s}", state)
                       for f, states in enumerate(folds) for s, state in enumerate(states)]
            for directory, state in active:
                if state != "active":
                    continue
                job = str(read(directory / "accepted-launch.json")["job_id"])
                if job in cpu_jobs:
                    missing_since.pop(job, None)
                elif time.monotonic() - missing_since.setdefault(job, time.monotonic()) > 300:
                    raise ValueError(f"Accepted job {job} has neither live queue nor retained terminal; no retry")
            next_item = next_stage(control, folds, len(cpu_jobs))
            snapshot = dict(at_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=checksum,
                            control=control, folds=folds, occupied_cpu_jobs=cpu_jobs,
                            next_stage=next_item, automatic_retry=False, candidate_promotion=False)
            temporary = ROOT / "coordinator-state.partial"
            temporary.write_text(json.dumps(snapshot, indent=2) + "\n")
            temporary.replace(ROOT / "coordinator-state.json")
            if next_item is not None:
                if datetime.now(timezone.utc) >= datetime.fromisoformat(projection["science_freeze_utc"]):
                    raise ValueError("Science freeze reached; no further fitting launch")
                fold, stage = next_item
                print(f"Advancing registered fold {fold}, stage {stage}", flush=True)
                env = launch_environment(plan)
                subprocess.run([str(PROJECT / ".venv/bin/python"),
                                str(Path(plan["source_snapshot"]) / "ticket64_full_backbone_comparison.py"),
                                "launch", checksum, "full", str(fold), str(stage)], env=env, check=True)
                continue
            time.sleep(30)


if __name__ == "__main__":
    run(sys.argv[1])
