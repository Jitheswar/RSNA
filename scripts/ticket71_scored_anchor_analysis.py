"""Execute the immutable fixed Ticket71 analysis only after full admission."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import platform
import signal
import subprocess
import sys
import time

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))
from rsna_knee import scored_anchor_comparison
from rsna_knee.scored_anchor_comparison import analyze, digest, input_files_digest, load_verified_inputs, read, require, successful_cpu_terminal


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def check_allocation(resource_admission):
    import torch
    job = os.environ["SLURM_JOB_ID"]
    actual = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True, timeout=15)
    fields = dict(token.split("=", 1) for token in actual.split() if "=" in token)
    expected = dict(JobId=job, JobState="RUNNING", Partition="cluster1-cpu", NumNodes="1", NumCPUs="16",
                    NumTasks="1", MinMemoryNode="64G", TimeLimit=resource_admission["time_limit"],
                    OverSubscribe="OK", Requeue="0", Restarts="0")
    hours, minutes, seconds = map(int, resource_admission["time_limit"].split(":"))
    total = hours * 3600 + minutes * 60 + seconds
    require(60 < total <= 21600 and type(resource_admission["watchdog_seconds"]) is int
            and 0 < resource_admission["watchdog_seconds"] <= total - 60,
            "Admitted CPU wall/watchdog bounds required")
    require(all(fields.get(key) == value for key, value in expected.items())
            and "gpu" not in fields.get("TRES", "").lower() and torch.version.cuda is None
            and not torch.cuda.is_available(), "Exact admitted CPU-only allocation required")
    return actual


def verified_primary(root, decision, plan_sha256, admission_sha256, decision_sha256):
    """Establish the primary's exact successful terminal before replay scoring."""
    accepted = read(root / "analysis/accepted-launch.json")
    require(accepted["plan_sha256"] == plan_sha256 and accepted["admission_sha256"] == admission_sha256
            and accepted["launch_attempt"] == 1, "Exact sole primary analysis must precede verification")
    job = accepted["job_id"]
    terminal_path = root / "analysis" / f"controller-{job}/terminal-controller.txt"
    observation_path = terminal_path.parent / "terminal-observation.json"
    bindings = dict(primary_terminal=dict(path=str(terminal_path), sha256=digest(terminal_path)),
                    primary_observation=dict(path=str(observation_path), sha256=digest(observation_path)))
    allocated_seconds, terminal_sha = successful_cpu_terminal(root / "analysis", job,
                           decision["components"]["analysis"]["time_limit"], bindings, "primary")
    analysis_path = root / "analysis/fixed-paired-analysis.json"
    actual = read(analysis_path)
    require(actual["status"] == "COMPLETE" and actual["plan_sha256"] == plan_sha256
            and actual["input_admission_sha256"] == admission_sha256
            and actual["resource_decision_sha256"] == decision_sha256 and str(actual["job_id"]) == str(job)
            and actual["gpu_seconds"] == actual["tpu_seconds"] == actual["external_spend_usd"] == 0
            and 0 < actual["elapsed_seconds"] <= decision["components"]["analysis"]["watchdog_seconds"],
            "Exact successful primary analysis identity/runtime required")
    audit = dict(analysis_sha256=digest(analysis_path), analysis_accepted_launch_sha256=digest(root / "analysis/accepted-launch.json"),
                 analysis_terminal_sha256=terminal_sha, analysis_observation_sha256=digest(observation_path),
                 analysis_allocated_cpu_node_seconds=allocated_seconds, analysis_job_id=job)
    return actual, audit


def run(plan_path, plan_sha256, admission_path, admission_sha256, kind="analysis"):
    started = time.monotonic()
    require(digest(plan_path) == plan_sha256 and digest(admission_path) == admission_sha256,
            "Exact immutable analysis plan and final admission required")
    plan, admission = read(plan_path), read(admission_path)
    require(kind in ("analysis", "verification"), "Only the primary analysis and its independent verification are admitted")
    actual_runtime = dict(python=platform.python_version(), numpy=importlib.metadata.version("numpy"),
                          torch=importlib.metadata.version("torch"))
    require(actual_runtime == plan["analysis_runtime"], "Exact prepared CPU analysis runtime required")
    require(any(Path(row["path"]).resolve() == Path(__file__).resolve() for row in plan["frozen_files"])
            and any(Path(row["path"]).resolve() == Path(scored_anchor_comparison.__file__).resolve()
                    for row in plan["frozen_files"]), "Executing analysis script and module must belong to the frozen source inventory")
    root = Path(plan["run_root"])
    require(Path(plan_path).resolve() == (root / "plan.json").resolve()
            and Path(admission_path).resolve() == (root / "input-resource-admission.json").resolve(),
            "Exact plan and admission role paths required")
    decision_path = Path(admission["resource_decision"]["path"])
    require(digest(decision_path) == admission["resource_decision"]["sha256"], "Exact resource decision bytes required")
    decision = read(decision_path)
    require(decision["decision"] == "approved" and decision["plan_sha256"] == plan_sha256
            and decision["input_files_sha256"] == input_files_digest(admission["files"])
            and decision["external_spend_usd"] == 0 and decision["gpu_seconds"] == decision["tpu_seconds"] == 0
            and decision["maximum_launch_attempts"] == 1 and not decision["automatic_retry"],
            "Complete exact CPU-only resource admission required")
    component = decision["components"][kind]
    require(component["maximum_launch_attempts"] == 1, "Exact sole component attempt required")
    accepted = read(root / kind / "accepted-launch.json")
    require(accepted["plan_sha256"] == plan_sha256 and accepted["admission_sha256"] == admission_sha256
            and accepted["launch_attempt"] == 1 and str(accepted["job_id"]) == os.environ["SLURM_JOB_ID"],
            "Exact sole admitted analysis launch required")
    allocation = check_allocation(component)
    write(root / kind / "running-allocation.json", dict(controller=allocation, job_id=os.environ["SLURM_JOB_ID"]))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Fixed paired analysis watchdog")))
    signal.alarm(component["watchdog_seconds"])
    primary, primary_audit = verified_primary(root, decision, plan_sha256, admission_sha256,
                           digest(decision_path)) if kind == "verification" else (None, None)
    manifest, anchor, report, reference, zero, image_signal = load_verified_inputs(plan, admission, plan_sha256)
    result = analyze(anchor, report, reference, zero, manifest, expected_report_image=image_signal)
    if kind == "verification":
        envelope_keys = {"status", "plan_sha256", "input_admission_sha256", "resource_decision_sha256", "job_id",
                         "completed_at_utc", "elapsed_seconds", "peak_rss_bytes", "gpu_seconds", "tpu_seconds", "external_spend_usd"}
        require(set(primary) == set(result) | envelope_keys and all(primary[key] == value for key, value in result.items()),
                "Every fixed paired metric, support, vote, diagnostic and review gate must independently reproduce")
        result = dict(status="COMPLETE", all_fixed_statistics_reproduced=True,
                      **primary_audit,
                      full_input_native_source_terminal_chains_reverified=True,
                      disposition=result["disposition"], gates=result["gates"], candidate_promotion=False,
                      competition_submission=False, offline_delivery_verified=False)
    result.update(status="COMPLETE", plan_sha256=plan_sha256, input_admission_sha256=admission_sha256,
                  resource_decision_sha256=digest(decision_path), job_id=os.environ["SLURM_JOB_ID"],
                  completed_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  gpu_seconds=0, tpu_seconds=0, external_spend_usd=0)
    write(root / kind / ("fixed-paired-analysis.json" if kind == "analysis" else "independent-verification.json"), result)
    signal.alarm(0)
    print(json.dumps(dict(status="COMPLETE", disposition=result["disposition"], elapsed_seconds=result["elapsed_seconds"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan_path", type=Path)
    parser.add_argument("plan_sha256")
    parser.add_argument("admission_path", type=Path)
    parser.add_argument("admission_sha256")
    parser.add_argument("--kind", choices=("analysis", "verification"), default="analysis")
    args = parser.parse_args()
    run(args.plan_path, args.plan_sha256, args.admission_path, args.admission_sha256, args.kind)
