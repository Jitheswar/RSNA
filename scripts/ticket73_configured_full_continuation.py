"""Resume unchanged Ticket64 science through an explicit checkpoint recovery.

Missing historical terminals remain unverified. This route never repeats their
updates or treats its checkpoint audit as a replacement backend success record.
"""

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from ticket72_checkpoint_recovery_audit import read, write, digest, require, admission as audit_admission

PROJECT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
ROOT = PROJECT / "artifacts/private/ticket64-full-backbone-comparison-v4"
PRIOR = PROJECT / "artifacts/private/ticket64-full-backbone-comparison-v2"
FIRST = PROJECT / "artifacts/private/ticket64-full-backbone-comparison-v1"
PRIOR_SHA = "41028974908d8ea8c4fe336f86de0263782100c76b9375dc7693bf60ac213eaf"
SCIENCE_SHA = "557230fec672f6db9252b1895a4d174652fe217c0388a8c9598bd749183aa985"
AUDIT = PROJECT / "artifacts/private/ticket72-checkpoint-recovery-audit-v1"
AUDIT_SHA = "e5ab5477d6acaa636ab9c7aebb8ba531ad91d017ae4d02c29d8874f87c024d75"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket73-configured-continuation-20261002.json"
PARENT_DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket72-checkpoint-audit-20261002.json"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
START_STAGES = (4, 4, 1)
V3 = PROJECT / "artifacts/private/ticket64-full-backbone-comparison-v3"
V3_SHA = "c0cafdac0916705cb672be6f2c45e7db75a661841c8efa6706fbc2cf8a5a1eaf"
from carry_training_configuration import carry_configuration
ARMS = ("head-continuation", "full-backbone")


def successful_terminal(directory, checksum, limit):
    accepted = read(directory / "accepted-launch.json")
    job = str(accepted["job_id"])
    path = directory / f"controller-{job}/terminal-controller.txt"
    observation = read(path.with_name("terminal-observation.json"))
    fields = dict(x.split("=", 1) for x in path.read_text().split() if "=" in x)
    expected = dict(JobId=job, JobState="COMPLETED", ExitCode="0:0", Partition="cluster1-cpu",
                    NumNodes="1", NumCPUs="16", NumTasks="1", MinMemoryNode="64G", TimeLimit=limit,
                    OverSubscribe="OK", Requeue="0", Restarts="0")
    require(accepted["plan_sha256"] == checksum and accepted["launch_attempt"] == 1
            and observation["job_id"] == job and observation["state"] == "COMPLETED" and observation["returncode"] == 0
            and all(fields.get(k) == v for k, v in expected.items()) and "gpu" not in fields.get("TRES", "").lower(),
            "Exact successful actual CPU terminal required")
    seconds = (datetime.fromisoformat(fields["EndTime"]) - datetime.fromisoformat(fields["StartTime"])).total_seconds()
    h, m, s = map(int, limit.split(":"))
    require(0 < seconds <= h * 3600 + m * 60 + s, "Actual allocation interval must fit its bound")
    return seconds, path


def audited_boundaries():
    require(digest(AUDIT / "plan.json") == AUDIT_SHA, "Exact independent audit plan required")
    audit_admission(AUDIT_SHA)
    plan, result = read(AUDIT / "plan.json"), read(AUDIT / "checkpoint-audit.json")
    seconds, path = successful_terminal(AUDIT, AUDIT_SHA, "00:30:00")
    accepted = read(AUDIT / "accepted-launch.json")
    require(result["status"] == "COMPLETE" and result["plan_sha256"] == AUDIT_SHA
            and str(result["job_id"]) == str(accepted["job_id"]) and 0 < result["elapsed_seconds"] <= 1740
            and result["original_missing_allocation_charge_seconds"] == 39600
            and all(result[k] is False for k in ("historical_backend_terminals_verified", "original_strict_ancestry_passed",
                 "fitting_continuation_admitted", "optimizer_updates_repeated", "predictions_or_metrics_computed", "candidate_promotion"))
            and result["gpu_seconds"] == result["tpu_seconds"] == result["external_spend_usd"] == 0,
            "Actual sole audit success with honest historical limitations required")
    require(all(digest(r["path"]) == r["sha256"] for r in plan["frozen_files"]), "Frozen audit state/source changed")
    require(len(result["boundaries"]) == len(plan["boundaries"]) == 2, "Both complete interrupted states required")
    for expected, actual in zip(plan["boundaries"], result["boundaries"], strict=True):
        require(all(actual[k] == v for k, v in expected.items()) and all(actual.get(k) is True for k in
                    ("model_adam_scheduler_rng_checked", "original_full_source_pixels_targets_masks_exposures_checked",
                     "original_initial_tensors_and_normalization_checked", "strict_checkpoint_identity_checked")),
                "Every independent original source/model/state check must pass")
    return result["boundaries"], seconds, path


def stage_location(fold, stage, checksum):
    require(fold in (0, 1, 2) and stage in range(8), "Unknown registered stage")
    if (fold, stage) == (1, 3):
        return V3 / "full-backbone/fold-1/stage-3", V3_SHA
    if stage >= START_STAGES[fold]:
        return ROOT / f"full-backbone/fold-{fold}/stage-{stage}", checksum
    if fold == stage == 0:
        return FIRST / "full-backbone/fold-0/stage-0", SCIENCE_SHA
    return PRIOR / f"full-backbone/fold-{fold}/stage-{stage}", PRIOR_SHA


def stage_record(fold, stage, checksum):
    directory, sha = stage_location(fold, stage, checksum)
    record = read(directory / "stage-result.json")
    require(record["status"] == "COMPLETE" and record["plan_sha256"] == sha
            and record["successful_steps"] == min(7500, (stage + 1) * 1000)
            and record["progress"]["successful_optimizer_steps"] == record["progress"]["attempted_batches"] == record["successful_steps"]
            and record["intermediate_prediction"] is False and record["candidate_promotion"] is False,
            "Exact unchanged registered stage budget required")
    if (fold, stage) in ((0, 3), (1, 2)):
        rows, _, _ = audited_boundaries()
        row = next(r for r in rows if r["fold"] == fold)
        require(row["stage"] == stage and str(row["original_job_id"]) == record["job_id"]
                and row["checkpoint_sha256"] == record["checkpoint_sha256"]
                and row["successful_steps"] == record["successful_steps"]
                and row["historical_backend_terminal_verified"] is False,
                "Only the independently audited historical boundary may be inherited")
        return record, "independently_audited_checkpoint_boundary"
    successful_terminal(directory, sha, "05:30:00")
    return record, "actual_successful_terminal"


def control_record(fold):
    successful_terminal(PRIOR / "control", PRIOR_SHA, "02:00:00")
    record = read(PRIOR / f"head-continuation/fold-{fold}/run.json")
    require(record["status"] == "COMPLETE" and record["plan_sha256"] == PRIOR_SHA
            and record["scientific_plan_sha256"] == SCIENCE_SHA
            and record["progress"]["successful_optimizer_steps"] == record["progress"]["attempted_batches"] == 7500,
            "Reuse only the exact completed original matched control")
    return record


def helpers():
    from ticket64_full_backbone_recovery_v2 import original, scientific_identity, final_predictions
    return original, scientific_identity, final_predictions


def remaining_stages():
    return [dict(fold=f, stage=s, prior_steps=s * 1000, target_steps=min(7500, (s + 1) * 1000))
            for f in range(3) for s in range(START_STAGES[f], 8)]


def live_stage_ready():
    directory = V3 / "full-backbone/fold-1/stage-3"
    accepted = read(directory / "accepted-launch.json")
    require(accepted["job_id"] == 8113 and accepted["plan_sha256"] == V3_SHA
            and accepted["launch_attempt"] == 1, "Only existing 8113 may be inherited")
    terminal = directory / "controller-8113/terminal-controller.txt"
    if terminal.exists():
        successful_terminal(directory, V3_SHA, "05:30:00")
        record, kind = stage_record(1, 3, "not-yet-prepared")
        previous, _ = stage_record(1, 2, "not-yet-prepared")
        require(kind == "actual_successful_terminal" and record["prior_checkpoint_sha256"] == previous["checkpoint_sha256"]
                and record["job_id"] == "8113" and record["successful_steps"] == 4000
                and digest(record["checkpoint_path"]) == record["checkpoint_sha256"],
                "Exact completed live budget/state and predecessor required")
        return True
    raw = subprocess.run(["scontrol", "show", "job", "8113", "-o"], capture_output=True, text=True, timeout=15)
    require(not (raw.returncode != 0 and "Invalid job id" in raw.stderr), "Live inherited handle purged without terminal; never retry")
    fields = dict(x.split("=", 1) for x in raw.stdout.split() if "=" in x)
    if raw.returncode == 0:
        require(fields.get("JobId") == "8113" and fields.get("JobState") in ("PENDING", "RUNNING", "COMPLETING", "COMPLETED"),
                "Existing inherited job failed; stop without retry")
    return False


def close_live_input(checksum):
    plan = admission(checksum)
    if not live_stage_ready():
        return False
    directory = V3 / "full-backbone/fold-1/stage-3"
    record, _ = stage_record(1, 3, checksum)
    destination = ROOT / "fold-1-recovery-seed.pt"
    closure = ROOT / "live-fold-1-input.json"
    if closure.exists():
        row = read(closure)
        require(row["plan_sha256"] == checksum and row["checkpoint_sha256"] == record["checkpoint_sha256"]
                and digest(destination) == row["checkpoint_sha256"]
                and all(digest(r["path"]) == r["sha256"] for r in row["frozen_files"]), "Immutable live input closure changed")
        return True
    require(not destination.exists(), "Live seed without completed input closure requires investigation")
    shutil.copyfile(record["checkpoint_path"], destination)
    require(digest(destination) == record["checkpoint_sha256"], "Completed live state copy differs")
    paths = [directory / "stage-result.json", directory / "accepted-launch.json",
             directory / "controller-8113/terminal-controller.txt", directory / "controller-8113/terminal-observation.json", destination]
    write(closure, dict(plan_sha256=checksum, inherited_plan_sha256=V3_SHA, job_id=8113, fold=1, stage=3,
          successful_steps=4000, checkpoint_sha256=record["checkpoint_sha256"],
          frozen_files=[dict(path=str(p), sha256=digest(p)) for p in paths], optimizer_updates_repeated=False))
    return True


def prepare():
    require(not ROOT.exists() and not DECISION.exists(), "Distinct immutable configured route already prepared")
    import ticket72_full_backbone_continuation as previous_route
    previous_route.admission(V3_SHA)
    rows, audit_seconds, audit_terminal = audited_boundaries()
    require(digest(PRIOR / "plan.json") == PRIOR_SHA and digest(FIRST / "plan.json") == SCIENCE_SHA, "Exact original plans required")
    prior = read(PRIOR / "plan.json")
    require(all(digest(r["path"]) == r["sha256"] for r in prior["frozen_files"]), "Original fitted inputs/source changed")
    live_stage_ready()
    failed = V3 / "full-backbone/fold-0/stage-4"
    accepted = read(failed / "accepted-launch.json")
    terminal = failed / "controller-8111/terminal-controller.txt"
    observation = read(terminal.with_name("terminal-observation.json"))
    fields = dict(x.split("=", 1) for x in terminal.read_text().split() if "=" in x)
    diagnosis = read(V3 / "configuration-failure-diagnosis.json")
    require(accepted["job_id"] == 8111 and accepted["plan_sha256"] == V3_SHA
            and fields["JobId"] == "8111" and fields["JobState"] == "FAILED" and fields["ExitCode"] == "1:0"
            and observation["state"] == "FAILED" and diagnosis["failed_before_optimizer_loop"] is True
            and diagnosis["new_training_checkpoint_or_attempt_files"] is False
            and diagnosis["actual_allocated_cpu_node_seconds"] == 349
            and digest(terminal) == diagnosis["terminal_sha256"]
            and digest(failed / "slurm-8111.err") == diagnosis["trace_sha256"]
            and not (failed / "stage-result.json").exists(), "Exact failed-before-update allocation required")
    require(not any(p.is_file() for p in (V3 / "full-backbone/fold-0/training").rglob("*")), "Failed fold0 produced state; investigate before recovery")
    for fold, end in enumerate((4, 3, 1)):
        control_record(fold)
        for stage in range(end):
            current, _ = stage_record(fold, stage, "not-yet-prepared")
            if stage:
                require(current["prior_checkpoint_sha256"] == previous["checkpoint_sha256"], "Original ancestry differs")
            previous = current
    remaining = remaining_stages()
    require(len(remaining) == 15 and sum(r["target_steps"]-r["prior_steps"] for r in remaining) == 13500,
            "Preserve all remaining updates plus live1000, no repeated steps")
    for row in remaining:
        old = V3 / f"full-backbone/fold-{row['fold']}/stage-{row['stage']}"
        require(not (old / "launch-intent.json").exists() or (row['fold'], row['stage']) == (0, 4), "Another future stage already consumed")
    allocated = (datetime.fromisoformat(fields["EndTime"]) - datetime.fromisoformat(fields["StartTime"])).total_seconds()
    require(allocated == 349 and fields["NumCPUs"] == "16" and fields["MinMemoryNode"] == "64G"
            and fields["TimeLimit"] == "05:30:00" and fields["Requeue"] == fields["Restarts"] == "0"
            and "gpu" not in fields.get("TRES", "").lower(), "Actual failed allocation charge/resource identity differs")
    # Reproject only unfinished work. All historic charges remain charged. Two
    # slots: fold0 has four stages, fold1 live+four, fold2 seven starts after fold0.
    calendar = dict(remaining_fit_critical_path_seconds=11 * 19800, conditional_final_fit_seconds=8 * 19800,
        independent_native_replay_seconds=5400, offline_delivery_seconds=7200, queue_transition_seconds=21600,
        additional_retained_cpu_delivery_seconds=21600, additional_retained_gpu_delivery_seconds=7200,
        retained_review_seconds=86400)
    protected = sum(calendar.values())
    freeze = prior["projection"]["science_freeze_utc"]
    finish = datetime.now(timezone.utc) + timedelta(seconds=protected)
    require(finish < datetime.fromisoformat(freeze), "Entire remaining scientific/delivery/review route exceeds freeze")
    usage = subprocess.check_output(["du", "-sb", str(PROJECT), "/home/ch.jitheswar/models/qwen3-8b-b968826d9c46dd6066d109eabc6255188de91218"], text=True)
    existing = sum(int(r.split()[0]) for r in usage.splitlines())
    old_decision = read(V3 / "plan.json")["decision_path"]
    old_storage = read(old_decision)["storage"]
    protected_bytes = old_storage["protected_allowance_bytes"] + old_storage["additional_allowance_bytes"]
    extra_bytes = 1_000_000_000
    require(existing + protected_bytes + extra_bytes < 2_000_000_000_000
            and shutil.disk_usage(PROJECT).free > protected_bytes + extra_bytes, "Whole remaining storage exceeds bound")
    configs = []
    for fold in range(3):
        path = (FIRST if fold == 0 else PRIOR) / f"full-backbone/fold-{fold}/training/configuration.json"
        value = read(path)
        _, identity, _ = helpers()
        require(value["fold"] == fold and value["identity"] == identity(prior, PRIOR_SHA, fold, "full-backbone")
                and value["scientific_configuration_fingerprint"] == stage_record(fold, 0, "not-yet-prepared")[0]["scientific_config_fingerprint"],
                "Original complete configuration/fold/source identity required")
        configs.append(dict(fold=fold, path=str(path), sha256=digest(path)))
    ROOT.mkdir(mode=0o700)
    snapshot = ROOT / "source-snapshot"
    shutil.copytree(read(V3 / "plan.json")["source_snapshot"], snapshot, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in (Path(__file__).name, "carry_training_configuration.py", "validate_ticket73_configured_full_comparison.py", "coordinate_ticket73_configured_full_continuation.py"):
        shutil.copyfile(PROJECT / "scripts" / name, snapshot / name)
    third, _ = stage_record(2, 0, "not-yet-prepared")
    third_seconds, third_terminal = successful_terminal(PRIOR / "full-backbone/fold-2/stage-0", PRIOR_SHA, "05:30:00")
    rows = [r for r in rows if r["fold"] == 0] + [dict(fold=2, stage=0, original_job_id=8081, successful_steps=1000,
        checkpoint_path=third["checkpoint_path"], checkpoint_sha256=third["checkpoint_sha256"], historical_backend_terminal_verified=True)]
    frozen = [FIRST / "plan.json", PRIOR / "plan.json", V3 / "plan.json", Path(old_decision), AUTH,
        AUDIT / "checkpoint-audit.json", audit_terminal, audit_terminal.with_name("terminal-observation.json"),
        V3 / "configuration-failure-diagnosis.json", V3 / "configuration-handoff-8113.json",
        failed / "accepted-launch.json", terminal, terminal.with_name("terminal-observation.json"), failed / "slurm-8111.err",
        V3 / "full-backbone/fold-1/stage-3/accepted-launch.json", third_terminal, third_terminal.with_name("terminal-observation.json")]
    recovered = []
    for row in rows:
        # Use the already frozen v3 seeds; never read a moving old checkpoint.
        source = V3 / f"fold-{row['fold']}-recovery-seed.pt"
        require(digest(source) == row["checkpoint_sha256"], "Exact original audited seed required")
        destination = ROOT / source.name
        shutil.copyfile(source, destination)
        frozen.append(destination)
        recovered.append(dict(row, continuation_seed_path=str(destination)))
    for config in configs:
        destination = ROOT / f"full-backbone/fold-{config['fold']}/training/configuration.json"
        carry_configuration(config["path"], destination, expected_sha256=config["sha256"])
        frozen.extend([Path(config["path"]), destination])
    frozen.extend(p for p in snapshot.rglob("*") if p.is_file())
    plan = dict(prior, schema_version="rsna-ticket73-configured-continuation-v1", run_root=str(ROOT), source_snapshot=str(snapshot),
        decision_path=str(DECISION), created_at_utc=datetime.now(timezone.utc).isoformat(),
        recovery=dict(checkpoint_audit_plan_sha256=AUDIT_SHA, checkpoint_audit_sha256=digest(AUDIT / "checkpoint-audit.json"),
            boundaries=recovered, configuration_inputs=configs, remaining_stages=remaining,
            remaining_optimizer_updates=13500, inherited_live_job_id=8113, inherited_live_plan_sha256=V3_SHA,
            inherited_live_remaining_optimizer_updates=1000, live_input_closure_required=True,
            historical_backend_terminals_verified=False, original_strict_ancestry_passed=False,
            original_missing_allocation_charge_seconds=39600, actual_failed_allocation_seconds=349,
            optimizer_updates_repeated=False, scientific_recipe_changed=False),
        remaining_calendar=calendar, retained_complete_route_finish_utc=finish.isoformat(),
        frozen_files=[dict(path=str(p), sha256=digest(p)) for p in frozen],
        conditional_final_fit_authorized=False, candidate_promotion=False, competition_submission=False)
    write(ROOT / "plan.json", plan)
    checksum = digest(ROOT / "plan.json")
    write(DECISION, dict(decision="approved", plan_sha256=checksum, authorization_sha256=digest(AUTH),
        resources=plan["resources"], projection=plan["projection"], remaining_calendar=calendar,
        decision_made_by="campaign agent under standing owner authorization", new_exact_owner_numeric_approval_claimed=False,
        remaining_stage_count=15, maximum_new_fit_cpu_node_seconds=15 * 19800,
        retained_live_job_maximum_cpu_node_seconds=19800, remaining_optimizer_updates=13500,
        previously_retained_unused_stage_allowances_reused=True, separately_admitted_failed_fold_recovery=True,
        complete_route_finish_utc=finish.isoformat(), science_freeze_utc=freeze,
        storage=dict(existing_apparent_bytes=existing, protected_allowance_bytes=protected_bytes,
            additional_allowance_bytes=extra_bytes, projected_apparent_bytes=existing + protected_bytes + extra_bytes),
        original_missing_allocation_charge_seconds=39600, actual_failed_allocation_seconds=349,
        scientific_recipe_changed=False, external_spend_usd=0, gpu_seconds=0, tpu_seconds=0,
        automatic_retry=False, candidate_promotion=False))
    launchers = []
    for row in remaining:
        directory = ROOT / f"full-backbone/fold-{row['fold']}/stage-{row['stage']}"
        directory.mkdir(mode=0o700, parents=True)
        launcher = directory / "run.sbatch"
        launcher.write_text(f"""#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket73-full-{row['fold']}-{row['stage']}
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:30:00
#SBATCH --no-requeue
#SBATCH --output={directory}/slurm-%j.out
#SBATCH --error={directory}/slurm-%j.err
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
{PROJECT}/.venv/bin/python {snapshot}/{Path(__file__).name} full {checksum} {row['fold']} {row['stage']}
""")
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        launchers.append(dict(path=str(launcher), sha256=digest(launcher), fold=row["fold"], stage=row["stage"]))
    write(ROOT / "prepared-validation.json", dict(plan_sha256=checksum, decision_sha256=digest(DECISION), launchers=launchers, launched=False))
    print(json.dumps(dict(status="PREPARED_ONLY", plan_sha256=checksum, new_remaining_updates=13500, live_remaining_updates=1000)))


def admission(checksum):
    require(digest(ROOT / "plan.json") == checksum, "Exact immutable continuation plan required")
    plan, decision, prepared = read(ROOT / "plan.json"), read(DECISION), read(ROOT / "prepared-validation.json")
    require(decision["decision"] == "approved" and decision["plan_sha256"] == prepared["plan_sha256"] == checksum
            and digest(DECISION) == prepared["decision_sha256"] and decision["resources"] == plan["resources"]
            and decision["projection"] == plan["projection"] and digest(AUTH) == plan["authorization_sha256"], "Exact continuation resource admission required")
    for row in plan["frozen_files"] + read(PRIOR / "plan.json")["frozen_files"]:
        require(digest(row["path"]) == row["sha256"], "Frozen scientific input, seed or source changed")
    closure = ROOT / "live-fold-1-input.json"
    if closure.exists():
        row = read(closure)
        require(row["plan_sha256"] == checksum and row["successful_steps"] == 4000 and row["job_id"] == 8113
                and all(digest(r["path"]) == r["sha256"] for r in row["frozen_files"]), "Live completed input closure changed")
    return plan


def previous_stage(fold, stage, checksum):
    require(stage >= START_STAGES[fold] and stage in range(8), "Unknown unused continuation stage")
    if stage == 0:
        return None, None
    record, kind = stage_record(fold, stage - 1, checksum)
    if stage == START_STAGES[fold]:
        if fold == 1:
            require((ROOT / "live-fold-1-input.json").exists(), "Completed live input closure required before next fold1 stage")
            row = read(ROOT / "live-fold-1-input.json")
            require(row["plan_sha256"] == checksum and row["checkpoint_sha256"] == record["checkpoint_sha256"], "Exact inherited live seed required")
            path = ROOT / "fold-1-recovery-seed.pt"
        else:
            row = next(r for r in read(ROOT / "plan.json")["recovery"]["boundaries"] if r["fold"] == fold)
            path = Path(row["continuation_seed_path"])
        require(kind == ("independently_audited_checkpoint_boundary" if fold == 0 else "actual_successful_terminal"), "Exact admitted inherited boundary required")
    else:
        path = ROOT / f"full-backbone/fold-{fold}/training/checkpoints/latest.pt"
        require(kind == "actual_successful_terminal", "Every new predecessor needs its actual successful terminal")
    require(record["successful_steps"] == stage * 1000 and digest(path) == record["checkpoint_sha256"], "Exact predecessor state bytes required")
    return record, path


def _launch_under_lock(checksum, fold, stage):
    plan = admission(checksum)
    require(datetime.now(timezone.utc) < datetime.fromisoformat(plan["projection"]["science_freeze_utc"]), "Science freeze reached")
    previous_stage(fold, stage, checksum)
    directory = ROOT / f"full-backbone/fold-{fold}/stage-{stage}"
    require(not (directory / "launch-intent.json").exists(), "Consumed stage intent; never retry")
    row = next(r for r in read(ROOT / "prepared-validation.json")["launchers"] if r["fold"] == fold and r["stage"] == stage)
    require(digest(row["path"]) == row["sha256"], "Exact frozen launcher required")
    jobs = subprocess.check_output(["squeue", "-u", "ch.jitheswar", "-h", "-o", "%i|%P"], text=True).splitlines()
    if sum(r.split("|")[1] == "cluster1-cpu" for r in jobs) >= 2:
        print(json.dumps(dict(status="WAIT_FOR_CPU_SLOT", fold=fold, stage=stage)), flush=True)
        return False
    write(directory / "launch-intent.json", dict(plan_sha256=checksum, launcher_sha256=row["sha256"], launch_attempt=1))
    job = int(subprocess.check_output(["sbatch", "--parsable", row["path"]], text=True).strip().split(";")[0])
    write(directory / "accepted-launch.json", dict(job_id=job, plan_sha256=checksum, launcher_sha256=row["sha256"], launch_attempt=1))
    subprocess.run(["systemd-run", "--user", f"--unit=rsna-ticket73-full-{fold}-{stage}-{job}-observer", "--collect",
                    str(PROJECT / ".venv/bin/python"), str(PROJECT / "scripts/monitor_ticket48_cpu_pilot.py"), str(directory), str(job)], check=True)
    print(json.dumps(dict(status="ACCEPTED", job_id=job, fold=fold, stage=stage)), flush=True)



def launch(checksum, fold, stage):
    with (PROJECT / "artifacts/private/cpu-scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _launch_under_lock(checksum, fold, stage)


def runtime(checksum):
    plan = admission(checksum)
    require(Path(__file__).resolve().parent == Path(plan["source_snapshot"])
            and os.environ.get("PYTHONPATH") == str(Path(plan["source_snapshot"]) / "src")
            and all(importlib.metadata.version(k) == v for k, v in plan["runtime_versions"].items()), "Frozen original CPU runtime required")
    raw = subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], text=True, timeout=15)
    fields = dict(x.split("=", 1) for x in raw.split() if "=" in x)
    expected = dict(JobId=os.environ["SLURM_JOB_ID"], JobState="RUNNING", Partition="cluster1-cpu", NumNodes="1",
                    NumCPUs="16", NumTasks="1", MinMemoryNode="64G", TimeLimit="05:30:00", OverSubscribe="OK", Requeue="0", Restarts="0")
    require(all(fields.get(k) == v for k, v in expected.items()) and "gpu" not in fields.get("TRES", "").lower(), "Exact bounded CPU fitting allocation required")
    import torch
    require(torch.version.cuda is None and not torch.cuda.is_available(), "CPU-only fitting required")
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Unchanged fitting watchdog")))
    signal.alarm(19740)
    return plan, raw


def full_stage(checksum, fold, stage):
    import resource
    import torch
    require(fold in (0, 1, 2) and stage in range(START_STAGES[fold], 8), "Unknown continuation stage")
    plan, controller = runtime(checksum)
    original, scientific_identity, final_predictions = helpers()
    start = time.monotonic()
    directory = ROOT / f"full-backbone/fold-{fold}/stage-{stage}"
    accepted = read(directory / "accepted-launch.json")
    require(str(accepted["job_id"]) == os.environ["SLURM_JOB_ID"] and accepted["plan_sha256"] == checksum
            and accepted["launch_attempt"] == 1 and not (directory / "stage-result.json").exists(), "Exact sole unused stage job required")
    bank, gold, weak = original.fold_inputs(plan, fold)
    model = original.initial_for_arm(fold, "full-backbone")
    initial_sha = original._model_state_sha256(model)
    buffers = {k: v.clone() for k, v in model.state_dict().items() if k.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    parity = None
    if stage == 0:
        selected = {pool[i].study_instance_uid for pool, seed in ((gold, 20260912), (weak, 20260912 + 34034034))
                    for i in original._batch_indices(len(pool), 2, 0, seed)}
        parity = original.verify_initial_batch(model, gold, weak, original.feature_bank({k: v for k, v in bank.items() if k in selected}))
    previous, resume = previous_stage(fold, stage, checksum)
    training = ROOT / f"full-backbone/fold-{fold}/training"
    config = next(r for r in plan["recovery"]["configuration_inputs"] if r["fold"] == fold)
    carry_configuration(config["path"], training / "configuration.json", expected_sha256=config["sha256"])
    stop = min(7500, (stage + 1) * 1000)
    tick = time.monotonic()
    stopped = False
    try:
        original.train_fixed_budget(gold, original.config_for_arm("full-backbone"), additional_examples=weak,
            additional_batch_order_seed=20260912 + 34034034, model=model, artifact_dir=training, fold=fold,
            identity=scientific_identity(plan, PRIOR_SHA, fold, "full-backbone"), resume_from=resume,
            requested_resources=plan["resources"], interrupt_after_steps=stop if stop < 7500 else None)
    except original.TrainingInterrupted:
        if stop == 7500:
            raise
        stopped = True
    trainer_seconds = time.monotonic() - tick
    path = training / "checkpoints" / ("latest.pt" if stopped else "final.pt")
    checkpoint = original.load_training_checkpoint(path, model=model, expected_config=original.config_for_arm("full-backbone"),
        expected_fold=fold, expected_identity=scientific_identity(plan, PRIOR_SHA, fold, "full-backbone"))
    require(checkpoint["progress"]["successful_optimizer_steps"] == checkpoint["progress"]["attempted_batches"] == stop
            and all(torch.equal(model.state_dict()[k], v) for k, v in buffers.items()), "Exact resumed boundary and fixed normalization required")
    changed = None
    if stop == 7500:
        initial = original.initial_model(fold).state_dict()
        changed = {prefix: any(not torch.equal(v, initial[k]) for k, v in model.state_dict().items() if k.startswith(prefix))
                   for prefix in ("encoder.conv1", "encoder.layer1", "encoder.layer2", "encoder.layer3", "encoder.layer4", "head")}
        require(all(changed.values()), "Every registered treatment stage must update")
    predictions = final_predictions(model, plan, fold, "full-backbone") if stop == 7500 else {}
    write(directory / "stage-result.json", dict(status="COMPLETE", plan_sha256=checksum, scientific_plan_sha256=SCIENCE_SHA,
        job_id=os.environ["SLURM_JOB_ID"], fold=fold, stage=stage, controller=controller, successful_steps=stop, planned_boundary=stopped,
        scientific_config_fingerprint=original.config_for_arm("full-backbone").config_fingerprint,
        checkpoint_path=str(path), checkpoint_sha256=digest(path), prior_checkpoint_sha256=None if previous is None else previous["checkpoint_sha256"],
        initial_model_state_sha256=initial_sha, initial_native_feature_logit_error=parity, trainer_seconds=trainer_seconds,
        elapsed_seconds=time.monotonic() - start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        progress=checkpoint["progress"], normalization_buffers_unchanged=True, all_encoder_stages_changed=changed,
        historical_backend_terminals_verified=False, original_strict_ancestry_passed=False,
        intermediate_prediction=False, candidate_promotion=False, **predictions))


def analyze(checksum):
    admission(checksum)
    original, _, _ = helpers()
    values = {a: {} for a in ARMS}
    zero = {a: {} for a in ARMS}
    parent, records = {}, []
    for fold in range(3):
        for stage in range(8):
            current, _ = stage_record(fold, stage, checksum)
            if stage:
                require(current["prior_checkpoint_sha256"] == previous["checkpoint_sha256"], "Full unchanged stage ancestry differs")
            previous = current
        control, full = control_record(fold), current
        require(control["initial_model_state_sha256"] == full["initial_model_state_sha256"]
                and all(control["progress"][k] == full["progress"][k] for k in
                    ("attempted_batches", "sample_exposures", "supervised_entry_exposures", "exposures_by_source")), "Matched scientific initialization/exposures differ")
        for arm, record in zip(ARMS, (control, full), strict=True):
            require(digest(record["checkpoint_path"]) == record["checkpoint_sha256"], "Exact six final checkpoint bytes required")
            values[arm].update(record["predictions"])
            zero[arm].update(record["encoded_image_ablation"])
            records.append(dict(arm=arm, fold=fold, checkpoint_path=record["checkpoint_path"], checkpoint_sha256=record["checkpoint_sha256"]))
        parent.update(read(original.PARENT / f"report-supervision/fold-{fold}/run.json")["predictions"])
    manifest = original.load_fold_manifest(original.t36.FOLD_MANIFEST_PATH)
    def compare(base, new):
        return original.compare_image_model_predictions(base, new, manifest, bootstrap_resamples=2000, bootstrap_seed=20261002)
    comparisons = dict(primary=compare(values[ARMS[0]], values[ARMS[1]]))
    for arm in ARMS:
        comparisons[f"{arm}-vs-parent"] = compare(parent, values[arm])
        comparisons[f"{arm}-image-signal"] = compare(zero[arm], values[arm])
    primary = comparisons["primary"]
    lower = primary["paired_study_bootstrap"]["macro_auc"]["lower"]
    passed = lower is not None and lower > 0 and all(r["delta_macro_auc"] is not None and r["delta_macro_auc"] > 0 for r in primary["delta"]["folds"])
    passed = passed and all(comparisons[f"{arm}-image-signal"]["paired_study_bootstrap"]["macro_auc"]["lower"] is not None
                           and comparisons[f"{arm}-image-signal"]["paired_study_bootstrap"]["macro_auc"]["lower"] > 0 for arm in ARMS)
    write(ROOT / "analysis.json", dict(status="COMPLETE", plan_sha256=checksum, scientific_plan_sha256=SCIENCE_SHA, comparisons=comparisons,
        run_records=records, provisional_review_gate_passed=passed, independent_original_mri_replay_still_required=True,
        historical_backend_terminals_verified=False, original_strict_ancestry_passed=False, original_missing_allocation_charge_seconds=39600,
        candidate_promotion=False, automatic_promotion=False, first_place_established=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch", "full", "analyze"))
    parser.add_argument("checksum", nargs="?")
    parser.add_argument("fold", nargs="?", type=int)
    parser.add_argument("stage", nargs="?", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "analyze":
        analyze(args.checksum)
    else:
        {"launch": launch, "full": full_stage}[args.command](args.checksum, args.fold, args.stage)
