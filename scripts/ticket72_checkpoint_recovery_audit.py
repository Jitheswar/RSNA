"""Audit existing interrupted-stage checkpoints without fitting or scoring.

Successful audit proves resumable state consistency, not the missing original
Slurm terminal. Each missing allocation retains its complete 19800-second bound.
"""

import argparse
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

PROJECT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
ROOT = PROJECT / "artifacts/private/ticket72-checkpoint-recovery-audit-v1"
OLD = PROJECT / "artifacts/private/ticket64-full-backbone-comparison-v2"
OLD_SHA = "41028974908d8ea8c4fe336f86de0263782100c76b9375dc7693bf60ac213eaf"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket72-checkpoint-audit-20261002.json"
PARENT_DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket70-exact-anchor-recovery-v4-20261002.json"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
BOUNDARIES = ((0, 3, 8074, 4000, "c6fdb7e2d1d11b18264d69529e9e20a19d4e41e6560c423dd6835eff3d8a7746"),
              (1, 2, 8072, 3000, "70af969f9161fd4d7e4d94eb0046665bdfe1d9c37412de07794c472c32219f14"))


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_progress(payload, record, steps, exposures, reference_count):
    progress = payload["progress"]
    require(progress == record["progress"], "Worker/checkpoint progress differs")
    require(all(type(progress[k]) is int and progress[k] == steps for k in
                ("successful_optimizer_steps", "attempted_batches", "scheduler_steps", "global_step", "next_batch_index")),
            "Exact complete planned optimizer/scheduler/sampler boundary required")
    require(progress["skipped_all_masked_batches"] == progress["consecutive_no_progress"] == 0
            and progress["sample_exposures"] == 4 * steps
            and progress["exposures_by_source"] == exposures
            and progress["supervised_entry_exposures"] == sum(x["supervised_entry_exposures"] for x in exposures.values()),
            "Exact original-source target/mask exposures required")
    batches = (reference_count + 1) // 2
    require(progress["epoch"] == steps // batches and progress["batch_in_epoch"] == steps % batches,
            "Exact original reference epoch position required")
    require(payload["sampler_state"] == {"kind": "deterministic-cyclic", "seed": 20260912, "next_batch_index": steps},
            "Original deterministic sampler state required")
    require(payload["checkpoint_kind"] == "latest_resumable" and "planned_interruption" in payload["checkpoint_reason"].split("+"),
            "Complete planned interrupt checkpoint required")


def validate_adam(optimizer, steps):
    import torch
    state = optimizer.state_dict()
    require([g["lr"] for g in state["param_groups"]] == [1e-5, 1e-4], "Original full-backbone Adam rates required")
    ids = [p for g in state["param_groups"] for p in g["params"]]
    require(len(set(ids)) == len(ids) and set(ids) == set(state["state"]), "Every trainable parameter needs one Adam state")
    for group in optimizer.param_groups:
        require(group["betas"] == (0.9, 0.999) and group["eps"] == 1e-8 and group["weight_decay"] == 0
                and not group["amsgrad"] and not group["maximize"], "Original Adam hyperparameters required")
        for param in group["params"]:
            item = optimizer.state[param]
            require(set(item) == {"step", "exp_avg", "exp_avg_sq"}
                    and item["step"].numel() == 1 and float(item["step"]) == steps
                    and item["exp_avg"].shape == item["exp_avg_sq"].shape == param.shape
                    and item["exp_avg"].dtype == item["exp_avg_sq"].dtype == param.dtype
                    and torch.isfinite(item["exp_avg"]).all() and torch.isfinite(item["exp_avg_sq"]).all()
                    and (item["exp_avg_sq"] >= 0).all(), "Complete finite shape-compatible Adam moments and exact duration required")


def prepare():
    require(not ROOT.exists() and not DECISION.exists(), "Immutable audit already prepared")
    require(digest(OLD / "plan.json") == OLD_SHA, "Exact original execution plan required")
    old = read(OLD / "plan.json")
    require(all(digest(r["path"]) == r["sha256"] for r in old["frozen_files"]), "Original frozen input/source changed")
    previous = read(PARENT_DECISION)
    protected = previous["calendar"]["protected_prior_seconds"] + previous["calendar"]["additional_route_seconds"]
    review = previous["calendar"]["review_seconds"]
    finish = datetime.now(timezone.utc) + timedelta(seconds=protected + review + 5400)
    require(finish < datetime.fromisoformat(previous["calendar"]["science_freeze_utc"]), "Entire retained route plus audit/review lacks calendar headroom")
    usage = subprocess.check_output(["du", "-sb", str(PROJECT), "/home/ch.jitheswar/models/qwen3-8b-b968826d9c46dd6066d109eabc6255188de91218"], text=True)
    existing = sum(int(r.split()[0]) for r in usage.splitlines())
    protected_bytes = previous["storage"]["protected_allowance_bytes"] + previous["storage"]["additional_allowance_bytes"]
    require(existing + protected_bytes + 1_000_000_000 < 2_000_000_000_000
            and shutil.disk_usage(PROJECT).free > protected_bytes + 1_000_000_000, "Entire retained storage plus checkpoint copies lacks headroom")
    ROOT.mkdir(mode=0o700)
    snapshot = ROOT / "source-snapshot"
    shutil.copytree(old["source_snapshot"], snapshot, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(__file__, snapshot / Path(__file__).name)
    frozen = [OLD / "plan.json", AUTH, PARENT_DECISION]
    rows = []
    for fold, stage, job, steps, sha in BOUNDARIES:
        directory = OLD / f"full-backbone/fold-{fold}/stage-{stage}"
        accepted, result = read(directory / "accepted-launch.json"), read(directory / "stage-result.json")
        require(accepted["job_id"] == job and accepted["plan_sha256"] == OLD_SHA and accepted["launch_attempt"] == 1
                and result["job_id"] == str(job) and result["plan_sha256"] == OLD_SHA
                and result["status"] == "COMPLETE" and result["successful_steps"] == steps
                and result["checkpoint_sha256"] == sha and result["planned_boundary"] is True
                and result["intermediate_prediction"] is False and result["candidate_promotion"] is False,
                "Exact sole original worker boundary required")
        require(not (directory / f"controller-{job}/terminal-controller.txt").exists(), "Actual terminal appeared; reconcile original strict route instead")
        raw = subprocess.run(["scontrol", "show", "job", str(job), "-o"], capture_output=True, text=True, timeout=15)
        require(raw.returncode != 0 and "Invalid job id" in raw.stderr, "Original handle must actually be purged, never replace a live worker")
        path = Path(result["checkpoint_path"])
        require(digest(path) == sha, "Exact durable original checkpoint required")
        destination = ROOT / f"fold-{fold}-boundary.pt"
        shutil.copyfile(path, destination)
        require(digest(destination) == sha, "Frozen checkpoint copy differs")
        prior = directory.parent / f"stage-{stage - 1}"
        require(result["prior_checkpoint_sha256"] == read(prior / "stage-result.json")["checkpoint_sha256"], "Original predecessor checkpoint chain differs")
        for name in ("accepted-launch.json", "stage-result.json", "launch-intent.json", "run.sbatch", f"slurm-{job}.err"):
            frozen.append(directory / name)
        frozen += [destination, prior / "stage-result.json"]
        rows.append(dict(fold=fold, stage=stage, original_job_id=job, successful_steps=steps,
                         checkpoint_path=str(destination), checkpoint_sha256=sha, original_stage_directory=str(directory),
                         historical_backend_terminal_verified=False, conservative_original_allocation_seconds=19800,
                         purged_handle_observation=dict(returncode=raw.returncode, stderr=raw.stderr.strip())))
    frozen += [p for p in snapshot.rglob("*") if p.is_file()]
    plan = dict(schema_version="rsna-ticket72-checkpoint-audit-v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
                source_snapshot=str(snapshot), original_execution_plan_sha256=OLD_SHA, boundaries=rows,
                frozen_files=[dict(path=str(p), sha256=digest(p)) for p in frozen],
                historical_backend_terminals_verified=False, original_strict_ancestry_passed=False,
                optimizer_updates_repeated=False, predictions_or_metrics_computed=False,
                fitting_continuation_admitted=False, candidate_promotion=False, competition_submission=False)
    write(ROOT / "plan.json", plan)
    sha = digest(ROOT / "plan.json")
    write(DECISION, dict(decision="approved", plan_sha256=sha, authorization_sha256=digest(AUTH),
            decision_made_by="campaign agent under standing owner authorization", new_exact_owner_numeric_approval_claimed=False,
            cpu_node_seconds=1800, queue_calendar_seconds=3600, maximum_launch_attempts=1, automatic_retry=False,
            time_limit="00:30:00", watchdog_seconds=1740, gpu_seconds=0, tpu_seconds=0, external_spend_usd=0,
            complete_route_finish_utc=finish.isoformat(), protected_prior_calendar_seconds=protected,
            retained_review_seconds=review, science_freeze_utc=previous["calendar"]["science_freeze_utc"],
            storage=dict(existing_apparent_bytes=existing, protected_allowance_bytes=protected_bytes,
                         additional_allowance_bytes=1_000_000_000, projected_apparent_bytes=existing + protected_bytes + 1_000_000_000),
            original_missing_allocation_charge_seconds=39600, fitting_continuation_admitted=False))
    launcher = ROOT / "run.sbatch"
    launcher.write_text(f"""#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket72-checkpoint-audit
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output={ROOT}/slurm-%j.out
#SBATCH --error={ROOT}/slurm-%j.err
#SBATCH --chdir={PROJECT}
set -euo pipefail
umask 077
export RSNA_PROJECT_ROOT={PROJECT}
export PYTHONPATH={snapshot}/src
export CUDA_VISIBLE_DEVICES=-1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
{PROJECT}/.venv/bin/python {snapshot}/{Path(__file__).name} run {sha}
""")
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    write(ROOT / "prepared-validation.json", dict(plan_sha256=sha, decision_sha256=digest(DECISION), launcher_sha256=digest(launcher), launched=False))
    print(json.dumps(dict(status="PREPARED_ONLY", plan_sha256=sha, complete_route_finish_utc=finish.isoformat())))


def admission(checksum):
    require(digest(ROOT / "plan.json") == checksum, "Exact immutable audit plan required")
    plan, prepared, decision = read(ROOT / "plan.json"), read(ROOT / "prepared-validation.json"), read(DECISION)
    require(prepared["plan_sha256"] == checksum == decision["plan_sha256"] and decision["decision"] == "approved"
            and digest(DECISION) == prepared["decision_sha256"] and digest(ROOT / "run.sbatch") == prepared["launcher_sha256"]
            and digest(AUTH) == decision["authorization_sha256"], "Exact resource/source launcher admission required")
    require(all(digest(r["path"]) == r["sha256"] for r in plan["frozen_files"]), "Frozen audit source/input changed")
    require(all(digest(r["path"]) == r["sha256"] for r in read(OLD / "plan.json")["frozen_files"]), "Original scientific input/source changed")
    return plan


def launch(checksum):
    plan = admission(checksum)
    require(not (ROOT / "launch-intent.json").exists(), "Consumed sole audit launch; never retry")
    require(digest(__file__) == digest(Path(plan["source_snapshot"]) / Path(__file__).name), "Exact prepared lifecycle source required")
    jobs = subprocess.check_output(["squeue", "-u", "ch.jitheswar", "-h", "-o", "%i|%P"], text=True).splitlines()
    require(sum(r.split("|")[1] == "cluster1-cpu" for r in jobs) < 2, "Both owner CPU slots occupied")
    write(ROOT / "launch-intent.json", dict(plan_sha256=checksum, launch_attempt=1, observed_owner_queue=jobs))
    job = int(subprocess.check_output(["sbatch", "--parsable", str(ROOT / "run.sbatch")], text=True).strip().split(";")[0])
    write(ROOT / "accepted-launch.json", dict(job_id=job, plan_sha256=checksum, launch_attempt=1))
    subprocess.run(["systemd-run", "--user", f"--unit=rsna-ticket72-audit-{job}-observer", "--collect",
                    str(PROJECT / ".venv/bin/python"), str(PROJECT / "scripts/monitor_ticket48_cpu_pilot.py"), str(ROOT), str(job)], check=True)
    print(json.dumps(dict(status="ACCEPTED", job_id=job, plan_sha256=checksum)))


def run(checksum):
    started = time.monotonic()
    plan = admission(checksum)
    require(Path(__file__).resolve().parent == Path(plan["source_snapshot"]), "Frozen audit executable required")
    job = os.environ["SLURM_JOB_ID"]
    accepted = read(ROOT / "accepted-launch.json")
    require(str(accepted["job_id"]) == job and accepted["plan_sha256"] == checksum and accepted["launch_attempt"] == 1,
            "Exact admitted sole audit job required")
    raw = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True, timeout=15)
    fields = dict(x.split("=", 1) for x in raw.split() if "=" in x)
    expected = dict(JobId=job, JobState="RUNNING", Partition="cluster1-cpu", NumNodes="1", NumCPUs="16", NumTasks="1",
                    MinMemoryNode="64G", TimeLimit="00:30:00", OverSubscribe="OK", Requeue="0", Restarts="0")
    require(all(fields.get(k) == v for k, v in expected.items()) and "gpu" not in fields.get("TRES", "").lower(), "Exact CPU allocation required")
    write(ROOT / "running-allocation.json", dict(job_id=job, controller=raw))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Checkpoint audit watchdog")))
    signal.alarm(1740)
    sys.path.insert(0, str(Path(plan["source_snapshot"])))
    sys.path.insert(0, str(Path(plan["source_snapshot"]) / "src"))
    import torch
    import numpy as np
    from ticket64_full_backbone_recovery_v2 import original, scientific_identity
    from rsna_knee.training import _rng_state, _restore_rng_state
    require(torch.version.cuda is None and not torch.cuda.is_available(), "CPU-only audit required")
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    old = read(OLD / "plan.json")
    require({k: importlib.metadata.version(k) for k in old["runtime_versions"]} == old["runtime_versions"], "Exact original CPU runtime required")
    original.terminal(OLD / "control", OLD_SHA, "02:00:00")
    rows = []
    for boundary in plan["boundaries"]:
        fold, steps = boundary["fold"], boundary["successful_steps"]
        directory = Path(boundary["original_stage_directory"])
        record = read(directory / "stage-result.json")
        for stage in range(boundary["stage"]):
            prior = PROJECT / "artifacts/private/ticket64-full-backbone-comparison-v1/full-backbone/fold-0/stage-0" if fold == stage == 0 else OLD / f"full-backbone/fold-{fold}/stage-{stage}"
            prior_sha = "557230fec672f6db9252b1895a4d174652fe217c0388a8c9598bd749183aa985" if fold == stage == 0 else OLD_SHA
            original.terminal(prior, prior_sha, "05:30:00")
            current = read(prior / "stage-result.json")
            require(current["successful_steps"] == (stage + 1) * 1000 and current["intermediate_prediction"] is False, "Original completed predecessor budgets required")
            if stage:
                require(current["prior_checkpoint_sha256"] == previous["checkpoint_sha256"], "Original completed predecessor chain differs")
            previous = current
        require(record["prior_checkpoint_sha256"] == previous["checkpoint_sha256"], "Recovered boundary must extend the completed predecessor")
        require(not (directory / f"controller-{boundary['original_job_id']}/terminal-controller.txt").exists(), "Historical terminal appeared; reconcile strict ancestry separately")
        # Original bank loader checks every selected fitted pixel byte, target,
        # mask and fingerprint. No optimizer update or prediction is performed.
        bank, gold, weak = original.fold_inputs(old, fold)
        exposures = {k: dict(sample_exposures=0, supervised_entry_exposures=0) for k in ("reference", "weak", "mixed", "pseudo")}
        for attempt in range(steps):
            for kind, pool, seed in (("reference", gold, 20260912), ("weak", weak, 20260912 + 34034034)):
                for index in original._batch_indices(len(pool), 2, attempt, seed):
                    example = pool[index]
                    require(example.supervision_mask.any(), "Empty original fitted supervision")
                    exposures[kind]["sample_exposures"] += 1
                    exposures[kind]["supervised_entry_exposures"] += int(example.supervision_mask.sum())
        reference_count = len(gold)
        del bank, gold, weak
        gc.collect()
        model = original.initial_for_arm(fold, "full-backbone")
        require(original._model_state_sha256(model) == record["initial_model_state_sha256"], "Original initialized model differs")
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        optimizer = torch.optim.Adam([{"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": 1e-5},
                                      {"params": [p for p in model.head.parameters() if p.requires_grad], "lr": 1e-4}], lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        checkpoint = original.load_training_checkpoint(boundary["checkpoint_path"], model=model, optimizer=optimizer, scheduler=scheduler,
                        expected_config=original.config_for_arm("full-backbone"), expected_fold=fold,
                        expected_identity=scientific_identity(old, OLD_SHA, fold, "full-backbone"),
                        expected_checkpoint_kind="latest_resumable", restore_rng=True)
        validate_progress(checkpoint, record, steps, exposures, reference_count)
        validate_adam(optimizer, steps)
        require(scheduler.last_epoch == steps and scheduler.get_last_lr() == [1e-5, 1e-4], "Original complete scheduler state required")
        state = model.state_dict()
        require(all(torch.isfinite(v).all() for v in state.values()) and all(torch.equal(v, initial[k]) for k, v in state.items()
                    if k.endswith(("running_mean", "running_var", "num_batches_tracked"))), "Finite model and unchanged normalization buffers required")
        require(all(any(not torch.equal(v, initial[k]) for k, v in state.items() if k.startswith(prefix))
                    for prefix in ("encoder.conv1", "encoder.layer1", "encoder.layer2", "encoder.layer3", "encoder.layer4", "head")),
                "All originally registered encoder/head stages must have updated")
        rng = _rng_state()
        require(set(checkpoint["rng_state"]) == {"python", "numpy", "torch"}
                and rng["python"] == checkpoint["rng_state"]["python"]
                and all(np.array_equal(a, b) for a, b in zip(rng["numpy"], checkpoint["rng_state"]["numpy"], strict=True))
                and torch.equal(rng["torch"], checkpoint["rng_state"]["torch"]), "Complete RNG state must round-trip exactly")
        rows.append(dict(boundary, model_adam_scheduler_rng_checked=True, original_full_source_pixels_targets_masks_exposures_checked=True,
                         original_initial_tensors_and_normalization_checked=True, strict_checkpoint_identity_checked=True))
        print(json.dumps(dict(fold=fold, steps=steps, checkpoint_state_verified=True)), flush=True)
        del model, initial, optimizer, scheduler, checkpoint
        gc.collect()
    write(ROOT / "checkpoint-audit.json", dict(status="COMPLETE", plan_sha256=checksum, job_id=job, boundaries=rows,
            elapsed_seconds=time.monotonic() - started, original_missing_allocation_charge_seconds=39600,
            historical_backend_terminals_verified=False, original_strict_ancestry_passed=False,
            fitting_continuation_admitted=False, optimizer_updates_repeated=False, predictions_or_metrics_computed=False,
            candidate_promotion=False, gpu_seconds=0, tpu_seconds=0, external_spend_usd=0))
    signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch", "run"))
    parser.add_argument("checksum", nargs="?")
    args = parser.parse_args()
    {"prepare": lambda: prepare(), "launch": lambda: launch(args.checksum), "run": lambda: run(args.checksum)}[args.command]()
