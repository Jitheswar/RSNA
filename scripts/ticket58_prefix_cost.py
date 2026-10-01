"""Measure real cohort loading and fixed matched adaptation batches before fitting."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import threading
import time


PROJECT = Path(os.environ.get("RSNA_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
PREFIX = PROJECT / "artifacts/private/ticket57-resnet-prefix-cohort"
PARENT = PROJECT / "artifacts/private/ticket55-resnet-long-report-head-v2"
ROOT = PROJECT / "artifacts/private/ticket58-prefix-cost-v1"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket58-prefix-cost-20261001.json"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def prepare():
    if ROOT.exists() or DECISION.exists():
        raise FileExistsError("Immutable actual-cohort cost request already prepared")
    verification = read(PREFIX / "verification/verification-result.json")
    job = verification["job_id"]
    terminal = PREFIX / f"verification/controller-{job}/terminal-controller.txt"
    fields = dict(token.split("=", 1) for token in terminal.read_text().split() if "=" in token)
    manifest_path = PREFIX / "features/feature-manifest.json"
    manifest = read(manifest_path)
    if (verification["status"] != "COMPLETE" or fields.get("JobState") != "COMPLETED"
            or fields.get("ExitCode") != "0:0" or fields.get("JobId") != job
            or manifest["study_count"] != 4228 or manifest["valid_slices"] != 187272
            or digest(manifest_path) != verification["feature_manifest_sha256"]
            or not manifest["all_native_encoder_features_verified"]):
        raise ValueError("Independently complete successful activation verification is required")
    parent = read(PARENT / "plan.json")
    parent_validation = read(PARENT / "validation.json")
    if parent_validation["status"] != "COMPLETE" or parent_validation["final_predictions_verified"] != 116:
        raise ValueError("Independently verified warm report heads are required")
    cohort_parent = Path(parent["parent_plan"])
    from rsna_knee import ticket36 as t36
    inputs = [AUTH, PREFIX / "plan.json", manifest_path,
              PREFIX / "verification/verification-result.json", terminal,
              terminal.with_name("terminal-observation.json"), PARENT / "plan.json",
              PARENT / "validation.json", PARENT / "pooled-cohort.npz",
              PARENT / "pooled-cohort-evidence.json", cohort_parent,
              t36.FOLD_MANIFEST_PATH, t36.TICKET30_ARTIFACT, t36.WEIGHTS_PATH,
              PROJECT / ".scratch/rsna-leaderboard/issues/58-test-report-supervised-layer4-adaptation.md"]
    for fold in range(3):
        inputs.extend((PARENT / f"report-supervision/fold-{fold}/run.json",
                       PARENT / f"report-supervision/fold-{fold}/checkpoints/final.pt"))
    ROOT.mkdir(mode=0o700)
    snapshot = ROOT / "source-snapshot"
    shutil.copytree(PROJECT / "src/rsna_knee", snapshot / "src/rsna_knee",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(Path(__file__), snapshot / "cost.py")
    inputs.extend(path for path in snapshot.rglob("*") if path.is_file())
    resources = {"partition": "cluster1-cpu", "nodes": 1, "tasks": 1, "cpus_per_task": 16,
                 "memory_gib": 64, "walltime_seconds": 1800, "code_watchdog_seconds": 1680,
                 "maximum_launch_attempts": 1, "gpu_count": 0, "exclusive": False,
                 "requeue": False, "external_spend_usd": 0}
    plan = {"schema_version": "rsna-ticket58-actual-prefix-cost-plan-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_root": str(ROOT),
            "prefix_root": str(PREFIX), "parent_root": str(PARENT), "cohort_parent_plan": str(cohort_parent),
            "source_snapshot": str(snapshot), "decision_path": str(DECISION), "resources": resources,
            "feature_manifest_sha256": digest(manifest_path), "random_read_studies": 128,
            "random_read_seed": 20261001, "timed_batch_attempts": [0, 42, 100, 400, 1000, 1499],
            "warmup_batch_attempt": 0, "draws_per_source": 2, "batch_order_seed": 20260912,
            "weak_seed_offset": 34034034, "successful_updates_per_fold": 1500,
            "initialization_seed": 20261001, "layer4_learning_rate": 1e-5, "head_learning_rate": 1e-4,
            "frozen_logit_parity_tolerance": 1e-6, "cost_allowance": 2,
            "prospective_fit_wall_limit_seconds": 21600, "memory_admission_ceiling_bytes": 60 * 1024**3,
            "runtime_versions": {name: importlib.metadata.version(name) for name in ("numpy", "torch")},
            "frozen_files": [{"path": str(path), "sha256": digest(path)} for path in inputs],
            "fitted_probe_states_discarded": True, "held_out_predictions": False,
            "cold_page_cache_claimed": False, "full_experiment_fitting": False,
            "candidate_promotion": False, "competition_submission": False}
    write(ROOT / "plan.json", plan)
    checksum = digest(ROOT / "plan.json")
    write(DECISION, {"schema_version": "rsna-campaign-autonomous-decision-v1", "decision": "approved",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_made_by": "campaign agent under standing owner authorization", "authorization_sha256": digest(AUTH),
        "plan_sha256": checksum, "resources": resources,
        "basis": "Measure actual complete activation loading/memory and real matched training batches before separately admitting the fixed six-run experiment. Probe weights are discarded and held-out predictions are not computed.",
        "new_exact_owner_numeric_approval_claimed": False, "new_gpu_request": False,
        "gpu_ledger_cap_changed": False, "candidate_promotion": False})
    launcher = ROOT / "run.sbatch"
    with launcher.open("x") as handle:
        handle.write(f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket58-prefix-cost
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
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
{PROJECT}/.venv/bin/python {snapshot}/cost.py run {ROOT}/plan.json {checksum}
''')
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    print(json.dumps({"plan_sha256": checksum, "launcher": str(launcher), "launched": False}))


def run(plan_path, checksum):
    started = time.monotonic()
    if digest(plan_path) != checksum:
        raise ValueError("Exact prospective cost request changed")
    plan = read(plan_path)
    root = Path(plan["run_root"])
    decision = read(plan["decision_path"])
    if decision["decision"] != "approved" or decision["plan_sha256"] != checksum or decision["resources"] != plan["resources"]:
        raise ValueError("Exact cost resource decision differs")
    allocation = subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], text=True)
    fields = dict(token.split("=", 1) for token in allocation.split() if "=" in token)
    expected = {"JobState": "RUNNING", "Partition": "cluster1-cpu", "NumNodes": "1", "NumCPUs": "16",
                "NumTasks": "1", "MinMemoryNode": "64G", "TimeLimit": "00:30:00", "OverSubscribe": "OK",
                "Requeue": "0", "Restarts": "0"}
    if any(fields.get(key) != value for key, value in expected.items()) or "gpu" in fields.get("TRES", "").lower():
        raise ValueError("Actual bounded CPU cost allocation differs")
    (root / "running-allocation.txt").write_text(allocation)
    watchdog = threading.Timer(plan["resources"]["code_watchdog_seconds"], lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    for row in plan["frozen_files"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("Frozen actual-cohort cost input changed")
    for name, version in plan["runtime_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Fixed cost runtime changed")

    import numpy as np
    import torch
    from rsna_knee import ticket36 as t36
    from rsna_knee.folds import load_fold_manifest
    from rsna_knee.model import ModelConfig, build_study_model
    from rsna_knee.prefix_cache import load_prefix_entry
    from rsna_knee.prefix_adaptation import prefix_logits
    from rsna_knee.ticket34 import admit_ticket30_fold_supervision
    from rsna_knee.training import _array_sha256, _batch_indices, masked_bce_loss, seed_everything

    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise ValueError("Actual cohort cost probe is CPU only")
    prefix_root = Path(plan["prefix_root"])
    manifest = read(prefix_root / "features/feature-manifest.json")
    extraction = read(prefix_root / "plan.json")
    rows = sorted(manifest["features"], key=lambda row: row["study_instance_uid"])
    def load(row):
        return load_prefix_entry(prefix_root / "features" / row["feature_file"],
            expected_sha256=row["feature_file_sha256"], expected_study_uid=row["study_instance_uid"],
            expected_identity_sha256=manifest["feature_identity_sha256"],
            expected_source_identity_sha256=row["source_identity_sha256"],
            expected_plan_sha256=manifest["extraction_plan_sha256"])
    random_indices = np.random.default_rng(plan["random_read_seed"]).choice(len(rows),
        size=plan["random_read_studies"], replace=False)
    random_started = time.monotonic()
    random_bytes = 0
    for index in random_indices:
        entry = load(rows[index])
        random_bytes += rows[index]["file_bytes"]
        del entry
    random_seconds = time.monotonic() - random_started
    bank_started = time.monotonic()
    entries = []
    for index, row in enumerate(rows):
        entries.append(load(row))
        if (index + 1) % 256 == 0:
            print(json.dumps({"loaded_studies": index + 1,
                              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}), flush=True)
    load_seconds = time.monotonic() - bank_started
    payload_bytes = sum(entry.prefix.nbytes for entry in entries)
    if payload_bytes != 37586239488:
        raise ValueError("Full actual activation bank differs from the admitted payload")
    uids = [row["study_instance_uid"] for row in rows]
    index = {uid: position for position, uid in enumerate(uids)}
    parent = Path(plan["parent_root"])
    with np.load(parent / "pooled-cohort.npz", allow_pickle=False) as value:
        pooled = torch.from_numpy(value["pooled"].copy())
        if value["study_uids"].astype(str).tolist() != uids:
            raise ValueError("Matched original pooled bank order differs")
    pooled_proof = read(parent / "pooled-cohort-evidence.json")
    if _array_sha256(pooled.numpy()) != pooled_proof["pooled_array_sha256"]:
        raise ValueError("Actual control pooled arrays changed")
    folds = read(plan["cohort_parent_plan"])["folds"]
    fold_manifest = load_fold_manifest(t36.FOLD_MANIFEST_PATH)
    gold = {member.study_instance_uid: member for member in fold_manifest.gold_memberships}
    weights = torch.load(extraction["checkpoint_path"], map_location="cpu", weights_only=True)
    results = []
    for fold in folds:
        training = fold["gold_training_study_uids"]
        reports = fold["weak_training_study_uids"]
        heldout = set(fold_manifest.held_out_study_uids(fold["fold"]))
        if set(training + reports) & heldout:
            raise ValueError("Held-out study entered the cost-update batches")
        admission = admit_ticket30_fold_supervision(fold_manifest, fold["fold"], t36.TICKET30_ARTIFACT,
            expected_artifact_sha256=t36.TICKET30_AUDITED_ARTIFACT_SHA256).for_image_studies(uids)
        if admission.to_public_dict() != fold["supervision_admission"]:
            raise ValueError("Original report supervision admission differs")
        targets = torch.zeros((len(uids), 12))
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for uid in training + reports:
            values = gold[uid].reference_labels if uid in gold else admission.by_study[uid].targets
            for column, value in enumerate(values):
                if value is not None:
                    targets[index[uid], column] = float(value)
                    mask[index[uid], column] = True
        def batch(attempt):
            return [index[pool[item]] for pool, seed in (
                (training, plan["batch_order_seed"]),
                (reports, plan["batch_order_seed"] + plan["weak_seed_offset"]))
                for item in _batch_indices(len(pool), plan["draws_per_source"], attempt, seed)]
        seed_everything(plan["initialization_seed"])
        config = {**extraction["model"], "model_id": "resnet18-layer4-finetune-v1"}
        model = build_study_model(ModelConfig.from_dict(config))
        model.load_pretrained_state(weights)
        checkpoint = torch.load(parent / f"report-supervision/fold-{fold['fold']}/checkpoints/final.pt",
                                map_location="cpu", weights_only=True)
        model.head.load_state_dict(checkpoint["head_state_dict"], strict=True)
        model.train()
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        maximum_error = 0.0
        with torch.no_grad():
            for attempt in plan["timed_batch_attempts"]:
                positions = batch(attempt)
                error = float((prefix_logits(model, [entries[i] for i in positions])
                               - model.head(pooled[positions])).abs().max())
                maximum_error = max(maximum_error, error)
        if maximum_error > plan["frozen_logit_parity_tolerance"]:
            raise ValueError("Original pooled control and frozen prefix logits differ")
        optimizer = torch.optim.Adam([
            {"params": model.encoder.layer4.parameters(), "lr": plan["layer4_learning_rate"]},
            {"params": model.head.parameters(), "lr": plan["head_learning_rate"]}])
        measurements = []
        for number, attempt in enumerate([plan["warmup_batch_attempt"]] + plan["timed_batch_attempts"]):
            positions = batch(attempt)
            if any(uids[i] in heldout for i in positions):
                raise ValueError("Held-out inputs entered a profiling update")
            update_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss = masked_bce_loss(prefix_logits(model, [entries[i] for i in positions]), targets[positions], mask[positions])
            if loss is None or not torch.isfinite(loss):
                raise ValueError("Missing or nonfinite profiling supervision")
            loss.backward()
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters() if parameter.requires_grad):
                raise ValueError("Missing or nonfinite actual adaptation gradient")
            optimizer.step()
            if number:
                measurements.append({"batch_attempt": attempt, "seconds": time.monotonic() - update_started,
                                     "valid_slices": sum(len(entries[i].prefix) for i in positions),
                                     "supervised_entries": int(mask[positions].sum())})
        final = model.state_dict()
        frozen_error = max(float((value - initial[key]).abs().max()) for key, value in final.items()
                           if not key.startswith(("encoder.layer4.", "head.")))
        buffer_error = max(float((value - initial[key]).abs().max()) for key, value in final.items()
                           if key.endswith(("running_mean", "running_var", "num_batches_tracked")))
        changes = {group: max(float((value - initial[key]).abs().max()) for key, value in final.items()
                             if key.startswith(group)) for group in ("encoder.layer4.", "head.")}
        if frozen_error != 0 or buffer_error != 0 or any(value <= 0 for value in changes.values()):
            raise ValueError("Actual probe adaptation scope or normalization differs")
        total_slices = sum(sum(len(entries[i].prefix) for i in batch(attempt))
                           for attempt in range(plan["successful_updates_per_fold"]))
        seconds_per_slice = max(value["seconds"] / value["valid_slices"] for value in measurements)
        projected = plan["cost_allowance"] * total_slices * seconds_per_slice
        results.append({"fold": fold["fold"], "measurements": measurements,
                        "maximum_frozen_logit_error": maximum_error,
                        "actual_registered_training_slice_exposures": total_slices,
                        "projected_training_seconds_with_allowance": projected,
                        "frozen_parameter_max_abs_change": frozen_error,
                        "normalization_buffer_max_abs_change": buffer_error,
                        "parameter_max_abs_changes": changes})
        del optimizer, model, initial, final, targets, mask
        print(json.dumps({"profiled_fold": fold["fold"], "projected_training_seconds": projected}), flush=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    # Extra half-hour covers controls, checkpointing, final/native replay and
    # evaluation; measured prefix-load cost is doubled in addition to training.
    projected_total = 2 * load_seconds + sum(row["projected_training_seconds_with_allowance"] for row in results) + 1800
    eligible = projected_total < plan["prospective_fit_wall_limit_seconds"] and peak < plan["memory_admission_ceiling_bytes"]
    write(root / "cost-result.json", {"schema_version": "rsna-ticket58-actual-prefix-cost-result-v1",
        "status": "COMPLETE", "observed_at_utc": datetime.now(timezone.utc).isoformat(), "job_id": os.environ["SLURM_JOB_ID"],
        "plan_sha256": checksum, "random_read_studies": len(random_indices), "random_read_bytes": random_bytes,
        "random_read_seconds": random_seconds, "full_loaded_studies": len(entries), "full_payload_bytes": payload_bytes,
        "full_verified_load_seconds": load_seconds, "peak_process_rss_bytes": peak, "folds": results,
        "projected_complete_fit_seconds_with_allowance": projected_total,
        "cost_disposition": "eligible_for_separate_fixed_fit_decision" if eligible else "cost_gate_failed",
        "cost_limits": "Warm/shared filesystem page cache; no cold-read or contention guarantee. Slice scaling extrapolates six measured batches per fold. Conservative cost includes2x loading/training and1800s for controls, checkpoints and final/native replay.",
        "fitted_probe_states_discarded": True, "held_out_predictions_computed": False,
        "full_experiment_fitting": False, "candidate_promotion": False, "competition_submission": False,
        "gpu_count": 0, "external_spend_usd": 0, "elapsed_seconds": time.monotonic() - started})
    watchdog.cancel()
    print(json.dumps({"status": "COMPLETE", "cost_gate_passed": eligible,
                      "projected_complete_seconds": projected_total, "peak_rss_bytes": peak}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        run(Path(sys.argv[2]), sys.argv[3])
