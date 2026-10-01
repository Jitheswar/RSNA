"""Independently check fixed adaptation checkpoints, exposures and native MRI."""

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
ROOT = PROJECT / "artifacts/private/ticket58-report-layer4-adaptation-v1"
OUTPUT = ROOT / "independent-verification"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket58-independent-native-replay-20261001.json"
ARMS = ("head-continuation", "layer4-adaptation")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def terminal(root, checksum, *, time_limit):
    accepted = read(root / "accepted-launch.json")
    path = root / f"controller-{accepted['job_id']}/terminal-controller.txt"
    controller = path.read_text()
    observation = read(path.with_name("terminal-observation.json"))
    fields = dict(token.split("=", 1) for token in controller.split() if "=" in token)
    expected = {"JobId": str(accepted["job_id"]), "JobState": "COMPLETED", "ExitCode": "0:0",
        "Partition": "cluster1-cpu", "NumNodes": "1", "NumCPUs": "16", "NumTasks": "1",
        "MinMemoryNode": "64G", "TimeLimit": time_limit, "OverSubscribe": "OK", "Requeue": "0", "Restarts": "0"}
    require(accepted["plan_sha256"] == checksum
        and str(observation["job_id"]) == str(accepted["job_id"])
        and observation["state"] == "COMPLETED" and observation["returncode"] == 0
        and all(fields.get(key) == value for key, value in expected.items())
        and "gpu" not in fields.get("TRES", "").lower(), "Matching successful CPU terminal evidence is required")
    seconds = (datetime.fromisoformat(fields["EndTime"]) - datetime.fromisoformat(fields["StartTime"])).total_seconds()
    return accepted, path, seconds


def prepare():
    require(not OUTPUT.exists() and not DECISION.exists(), "Immutable independent replay request already exists")
    plan = read(ROOT / "plan.json")
    checksum = digest(ROOT / "plan.json")
    accepted, controller, _ = terminal(ROOT, checksum, time_limit="03:00:00")
    analysis = read(ROOT / "analysis.json")
    require(analysis["status"] == "COMPLETE" and analysis["plan_sha256"] == checksum
        and analysis["job_id"] == str(accepted["job_id"]) and len(analysis["run_records"]) == 6,
        "All six fixed training runs must complete before independent replay")
    OUTPUT.mkdir(mode=0o700)
    snapshot = OUTPUT / "source-snapshot"
    shutil.copytree(Path(plan["source_snapshot"]) / "src", snapshot / "src")
    script = snapshot / Path(__file__).name
    shutil.copyfile(__file__, script)
    inputs = [AUTH, ROOT / "plan.json", ROOT / "analysis.json", ROOT / "accepted-launch.json",
        controller, controller.with_name("terminal-observation.json")]
    for arm in ARMS:
        for fold in range(3):
            inputs.extend((ROOT / arm / f"fold-{fold}/run.json", ROOT / arm / f"fold-{fold}/checkpoints/final.pt"))
    inputs.extend(path for path in snapshot.rglob("*") if path.is_file() and path.suffix != ".pyc")
    resources = {"partition": "cluster1-cpu", "nodes": 1, "tasks": 1, "cpus_per_task": 16,
        "memory_gib": 64, "walltime_seconds": 1800, "code_watchdog_seconds": 1740,
        "maximum_launch_attempts": 1, "maximum_concurrent_jobs": 1, "gpu_count": 0,
        "exclusive": False, "requeue": False, "external_spend_usd": 0}
    request = {"schema_version": "rsna-ticket58-independent-native-replay-plan-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_root": str(OUTPUT),
        "fit_root": str(ROOT), "fit_plan_sha256": checksum, "decision_path": str(DECISION),
        "resources": resources, "source_snapshot": str(snapshot),
        "inference": plan["inference"], "runtime_versions": plan["runtime_versions"],
        "frozen_files": [{"path": str(path), "sha256": digest(path)} for path in inputs],
        "network_during_execution": False, "fitting": False, "candidate_promotion": False,
        "competition_submission": False, "final_selection": False}
    write(OUTPUT / "plan.json", request)
    request_sha = digest(OUTPUT / "plan.json")
    write(DECISION, {"schema_version": "rsna-campaign-autonomous-decision-v1", "decision": "approved",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": request_sha,
        "resources": resources, "authorization_sha256": digest(AUTH),
        "decision_made_by": "campaign agent under standing owner authorization",
        "basis": "Independently reload all six fixed final checkpoints, reconstruct training membership/masks/exposures and verify actual native MRI predictions and registered controls/metrics before candidate disposition.",
        "new_exact_owner_numeric_approval_claimed": False, "new_gpu_request": False,
        "gpu_ledger_cap_changed": False, "candidate_promotion": False})
    with (OUTPUT / "run.sbatch").open("x") as handle:
        handle.write(f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket58-native-verify
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output={OUTPUT}/slurm-%j.out
#SBATCH --error={OUTPUT}/slurm-%j.err
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
{PROJECT}/.venv/bin/python {script} run {OUTPUT}/plan.json {request_sha}
''')
    subprocess.run(["bash", "-n", str(OUTPUT / "run.sbatch")], check=True)
    print(json.dumps({"plan_sha256": request_sha, "launcher": str(OUTPUT / "run.sbatch"), "launched": False}))


def maximum_error(actual, expected):
    import numpy as np
    require(set(actual) == set(expected), "Independently replayed prediction membership differs")
    require(all(np.asarray(actual[uid]).shape == np.asarray(expected[uid]).shape == (12,)
        and np.isfinite(actual[uid]).all() and np.isfinite(expected[uid]).all() for uid in actual),
        "Finite twelve-target prediction shape is required")
    return max(float(np.max(np.abs(np.asarray(actual[uid]) - np.asarray(expected[uid])))) for uid in actual)


def run(request_path, request_sha):
    started = time.monotonic()
    require(digest(request_path) == request_sha, "Frozen independent replay request changed")
    request = read(request_path)
    output, root = Path(request["run_root"]), Path(request["fit_root"])
    decision = read(request["decision_path"])
    require(decision["decision"] == "approved" and decision["plan_sha256"] == request_sha
        and decision["resources"] == request["resources"], "Exact independent resource decision differs")
    allocation = subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], text=True)
    (output / "running-allocation.txt").write_text(allocation)
    fields = dict(token.split("=", 1) for token in allocation.split() if "=" in token)
    expected = {"JobId": os.environ["SLURM_JOB_ID"], "JobState": "RUNNING", "Partition": "cluster1-cpu",
        "NumNodes": "1", "NumCPUs": "16", "NumTasks": "1", "MinMemoryNode": "64G",
        "TimeLimit": "00:30:00", "OverSubscribe": "OK", "Requeue": "0", "Restarts": "0"}
    require(all(fields.get(key) == value for key, value in expected.items())
        and "gpu" not in fields.get("TRES", "").lower(), "Actual bounded independent CPU allocation differs")
    watchdog = threading.Timer(request["resources"]["code_watchdog_seconds"], lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    plan = read(root / "plan.json")
    checksum = digest(root / "plan.json")
    require(checksum == request["fit_plan_sha256"] and Path(plan["run_root"]) == root,
        "Exact completed fit identity differs")
    for row in request["frozen_files"] + plan["frozen_files"]:
        require(digest(row["path"]) == row["sha256"], "Frozen scientific input, checkpoint or verifier changed")
    require(all(importlib.metadata.version(name) == version for name, version in plan["runtime_versions"].items()),
        "Frozen independent replay runtime differs")
    accepted, terminal_path, allocated_fit_seconds = terminal(root, checksum, time_limit="03:00:00")
    require(plan["optimization"] == {"successful_steps": 1500, "checkpoint_interval_steps": 500,
        "draws_per_source": 2, "batch_order_seed": 20260912, "weak_seed_offset": 34034034,
        "initialization_seed": 20261001, "head_learning_rate": 1e-4, "layer4_learning_rate": 1e-5,
        "optimizer": "Adam", "augmentation": False, "normalization_buffers_fixed": True,
        "optimizer_restarted_from_warm_weights": True}
        and plan["inference"] == request["inference"] == {"threads": 16, "batch_size": 2, "maximum_absolute_error": 1e-6}
        and plan["analysis"]["bootstrap_resamples"] == 2000 and plan["analysis"]["seed"] == 20261001,
        "Registered optimization, inference or comparison settings differ")

    import numpy as np
    import torch
    from rsna_knee import ticket36 as t36
    from rsna_knee.folds import load_fold_manifest
    from rsna_knee.frozen_head_experiment import sequence_digest
    from rsna_knee.image_model_comparison import compare_image_model_predictions
    from rsna_knee.model import ModelConfig, build_study_model
    from rsna_knee.prefix_adaptation import prefix_logits
    from rsna_knee.prefix_cache import load_prefix_entry
    from rsna_knee.preprocessing import pack_study, prepare_study
    from rsna_knee.real_inference import iter_real_studies, load_integration_manifest
    from rsna_knee.ticket34 import admit_ticket30_fold_supervision
    from rsna_knee.training import _array_sha256, _batch_indices, _model_state_sha256, seed_everything

    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_available() and torch.version.cuda is None, "Admitted CPU-only replay is required")
    prefix_root, parent = Path(plan["prefix_root"]), Path(plan["parent_root"])
    prefix_plan = read(prefix_root / "plan.json")
    prefix_manifest = read(prefix_root / "features/feature-manifest.json")
    require(digest(prefix_root / "features/feature-manifest.json") == plan["feature_manifest_sha256"]
        and prefix_manifest["all_native_encoder_features_verified"] and prefix_manifest["study_count"] == 4228,
        "Complete independent activation admission differs")
    rows = sorted(prefix_manifest["features"], key=lambda row: row["study_instance_uid"])
    uids = [row["study_instance_uid"] for row in rows]
    require(len(uids) == len(set(uids)) == 4228, "Complete admitted study mapping differs")
    index = {uid: i for i, uid in enumerate(uids)}
    by_uid = {row["study_instance_uid"]: row for row in rows}
    manifest = load_fold_manifest(plan["fold_manifest_path"])
    require(manifest.manifest_checksum == plan["fold_manifest_checksum"] and not manifest.superseded,
        "Corrected three-fold identity differs")
    gold = {member.study_instance_uid: member for member in manifest.gold_memberships}
    require(len(gold) == 58 and set(gold) <= set(uids), "Complete reference membership differs")
    with np.load(parent / "pooled-cohort.npz", allow_pickle=False) as bank:
        require(bank["study_uids"].astype(str).tolist() == uids, "Pooled control study order differs")
        pooled_values = bank["pooled"].copy()
    require(pooled_values.shape == (4228, 2076) and np.isfinite(pooled_values).all()
        and _array_sha256(pooled_values) == read(parent / "pooled-cohort-evidence.json")["pooled_array_sha256"],
        "Admitted pooled control arrays differ")
    pooled = torch.from_numpy(pooled_values)
    zero_pool = pooled.clone().reshape(-1, 4, 519)
    zero_pool[:, :, :512] = 0
    zero_pool = zero_pool.reshape(-1, 2076)

    # Load original DICOMs, rather than the prepared-series or feature caches.
    integration = load_integration_manifest(prefix_plan["integration_path"])
    require(integration.manifest_checksum == prefix_plan["integration_manifest_checksum"], "Native integration differs")
    archive = Path(prefix_plan["source_archive_path"])
    stat = archive.stat()
    require(stat.st_size == prefix_plan["source_archive_bytes"] and stat.st_mtime_ns == prefix_plan["source_archive_mtime_ns"],
        "Original MRI archive stat identity changed")
    packed, entries, source_audit = {}, {}, []
    for loaded in iter_real_studies(integration, archive, require_complete_series=True, study_uids=sorted(gold)):
        study = loaded.study
        uid = study.study_instance_uid
        require(uid in gold and uid not in packed, "Native reference MRI membership differs")
        prepared = prepare_study(study, integration.preprocess_config)
        require(not prepared.failures and prepared.series, "Actual native MRI preparation failed")
        value = pack_study(study, prepared.series, integration.preprocess_config, packing_config=integration.packing_config)
        source_identity = t36._json_sha256({"archive_fingerprint": integration.archive_fingerprint,
            "study_instance_uid": uid, "series": [{"series_instance_uid": item.series_instance_uid,
                "entry_names": list(item.source_entry_names),
                "source_slice_uids": [part.sop_instance_uid for part in item.slices]}
                for item in sorted(study.series, key=lambda item: item.series_instance_uid)]})
        row = by_uid[uid]
        require(row["source_kind"] == "frozen-rsna-archive" and row["source_identity_sha256"] == source_identity,
            "Native source identity differs from the admitted cache")
        entry = load_prefix_entry(prefix_root / "features" / row["feature_file"],
            expected_sha256=row["feature_file_sha256"], expected_study_uid=uid,
            expected_identity_sha256=prefix_manifest["feature_identity_sha256"],
            expected_source_identity_sha256=source_identity, expected_plan_sha256=prefix_manifest["extraction_plan_sha256"])
        require(all(np.array_equal(getattr(value, key), getattr(entry, key))
            for key in ("slice_mask", "series_mask", "series_descriptors"))
            and [list(indices) for indices in value.sampled_source_indices] == entry.metadata["sampled_source_indices"],
            "Actual native packing, selected slices, masks or descriptors differ")
        packed[uid], entries[uid] = value, entry
        source_audit.append({"study_instance_uid": uid, "source_identity_sha256": source_identity,
            "valid_slices": len(entry.prefix), "prefix_file_sha256": row["feature_file_sha256"]})
    require(set(packed) == set(gold), "All 58 actual reference MRI studies must be read")
    print(json.dumps({"actual_native_studies_loaded": len(packed)}), flush=True)
    weights = torch.load(prefix_plan["checkpoint_path"], map_location="cpu", weights_only=True)
    folds = read(plan["cohort_parent_plan"])["folds"]
    opt = plan["optimization"]
    cached, native, ablated, native_ablated = [{arm: {} for arm in ARMS} for _ in range(4)]
    parent_predictions, native_parent, records, audits = {}, {}, [], []

    def build(head):
        seed_everything(opt["initialization_seed"])
        model = build_study_model(ModelConfig.from_dict(plan["model"]))
        model.load_pretrained_state(weights)
        model.head.load_state_dict(head, strict=True)
        return model.requires_grad_(False).eval()

    def predict(model, members, route, *, zero_images=False):
        values = {}
        with torch.inference_mode():
            for offset in range(0, len(members), 2):
                batch = members[offset:offset + 2]
                if route == "native":
                    slices, series, desc = [torch.from_numpy(np.stack([getattr(packed[uid], key) for uid in batch]))
                        for key in ("slice_mask", "series_mask", "series_descriptors")]
                    if zero_images:
                        logits = model.head(model._pool_features(torch.zeros((len(batch), 16, 8, 512)), slices, series, desc))
                    else:
                        images = torch.from_numpy(np.stack([packed[uid].images for uid in batch]))
                        logits = model(images, slices, series, desc)
                elif route == "prefix":
                    logits = prefix_logits(model, [entries[uid] for uid in batch], zero_images=zero_images)
                else:
                    logits = model.head((zero_pool if zero_images else pooled)[[index[uid] for uid in batch]])
                scores = torch.sigmoid(logits).numpy()
                require(np.isfinite(scores).all(), "Nonfinite independently replayed score")
                values.update({uid: value.tolist() for uid, value in zip(batch, scores, strict=True)})
        return values

    for fold in folds:
        fold_id = fold["fold"]
        training, reports, heldout = [fold[key] for key in
            ("gold_training_study_uids", "weak_training_study_uids", "gold_heldout_study_uids")]
        require(set(heldout) == {uid for uid, member in gold.items() if member.fold == fold_id}
            and set(training) == set(gold) - set(heldout)
            and not set(training + reports) & set(manifest.held_out_study_uids(fold_id)),
            "Training, report or held-out partition differs")
        admission = admit_ticket30_fold_supervision(manifest, fold_id, t36.TICKET30_ARTIFACT,
            expected_artifact_sha256=t36.TICKET30_AUDITED_ARTIFACT_SHA256).for_image_studies(uids)
        require(admission.to_public_dict() == fold["supervision_admission"], "Report-label source admission differs")
        targets = torch.zeros((len(uids), 12))
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for uid in training + reports:
            values = gold[uid].reference_labels if uid in gold else admission.by_study[uid].targets
            for column, value in enumerate(values):
                if value is not None:
                    targets[index[uid], column] = float(value)
                    mask[index[uid], column] = True
        require(not mask[[index[uid] for uid in heldout]].any(), "Held-out target entries entered fitting masks")
        parent_path = parent / f"report-supervision/fold-{fold_id}/checkpoints/final.pt"
        parent_run = read(parent_path.parents[1] / "run.json")
        parent_checkpoint = torch.load(parent_path, map_location="cpu", weights_only=True)
        warm = build(parent_checkpoint["head_state_dict"])
        require(digest(parent_path) == parent_run["checkpoint_sha256"]
            and parent_run["held_out_study_uids"] == heldout,
            "Same-fold independently admitted warm parent differs")
        parent_native = predict(warm, heldout, "native")
        parent_error = maximum_error(parent_native, parent_run["predictions"])
        require(parent_error <= 1e-6, "Actual native warm-parent prediction replay differs")
        parent_predictions.update(parent_run["predictions"])
        native_parent.update(parent_native)
        initial = {key: value.clone() for key, value in warm.state_dict().items()}
        initial_hashes = {"initial_model_state_sha256": _model_state_sha256(warm),
            "initial_head_state_sha256": _model_state_sha256(warm.head),
            "initial_encoder_state_sha256": _model_state_sha256(warm.encoder)}
        expected_sources = {}
        for kind, members, seed in (("reference", training, opt["batch_order_seed"]),
            ("weak", reports, opt["batch_order_seed"] + opt["weak_seed_offset"])):
            sequence = [members[position] for step in range(1500)
                for position in _batch_indices(len(members), 2, step, seed)]
            expected_sources[kind] = {"sequence_sha256": sequence_digest(sequence),
                "distinct_studies": len(set(sequence)), "sample_exposures": len(sequence),
                "supervised_entry_exposures": sum(int(mask[index[uid]].sum()) for uid in sequence)}
        for arm in ARMS:
            path = root / arm / f"fold-{fold_id}/run.json"
            record = read(path)
            checkpoint_path = path.parent / "checkpoints/final.pt"
            require(digest(checkpoint_path) == record["checkpoint_sha256"], "Final checkpoint digest differs")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            identity = {"plan_sha256": checksum, "fold": fold_id, "arm": arm, **initial_hashes,
                "parent_checkpoint_sha256": digest(parent_path), "feature_manifest_sha256": plan["feature_manifest_sha256"],
                "fold_manifest_checksum": manifest.manifest_checksum,
                "target_array_sha256": _array_sha256(targets.numpy()), "target_mask_sha256": _array_sha256(mask.numpy())}
            require(record["status"] == "COMPLETE" and record["identity"] == checkpoint["identity"] == identity
                and checkpoint["counts"] == record["counts"] and checkpoint["checkpoint_kind"] == "final_selected"
                and checkpoint["model_config"] == plan["model"]
                and record["training_study_uids"] == training + reports
                and record["held_out_study_uids"] == heldout
                and record["held_out_inputs_excluded_from_fitting"] and record["held_out_target_entries_masked"],
                "Independent checkpoint, initialization, label or membership identity differs")
            counts = record["counts"]
            require(counts["successful_optimizer_steps"] == counts["attempted_batches"] == 1500
                and counts["skipped_batches"] == 0 and len(counts["loss_history"]) == 1500
                and np.isfinite(counts["loss_history"]).all()
                and counts["distinct_weak_studies_exposed"] == expected_sources["weak"]["distinct_studies"]
                and counts["weak_training_pool_count"] == len(reports), "Fixed update budget or report coverage differs")
            for kind, key in (("reference", "gold_exposure_sequence_sha256"), ("weak", "weak_exposure_sequence_sha256")):
                expected_source = expected_sources[kind]
                require(counts[key] == expected_source["sequence_sha256"]
                    and counts["exposures_by_source"][kind] == {name: expected_source[name]
                        for name in ("sample_exposures", "supervised_entry_exposures")},
                    "Actual ordered exposure or supervised-entry counts differ")
            optimizer = checkpoint["optimizer_state_dict"]
            require([group["lr"] for group in optimizer["param_groups"]]
                == ([1e-5, 1e-4] if arm == ARMS[1] else [1e-4])
                and len(optimizer["state"]) == sum(len(group["params"]) for group in optimizer["param_groups"])
                and all(float(state["step"]) == 1500 and torch.isfinite(state["exp_avg"]).all()
                    and torch.isfinite(state["exp_avg_sq"]).all() for state in optimizer["state"].values()),
                "Final Adam learning rates, moments or step budget differ")
            model = build(parent_checkpoint["head_state_dict"])
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            current = model.state_dict()
            require(all(torch.isfinite(value).all() for value in current.values()), "Nonfinite final model tensor")
            changes = {group: max(float((value - initial[key]).abs().max())
                for key, value in current.items() if key.startswith(group)) for group in ("encoder.layer4.", "head.")}
            frozen_error = max(float((value - initial[key]).abs().max()) for key, value in current.items()
                if not key.startswith(("encoder.layer4.", "head.")))
            buffer_error = max(float((value - initial[key]).abs().max()) for key, value in current.items()
                if key.endswith(("running_mean", "running_var", "num_batches_tracked")))
            require(frozen_error == buffer_error == 0 and changes["head."] > 0
                and (changes["encoder.layer4."] > 0 if arm == ARMS[1] else changes["encoder.layer4."] == 0)
                and record["frozen_parameter_max_abs_change"] == frozen_error
                and record["normalization_buffer_max_abs_change"] == buffer_error
                and record["parameter_max_abs_changes"] == changes, "Independent trainable scope or fixed buffers differ")
            route = "prefix" if arm == ARMS[1] else "pooled"
            full = predict(model, heldout, route)
            zero = predict(model, heldout, route, zero_images=True)
            native_full = predict(model, heldout, "native")
            native_zero = predict(model, heldout, "native", zero_images=True)
            errors = {"cached_checkpoint_error": maximum_error(full, record["predictions"]),
                "zero_control_checkpoint_error": maximum_error(zero, record["zero_image_predictions"]),
                "native_prediction_error": maximum_error(native_full, record["predictions"]),
                "native_zero_control_error": maximum_error(native_zero, record["zero_image_predictions"]),
                "native_warm_parent_error": parent_error}
            require(max(errors.values()) <= 1e-6, "Independent cached or actual native final replay exceeds 1e-6")
            require(not set(cached[arm]) & set(full), "Independent held-out fold predictions overlap")
            cached[arm].update(full)
            ablated[arm].update(zero)
            native[arm].update(native_full)
            native_ablated[arm].update(native_zero)
            records.append({"fold": fold_id, "arm": arm, "path": str(path), "sha256": digest(path)})
            audits.append({"fold": fold_id, "arm": arm, "checkpoint_sha256": digest(checkpoint_path),
                **errors, "initial_state_and_targets_verified": True, "ordered_exposures_verified": True,
                "frozen_parameter_max_abs_change": frozen_error, "normalization_buffer_max_abs_change": buffer_error,
                "parameter_max_abs_changes": changes, "inference_threads": 16, "inference_batch_size": 2})
            print(json.dumps({"fold": fold_id, "arm": arm, **errors}), flush=True)
            del model, checkpoint

    def comparisons_for(values, zero, warm_values):
        pairs = {"adaptation_minus_matched_head_continuation": (values[ARMS[0]], values[ARMS[1]]),
            "head_continuation_minus_warm_parent": (warm_values, values[ARMS[0]]),
            "adaptation_minus_warm_parent": (warm_values, values[ARMS[1]]),
            "head_continuation_minus_zero_image": (zero[ARMS[0]], values[ARMS[0]]),
            "adaptation_minus_zero_image": (zero[ARMS[1]], values[ARMS[1]])}
        return {name: compare_image_model_predictions(base, candidate, manifest,
            bootstrap_resamples=2000, bootstrap_seed=20261001) for name, (base, candidate) in pairs.items()}

    comparisons = comparisons_for(cached, ablated, parent_predictions)
    native_comparisons = comparisons_for(native, native_ablated, native_parent)
    require(native_comparisons == comparisons, "Actual native prediction ranks change registered comparison results")
    def passes(name):
        value = comparisons[name]
        lower = value["paired_study_bootstrap"]["macro_auc"]["lower"]
        return lower is not None and lower > 0 and all(row["delta_macro_auc"] is not None
            and row["delta_macro_auc"] > 0 for row in value["delta"]["folds"])
    disposition = ("provisional_adaptation_review_pending_independent_native_replay"
        if passes(plan["analysis"]["primary"]) and passes("adaptation_minus_zero_image") else "inconclusive_no_promotion")
    analysis = read(root / "analysis.json")
    require(analysis["status"] == "COMPLETE" and analysis["plan_sha256"] == checksum
        and analysis["job_id"] == str(accepted["job_id"]) and analysis["run_records"] == records
        and analysis["comparisons"] == comparisons and analysis["disposition"] == disposition
        and not analysis["candidate_promotion"] and not analysis["competition_submission"]
        and analysis["development_cohort_reused"] and analysis["gpu_count"] == 0
        and analysis["external_spend_usd"] == 0, "Independent metrics, analysis or declared disposition differs")
    write(output / "validation.json", {"schema_version": "rsna-ticket58-independent-native-validation-v1",
        "status": "COMPLETE", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"], "plan_sha256": request_sha, "fit_plan_sha256": checksum,
        "fit_job_id": accepted["job_id"], "fit_terminal_sha256": digest(terminal_path),
        "analysis_sha256": digest(root / "analysis.json"), "allocated_fit_cpu_node_seconds": allocated_fit_seconds,
        "frozen_files_verified": len(request["frozen_files"]) + len(plan["frozen_files"]),
        "runs_verified": len(audits), "final_predictions_verified": sum(len(values) for values in native.values()),
        "native_warm_parent_predictions_verified": len(native_parent), "actual_native_studies_verified": len(packed),
        "native_mri_sources": source_audit, "independent_replays": audits,
        "maximum_prediction_error": max(row[key] for row in audits for key in row if key.endswith("error")),
        "actual_native_comparisons_identical": True, "comparisons_recomputed": list(comparisons),
        "training_exclusions_and_source_masks_verified": True, "native_final_replay_verified": True,
        "disposition": "provisional_adaptation_review" if disposition.startswith("provisional") else disposition,
        "candidate_promotion": False, "development_cohort_reused": True, "fitting": False,
        "gpu_count": 0, "external_spend_usd": 0, "elapsed_seconds": time.monotonic() - started,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
    watchdog.cancel()
    print(json.dumps({"status": "COMPLETE", "runs_verified": len(audits),
        "actual_native_studies_verified": len(packed), "disposition": disposition}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        run(Path(sys.argv[2]), sys.argv[3])
