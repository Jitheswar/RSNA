"""Matched long ResNet heads and one fixed blend on existing frozen features."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import resource

import numpy as np
import torch

from rsna_knee import ticket36 as t36
from rsna_knee import frozen_head_experiment as heads
from rsna_knee.feature_pooling import pool_frozen_features
from rsna_knee.folds import load_fold_manifest
from rsna_knee.image_model_comparison import compare_image_model_predictions
from rsna_knee.model import ModelConfig, build_study_model
from rsna_knee.ticket34 import admit_ticket30_fold_supervision
from rsna_knee.training import _array_sha256, _model_state_sha256, seed_everything
from ticket54_convnext_long_report_head import digest, read, write


PROJECT = t36.PROJECT_ROOT
REVISION = int(os.environ.get("RSNA_TICKET55_REVISION", "1"))
if REVISION not in (1, 2):
    raise ValueError("This stage admits the original request and one diagnosed recovery")
BASE_ROOT = PROJECT / "artifacts/private/ticket55-resnet-long-report-head"
ROOT = BASE_ROOT if REVISION == 1 else BASE_ROOT.with_name(BASE_ROOT.name + "-v2")
OLD = PROJECT / "artifacts/private/ticket44-full-weak-exposure"
CONVNEXT = PROJECT / "artifacts/private/ticket54-convnext-long-report-head"
PARENT = PROJECT / "artifacts/private/ticket50-convnext-supervision/plan-v2.json"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
DECISION = PROJECT / (".scratch/rsna-leaderboard/resource-approvals/ticket55-resnet-long-report-head-20261001"
                      + ("-v2" if REVISION == 2 else "") + ".json")


def weak_sequence_matches(current, historical):
    """Accept equivalent absent-report metadata only with zero exposures."""
    old_digest = historical["weak_exposure_sequence_sha256"]
    if old_digest is not None:
        return current["weak_exposure_sequence_sha256"] == old_digest
    return (historical["exposures_by_source"]["weak"] == {
                "sample_exposures": 0, "supervised_entry_exposures": 0}
            and current["exposures_by_source"]["weak"] == {
                "sample_exposures": 0, "supervised_entry_exposures": 0}
            and current["weak_exposure_sequence_sha256"] == heads.sequence_digest([]))


def prepare():
    if ROOT.exists() or DECISION.exists():
        raise FileExistsError("Immutable Ticket 55 request already exists")
    validation = read(CONVNEXT / "validation.json")
    if validation["status"] != "COMPLETE" or validation["final_predictions_verified"] != 116:
        raise ValueError("Complete independently verified long ConvNeXt comparison is required")
    parent = read(PARENT)
    original = read(OLD / "plan.json")
    features = read(t36.FEATURE_ROOT / "feature-manifest.json")
    if len(features["features"]) != 4228 or original["fold_manifest_checksum"] != parent["fold_manifest_checksum"]:
        raise ValueError("Original feature cohort or fold identity differs")
    inputs = [PARENT, OLD / "plan.json", CONVNEXT / "validation.json", CONVNEXT / "analysis.json",
              t36.FOLD_MANIFEST_PATH, t36.TICKET30_ARTIFACT, t36.FEATURE_ROOT / "feature-manifest.json",
              t36.WEIGHTS_PATH, AUTH]
    if REVISION == 2:
        diagnosis = read(BASE_ROOT / "failure-diagnosis.json")
        if (diagnosis["job_id"] != 7977 or diagnosis["scientific_changes"]
                or diagnosis["replay_tolerance_changed"]
                or not diagnosis["checks"]["initial_full_state_matches"]
                or not diagnosis["checks"]["gold_exposure_matches"]):
            raise ValueError("Exact operational failure diagnosis is required")
        inputs.extend((BASE_ROOT / "plan.json", BASE_ROOT / "failure-diagnosis.json"))
    for arm in ("reference-only", "report-supervision"):
        for fold in range(3):
            path = OLD / arm / f"fold-{fold}/run.json"
            record = read(path)
            checkpoint = path.parent / "checkpoints/final.pt"
            if (record["status"] != "completed_fixed_budget_training"
                    or record["counts"]["successful_optimizer_steps"] != 1500
                    or digest(checkpoint) != record["checkpoint"]["final_sha256"]):
                raise ValueError("Original ResNet duration comparator differs")
            inputs.extend((path, checkpoint, CONVNEXT / arm / f"fold-{fold}/run.json"))
    ROOT.mkdir(mode=0o700, parents=True)
    snapshot = ROOT / "source-snapshot"
    shutil.copytree(PROJECT / "src/rsna_knee", snapshot / "src/rsna_knee",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for script in (Path(__file__), PROJECT / "scripts/ticket54_convnext_long_report_head.py"):
        shutil.copyfile(script, snapshot / script.name)
    inputs.extend(path for path in snapshot.rglob("*") if path.is_file())
    resources = {"partition": "cluster1-cpu", "nodes": 1, "tasks": 1, "cpus_per_task": 16,
                 "memory_gib": 64, "walltime_seconds": 1800, "code_watchdog_seconds": 1200,
                 "gpu_count": 0, "exclusive": False, "maximum_concurrent_jobs": 1,
                 "maximum_launch_attempts": 1, "requeue": False, "external_spend_usd": 0}
    plan = {"schema_version": "rsna-ticket55-matched-long-head-plan-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_root": str(ROOT),
            "preparation_revision": REVISION, "stage_attempt_number": REVISION, "maximum_stage_attempts": 2,
            "source_snapshot": str(snapshot), "parent_plan": str(PARENT), "decision_path": str(DECISION),
            "authorization_sha256": digest(AUTH), "fold_manifest_checksum": parent["fold_manifest_checksum"],
            "original_resnet_configuration": original["training_configuration"],
            "optimization": {**parent["optimization"], "successful_steps": 7500, "checkpoint_interval_steps": 1500},
            "head": {"head_id": "resnet18-frozen-plane-pooled-mlp32-v1", "feature_width": 512,
                     "input_width": 2076, "hidden_width": 32, "targets": 12},
            "prefix_maximum_absolute_error": 1e-6,
            "analysis": {"bootstrap_resamples": 2000, "seed": 20261001, "fixed_convnext_weight": 0.5,
                         "provisional_review_gate": "Positive paired lower bound and positive directions in every fold, with image-signal controls; no automatic promotion.",
                         "reused_development_cohort": True},
            "resources": resources, "runtime_versions": {name: importlib.metadata.version(name) for name in ("numpy", "torch")},
            "frozen_files": [{"path": str(path), "sha256": digest(path)} for path in inputs],
            "feature_files": [{"path": str(t36.FEATURE_ROOT / row["feature_file"]),
                               "sha256": row["feature_file_sha256"]} for row in features["features"]],
            "original_feature_identity_fingerprint": features["identity_fingerprint"],
            "new_downloads": False, "network_during_execution": False, "competition_submission": False,
            "candidate_promotion": False, "final_selection": False, "publication": False}
    write(ROOT / "plan.json", plan)
    checksum = digest(ROOT / "plan.json")
    write(DECISION, {"schema_version": "rsna-campaign-autonomous-decision-v1", "decision": "approved",
          "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
          "decision_made_by": "campaign agent under standing owner authorization",
          "authorization_sha256": digest(AUTH), "plan_sha256": checksum, "resources": resources,
          "basis": "Complete the matched long-budget architecture comparison and one prospectively fixed half blend using admitted frozen features.",
          "new_exact_owner_numeric_approval_claimed": False, "gpu_ledger_cap_changed": False,
          "new_gpu_request": False, "new_weight_download": False, "candidate_promotion": False})
    launcher = ROOT / "run.sbatch"
    with launcher.open("x") as handle:
        handle.write(f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket55-heads
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
export RSNA_TICKET55_REVISION={REVISION}
export PYTHONPATH={snapshot}/src
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
{PROJECT}/.venv/bin/python {snapshot / Path(__file__).name} run {ROOT / 'plan.json'} {checksum}
''')
    write(ROOT / "preparation-validation.json", {"plan_sha256": checksum, "launcher_sha256": digest(launcher),
          "compiled": True, "training_runs": 6, "fixed_blend_variants": 1, "launched": False})
    print(json.dumps({"plan_sha256": checksum, "launcher": str(launcher)}))


def predict(head, pooled, indices, uids):
    values = {}
    with torch.inference_mode():
        for start in range(0, len(indices), 2):
            batch = indices[start:start + 2]
            scores = torch.sigmoid(head(pooled[batch])).numpy()
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite held-out head prediction")
            values.update({uids[index]: score.tolist() for index, score in zip(batch, scores, strict=True)})
    return values


def run(plan_path, checksum):
    started = time.monotonic()
    if digest(plan_path) != checksum:
        raise ValueError("Frozen matched-head plan changed")
    plan = read(plan_path)
    decision = read(plan["decision_path"])
    if (decision["decision"] != "approved" or decision["plan_sha256"] != checksum
            or decision["resources"] != plan["resources"] or digest(AUTH) != plan["authorization_sha256"]):
        raise ValueError("Exact autonomous resource decision differs")
    controller = subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], text=True)
    fields = dict(token.split("=", 1) for token in controller.split() if "=" in token)
    expected = {"JobState": "RUNNING", "Partition": "cluster1-cpu", "NumNodes": "1", "NumCPUs": "16",
                "NumTasks": "1", "MinMemoryNode": "64G", "TimeLimit": "00:30:00", "OverSubscribe": "OK", "Requeue": "0", "Restarts": "0"}
    if any(fields.get(key) != value for key, value in expected.items()) or "gpu" in fields.get("TRES", "").lower() or torch.cuda.is_available():
        raise ValueError("Live CPU allocation differs")
    root = Path(plan["run_root"])
    write(root / f"running-allocation-{os.environ['SLURM_JOB_ID']}.json", {"controller": controller})
    watchdog = threading.Timer(plan["resources"]["code_watchdog_seconds"], lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    for row in plan["frozen_files"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("Frozen source, original checkpoint or metadata changed")
    for name, version in plan["runtime_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Frozen CPU runtime differs")
    if os.environ.get("PYTHONPATH") != str(Path(plan["source_snapshot"]) / "src"):
        raise ValueError("Immutable source runtime is required")
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    parent = read(plan["parent_plan"])
    manifest = load_fold_manifest(t36.FOLD_MANIFEST_PATH)
    if manifest.manifest_checksum != plan["fold_manifest_checksum"]:
        raise ValueError("Corrected development folds differ")
    feature_manifest = read(t36.FEATURE_ROOT / "feature-manifest.json")
    rows = {row["study_instance_uid"]: row for row in feature_manifest["features"]}
    uids = sorted(rows)
    index = {uid: i for i, uid in enumerate(uids)}
    pooled_rows = []
    for uid in uids:
        row = rows[uid]
        loaded = t36._load_feature_file(t36.FEATURE_ROOT / row["feature_file"], study_uid=uid,
                 batch_index=row["batch_index"], feature_identity_fingerprint=plan["original_feature_identity_fingerprint"])
        if loaded["file_sha256"] != row["feature_file_sha256"]:
            raise ValueError("Frozen slice feature bytes changed")
        with torch.no_grad():
            pooled_rows.append(pool_frozen_features(torch.from_numpy(loaded["features"])[None],
                torch.from_numpy(loaded["slice_mask"])[None], torch.from_numpy(loaded["series_mask"])[None],
                torch.from_numpy(loaded["series_descriptors"])[None], feature_width=512,
                pooling_id=parent["head"]["pooling_id"]).squeeze(0))
    pooled = torch.stack(pooled_rows)
    if pooled.shape != (4228, 2076) or not torch.isfinite(pooled).all():
        raise ValueError("ResNet pooled bank dimensions differ")
    np.savez(root / "pooled-cohort.npz", pooled=pooled.numpy(), study_uids=np.asarray(uids))
    write(root / "pooled-cohort-evidence.json", {"source_feature_manifest_sha256": digest(t36.FEATURE_ROOT / "feature-manifest.json"),
          "source_entries_verified": len(uids), "pooled_array_sha256": _array_sha256(pooled.numpy()),
          "pooled_file_sha256": digest(root / "pooled-cohort.npz"), "fitted": False})
    # Adapt the unchanged sampler/loss/checkpoint helper inside this isolated
    # process to the original ResNet dimensions and initialization. No shared
    # source is edited; the six original-prefix checks verify equivalence.
    heads.FEATURE_WIDTH, heads.POOLED_WIDTH = 512, 2076
    heads.HEAD_ID = plan["head"]["head_id"]
    weight_state = torch.load(t36.WEIGHTS_PATH, map_location="cpu", weights_only=True)
    initialization = {}
    def original_head(seed):
        seed_everything(seed)
        model = build_study_model(ModelConfig.from_dict(plan["original_resnet_configuration"]["model"]))
        model.load_pretrained_state(weight_state)
        initialization["full_model_sha256"] = _model_state_sha256(model)
        return model.head
    heads.build_head = original_head
    gold = {member.study_instance_uid: member for member in manifest.gold_memberships}
    arms = ("reference-only", "report-supervision")
    predictions, short, ablated, convnext = [{arm: {} for arm in arms} for _ in range(4)]
    records = []
    optimization = plan["optimization"]
    parameters = {key: optimization[key] for key in ("successful_steps", "batch_size", "initialization_seed",
                  "batch_order_seed", "weak_seed_offset", "learning_rate", "checkpoint_interval_steps")}
    for row in parent["folds"]:
        fold = row["fold"]
        admission = admit_ticket30_fold_supervision(manifest, fold, t36.TICKET30_ARTIFACT,
            expected_artifact_sha256=t36.TICKET30_AUDITED_ARTIFACT_SHA256).for_image_studies(uids)
        if admission.to_public_dict() != row["supervision_admission"]:
            raise ValueError("Original report admission differs")
        training, heldout, weak = [row[key] for key in ("gold_training_study_uids", "gold_heldout_study_uids", "weak_training_study_uids")]
        if set(training + weak) & set(manifest.held_out_study_uids(fold)):
            raise ValueError("Held-out study entered fitting partitions")
        targets = torch.zeros((len(uids), 12), dtype=torch.float32)
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for uid in training + weak:
            values = gold[uid].reference_labels if uid in gold else admission.by_study[uid].targets
            for column, value in enumerate(values):
                if value is not None:
                    targets[index[uid], column] = float(value)
                    mask[index[uid], column] = True
        heldout_indices = [index[uid] for uid in heldout]
        if mask[heldout_indices].any():
            raise ValueError("Held-out target entries entered training tensors")
        matched = []
        for arm in arms:
            old_path = OLD / arm / f"fold-{fold}/run.json"
            original = read(old_path)
            checkpoint_path = old_path.parent / "checkpoints/final.pt"
            if digest(checkpoint_path) != original["checkpoint"]["final_sha256"]:
                raise ValueError("Original trusted checkpoint bytes changed")
            # Own hash-verified checkpoint contains the historical NumPy RNG
            # state, which the weights-only loader cannot decode.
            old_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            old_head = {key.removeprefix("head."): value for key, value in old_checkpoint["model_state_dict"].items() if key.startswith("head.")}
            short[arm].update(original["predictions"])
            convnext_record = read(CONVNEXT / arm / f"fold-{fold}/run.json")
            if set(convnext_record["predictions"]) != set(heldout):
                raise ValueError("Matched long ConvNeXt cohort differs")
            convnext[arm].update(convnext_record["predictions"])
            run_root = root / arm / f"fold-{fold}"
            identity = {"plan_sha256": checksum, "fold": fold, "arm": arm, "head": plan["head"],
                        "fold_manifest_checksum": manifest.manifest_checksum,
                        "pooled_cohort_sha256": digest(root / "pooled-cohort.npz")}
            prefix = []
            def checkpoint(head, optimizer, counts):
                step = counts["successful_optimizer_steps"]
                if step == 1500:
                    trial = predict(head, pooled, heldout_indices, uids)
                    error = max(abs(x - y) for uid in trial for x, y in zip(trial[uid], original["predictions"][uid], strict=True))
                    state_error = max(float((value - old_head[key]).abs().max()) for key, value in head.state_dict().items())
                    if (error > plan["prefix_maximum_absolute_error"] or state_error > plan["prefix_maximum_absolute_error"]
                            or initialization["full_model_sha256"] != original["counts"]["initial_model_state_sha256"]
                            or counts["gold_exposure_sequence_sha256"] != original["counts"]["gold_exposure_sequence_sha256"]
                            or not weak_sequence_matches(counts, original["counts"])):
                        raise ValueError(f"Original ResNet prefix differs: prediction={error}, state={state_error}")
                    prefix.append({"prediction_max_abs_error": error, "state_max_abs_error": state_error,
                                   "initial_full_model_state_matches": True, "ordered_exposures_match": True})
                heads.save_head_checkpoint(run_root / "checkpoints" / ("final.pt" if step == 7500 else "latest.pt"),
                                          head, optimizer, counts, identity, final=step == 7500)
            head, counts = heads.train_head(pooled=pooled, targets=targets, mask=mask, study_uids=uids,
                reference_indices=[index[uid] for uid in training],
                weak_indices=[index[uid] for uid in weak] if arm == "report-supervision" else [],
                checkpoint=checkpoint, **parameters)
            restored, replay_counts = heads.reload_head(run_root / "checkpoints/final.pt", identity)
            if replay_counts != counts or len(prefix) != 1 or counts["skipped_batches"]:
                raise ValueError("Final budget or source prefix differs")
            full = predict(head, pooled, heldout_indices, uids)
            replayed = predict(restored, pooled, heldout_indices, uids)
            zero = predict(restored, heads.zero_image_features(pooled), heldout_indices, uids)
            if full != replayed:
                raise ValueError("Final head replay differs")
            predictions[arm].update(replayed)
            ablated[arm].update(zero)
            record = {"schema_version": "rsna-ticket55-matched-head-run-v1", "status": "COMPLETE",
                      "identity": identity, "counts": counts, "prefix_replay": prefix[0], "predictions": replayed,
                      "training_study_uids": training + (weak if arm == "report-supervision" else []),
                      "held_out_study_uids": heldout, "held_out_inputs_excluded_from_fitting": True,
                      "held_out_target_entries_masked": True, "checkpoint_replay_max_abs_error": 0.0,
                      "checkpoint_sha256": digest(run_root / "checkpoints/final.pt")}
            write(run_root / "run.json", record)
            records.append({"fold": fold, "arm": arm, "path": str(run_root / "run.json"), "sha256": digest(run_root / "run.json")})
            matched.append(counts)
            print(json.dumps({"fold": fold, "arm": arm, "steps": 7500, "elapsed_seconds": time.monotonic() - started}), flush=True)
        if any(matched[0][key] != matched[1][key] for key in ("initial_head_state_sha256", "gold_exposure_sequence_sha256")):
            raise ValueError("Matched reference/report initialization or exposures differ")
    blend = {uid: ((np.asarray(predictions["report-supervision"][uid]) + convnext["report-supervision"][uid]) / 2).tolist() for uid in gold}
    pairs = {"gold_long_minus_short": (short["reference-only"], predictions["reference-only"]),
             "report_long_minus_short": (short["report-supervision"], predictions["report-supervision"]),
             "report_minus_gold": (predictions["reference-only"], predictions["report-supervision"]),
             "gold_minus_zero_image": (ablated["reference-only"], predictions["reference-only"]),
             "report_minus_zero_image": (ablated["report-supervision"], predictions["report-supervision"]),
             "convnext_long_minus_resnet_long": (predictions["report-supervision"], convnext["report-supervision"]),
             "fixed_half_blend_minus_resnet_long": (predictions["report-supervision"], blend)}
    comparisons = {name: compare_image_model_predictions(base, candidate, manifest,
        bootstrap_resamples=plan["analysis"]["bootstrap_resamples"], bootstrap_seed=plan["analysis"]["seed"])
        for name, (base, candidate) in pairs.items()}
    def passes(name):
        c = comparisons[name]
        lower = c["paired_study_bootstrap"]["macro_auc"]["lower"]
        return lower is not None and lower > 0 and all(row["delta_macro_auc"] is not None and row["delta_macro_auc"] > 0 for row in c["delta"]["folds"])
    disposition = "provisional_fixed_blend_review" if passes("fixed_half_blend_minus_resnet_long") and passes("report_minus_zero_image") else "inconclusive_no_promotion"
    result = {"schema_version": "rsna-ticket55-matched-head-analysis-v1", "status": "COMPLETE", "plan_sha256": checksum,
              "observed_at_utc": datetime.now(timezone.utc).isoformat(), "job_id": os.environ["SLURM_JOB_ID"],
              "comparisons": comparisons, "run_records": records, "fixed_half_blend_predictions": blend,
              "disposition": disposition, "candidate_promotion": False, "development_cohort_reused": True,
              "elapsed_seconds": time.monotonic() - started,
              "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "gpu_count": 0, "external_spend_usd": 0}
    write(root / "analysis.json", result)
    watchdog.cancel()
    print(json.dumps({"status": "COMPLETE", "disposition": disposition, "analysis": str(root / "analysis.json")}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        run(Path(sys.argv[2]), sys.argv[3])
