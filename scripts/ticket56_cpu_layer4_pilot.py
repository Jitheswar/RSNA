"""Bounded real-MRI CPU timing for a prospective report-adaptation route."""

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

import numpy as np
import torch

from rsna_knee import ticket36 as t36
from rsna_knee.domain import SeriesRecord, StudyRecord
from rsna_knee.folds import load_fold_manifest
from rsna_knee.model import ModelConfig, build_study_model
from rsna_knee.preprocessing import pack_study
from rsna_knee.real_inference import load_integration_manifest
from rsna_knee.resnet18 import RESNET18_FINETUNE_MODEL_ID
from rsna_knee.series_cache import SeriesCache
from rsna_knee.ticket34 import admit_ticket30_fold_supervision
from rsna_knee.training import _array_sha256, masked_bce_loss, seed_everything


PROJECT = t36.PROJECT_ROOT
ROOT = PROJECT / "artifacts/private/ticket56-cpu-layer4-pilot"
PARENT = PROJECT / "artifacts/private/ticket55-resnet-long-report-head-v2"
COHORT_PLAN = PROJECT / "artifacts/private/ticket49-convnext-feature-cohort/plan.json"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket56-cpu-layer4-pilot-20261001.json"
STORAGE = PROJECT / "artifacts/private/campaign-continuation-20261001/ticket56-storage-observation.json"


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
        raise FileExistsError("Immutable CPU adaptation pilot already exists")
    validation = read(PARENT / "validation.json")
    if validation["status"] != "COMPLETE" or validation["final_predictions_verified"] != 116:
        raise ValueError("Independently verified parent head is required")
    parent = read(PARENT / "plan.json")
    folds = read(parent["parent_plan"])["folds"]
    fold0 = next(row for row in folds if row["fold"] == 0)
    cohort = read(COHORT_PLAN)
    counts = cohort["valid_slice_counts"]
    ordered = sorted(fold0["weak_training_study_uids"],
                     key=lambda uid: (counts[uid], hashlib.sha256(uid.encode()).hexdigest()))
    selected = [ordered[round(q * (len(ordered) - 1))] for q in (0.1, 0.4, 0.7, 1.0)]
    manifest = load_fold_manifest(t36.FOLD_MANIFEST_PATH)
    if (len(set(selected)) != 4 or set(selected) & set(manifest.held_out_study_uids(0))
            or set(selected) & {m.study_instance_uid for m in manifest.gold_memberships}):
        raise ValueError("Pilot images entered a held-out or gold cohort")
    ROOT.mkdir(mode=0o700, parents=True)
    snapshot = ROOT / "source-snapshot"
    shutil.copytree(PROJECT / "src/rsna_knee", snapshot / "src/rsna_knee",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(Path(__file__), snapshot / Path(__file__).name)
    fixed = [AUTH, PARENT / "plan.json", PARENT / "validation.json",
             PARENT / "report-supervision/fold-0/run.json",
             PARENT / "report-supervision/fold-0/checkpoints/final.pt",
             Path(parent["parent_plan"]), COHORT_PLAN, t36.INTEGRATION_PATH,
             t36.SERIES_INDEX_PATH, t36.FEATURE_ROOT / "feature-manifest.json",
             t36.FOLD_MANIFEST_PATH, t36.TICKET30_ARTIFACT, t36.WEIGHTS_PATH, STORAGE]
    fixed.extend(path for path in snapshot.rglob("*") if path.is_file())
    resources = {"partition": "cluster1-cpu", "nodes": 1, "tasks": 1, "cpus_per_task": 16,
                 "memory_gib": 64, "walltime_seconds": 900, "code_watchdog_seconds": 780,
                 "gpu_count": 0, "exclusive": False, "maximum_launch_attempts": 1,
                 "requeue": False, "external_spend_usd": 0}
    plan = {"schema_version": "rsna-ticket56-cpu-adaptation-pilot-plan-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_root": str(ROOT),
            "source_snapshot": str(snapshot), "decision_path": str(DECISION), "resources": resources,
            "authorization_sha256": digest(AUTH), "fold_manifest_checksum": manifest.manifest_checksum,
            "training_fold": 0, "selected_training_report_uids": selected,
            "selected_valid_slice_counts": {uid: counts[uid] for uid in selected},
            "selection": "10th/40th/70th/100th valid-slice-count order statistics within admitted fold-0 reports; SHA256(uid) tie-break, no labels",
            "model": {**parent["original_resnet_configuration"]["model"], "model_id": RESNET18_FINETUNE_MODEL_ID},
            "optimizer": {"id": "Adam", "encoder_learning_rate": 1e-5, "head_learning_rate": 1e-4},
            "profiling": {"batch_size": 4, "warmup_updates": 1, "timed_updates": 4,
                          "seed": 20261001, "parity_tolerance": 1e-6},
            "projection": {"adaptation_updates_per_fold": 1500, "reference_draws_per_update": 2,
                           "report_draws_per_update": 2, "cost_allowance": 2,
                           "maximum_fold_seconds": 21600, "maximum_total_node_seconds": 86400,
                           "full_training_selected": False},
            "storage_observation": read(STORAGE),
            "frozen_files": [{"path": str(path), "sha256": digest(path)} for path in fixed],
            "runtime_versions": {name: importlib.metadata.version(name) for name in ("numpy", "torch")},
            "network_during_execution": False, "new_downloads": False, "candidate_checkpoint_saved": False,
            "competition_submission": False, "candidate_promotion": False, "final_selection": False}
    write(ROOT / "plan.json", plan)
    checksum = digest(ROOT / "plan.json")
    write(DECISION, {"schema_version": "rsna-campaign-autonomous-decision-v1", "decision": "approved",
          "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
          "decision_made_by": "campaign agent under standing owner authorization",
          "authorization_sha256": digest(AUTH), "plan_sha256": checksum, "resources": resources,
          "basis": "Measure a bounded report-supervised layer4 adaptation route after verified frozen-head comparisons; use admitted resources and training-side data only.",
          "new_exact_owner_numeric_approval_claimed": False, "gpu_ledger_cap_changed": False,
          "new_gpu_request": False, "new_weight_download": False, "full_training_authorized": False,
          "candidate_promotion": False})
    launcher = ROOT / "run.sbatch"
    with launcher.open("x") as handle:
        handle.write(f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket56-pilot
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:15:00
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
{PROJECT}/.venv/bin/python {snapshot / Path(__file__).name} run {ROOT / 'plan.json'} {checksum}
''')
    print(json.dumps({"plan_sha256": checksum, "launcher": str(launcher)}))


def run(plan_path, checksum):
    started = time.monotonic()
    if digest(plan_path) != checksum:
        raise ValueError("Frozen CPU pilot plan changed")
    plan = read(plan_path)
    decision = read(plan["decision_path"])
    if (decision["decision"] != "approved" or decision["plan_sha256"] != checksum
            or decision["resources"] != plan["resources"]
            or digest(AUTH) != plan["authorization_sha256"]):
        raise ValueError("Exact CPU resource decision differs")
    root = Path(plan["run_root"])
    job = os.environ["SLURM_JOB_ID"]
    controller = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True)
    fields = dict(token.split("=", 1) for token in controller.split() if "=" in token)
    expected = {"JobState": "RUNNING", "Partition": "cluster1-cpu", "NumNodes": "1", "NumCPUs": "16",
                "NumTasks": "1", "MinMemoryNode": "64G", "TimeLimit": "00:15:00",
                "OverSubscribe": "OK", "Requeue": "0", "Restarts": "0"}
    if any(fields.get(key) != value for key, value in expected.items()) or "gpu" in fields.get("TRES", "").lower():
        raise ValueError("Live CPU pilot allocation differs")
    write(root / "running-allocation.json", {"controller": controller})
    watchdog = threading.Timer(plan["resources"]["code_watchdog_seconds"], lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    for row in plan["frozen_files"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("Frozen source or pilot input bytes changed")
    for name, version in plan["runtime_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Pinned CPU runtime differs")
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise ValueError("CPU-only pilot is required")
    manifest = load_fold_manifest(t36.FOLD_MANIFEST_PATH)
    parent = read(read(PARENT / "plan.json")["parent_plan"])
    fold0 = next(row for row in parent["folds"] if row["fold"] == 0)
    selected = plan["selected_training_report_uids"]
    if (manifest.manifest_checksum != plan["fold_manifest_checksum"]
            or not set(selected) <= set(fold0["weak_training_study_uids"])
            or set(selected) & set(manifest.held_out_study_uids(0))
            or set(selected) & {m.study_instance_uid for m in manifest.gold_memberships}):
        raise ValueError("Held-out or gold input entered report profiling")
    admission = admit_ticket30_fold_supervision(manifest, 0, t36.TICKET30_ARTIFACT,
        expected_artifact_sha256=t36.TICKET30_AUDITED_ARTIFACT_SHA256)
    targets = torch.tensor([[float(value) if value is not None else 0.0
                             for value in admission.by_study[uid].targets] for uid in selected])
    mask = torch.tensor([[value is not None for value in admission.by_study[uid].targets]
                         for uid in selected], dtype=torch.bool)
    if not mask.any(dim=1).all():
        raise ValueError("Every profiled report requires admitted supervision")
    integration = load_integration_manifest(t36.INTEGRATION_PATH)
    index = read(t36.SERIES_INDEX_PATH)["entries"]
    reference = {r["study_instance_uid"]: r for r in read(t36.FEATURE_ROOT / "feature-manifest.json")["features"]}
    studies = {s.study_instance_uid: s for s in integration.studies}
    cache = SeriesCache(t36.TICKET35_SERIES_CACHE, read_only=True)
    packed_rows = []
    for uid in selected:
        prior = reference[uid]
        if prior["source_kind"] != "ticket35-prepared-series-cache-read-only":
            raise ValueError("Probe must use existing report image caches")
        source = studies[uid]
        prepared, records = [], []
        for item in source.series:
            row = index[item.series_instance_uid]
            if row["study_instance_uid"] != uid or digest(cache.entry_path(row["cache_key"]) / "metadata.json") != row["metadata_sha256"]:
                raise ValueError("Prepared series identity differs")
            prepared.append(cache.load_validated_entry(row["cache_key"], study_instance_uid=uid,
                series_instance_uid=item.series_instance_uid, expected_source_entry_names=item.entries,
                config=integration.preprocess_config))
            records.append(SeriesRecord(study_instance_uid=uid, series_instance_uid=item.series_instance_uid,
                slices=(), anatomical_plane=item.anatomical_plane, fluid_sensitive=item.fluid_sensitive,
                fat_suppression=item.fat_suppression, diagnostic_status=item.diagnostic_status,
                source_entry_names=item.entries))
        if t36._study_source_digest(source, index, archive_fingerprint=integration.archive_fingerprint) != prior["source_identity_sha256"]:
            raise ValueError("MRI source differs from the admitted representation")
        packed = pack_study(StudyRecord(uid, tuple(records)), prepared, integration.preprocess_config,
                            packing_config=integration.packing_config)
        prior_path = t36.FEATURE_ROOT / prior["feature_file"]
        if digest(prior_path) != prior["feature_file_sha256"]:
            raise ValueError("Admitted source feature identity differs")
        with np.load(prior_path, allow_pickle=False) as previous:
            for key, value in {"slice_mask": packed.slice_mask, "series_mask": packed.series_mask,
                               "series_descriptors": packed.series_descriptors}.items():
                if not np.array_equal(previous[key], value):
                    raise ValueError("Packing or anatomical descriptors differ")
        if int((packed.slice_mask & packed.series_mask[:, None]).sum()) != plan["selected_valid_slice_counts"][uid]:
            raise ValueError("Selected actual slice count differs")
        packed_rows.append(packed)
    images = torch.from_numpy(np.stack([row.images for row in packed_rows])).float()
    slices = torch.from_numpy(np.stack([row.slice_mask for row in packed_rows])).bool()
    series = torch.from_numpy(np.stack([row.series_mask for row in packed_rows])).bool()
    descriptors = torch.from_numpy(np.stack([row.series_descriptors for row in packed_rows])).float()
    input_digest = _array_sha256(images.numpy())
    seed_everything(plan["profiling"]["seed"])
    model = build_study_model(ModelConfig.from_dict(plan["model"]))
    model.load_pretrained_state(torch.load(t36.WEIGHTS_PATH, map_location="cpu", weights_only=True))
    initial_head = torch.load(PARENT / "report-supervision/fold-0/checkpoints/final.pt",
                              map_location="cpu", weights_only=True)
    model.head.load_state_dict(initial_head["head_state_dict"], strict=True)
    model.train()
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    valid_indices = torch.nonzero((slices & series[:, :, None]).flatten(), as_tuple=False).flatten()
    flat = images.reshape(-1, 1, 224, 224)
    prefix_started = time.monotonic()
    with torch.no_grad():
        prefix = torch.cat([model.encoder.forward_before_layer4(
            (flat[valid_indices[offset:offset + 16]].expand(-1, 3, -1, -1)
             - model.imagenet_mean) / model.imagenet_std)
            for offset in range(0, len(valid_indices), 16)])
    prefix_seconds = time.monotonic() - prefix_started
    if prefix.shape != (len(valid_indices), 256, 14, 14) or not torch.isfinite(prefix).all():
        raise ValueError("Pre-adaptation activation contract differs")
    activation_path = root / "temporary-prefix.npy"
    io_start = time.monotonic()
    np.save(activation_path, prefix.numpy(), allow_pickle=False)
    write_seconds = time.monotonic() - io_start
    activation_file_sha = digest(activation_path)
    io_start = time.monotonic()
    restored = np.load(activation_path, allow_pickle=False)
    read_seconds = time.monotonic() - io_start
    if _array_sha256(restored) != _array_sha256(prefix.numpy()):
        raise ValueError("Temporary activation readback changed bytes")
    activation_bytes = activation_path.stat().st_size
    activation_path.unlink()
    def cached_forward():
        chunks = [model.encoder.forward_from_layer4(prefix[offset:offset + 16])
                  for offset in range(0, len(prefix), 16)]
        values = torch.zeros((flat.shape[0], 512)).index_copy(0, valid_indices, torch.cat(chunks))
        return model.forward_from_features(values.reshape(4, 16, 8, 512), slices, series, descriptors)
    with torch.no_grad():
        score_error = float((model(images, slices, series, descriptors) - cached_forward()).abs().max())
    if score_error > plan["profiling"]["parity_tolerance"]:
        raise ValueError("Native/cached pre-update logits differ")
    results, final_states = {}, {}
    for mode in ("native", "cached_prefix"):
        model.load_state_dict(initial, strict=True)
        optimizer = torch.optim.Adam([
            {"params": model.encoder.layer4.parameters(), "lr": plan["optimizer"]["encoder_learning_rate"]},
            {"params": model.head.parameters(), "lr": plan["optimizer"]["head_learning_rate"]}])
        timings, losses = [], []
        for step in range(plan["profiling"]["warmup_updates"] + plan["profiling"]["timed_updates"]):
            update_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, slices, series, descriptors) if mode == "native" else cached_forward()
            loss = masked_bce_loss(logits, targets, mask)
            if loss is None or not torch.isfinite(loss):
                raise ValueError("Nonfinite or entirely masked real-report loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
                raise ValueError("Missing or nonfinite adaptation gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            if step >= plan["profiling"]["warmup_updates"]:
                timings.append(time.monotonic() - update_started)
        final_states[mode] = {key: value.detach().clone() for key, value in model.state_dict().items()}
        groups = {}
        for group, matches in {"layer4": lambda key: key.startswith("encoder.layer4."),
                               "head": lambda key: key.startswith("head.")}.items():
            groups[group] = max(float((value - initial[key]).abs().max())
                                for key, value in final_states[mode].items() if matches(key))
        frozen_error = max(float((value - initial[key]).abs().max())
                           for key, value in final_states[mode].items()
                           if not key.startswith(("encoder.layer4.", "head.")))
        # Normalization buffers, including layer4's, remain in eval mode.
        buffer_error = max(float((value - initial[key]).abs().max())
                           for key, value in final_states[mode].items()
                           if key.endswith(("running_mean", "running_var", "num_batches_tracked")))
        if any(value <= 0 for value in groups.values()) or frozen_error != 0 or buffer_error != 0:
            raise ValueError("Adaptation scope or fixed normalization changed")
        results[mode] = {"timed_update_seconds": timings, "median_update_seconds": float(np.median(timings)),
                         "training_losses": losses, "parameter_max_abs_changes": groups,
                         "frozen_prefix_max_abs_change": frozen_error, "normalization_buffer_max_abs_change": buffer_error}
        print(json.dumps({"mode": mode, "median_update_seconds": results[mode]["median_update_seconds"]}), flush=True)
    state_error = max(float((value - final_states["cached_prefix"][key]).abs().max())
                      for key, value in final_states["native"].items())
    if state_error > plan["profiling"]["parity_tolerance"]:
        raise ValueError("Equivalent native/cached updates changed final state")
    cohort = read(COHORT_PLAN)
    counts = cohort["valid_slice_counts"]
    allowance = plan["projection"]["cost_allowance"]
    rate = results["cached_prefix"]["median_update_seconds"] / len(valid_indices)
    projection = []
    for row in parent["folds"]:
        expected_slices = sum(plan["projection"][key] * np.mean([counts[uid] for uid in row[uids]])
            for key, uids in (("reference_draws_per_update", "gold_training_study_uids"),
                              ("report_draws_per_update", "weak_training_study_uids")))
        seconds = float(allowance * expected_slices * rate * plan["projection"]["adaptation_updates_per_fold"])
        projection.append({"fold": row["fold"], "expected_valid_slices_per_update": float(expected_slices),
                           "projected_training_seconds_with_allowance": seconds})
    cohort_slices = sum(counts.values())
    storage = cohort_slices * 256 * 14 * 14 * 4
    storage_observation = plan["storage_observation"]
    projected_project_bytes = (storage_observation["project_apparent_bytes"]
                               + storage_observation["tracked_external_model_apparent_bytes"] + storage)
    storage_gate = (projected_project_bytes < storage_observation["storage_soft_cap_bytes"]
                    and storage < storage_observation["filesystem_available_bytes"])
    extraction = allowance * prefix_seconds / len(prefix) * cohort_slices
    cache_io = allowance * (write_seconds + read_seconds) / len(prefix) * cohort_slices
    total = extraction + cache_io + sum(row["projected_training_seconds_with_allowance"] for row in projection)
    eligible = (storage_gate and total <= plan["projection"]["maximum_total_node_seconds"]
                and all(row["projected_training_seconds_with_allowance"] <= plan["projection"]["maximum_fold_seconds"] for row in projection)
                and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 64 * 1024**3)
    write(root / "pilot-result.json", {"schema_version": "rsna-ticket56-cpu-adaptation-pilot-result-v1",
        "status": "COMPLETE", "observed_at_utc": datetime.now(timezone.utc).isoformat(), "job_id": job,
        "plan_sha256": checksum, "selected_report_study_count": 4, "valid_slices": len(valid_indices),
        "held_out_inputs_excluded": True, "admitted_supervised_entries_per_update": int(mask.sum()),
        "input_array_sha256": input_digest, "prefix_array_sha256": _array_sha256(prefix.numpy()),
        "prefix_extraction_seconds": prefix_seconds, "temporary_activation_bytes": activation_bytes,
        "temporary_activation_sha256": activation_file_sha, "temporary_write_seconds": write_seconds,
        "temporary_read_seconds": read_seconds, "temporary_readback_identical": True,
        "native_cached_logit_max_abs_error": score_error, "native_cached_final_state_max_abs_error": state_error,
        "measurements": results, "projected_folds": projection, "projected_prefix_payload_bytes": storage,
        "projected_project_apparent_bytes": projected_project_bytes, "storage_projection_gate_passed": storage_gate,
        "projected_extraction_seconds_with_allowance": extraction, "projected_cache_io_seconds_with_allowance": cache_io,
        "projected_total_node_seconds_with_allowance": total,
        "cost_disposition": "eligible_for_separate_full_route_planning" if eligible else "cpu_cost_gate_failed",
        "projection_limits": "Four-report warm-cache microbenchmark; linear slice scaling is unverified at full cohort. Projection omits unmeasured scheduling, final evaluation, repeated random-read I/O and checkpoint/delivery overhead.",
        "full_training_selected": False, "candidate_promotion": False, "competition_submission": False,
        "elapsed_seconds": time.monotonic() - started,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "gpu_count": 0, "external_spend_usd": 0})
    watchdog.cancel()
    print(json.dumps({"status": "COMPLETE", "total_projected_node_hours": total / 3600,
                      "cost_gate_passed": eligible}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        run(Path(sys.argv[2]), sys.argv[3])
