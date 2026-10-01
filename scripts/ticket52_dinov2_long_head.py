"""Separate frozen DINOv2 duration experiment; no encoder training or release."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from torch import nn

from rsna_knee.dinov2 import load_dinov2_resource_approval
from rsna_knee.evaluation import evaluate_oof_predictions
from rsna_knee.feature_pooling import ANATOMICAL_POOLING_ID, pool_frozen_features
from rsna_knee.folds import load_fold_manifest
from rsna_knee.image_model_comparison import compare_image_model_predictions
from rsna_knee.model import ModelConfig, build_study_model
from rsna_knee.ticket36 import _load_feature_file
from rsna_knee.training import _array_sha256, _batch_indices, _model_state_sha256, masked_bce_loss, seed_everything


WIDTH = 384
POOLED_WIDTH = 4 * (WIDTH + 7)
HEAD_ID = "dinov2-vits14-frozen-plane-pooled-mlp32-v1"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    path.chmod(0o600)


def sequence_digest(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def predictions(head: nn.Module, pooled: torch.Tensor, indices: list[int], uids: list[str]) -> dict:
    result = {}
    with torch.no_grad():
        for start in range(0, len(indices), 2):
            batch = indices[start:start + 2]
            scores = torch.sigmoid(head(pooled[batch])).numpy()
            for index, score in zip(batch, scores, strict=True):
                if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
                    raise RuntimeError("head scores invalid")
                result[uids[index]] = score.tolist()
    return result


def main() -> None:
    started = time.monotonic()
    plan_path, expected_sha = Path(sys.argv[1]), sys.argv[2]
    if digest(plan_path) != expected_sha:
        raise RuntimeError("frozen plan changed")
    plan = read(plan_path)
    approval = read(Path(plan["approval_path"]))
    if (approval.get("plan_sha256") != expected_sha or approval.get("decision") != "approved"
            or approval.get("resources") != plan["resources"]):
        raise RuntimeError("exact CPU plan admission differs")
    if (os.environ.get("SLURM_JOB_PARTITION") != "cluster1-cpu"
            or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) != 16
            or int(os.environ.get("SLURM_MEM_PER_NODE", 0)) != 65536
            or torch.cuda.is_available()):
        raise RuntimeError("exact CPU allocation required")
    root = Path(plan["run_root"])
    output = root / ("job-" + os.environ["SLURM_JOB_ID"])
    output.mkdir(mode=0o700, exist_ok=False)
    for row in plan["frozen_files"]:
        path = Path(row["path"])
        if path.is_symlink() or not path.is_file() or digest(path) != row["sha256"]:
            raise RuntimeError("frozen source, runtime, or input changed")
    for name, version in plan["runtime_versions"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("frozen runtime changed")
    load_dinov2_resource_approval(plan["original_encoder_approval"])
    torch.set_num_threads(16)
    manifest = load_fold_manifest(plan["fold_manifest_path"])
    if manifest.manifest_checksum != plan["fold_manifest_checksum"]:
        raise RuntimeError("corrected folds changed")
    members = {row.study_instance_uid: row for row in manifest.gold_memberships}
    uids = sorted(members)
    if len(uids) != 58:
        raise RuntimeError("gold cohort changed")
    old_integration, expanded = [read(Path(plan[key])) for key in ("original_integration", "expanded_integration")]
    old_studies = {s["StudyInstanceUID"]: s for s in old_integration["studies"]}
    expanded_studies = {s["StudyInstanceUID"]: s for s in expanded["studies"]}
    if any(old_studies[uid] != expanded_studies[uid] for uid in uids):
        raise RuntimeError("gold series entries, descriptors or order differ")
    cache_identity = read(Path(plan["feature_cache"]) / "identity.json")
    reference_manifest = read(Path(plan["reference_feature_manifest"]))
    a, b = [dict(v["preprocessing"]) for v in (cache_identity["identity"], reference_manifest["feature_identity"])]
    a.pop("representation_id"); b.pop("representation_id")
    if a != b or cache_identity["identity"]["packing"] != reference_manifest["feature_identity"]["packing"]:
        raise RuntimeError("mask and feature pixel/packing policies differ")
    reference_rows = {row["study_instance_uid"]: row for row in reference_manifest["features"]}
    pooled_rows = []
    for uid in uids:
        key = hashlib.sha256(uid.encode()).hexdigest()
        feature_path = Path(plan["feature_cache"]) / "entries" / (key + ".npy")
        metadata = read(feature_path.with_suffix(".json"))
        features = np.load(feature_path, allow_pickle=False)
        if (metadata["study_instance_uid"] != uid or metadata["identity_fingerprint"] != cache_identity["identity_fingerprint"]
                or metadata["feature_file_sha256"] != digest(feature_path)
                or metadata["feature_array_sha256"] != _array_sha256(features)
                or features.shape != (16, 8, WIDTH) or features.dtype != np.float32 or not np.isfinite(features).all()):
            raise RuntimeError("DINO feature cache identity invalid")
        row = reference_rows[uid]
        reference_path = Path(row["feature_file"])
        if not reference_path.is_absolute():
            reference_path = Path(plan["reference_feature_manifest"]).parent / reference_path
        reference = _load_feature_file(reference_path, study_uid=uid,
                                      batch_index=row["batch_index"], feature_identity_fingerprint=reference_manifest["identity_fingerprint"])
        if reference["file_sha256"] != row["feature_file_sha256"] or row["source_kind"] != "frozen-rsna-archive":
            raise RuntimeError("gold mask cache identity invalid")
        valid = reference["slice_mask"] & reference["series_mask"][:, None]
        if not np.array_equal(np.any(features != 0, axis=-1), valid):
            raise RuntimeError("DINO cache padding differs from masks")
        with torch.no_grad():
            pooled_rows.append(pool_frozen_features(torch.from_numpy(features)[None],
                torch.from_numpy(reference["slice_mask"])[None], torch.from_numpy(reference["series_mask"])[None],
                torch.from_numpy(reference["series_descriptors"])[None], feature_width=WIDTH,
                pooling_id=ANATOMICAL_POOLING_ID).squeeze(0))
    pooled = torch.stack(pooled_rows)
    targets = torch.tensor([members[uid].reference_labels for uid in uids], dtype=torch.float32)
    mask = torch.ones_like(targets, dtype=torch.bool)
    if pooled.shape != (58, POOLED_WIDTH) or targets.shape != (58, 12) or not torch.isfinite(pooled).all():
        raise RuntimeError("pooled gold inputs invalid")
    zero = pooled.clone().reshape(58, 4, WIDTH + 7)
    zero[:, :, :WIDTH] = 0
    zero = zero.reshape(58, POOLED_WIDTH)
    config = read(Path(plan["original_configuration"]))["config"]
    short, baseline, full, ablated, fold_evidence = {}, {}, {}, {}, []
    for fold in range(3):
        fold_started = time.monotonic()
        source = read(Path(plan["short_root"]) / f"fold-{fold}/run.json")
        control = read(Path(plan["baseline_root"]) / f"fold-{fold}/run.json")
        training = [i for i, uid in enumerate(uids) if members[uid].fold != fold]
        heldout = [i for i, uid in enumerate(uids) if members[uid].fold == fold]
        heldout_uids = {uids[i] for i in heldout}
        if (set(source["training_study_uids"]) != {uids[i] for i in training}
                or set(source["predictions"]) != heldout_uids or set(control["predictions"]) != heldout_uids
                or source["counts"]["successful_optimizer_steps"] != 100
                or control["counts"]["successful_optimizer_steps"] != 1500
                or source["status"] != "completed_fixed_budget_training"
                or control["status"] != "completed_fixed_budget_training"):
            raise RuntimeError("short/control fold isolation differs")
        seed_everything(config["seeds"]["initialization"])
        model = build_study_model(ModelConfig.from_dict(config["model"]))
        head = model.head
        del model
        initial_sha = _model_state_sha256(head)
        optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
        exposure, losses = [], []
        replay = None
        for step in range(1500):
            if time.monotonic() - started > plan["resources"]["code_watchdog_seconds"]:
                raise RuntimeError("bounded CPU watchdog exceeded")
            indices = [training[i] for i in _batch_indices(len(training), 2, step, config["seeds"]["batch_order"])]
            if set(indices) & set(heldout):
                raise RuntimeError("held-out input entered fitting")
            exposure.extend(uids[i] for i in indices)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_bce_loss(head(pooled[indices]), targets[indices], mask[indices])
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("gold loss invalid")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise RuntimeError("head gradient invalid")
            optimizer.step()
            losses.append(float(loss.detach()))
            if step == 99:
                trial = predictions(head, pooled, heldout, uids)
                error = max(abs(x - y) for uid in trial for x, y in zip(trial[uid], source["predictions"][uid], strict=True))
                if sequence_digest(exposure) != source["counts"]["gold_exposure_sequence_sha256"] or error > 1e-6:
                    raise RuntimeError(f"100-step source replay failed: max error {error:.12g}")
                replay = {"maximum_absolute_score_error": error, "ordered_gold_exposure_digest_matches": True}
        head.eval()
        values = predictions(head, pooled, heldout, uids)
        identity = {"plan_sha256": expected_sha, "fold": fold, "head_id": HEAD_ID,
                    "feature_identity_fingerprint": cache_identity["identity_fingerprint"],
                    "training_study_uids": [uids[i] for i in training], "held_out_study_uids": [uids[i] for i in heldout]}
        checkpoint = output / f"fold-{fold}/final-head.pt"
        checkpoint.parent.mkdir(mode=0o700)
        with checkpoint.open("xb") as handle:
            torch.save({"identity": identity, "head_state_dict": head.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "successful_steps": 1500,
                        "checkpoint_selection": "final_step", "rng_state": torch.get_rng_state()}, handle)
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if loaded["identity"] != identity or loaded["successful_steps"] != 1500:
            raise RuntimeError("final head checkpoint identity differs")
        reloaded = nn.Sequential(nn.Linear(POOLED_WIDTH, 32), nn.ReLU(inplace=True), nn.Linear(32, 12))
        reloaded.load_state_dict(loaded["head_state_dict"], strict=True)
        reloaded.eval()
        replayed = predictions(reloaded, pooled, heldout, uids)
        replay_error = max(abs(x - y) for uid in values for x, y in zip(values[uid], replayed[uid], strict=True))
        if replay_error > 1e-6:
            raise RuntimeError("final checkpoint replay failed")
        full.update(values); ablated.update(predictions(reloaded, zero, heldout, uids))
        short.update(source["predictions"]); baseline.update(control["predictions"])
        record = {**identity, "status": "completed_fixed_budget_training", "predictions": values,
                  "counts": {"successful_optimizer_steps": 1500, "gold_sample_exposures": len(exposure),
                             "supervised_entry_exposures": 12 * len(exposure), "initial_head_sha256": initial_sha,
                             "gold_exposure_sequence_sha256": sequence_digest(exposure)},
                  "short_source_replay": replay, "final_checkpoint_replay_max_error": replay_error,
                  "checkpoint_sha256": digest(checkpoint), "loss_history": losses,
                  "elapsed_seconds": time.monotonic() - fold_started}
        save(checkpoint.parent / "run.json", record)
        fold_evidence.append({k: v for k, v in record.items() if k not in {"predictions", "loss_history", "training_study_uids", "held_out_study_uids"}})
        print(json.dumps({"completed_fold": fold, "successful_steps": 1500, "source_replay_error": replay["maximum_absolute_score_error"]}), flush=True)
    compare = lambda a, b: compare_image_model_predictions(a, b, manifest, bootstrap_resamples=2000, bootstrap_seed=20260930)
    primary = compare(baseline, full)
    lower = primary["paired_study_bootstrap"]["macro_auc"]["lower"]
    fold_deltas = [row["delta_macro_auc"] for row in primary["delta"]["folds"]]
    disposition = ("eligible_for_followup_no_promotion" if lower is not None and lower > 0
                   and all(x is not None and x > 0 for x in fold_deltas) else "inconclusive_no_promotion")
    evidence = {"schema_version": "rsna-ticket52-dinov2-long-head-evidence-v1", "plan_sha256": expected_sha,
                "job_id": os.environ["SLURM_JOB_ID"], "fold_evidence": fold_evidence,
                "primary_dino1500_minus_resnet1500": primary,
                "secondary_dino1500_minus_dino100": compare(short, full),
                "image_control_full_minus_zero": compare(ablated, full),
                "oof_diagnostics": evaluate_oof_predictions(full, manifest, bootstrap_resamples=2000,
                    bootstrap_seed=20260930, permutation_resamples=10000, permutation_seed=20260930).to_public_dict(),
                "constant_control": evaluate_oof_predictions({uid: [0.5] * 12 for uid in uids}, manifest, include_diagnostics=False).to_public_dict(),
                "disposition": disposition, "candidate_promotion": False, "external_spend_usd": 0,
                "gpu_seconds": 0, "development_cv_not_untouched_test": True,
                "elapsed_seconds": time.monotonic() - started,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    save(output / "evidence.json", evidence)
    print(json.dumps({"status": "completed", "disposition": disposition,
                      "macro_auc": primary["candidate"]["pooled_macro_auc"]}), flush=True)


if __name__ == "__main__":
    main()
