"""Freeze a private CPU-only DINOv2 duration experiment before fitting."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "artifacts/private/ticket52-dinov2-long-head"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    path.chmod(0o600)


def main() -> None:
    if ROOT.exists():
        raise FileExistsError("experiment root already exists")
    snapshot = ROOT / "source-snapshot"
    shutil.copytree(PROJECT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(PROJECT / "scripts/ticket52_dinov2_long_head.py", snapshot / "run.py")
    short_root = PROJECT / "artifacts/private/ticket38-dinov2/attempt-2"
    cache = short_root / "frozen-feature-cache/22b2017db167e1ebe05d8a6c421d80387e4440ecc226a4f82f701bfad8e95f36"
    baseline = PROJECT / "artifacts/private/ticket44-full-weak-exposure/reference-only"
    fold_path = PROJECT / "artifacts/private/ticket-04-real-v2/private/fold-manifest.json"
    feature_manifest = PROJECT / "artifacts/private/ticket36-expanded-supervision/features/feature-manifest.json"
    original_integration = PROJECT / "artifacts/private/ticket-05-real-training-input/private/integration-manifest.json"
    expanded = PROJECT / "artifacts/private/ticket36-expanded-supervision/private/expanded-integration-manifest.json"
    encoder_approval = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket38-dinov2-vits14-cpu.json"
    configuration = short_root / "fold-0/configuration.json"
    files = [fold_path, feature_manifest, original_integration, expanded, encoder_approval, configuration,
             PROJECT / "requirements-cpu.lock", PROJECT / ".venv/pyvenv.cfg"]
    files += sorted(p for p in snapshot.rglob("*") if p.is_file())
    files += sorted(p for p in cache.rglob("*") if p.is_file())
    gold = {r["study_instance_uid"] for r in json.loads(feature_manifest.read_text())["features"]
            if r["source_kind"] == "frozen-rsna-archive"}
    if len(gold) != 58:
        raise RuntimeError("sealed gold cache differs")
    for row in json.loads(feature_manifest.read_text())["features"]:
        if row["study_instance_uid"] in gold:
            path = Path(row["feature_file"])
            files.append(path if path.is_absolute() else feature_manifest.parent / path)
    for fold in range(3):
        files += [short_root / f"fold-{fold}/run.json", short_root / f"fold-{fold}/configuration.json",
                  baseline / f"fold-{fold}/run.json"]
    approval_path = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket52-dinov2-long-head.json"
    resources = {"partition": "cluster1-cpu", "nodes": 1, "tasks": 1, "cpus_per_task": 16,
                 "memory_gib": 64, "walltime_seconds": 3600, "code_watchdog_seconds": 3500,
                 "gpu_count": 0, "maximum_concurrent_jobs": 1, "exclusive": False,
                 "maximum_launch_attempts": 1, "external_spend_usd": 0}
    plan = {"schema_version": "rsna-ticket52-dinov2-long-head-plan-v1", "ticket": 52,
            "run_root": str(ROOT), "source_snapshot": str(snapshot), "resources": resources,
            "approval_path": str(approval_path), "original_encoder_approval": str(encoder_approval),
            "fold_manifest_path": str(fold_path), "fold_manifest_checksum": "8460234d0fcbfd2eb9ca3a3e4165929f50a86acac37d2cd03be1164c2f287fb8",
            "original_integration": str(original_integration), "expanded_integration": str(expanded),
            "feature_cache": str(cache), "reference_feature_manifest": str(feature_manifest),
            "original_configuration": str(configuration), "short_root": str(short_root), "baseline_root": str(baseline),
            "runtime_versions": {name: importlib.metadata.version(name) for name in ["torch", "numpy", "transformers", "safetensors"]},
            "frozen_files": [{"path": str(p), "sha256": digest(p)} for p in sorted(set(files))],
            "training": {"encoder": "frozen previously admitted DINOv2 cache; no encoding or downloads",
                         "pooling": "plane-stratified-series-mean-v1", "head_hidden_size": 32,
                         "feature_width": 384, "pooled_width": 1564, "successful_steps": 1500,
                         "batch_size": 2, "optimizer": "Adam", "learning_rate": 0.001, "schedule": "constant",
                         "loss": "masked_bce", "supervision": "reference only", "initialization_seed": 20260911,
                         "batch_order_seed": 20260912, "selection_policy": "final_checkpoint",
                         "folds": [0, 1, 2], "maximum_cpu_training_runs": 3},
            "source_replay_gate": {"at_step": 100, "maximum_absolute_score_error": 1e-6,
                                   "ordered_gold_exposure_digest_must_match": True},
            "analysis": {"primary": "DINOv2 1500 minus preserved ResNet gold-only 1500",
                         "secondary": "DINOv2 1500 minus preserved DINOv2 100",
                         "image_control": "zero image features retaining masks, descriptors and presence",
                         "bootstrap_resamples": 2000, "bootstrap_seed": 20260930,
                         "permutation_resamples": 10000, "permutation_seed": 20260930,
                         "followup_gate": "primary paired macro AUC lower 95% bound >0 AND every fold delta >0; otherwise inconclusive",
                         "no_candidate_promotion": True, "development_cv_not_untouched_test": True},
            "network_during_execution": False, "external_spend_usd": 0,
            "final_candidate_selection": False, "publication": False, "ticket39_or_ticket40_changed": False}
    save(ROOT / "plan.json", plan)
    plan_sha = digest(ROOT / "plan.json")
    save(approval_path, {"schema_version": "rsna-ticket52-resource-approval-v1", "decision": "approved",
                         "plan_path": str(ROOT / "plan.json"), "plan_sha256": plan_sha, "resources": resources,
                         "approved_by": "Project owner through existing conversation-wide campaign authorization",
                         "authorization_basis": "approved for anything everything; no matter what 1st place",
                         "authorization_interpretation": "Separate bounded zero-spend private CPU experiment. No Ticket 52-specific owner words claimed; no expansion of GPU cap or amendment of Ticket 39/40.",
                         "rights_basis": "Reuses already admitted private DINOv2 feature cache; underlying pretraining image rights remain unresolved for release.",
                         "final_candidate_selection": False, "publication": False})
    launch = f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket52-dino
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
{PROJECT}/.venv/bin/python {snapshot}/run.py {ROOT}/plan.json {plan_sha}
'''
    with (ROOT / "run-head.sbatch").open("x") as handle:
        handle.write(launch)
    save(ROOT / "prepared-validation.json", {"plan_sha256": plan_sha, "launch_sha256": digest(ROOT / "run-head.sbatch"),
                                            "frozen_file_count": len(set(files)), "training_started": False})
    print(json.dumps({"plan_sha256": plan_sha, "frozen_files": len(set(files)), "gold_studies": len(gold)}))


if __name__ == "__main__":
    main()
