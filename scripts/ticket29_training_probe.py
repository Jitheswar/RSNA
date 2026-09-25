"""Bounded CPU training probe on Ticket 29's frozen real training cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping, cast

from rsna_knee.image_diagnostic import (
    COHORT_SIZE,
    DIAGNOSTIC_SEED,
    BATCH_ORDER_SEED,
    FOLD_ID,
    SLURM_RESOURCES,
    _check_slurm_context,
    _evaluate,
    _prevalence_control,
)
from rsna_knee.model import build_study_model
from rsna_knee.real_inference import load_integration_manifest
from rsna_knee.training import (
    TrainingConfig,
    require_cpu_preflight,
    run_cpu_preflight,
    seed_everything,
    train_fixed_budget,
)

from ticket29_memorization_probe import load_cache


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, value: Mapping[str, object]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as target:
        json.dump(value, target, sort_keys=True, separators=(",", ":"))
        target.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--integration-manifest", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--channels", nargs=3, required=True, type=int)
    parser.add_argument("--hypothesis", required=True)
    args = parser.parse_args()

    slurm = _check_slurm_context()
    cache_dir = Path(args.cache)
    metadata, examples = load_cache(cache_dir)
    if len(examples) != COHORT_SIZE or any(item.provenance.source_kind != "reference" for item in examples):
        raise RuntimeError("probe must use the eight reference-only frozen training studies")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    output.chmod(0o700)
    preflight = run_cpu_preflight(
        output_path=output / "cpu-preflight.json",
        require_isolated_python=True,
        require_slurm=True,
        required_slurm_resources=SLURM_RESOURCES,
    )
    require_cpu_preflight(preflight, require_slurm=True, require_isolated_python=True)
    integration = load_integration_manifest(args.integration_manifest)
    config = TrainingConfig(
        image_size=integration.preprocess_config.image_size,
        channels=tuple(args.channels),
        max_slices=integration.packing_config.max_slices,
        max_series=integration.packing_config.max_series,
        batch_size=2,
        successful_optimizer_steps=args.steps,
        optimizer="Adam",
        learning_rate=args.learning_rate,
        schedule="constant",
        seed=DIAGNOSTIC_SEED,
        batch_order_seed=BATCH_ORDER_SEED,
        checkpoint_interval_steps=100,
        checkpoint_interval_seconds=900.0,
        augmentation="none",
        device="cpu",
        selection_policy="final_checkpoint",
        preprocessing_config=integration.preprocess_config,
    )
    parent_path = Path(args.parent_checkpoint)
    parent_sha = _sha256(parent_path)
    identity: dict[str, object] = {
        "ticket": "29-repair-probe",
        "diagnostic_plan_sha256": metadata["plan_sha256"],
        "training_cache_sha256": metadata["cache_file_sha256"],
        "hypothesis": args.hypothesis,
        "parent_checkpoint_sha256": parent_sha,
        "fold": FOLD_ID,
        "source": "same eight reference-training MRI studies; no held-out data",
    }
    seed_everything(config.seed)
    model = build_study_model(config.model_config())
    initial = _evaluate(model, examples)
    prevalence = _prevalence_control(examples)
    started = time.monotonic()
    result = train_fixed_budget(
        examples,
        config,
        model=model,
        artifact_dir=output / "training",
        fold=FOLD_ID,
        identity=identity,
        parent_provenance={
            "parent_checkpoint_sha256": parent_sha,
            "parent_diagnostic_plan_sha256": metadata["plan_sha256"],
        },
        requested_resources={
            "partition": SLURM_RESOURCES["partition"],
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 8,
            "memory_gb": 32,
            "wall_time_hours": 6,
            "exclusive": False,
            "allocated_gpu_count": 0,
        },
    )
    elapsed = time.monotonic() - started
    final = _evaluate(result.model, examples)
    if result.final_checkpoint_path is None or result.successful_optimizer_steps != args.steps:
        raise RuntimeError("bounded probe did not complete its frozen optimizer steps")
    gate = (
        final["supported_targets"] == 12
        and float(cast(float, final["supported_target_macro_auc"])) >= 0.85
        and float(cast(float, final["training_cohort_mean_masked_bce"]))
        <= 0.5 * float(cast(float, prevalence["constant_prevalence_masked_bce"]))
    )
    record: dict[str, object] = {
        "status": "completed_training_only_probe",
        "hypothesis": args.hypothesis,
        "parent_checkpoint_sha256": parent_sha,
        "diagnostic_plan_sha256": metadata["plan_sha256"],
        "cache_sha256": metadata["cache_file_sha256"],
        "slurm_job_id": slurm["job_id"],
        "scientific_config_fingerprint": config.config_fingerprint,
        "learning_rate": args.learning_rate,
        "steps": args.steps,
        "channels": args.channels,
        "initial_loss": initial["training_cohort_mean_masked_bce"],
        "initial_training_macro_auc": initial["supported_target_macro_auc"],
        "prevalence_control_bce": prevalence["constant_prevalence_masked_bce"],
        "final_loss": final["training_cohort_mean_masked_bce"],
        "final_training_macro_auc": final["supported_target_macro_auc"],
        "supported_targets": final["supported_targets"],
        "frozen_gate_passed": gate,
        "training_seconds": elapsed,
        "final_checkpoint_sha256": _sha256(result.final_checkpoint_path),
        "final_checkpoint_path": str(result.final_checkpoint_path),
        "held_out_studies_used": 0,
        "gpu_allocations": 0,
    }
    _write_new(output / "result.json", record)
    print(json.dumps({key: value for key, value in record.items() if key != "final_checkpoint_path"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"ticket29 training probe failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
