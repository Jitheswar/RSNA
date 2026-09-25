"""Re-run a successful Ticket 29 probe from real DICOM under the frozen CPU gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping, cast

import numpy as np

from rsna_knee.image_diagnostic import (
    FOLD_ID,
    SLURM_RESOURCES,
    SUCCESS_CRITERIA,
    _check_inputs,
    _check_slurm_context,
    _evaluate,
    _example_summary,
    _gradient_summary,
    _pixel_and_geometry_summary,
    _prevalence_control,
    _relative_parameter_change,
    _verify_manifest_identity,
    load_diagnostic_plan,
)
from rsna_knee.model import build_study_model
from rsna_knee.folds import load_fold_manifest
from rsna_knee.inventory import identify_archive
from rsna_knee.real_inference import iter_real_studies, load_integration_manifest
from rsna_knee.training import (
    TrainingConfig,
    build_fold_training_examples,
    load_training_checkpoint,
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


def _write_new(path: Path, payload: Mapping[str, object]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as target:
        json.dump(payload, target, sort_keys=True, separators=(",", ":"))
        target.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "plan", "integration-manifest", "fold-manifest", "selection-report", "source-archive",
        "cache", "probe-result", "repair-plan", "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    slurm = _check_slurm_context()
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    output.chmod(0o700)
    preflight = run_cpu_preflight(
        output_path=output / "cpu-preflight.json",
        require_isolated_python=True,
        require_slurm=True,
        required_slurm_resources=SLURM_RESOURCES,
    )
    require_cpu_preflight(preflight, require_slurm=True, require_isolated_python=True)

    plan = load_diagnostic_plan(args.plan)
    integration = load_integration_manifest(args.integration_manifest)
    fold = load_fold_manifest(args.fold_manifest)
    selected, reconciliation = _verify_manifest_identity(
        plan, integration, fold, args.integration_manifest, args.fold_manifest, args.selection_report,
    )
    if args.source_archive.stat().st_size != cast(Mapping[str, object], plan["source"])["source_archive_bytes"]:
        raise RuntimeError("real source archive size changed")
    if identify_archive(args.source_archive, source_version=integration.source_version).fingerprint != integration.archive_fingerprint:
        raise RuntimeError("real source archive fingerprint changed")
    with args.probe_result.open(encoding="utf-8") as source:
        probe: dict[str, object] = json.load(source)
    with args.repair_plan.open(encoding="utf-8") as source:
        repair_plan: dict[str, object] = json.load(source)
    if (
        repair_plan.get("schema_version") != "rsna-ticket29-real-dicom-repair-plan-v1"
        or repair_plan.get("original_plan_sha256") != plan["plan_sha256"]
        or repair_plan.get("passing_probe_result_sha256") != _sha256(args.probe_result)
        or repair_plan.get("source_archive_fingerprint") != integration.archive_fingerprint
        or repair_plan.get("cohort_membership_sha256") != cast(Mapping[str, object], plan["cohort"])["cohort_membership_sha256"]
        or repair_plan.get("learning_rate") != probe.get("learning_rate")
        or repair_plan.get("steps") != probe.get("steps")
        or repair_plan.get("channels") != probe.get("channels")
        or repair_plan.get("success_criteria") != SUCCESS_CRITERIA
        or repair_plan.get("gpu_allocations") != 0
        or repair_plan.get("held_out_studies") != 0
    ):
        raise RuntimeError("real repair plan does not match the frozen successful probe")
    if (
        probe.get("status") != "completed_training_only_probe"
        or probe.get("frozen_gate_passed") is not True
        or probe.get("diagnostic_plan_sha256") != plan["plan_sha256"]
        or probe.get("held_out_studies_used") != 0
        or probe.get("gpu_allocations") != 0
    ):
        raise RuntimeError("repair rerun requires a passing, matching training-only probe")
    cache_metadata, cached_examples = load_cache(args.cache)
    if cache_metadata["cache_file_sha256"] != probe["cache_sha256"]:
        raise RuntimeError("probe cache identity changed")

    load_started = time.monotonic()
    loaded = tuple(iter_real_studies(integration, args.source_archive, require_complete_series=True, study_uids=selected))
    load_seconds = time.monotonic() - load_started
    pixel_summary, private_pixel_detail = _pixel_and_geometry_summary(loaded)
    studies_by_uid = {item.study.study_instance_uid: item.study for item in loaded}
    examples = build_fold_training_examples(
        studies_by_uid,
        fold,
        FOLD_ID,
        preprocess_config=integration.preprocess_config,
        packing_config=integration.packing_config,
        allow_partial_inputs=False,
        selected_training_study_uids=selected,
    )
    input_summary = _example_summary(examples, expected_ids=selected)
    checks = _check_inputs(pixel_summary, input_summary)
    if not all(checks.values()):
        raise RuntimeError("reloaded real input failed original frozen checks")
    if len(examples) != len(cached_examples):
        raise RuntimeError("reloaded cohort size differs from probe cache")
    for fresh, cached in zip(examples, cached_examples, strict=True):
        if (
            fresh.study_instance_uid != cached.study_instance_uid
            or fresh.provenance.source_kind != "reference"
            or fresh.provenance.fold != FOLD_ID
            or not fresh.provenance.held_out_excluded
            or not np.array_equal(fresh.images, cached.images)
            or not np.array_equal(fresh.slice_mask, cached.slice_mask)
            or not np.array_equal(fresh.series_mask, cached.series_mask)
            or not np.array_equal(fresh.targets, cached.targets)
            or not np.array_equal(fresh.supervision_mask, cached.supervision_mask)
        ):
            raise RuntimeError("real DICOM training input differs from the probed cached input")

    channels = tuple(cast(list[int], probe["channels"]))
    if len(channels) != 3:
        raise RuntimeError("probe model width malformed")
    config = TrainingConfig(
        image_size=integration.preprocess_config.image_size,
        channels=cast(tuple[int, int, int], channels),
        max_slices=integration.packing_config.max_slices,
        max_series=integration.packing_config.max_series,
        batch_size=2,
        successful_optimizer_steps=int(cast(int, probe["steps"])),
        optimizer="Adam",
        learning_rate=float(cast(float, probe["learning_rate"])),
        schedule="constant",
        seed=20260925,
        batch_order_seed=20260926,
        checkpoint_interval_steps=100,
        checkpoint_interval_seconds=900.0,
        augmentation="none",
        device="cpu",
        selection_policy="final_checkpoint",
        preprocessing_config=integration.preprocess_config,
    )
    if config.config_fingerprint != probe["scientific_config_fingerprint"]:
        raise RuntimeError("probe scientific config fingerprint changed")
    seed_everything(config.seed)
    model = build_study_model(config.model_config())
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    initial = _evaluate(model, examples)
    gradient = _gradient_summary(model, examples)
    prevalence = _prevalence_control(examples)
    identity: dict[str, object] = {
        "ticket": "29-repair-real-dicom-rerun",
        "original_plan_sha256": plan["plan_sha256"],
        "repair_plan_sha256": _sha256(args.repair_plan),
        "passing_probe_result_sha256": _sha256(args.probe_result),
        "probe_checkpoint_sha256": probe["final_checkpoint_sha256"],
        "cache_sha256": cache_metadata["cache_file_sha256"],
        "source_manifest_checksum": integration.source_manifest_checksum,
        "integration_manifest_checksum": integration.manifest_checksum,
        "fold_manifest_checksum": fold.manifest_checksum,
        "fold": FOLD_ID,
        "held_out_used": 0,
    }
    started = time.monotonic()
    result = train_fixed_budget(
        examples,
        config,
        model=model,
        artifact_dir=output / "training",
        fold=FOLD_ID,
        identity=identity,
        parent_provenance={"passing_probe_result_sha256": _sha256(args.probe_result)},
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
    training_seconds = time.monotonic() - started
    if result.final_checkpoint_path is None or result.successful_optimizer_steps != config.successful_optimizer_steps:
        raise RuntimeError("real DICOM repair training did not finish its frozen step budget")
    final = _evaluate(result.model, examples)
    change = _relative_parameter_change(initial_state, result.model.state_dict())
    curve: list[dict[str, object]] = [{"steps": 0, **initial, "checkpoint_sha256": None}]
    curve_steps = sorted({100, 500, config.successful_optimizer_steps})
    checkpoint_paths: dict[int, Path] = {config.successful_optimizer_steps: result.final_checkpoint_path}
    for path in sorted((output / "training" / "checkpoints" / "history").glob("*.pt")):
        checkpoint = load_training_checkpoint(path)
        progress = checkpoint.get("progress")
        if isinstance(progress, Mapping):
            step = progress.get("successful_optimizer_steps")
            if isinstance(step, int) and step in curve_steps:
                checkpoint_paths[step] = path
    if set(curve_steps) != set(checkpoint_paths):
        raise RuntimeError("repair run is missing a frozen learning-curve checkpoint")
    for step in curve_steps:
        checkpoint_path = checkpoint_paths[step]
        evaluation_model = build_study_model(config.model_config())
        load_training_checkpoint(checkpoint_path, model=evaluation_model)
        curve.append({"steps": step, **_evaluate(evaluation_model, examples), "checkpoint_sha256": _sha256(checkpoint_path)})
    final_auc = float(cast(float, final["supported_target_macro_auc"]))
    final_bce = float(cast(float, final["training_cohort_mean_masked_bce"]))
    control_bce = float(cast(float, prevalence["constant_prevalence_masked_bce"]))
    gradient_l2 = float(cast(float, gradient["global_gradient_l2_norm"]))
    relative_change = float(change["relative_parameter_change_l2"])
    gate = bool(
        gradient["all_parameter_gradients_present_and_finite"]
        and gradient_l2 >= float(SUCCESS_CRITERIA["gradient_global_norm_minimum"])
        and relative_change >= float(SUCCESS_CRITERIA["relative_parameter_l2_change_minimum"])
        and int(cast(int, final["supported_targets"])) >= int(SUCCESS_CRITERIA["long_run_supported_targets_minimum"])
        and final_auc >= float(SUCCESS_CRITERIA["long_run_supported_target_macro_auc_minimum"])
        and final_bce <= control_bce * float(SUCCESS_CRITERIA["long_run_loss_fraction_of_training_prevalence_control_maximum"])
    )
    payload: dict[str, object] = {
        "schema_version": "rsna-ticket29-real-dicom-repair-v1",
        "status": "completed" if gate else "failed_frozen_memorization_gate",
        "original_plan_sha256": plan["plan_sha256"],
        "repair_plan_sha256": _sha256(args.repair_plan),
        "passing_probe_result_sha256": _sha256(args.probe_result),
        "real_source_archive_fingerprint": integration.archive_fingerprint,
        "source_archive_bytes": args.source_archive.stat().st_size,
        "real_input_reloaded_equal_to_probed_cache": True,
        "reconciliation": reconciliation,
        "aggregate_real_pixel_diagnostic": pixel_summary,
        "aggregate_training_input_diagnostic": input_summary,
        "input_checks": checks,
        "private_pixel_detail": private_pixel_detail,
        "scientific_config_fingerprint": config.config_fingerprint,
        "learning_rate": config.learning_rate,
        "steps": config.successful_optimizer_steps,
        "channels": list(config.channels),
        "initial": initial,
        "initial_gradient": gradient,
        "prevalence_control": prevalence,
        "final": final,
        "learning_curve": curve,
        "parameter_change": change,
        "frozen_gate_passed": gate,
        "real_load_seconds": load_seconds,
        "training_seconds": training_seconds,
        "final_checkpoint_sha256": _sha256(result.final_checkpoint_path),
        "final_checkpoint_path": str(result.final_checkpoint_path),
        "slurm_job_id": slurm["job_id"],
        "slurm_resources": slurm,
        "held_out_studies_used": 0,
        "gpu_allocations": 0,
        "generalization_claim": False,
    }
    _write_new(output / "private-evidence.json", payload)
    print(json.dumps({
        "status": payload["status"], "frozen_gate_passed": gate, "auc": final_auc,
        "bce": final_bce, "control_bce": control_bce,
        "gradient_l2": gradient_l2, "relative_parameter_change": relative_change,
        "slurm_job_id": slurm["job_id"],
    }, sort_keys=True))
    return 0 if gate else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"ticket29 real DICOM repair failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
