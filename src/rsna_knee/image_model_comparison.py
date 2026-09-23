"""Controlled paired OOF comparison of two image-only model architectures."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence, cast

import numpy as np

from .archive import atomic_json_write
from .evaluation import evaluate_oof_predictions, roc_auc_score
from .folds import FOLD_IDS, FoldManifest, load_fold_manifest
from .image_variant import TICKET19_ENCODER_CHANNELS
from .model import HierarchicalStudyModel, ModelConfig
from .oof import OOFArtifactError, OOFRunRecord, aggregate_oof_run_records, load_oof_run_record
from .targets import NUM_TARGETS, TARGET_COLUMNS


BASELINE_ENCODER_CHANNELS = (8, 16, 32)
COMPARISON_SCHEMA_VERSION = "rsna-knee-image-model-comparison-v1"


class ImageModelComparisonError(ValueError):
    """Raised when two image-model OOF arms do not form a controlled comparison."""


def _interval(
    values: Sequence[float],
    *,
    total_resamples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, object]:
    if values:
        alpha = (1.0 - confidence_level) / 2.0
        lower, median, upper = np.quantile(
            np.asarray(values, dtype=np.float64), [alpha, 0.5, 1.0 - alpha]
        )
        result: dict[str, object] = {
            "median": float(median),
            "lower": float(lower),
            "upper": float(upper),
        }
    else:
        result = {"median": None, "lower": None, "upper": None}
    return {
        **result,
        "defined_resamples": len(values),
        "undefined_resamples": total_resamples - len(values),
        "resamples": total_resamples,
        "seed": seed,
        "confidence_level": confidence_level,
        "method": "paired_study_percentile_bootstrap",
    }


def compare_image_model_predictions(
    baseline_predictions: Mapping[str, Sequence[float]],
    candidate_predictions: Mapping[str, Sequence[float]],
    fold_manifest: FoldManifest,
    *,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 20260919,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Compare complete paired OOF predictions without imputing undefined AUCs."""

    if bootstrap_resamples <= 0 or bootstrap_seed < 0:
        raise ImageModelComparisonError("bootstrap resamples must be positive and seed non-negative")
    if not 0.0 < confidence_level < 1.0:
        raise ImageModelComparisonError("confidence level must be between zero and one")
    expected_uids = {item.study_instance_uid for item in fold_manifest.gold_memberships}
    if set(baseline_predictions) != expected_uids or set(candidate_predictions) != expected_uids:
        raise ImageModelComparisonError("both model arms must contain the same complete gold cohort")

    try:
        baseline = evaluate_oof_predictions(
            baseline_predictions,
            fold_manifest,
            include_diagnostics=False,
        )
        candidate = evaluate_oof_predictions(
            candidate_predictions,
            fold_manifest,
            include_diagnostics=False,
        )
    except ValueError as exc:
        raise ImageModelComparisonError(str(exc)) from exc
    if baseline.study_instance_uids != candidate.study_instance_uids:
        raise ImageModelComparisonError("paired model arms have different OOF prediction identities")

    memberships = {item.study_instance_uid: item for item in fold_manifest.gold_memberships}
    ordered_uids = baseline.study_instance_uids
    labels = np.asarray(
        [memberships[uid].reference_labels for uid in ordered_uids], dtype=np.int8
    ).reshape(len(ordered_uids), NUM_TARGETS)
    baseline_scores = np.asarray(
        [baseline_predictions[uid] for uid in ordered_uids], dtype=np.float64
    ).reshape(len(ordered_uids), NUM_TARGETS)
    candidate_scores = np.asarray(
        [candidate_predictions[uid] for uid in ordered_uids], dtype=np.float64
    ).reshape(len(ordered_uids), NUM_TARGETS)

    rng = np.random.default_rng(bootstrap_seed)
    macro_deltas: list[float] = []
    target_deltas: list[list[float]] = [[] for _ in TARGET_COLUMNS]
    for _ in range(bootstrap_resamples):
        sample = rng.integers(0, len(ordered_uids), size=len(ordered_uids))
        deltas: list[float | None] = []
        for target_index in range(NUM_TARGETS):
            baseline_auc = roc_auc_score(
                labels[sample, target_index], baseline_scores[sample, target_index]
            )
            candidate_auc = roc_auc_score(
                labels[sample, target_index], candidate_scores[sample, target_index]
            )
            delta = (
                None
                if baseline_auc is None or candidate_auc is None
                else float(candidate_auc - baseline_auc)
            )
            deltas.append(delta)
            if delta is not None:
                target_deltas[target_index].append(delta)
        if all(delta is not None for delta in deltas):
            macro_deltas.append(float(np.mean([cast(float, delta) for delta in deltas])))

    per_target = []
    for index, (baseline_metric, candidate_metric) in enumerate(
        zip(baseline.pooled_targets, candidate.pooled_targets, strict=True)
    ):
        delta = (
            None
            if baseline_metric.auc is None or candidate_metric.auc is None
            else candidate_metric.auc - baseline_metric.auc
        )
        per_target.append(
            {
                "target": TARGET_COLUMNS[index],
                "baseline_auc": baseline_metric.auc,
                "candidate_auc": candidate_metric.auc,
                "delta_auc": delta,
                "positive_count": baseline_metric.positive_count,
                "negative_count": baseline_metric.negative_count,
                "paired_study_bootstrap": _interval(
                    target_deltas[index],
                    total_resamples=bootstrap_resamples,
                    seed=bootstrap_seed,
                    confidence_level=confidence_level,
                ),
            }
        )
    fold_results = [
        {
            "fold": base_fold.fold,
            "baseline_macro_auc": base_fold.macro_auc,
            "candidate_macro_auc": candidate_fold.macro_auc,
            "delta_macro_auc": (
                None
                if base_fold.macro_auc is None or candidate_fold.macro_auc is None
                else candidate_fold.macro_auc - base_fold.macro_auc
            ),
            "prediction_count": len(base_fold.study_instance_uids),
        }
        for base_fold, candidate_fold in zip(baseline.folds, candidate.folds, strict=True)
    ]
    macro_delta = (
        None
        if baseline.pooled_macro_auc is None or candidate.pooled_macro_auc is None
        else candidate.pooled_macro_auc - baseline.pooled_macro_auc
    )
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "fold_manifest_checksum": fold_manifest.manifest_checksum,
        "repeat_name": fold_manifest.repeat_name,
        "prediction_count": baseline.prediction_count,
        "baseline": {
            "pooled_macro_auc": baseline.pooled_macro_auc,
            "mean_fold_macro_auc": baseline.mean_fold_macro_auc,
            "fold_macro_auc_std": baseline.fold_macro_auc_std,
            "pooled_targets": [item.to_dict() for item in baseline.pooled_targets],
        },
        "candidate": {
            "pooled_macro_auc": candidate.pooled_macro_auc,
            "mean_fold_macro_auc": candidate.mean_fold_macro_auc,
            "fold_macro_auc_std": candidate.fold_macro_auc_std,
            "pooled_targets": [item.to_dict() for item in candidate.pooled_targets],
        },
        "delta": {
            "definition": "candidate minus baseline",
            "pooled_macro_auc": macro_delta,
            "mean_fold_macro_auc": (
                None
                if baseline.mean_fold_macro_auc is None or candidate.mean_fold_macro_auc is None
                else candidate.mean_fold_macro_auc - baseline.mean_fold_macro_auc
            ),
            "folds": fold_results,
        },
        "paired_study_bootstrap": {
            "macro_auc": _interval(
                macro_deltas,
                total_resamples=bootstrap_resamples,
                seed=bootstrap_seed,
                confidence_level=confidence_level,
            )
        },
        "per_target": per_target,
        "undefined_auc_is_not_imputed": True,
        "interpretation_limit": (
            "Repeated development cross-validation for model selection; not an untouched internal "
            "test or clinical-generalization estimate."
        ),
    }


def _normalised_config(config: Mapping[str, object]) -> dict[str, object]:
    result = json.loads(json.dumps(config))
    if not isinstance(result, dict):
        raise ImageModelComparisonError("run configuration must be an object")
    model = result.get("model")
    if not isinstance(model, dict):
        raise ImageModelComparisonError("run configuration is missing its model settings")
    model.pop("channels", None)
    return cast(dict[str, object], result)


def _channels(record: OOFRunRecord) -> tuple[int, ...]:
    model = record.config.get("model")
    if not isinstance(model, Mapping):
        raise ImageModelComparisonError("run configuration is missing model settings")
    value = model.get("channels")
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(channel, int) or isinstance(channel, bool) or channel <= 0
        for channel in value
    ):
        raise ImageModelComparisonError("run configuration has invalid encoder channels")
    if len(value) != 3:
        raise ImageModelComparisonError("run configuration must declare three encoder stages")
    return tuple(value)


def _validate_cpu_resources(record: OOFRunRecord) -> dict[str, object]:
    resources = record.resources
    expected = {
        "requested_partition": "cluster1-cpu",
        "requested_nodes": 1,
        "requested_cpus": 16,
        "requested_memory_gb": 64,
        "requested_wall_time_hours": 6,
        "exclusive": False,
    }
    if any(resources.get(key) != value for key, value in expected.items()):
        raise ImageModelComparisonError("fold run exceeds or differs from the Ticket 19 CPU envelope")
    return {key: resources.get(key) for key in expected}


def _compute_evidence(record: OOFRunRecord) -> dict[str, object]:
    path = record.path.parent.with_suffix(".compute.json")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ImageModelComparisonError(f"fold run is missing valid observed compute evidence: {path}") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != "rsna-knee-observed-compute-v1":
        raise ImageModelComparisonError(f"fold run has unsupported compute evidence: {path}")
    required = ("elapsed_seconds", "user_cpu_seconds", "system_cpu_seconds", "peak_rss_kb")
    result: dict[str, object] = {}
    for field in required:
        item = value.get(field)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or item < 0:
            raise ImageModelComparisonError(f"compute evidence has invalid {field}: {path}")
        result[field] = item
    return result


def _arm_cost(records: Sequence[OOFRunRecord]) -> dict[str, object]:
    folds = []
    total_checkpoint_bytes = 0
    parameter_count: int | None = None
    heldout_inference_seconds: list[float] = []
    for record in sorted(records, key=lambda item: item.fold):
        checkpoint = record.path.parent / "checkpoints" / "final.pt"
        try:
            checkpoint_bytes = checkpoint.stat().st_size
            run_record = json.loads(record.path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ImageModelComparisonError(f"could not stat final checkpoint: {checkpoint}") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ImageModelComparisonError(f"could not read fold run record: {record.path}") from exc
        if not isinstance(run_record, Mapping) or not isinstance(run_record.get("checkpoint"), Mapping):
            raise ImageModelComparisonError(f"fold run is missing final checkpoint evidence: {record.path}")
        checkpoint_evidence = cast(Mapping[str, object], run_record["checkpoint"])
        total_checkpoint_bytes += checkpoint_bytes
        model_settings = cast(Mapping[str, object], record.config["model"])
        image_size = cast(Sequence[int], model_settings["image_size"])
        channels = _channels(record)
        if len(image_size) != 2 or len(channels) != 3:
            raise ImageModelComparisonError("run configuration has an invalid model shape")
        model = HierarchicalStudyModel(
            ModelConfig(
                image_size=(image_size[0], image_size[1]),
                channels=cast(tuple[int, int, int], channels),
                input_channels=cast(int, model_settings["input_channels"]),
                num_targets=cast(int, model_settings["num_targets"]),
            )
        )
        current_parameter_count = sum(item.numel() for item in model.parameters())
        if parameter_count is not None and current_parameter_count != parameter_count:
            raise ImageModelComparisonError("parameter count differs between folds in one model arm")
        parameter_count = current_parameter_count
        inference_time = record.counts.get("heldout_model_inference_seconds")
        if isinstance(inference_time, (int, float)) and not isinstance(inference_time, bool):
            if math.isfinite(float(inference_time)) and inference_time >= 0:
                heldout_inference_seconds.append(float(inference_time))
        folds.append(
            {
                "fold": record.fold,
                "run_id": record.run_id,
                "slurm_job_id": record.resources.get("slurm_attempt_id"),
                "config_fingerprint": record.config_fingerprint,
                "final_checkpoint_sha256": checkpoint_evidence.get("final_sha256"),
                "final_checkpoint_bytes": checkpoint_bytes,
                "requested_resources": _validate_cpu_resources(record),
                "observed_compute": _compute_evidence(record),
                "heldout_model_inference_seconds": (
                    float(inference_time)
                    if isinstance(inference_time, (int, float)) and not isinstance(inference_time, bool)
                    else None
                ),
                "heldout_studies": record.counts.get("held_out_studies"),
            }
        )
    return {
        "encoder_channels": list(_channels(records[0])),
        "code_revision": records[0].environment.get("code_revision"),
        "environment_fingerprint": records[0].environment.get("environment_fingerprint"),
        "parameter_count": parameter_count,
        "total_final_checkpoint_bytes": total_checkpoint_bytes,
        "mean_final_checkpoint_bytes": total_checkpoint_bytes / len(records),
        "folds": folds,
        "heldout_model_inference_seconds_total": (
            sum(heldout_inference_seconds) if len(heldout_inference_seconds) == len(records) else None
        ),
        "inference_scope": (
            "model scoring on already decoded and packed held-out studies; excludes process startup, "
            "DICOM loading, preprocessing, checkpoint loading, submission creation and hidden test scale"
        ),
        "peak_vram_bytes": None,
        "peak_vram_status": "not measured; the experiment uses the CPU partition",
        "package_size_bytes": None,
        "package_size_status": "only final checkpoint bytes are measured; a delivery bundle is not built by Ticket 19",
    }


def compare_image_model_runs(
    baseline_run_paths: Sequence[str | Path],
    candidate_run_paths: Sequence[str | Path],
    fold_manifest: FoldManifest,
    *,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 20260919,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Validate and compare the Ticket 19 candidate against gold-only baseline OOF runs."""

    if fold_manifest.superseded:
        raise ImageModelComparisonError("a superseded fold manifest cannot support Ticket 19")
    if len(fold_manifest.gold_memberships) != 58:
        raise ImageModelComparisonError("Ticket 19 requires the established 58-study gold cohort")
    try:
        aggregate_oof_run_records(
            baseline_run_paths,
            fold_manifest,
            expected_supervision="reference",
        )
        aggregate_oof_run_records(
            candidate_run_paths,
            fold_manifest,
            expected_supervision="reference",
        )
        baseline_records = tuple(
            load_oof_run_record(path, expected_supervision="reference")
            for path in baseline_run_paths
        )
        candidate_records = tuple(
            load_oof_run_record(path, expected_supervision="reference")
            for path in candidate_run_paths
        )
    except OOFArtifactError as exc:
        raise ImageModelComparisonError(str(exc)) from exc

    all_records = (*baseline_records, *candidate_records)
    if len({record.path for record in all_records}) != len(all_records):
        raise ImageModelComparisonError("a fold run cannot appear in both model arms")
    if len({record.run_id for record in all_records}) != len(all_records):
        raise ImageModelComparisonError("run identities must be unique across model arms")
    baseline_by_fold = {record.fold: record for record in baseline_records}
    candidate_by_fold = {record.fold: record for record in candidate_records}
    if set(baseline_by_fold) != set(FOLD_IDS) or set(candidate_by_fold) != set(FOLD_IDS):
        raise ImageModelComparisonError("each image-model arm must contain exactly one run per fold")
    if any(_channels(record) != BASELINE_ENCODER_CHANNELS for record in baseline_records):
        raise ImageModelComparisonError("baseline runs do not use the established 8/16/32 encoder")
    if any(_channels(record) != TICKET19_ENCODER_CHANNELS for record in candidate_records):
        raise ImageModelComparisonError("candidate runs do not use the frozen Ticket 19 16/32/64 encoder")
    normalized = [_normalised_config(record.config) for record in all_records]
    if any(config != normalized[0] for config in normalized[1:]):
        raise ImageModelComparisonError("training settings differ beyond the encoder channel widths")
    for record in all_records:
        counts = record.counts
        if (
            counts.get("successful_optimizer_steps") != 100
            or counts.get("scheduler_steps") != 100
            or counts.get("attempted_batches") != 100
            or counts.get("skipped_all_masked_batches") != 0
        ):
            raise ImageModelComparisonError("all folds must complete the fixed 100-step budget")
    for fold in FOLD_IDS:
        baseline_record = baseline_by_fold[fold]
        candidate_record = candidate_by_fold[fold]
        if baseline_record.training_study_uids != candidate_record.training_study_uids:
            raise ImageModelComparisonError(f"fold {fold} training cohort differs between model arms")
        if baseline_record.held_out_study_uids != candidate_record.held_out_study_uids:
            raise ImageModelComparisonError(f"fold {fold} held-out cohort differs between model arms")
        for field in (
            "source_version",
            "source_fingerprint",
            "integration_manifest_checksum",
        ):
            if getattr(baseline_record, field) != getattr(candidate_record, field):
                raise ImageModelComparisonError(f"fold {fold} {field} differs between model arms")
        for field in ("data_fingerprint", "fold_manifest_checksum", "source_fingerprint", "source_version"):
            if baseline_record.identity.get(field) != candidate_record.identity.get(field):
                raise ImageModelComparisonError(f"fold {fold} identity {field} differs between model arms")
        if baseline_record.counts.get("exposures_by_source") != candidate_record.counts.get("exposures_by_source"):
            raise ImageModelComparisonError(f"fold {fold} supervision exposure counts differ between model arms")
        if baseline_record.counts.get("gold_exposure_sequence_sha256") != candidate_record.counts.get("gold_exposure_sequence_sha256"):
            raise ImageModelComparisonError(f"fold {fold} gold exposure sequence differs between model arms")
    for records in (baseline_records, candidate_records):
        revisions = {record.environment.get("code_revision") for record in records}
        if len(revisions) != 1 or None in revisions:
            raise ImageModelComparisonError("each model arm requires one pinned code revision")

    try:
        result = compare_image_model_predictions(
            {
                uid: values
                for record in baseline_records
                for uid, values in record.predictions.items()
            },
            {
                uid: values
                for record in candidate_records
                for uid, values in record.predictions.items()
            },
            fold_manifest,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        )
    except ImageModelComparisonError:
        raise
    result["experiment"] = "Ticket 19 wider slice encoder versus mechanics baseline"
    result["comparison_controls"] = {
        "changed_setting": "shared slice encoder channels: 8/16/32 to 16/32/64",
        "held_fixed": [
            "gold-only reference supervision",
            "frozen fold manifest and cohort",
            "image size and preprocessing",
            "series and slice sampling",
            "initialization seed and batch-order seed",
            "optimizer, learning rate, schedule and loss",
            "augmentation, batch size and successful optimizer-step budget",
            "final-checkpoint OOF evaluation",
        ],
    }
    result["training_cost"] = {
        "baseline": _arm_cost(baseline_records),
        "candidate": _arm_cost(candidate_records),
    }
    result["resource_notes"] = {
        "gpu_execution": False,
        "vram": "not measured; all Ticket 19 runs use cluster1-cpu",
        "inference_package": "full notebook and delivery bundle costs remain unmeasured",
    }
    return result


def render_image_model_comparison(comparison: Mapping[str, object]) -> str:
    """Render an identifier-free Markdown report from the public comparison record."""

    baseline = cast(Mapping[str, object], comparison["baseline"])
    candidate = cast(Mapping[str, object], comparison["candidate"])
    delta = cast(Mapping[str, object], comparison["delta"])
    interval = cast(
        Mapping[str, object],
        cast(Mapping[str, object], comparison["paired_study_bootstrap"])["macro_auc"],
    )
    lines = [
        "# Ticket 19: Wider image encoder comparison",
        "",
        f"- OOF studies: {comparison['prediction_count']}",
        f"- Fold manifest SHA-256: `{comparison['fold_manifest_checksum']}`",
        f"- Baseline pooled macro ROC-AUC: `{baseline['pooled_macro_auc']}`",
        f"- Candidate pooled macro ROC-AUC: `{candidate['pooled_macro_auc']}`",
        f"- Candidate minus baseline: `{delta['pooled_macro_auc']}`",
        "- Paired study-bootstrap 95% macro-delta interval: "
        f"`{interval['lower']}` to `{interval['upper']}` "
        f"({interval['undefined_resamples']} undefined of {interval['resamples']}; seed {interval['seed']})",
        "",
        "## Fold results",
        "",
        "| Fold | Baseline macro AUC | Candidate macro AUC | Delta | OOF studies |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in cast(Sequence[Mapping[str, object]], delta["folds"]):
        lines.append(
            f"| {item['fold']} | {item['baseline_macro_auc']} | {item['candidate_macro_auc']} | "
            f"{item['delta_macro_auc']} | {item['prediction_count']} |"
        )
    lines.extend(
        [
            "",
            "## Per-target results",
            "",
            "| Target | Baseline AUC | Candidate AUC | Delta | Positives | Negatives | Paired 95% interval |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in cast(Sequence[Mapping[str, object]], comparison["per_target"]):
        target_interval = cast(Mapping[str, object], item["paired_study_bootstrap"])
        lines.append(
            f"| {item['target']} | {item['baseline_auc']} | {item['candidate_auc']} | "
            f"{item['delta_auc']} | {item['positive_count']} | {item['negative_count']} | "
            f"{target_interval['lower']} to {target_interval['upper']} "
            f"({target_interval['undefined_resamples']} undefined) |"
        )
    costs = cast(Mapping[str, object], comparison["training_cost"])
    baseline_cost = cast(Mapping[str, object], costs["baseline"])
    candidate_cost = cast(Mapping[str, object], costs["candidate"])
    lines.extend(
        [
            "",
            "## Runtime and model size",
            "",
            "| Arm | Fold | Elapsed seconds | User CPU seconds | Peak RSS KB | Final checkpoint bytes |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm_name, arm in (("Baseline", baseline_cost), ("Candidate", candidate_cost)):
        for item in cast(Sequence[Mapping[str, object]], arm["folds"]):
            compute = cast(Mapping[str, object], item["observed_compute"])
            lines.append(
                f"| {arm_name} | {item['fold']} | {compute['elapsed_seconds']} | "
                f"{compute['user_cpu_seconds']} | {compute['peak_rss_kb']} | "
                f"{item['final_checkpoint_bytes']} |"
            )
    lines.extend(
        [
            "",
            "Only final checkpoint sizes and the trainer's in-memory held-out model-scoring timer are "
            "recorded here. This is not a full inference run: it excludes DICOM loading, preprocessing, "
            "startup, model loading, CSV creation and the approximately 1,300-study hidden workload.",
            "GPU/VRAM was not measured because the experiment uses CPU Slurm jobs. The final checkpoint "
            "is not a complete offline package.",
            "",
            "## Interpretation",
            "",
            str(comparison["interpretation_limit"]),
            "",
        ]
    )
    for arm_name, arm in (("baseline", baseline_cost), ("candidate", candidate_cost)):
        lines.append(
            f"- {arm_name.title()}: {arm['parameter_count']} parameters; "
            f"{arm['total_final_checkpoint_bytes']} total final-checkpoint bytes across folds; "
            f"model-only held-out inference seconds `{arm['heldout_model_inference_seconds_total']}`."
        )
    return "\n".join(lines) + "\n"


def _atomic_text_write(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.partial-", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def _reject_run_destinations(
    destinations: Sequence[Path], run_paths: Sequence[str | Path]
) -> None:
    resolved_destinations = tuple(path.resolve() for path in destinations)
    if len(set(resolved_destinations)) != len(resolved_destinations):
        raise ImageModelComparisonError("comparison output paths must be distinct")
    run_directories = tuple(
        (Path(path) if Path(path).is_dir() else Path(path).parent).resolve()
        for path in run_paths
    )
    for destination in resolved_destinations:
        for run_directory in run_directories:
            try:
                destination.relative_to(run_directory)
            except ValueError:
                continue
            raise ImageModelComparisonError(
                f"comparison output cannot be written inside an immutable fold run: {destination}"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--baseline-run", action="append", required=True)
    parser.add_argument("--candidate-run", action="append", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260919)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _reject_run_destinations(
            (args.json_output, args.markdown_output),
            (*args.baseline_run, *args.candidate_run),
        )
        result = compare_image_model_runs(
            args.baseline_run,
            args.candidate_run,
            load_fold_manifest(args.fold_manifest),
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            confidence_level=args.confidence_level,
        )
        atomic_json_write(args.json_output, result, overwrite=True)
        _atomic_text_write(args.markdown_output, render_image_model_comparison(result))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Ticket 19 image-model comparison failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "BASELINE_ENCODER_CHANNELS",
    "COMPARISON_SCHEMA_VERSION",
    "ImageModelComparisonError",
    "compare_image_model_predictions",
    "compare_image_model_runs",
    "main",
    "render_image_model_comparison",
]
