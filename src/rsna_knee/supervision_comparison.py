"""Paired, fold-safe OOF comparison of reference and report supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np

from .archive import atomic_json_write
from .evaluation import (
    OOFEvaluation,
    TargetEvaluation,
    evaluate_oof_predictions,
    roc_auc_score,
)
from .folds import FOLD_IDS, FoldManifest, load_fold_manifest
from .oof import OOFArtifactError, OOFRunRecord, aggregate_oof_run_records, load_oof_run_record
from .targets import NUM_TARGETS, TARGET_COLUMNS


class SupervisionComparisonError(ValueError):
    """Raised when the two supervision arms do not form a controlled OOF pair."""


_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_EXPOSURE_KINDS = ("reference", "weak", "mixed")
_EXPOSURE_FIELDS = ("sample_exposures", "supervised_entry_exposures")
_PUBLIC_ADMISSION_FIELDS = (
    "schema_version",
    "fold",
    "fold_manifest_checksum",
    "source_artifact_digest",
    "source_schema_version",
    "weak_label_version",
    "mixture_policy",
    "extractor",
    "counts",
    "boundaries",
)
_TICKET18_ADDITIVE_DESIGN = "ticket18-additive-report-supervision-v2"
_TICKET18_WEAK_SAMPLING_POLICY = "ticket18-report-only-cyclic-shuffle-v1"
_TICKET34_ADDITIVE_DESIGN = "ticket34-additive-report-supervision-v1"
_TICKET34_WEAK_SAMPLING_POLICY = "ticket34-report-only-cyclic-shuffle-v1"


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SupervisionComparisonError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SupervisionComparisonError(f"{field} must be an integer of at least {minimum}")
    return value


def _read_json_object(path: Path, description: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SupervisionComparisonError(f"could not read {description}: {path}") from exc
    return _mapping(payload, description)


@dataclass(frozen=True)
class PairedBootstrapInterval:
    """Study-paired percentile interval for a report-minus-reference metric."""

    median: float | None
    lower: float | None
    upper: float | None
    defined_resamples: int
    undefined_resamples: int
    resamples: int
    seed: int
    confidence_level: float

    def __post_init__(self) -> None:
        if self.resamples <= 0 or self.seed < 0:
            raise SupervisionComparisonError("bootstrap resamples and seed are invalid")
        if self.defined_resamples + self.undefined_resamples != self.resamples:
            raise SupervisionComparisonError("bootstrap defined and undefined counts do not reconcile")
        if not 0.0 < self.confidence_level < 1.0:
            raise SupervisionComparisonError("bootstrap confidence level must be between zero and one")
        values = (self.median, self.lower, self.upper)
        if self.defined_resamples == 0 and any(value is not None for value in values):
            raise SupervisionComparisonError("undefined bootstrap metric cannot have interval values")
        if self.defined_resamples > 0 and any(value is None for value in values):
            raise SupervisionComparisonError("defined bootstrap metric requires interval values")
        if any(value is not None and not -1.0 <= value <= 1.0 for value in values):
            raise SupervisionComparisonError("AUC delta interval values must be within [-1, 1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "paired_study_percentile_bootstrap",
            "median": self.median,
            "lower": self.lower,
            "upper": self.upper,
            "defined_resamples": self.defined_resamples,
            "undefined_resamples": self.undefined_resamples,
            "resamples": self.resamples,
            "seed": self.seed,
            "confidence_level": self.confidence_level,
        }


@dataclass(frozen=True)
class TargetAucComparison:
    target: str
    gold_only_auc: float | None
    report_supervised_auc: float | None
    delta_auc: float | None
    positive_count: int
    negative_count: int
    uncertainty: PairedBootstrapInterval

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "gold_only_auc": self.gold_only_auc,
            "report_supervised_auc": self.report_supervised_auc,
            "delta_auc_report_minus_gold": self.delta_auc,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "uncertainty": self.uncertainty.to_dict(),
        }


@dataclass(frozen=True)
class ArmExposureEvidence:
    run_id: str
    training_studies: int
    reference_training_studies: int
    report_training_studies: int
    successful_optimizer_steps: int
    attempted_batches: int
    skipped_all_masked_batches: int
    sample_exposures: int
    supervised_entry_exposures: int
    exposures_by_source: Mapping[str, Mapping[str, int]]
    requested_partition: str
    requested_nodes: int
    requested_cpus: int
    requested_memory_gb: int
    requested_wall_time_hours: int
    exclusive: bool
    slurm_job_id: str
    final_checkpoint_sha256: str
    observed_elapsed_seconds: float
    observed_user_cpu_seconds: float
    observed_system_cpu_seconds: float
    observed_peak_rss_kb: int
    gold_exposure_sequence_sha256: str | None = None
    weak_exposure_sequence_sha256: str | None = None
    initial_model_state_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "training_studies": self.training_studies,
            "reference_training_studies": self.reference_training_studies,
            "report_training_studies": self.report_training_studies,
            "successful_optimizer_steps": self.successful_optimizer_steps,
            "attempted_batches": self.attempted_batches,
            "skipped_all_masked_batches": self.skipped_all_masked_batches,
            "sample_exposures": self.sample_exposures,
            "supervised_entry_exposures": self.supervised_entry_exposures,
            "exposures_by_source": {
                kind: dict(counts) for kind, counts in self.exposures_by_source.items()
            },
            "requested_resources": {
                "partition": self.requested_partition,
                "nodes": self.requested_nodes,
                "cpus": self.requested_cpus,
                "memory_gb": self.requested_memory_gb,
                "wall_time_hours": self.requested_wall_time_hours,
                "exclusive": self.exclusive,
            },
            "slurm_job_id": self.slurm_job_id,
            "final_checkpoint_sha256": self.final_checkpoint_sha256,
            "observed_compute": {
                "elapsed_seconds": self.observed_elapsed_seconds,
                "user_cpu_seconds": self.observed_user_cpu_seconds,
                "system_cpu_seconds": self.observed_system_cpu_seconds,
                "peak_rss_kb": self.observed_peak_rss_kb,
            },
            "gold_exposure_sequence_sha256": self.gold_exposure_sequence_sha256,
            "weak_exposure_sequence_sha256": self.weak_exposure_sequence_sha256,
            "initial_model_state_sha256": self.initial_model_state_sha256,
        }


@dataclass(frozen=True)
class FoldSupervisionComparison:
    fold: int
    gold_only_auc: float | None
    report_supervised_auc: float | None
    macro_auc_delta: float | None
    gold_only_exposure: ArmExposureEvidence
    report_supervised_exposure: ArmExposureEvidence

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "gold_only_macro_auc": self.gold_only_auc,
            "report_supervised_macro_auc": self.report_supervised_auc,
            "macro_auc_delta_report_minus_gold": self.macro_auc_delta,
            "gold_only_training": self.gold_only_exposure.to_dict(),
            "report_supervised_training": self.report_supervised_exposure.to_dict(),
        }


@dataclass(frozen=True)
class SupervisionComparison:
    gold_only: OOFEvaluation
    report_supervised: OOFEvaluation
    pooled_macro_auc_delta: float | None
    macro_auc_delta_interval: PairedBootstrapInterval
    pooled_targets: tuple[TargetAucComparison, ...]
    folds: tuple[FoldSupervisionComparison, ...]
    config_fingerprint: str
    code_revision: str
    source_version: str
    source_fingerprint: str
    integration_manifest_checksum: str
    fold_manifest_checksum: str
    report_artifact_sha256: str
    report_semantic_digest: str
    weak_label_version: str
    mixture_policy: str
    positive_folds: int
    negative_folds: int
    unchanged_folds: int
    undefined_folds: int
    interpretation: str
    additive_design: bool = False
    report_extractor: str = "deterministic Ticket 15 extractor"
    report_ticket_number: int = 18

    def __post_init__(self) -> None:
        if tuple(item.target for item in self.pooled_targets) != TARGET_COLUMNS:
            raise SupervisionComparisonError("target comparison order must match the competition contract")
        if tuple(item.fold for item in self.folds) != FOLD_IDS:
            raise SupervisionComparisonError("supervision comparison must include all folds")
        if (
            self.positive_folds
            + self.negative_folds
            + self.unchanged_folds
            + self.undefined_folds
            != len(FOLD_IDS)
        ):
            raise SupervisionComparisonError("fold consistency counts do not reconcile")
        for value, name in (
            (self.report_artifact_sha256, "report artifact SHA-256"),
            (self.report_semantic_digest, "report semantic digest"),
            (self.source_fingerprint, "source fingerprint"),
            (self.integration_manifest_checksum, "integration manifest checksum"),
            (self.fold_manifest_checksum, "fold manifest checksum"),
        ):
            if not _DIGEST_RE.fullmatch(value):
                raise SupervisionComparisonError(f"{name} is invalid")

    @property
    def prediction_count(self) -> int:
        return self.gold_only.prediction_count

    def to_public_dict(self) -> dict[str, object]:
        return {
            "ticket": self.report_ticket_number,
            "schema_version": (
                "rsna-knee-supervision-comparison-additive-v2"
                if self.additive_design
                else "rsna-knee-supervision-comparison-v1"
            ),
            "experiment": (
                "gold-only versus additive gold-plus-clean-report supervision"
                if self.additive_design
                else "gold-only versus gold-plus-clean-report supervision"
            ),
            "design": (
                "additive weak supervision with identical paired gold exposure"
                if self.additive_design
                else "original mixture-policy comparison"
            ),
            "prediction_count": self.prediction_count,
            "metric": "pooled twelve-target macro ROC-AUC",
            "delta_definition": "report-supervised minus gold-only",
            "gold_only": {
                "pooled_macro_auc": self.gold_only.pooled_macro_auc,
                "defined_target_macro_auc": self.gold_only.defined_target_macro_auc,
                "fold_macro_auc": {
                    "mean": self.gold_only.mean_fold_macro_auc,
                    "standard_deviation": self.gold_only.fold_macro_auc_std,
                    "minimum": self.gold_only.fold_macro_auc_min,
                    "maximum": self.gold_only.fold_macro_auc_max,
                },
                "targets": [item.to_dict() for item in self.gold_only.pooled_targets],
                "folds": [item.to_public_dict() for item in self.gold_only.folds],
            },
            "report_supervised": {
                "pooled_macro_auc": self.report_supervised.pooled_macro_auc,
                "defined_target_macro_auc": self.report_supervised.defined_target_macro_auc,
                "fold_macro_auc": {
                    "mean": self.report_supervised.mean_fold_macro_auc,
                    "standard_deviation": self.report_supervised.fold_macro_auc_std,
                    "minimum": self.report_supervised.fold_macro_auc_min,
                    "maximum": self.report_supervised.fold_macro_auc_max,
                },
                "targets": [item.to_dict() for item in self.report_supervised.pooled_targets],
                "folds": [item.to_public_dict() for item in self.report_supervised.folds],
            },
            "paired_delta": {
                "pooled_macro_auc": self.pooled_macro_auc_delta,
                "uncertainty": self.macro_auc_delta_interval.to_dict(),
                "positive_folds": self.positive_folds,
                "negative_folds": self.negative_folds,
                "unchanged_folds": self.unchanged_folds,
                "undefined_folds": self.undefined_folds,
                "interpretation": self.interpretation,
            },
            "per_target": [item.to_dict() for item in self.pooled_targets],
            "fold_comparison": [item.to_dict() for item in self.folds],
            "supervision_provenance": {
                "weak_label_version": self.weak_label_version,
                "mixture_policy": self.mixture_policy,
                "extractor": self.report_extractor,
                "source_id": self.report_extractor,
                "report_artifact_sha256": self.report_artifact_sha256,
                "report_semantic_digest": self.report_semantic_digest,
            },
            "shared_experiment_identity": {
                "training_configuration_fingerprint": self.config_fingerprint,
                "code_revision": self.code_revision,
                "source_version": self.source_version,
                "source_fingerprint": self.source_fingerprint,
                "integration_manifest_checksum": self.integration_manifest_checksum,
                "fold_manifest_checksum": self.fold_manifest_checksum,
            },
            "interpretation_limitations": [
                "three-fold development CV guides model selection and is not an untouched internal test",
                "study-level uncertainty is used because patient grouping is unavailable",
                "no clinical-generalization claim is established",
                "undefined AUCs and bootstrap resamples are reported rather than imputed",
            ],
        }

    def to_private_dict(self) -> dict[str, object]:
        return self.to_public_dict()


def _load_report_evidence(
    record: OOFRunRecord,
    *,
    additive_design: bool = False,
    expected_ticket: int | None = None,
) -> tuple[str, str]:
    run_directory = record.path.parent
    evidence_path = run_directory.parent / f"{run_directory.name}.public.json"
    evidence = _read_json_object(evidence_path, "report-supervision run evidence")
    selected_ticket = expected_ticket or (18 if additive_design else 17)
    if evidence.get("ticket") not in (selected_ticket, str(selected_ticket)):
        raise SupervisionComparisonError(
            f"report run evidence is not from the Ticket {selected_ticket} training adapter"
        )
    if evidence.get("fold") != record.fold or evidence.get("fold_manifest_checksum") != record.fold_manifest_checksum:
        raise SupervisionComparisonError("report run evidence fold identity disagrees with its run record")
    if evidence.get("preflight_passed") is not True:
        raise SupervisionComparisonError("report run evidence does not contain a passed CPU preflight")
    supervision = record.supervision
    semantic_digest = evidence.get("report_extraction_semantic_digest")
    if semantic_digest != supervision.get("source_artifact_digest"):
        raise SupervisionComparisonError("report run semantic digest disagrees with admitted supervision")
    if not isinstance(semantic_digest, str) or not _DIGEST_RE.fullmatch(semantic_digest):
        raise SupervisionComparisonError("report run has an invalid semantic digest")
    artifact_sha256 = evidence.get("frozen_report_artifact_sha256")
    if not isinstance(artifact_sha256, str) or not _DIGEST_RE.fullmatch(artifact_sha256):
        raise SupervisionComparisonError("report run evidence has no frozen artifact checksum")
    image_admission = _mapping(evidence.get("supervision_admission"), "image-scope supervision evidence")
    image_admission_fields = (
        (*_PUBLIC_ADMISSION_FIELDS, "source_id")
        if selected_ticket == 36
        else _PUBLIC_ADMISSION_FIELDS
    )
    expected_image_admission = {
        field: supervision.get(field) for field in image_admission_fields
    }
    observed_image_admission = {
        field: image_admission.get(field) for field in image_admission_fields
    }
    image_admission_matches = (
        set(image_admission_fields).issubset(image_admission)
        and observed_image_admission == expected_image_admission
    )
    if selected_ticket != 36:
        image_admission_matches = dict(image_admission) == expected_image_admission
    if not image_admission_matches:
        raise SupervisionComparisonError("public report admission evidence disagrees with the private run")
    full_admission = _mapping(evidence.get("full_fold_supervision_admission"), "full-fold supervision evidence")
    full_admission_fields: tuple[str, ...] = (
        "schema_version",
        "fold",
        "fold_manifest_checksum",
        "source_artifact_digest",
        "source_schema_version",
        "weak_label_version",
        "mixture_policy",
    )
    if selected_ticket == 36:
        full_admission_fields = (*full_admission_fields, "source_id")
    for field in full_admission_fields:
        if full_admission.get(field) != supervision.get(field):
            raise SupervisionComparisonError("full-fold report lineage disagrees with the training run")
    return artifact_sha256, semantic_digest


def _arm_exposure(
    record: OOFRunRecord,
    *,
    report_arm: bool,
    additive_report_arm: bool = False,
) -> ArmExposureEvidence:
    counts = record.counts
    training_studies = _integer(counts.get("training_studies"), "counts.training_studies", minimum=1)
    held_out_studies = _integer(counts.get("held_out_studies"), "counts.held_out_studies", minimum=1)
    if held_out_studies != len(record.held_out_study_uids):
        raise SupervisionComparisonError("held-out count disagrees with the recorded identities")
    if training_studies != len(record.training_study_uids):
        raise SupervisionComparisonError("training count disagrees with the recorded identities")
    optimizer = _mapping(record.config.get("optimization"), "training configuration optimization")
    expected_steps = _integer(optimizer.get("successful_optimizer_steps"), "configured successful steps", minimum=1)
    successful = _integer(counts.get("successful_optimizer_steps"), "counts.successful_optimizer_steps", minimum=1)
    scheduler_steps = _integer(counts.get("scheduler_steps"), "counts.scheduler_steps", minimum=0)
    attempted = _integer(counts.get("attempted_batches"), "counts.attempted_batches", minimum=1)
    skipped = _integer(counts.get("skipped_all_masked_batches"), "counts.skipped_all_masked_batches", minimum=0)
    batch_size = _integer(optimizer.get("batch_size"), "configured batch size", minimum=1)
    if successful != expected_steps or scheduler_steps != successful:
        raise SupervisionComparisonError("training arms do not use the configured successful-step budget")
    if attempted - skipped != successful:
        raise SupervisionComparisonError("attempted, skipped and successful batch counts do not reconcile")

    sample_exposures = _integer(counts.get("sample_exposures"), "counts.sample_exposures", minimum=1)
    supervised_exposures = _integer(
        counts.get("supervised_entry_exposures"), "counts.supervised_entry_exposures", minimum=1
    )
    expected_sample_exposures = attempted * batch_size * (
        2 if report_arm and additive_report_arm else 1
    )
    if sample_exposures != expected_sample_exposures:
        raise SupervisionComparisonError("sample exposure count does not match the declared batches")
    raw_by_source = _mapping(counts.get("exposures_by_source"), "counts.exposures_by_source")
    if set(raw_by_source) != set(_EXPOSURE_KINDS):
        raise SupervisionComparisonError("exposure counts must separate reference, weak and mixed sources")
    exposures: dict[str, dict[str, int]] = {}
    for kind in _EXPOSURE_KINDS:
        group = _mapping(raw_by_source[kind], f"counts.exposures_by_source.{kind}")
        if set(group) != set(_EXPOSURE_FIELDS):
            raise SupervisionComparisonError(f"{kind} exposure counts are incomplete")
        exposures[kind] = {
            name: _integer(group.get(name), f"{kind}.{name}") for name in _EXPOSURE_FIELDS
        }
    if sum(item["sample_exposures"] for item in exposures.values()) != sample_exposures:
        raise SupervisionComparisonError("source sample exposures do not reconcile with total exposure")
    if sum(item["supervised_entry_exposures"] for item in exposures.values()) != supervised_exposures:
        raise SupervisionComparisonError("source supervised-entry exposures do not reconcile")
    if any(item["supervised_entry_exposures"] > NUM_TARGETS * item["sample_exposures"] for item in exposures.values()):
        raise SupervisionComparisonError("supervised-entry exposure exceeds target capacity")

    if report_arm:
        resolved = record.supervision.get("resolved_studies")
        if not isinstance(resolved, list):
            raise SupervisionComparisonError("report run is missing resolved study sources")
        reference_count = sum(_mapping(item, "resolved supervision row").get("source_kind") == "reference" for item in resolved)
        report_count = sum(_mapping(item, "resolved supervision row").get("source_kind") == "weak" for item in resolved)
        if reference_count + report_count != training_studies:
            raise SupervisionComparisonError("report training rows do not separate gold and report-only studies")
        if report_count == 0:
            raise SupervisionComparisonError("report supervision has no report-only training studies")
        report_supervised_entries = sum(
            exposures[kind]["supervised_entry_exposures"] for kind in ("weak", "mixed")
        )
        if report_supervised_entries == 0:
            raise SupervisionComparisonError(
                "report supervision has no report-derived supervised-entry exposure"
            )
    else:
        reference_count = training_studies
        report_count = 0
        if exposures["weak"]["sample_exposures"] or exposures["mixed"]["sample_exposures"]:
            raise SupervisionComparisonError("gold-only training contains non-reference sample exposures")

    resources = record.resources
    partition = str(resources.get("requested_partition", ""))
    nodes = _integer(resources.get("requested_nodes"), "resources.requested_nodes", minimum=1)
    cpus = _integer(resources.get("requested_cpus"), "resources.requested_cpus", minimum=1)
    memory = _integer(resources.get("requested_memory_gb"), "resources.requested_memory_gb", minimum=1)
    wall_time = _integer(resources.get("requested_wall_time_hours"), "resources.requested_wall_time_hours", minimum=1)
    if partition != "cluster1-cpu" or nodes != 1 or cpus > 16 or memory > 64 or wall_time > 6:
        raise SupervisionComparisonError("training resources exceed the autonomous CPU allocation envelope")
    if resources.get("exclusive") is not False:
        raise SupervisionComparisonError("training run requested an exclusive allocation")
    if record.config.get("device") != "cpu":
        raise SupervisionComparisonError("report supervision comparison requires CPU-only runs")
    job_id = resources.get("slurm_attempt_id")
    if job_id is None or not str(job_id).strip():
        raise SupervisionComparisonError("training run is missing its Slurm job identity")
    checkpoint_path = record.path.parent / "checkpoints" / "final.pt"
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    compute_path = record.path.parent.with_name(f"{record.path.parent.name}.compute.json")
    try:
        compute = _mapping(json.loads(compute_path.read_text(encoding="utf-8")), "compute evidence")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SupervisionComparisonError(
            f"training run is missing its observed compute evidence: {compute_path}"
        ) from exc
    if compute.get("schema_version") != "rsna-knee-observed-compute-v1":
        raise SupervisionComparisonError("observed compute evidence has an unsupported schema")

    def measured_float(field: str) -> float:
        raw = compute.get(field)
        if isinstance(raw, bool):
            raise SupervisionComparisonError(f"observed compute {field} must be numeric")
        try:
            value = float(cast(Any, raw))
        except (TypeError, ValueError, OverflowError) as exc:
            raise SupervisionComparisonError(f"observed compute {field} must be numeric") from exc
        if not math.isfinite(value) or value < 0:
            raise SupervisionComparisonError(f"observed compute {field} must be finite and non-negative")
        return value

    elapsed_seconds = measured_float("elapsed_seconds")
    user_cpu_seconds = measured_float("user_cpu_seconds")
    system_cpu_seconds = measured_float("system_cpu_seconds")
    peak_rss_kb = _integer(compute.get("peak_rss_kb"), "observed compute peak_rss_kb", minimum=1)
    if elapsed_seconds <= 0:
        raise SupervisionComparisonError("observed compute elapsed_seconds must be positive")
    if elapsed_seconds > wall_time * 60 * 60:
        raise SupervisionComparisonError("observed runtime exceeds the requested wall-time limit")
    if peak_rss_kb > memory * 1024 * 1024:
        raise SupervisionComparisonError("observed peak RSS exceeds the requested memory limit")
    measured_cpu_seconds = user_cpu_seconds + system_cpu_seconds
    cpu_time_tolerance = max(1.0, elapsed_seconds * 0.01)
    if measured_cpu_seconds > cpus * elapsed_seconds + cpu_time_tolerance:
        raise SupervisionComparisonError("observed CPU time exceeds the requested CPU allocation")

    gold_sequence = counts.get("gold_exposure_sequence_sha256")
    weak_sequence = counts.get("weak_exposure_sequence_sha256")
    initial_model_hash = counts.get("initial_model_state_sha256")
    for value, name in (
        (gold_sequence, "gold exposure sequence SHA-256"),
        (initial_model_hash, "initial model state SHA-256"),
    ):
        if value is not None and (not isinstance(value, str) or not _DIGEST_RE.fullmatch(value)):
            raise SupervisionComparisonError(f"{name} is invalid")
    if weak_sequence is not None and (
        not isinstance(weak_sequence, str) or not _DIGEST_RE.fullmatch(weak_sequence)
    ):
        raise SupervisionComparisonError("weak exposure sequence SHA-256 is invalid")

    return ArmExposureEvidence(
        run_id=record.run_id,
        training_studies=training_studies,
        reference_training_studies=reference_count,
        report_training_studies=report_count,
        successful_optimizer_steps=successful,
        attempted_batches=attempted,
        skipped_all_masked_batches=skipped,
        sample_exposures=sample_exposures,
        supervised_entry_exposures=supervised_exposures,
        exposures_by_source=exposures,
        requested_partition=partition,
        requested_nodes=nodes,
        requested_cpus=cpus,
        requested_memory_gb=memory,
        requested_wall_time_hours=wall_time,
        exclusive=False,
        slurm_job_id=str(job_id),
        final_checkpoint_sha256=checkpoint_sha256,
        observed_elapsed_seconds=elapsed_seconds,
        observed_user_cpu_seconds=user_cpu_seconds,
        observed_system_cpu_seconds=system_cpu_seconds,
        observed_peak_rss_kb=peak_rss_kb,
        gold_exposure_sequence_sha256=cast(str | None, gold_sequence),
        weak_exposure_sequence_sha256=cast(str | None, weak_sequence),
        initial_model_state_sha256=cast(str | None, initial_model_hash),
    )


def _paired_interval(
    values: Sequence[float],
    *,
    undefined_resamples: int,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> PairedBootstrapInterval:
    if not values:
        return PairedBootstrapInterval(
            median=None,
            lower=None,
            upper=None,
            defined_resamples=0,
            undefined_resamples=undefined_resamples,
            resamples=resamples,
            seed=seed,
            confidence_level=confidence_level,
        )
    alpha = (1.0 - confidence_level) / 2.0
    lower, median, upper = np.quantile(np.asarray(values, dtype=np.float64), [alpha, 0.5, 1.0 - alpha])
    return PairedBootstrapInterval(
        median=float(median),
        lower=float(lower),
        upper=float(upper),
        defined_resamples=len(values),
        undefined_resamples=undefined_resamples,
        resamples=resamples,
        seed=seed,
        confidence_level=confidence_level,
    )


def _paired_bootstrap(
    gold_predictions: Mapping[str, Sequence[float]],
    report_predictions: Mapping[str, Sequence[float]],
    manifest: FoldManifest,
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> tuple[PairedBootstrapInterval, tuple[PairedBootstrapInterval, ...]]:
    gold_uids = tuple(sorted(_gold_uid_set(manifest)))
    memberships = {item.study_instance_uid: item for item in manifest.gold_memberships}
    labels = np.asarray([memberships[uid].reference_labels for uid in gold_uids], dtype=np.int8)
    gold_scores = np.asarray([gold_predictions[uid] for uid in gold_uids], dtype=np.float64)
    report_scores = np.asarray([report_predictions[uid] for uid in gold_uids], dtype=np.float64)
    if gold_scores.shape != labels.shape or report_scores.shape != labels.shape:
        raise SupervisionComparisonError("paired bootstrap inputs do not align with the gold cohort")

    rng = np.random.default_rng(seed)
    pooled_deltas: list[float] = []
    target_deltas: list[list[float]] = [[] for _ in TARGET_COLUMNS]
    for _ in range(resamples):
        sample = rng.integers(0, len(gold_uids), size=len(gold_uids))
        deltas: list[float | None] = []
        for index in range(NUM_TARGETS):
            gold_auc = roc_auc_score(labels[sample, index], gold_scores[sample, index])
            report_auc = roc_auc_score(labels[sample, index], report_scores[sample, index])
            delta = None if gold_auc is None or report_auc is None else report_auc - gold_auc
            deltas.append(delta)
            if delta is not None:
                target_deltas[index].append(float(delta))
        if all(delta is not None for delta in deltas):
            pooled_deltas.append(float(np.mean([cast(float, item) for item in deltas])))

    pooled_interval = _paired_interval(
        pooled_deltas,
        undefined_resamples=resamples - len(pooled_deltas),
        resamples=resamples,
        seed=seed,
        confidence_level=confidence_level,
    )
    target_intervals = tuple(
        _paired_interval(
            values,
            undefined_resamples=resamples - len(values),
            resamples=resamples,
            seed=seed,
            confidence_level=confidence_level,
        )
        for values in target_deltas
    )
    return pooled_interval, target_intervals


def _gold_uid_set(manifest: FoldManifest) -> set[str]:
    return {item.study_instance_uid for item in manifest.gold_memberships}


def compare_supervision_oof(
    gold_only_run_paths: Sequence[str | Path],
    report_supervised_run_paths: Sequence[str | Path],
    fold_manifest: FoldManifest,
    *,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 20260918,
    bootstrap_confidence_level: float = 0.95,
    require_additive_gold_exposure: bool = False,
    additive_design_id: str | None = None,
    weak_sampling_policy: str | None = None,
    report_ticket_number: int | None = None,
) -> SupervisionComparison:
    """Validate and compare matched three-fold reference/report OOF artifacts."""

    if fold_manifest.superseded:
        raise SupervisionComparisonError("superseded fold manifests cannot support Ticket 18")
    if bootstrap_resamples <= 0 or bootstrap_seed < 0:
        raise SupervisionComparisonError("bootstrap resamples must be positive and seed non-negative")
    if not 0.0 < bootstrap_confidence_level < 1.0:
        raise SupervisionComparisonError("bootstrap confidence level must be between zero and one")
    selected_design = additive_design_id or _TICKET18_ADDITIVE_DESIGN
    selected_weak_sampling = weak_sampling_policy or _TICKET18_WEAK_SAMPLING_POLICY
    selected_ticket = (
        18 if require_additive_gold_exposure else 17
    ) if report_ticket_number is None else report_ticket_number
    if selected_ticket not in {17, 18, 36}:
        raise SupervisionComparisonError("unsupported report-supervision ticket identity")
    try:
        gold_oof = aggregate_oof_run_records(
            gold_only_run_paths,
            fold_manifest,
            expected_supervision="reference",
        )
        report_oof = aggregate_oof_run_records(
            report_supervised_run_paths,
            fold_manifest,
            expected_supervision="report",
            expected_report_ticket=selected_ticket,
        )
        gold_records = tuple(
            load_oof_run_record(path, expected_supervision="reference")
            for path in gold_only_run_paths
        )
        report_records = tuple(
            load_oof_run_record(
                path,
                expected_supervision="report",
                expected_report_ticket=selected_ticket,
            )
            for path in report_supervised_run_paths
        )
    except OOFArtifactError as exc:
        raise SupervisionComparisonError(str(exc)) from exc

    all_records = (*gold_records, *report_records)
    if len({record.path for record in all_records}) != len(all_records):
        raise SupervisionComparisonError("a fold run cannot be used in both supervision arms")
    if {record.fold for record in gold_records} != set(FOLD_IDS) or {record.fold for record in report_records} != set(FOLD_IDS):
        raise SupervisionComparisonError("each supervision arm must contain exactly one run per fold")
    if len({record.run_id for record in all_records}) != len(all_records):
        raise SupervisionComparisonError("run identities must be unique across both arms")

    configs = [dict(record.config) for record in all_records]
    if any(config != configs[0] for config in configs[1:]):
        raise SupervisionComparisonError("training configuration differs between supervision arms or folds")
    fingerprints = {record.config_fingerprint for record in all_records}
    if len(fingerprints) != 1:
        raise SupervisionComparisonError("training configuration fingerprint differs between arms")
    revisions = {
        record.environment.get("code_revision")
        for record in all_records
    }
    if len(revisions) != 1 or None in revisions:
        raise SupervisionComparisonError("training code revision is missing or differs between arms")
    for field, label in (
        ("fold_manifest_checksum", "fold manifest"),
        ("source_version", "source version"),
        ("source_fingerprint", "source fingerprint"),
        ("integration_manifest_checksum", "image integration"),
    ):
        values = {getattr(record, field) for record in all_records}
        if len(values) != 1 or None in values:
            raise SupervisionComparisonError(f"{label} identity differs between arms or folds")
    if gold_oof.fold_manifest_checksum != report_oof.fold_manifest_checksum:
        raise SupervisionComparisonError("gold and report runs use different fold manifests")

    gold_by_fold = {record.fold: record for record in gold_records}
    report_by_fold = {record.fold: record for record in report_records}
    gold_exposures = {
        fold: _arm_exposure(gold_by_fold[fold], report_arm=False) for fold in FOLD_IDS
    }
    report_exposures = {
        fold: _arm_exposure(
            report_by_fold[fold],
            report_arm=True,
            additive_report_arm=require_additive_gold_exposure,
        )
        for fold in FOLD_IDS
    }
    if require_additive_gold_exposure:
        for fold in FOLD_IDS:
            gold_record = gold_by_fold[fold]
            report_record = report_by_fold[fold]
            gold = gold_exposures[fold]
            report = report_exposures[fold]
            if gold_record.counts.get("training_design") != selected_design:
                raise SupervisionComparisonError("gold control has a different additive training design")
            if report_record.counts.get("training_design") != selected_design:
                raise SupervisionComparisonError("report treatment has a different additive training design")
            if report_record.counts.get("weak_sampling_policy") != selected_weak_sampling:
                raise SupervisionComparisonError("report treatment uses an undeclared weak sampling policy")
            if (
                gold.exposures_by_source["reference"]
                != report.exposures_by_source["reference"]
            ):
                raise SupervisionComparisonError(
                    f"fold {fold} gold sample or supervised-entry exposures differ between arms"
                )
            if (
                gold.gold_exposure_sequence_sha256 is None
                or report.gold_exposure_sequence_sha256 != gold.gold_exposure_sequence_sha256
            ):
                raise SupervisionComparisonError(
                    f"fold {fold} ordered gold exposure sequence differs between arms"
                )
            if not report.weak_exposure_sequence_sha256:
                raise SupervisionComparisonError(
                    f"fold {fold} report treatment has no weak sampling sequence fingerprint"
                )
            if (
                gold.initial_model_state_sha256 is None
                or report.initial_model_state_sha256 != gold.initial_model_state_sha256
            ):
                raise SupervisionComparisonError(
                    f"fold {fold} model initialization differs between arms"
                )
            gold_batch = _integer(
                _mapping(gold_record.config.get("optimization"), "gold optimization config").get("batch_size"),
                "gold batch size",
                minimum=1,
            )
            if report.exposures_by_source["reference"]["sample_exposures"] != (
                gold.successful_optimizer_steps * gold_batch
            ):
                raise SupervisionComparisonError(
                    f"fold {fold} gold reference exposure does not match the fixed-step batch sequence"
                )
            if report.exposures_by_source["weak"]["sample_exposures"] != (
                report.successful_optimizer_steps * gold_batch
            ):
                raise SupervisionComparisonError(
                    f"fold {fold} weak exposure does not match the declared microbatch size"
                )
            if report.successful_optimizer_steps != gold.successful_optimizer_steps:
                raise SupervisionComparisonError(
                    f"fold {fold} optimizer-step budgets differ between arms"
                )
    report_evidence = tuple(
        _load_report_evidence(
            report_by_fold[fold],
            additive_design=require_additive_gold_exposure,
            expected_ticket=selected_ticket,
        )
        for fold in FOLD_IDS
    )
    artifact_hashes = {item[0] for item in report_evidence}
    semantic_digests = {item[1] for item in report_evidence}
    if len(artifact_hashes) != 1 or len(semantic_digests) != 1:
        raise SupervisionComparisonError("report folds do not share one frozen extraction artifact")

    gold_predictions = {
        uid: scores for record in gold_records for uid, scores in record.predictions.items()
    }
    report_predictions = {
        uid: scores for record in report_records for uid, scores in record.predictions.items()
    }
    pooled_interval, target_intervals = _paired_bootstrap(
        gold_predictions,
        report_predictions,
        fold_manifest,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        confidence_level=bootstrap_confidence_level,
    )
    gold_targets = {item.target: item for item in gold_oof.pooled_targets}
    report_targets = {item.target: item for item in report_oof.pooled_targets}
    target_comparison_items: list[TargetAucComparison] = []
    for index, target in enumerate(TARGET_COLUMNS):
        gold_auc = gold_targets[target].auc
        report_auc = report_targets[target].auc
        delta_auc = (
            None
            if gold_auc is None or report_auc is None
            else report_auc - gold_auc
        )
        target_comparison_items.append(
            TargetAucComparison(
                target=target,
                gold_only_auc=gold_auc,
                report_supervised_auc=report_auc,
                delta_auc=delta_auc,
                positive_count=gold_targets[target].positive_count,
                negative_count=gold_targets[target].negative_count,
                uncertainty=target_intervals[index],
            )
        )
    target_comparisons = tuple(target_comparison_items)
    folds: list[FoldSupervisionComparison] = []
    positive_folds = 0
    negative_folds = 0
    unchanged_folds = 0
    undefined_folds = 0
    for fold in FOLD_IDS:
        gold_metric = gold_oof.folds[fold]
        report_metric = report_oof.folds[fold]
        delta = (
            None
            if gold_metric.macro_auc is None or report_metric.macro_auc is None
            else report_metric.macro_auc - gold_metric.macro_auc
        )
        if delta is None:
            undefined_folds += 1
        elif delta > 0:
            positive_folds += 1
        elif delta < 0:
            negative_folds += 1
        else:
            unchanged_folds += 1
        folds.append(
            FoldSupervisionComparison(
                fold=fold,
                gold_only_auc=gold_metric.macro_auc,
                report_supervised_auc=report_metric.macro_auc,
                macro_auc_delta=delta,
                gold_only_exposure=gold_exposures[fold],
                report_supervised_exposure=report_exposures[fold],
            )
        )
    if pooled_interval.lower is not None and pooled_interval.lower > 0 and positive_folds >= 2:
        interpretation = "paired evidence favors report supervision in this development-CV comparison"
    elif pooled_interval.upper is not None and pooled_interval.upper < 0 and negative_folds >= 2:
        interpretation = "paired evidence is negative for report supervision in this development-CV comparison"
    else:
        interpretation = "inconclusive; no report-supervision benefit is claimed"

    first_weak = report_by_fold[0].supervision
    return SupervisionComparison(
        gold_only=gold_oof,
        report_supervised=report_oof,
        pooled_macro_auc_delta=(
            None
            if gold_oof.pooled_macro_auc is None or report_oof.pooled_macro_auc is None
            else report_oof.pooled_macro_auc - gold_oof.pooled_macro_auc
        ),
        macro_auc_delta_interval=pooled_interval,
        pooled_targets=target_comparisons,
        folds=tuple(folds),
        config_fingerprint=next(iter(fingerprints)),
        code_revision=cast(str, next(iter(revisions))),
        source_version=cast(str, next(iter({record.source_version for record in all_records}))),
        source_fingerprint=cast(str, next(iter({record.source_fingerprint for record in all_records}))),
        integration_manifest_checksum=cast(
            str, next(iter({record.integration_manifest_checksum for record in all_records}))
        ),
        fold_manifest_checksum=fold_manifest.manifest_checksum,
        report_artifact_sha256=next(iter(artifact_hashes)),
        report_semantic_digest=next(iter(semantic_digests)),
        weak_label_version=str(first_weak.get("weak_label_version")),
        mixture_policy=str(first_weak.get("mixture_policy")),
        positive_folds=positive_folds,
        negative_folds=negative_folds,
        unchanged_folds=unchanged_folds,
        undefined_folds=undefined_folds,
        interpretation=interpretation,
        additive_design=require_additive_gold_exposure,
        report_extractor=str(report_by_fold[0].supervision.get("source_id", "unknown report extractor")),
        report_ticket_number=selected_ticket,
    )


def _format(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"


def _format_confidence_level(value: float) -> str:
    return f"{value * 100:.17g}%"


def render_supervision_comparison_markdown(comparison: SupervisionComparison) -> str:
    interval = comparison.macro_auc_delta_interval
    title = (
        "# Ticket 36 — Expanded report-supervision OOF comparison"
        if comparison.report_ticket_number == 36
        else (
            "# Ticket 18 — Additive report-supervision OOF comparison v2"
            if comparison.additive_design
            else "# Ticket 18 — Controlled report-supervision OOF comparison"
        )
    )
    lines = [
        title,
        "",
        "The gold-only and gold-plus-report arms use the same image model, folds, initialization, preprocessing, optimizer, loss, successful-step budget, and final-checkpoint evaluation. The report arm uses the frozen "
        + comparison.report_extractor
        + " source and reference-precedence mixture policy.",
        "",
        f"- Gold reference predictions: {comparison.prediction_count}",
        f"- Fold manifest: `{comparison.fold_manifest_checksum}`",
        f"- Shared training configuration: `{comparison.config_fingerprint}`",
        f"- Code revision: `{comparison.code_revision}`",
        f"- Frozen report artifact SHA-256: `{comparison.report_artifact_sha256}`",
        f"- Report semantic digest: `{comparison.report_semantic_digest}`",
        f"- Weak-label policy: `{comparison.weak_label_version}`",
        f"- Mixture policy: `{comparison.mixture_policy}`",
        "- All study, series, report, and source-row identifiers are omitted.",
        "",
        "## Pooled result",
        "",
        "| Arm | Pooled 12-target macro ROC-AUC | Mean fold macro AUC | Fold macro AUC standard deviation | Minimum fold macro AUC | Maximum fold macro AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Gold-only | {_format(comparison.gold_only.pooled_macro_auc)} | "
        f"{_format(comparison.gold_only.mean_fold_macro_auc)} | "
        f"{_format(comparison.gold_only.fold_macro_auc_std)} | "
        f"{_format(comparison.gold_only.fold_macro_auc_min)} | "
        f"{_format(comparison.gold_only.fold_macro_auc_max)} |",
        f"| Gold + admitted reports | {_format(comparison.report_supervised.pooled_macro_auc)} | "
        f"{_format(comparison.report_supervised.mean_fold_macro_auc)} | "
        f"{_format(comparison.report_supervised.fold_macro_auc_std)} | "
        f"{_format(comparison.report_supervised.fold_macro_auc_min)} | "
        f"{_format(comparison.report_supervised.fold_macro_auc_max)} |",
        f"| Delta (report − gold) | {_format(comparison.pooled_macro_auc_delta)} | | | | |",
        "",
        f"Paired study bootstrap {_format_confidence_level(interval.confidence_level)} interval for the macro delta: `{_format(interval.lower)}` to `{_format(interval.upper)}`; {interval.defined_resamples:,} defined and {interval.undefined_resamples:,} undefined resamples out of {interval.resamples:,} (seed {interval.seed}).",
        "",
        "## Per-target OOF results",
        "",
        f"| Target | Gold AUC | Report AUC | Delta | Positive | Negative | Paired {_format_confidence_level(interval.confidence_level)} interval |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in comparison.pooled_targets:
        lines.append(
            f"| {item.target} | {_format(item.gold_only_auc)} | {_format(item.report_supervised_auc)} | "
            f"{_format(item.delta_auc)} | {item.positive_count} | {item.negative_count} | "
            f"{_format(item.uncertainty.lower)} to {_format(item.uncertainty.upper)} |"
        )
    lines.extend(
        [
            "",
            "## Fold consistency and training exposure",
            "",
            "| Fold | Gold macro AUC | Report macro AUC | Delta | Gold studies / exposures / supervised entries | Report studies (gold + weak) / exposures / supervised entries |",
            "| ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for fold_comparison in comparison.folds:
        gold = fold_comparison.gold_only_exposure
        report = fold_comparison.report_supervised_exposure
        lines.append(
            f"| {fold_comparison.fold} | {_format(fold_comparison.gold_only_auc)} | "
            f"{_format(fold_comparison.report_supervised_auc)} | "
            f"{_format(fold_comparison.macro_auc_delta)} | {gold.training_studies} / {gold.sample_exposures} / "
            f"{gold.supervised_entry_exposures} | {report.training_studies} "
            f"({report.reference_training_studies} + {report.report_training_studies}) / "
            f"{report.sample_exposures} / {report.supervised_entry_exposures} |"
        )
    lines.extend(
        [
            "",
            "### Source-stratified training exposure",
            "",
            "Sample and supervised-entry exposures are listed as `samples / entries` for each source type.",
            "",
            "| Fold | Arm | Training studies (reference + weak) | Reference sample / entry exposures | Weak sample / entry exposures | Mixed sample / entry exposures |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for fold_comparison in comparison.folds:
        for arm_name, evidence in (
            ("Gold-only", fold_comparison.gold_only_exposure),
            ("Gold + clean reports", fold_comparison.report_supervised_exposure),
        ):
            reference_exposure = evidence.exposures_by_source["reference"]
            weak_exposure = evidence.exposures_by_source["weak"]
            mixed_exposure = evidence.exposures_by_source["mixed"]
            lines.append(
                f"| {fold_comparison.fold} | {arm_name} | "
                f"{evidence.training_studies} "
                f"({evidence.reference_training_studies} + {evidence.report_training_studies}) | "
                f"{reference_exposure['sample_exposures']} / "
                f"{reference_exposure['supervised_entry_exposures']} | "
                f"{weak_exposure['sample_exposures']} / "
                f"{weak_exposure['supervised_entry_exposures']} | "
                f"{mixed_exposure['sample_exposures']} / "
                f"{mixed_exposure['supervised_entry_exposures']} |"
            )
    lines.extend(
        [
            "",
            "## Slurm compute requested",
            "",
            "Allocation values are Slurm requests. Timing and peak memory are measured around the full training CLI process.",
            "",
            "| Fold | Arm | Partition | Nodes | CPUs | Memory | Wall time | Exclusive | Job | Successful steps / attempts / skipped | Elapsed / user CPU / system CPU | Peak RSS |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | ---: |",
        ]
    )
    for fold_comparison in comparison.folds:
        for arm_name, evidence in (
            ("Gold-only", fold_comparison.gold_only_exposure),
            ("Gold + reports", fold_comparison.report_supervised_exposure),
        ):
            lines.append(
                f"| {fold_comparison.fold} | {arm_name} | {evidence.requested_partition} | "
                f"{evidence.requested_nodes} | {evidence.requested_cpus} | "
                f"{evidence.requested_memory_gb} GB | {evidence.requested_wall_time_hours} h | "
                f"{'yes' if evidence.exclusive else 'no'} | `{evidence.slurm_job_id}` | "
                f"{evidence.successful_optimizer_steps} / {evidence.attempted_batches} / "
                f"{evidence.skipped_all_masked_batches} | "
                f"{evidence.observed_elapsed_seconds:.1f} s / {evidence.observed_user_cpu_seconds:.1f} s / "
                f"{evidence.observed_system_cpu_seconds:.1f} s | {evidence.observed_peak_rss_kb} KB |"
            )
    lines.extend(
        [
            "",
            "## Paired run identities and fingerprints",
            "",
            "The gold exposure digest hashes the ordered sequence of sample identities without disclosing those identities. The initial-state digest covers every initialized model tensor. The weak digest hashes the added report-only sequence.",
            "",
            "| Fold | Arm | Run ID | Gold exposure sequence SHA-256 | Weak exposure sequence SHA-256 | Initial model state SHA-256 | Final checkpoint SHA-256 |",
            "| ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for fold_comparison in comparison.folds:
        for arm_name, evidence in (
            ("Gold-only", fold_comparison.gold_only_exposure),
            ("Gold + clean reports", fold_comparison.report_supervised_exposure),
        ):
            lines.append(
                f"| {fold_comparison.fold} | {arm_name} | `{evidence.run_id}` | "
                f"`{evidence.gold_exposure_sequence_sha256 or 'undefined'}` | "
                f"`{evidence.weak_exposure_sequence_sha256 or '—'}` | "
                f"`{evidence.initial_model_state_sha256 or 'undefined'}` | "
                f"`{evidence.final_checkpoint_sha256}` |"
            )
    lines.extend(
        [
            "",
            f"Fold deltas: {comparison.positive_folds} positive, {comparison.negative_folds} negative, "
            f"{comparison.unchanged_folds} unchanged, {comparison.undefined_folds} undefined.",
            "",
            "## Interpretation",
            "",
            comparison.interpretation + ".",
            "",
            "This is repeated development CV for model selection, not an untouched internal test. Study bootstrap uncertainty is reported because an attested patient/group key is unavailable. Undefined AUCs remain undefined, and undefined bootstrap resamples are counted rather than imputed.",
        ]
    )
    if comparison.additive_design:
        lines.extend(
            [
                "",
                "The report-supervised arm retains the exact ordered gold exposure sequence and adds one deterministic two-study weak microbatch to each gold batch. Gold and weak entries are averaged together by the existing masked BCE, with one optimizer and scheduler step per batch.",
            ]
        )
    lines.extend(
        [
            "",
            "No focal, class-weighted, or soft-label loss was used. The gold-only training path remains available regardless of this result.",
            "",
        ]
    )
    return "\n".join(lines)


def _ensure_output_outside_runs(
    destinations: Sequence[Path], source_run_paths: Sequence[str | Path]
) -> None:
    if not source_run_paths:
        raise SupervisionComparisonError("source run paths are required to protect input artifacts")
    directories = tuple(
        (Path(path).parent if Path(path).name == "run.json" else Path(path)).resolve()
        for path in source_run_paths
    )
    for destination in destinations:
        resolved = destination.resolve()
        for directory in directories:
            try:
                resolved.relative_to(directory)
            except ValueError:
                continue
            raise SupervisionComparisonError("comparison output cannot be written inside a source run")


def _atomic_text_write(path: str | Path, content: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-", dir=destination.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
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


def write_supervision_comparison(
    comparison: SupervisionComparison,
    *,
    public_path: str | Path,
    markdown_path: str | Path,
    source_run_paths: Sequence[str | Path],
) -> tuple[Path, Path]:
    destinations = (Path(public_path), Path(markdown_path))
    _ensure_output_outside_runs(destinations, source_run_paths)
    atomic_json_write(destinations[0], comparison.to_public_dict())
    _atomic_text_write(destinations[1], render_supervision_comparison_markdown(comparison))
    return destinations[0], destinations[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--gold-only-run", type=Path, action="append", required=True)
    parser.add_argument("--report-supervised-run", type=Path, action="append", required=True)
    parser.add_argument("--public-json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260918)
    parser.add_argument("--bootstrap-confidence-level", type=float, default=0.95)
    parser.add_argument("--require-additive-gold-exposure", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_fold_manifest(args.fold_manifest)
        comparison = compare_supervision_oof(
            args.gold_only_run,
            args.report_supervised_run,
            manifest,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_confidence_level=args.bootstrap_confidence_level,
            require_additive_gold_exposure=args.require_additive_gold_exposure,
        )
        write_supervision_comparison(
            comparison,
            public_path=args.public_json,
            markdown_path=args.markdown,
            source_run_paths=(*args.gold_only_run, *args.report_supervised_run),
        )
        print(json.dumps(comparison.to_public_dict(), indent=2, sort_keys=True))
    except (OSError, RuntimeError, ValueError, OOFArtifactError, SupervisionComparisonError) as exc:
        print(f"Ticket 18 supervision comparison failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArmExposureEvidence",
    "FoldSupervisionComparison",
    "PairedBootstrapInterval",
    "SupervisionComparison",
    "SupervisionComparisonError",
    "TargetAucComparison",
    "compare_supervision_oof",
    "main",
    "render_supervision_comparison_markdown",
    "write_supervision_comparison",
]
