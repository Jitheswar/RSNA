"""Fold-isolated Ticket 22 soft-target MRI student experiment."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence, cast

import numpy as np
import torch

from .archive import atomic_json_write
from .evaluation import predict_training_examples
from .domain import StudyRecord
from .folds import FOLD_IDS, FoldManifest
from .image_model_comparison import compare_image_model_predictions
from .image_variant import TICKET19_VARIANT, image_model_training_config
from .inference import InferenceEngine
from .model import HierarchicalStudyModel
from .oof import OOFRunRecord, load_oof_run_record
from .real_inference import IntegrationManifest
from .series_cache import SeriesCache
from .training import (
    CpuPreflightResult,
    SupervisionProvenance,
    TrainingConfig,
    TrainingDataError,
    TrainingExample,
    TrainingResult,
    build_fold_held_out_examples,
    build_fold_pseudo_candidate_examples,
    build_fold_training_examples,
    checkpoint_sha256,
    fingerprint_training_examples,
    load_training_checkpoint,
    train_fixed_budget,
    train_one_fold,
)
from .targets import NUM_TARGETS


TICKET22_SCHEMA_VERSION = "rsna-knee-ticket22-distillation-v1"
TICKET22_SOFT_LABEL_VERSION = "ticket22-teacher-soft-probabilities-v1"
TICKET22_STUDENT_CHANNELS = (4, 8, 16)
TICKET22_STEPS = 100
TICKET22_SOFT_BATCH_SEED_OFFSET = 22_022_022
TICKET22_BOOTSTRAP_SEED = 20260923
TICKET22_BOOTSTRAP_RESAMPLES = 2000


class DistillationExperimentError(ValueError):
    """Raised when the declared Ticket 22 experiment is incomplete or contaminated."""


class _Ticket22FoldEnsemble(torch.nn.Module):
    """Match candidate inference by preprocessing once and averaging fold probabilities."""

    def __init__(self, models: Sequence[HierarchicalStudyModel]) -> None:
        super().__init__()
        if not models:
            raise DistillationExperimentError("Ticket 22 inference requires all fold models")
        first_config = models[0].config
        if any(model.config != first_config for model in models[1:]):
            raise DistillationExperimentError("Ticket 22 inference fold model configurations differ")
        self.models = torch.nn.ModuleList(models)
        self.config = first_config

    def predict_proba(
        self,
        images: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> torch.Tensor:
        fold_scores = tuple(
            model.predict_proba(images, slice_mask, series_mask)
            for model in self.models
        )
        return torch.stack(fold_scores, dim=0).mean(dim=0).clamp(0.0, 1.0)


@dataclass(frozen=True)
class Ticket22FoldPlan:
    """Private identity partition for one fold's teacher and student adaptation."""

    fold: int
    teacher_training_study_uids: tuple[str, ...]
    held_out_study_uids: tuple[str, ...]
    held_out_gold_study_uids: tuple[str, ...]
    eligible_soft_target_study_uids: tuple[str, ...]
    excluded_report_only_study_uids: tuple[str, ...]

    def to_private_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "teacher_training_study_uids": list(self.teacher_training_study_uids),
            "student_gold_training_study_uids": list(self.teacher_training_study_uids),
            "held_out_study_uids": list(self.held_out_study_uids),
            "held_out_gold_study_uids": list(self.held_out_gold_study_uids),
            "eligible_soft_target_study_uids": list(self.eligible_soft_target_study_uids),
            "excluded_report_only_study_uids": list(self.excluded_report_only_study_uids),
            "teacher_training_set_sha256": _uid_set_sha256(self.teacher_training_study_uids),
            "student_gold_training_set_sha256": _uid_set_sha256(self.teacher_training_study_uids),
            "held_out_study_set_sha256": _uid_set_sha256(self.held_out_study_uids),
            "held_out_gold_set_sha256": _uid_set_sha256(self.held_out_gold_study_uids),
            "eligible_soft_target_set_sha256": _uid_set_sha256(self.eligible_soft_target_study_uids),
            "excluded_report_only_set_sha256": _uid_set_sha256(self.excluded_report_only_study_uids),
            "held_out_study_count": len(self.held_out_study_uids),
            "held_out_gold_count": len(self.held_out_gold_study_uids),
            "eligible_soft_target_count": len(self.eligible_soft_target_study_uids),
            "excluded_report_only_count": len(self.excluded_report_only_study_uids),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _uid_set_sha256(uids: Sequence[str]) -> str:
    payload = json.dumps(sorted(uids), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _code_sha256(repository_root: Path) -> str:
    source_files = (
        "src/rsna_knee/distillation.py",
        "src/rsna_knee/distillation_cli.py",
        "src/rsna_knee/evaluation.py",
        "src/rsna_knee/folds.py",
        "src/rsna_knee/image_model_comparison.py",
        "src/rsna_knee/image_variant.py",
        "src/rsna_knee/inference.py",
        "src/rsna_knee/model.py",
        "src/rsna_knee/oof.py",
        "src/rsna_knee/preprocessing.py",
        "src/rsna_knee/real_inference.py",
        "src/rsna_knee/series_cache.py",
        "src/rsna_knee/training.py",
    )
    digest = hashlib.sha256()
    for relative_path in source_files:
        path = repository_root / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _teacher_config(integration_manifest: IntegrationManifest) -> TrainingConfig:
    return image_model_training_config(
        TICKET19_VARIANT,
        preprocessing_config=integration_manifest.preprocess_config,
    )


def _student_config(integration_manifest: IntegrationManifest) -> TrainingConfig:
    return TrainingConfig(
        channels=TICKET22_STUDENT_CHANNELS,
        successful_optimizer_steps=TICKET22_STEPS,
        preprocessing_config=integration_manifest.preprocess_config,
    )


def build_ticket22_fold_plan(
    fold_manifest: FoldManifest,
    fold: int,
    available_study_uids: Sequence[str],
) -> Ticket22FoldPlan:
    """Resolve one fold's gold, held-out, and eligible soft-target identities."""

    if fold not in FOLD_IDS:
        raise DistillationExperimentError(f"fold must be one of {FOLD_IDS}")
    available = tuple(available_study_uids)
    if any(not isinstance(uid, str) or not uid.strip() for uid in available):
        raise DistillationExperimentError("available study identities must be non-empty")
    if len(set(available)) != len(available):
        raise DistillationExperimentError("available study identities must be unique")
    known = set(fold_manifest.study_to_fold)
    if not set(available).issubset(known):
        raise DistillationExperimentError("available study identities are missing from the frozen folds")
    training_gold = tuple(
        sorted(item.study_instance_uid for item in fold_manifest.gold_memberships if item.fold != fold)
    )
    held_out = tuple(sorted(fold_manifest.held_out_study_uids(fold)))
    held_out_gold = tuple(
        sorted(item.study_instance_uid for item in fold_manifest.gold_memberships if item.fold == fold)
    )
    report_only = fold_manifest.report_only_memberships
    eligible = tuple(
        sorted(item.study_instance_uid for item in report_only if item.fold != fold and item.study_instance_uid in set(available))
    )
    excluded = tuple(
        sorted(item.study_instance_uid for item in report_only if item.study_instance_uid not in set(eligible))
    )
    fold_manifest.assert_adaptation_input_is_training_only(fold, training_gold)
    fold_manifest.assert_adaptation_input_is_training_only(fold, eligible)
    if set(training_gold).intersection(eligible) or set(held_out).intersection(eligible):
        raise DistillationExperimentError("soft-target candidates overlap gold training or held-out studies")
    return Ticket22FoldPlan(
        fold=fold,
        teacher_training_study_uids=training_gold,
        held_out_study_uids=held_out,
        held_out_gold_study_uids=held_out_gold,
        eligible_soft_target_study_uids=eligible,
        excluded_report_only_study_uids=excluded,
    )


def _load_ticket19_teacher(
    teacher_root: Path,
    fold: int,
    fold_manifest: FoldManifest,
    integration_manifest: IntegrationManifest,
    source_fingerprint: str,
) -> tuple[OOFRunRecord, HierarchicalStudyModel, dict[str, object]]:
    """Load and reconcile one frozen Ticket 19 teacher and its OOF predictions."""

    record = load_oof_run_record(teacher_root / "runs" / f"fold-{fold}", expected_supervision="reference")
    expected_training = tuple(
        sorted(
            item.study_instance_uid
            for item in fold_manifest.gold_memberships
            if item.fold != fold
        )
    )
    expected_held_out = tuple(
        sorted(
            item.study_instance_uid
            for item in fold_manifest.gold_memberships
            if item.fold == fold
        )
    )
    if record.fold != fold or record.fold_manifest_checksum != fold_manifest.manifest_checksum:
        raise DistillationExperimentError("Ticket 19 teacher fold identity does not match frozen folds")
    if record.integration_manifest_checksum != integration_manifest.manifest_checksum:
        raise DistillationExperimentError("Ticket 19 teacher used a different integration manifest")
    if record.source_fingerprint != source_fingerprint:
        raise DistillationExperimentError("Ticket 19 teacher used a different source fingerprint")
    if tuple(sorted(record.training_study_uids)) != expected_training:
        raise DistillationExperimentError("Ticket 19 teacher training cohort is not the declared gold partition")
    if tuple(sorted(record.held_out_study_uids)) != expected_held_out:
        raise DistillationExperimentError("Ticket 19 teacher held-out cohort is not the declared fold")
    if set(record.predictions) != set(expected_held_out):
        raise DistillationExperimentError("Ticket 19 teacher does not contain complete held-out predictions")
    model_config = record.config.get("model")
    if not isinstance(model_config, Mapping) or tuple(model_config.get("channels", ())) != (16, 32, 64):
        raise DistillationExperimentError("Ticket 19 teacher architecture is not the frozen wider encoder")
    if record.counts.get("successful_optimizer_steps") != TICKET22_STEPS:
        raise DistillationExperimentError("Ticket 19 teacher step budget differs from Ticket 22 design")
    if record.environment.get("code_revision") is None:
        raise DistillationExperimentError("Ticket 19 teacher has no pinned source revision")

    run_path = record.path
    configuration_path = run_path.parent / "configuration.json"
    checkpoint_path = run_path.parent / "checkpoints" / "final.pt"
    try:
        checkpoint_evidence = json.loads(run_path.read_text(encoding="utf-8"))
        configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DistillationExperimentError("Ticket 19 teacher records are unreadable") from exc
    if not isinstance(checkpoint_evidence, Mapping) or not isinstance(configuration, Mapping):
        raise DistillationExperimentError("Ticket 19 teacher records have invalid structure")
    checkpoint_record = checkpoint_evidence.get("checkpoint")
    if not isinstance(checkpoint_record, Mapping):
        raise DistillationExperimentError("Ticket 19 teacher record has no checkpoint checksum")
    checkpoint_digest = _sha256_file(checkpoint_path)
    if checkpoint_record.get("final_sha256") != checkpoint_digest:
        raise DistillationExperimentError("Ticket 19 teacher checkpoint checksum does not match its run record")

    config = _teacher_config(integration_manifest)
    model = HierarchicalStudyModel(config.model_config())
    load_training_checkpoint(
        checkpoint_path,
        model=model,
        expected_config=config,
        expected_fold=fold,
        expected_checkpoint_kind="final_selected",
    )
    lineage: dict[str, object] = {
        "fold": fold,
        "run_id": record.run_id,
        "code_revision": record.environment.get("code_revision"),
        "run_sha256": _sha256_file(run_path),
        "configuration_sha256": _sha256_file(configuration_path),
        "checkpoint_sha256": checkpoint_digest,
        "fold_manifest_checksum": record.fold_manifest_checksum,
        "integration_manifest_checksum": record.integration_manifest_checksum,
        "source_fingerprint": record.source_fingerprint,
        "training_study_set_sha256": _uid_set_sha256(expected_training),
        "held_out_study_set_sha256": _uid_set_sha256(expected_held_out),
        "configuration_fingerprint": record.config_fingerprint,
    }
    return record, model, lineage


def create_fold_soft_target_examples(
    candidate_examples: Sequence[TrainingExample],
    teacher_predictions: Mapping[str, Sequence[float]],
    *,
    fold_manifest: FoldManifest,
    fold: int,
    eligible_study_uids: Sequence[str],
    excluded_study_uids: Sequence[str],
    teacher_checkpoint_sha256: str,
) -> tuple[TrainingExample, ...]:
    """Bind all MRI-only teacher probabilities to the fold-scoped student pool."""

    expected = {item.study_instance_uid for item in candidate_examples}
    declared_eligible = set(eligible_study_uids)
    declared_excluded = set(excluded_study_uids)
    if not expected:
        raise DistillationExperimentError("at least one non-held-out MRI candidate is required")
    report_only_training = {
        item.study_instance_uid
        for item in fold_manifest.report_only_memberships
        if item.fold != fold
    }
    held_out_report_only = {
        item.study_instance_uid
        for item in fold_manifest.report_only_memberships
        if item.fold == fold
    }
    if not expected.issubset(report_only_training):
        raise DistillationExperimentError("held-out or non-report-only studies cannot receive soft targets")
    if expected != declared_eligible:
        raise DistillationExperimentError("soft-target example cohort does not match declared eligibility")
    if not held_out_report_only.issubset(declared_excluded):
        raise DistillationExperimentError("every held-out report-only study must be excluded from soft targets")
    if expected.intersection(declared_excluded):
        raise DistillationExperimentError("eligible and excluded soft-target cohorts overlap")
    if set(teacher_predictions) != expected:
        raise DistillationExperimentError("teacher scores do not cover exactly the eligible MRI candidates")
    if any(
        item.provenance.fold != fold
        or item.provenance.source_kind != "reference"
        or bool(np.asarray(item.supervision_mask, dtype=bool).any())
        for item in candidate_examples
    ):
        raise DistillationExperimentError("soft-target inputs must be unlabelled report-only examples for this fold")
    provenance = SupervisionProvenance(
        source_kind="pseudo",
        source_id=f"ticket22-fold-{fold}-teacher-soft-targets",
        fold=fold,
        held_out_excluded=True,
        eligible_study_uids=tuple(sorted(eligible_study_uids)),
        excluded_study_uids=tuple(sorted(excluded_study_uids)),
        mixture_policy="ticket22-reference-plus-unthresholded-teacher-soft-bce-v1",
        pseudo_label_version=TICKET22_SOFT_LABEL_VERSION,
        teacher_checkpoint_sha256=teacher_checkpoint_sha256,
    )
    result: list[TrainingExample] = []
    for example in candidate_examples:
        scores = np.asarray(teacher_predictions[example.study_instance_uid], dtype=np.float32)
        if scores.shape != (NUM_TARGETS,) or not np.isfinite(scores).all():
            raise DistillationExperimentError("teacher soft target must contain twelve finite scores")
        if (scores < 0.0).any() or (scores > 1.0).any():
            raise DistillationExperimentError("teacher soft targets must be probabilities in [0, 1]")
        result.append(
            replace(
                example,
                targets=scores.copy(),
                supervision_mask=np.ones(NUM_TARGETS, dtype=bool),
                provenance=provenance,
                target_kind="teacher_probability",
            )
        )
    return tuple(result)


def compare_ticket22_oof_predictions(
    teacher_predictions: Mapping[str, Sequence[float]],
    undistilled_student_predictions: Mapping[str, Sequence[float]],
    distilled_student_predictions: Mapping[str, Sequence[float]],
    fold_manifest: FoldManifest,
    *,
    bootstrap_resamples: int = TICKET22_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = TICKET22_BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Compare all three MRI-only arms on the same complete gold OOF cohort."""

    return {
        "undistilled_student_vs_teacher": compare_image_model_predictions(
            teacher_predictions,
            undistilled_student_predictions,
            fold_manifest,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
        "distilled_student_vs_undistilled_student": compare_image_model_predictions(
            undistilled_student_predictions,
            distilled_student_predictions,
            fold_manifest,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
        "distilled_student_vs_teacher": compare_image_model_predictions(
            teacher_predictions,
            distilled_student_predictions,
            fold_manifest,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
    }


def _private_predictions(
    arms: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, object]:
    return {
        arm: {uid: [float(score) for score in scores] for uid, scores in sorted(values.items())}
        for arm, values in sorted(arms.items())
    }


def _score_sequences(
    predictions: Mapping[str, Sequence[float]] | Mapping[str, np.ndarray],
) -> dict[str, tuple[float, ...]]:
    return {
        uid: tuple(float(score) for score in np.asarray(values, dtype=np.float64).reshape(-1))
        for uid, values in predictions.items()
    }


def _checkpoint_costs(
    results: Mapping[str, Sequence[TrainingResult]],
) -> dict[str, object]:
    evidence: dict[str, object] = {}
    for arm, folds in results.items():
        fold_items: list[dict[str, object]] = []
        total_bytes = 0
        total_parameters: int | None = None
        for fold, result in enumerate(folds):
            checkpoint = result.final_checkpoint_path
            if checkpoint is None or not checkpoint.is_file():
                raise DistillationExperimentError(f"{arm} fold {fold} has no final checkpoint")
            size = checkpoint.stat().st_size
            total_bytes += size
            parameters = sum(parameter.numel() for parameter in result.model.parameters())
            total_parameters = parameters if total_parameters is None else total_parameters
            if parameters != total_parameters:
                raise DistillationExperimentError(f"{arm} fold parameter counts differ")
            fold_items.append(
                {
                    "fold": fold,
                    "checkpoint_bytes": size,
                    "checkpoint_sha256": checkpoint_sha256(checkpoint),
                    "parameter_count": parameters,
                    "optimizer_steps": result.successful_optimizer_steps,
                }
            )
        evidence[arm] = {
            "folds": fold_items,
            "total_checkpoint_bytes": total_bytes,
            "total_parameters_per_fold": total_parameters,
            "three_fold_checkpoint_bundle_bytes": total_bytes,
        }
    return evidence


def _teacher_elapsed_seconds(run_path: Path) -> float | None:
    compute_path = run_path.parent.parent / f"{run_path.parent.name}.compute.json"
    try:
        value = json.loads(compute_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping):
        return None
    elapsed = value.get("elapsed_seconds")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        return None
    if not math.isfinite(float(elapsed)) or elapsed < 0:
        return None
    return float(elapsed)


def _measure_ensemble_runtime(
    fold_checkpoints: Mapping[str, Sequence[Path]],
    arm_configs: Mapping[str, TrainingConfig],
    studies: Sequence[StudyRecord],
    integration_manifest: IntegrationManifest,
    series_cache: SeriesCache | None,
    input_loading_seconds: float,
) -> dict[str, object]:
    """Measure candidate-style ensemble inference from loading through model calls."""

    evidence: dict[str, object] = {}
    ordered_studies = tuple(studies)
    for arm, checkpoints in fold_checkpoints.items():
        if len(checkpoints) != len(FOLD_IDS):
            raise DistillationExperimentError(f"{arm} inference bundle must contain all three folds")
        config = arm_configs[arm]
        started = time.perf_counter()
        models: list[HierarchicalStudyModel] = []
        for fold, checkpoint in enumerate(checkpoints):
            model = HierarchicalStudyModel(config.model_config())
            load_training_checkpoint(
                checkpoint,
                model=model,
                expected_config=config,
                expected_fold=fold,
                expected_checkpoint_kind="final_selected",
            )
            models.append(model)
        engine = InferenceEngine(
            _Ticket22FoldEnsemble(models),  # type: ignore[arg-type]
            integration_manifest.preprocess_config,
            packing_config=integration_manifest.packing_config,
            series_cache=series_cache,
        )
        for study in ordered_studies:
            engine.predict_study(study)
        elapsed_seconds = time.perf_counter() - started
        evidence[arm] = {
            "three_fold_ensemble_seconds": elapsed_seconds,
            "runtime_including_shared_input_load_seconds": input_loading_seconds + elapsed_seconds,
            "ensemble_study_predictions": len(ordered_studies),
            "fold_model_predictions": len(ordered_studies) * len(checkpoints),
            "input_study_count": len(ordered_studies),
            "timed_stages": [
                "all three checkpoint loads",
                "series preprocessing once per study",
                "study packing once per study",
                "three-fold probability averaging",
                "score validation",
            ],
            "scope": "all studies selected by the private integration manifest; training-source proxy, not competition test scale",
        }
    return evidence


def run_ticket22_distillation_experiment(
    studies: Mapping[str, StudyRecord],
    fold_manifest: FoldManifest,
    integration_manifest: IntegrationManifest,
    teacher_run_root: str | Path,
    output_root: str | Path,
    *,
    code_revision: str,
    source_fingerprint: str,
    preflight: CpuPreflightResult,
    input_loading_seconds: float,
    series_cache: SeriesCache | None = None,
    requested_resources: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run the fixed three-fold teacher/student/undistilled comparison on CPU."""

    repository_root = Path(__file__).resolve().parents[2]
    destination = Path(output_root).resolve()
    teacher_root = Path(teacher_run_root).resolve()
    if destination.is_relative_to(repository_root) or teacher_root.is_relative_to(repository_root):
        raise TrainingDataError("Ticket 22 private experiment and teacher artifacts must stay outside the repository")
    if not code_revision or len(code_revision) != 40 or any(
        character not in "0123456789abcdef" for character in code_revision
    ):
        raise TrainingDataError("Ticket 22 requires the committed full Git code revision")
    if source_fingerprint != integration_manifest.archive_fingerprint:
        raise TrainingDataError("Ticket 22 source fingerprint disagrees with its integration manifest")
    if fold_manifest.superseded or fold_manifest.fold_ids != FOLD_IDS:
        raise TrainingDataError("Ticket 22 requires the canonical frozen three-fold manifest")
    if fold_manifest.source_fingerprint not in {None, source_fingerprint}:
        raise TrainingDataError("Ticket 22 source fingerprint disagrees with frozen folds")
    if fold_manifest.source_manifest_checksum != integration_manifest.source_manifest_checksum:
        raise TrainingDataError("Ticket 22 fold and integration manifests have different source checksums")
    if not preflight.passed or preflight.details.get("required_slurm") is not True:
        raise TrainingDataError("Ticket 22 requires a passed cluster1-cpu preflight")
    if not math.isfinite(input_loading_seconds) or input_loading_seconds < 0:
        raise TrainingDataError("Ticket 22 input-loading time must be finite and non-negative")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite Ticket 22 output {destination}")
    available_uids = set(integration_manifest.study_uids).intersection(studies)
    gold_uids = {item.study_instance_uid for item in fold_manifest.gold_memberships}
    if not gold_uids.issubset(available_uids):
        raise TrainingDataError("Ticket 22 input manifest does not cover every frozen gold study")
    if any(key != getattr(study, "study_instance_uid", None) for key, study in studies.items()):
        raise TrainingDataError("Ticket 22 study mapping keys must match study identities")

    teacher_config = _teacher_config(integration_manifest)
    student_config = _student_config(integration_manifest)
    teacher_records: list[OOFRunRecord] = []
    teachers: list[HierarchicalStudyModel] = []
    teacher_lineage: list[dict[str, object]] = []
    teacher_results: list[TrainingResult] = []
    for fold in FOLD_IDS:
        record, model, lineage = _load_ticket19_teacher(
            teacher_root,
            fold,
            fold_manifest,
            integration_manifest,
            source_fingerprint,
        )
        teacher_records.append(record)
        teachers.append(model)
        teacher_lineage.append(lineage)
        teacher_results.append(
            TrainingResult(
                model=model,
                config=teacher_config,
                successful_optimizer_steps=TICKET22_STEPS,
                scheduler_steps=TICKET22_STEPS,
                attempted_batches=TICKET22_STEPS,
                skipped_batches=0,
                loss_history=(),
                final_checkpoint_path=record.path.parent / "checkpoints" / "final.pt",
                fold=fold,
                predictions={
                    uid: np.asarray(values, dtype=np.float64)
                    for uid, values in record.predictions.items()
                },
                training_study_uids=record.training_study_uids,
                held_out_study_uids=record.held_out_study_uids,
                sample_exposures=cast(int, record.counts.get("sample_exposures", 0)),
            )
        )

    fold_plans = [
        build_ticket22_fold_plan(fold_manifest, fold, tuple(available_uids))
        for fold in FOLD_IDS
    ]
    fold_designs = [plan.to_private_dict() for plan in fold_plans]

    atomic_json_write(
        destination / "design.private.json",
        {
            "schema_version": TICKET22_SCHEMA_VERSION,
            "ticket": 22,
            "code_revision": code_revision,
            "code_sha256": _code_sha256(repository_root),
            "source_fingerprint": source_fingerprint,
            "fold_manifest_checksum": fold_manifest.manifest_checksum,
            "integration_manifest_checksum": integration_manifest.manifest_checksum,
            "repeat_name": fold_manifest.repeat_name,
            "teacher": {
                "parent_ticket": 19,
                "variant": TICKET19_VARIANT,
                "oof_prediction_role": "held-out evaluation only",
                "training_config": teacher_config.to_dict(),
                "fold_lineage": teacher_lineage,
            },
            "student": {
                "channels": list(TICKET22_STUDENT_CHANNELS),
                "training_config": student_config.to_dict(),
                "gold_only_steps": TICKET22_STEPS,
                "distilled_steps": TICKET22_STEPS,
                "gold_batch_size": student_config.batch_size,
                "soft_target_batch_size": student_config.batch_size,
                "soft_target_temperature": 1.0,
                "loss": "masked BCE on reference labels plus equally sized microbatches of unthresholded teacher probabilities; one Adam update per step",
                "soft_target_artifact_role": "fold-training targets; held-out outputs are not admitted",
                "sampling_policy": "independent deterministic cyclic-shuffle batches for gold and teacher-target cohorts",
                "soft_target_seed": student_config.batch_order_seed + TICKET22_SOFT_BATCH_SEED_OFFSET,
            },
            "execution_budget": {
                "partition": "cluster1-cpu",
                "nodes": 1,
                "cpus": 16,
                "requested_memory_gb": 64,
                "wall_time_hours": 6,
                "maximum_concurrent_jobs": 2,
                "exclusive": False,
                "gpu": False,
                "student_optimizer_steps_total": len(FOLD_IDS) * 2 * TICKET22_STEPS,
            },
            "inference_is_mri_only": True,
            "reports_read": False,
            "folds": fold_designs,
        },
    )
    design_sha256 = _sha256_file(destination / "design.private.json")

    teacher_predictions: dict[str, Sequence[float]] = {}
    baseline_predictions: dict[str, Sequence[float]] = {}
    distilled_predictions: dict[str, Sequence[float]] = {}
    results_by_arm: dict[str, list[TrainingResult]] = {
        "teacher": teacher_results,
        "undistilled_student": [],
        "distilled_student": [],
    }
    fold_costs: list[dict[str, object]] = []

    for fold in FOLD_IDS:
        record = teacher_records[fold]
        teacher_predictions.update(_score_sequences(record.predictions))
        training_examples = build_fold_training_examples(
            studies,
            fold_manifest,
            fold,
            preprocess_config=student_config.preprocess_config(),
            packing_config=student_config.packing_config(),
            series_cache=series_cache,
        )
        held_out_examples = build_fold_held_out_examples(
            studies,
            fold_manifest,
            fold,
            preprocess_config=student_config.preprocess_config(),
            packing_config=student_config.packing_config(),
            series_cache=series_cache,
        )
        candidate_uids = fold_plans[fold].eligible_soft_target_study_uids
        excluded_uids = fold_plans[fold].excluded_report_only_study_uids
        candidate_examples = build_fold_pseudo_candidate_examples(
            studies,
            fold_manifest,
            fold,
            candidate_uids,
            preprocess_config=student_config.preprocess_config(),
            packing_config=student_config.packing_config(),
            series_cache=series_cache,
        )

        teacher_target_started = time.perf_counter()
        candidate_scores = predict_training_examples(
            teachers[fold], candidate_examples, device="cpu", batch_size=student_config.batch_size
        )
        candidate_score_sequences = _score_sequences(candidate_scores)
        target_path = destination / "teacher-targets" / f"fold-{fold}.private.json"
        atomic_json_write(
            target_path,
            {
                "schema_version": TICKET22_SCHEMA_VERSION,
                "artifact_role": "fold_training_soft_targets",
                "fold": fold,
                "teacher_checkpoint_sha256": teacher_lineage[fold]["checkpoint_sha256"],
                "teacher_run_sha256": teacher_lineage[fold]["run_sha256"],
                "fold_manifest_checksum": fold_manifest.manifest_checksum,
                "eligible_study_uids": list(candidate_uids),
                "excluded_study_uids": list(excluded_uids),
                "predictions": _private_predictions({"teacher": candidate_score_sequences})["teacher"],
            },
        )
        target_sha256 = _sha256_file(target_path)
        soft_examples = create_fold_soft_target_examples(
            candidate_examples,
            candidate_score_sequences,
            fold_manifest=fold_manifest,
            fold=fold,
            eligible_study_uids=candidate_uids,
            excluded_study_uids=excluded_uids,
            teacher_checkpoint_sha256=str(teacher_lineage[fold]["checkpoint_sha256"]),
        )
        target_generation_seconds = time.perf_counter() - teacher_target_started

        baseline_started = time.perf_counter()
        baseline_result = train_one_fold(
            studies,  # type: ignore[arg-type]
            fold_manifest,
            fold,
            destination / "undistilled-student" / f"fold-{fold}",
            config=student_config,
            source_version=integration_manifest.source_version,
            source_fingerprint=source_fingerprint,
            integration_manifest_checksum=integration_manifest.manifest_checksum,
            series_cache=series_cache,
            preflight=preflight,
            requested_resources=requested_resources,
        )
        baseline_training_seconds = time.perf_counter() - baseline_started
        if baseline_result.final_checkpoint_path is None:
            raise DistillationExperimentError("undistilled student did not publish its checkpoint")
        baseline_predictions.update(_score_sequences(baseline_result.predictions))
        results_by_arm["undistilled_student"].append(baseline_result)

        target_fingerprint = fingerprint_training_examples(soft_examples)
        heldout_fingerprint = fingerprint_training_examples(held_out_examples)
        additional_seed = student_config.batch_order_seed + TICKET22_SOFT_BATCH_SEED_OFFSET
        distill_identity = {
            "ticket": 22,
            "experiment_schema_version": TICKET22_SCHEMA_VERSION,
            "fold": fold,
            "fold_manifest_checksum": fold_manifest.manifest_checksum,
            "integration_manifest_checksum": integration_manifest.manifest_checksum,
            "source_fingerprint": source_fingerprint,
            "code_revision": code_revision,
            "code_sha256": _code_sha256(repository_root),
            "design_sha256": design_sha256,
            "gold_data_fingerprint": fingerprint_training_examples(training_examples),
            "held_out_data_fingerprint": heldout_fingerprint,
            "held_out_inputs_excluded_from_fitting": True,
            "eligible_soft_target_set_sha256": _uid_set_sha256(candidate_uids),
            "excluded_soft_target_set_sha256": _uid_set_sha256(excluded_uids),
            "soft_target_artifact_sha256": target_sha256,
            "soft_target_data_fingerprint": target_fingerprint,
            "soft_target_temperature": 1.0,
            "teacher_checkpoint_sha256": teacher_lineage[fold]["checkpoint_sha256"],
            "teacher_run_sha256": teacher_lineage[fold]["run_sha256"],
            "teacher_configuration_fingerprint": teacher_lineage[fold]["configuration_fingerprint"],
            "training_design": "ticket22-gold-bce-plus-unthresholded-teacher-soft-bce-v1",
            "soft_target_batch_order_seed": additional_seed,
        }
        distill_started = time.perf_counter()
        distilled_result = train_fixed_budget(
            training_examples,
            student_config,
            additional_examples=soft_examples,
            additional_batch_order_seed=additional_seed,
            artifact_dir=destination / "distilled-student" / f"fold-{fold}",
            fold=fold,
            identity=distill_identity,
            parent_provenance={
                "parent_ticket": 19,
                "parent_run_id": record.run_id,
                "parent_checkpoint_sha256": teacher_lineage[fold]["checkpoint_sha256"],
                "parent_code_revision": teacher_lineage[fold]["code_revision"],
                "soft_target_artifact_sha256": target_sha256,
            },
            requested_resources=requested_resources,
        )
        if distilled_result.final_checkpoint_path is None:
            raise DistillationExperimentError("distilled student did not publish its checkpoint")
        load_training_checkpoint(
            distilled_result.final_checkpoint_path,
            model=distilled_result.model,
            expected_config=student_config,
            expected_fold=fold,
            expected_identity=distill_identity,
            expected_checkpoint_kind="final_selected",
        )
        distill_training_seconds = time.perf_counter() - distill_started
        if baseline_result.initial_model_state_sha256 != distilled_result.initial_model_state_sha256:
            raise DistillationExperimentError("student arms did not use identical initialization")
        if baseline_result.gold_exposure_sequence_sha256 != distilled_result.gold_exposure_sequence_sha256:
            raise DistillationExperimentError("student arms did not preserve the same gold exposure sequence")
        if distilled_result.successful_optimizer_steps != student_config.successful_optimizer_steps:
            raise DistillationExperimentError("distilled student did not complete its frozen optimizer budget")
        exposures_by_source = distilled_result.exposures_by_source
        if exposures_by_source is None or "reference" not in exposures_by_source or "pseudo" not in exposures_by_source:
            raise DistillationExperimentError("distilled student did not record both target exposure sources")

        student_scores = predict_training_examples(
            distilled_result.model, held_out_examples, device="cpu", batch_size=student_config.batch_size
        )
        student_score_sequences = _score_sequences(student_scores)
        distilled_predictions.update(student_score_sequences)
        results_by_arm["distilled_student"].append(distilled_result)

        atomic_json_write(
            destination / "distilled-student" / f"fold-{fold}" / "ticket22.private.json",
            {
                "fold": fold,
                "training_study_uids": [item.study_instance_uid for item in training_examples],
                "held_out_study_uids": [item.study_instance_uid for item in held_out_examples],
                "soft_target_study_uids": list(candidate_uids),
                "soft_target_artifact_sha256": target_sha256,
                "teacher_checkpoint_sha256": teacher_lineage[fold]["checkpoint_sha256"],
                "held_out_inputs_excluded": True,
                "training": distilled_result.to_private_dict(),
                "held_out_predictions": _private_predictions({"distilled_student": student_score_sequences})[
                    "distilled_student"
                ],
            },
        )
        fold_costs.append(
            {
                "fold": fold,
                "teacher_target_generation_seconds": target_generation_seconds,
                "undistilled_student_training_seconds": baseline_training_seconds,
                "distilled_student_training_seconds": distill_training_seconds,
                "teacher_training_seconds": _teacher_elapsed_seconds(record.path),
                "teacher_heldout_inference_seconds": record.counts.get(
                    "heldout_model_inference_seconds"
                ),
                "undistilled_gold_sample_exposures": baseline_result.sample_exposures,
                "distilled_gold_sample_exposures": exposures_by_source["reference"][
                    "sample_exposures"
                ],
                "distilled_soft_target_sample_exposures": exposures_by_source["pseudo"][
                    "sample_exposures"
                ],
                "gold_exposure_sequence_sha256": baseline_result.gold_exposure_sequence_sha256,
                "initialization_state_sha256": baseline_result.initial_model_state_sha256,
                "teacher_target_artifact_sha256": target_sha256,
                "teacher_checkpoint_sha256": teacher_lineage[fold]["checkpoint_sha256"],
                "baseline_checkpoint_sha256": checkpoint_sha256(
                    cast(Path, baseline_result.final_checkpoint_path)
                ),
                "distilled_checkpoint_sha256": checkpoint_sha256(
                    cast(Path, distilled_result.final_checkpoint_path)
                ),
            }
        )

    comparisons = compare_ticket22_oof_predictions(
        teacher_predictions,
        baseline_predictions,
        distilled_predictions,
        fold_manifest,
        bootstrap_resamples=TICKET22_BOOTSTRAP_RESAMPLES,
        bootstrap_seed=TICKET22_BOOTSTRAP_SEED,
    )
    public_comparison = {
        "schema_version": TICKET22_SCHEMA_VERSION,
        "ticket": 22,
        "status": "completed_three_fold_oof_comparison",
        "fold_manifest_checksum": fold_manifest.manifest_checksum,
        "integration_manifest_checksum": integration_manifest.manifest_checksum,
        "code_revision": code_revision,
        "code_sha256": _code_sha256(repository_root),
        "design_sha256": design_sha256,
        "teacher_lineage": [
            {
                key: value
                for key, value in item.items()
                if key not in {"source_fingerprint"}
            }
            for item in teacher_lineage
        ],
        "comparisons": comparisons,
        "fold_training_costs": fold_costs,
        "checkpoint_and_model_costs": _checkpoint_costs(results_by_arm),
        "inference_runtime": _measure_ensemble_runtime(
            {
                arm: [
                    cast(Path, result.final_checkpoint_path)
                    for result in arm_results
                ]
                for arm, arm_results in results_by_arm.items()
            },
            {
                "teacher": teacher_config,
                "undistilled_student": student_config,
                "distilled_student": student_config,
            },
            [studies[uid] for uid in integration_manifest.study_uids if uid in studies],
            integration_manifest,
            series_cache,
            input_loading_seconds,
        ),
        "input_loading_seconds": input_loading_seconds,
        "gpu_used": False,
        "peak_vram_bytes": None,
        "peak_vram_status": "not measured; this experiment used the approved CPU partition",
        "training_bundle_size_bytes": None,
        "training_bundle_size_status": "no delivery package was assembled; final checkpoint bytes are measured per fold",
        "limitations": [
            "Development cross-validation guided model selection and is not an untouched internal test.",
            "The inference timing covers the selected training-source integration cohort, not the hidden competition test scale.",
            "No GPU run was authorized, so peak VRAM is unmeasured.",
            "The Ticket 19 teacher is retained as a tested parent checkpoint; it was not promoted by Ticket 19.",
        ],
    }
    atomic_json_write(destination / "comparison.json", public_comparison)
    atomic_json_write(
        destination / "predictions.private.json",
        {
            "schema_version": TICKET22_SCHEMA_VERSION,
            "fold_manifest_checksum": fold_manifest.manifest_checksum,
            "prediction_roles": {
                "teacher": "held-out evaluation only",
                "undistilled_student": "held-out evaluation only",
                "distilled_student": "held-out evaluation only",
            },
            "teacher": _private_predictions({"teacher": teacher_predictions})["teacher"],
            "undistilled_student": _private_predictions({"undistilled_student": baseline_predictions})[
                "undistilled_student"
            ],
            "distilled_student": _private_predictions({"distilled_student": distilled_predictions})[
                "distilled_student"
            ],
        },
    )
    return public_comparison


__all__ = [
    "DistillationExperimentError",
    "Ticket22FoldPlan",
    "TICKET22_BOOTSTRAP_RESAMPLES",
    "TICKET22_BOOTSTRAP_SEED",
    "TICKET22_SCHEMA_VERSION",
    "TICKET22_SOFT_LABEL_VERSION",
    "TICKET22_STUDENT_CHANNELS",
    "TICKET22_STEPS",
    "build_ticket22_fold_plan",
    "compare_ticket22_oof_predictions",
    "create_fold_soft_target_examples",
    "run_ticket22_distillation_experiment",
]
