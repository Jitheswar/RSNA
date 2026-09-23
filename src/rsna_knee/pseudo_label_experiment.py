"""Run the predeclared one-fold teacher-to-pseudo-label-to-student trial."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

from .archive import atomic_json_write
from .evaluation import predict_training_examples
from .folds import FOLD_IDS, FoldManifest
from .image_variant import MECHANICS_BASELINE_VARIANT, image_model_training_config
from .pseudo_label_comparison import compare_fold_pseudo_label_predictions
from .pseudo_labels import (
    PSEUDO_LABEL_POLICY,
    PseudoLabelTeacherProvenance,
    admit_fold_pseudo_labels,
    create_pseudo_label_generation,
)
from .real_inference import IntegrationManifest
from .training import (
    CpuPreflightResult,
    TrainingDataError,
    build_fold_pseudo_candidate_examples,
    checkpoint_sha256,
    train_one_fold,
)
from .domain import StudyRecord
from .series_cache import SeriesCache


TICKET21_FOLD = 0
TICKET21_BOOTSTRAP_RESAMPLES = 2000
TICKET21_BOOTSTRAP_SEED = 20260923
_TEACHER_CODE_FILES = (
    "src/rsna_knee/evaluation.py",
    "src/rsna_knee/folds.py",
    "src/rsna_knee/model.py",
    "src/rsna_knee/preprocessing.py",
    "src/rsna_knee/pseudo_labels.py",
    "src/rsna_knee/targets.py",
    "src/rsna_knee/training.py",
)


def _code_sha256(repository_root: Path) -> str:
    digest = hashlib.sha256()
    for relative_path in _TEACHER_CODE_FILES:
        path = repository_root / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _study_set_sha256(study_uids: Sequence[str]) -> str:
    serialized = json.dumps(
        sorted(study_uids), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def run_ticket21_pseudo_label_experiment(
    studies: Mapping[str, StudyRecord],
    fold_manifest: FoldManifest,
    integration_manifest: IntegrationManifest,
    output_root: str | Path,
    *,
    fold: int,
    code_revision: str,
    source_fingerprint: str,
    preflight: CpuPreflightResult,
    input_loading_seconds: float,
    series_cache: SeriesCache | None = None,
    requested_resources: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run a fixed gold-teacher comparator and one additive pseudo-label student."""

    if fold != TICKET21_FOLD or fold not in FOLD_IDS:
        raise TrainingDataError("Ticket 21 predeclares fold 0 only")
    if fold_manifest.superseded or fold_manifest.fold_ids != FOLD_IDS:
        raise TrainingDataError("Ticket 21 requires the canonical frozen fold manifest")
    repository_root = Path(__file__).resolve().parents[2]
    destination = Path(output_root).resolve()
    if destination.is_relative_to(repository_root):
        raise TrainingDataError("Ticket 21 private experiment output must be outside the repository")
    if not code_revision or len(code_revision) != 40 or any(
        character not in "0123456789abcdef" for character in code_revision
    ):
        raise TrainingDataError("Ticket 21 requires the committed full Git code revision")
    if not source_fingerprint.strip():
        raise TrainingDataError("Ticket 21 requires a source fingerprint")
    if not preflight.passed or preflight.details.get("required_slurm") is not True:
        raise TrainingDataError("Ticket 21 requires a passed cluster1-cpu preflight")

    destination.mkdir(parents=True, exist_ok=True)
    design_path = destination / "design.private.json"
    if design_path.exists():
        raise FileExistsError(f"refusing to overwrite the frozen Ticket 21 design {design_path}")
    expected_report_only = {
        item.study_instance_uid
        for item in fold_manifest.report_only_memberships
        if item.fold != fold
    }
    image_uids = set(integration_manifest.study_uids).intersection(studies)
    eligible_candidates = tuple(sorted(expected_report_only.intersection(image_uids)))
    excluded_candidates = tuple(sorted(expected_report_only - set(eligible_candidates)))
    if not eligible_candidates:
        raise TrainingDataError("the integration manifest has no fold-0 report-only pseudo candidates")
    gold_training_uids = tuple(
        sorted(item.study_instance_uid for item in fold_manifest.gold_memberships if item.fold != fold)
    )
    expected_gold = {item.study_instance_uid for item in fold_manifest.gold_memberships}
    if not expected_gold.issubset(integration_manifest.study_uids):
        raise TrainingDataError("the integration manifest does not cover every frozen gold study")

    config = image_model_training_config(
        MECHANICS_BASELINE_VARIANT,
        preprocessing_config=integration_manifest.preprocess_config,
    )
    source_code_sha256 = _code_sha256(repository_root)
    design = {
        "ticket": 21,
        "schema_version": "rsna-knee-ticket21-design-v1",
        "fold": fold,
        "repeat_name": fold_manifest.repeat_name,
        "fold_manifest_checksum": fold_manifest.manifest_checksum,
        "integration_manifest_checksum": integration_manifest.manifest_checksum,
        "source_fingerprint": source_fingerprint,
        "fixed_comparator": {
            "description": "fold-0 gold-only active mechanics baseline teacher",
            "image_variant": MECHANICS_BASELINE_VARIANT,
            "training_cohort_study_uids": list(gold_training_uids),
            "training_cohort_sha256": _study_set_sha256(gold_training_uids),
            "training_config": config.to_dict(),
            "successful_optimizer_steps": config.successful_optimizer_steps,
        },
        "student": {
            "description": "same fold, architecture, initialization, gold exposure sequence and step budget; additive pseudo-label pool",
            "successful_optimizer_steps": config.successful_optimizer_steps,
            "pseudo_sampling_policy": "ticket21-report-only-pseudo-cyclic-shuffle-v1",
        },
        "pseudo_label_generation": {
            "policy": PSEUDO_LABEL_POLICY.to_dict(),
            "eligible_study_uids": list(eligible_candidates),
            "eligible_study_set_sha256": _study_set_sha256(eligible_candidates),
            "excluded_study_uids": list(excluded_candidates),
            "excluded_study_set_sha256": _study_set_sha256(excluded_candidates),
            "source": "MRI-only teacher predictions; no reports or external learned resources",
        },
        "teacher_provenance": {
            "code_revision": code_revision,
            "code_sha256": source_code_sha256,
            "configuration_fingerprint": config.config_fingerprint,
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
            "teacher_optimizer_steps": config.successful_optimizer_steps,
            "student_optimizer_steps": config.successful_optimizer_steps,
        },
    }
    atomic_json_write(design_path, design)
    design_sha256 = hashlib.sha256(design_path.read_bytes()).hexdigest()

    teacher_started = time.perf_counter()
    teacher_result = train_one_fold(
        studies,
        fold_manifest,
        fold,
        destination / "teacher-fold-0",
        config=config,
        source_version=integration_manifest.source_version,
        source_fingerprint=source_fingerprint,
        integration_manifest_checksum=integration_manifest.manifest_checksum,
        series_cache=series_cache,
        preflight=preflight,
        requested_resources=requested_resources,
    )
    teacher_elapsed = time.perf_counter() - teacher_started
    if teacher_result.final_checkpoint_path is None:
        raise TrainingDataError("Ticket 21 teacher did not publish a final checkpoint")
    if teacher_result.training_study_uids != gold_training_uids:
        raise TrainingDataError("Ticket 21 teacher training identities differ from the predeclared cohort")

    teacher = PseudoLabelTeacherProvenance(
        fold=fold,
        fold_manifest_checksum=fold_manifest.manifest_checksum,
        repeat_name=fold_manifest.repeat_name,
        training_study_uids=gold_training_uids,
        held_out_study_uids=tuple(sorted(fold_manifest.held_out_study_uids(fold))),
        source_fingerprint=source_fingerprint,
        code_revision=code_revision,
        code_sha256=source_code_sha256,
        configuration_fingerprint=config.config_fingerprint,
        checkpoint_sha256=checkpoint_sha256(teacher_result.final_checkpoint_path),
    )

    candidate_started = time.perf_counter()
    candidate_examples = build_fold_pseudo_candidate_examples(
        studies,
        fold_manifest,
        fold,
        eligible_candidates,
        preprocess_config=integration_manifest.preprocess_config,
        packing_config=integration_manifest.packing_config,
        series_cache=series_cache,
    )
    candidate_prepare_elapsed = time.perf_counter() - candidate_started
    generation_started = time.perf_counter()
    predicted_scores = predict_training_examples(
        teacher_result.model,
        candidate_examples,
        device=config.device,
    )
    scores = {
        uid: tuple(float(score) for score in values)
        for uid, values in predicted_scores.items()
    }
    generation_elapsed = time.perf_counter() - generation_started
    generation = create_pseudo_label_generation(
        teacher,
        scores,
        eligible_study_uids=eligible_candidates,
        excluded_study_uids=excluded_candidates,
    )
    atomic_json_write(
        destination / "pseudo-labels.private.json",
        generation.to_private_dict(),
    )
    admitted = admit_fold_pseudo_labels(fold_manifest, fold, generation)

    student_result = None
    comparison = None
    student_elapsed = 0.0
    if admitted.pseudo_study_count:
        student_started = time.perf_counter()
        student_result = train_one_fold(
            studies,
            fold_manifest,
            fold,
            destination / "student-fold-0",
            config=config,
            source_version=integration_manifest.source_version,
            source_fingerprint=source_fingerprint,
            integration_manifest_checksum=integration_manifest.manifest_checksum,
            series_cache=series_cache,
            supervision=admitted,
            ticket21_pseudolabel_design=True,
            preflight=preflight,
            requested_resources=requested_resources,
        )
        student_elapsed = time.perf_counter() - student_started
        if student_result.training_study_uids != admitted.training_study_uids:
            raise TrainingDataError("Ticket 21 student identities disagree with admitted lineage")
        if student_result.config.config_fingerprint != teacher_result.config.config_fingerprint:
            raise TrainingDataError("Ticket 21 teacher and student training configurations differ")
        if student_result.successful_optimizer_steps != teacher_result.successful_optimizer_steps:
            raise TrainingDataError("Ticket 21 teacher and student optimizer-step budgets differ")
        if student_result.initial_model_state_sha256 != teacher_result.initial_model_state_sha256:
            raise TrainingDataError("Ticket 21 teacher and student initialization states differ")
        if (
            student_result.gold_exposure_sequence_sha256
            != teacher_result.gold_exposure_sequence_sha256
        ):
            raise TrainingDataError("Ticket 21 student changed the teacher's gold exposure sequence")
        comparison = compare_fold_pseudo_label_predictions(
            {
                uid: tuple(float(score) for score in values)
                for uid, values in teacher_result.predictions.items()
            },
            {
                uid: tuple(float(score) for score in values)
                for uid, values in student_result.predictions.items()
            },
            fold_manifest,
            fold,
            bootstrap_resamples=TICKET21_BOOTSTRAP_RESAMPLES,
            bootstrap_seed=TICKET21_BOOTSTRAP_SEED,
        )

    total_elapsed = time.perf_counter() - teacher_started
    evidence: dict[str, object] = {
        "schema_version": "rsna-knee-ticket21-experiment-evidence-v1",
        "status": "completed" if student_result is not None else "completed_without_admitted_pseudo_labels",
        "design_sha256": design_sha256,
        "fold_manifest_checksum": fold_manifest.manifest_checksum,
        "integration_manifest_checksum": integration_manifest.manifest_checksum,
        "code_revision": code_revision,
        "teacher_code_sha256": source_code_sha256,
        "teacher_checkpoint_sha256": teacher.checkpoint_sha256,
        "pseudo_label_output_sha256": generation.output_sha256,
        "pseudo_label_set_digest": generation.semantic_digest,
        "predeclared_policy": PSEUDO_LABEL_POLICY.to_dict(),
        "admission": admitted.to_public_dict(),
        "teacher": teacher_result.to_public_dict(),
        "student": student_result.to_public_dict() if student_result is not None else None,
        "comparison": comparison,
        "cost": {
            "input_loading_seconds": input_loading_seconds,
            "teacher_train_prepare_checkpoint_and_oof_seconds": teacher_elapsed,
            "pseudo_candidate_preparation_seconds": candidate_prepare_elapsed,
            "pseudo_generation_inference_seconds": generation_elapsed,
            "student_train_prepare_checkpoint_and_oof_seconds": student_elapsed,
            "teacher_held_out_model_inference_seconds": teacher_result.heldout_model_inference_seconds,
            "student_held_out_model_inference_seconds": (
                student_result.heldout_model_inference_seconds
                if student_result is not None
                else None
            ),
            "experiment_elapsed_seconds_excluding_preflight": total_elapsed,
            "full_slurm_job_cost_record": "observed compute is written by the Slurm adapter",
        },
        "limits": {
            "one_fold_only": True,
            "complete_three_fold_oof": False,
            "whole_kaggle_notebook_runtime_measured": False,
            "candidate_promotion": False,
            "final_model_replacement": False,
        },
    }
    atomic_json_write(destination / "experiment.public.json", evidence)
    return evidence


__all__ = ["TICKET21_BOOTSTRAP_RESAMPLES", "TICKET21_BOOTSTRAP_SEED", "run_ticket21_pseudo_label_experiment"]
