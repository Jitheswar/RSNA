"""One-fold paired OOF comparison for a pseudo-label student."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .evaluation import EvaluationError, evaluate_fold_predictions, roc_auc_score
from .folds import FOLD_IDS, FoldManifest
from .targets import NUM_TARGETS, TARGET_COLUMNS


class PseudoLabelComparisonError(ValueError):
    """Raised when teacher and student predictions are not a valid paired fold."""


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
            "lower": float(lower),
            "median": float(median),
            "upper": float(upper),
        }
    else:
        result = {"lower": None, "median": None, "upper": None}
    return {
        **result,
        "defined_resamples": len(values),
        "undefined_resamples": total_resamples - len(values),
        "resamples": total_resamples,
        "seed": seed,
        "confidence_level": confidence_level,
        "method": "paired_study_percentile_bootstrap",
    }


def compare_fold_pseudo_label_predictions(
    teacher_predictions: Mapping[str, Sequence[float]],
    student_predictions: Mapping[str, Sequence[float]],
    fold_manifest: FoldManifest,
    fold: int,
    *,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 20260921,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Compare paired teacher/student outputs on exactly one held-out gold fold."""

    if fold not in FOLD_IDS:
        raise PseudoLabelComparisonError(f"fold must be one of {FOLD_IDS}")
    if bootstrap_resamples <= 0 or bootstrap_seed < 0:
        raise PseudoLabelComparisonError("bootstrap resamples must be positive and seed non-negative")
    if not 0.0 < confidence_level < 1.0:
        raise PseudoLabelComparisonError("confidence level must be between zero and one")
    expected = {
        item.study_instance_uid
        for item in fold_manifest.gold_memberships
        if item.fold == fold
    }
    if set(teacher_predictions) != expected or set(student_predictions) != expected:
        raise PseudoLabelComparisonError(
            "teacher and student predictions must match the held-out gold identities"
        )
    for arm, predictions in (
        ("teacher", teacher_predictions),
        ("student", student_predictions),
    ):
        for uid, vector in predictions.items():
            values = np.asarray(vector, dtype=np.float64)
            if values.shape != (NUM_TARGETS,) or not np.isfinite(values).all():
                raise PseudoLabelComparisonError(
                    f"{arm} prediction for one study must contain twelve finite scores"
                )
            if np.any((values < 0.0) | (values > 1.0)):
                raise PseudoLabelComparisonError(f"{arm} scores must be within [0, 1]")
    try:
        teacher = evaluate_fold_predictions(teacher_predictions, fold_manifest, fold)
        student = evaluate_fold_predictions(student_predictions, fold_manifest, fold)
    except EvaluationError as exc:
        raise PseudoLabelComparisonError(str(exc)) from exc

    memberships = {
        item.study_instance_uid: item
        for item in fold_manifest.gold_memberships
        if item.fold == fold
    }
    ordered_uids = tuple(sorted(expected))
    labels = np.asarray(
        [memberships[uid].reference_labels for uid in ordered_uids], dtype=np.int8
    ).reshape(len(ordered_uids), NUM_TARGETS)
    teacher_scores = np.asarray(
        [teacher_predictions[uid] for uid in ordered_uids], dtype=np.float64
    )
    student_scores = np.asarray(
        [student_predictions[uid] for uid in ordered_uids], dtype=np.float64
    )

    rng = np.random.default_rng(bootstrap_seed)
    macro_deltas: list[float] = []
    target_deltas: list[list[float]] = [[] for _ in TARGET_COLUMNS]
    for _ in range(bootstrap_resamples):
        sample = rng.integers(0, len(ordered_uids), size=len(ordered_uids))
        deltas: list[float | None] = []
        for index in range(NUM_TARGETS):
            teacher_auc = roc_auc_score(labels[sample, index], teacher_scores[sample, index])
            student_auc = roc_auc_score(labels[sample, index], student_scores[sample, index])
            delta = (
                None
                if teacher_auc is None or student_auc is None
                else float(student_auc - teacher_auc)
            )
            deltas.append(delta)
            if delta is not None:
                target_deltas[index].append(delta)
        if all(delta is not None for delta in deltas):
            macro_deltas.append(float(np.mean([delta for delta in deltas if delta is not None])))

    per_target: list[dict[str, object]] = []
    for index, (teacher_metric, student_metric) in enumerate(
        zip(teacher.targets, student.targets, strict=True)
    ):
        delta = (
            None
            if teacher_metric.auc is None or student_metric.auc is None
            else student_metric.auc - teacher_metric.auc
        )
        per_target.append(
            {
                "target": TARGET_COLUMNS[index],
                "teacher_auc": teacher_metric.auc,
                "student_auc": student_metric.auc,
                "delta_auc": delta,
                "positive_count": teacher_metric.positive_count,
                "negative_count": teacher_metric.negative_count,
                "paired_study_bootstrap": _interval(
                    target_deltas[index],
                    total_resamples=bootstrap_resamples,
                    seed=bootstrap_seed,
                    confidence_level=confidence_level,
                ),
            }
        )
    delta_macro = (
        None
        if teacher.macro_auc is None or student.macro_auc is None
        else student.macro_auc - teacher.macro_auc
    )
    return {
        "schema_version": "rsna-knee-pseudo-label-fold-comparison-v1",
        "oof": {
            "repeat_name": fold_manifest.repeat_name,
            "fold": fold,
            "fold_manifest_checksum": fold_manifest.manifest_checksum,
            "prediction_count": len(ordered_uids),
            "scope": "one held-out fold; not complete-repeat OOF",
        },
        "teacher": {
            "macro_auc": teacher.macro_auc,
            "defined_target_macro_auc": teacher.defined_target_macro_auc,
            "targets": [item.to_dict() for item in teacher.targets],
        },
        "student": {
            "macro_auc": student.macro_auc,
            "defined_target_macro_auc": student.defined_target_macro_auc,
            "targets": [item.to_dict() for item in student.targets],
        },
        "delta": {
            "definition": "student minus teacher",
            "macro_auc": delta_macro,
            "folds": [
                {
                    "fold": fold,
                    "teacher_macro_auc": teacher.macro_auc,
                    "student_macro_auc": student.macro_auc,
                    "delta_macro_auc": delta_macro,
                    "prediction_count": len(ordered_uids),
                }
            ],
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
        "fold_variation": {
            "estimable": False,
            "reason": "one fold was predeclared for this budgeted experiment",
        },
        "undefined_auc_is_not_imputed": True,
        "interpretation_limit": (
            "One-fold development-CV evidence; it is not complete-repeat OOF, an untouched "
            "internal test, or a clinical-generalization estimate."
        ),
    }


__all__ = ["PseudoLabelComparisonError", "compare_fold_pseudo_label_predictions"]
