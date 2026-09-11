"""Reference-only held-out evaluation for the twelve-target mechanics model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence, cast

import numpy as np
import torch

from .folds import FOLD_IDS, FoldManifest
from .targets import NUM_TARGETS, TARGET_COLUMNS

if TYPE_CHECKING:
    from .model import HierarchicalStudyModel
    from .training import TrainingExample


class EvaluationError(ValueError):
    """Raised when reference predictions cannot be evaluated safely."""


def roc_auc_score(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Return rank-based ROC-AUC, or ``None`` when one class is unavailable.

    Average ranks are assigned to tied scores, so perfect, reversed, and tied
    rankings have the conventional values 1, 0, and 0.5 respectively.
    """

    label_array = np.asarray(tuple(labels))
    score_array = np.asarray(tuple(scores), dtype=np.float64)
    if label_array.ndim != 1 or score_array.ndim != 1 or label_array.shape != score_array.shape:
        raise EvaluationError("labels and scores must be one-dimensional arrays of equal length")
    if label_array.size == 0:
        raise EvaluationError("ROC-AUC requires at least one prediction")
    if np.any(~np.isin(label_array, (0, 1))):
        raise EvaluationError("ROC-AUC labels must be binary")
    if not np.isfinite(score_array).all():
        raise EvaluationError("ROC-AUC scores must be finite")
    positives = int(np.sum(label_array == 1))
    negatives = int(np.sum(label_array == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(score_array, kind="mergesort")
    sorted_scores = score_array[order]
    ranks = np.empty(score_array.size, dtype=np.float64)
    index = 0
    while index < sorted_scores.size:
        end = index + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[index]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    positive_rank_sum = float(ranks[label_array == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


@dataclass(frozen=True)
class TargetEvaluation:
    """One target's reference support and AUC status."""

    target: str
    auc: float | None
    positive_count: int
    negative_count: int
    status: str

    def __post_init__(self) -> None:
        if self.target not in TARGET_COLUMNS:
            raise EvaluationError(f"unknown target {self.target!r}")
        if min(self.positive_count, self.negative_count) < 0:
            raise EvaluationError("target support counts cannot be negative")
        if self.status not in {"defined", "undefined_one_class"}:
            raise EvaluationError("unknown target evaluation status")
        if self.status == "defined" and self.auc is None:
            raise EvaluationError("defined target evaluation requires an AUC")
        if self.auc is not None and not 0.0 <= self.auc <= 1.0:
            raise EvaluationError("AUC must be within [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "auc": self.auc,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "status": self.status,
        }


@dataclass(frozen=True)
class FoldEvaluation:
    """Metrics for exactly one held-out fold."""

    fold: int
    study_instance_uids: tuple[str, ...]
    targets: tuple[TargetEvaluation, ...]
    macro_auc: float | None
    defined_target_macro_auc: float | None

    def __post_init__(self) -> None:
        if self.fold not in FOLD_IDS:
            raise EvaluationError(f"fold must be one of {FOLD_IDS}")
        if tuple(item.target for item in self.targets) != TARGET_COLUMNS:
            raise EvaluationError("fold evaluation targets must use competition order")
        if len(set(self.study_instance_uids)) != len(self.study_instance_uids):
            raise EvaluationError("fold evaluation study identifiers must be unique")
        if self.macro_auc is not None and not 0.0 <= self.macro_auc <= 1.0:
            raise EvaluationError("macro AUC must be within [0, 1]")
        if (
            self.defined_target_macro_auc is not None
            and not 0.0 <= self.defined_target_macro_auc <= 1.0
        ):
            raise EvaluationError("defined-target macro AUC must be within [0, 1]")

    @property
    def undefined_targets(self) -> tuple[str, ...]:
        return tuple(item.target for item in self.targets if item.auc is None)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "prediction_count": len(self.study_instance_uids),
            "targets": [item.to_dict() for item in self.targets],
            "macro_auc": self.macro_auc,
            "defined_target_macro_auc": self.defined_target_macro_auc,
            "undefined_targets": list(self.undefined_targets),
            "undefined_auc_is_not_imputed": True,
        }

    def to_private_dict(self) -> dict[str, object]:
        return {
            **self.to_public_dict(),
            "study_instance_uids": list(self.study_instance_uids),
        }


def _validate_prediction_vector(study_uid: str, values: Sequence[float]) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.shape != (NUM_TARGETS,):
        raise EvaluationError(f"prediction for {study_uid} must contain {NUM_TARGETS} scores")
    if not np.isfinite(array).all() or (array < 0.0).any() or (array > 1.0).any():
        raise EvaluationError(f"prediction for {study_uid} must be finite and bounded")
    return array


def evaluate_fold_predictions(
    predictions: Mapping[str, Sequence[float]],
    fold_manifest: FoldManifest,
    fold: int,
) -> FoldEvaluation:
    """Join final-checkpoint predictions to the declared held-out gold rows."""

    if fold not in FOLD_IDS:
        raise EvaluationError(f"fold must be one of {FOLD_IDS}")
    expected = tuple(
        sorted(
            item.study_instance_uid
            for item in fold_manifest.gold_memberships
            if item.fold == fold
        )
    )
    actual = tuple(sorted(predictions))
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise EvaluationError(f"held-out prediction identity mismatch: missing={missing[:3]}, extra={extra[:3]}")
    by_uid = {item.study_instance_uid: item for item in fold_manifest.gold_memberships}
    values = {uid: _validate_prediction_vector(uid, predictions[uid]) for uid in expected}
    metrics: list[TargetEvaluation] = []
    for target_index, target in enumerate(TARGET_COLUMNS):
        labels: list[int] = []
        for uid in expected:
            label = by_uid[uid].reference_labels[target_index]
            if label not in {0, 1}:
                raise EvaluationError(f"held-out reference label is missing for {uid}, {target}")
            labels.append(cast(int, label))
        scores = [float(values[uid][target_index]) for uid in expected]
        positives = sum(label == 1 for label in labels)
        negatives = sum(label == 0 for label in labels)
        auc = roc_auc_score(labels, scores)
        metrics.append(
            TargetEvaluation(
                target=target,
                auc=auc,
                positive_count=positives,
                negative_count=negatives,
                status="defined" if auc is not None else "undefined_one_class",
            )
        )
    defined = tuple(item.auc for item in metrics if item.auc is not None)
    defined_macro = float(np.mean(defined)) if defined else None
    macro = defined_macro if len(defined) == len(metrics) else None
    return FoldEvaluation(
        fold=fold,
        study_instance_uids=expected,
        targets=tuple(metrics),
        macro_auc=macro,
        defined_target_macro_auc=defined_macro,
    )


def predict_training_examples(
    model: "HierarchicalStudyModel",
    examples: Sequence["TrainingExample"],
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 2,
) -> dict[str, np.ndarray]:
    """Run packed examples through the same model aggregation used by inference."""

    if not examples:
        raise EvaluationError("at least one example is required for prediction")
    if batch_size <= 0:
        raise EvaluationError("batch_size must be positive")
    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()
    predictions: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            images = torch.from_numpy(np.stack([item.images for item in batch])).to(target_device)
            slice_mask = torch.from_numpy(
                np.stack([item.slice_mask for item in batch])
            ).to(target_device)
            series_mask = torch.from_numpy(
                np.stack([item.series_mask for item in batch])
            ).to(target_device)
            scores = model.predict_proba(images, slice_mask, series_mask).cpu().numpy()
            for item, row in zip(batch, scores, strict=True):
                values = _validate_prediction_vector(item.study_instance_uid, row)
                predictions[item.study_instance_uid] = values
    return predictions


__all__ = [
    "EvaluationError",
    "FoldEvaluation",
    "TargetEvaluation",
    "evaluate_fold_predictions",
    "predict_training_examples",
    "roc_auc_score",
]
