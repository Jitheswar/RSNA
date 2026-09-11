"""Reference-only held-out evaluation for the twelve-target mechanics model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence, cast

import numpy as np
import torch

from .folds import FOLD_IDS, FoldManifest, FoldMembership
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
class BootstrapMetric:
    """A percentile interval with explicit undefined-resample accounting."""

    statistic: str
    median: float | None
    lower: float | None
    upper: float | None
    defined_resamples: int
    undefined_resamples: int
    resamples: int

    def __post_init__(self) -> None:
        if not self.statistic.strip():
            raise EvaluationError("bootstrap statistic must be non-empty")
        if self.resamples <= 0:
            raise EvaluationError("bootstrap resamples must be positive")
        if min(self.defined_resamples, self.undefined_resamples) < 0:
            raise EvaluationError("bootstrap resample counts cannot be negative")
        if self.defined_resamples + self.undefined_resamples != self.resamples:
            raise EvaluationError("bootstrap resample counts do not add up")
        for name, value in (
            ("median", self.median),
            ("lower", self.lower),
            ("upper", self.upper),
        ):
            if value is not None and (not np.isfinite(value) or not 0.0 <= value <= 1.0):
                raise EvaluationError(f"bootstrap {name} must be finite and within [0, 1]")
        if self.defined_resamples == 0 and any(
            value is not None for value in (self.median, self.lower, self.upper)
        ):
            raise EvaluationError("undefined bootstrap statistic cannot have interval values")
        if self.defined_resamples > 0 and any(
            value is None for value in (self.median, self.lower, self.upper)
        ):
            raise EvaluationError("defined bootstrap statistic requires interval values")

    def to_dict(self) -> dict[str, object]:
        return {
            "statistic": self.statistic,
            "median": self.median,
            "lower": self.lower,
            "upper": self.upper,
            "defined_resamples": self.defined_resamples,
            "undefined_resamples": self.undefined_resamples,
            "resamples": self.resamples,
        }


@dataclass(frozen=True)
class BootstrapSummary:
    """Study- or reliable-group bootstrap uncertainty for one OOF result."""

    unit: str
    unit_count: int
    resamples: int
    seed: int
    confidence_level: float
    targets: tuple[BootstrapMetric, ...]
    macro_auc: BootstrapMetric

    def __post_init__(self) -> None:
        if self.unit not in {"study", "group"}:
            raise EvaluationError("bootstrap unit must be study or group")
        if self.unit_count <= 0:
            raise EvaluationError("bootstrap requires at least one sampled unit")
        if self.resamples <= 0 or self.seed < 0:
            raise EvaluationError("bootstrap resamples must be positive and seed non-negative")
        if not 0.0 < self.confidence_level < 1.0:
            raise EvaluationError("bootstrap confidence level must be between zero and one")
        if tuple(item.statistic for item in self.targets) != tuple(
            f"roc_auc:{target}" for target in TARGET_COLUMNS
        ):
            raise EvaluationError("bootstrap targets must use competition order")
        if any(item.resamples != self.resamples for item in self.targets) or (
            self.macro_auc.resamples != self.resamples
        ):
            raise EvaluationError("bootstrap metrics must use the summary resample count")

    def to_dict(self) -> dict[str, object]:
        return {
            "unit": self.unit,
            "unit_count": self.unit_count,
            "resamples": self.resamples,
            "seed": self.seed,
            "confidence_level": self.confidence_level,
            "interval_method": "percentile_on_defined_resamples_only",
            "targets": [item.to_dict() for item in self.targets],
            "macro_auc": self.macro_auc.to_dict(),
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


@dataclass(frozen=True)
class OOFEvaluation:
    """Pooled and per-fold metrics for one frozen development-CV repeat."""

    repeat_name: str
    fold_manifest_checksum: str
    study_instance_uids: tuple[str, ...]
    pooled_targets: tuple[TargetEvaluation, ...]
    pooled_macro_auc: float | None
    defined_target_macro_auc: float | None
    folds: tuple[FoldEvaluation, ...]
    mean_fold_macro_auc: float | None
    defined_fold_macro_auc_mean: float | None
    fold_macro_auc_std: float | None
    fold_macro_auc_min: float | None
    fold_macro_auc_max: float | None
    bootstrap: BootstrapSummary | None = None
    run_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.repeat_name.strip() or not self.fold_manifest_checksum.strip():
            raise EvaluationError("OOF evaluation requires repeat and fold-manifest identities")
        if tuple(item.target for item in self.pooled_targets) != TARGET_COLUMNS:
            raise EvaluationError("pooled evaluation targets must use competition order")
        if tuple(item.fold for item in self.folds) != FOLD_IDS:
            raise EvaluationError("OOF evaluation must contain every fold in order")
        if len(set(self.study_instance_uids)) != len(self.study_instance_uids):
            raise EvaluationError("OOF study identifiers must be unique")
        if len(set(self.run_ids)) != len(self.run_ids):
            raise EvaluationError("OOF run identifiers must be unique")
        for name, value in (
            ("pooled_macro_auc", self.pooled_macro_auc),
            ("defined_target_macro_auc", self.defined_target_macro_auc),
            ("mean_fold_macro_auc", self.mean_fold_macro_auc),
            ("defined_fold_macro_auc_mean", self.defined_fold_macro_auc_mean),
            ("fold_macro_auc_std", self.fold_macro_auc_std),
            ("fold_macro_auc_min", self.fold_macro_auc_min),
            ("fold_macro_auc_max", self.fold_macro_auc_max),
        ):
            if value is not None and (not np.isfinite(value) or not 0.0 <= value <= 1.0):
                raise EvaluationError(f"{name} must be finite and within [0, 1]")
        if self.pooled_macro_auc is not None and any(item.auc is None for item in self.pooled_targets):
            raise EvaluationError("pooled macro AUC cannot be defined with undefined targets")
        if self.mean_fold_macro_auc is not None and any(item.macro_auc is None for item in self.folds):
            raise EvaluationError("mean fold macro AUC cannot hide undefined fold metrics")

    @property
    def prediction_count(self) -> int:
        return len(self.study_instance_uids)

    @property
    def macro_auc(self) -> float | None:
        """Alias for the primary pooled twelve-target macro AUC."""

        return self.pooled_macro_auc

    @property
    def undefined_targets(self) -> tuple[str, ...]:
        return tuple(item.target for item in self.pooled_targets if item.auc is None)

    @property
    def undefined_fold_count(self) -> int:
        return sum(item.macro_auc is None for item in self.folds)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": "rsna-knee-oof-evaluation-v1",
            "repeat_name": self.repeat_name,
            "fold_manifest_checksum": self.fold_manifest_checksum,
            "prediction_count": self.prediction_count,
            "macro_auc": self.pooled_macro_auc,
            "defined_target_macro_auc": self.defined_target_macro_auc,
            "pooled": {
                "targets": [item.to_dict() for item in self.pooled_targets],
                "macro_auc": self.pooled_macro_auc,
                "defined_target_macro_auc": self.defined_target_macro_auc,
                "undefined_targets": list(self.undefined_targets),
            },
            "folds": [item.to_public_dict() for item in self.folds],
            "fold_macro_auc": {
                "mean": self.mean_fold_macro_auc,
                "defined_fold_mean": self.defined_fold_macro_auc_mean,
                "std": self.fold_macro_auc_std,
                "min": self.fold_macro_auc_min,
                "max": self.fold_macro_auc_max,
                "undefined_fold_count": self.undefined_fold_count,
                "std_definition": "population_over_three_folds_when_all_defined",
            },
            "bootstrap": self.bootstrap.to_dict() if self.bootstrap is not None else None,
            "undefined_auc_is_not_imputed": True,
            "oof_is_development_cv_not_untouched_internal_test": True,
            "clinical_generalization_not_established": True,
            "run_ids": list(self.run_ids),
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


def _validated_prediction_mapping(
    predictions: Mapping[str, Sequence[float]],
    fold_manifest: FoldManifest,
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    expected = tuple(sorted(item.study_instance_uid for item in fold_manifest.gold_memberships))
    actual = tuple(sorted(predictions))
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise EvaluationError(
            f"OOF prediction identity mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )
    return expected, {
        uid: _validate_prediction_vector(uid, predictions[uid]) for uid in expected
    }


def _evaluate_uid_rows(
    study_instance_uids: Sequence[str],
    memberships: Mapping[str, FoldMembership],
    values: Mapping[str, np.ndarray],
) -> tuple[TargetEvaluation, ...]:
    metrics: list[TargetEvaluation] = []
    for target_index, target in enumerate(TARGET_COLUMNS):
        labels: list[int] = []
        scores: list[float] = []
        for uid in study_instance_uids:
            label = memberships[uid].reference_labels[target_index]
            if label not in {0, 1}:
                raise EvaluationError(f"reference label is missing for {uid}, {target}")
            labels.append(cast(int, label))
            scores.append(float(values[uid][target_index]))
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
    return tuple(metrics)


def _macro_values(metrics: Sequence[TargetEvaluation]) -> tuple[float | None, float | None]:
    defined = tuple(item.auc for item in metrics if item.auc is not None)
    defined_macro = float(np.mean(defined)) if defined else None
    macro = defined_macro if len(defined) == len(metrics) else None
    return macro, defined_macro


def _bootstrap_metric(
    statistic: str,
    values: Sequence[float],
    *,
    resamples: int,
    confidence_level: float,
) -> BootstrapMetric:
    defined = tuple(float(value) for value in values)
    median: float | None
    lower: float | None
    upper: float | None
    if defined:
        interval = np.quantile(
            np.asarray(defined, dtype=np.float64),
            ((1.0 - confidence_level) / 2.0, 1.0 - (1.0 - confidence_level) / 2.0),
        )
        median = float(np.median(defined))
        lower = float(interval[0])
        upper = float(interval[1])
    else:
        median = lower = upper = None
    return BootstrapMetric(
        statistic=statistic,
        median=median,
        lower=lower,
        upper=upper,
        defined_resamples=len(defined),
        undefined_resamples=resamples - len(defined),
        resamples=resamples,
    )


def bootstrap_oof_predictions(
    predictions: Mapping[str, Sequence[float]],
    fold_manifest: FoldManifest,
    *,
    resamples: int,
    seed: int,
    unit: str = "auto",
    confidence_level: float = 0.95,
) -> BootstrapSummary:
    """Bootstrap OOF studies, or reliable patient groups, with explicit undefined draws."""

    if resamples <= 0:
        raise EvaluationError("bootstrap resamples must be positive")
    if seed < 0:
        raise EvaluationError("bootstrap seed must be non-negative")
    if not 0.0 < confidence_level < 1.0:
        raise EvaluationError("bootstrap confidence level must be between zero and one")
    if unit not in {"auto", "study", "group"}:
        raise EvaluationError("bootstrap unit must be auto, study or group")
    expected, values = _validated_prediction_mapping(predictions, fold_manifest)
    selected_unit = (
        "group"
        if unit == "auto" and fold_manifest.grouping.reliable
        else "study"
        if unit == "auto"
        else unit
    )
    if selected_unit == "group" and not fold_manifest.grouping.reliable:
        raise EvaluationError("group bootstrap requires a reliable grouping assessment")
    memberships = {item.study_instance_uid: item for item in fold_manifest.gold_memberships}
    if selected_unit == "group":
        group_by_uid = fold_manifest.grouping.group_by_study
        grouped: dict[str, list[str]] = {}
        for uid in expected:
            group = group_by_uid.get(uid)
            if group is None:
                raise EvaluationError("reliable grouping assessment does not cover every gold study")
            grouped.setdefault(group, []).append(uid)
        units = tuple(tuple(sorted(grouped[key])) for key in sorted(grouped))
    else:
        units = tuple((uid,) for uid in expected)
    rng = np.random.default_rng(seed)
    target_values: list[list[float]] = [[] for _ in TARGET_COLUMNS]
    macro_values: list[float] = []
    for _ in range(resamples):
        sampled_indices = rng.integers(0, len(units), size=len(units))
        sampled_uids = tuple(uid for index in sampled_indices for uid in units[int(index)])
        metrics = _evaluate_uid_rows(sampled_uids, memberships, values)
        for target_index, metric in enumerate(metrics):
            if metric.auc is not None:
                target_values[target_index].append(metric.auc)
        macro, _ = _macro_values(metrics)
        if macro is not None:
            macro_values.append(macro)
    return BootstrapSummary(
        unit=cast(str, selected_unit),
        unit_count=len(units),
        resamples=resamples,
        seed=seed,
        confidence_level=confidence_level,
        targets=tuple(
            _bootstrap_metric(
                f"roc_auc:{target}",
                target_values[index],
                resamples=resamples,
                confidence_level=confidence_level,
            )
            for index, target in enumerate(TARGET_COLUMNS)
        ),
        macro_auc=_bootstrap_metric(
            "macro_auc",
            macro_values,
            resamples=resamples,
            confidence_level=confidence_level,
        ),
    )


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
        raise EvaluationError(
            f"held-out prediction identity mismatch: missing={missing[:3]}, "
            f"extra={extra[:3]}"
        )
    by_uid = {item.study_instance_uid: item for item in fold_manifest.gold_memberships}
    values = {uid: _validate_prediction_vector(uid, predictions[uid]) for uid in expected}
    metrics = _evaluate_uid_rows(expected, by_uid, values)
    macro, defined_macro = _macro_values(metrics)
    return FoldEvaluation(
        fold=fold,
        study_instance_uids=expected,
        targets=tuple(metrics),
        macro_auc=macro,
        defined_target_macro_auc=defined_macro,
    )


def evaluate_oof_predictions(
    predictions: Mapping[str, Sequence[float]],
    fold_manifest: FoldManifest,
    *,
    bootstrap_resamples: int | None = None,
    bootstrap_seed: int = 20260911,
    bootstrap_unit: str = "auto",
    bootstrap_confidence_level: float = 0.95,
) -> OOFEvaluation:
    """Evaluate exactly one held-out prediction for every gold study in a repeat."""

    expected, values = _validated_prediction_mapping(predictions, fold_manifest)
    memberships = {item.study_instance_uid: item for item in fold_manifest.gold_memberships}
    pooled_targets = _evaluate_uid_rows(expected, memberships, values)
    pooled_macro, defined_target_macro = _macro_values(pooled_targets)
    folds = tuple(
        evaluate_fold_predictions(
            {
                uid: tuple(float(item) for item in values[uid])
                for uid in expected
                if memberships[uid].fold == fold
            },
            fold_manifest,
            fold,
        )
        for fold in FOLD_IDS
    )
    fold_values = tuple(item.macro_auc for item in folds if item.macro_auc is not None)
    all_fold_macros_defined = len(fold_values) == len(folds)
    mean_fold = float(np.mean(fold_values)) if all_fold_macros_defined else None
    defined_fold_mean = float(np.mean(fold_values)) if fold_values else None
    fold_std = float(np.std(fold_values)) if all_fold_macros_defined else None
    fold_min = float(min(fold_values)) if all_fold_macros_defined else None
    fold_max = float(max(fold_values)) if all_fold_macros_defined else None
    bootstrap = (
        bootstrap_oof_predictions(
            predictions,
            fold_manifest,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
            unit=bootstrap_unit,
            confidence_level=bootstrap_confidence_level,
        )
        if bootstrap_resamples is not None
        else None
    )
    return OOFEvaluation(
        repeat_name=fold_manifest.repeat_name,
        fold_manifest_checksum=fold_manifest.manifest_checksum,
        study_instance_uids=expected,
        pooled_targets=pooled_targets,
        pooled_macro_auc=pooled_macro,
        defined_target_macro_auc=defined_target_macro,
        folds=folds,
        mean_fold_macro_auc=mean_fold,
        defined_fold_macro_auc_mean=defined_fold_mean,
        fold_macro_auc_std=fold_std,
        fold_macro_auc_min=fold_min,
        fold_macro_auc_max=fold_max,
        bootstrap=bootstrap,
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
    "BootstrapMetric",
    "BootstrapSummary",
    "EvaluationError",
    "FoldEvaluation",
    "OOFEvaluation",
    "TargetEvaluation",
    "bootstrap_oof_predictions",
    "evaluate_fold_predictions",
    "evaluate_oof_predictions",
    "predict_training_examples",
    "roc_auc_score",
]
