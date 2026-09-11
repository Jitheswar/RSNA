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


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based ranks with tied values sharing their average rank."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    index = 0
    while index < sorted_values.size:
        end = index + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    return ranks


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
    ranks = _average_ranks(score_array)
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
    within_fold_macro_auc: BootstrapMetric | None = None

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
        ) or (
            self.within_fold_macro_auc is not None
            and self.within_fold_macro_auc.resamples != self.resamples
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
            "within_fold_macro_auc": (
                self.within_fold_macro_auc.to_dict()
                if self.within_fold_macro_auc is not None
                else None
            ),
        }


WITHIN_FOLD_SIGNAL_ALPHA = 0.05
WITHIN_FOLD_SIGNAL_RULE = (
    "Measurable image signal is claimed only if a one-sided within-fold label-permutation test "
    "of the pair-weighted within-fold macro AUC gives p < 0.05. Whole label vectors are permuted "
    "among the studies of each held-out fold, so fold-level score shifts cannot create signal."
)


@dataclass(frozen=True)
class TargetPairDecomposition:
    """Exact split of one target's pooled OOF pairs into within- and cross-fold pairs.

    ``pooled_auc`` equals the pair-count-weighted combination of the two
    components.  Cross-fold pairs compare scores from different fold models,
    so they also reward or penalize differences in fold score scale.
    """

    target: str
    blocks: tuple[tuple[int, int, float, int], ...]

    def __post_init__(self) -> None:
        if self.target not in TARGET_COLUMNS:
            raise EvaluationError(f"unknown target {self.target!r}")
        if tuple((positive, negative) for positive, negative, _, _ in self.blocks) != tuple(
            (positive, negative) for positive in FOLD_IDS for negative in FOLD_IDS
        ):
            raise EvaluationError("pair decomposition must contain every fold block in order")

    def _sum(self, within: bool) -> tuple[float, int]:
        wins = sum(item[2] for item in self.blocks if (item[0] == item[1]) == within)
        pairs = sum(item[3] for item in self.blocks if (item[0] == item[1]) == within)
        return wins, pairs

    @property
    def within_fold_pairs(self) -> int:
        return self._sum(True)[1]

    @property
    def cross_fold_pairs(self) -> int:
        return self._sum(False)[1]

    @property
    def pooled_pairs(self) -> int:
        return self.within_fold_pairs + self.cross_fold_pairs

    @property
    def within_fold_auc(self) -> float | None:
        wins, pairs = self._sum(True)
        return wins / pairs if pairs else None

    @property
    def cross_fold_auc(self) -> float | None:
        wins, pairs = self._sum(False)
        return wins / pairs if pairs else None

    @property
    def pooled_auc(self) -> float | None:
        pairs = self.pooled_pairs
        return (self._sum(True)[0] + self._sum(False)[0]) / pairs if pairs else None

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "pooled_auc": self.pooled_auc,
            "pooled_pairs": self.pooled_pairs,
            "within_fold_auc": self.within_fold_auc,
            "within_fold_pairs": self.within_fold_pairs,
            "within_fold_pair_share": (
                self.within_fold_pairs / self.pooled_pairs if self.pooled_pairs else None
            ),
            "cross_fold_auc": self.cross_fold_auc,
            "cross_fold_pairs": self.cross_fold_pairs,
            "cross_fold_pair_share": (
                self.cross_fold_pairs / self.pooled_pairs if self.pooled_pairs else None
            ),
            "blocks": [
                {
                    "positive_fold": positive_fold,
                    "negative_fold": negative_fold,
                    "auc": wins / pairs if pairs else None,
                    "pairs": pairs,
                }
                for positive_fold, negative_fold, wins, pairs in self.blocks
            ],
        }


@dataclass(frozen=True)
class PrevalenceBaseline:
    """Metrics for scoring every held-out study with its fold's training prevalence.

    The baseline has no image input.  Its pooled AUC measures how much of a
    pooled metric can arise from fold-level base rates alone; within a fold
    its scores are constant, so fold and within-fold AUCs are 0.5 when defined.
    """

    targets: tuple[TargetEvaluation, ...]
    macro_auc: float | None
    fold_macro_aucs: tuple[float | None, ...]
    within_fold_macro_auc: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "definition": (
                "score = prevalence of the target among the gold studies used to train the "
                "held-out study's fold; no image input"
            ),
            "targets": [item.to_dict() for item in self.targets],
            "pooled_macro_auc": self.macro_auc,
            "fold_macro_aucs": list(self.fold_macro_aucs),
            "pair_weighted_within_fold_macro_auc": self.within_fold_macro_auc,
        }


@dataclass(frozen=True)
class FoldScoreScale:
    """Location and spread of one fold model's held-out scores for one target."""

    fold: int
    target: str
    mean_score: float
    sd_score: float
    training_prevalence: float
    held_out_prevalence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "target": self.target,
            "mean_score": self.mean_score,
            "sd_score": self.sd_score,
            "training_prevalence": self.training_prevalence,
            "held_out_prevalence": self.held_out_prevalence,
        }


@dataclass(frozen=True)
class WithinFoldPermutationTest:
    """Label-permutation null for the pair-weighted within-fold macro AUC."""

    observed: float
    resamples: int
    seed: int
    null_mean: float
    null_sd: float
    null_quantile_95: float
    p_upper: float
    p_two_sided: float

    @property
    def rejects_no_signal(self) -> bool:
        return self.p_upper < WITHIN_FOLD_SIGNAL_ALPHA

    def to_dict(self) -> dict[str, object]:
        return {
            "statistic": "pair_weighted_within_fold_macro_auc",
            "permutation_unit": "whole reference-label vectors within each held-out fold",
            "observed": self.observed,
            "resamples": self.resamples,
            "seed": self.seed,
            "null_mean": self.null_mean,
            "null_sd": self.null_sd,
            "null_quantile_95": self.null_quantile_95,
            "p_upper": self.p_upper,
            "p_two_sided": self.p_two_sided,
            "p_value_definition": "(1 + resamples at least as extreme) / (1 + resamples)",
            "alpha": WITHIN_FOLD_SIGNAL_ALPHA,
            "rule": WITHIN_FOLD_SIGNAL_RULE,
            "measurable_image_signal": self.rejects_no_signal,
        }


@dataclass(frozen=True)
class OOFDiagnostics:
    """Fold-structure diagnostics that qualify a pooled OOF macro AUC."""

    pair_decomposition: tuple[TargetPairDecomposition, ...]
    within_fold_macro_auc: float | None
    cross_fold_macro_auc: float | None
    prevalence_baseline: PrevalenceBaseline
    fold_score_scale: tuple[FoldScoreScale, ...]
    mean_score_vs_training_prevalence_pearson: float | None
    mean_score_vs_training_prevalence_spearman: float | None
    mean_score_vs_held_out_prevalence_pearson: float | None
    between_fold_to_within_fold_sd_ratio: tuple[float | None, ...]
    fold_standardized_macro_auc: float | None
    permutation_test: WithinFoldPermutationTest | None

    def __post_init__(self) -> None:
        if tuple(item.target for item in self.pair_decomposition) != TARGET_COLUMNS:
            raise EvaluationError("pair decomposition must use competition order")
        if len(self.between_fold_to_within_fold_sd_ratio) != len(TARGET_COLUMNS):
            raise EvaluationError("score-scale ratios must cover every target")

    @property
    def mean_within_fold_pair_share(self) -> float | None:
        shares = [
            item.within_fold_pairs / item.pooled_pairs
            for item in self.pair_decomposition
            if item.pooled_pairs
        ]
        return float(np.mean(shares)) if shares else None

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_weighted_within_fold_macro_auc": self.within_fold_macro_auc,
            "cross_fold_macro_auc": self.cross_fold_macro_auc,
            "mean_within_fold_pair_share": self.mean_within_fold_pair_share,
            "pair_weighted_within_fold_definition": (
                "per target, AUC over positive/negative pairs whose studies share a held-out fold, "
                "pooled across folds by pair count; macro = unweighted mean of the twelve targets"
            ),
            "pair_decomposition": [item.to_dict() for item in self.pair_decomposition],
            "training_prevalence_baseline": self.prevalence_baseline.to_dict(),
            "fold_score_scale": {
                "cells": [item.to_dict() for item in self.fold_score_scale],
                "mean_score_vs_training_prevalence": {
                    "pearson": self.mean_score_vs_training_prevalence_pearson,
                    "spearman": self.mean_score_vs_training_prevalence_spearman,
                    "cells": "fold x target",
                },
                "mean_score_vs_held_out_prevalence_pearson": (
                    self.mean_score_vs_held_out_prevalence_pearson
                ),
                "between_fold_sd_of_means_over_mean_within_fold_sd": {
                    target: ratio
                    for target, ratio in zip(TARGET_COLUMNS, self.between_fold_to_within_fold_sd_ratio)
                },
                "fold_standardized_pooled_macro_auc": self.fold_standardized_macro_auc,
                "fold_standardized_definition": (
                    "diagnostic only: pooled AUC after z-scoring each fold's scores per target"
                ),
            },
            "within_fold_permutation_test": (
                self.permutation_test.to_dict() if self.permutation_test is not None else None
            ),
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
    diagnostics: OOFDiagnostics | None = None
    fold_generation_method: str | None = None
    fold_manifest_superseded: bool = False

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
            "schema_version": "rsna-knee-oof-evaluation-v2",
            "repeat_name": self.repeat_name,
            "fold_manifest_checksum": self.fold_manifest_checksum,
            "fold_generation_method": self.fold_generation_method,
            "fold_manifest_superseded": self.fold_manifest_superseded,
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
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics is not None else None,
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


def _oof_arrays(
    study_instance_uids: Sequence[str],
    memberships: Mapping[str, FoldMembership],
    values: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return label, score and fold arrays in the given study order."""

    labels = np.array(
        [memberships[uid].reference_labels for uid in study_instance_uids], dtype=np.int64
    ).reshape(len(study_instance_uids), NUM_TARGETS)
    scores = np.array([values[uid] for uid in study_instance_uids], dtype=np.float64).reshape(
        len(study_instance_uids), NUM_TARGETS
    )
    folds = np.array([memberships[uid].fold for uid in study_instance_uids], dtype=np.int64)
    return labels, scores, folds


def _pair_wins(positive: np.ndarray, negative: np.ndarray) -> tuple[float, int]:
    """Count positive-over-negative orderings, with ties as half, and all pairs."""

    if positive.size == 0 or negative.size == 0:
        return 0.0, 0
    difference = positive[:, None] - negative[None, :]
    wins = np.count_nonzero(difference > 0) + 0.5 * np.count_nonzero(difference == 0)
    return float(wins), int(difference.size)


def _array_target_evaluations(labels: np.ndarray, scores: np.ndarray) -> tuple[TargetEvaluation, ...]:
    metrics: list[TargetEvaluation] = []
    for index, target in enumerate(TARGET_COLUMNS):
        auc = roc_auc_score(labels[:, index].tolist(), scores[:, index].tolist())
        metrics.append(
            TargetEvaluation(
                target=target,
                auc=auc,
                positive_count=int(np.sum(labels[:, index] == 1)),
                negative_count=int(np.sum(labels[:, index] == 0)),
                status="defined" if auc is not None else "undefined_one_class",
            )
        )
    return tuple(metrics)


def _within_fold_macro(labels: np.ndarray, scores: np.ndarray, folds: np.ndarray) -> float | None:
    """Pair-weighted within-fold macro AUC; undefined when a target has no within-fold pair."""

    terms: list[float] = []
    for index in range(NUM_TARGETS):
        wins = 0.0
        pairs = 0
        for fold in FOLD_IDS:
            rows = folds == fold
            fold_wins, fold_pairs = _pair_wins(
                scores[rows & (labels[:, index] == 1), index],
                scores[rows & (labels[:, index] == 0), index],
            )
            wins += fold_wins
            pairs += fold_pairs
        if pairs == 0:
            return None
        terms.append(wins / pairs)
    return float(np.mean(terms))


def _pair_decomposition(
    labels: np.ndarray, scores: np.ndarray, folds: np.ndarray
) -> tuple[TargetPairDecomposition, ...]:
    decomposition: list[TargetPairDecomposition] = []
    for index, target in enumerate(TARGET_COLUMNS):
        blocks: list[tuple[int, int, float, int]] = []
        for positive_fold in FOLD_IDS:
            positive = scores[(folds == positive_fold) & (labels[:, index] == 1), index]
            for negative_fold in FOLD_IDS:
                negative = scores[(folds == negative_fold) & (labels[:, index] == 0), index]
                wins, pairs = _pair_wins(positive, negative)
                blocks.append((positive_fold, negative_fold, wins, pairs))
        decomposition.append(TargetPairDecomposition(target=target, blocks=tuple(blocks)))
    return tuple(decomposition)


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _within_fold_permutation_test(
    labels: np.ndarray,
    scores: np.ndarray,
    folds: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> WithinFoldPermutationTest | None:
    """Permute whole label vectors within each fold and recompute the within-fold macro.

    Within-fold ranks of every score column are fixed under the null, so each
    resample needs only the permuted positive rank sums.
    """

    if resamples <= 0:
        raise EvaluationError("permutation resamples must be positive")
    if seed < 0:
        raise EvaluationError("permutation seed must be non-negative")
    rng = np.random.default_rng(seed)
    null_u = np.zeros((resamples, NUM_TARGETS), dtype=np.float64)
    observed_u = np.zeros(NUM_TARGETS, dtype=np.float64)
    denominators = np.zeros(NUM_TARGETS, dtype=np.float64)
    chunk = 1024
    for fold in FOLD_IDS:
        rows = np.flatnonzero(folds == fold)
        if rows.size == 0:
            continue
        fold_labels = labels[rows].astype(np.float64)
        ranks = np.column_stack(
            [_average_ranks(scores[rows, index]) for index in range(NUM_TARGETS)]
        )
        positives = fold_labels.sum(axis=0)
        offset = positives * (positives + 1.0) / 2.0
        denominators += positives * (rows.size - positives)
        observed_u += (fold_labels * ranks).sum(axis=0) - offset
        for start in range(0, resamples, chunk):
            count = min(chunk, resamples - start)
            permutations = np.argsort(rng.random((count, rows.size)), axis=1)
            permuted = fold_labels[permutations]
            null_u[start : start + count] += np.einsum("rnt,nt->rt", permuted, ranks) - offset
    if np.any(denominators == 0.0):
        return None
    observed = float(np.mean(observed_u / denominators))
    null = (null_u / denominators).mean(axis=1)
    tolerance = 1e-12
    return WithinFoldPermutationTest(
        observed=observed,
        resamples=resamples,
        seed=seed,
        null_mean=float(np.mean(null)),
        null_sd=float(np.std(null)),
        null_quantile_95=float(np.quantile(null, 0.95)),
        p_upper=float((1 + np.count_nonzero(null >= observed - tolerance)) / (resamples + 1)),
        p_two_sided=float(
            (1 + np.count_nonzero(np.abs(null - 0.5) >= abs(observed - 0.5) - tolerance))
            / (resamples + 1)
        ),
    )


def compute_oof_diagnostics(
    labels: np.ndarray,
    scores: np.ndarray,
    folds: np.ndarray,
    *,
    permutation_resamples: int | None = None,
    permutation_seed: int = 20260911,
) -> OOFDiagnostics:
    """Qualify a pooled OOF AUC by fold structure, base rates and fold score scale."""

    if labels.shape != scores.shape or labels.shape[1:] != (NUM_TARGETS,):
        raise EvaluationError("diagnostics require matching study-by-target label and score arrays")
    if folds.shape != (labels.shape[0],):
        raise EvaluationError("diagnostics require one fold per study")
    for fold in FOLD_IDS:
        held_out = folds == fold
        if not held_out.any() or held_out.all():
            raise EvaluationError("diagnostics require held-out and training studies for every fold")
    decomposition = _pair_decomposition(labels, scores, folds)
    within_values = tuple(item.within_fold_auc for item in decomposition)
    cross_values = tuple(item.cross_fold_auc for item in decomposition)

    baseline_scores = np.zeros(labels.shape, dtype=np.float64)
    cells: list[FoldScoreScale] = []
    for fold in FOLD_IDS:
        held_out = folds == fold
        training_prevalence = labels[~held_out].mean(axis=0)
        baseline_scores[held_out] = training_prevalence
        for index, target in enumerate(TARGET_COLUMNS):
            fold_scores = scores[held_out, index]
            cells.append(
                FoldScoreScale(
                    fold=fold,
                    target=target,
                    mean_score=float(fold_scores.mean()),
                    sd_score=float(fold_scores.std()),
                    training_prevalence=float(training_prevalence[index]),
                    held_out_prevalence=float(labels[held_out, index].mean()),
                )
            )
    baseline_targets = _array_target_evaluations(labels, baseline_scores)
    baseline = PrevalenceBaseline(
        targets=baseline_targets,
        macro_auc=_macro_values(baseline_targets)[0],
        fold_macro_aucs=tuple(
            _macro_values(
                _array_target_evaluations(labels[folds == fold], baseline_scores[folds == fold])
            )[0]
            for fold in FOLD_IDS
        ),
        within_fold_macro_auc=_within_fold_macro(labels, baseline_scores, folds),
    )

    mean_scores = np.array([item.mean_score for item in cells])
    training_prevalences = np.array([item.training_prevalence for item in cells])
    held_out_prevalences = np.array([item.held_out_prevalence for item in cells])
    ratios: list[float | None] = []
    standardized = np.zeros(scores.shape, dtype=np.float64)
    for index in range(NUM_TARGETS):
        fold_means = np.array([scores[folds == fold, index].mean() for fold in FOLD_IDS])
        fold_sds = np.array([scores[folds == fold, index].std() for fold in FOLD_IDS])
        ratios.append(float(np.std(fold_means) / np.mean(fold_sds)) if np.mean(fold_sds) > 0 else None)
        for fold, mean, sd in zip(FOLD_IDS, fold_means, fold_sds):
            rows = folds == fold
            standardized[rows, index] = (scores[rows, index] - mean) / sd if sd > 0 else 0.0
    return OOFDiagnostics(
        pair_decomposition=decomposition,
        within_fold_macro_auc=(
            float(np.mean(cast(Sequence[float], within_values)))
            if all(value is not None for value in within_values)
            else None
        ),
        cross_fold_macro_auc=(
            float(np.mean(cast(Sequence[float], cross_values)))
            if all(value is not None for value in cross_values)
            else None
        ),
        prevalence_baseline=baseline,
        fold_score_scale=tuple(cells),
        mean_score_vs_training_prevalence_pearson=_correlation(mean_scores, training_prevalences),
        mean_score_vs_training_prevalence_spearman=_correlation(
            _average_ranks(mean_scores), _average_ranks(training_prevalences)
        ),
        mean_score_vs_held_out_prevalence_pearson=_correlation(mean_scores, held_out_prevalences),
        between_fold_to_within_fold_sd_ratio=tuple(ratios),
        fold_standardized_macro_auc=_macro_values(_array_target_evaluations(labels, standardized))[0],
        permutation_test=(
            _within_fold_permutation_test(
                labels,
                scores,
                folds,
                resamples=permutation_resamples,
                seed=permutation_seed,
            )
            if permutation_resamples
            else None
        ),
    )


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
    labels, scores, folds = _oof_arrays(expected, memberships, values)
    row_by_uid = {uid: row for row, uid in enumerate(expected)}
    unit_rows = tuple(tuple(row_by_uid[uid] for uid in unit) for unit in units)
    rng = np.random.default_rng(seed)
    target_values: list[list[float]] = [[] for _ in TARGET_COLUMNS]
    macro_values: list[float] = []
    within_values: list[float] = []
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
        rows = np.fromiter(
            (row for index in sampled_indices for row in unit_rows[int(index)]), dtype=np.int64
        )
        within = _within_fold_macro(labels[rows], scores[rows], folds[rows])
        if within is not None:
            within_values.append(within)
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
        within_fold_macro_auc=_bootstrap_metric(
            "pair_weighted_within_fold_macro_auc",
            within_values,
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
    include_diagnostics: bool = True,
    permutation_resamples: int | None = None,
    permutation_seed: int = 20260911,
) -> OOFEvaluation:
    """Evaluate exactly one held-out prediction for every gold study in a repeat.

    Diagnostics decompose pooled pairs into within- and cross-fold pairs and
    add the training-prevalence baseline and fold score-scale summaries; a
    positive ``permutation_resamples`` also runs the within-fold signal test.
    """

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
    diagnostics = None
    if include_diagnostics:
        labels, scores, fold_array = _oof_arrays(expected, memberships, values)
        diagnostics = compute_oof_diagnostics(
            labels,
            scores,
            fold_array,
            permutation_resamples=permutation_resamples,
            permutation_seed=permutation_seed,
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
        diagnostics=diagnostics,
        fold_generation_method=fold_manifest.generation_method,
        fold_manifest_superseded=fold_manifest.superseded,
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
    "FoldScoreScale",
    "OOFDiagnostics",
    "OOFEvaluation",
    "PrevalenceBaseline",
    "TargetEvaluation",
    "TargetPairDecomposition",
    "WITHIN_FOLD_SIGNAL_ALPHA",
    "WITHIN_FOLD_SIGNAL_RULE",
    "WithinFoldPermutationTest",
    "bootstrap_oof_predictions",
    "compute_oof_diagnostics",
    "evaluate_fold_predictions",
    "evaluate_oof_predictions",
    "predict_training_examples",
    "roc_auc_score",
]
