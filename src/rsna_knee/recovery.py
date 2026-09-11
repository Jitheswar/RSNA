"""Safe study-level fallback priors and recovery accounting primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .targets import NUM_TARGETS, TARGET_COLUMNS


RECOVERY_POLICY_VERSION = "rsna-knee-recovery-v1"
DEFAULT_SMOOTHING_ALPHA = 1.0


class RecoveryConfigurationError(ValueError):
    """Raised when fallback-prior inputs are not safe to use."""


def _study_uids(values: Sequence[str], field: str) -> tuple[str, ...]:
    if any(not isinstance(value, str) for value in values):
        raise RecoveryConfigurationError(f"{field} must contain string study identifiers")
    result = tuple(value.strip() for value in values)
    if any(not value for value in result):
        raise RecoveryConfigurationError(f"{field} must contain non-empty study identifiers")
    if len(set(result)) != len(result):
        raise RecoveryConfigurationError(f"{field} must not contain duplicate study identifiers")
    return tuple(sorted(result))


def _validated_labels(
    reference_labels: Mapping[str, Sequence[int | None]],
) -> dict[str, tuple[int | None, ...]]:
    result: dict[str, tuple[int | None, ...]] = {}
    for raw_uid, raw_labels in reference_labels.items():
        uid = str(raw_uid).strip()
        if not uid:
            raise RecoveryConfigurationError("reference labels contain an empty study identifier")
        if uid in result:
            raise RecoveryConfigurationError(f"reference labels repeat study {uid}")
        labels = tuple(raw_labels)
        if len(labels) != NUM_TARGETS:
            raise RecoveryConfigurationError(
                f"reference labels for {uid} must contain {NUM_TARGETS} targets"
            )
        if any(
            label is not None
            and (
                isinstance(label, bool)
                or not isinstance(label, (int, np.integer))
                or int(label) not in (0, 1)
            )
            for label in labels
        ):
            raise RecoveryConfigurationError(
                f"reference labels for {uid} must contain only 0, 1 or missing values"
            )
        result[uid] = tuple(None if label is None else int(label) for label in labels)
    return result


@dataclass(frozen=True)
class ReferenceFallbackPrior:
    """Laplace-smoothed fallback scores derived from eligible references.

    ``eligible_study_uids`` is retained for private provenance only.  It must
    identify reference-label rows supplied by the caller; report-derived or
    weak labels are intentionally not accepted as an implicit source.
    """

    scores: np.ndarray
    positive_counts: tuple[int, ...]
    negative_counts: tuple[int, ...]
    eligible_study_count: int
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA
    eligible_study_uids: tuple[str, ...] = ()
    source: str = "trusted_reference_labels"
    policy_version: str = RECOVERY_POLICY_VERSION

    def __post_init__(self) -> None:
        values = np.asarray(self.scores, dtype=np.float64)
        if values.shape != (NUM_TARGETS,) or not np.isfinite(values).all():
            raise RecoveryConfigurationError(
                f"fallback prior scores must contain {NUM_TARGETS} finite values"
            )
        if (values < 0.0).any() or (values > 1.0).any():
            raise RecoveryConfigurationError("fallback prior scores must be within [0, 1]")
        if len(self.positive_counts) != NUM_TARGETS or len(self.negative_counts) != NUM_TARGETS:
            raise RecoveryConfigurationError("fallback prior counts must contain every target")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 0
            for value in self.positive_counts + self.negative_counts
        ):
            raise RecoveryConfigurationError("fallback prior counts must be non-negative integers")
        if (
            isinstance(self.eligible_study_count, bool)
            or not isinstance(self.eligible_study_count, (int, np.integer))
            or self.eligible_study_count < 0
        ):
            raise RecoveryConfigurationError("eligible_study_count must be non-negative")
        try:
            valid_alpha = bool(np.isfinite(self.smoothing_alpha) and self.smoothing_alpha > 0.0)
        except TypeError as exc:
            raise RecoveryConfigurationError("smoothing_alpha must be positive and finite") from exc
        if not valid_alpha:
            raise RecoveryConfigurationError("smoothing_alpha must be positive and finite")
        if self.policy_version != RECOVERY_POLICY_VERSION:
            raise RecoveryConfigurationError(f"unsupported recovery policy {self.policy_version!r}")
        if not self.source.strip():
            raise RecoveryConfigurationError("fallback prior source must be non-empty")
        uids = _study_uids(self.eligible_study_uids, "eligible_study_uids")
        if uids and len(uids) != self.eligible_study_count:
            raise RecoveryConfigurationError(
                "eligible_study_uids does not agree with eligible_study_count"
            )
        object.__setattr__(self, "scores", values.copy())
        object.__setattr__(self, "eligible_study_uids", uids)
        object.__setattr__(self, "positive_counts", tuple(int(value) for value in self.positive_counts))
        object.__setattr__(self, "negative_counts", tuple(int(value) for value in self.negative_counts))

    @property
    def known_label_counts(self) -> tuple[int, ...]:
        """Number of known reference labels contributing to each target."""

        return tuple(
            positive + negative
            for positive, negative in zip(self.positive_counts, self.negative_counts)
        )

    def to_public_dict(self) -> dict[str, object]:
        """Return identifier-free prior evidence suitable for public output."""

        return {
            "policy_version": self.policy_version,
            "source": self.source,
            "smoothing": "laplace",
            "smoothing_alpha": self.smoothing_alpha,
            "eligible_reference_studies": self.eligible_study_count,
            "fallback_when_target_unavailable": 0.5,
            "targets": {
                target: {
                    "positive": self.positive_counts[index],
                    "negative": self.negative_counts[index],
                    "known": self.known_label_counts[index],
                    "score": float(self.scores[index]),
                }
                for index, target in enumerate(TARGET_COLUMNS)
            },
        }

    def to_private_dict(self) -> dict[str, object]:
        """Return prior evidence including the eligible identifier set."""

        payload = self.to_public_dict()
        payload["eligible_study_uids"] = list(self.eligible_study_uids)
        return payload


def compute_laplace_smoothed_reference_priors(
    reference_labels: Mapping[str, Sequence[int | None]],
    *,
    eligible_study_uids: Sequence[str] | None = None,
    held_out_study_uids: Sequence[str] = (),
    alpha: float = DEFAULT_SMOOTHING_ALPHA,
) -> ReferenceFallbackPrior:
    """Compute per-target emergency scores from explicitly eligible references.

    The mapping is a reference-label input, not a general supervision map.
    When ``eligible_study_uids`` is supplied, only those rows contribute.  A
    held-out study can never be eligible; missing labels for an eligible target
    are ignored and produce the documented midpoint when no known label exists.
    """

    labels_by_study = _validated_labels(reference_labels)
    eligible = (
        _study_uids(eligible_study_uids, "eligible_study_uids")
        if eligible_study_uids is not None
        else tuple(sorted(labels_by_study))
    )
    held_out = _study_uids(held_out_study_uids, "held_out_study_uids")
    overlap = sorted(set(eligible).intersection(held_out))
    if overlap:
        raise RecoveryConfigurationError(
            f"eligible reference labels overlap held-out studies: {overlap[:3]}"
        )
    missing = sorted(set(eligible).difference(labels_by_study))
    if missing:
        raise RecoveryConfigurationError(
            f"eligible reference labels are missing study rows: {missing[:3]}"
        )
    try:
        valid_alpha = bool(np.isfinite(alpha) and alpha > 0.0)
    except TypeError as exc:
        raise RecoveryConfigurationError("alpha must be positive and finite") from exc
    if not valid_alpha:
        raise RecoveryConfigurationError("alpha must be positive and finite")

    positive = [0] * NUM_TARGETS
    negative = [0] * NUM_TARGETS
    for uid in eligible:
        labels = labels_by_study[uid]
        for index, label in enumerate(labels):
            if label == 1:
                positive[index] += 1
            elif label == 0:
                negative[index] += 1

    scores = np.full(NUM_TARGETS, 0.5, dtype=np.float64)
    for index, (positive_count, negative_count) in enumerate(zip(positive, negative)):
        known = positive_count + negative_count
        if known:
            scores[index] = (positive_count + alpha) / (known + 2.0 * alpha)
    return ReferenceFallbackPrior(
        scores=scores,
        positive_counts=tuple(positive),
        negative_counts=tuple(negative),
        eligible_study_count=len(eligible),
        smoothing_alpha=float(alpha),
        eligible_study_uids=eligible,
        source="trusted_reference_labels" if eligible else "no_eligible_reference_labels",
    )


def default_reference_fallback_prior() -> ReferenceFallbackPrior:
    """Return the explicit 0.5 fallback used when no references are available."""

    return compute_laplace_smoothed_reference_priors({}, eligible_study_uids=())


__all__ = [
    "DEFAULT_SMOOTHING_ALPHA",
    "RECOVERY_POLICY_VERSION",
    "RecoveryConfigurationError",
    "ReferenceFallbackPrior",
    "compute_laplace_smoothed_reference_priors",
    "default_reference_fallback_prior",
]
