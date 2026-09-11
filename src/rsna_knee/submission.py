"""Strict competition-format output validation and writing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .targets import NUM_TARGETS, SUBMISSION_COLUMNS


class SubmissionError(ValueError):
    """Raised when predictions cannot satisfy the competition output contract."""


def _ordered_rows(
    expected_study_uids: Sequence[str],
    predictions: Mapping[str, Sequence[float] | np.ndarray],
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    expected = tuple(expected_study_uids)
    if not expected:
        raise SubmissionError("at least one expected study UID is required")
    if len(set(expected)) != len(expected):
        raise SubmissionError("expected study UIDs must be unique")
    if any(not isinstance(uid, str) or not uid for uid in expected):
        raise SubmissionError("expected study UIDs must be non-empty strings")

    expected_set = set(expected)
    prediction_set = set(predictions)
    missing = expected_set - prediction_set
    extra = prediction_set - expected_set
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if extra:
            details.append(f"extra={sorted(extra)}")
        raise SubmissionError("prediction UID set does not match expected UID set (" + ", ".join(details) + ")")

    rows: list[tuple[str, tuple[float, ...]]] = []
    for uid in expected:
        try:
            values = np.asarray(predictions[uid], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SubmissionError(f"scores for {uid} are not numeric") from exc
        if values.shape != (NUM_TARGETS,):
            raise SubmissionError(f"scores for {uid} must contain exactly {NUM_TARGETS} values")
        if not np.isfinite(values).all():
            raise SubmissionError(f"scores for {uid} must all be finite")
        if (values < 0.0).any() or (values > 1.0).any():
            raise SubmissionError(f"scores for {uid} must be within [0, 1]")
        rows.append((uid, tuple(float(value) for value in values)))
    return tuple(rows)


def validate_predictions(
    expected_study_uids: Sequence[str],
    predictions: Mapping[str, Sequence[float] | np.ndarray],
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Validate and return rows in the exact expected study order."""

    return _ordered_rows(expected_study_uids, predictions)


def write_submission(
    output_path: str | Path,
    expected_study_uids: Sequence[str],
    predictions: Mapping[str, Sequence[float] | np.ndarray],
) -> Path:
    """Write a validated, index-free ``submission.csv``-compatible file."""

    rows = _ordered_rows(expected_study_uids, predictions)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(SUBMISSION_COLUMNS)
        writer.writerows((uid, *values) for uid, values in rows)
    return destination
