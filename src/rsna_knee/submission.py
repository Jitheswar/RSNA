"""Strict competition-format output validation and writing."""

from __future__ import annotations

import csv
import os
import tempfile
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


def validate_submission_csv(
    input_path: str | Path,
    expected_study_uids: Sequence[str],
) -> dict[str, object]:
    """Validate an existing CSV against the complete competition contract."""

    expected = tuple(expected_study_uids)
    if not expected:
        raise SubmissionError("at least one expected study UID is required")
    if len(set(expected)) != len(expected) or any(not isinstance(uid, str) or not uid for uid in expected):
        raise SubmissionError("expected study UIDs must be unique non-empty strings")
    source = Path(input_path)
    if not source.is_file() or source.is_symlink():
        raise SubmissionError(f"submission CSV is not a regular file: {source}")
    try:
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise SubmissionError(f"could not read submission CSV {source}") from exc
    if not rows:
        raise SubmissionError("submission CSV is empty")
    if rows[0] != list(SUBMISSION_COLUMNS):
        raise SubmissionError("submission CSV header does not match the exact target order")
    data_rows = rows[1:]
    if len(data_rows) != len(expected):
        raise SubmissionError(
            f"submission CSV contains {len(data_rows)} rows; expected {len(expected)}"
        )
    observed_uids: list[str] = []
    for row in data_rows:
        if len(row) != len(SUBMISSION_COLUMNS):
            raise SubmissionError("submission CSV contains an incorrect number of columns")
        uid = row[0]
        if not uid:
            raise SubmissionError("submission CSV contains an empty StudyInstanceUID")
        observed_uids.append(uid)
        try:
            scores = np.asarray([float(value) for value in row[1:]], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SubmissionError(f"submission CSV scores are not numeric for {uid}") from exc
        if scores.shape != (NUM_TARGETS,) or not np.isfinite(scores).all():
            raise SubmissionError(f"submission CSV scores are not finite for {uid}")
        if (scores < 0.0).any() or (scores > 1.0).any():
            raise SubmissionError(f"submission CSV scores are outside [0, 1] for {uid}")
    if len(set(observed_uids)) != len(observed_uids):
        raise SubmissionError("submission CSV contains duplicate StudyInstanceUID rows")
    if tuple(observed_uids) != expected:
        raise SubmissionError("submission CSV StudyInstanceUID rows are not in expected order")
    return {
        "row_count": len(data_rows),
        "column_count": len(rows[0]),
        "exact_header": True,
        "exact_uid_order": True,
        "scores_finite": True,
        "scores_bounded": True,
    }


def write_submission(
    output_path: str | Path,
    expected_study_uids: Sequence[str],
    predictions: Mapping[str, Sequence[float] | np.ndarray],
) -> Path:
    """Write a validated, index-free ``submission.csv``-compatible file."""

    rows = _ordered_rows(expected_study_uids, predictions)
    destination = Path(output_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite protected submission {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-",
        dir=destination.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(SUBMISSION_COLUMNS)
            writer.writerows((uid, *values) for uid, values in rows)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite protected submission {destination}") from exc
    except BaseException:
        raise
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return destination
