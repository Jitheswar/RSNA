from __future__ import annotations

import csv

import numpy as np
import pytest

from rsna_knee.submission import SubmissionError, write_submission
from rsna_knee.targets import SUBMISSION_COLUMNS, TARGET_COLUMNS


def scores(value: float) -> list[float]:
    return [value] * len(TARGET_COLUMNS)


def test_write_submission_preserves_expected_uid_order_and_exact_schema(tmp_path) -> None:
    output_path = tmp_path / "submission.csv"

    write_submission(
        output_path,
        expected_study_uids=("study-b", "study-a"),
        predictions={"study-a": scores(0.25), "study-b": scores(0.75)},
    )

    with output_path.open(newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == list(SUBMISSION_COLUMNS)
    assert [row[0] for row in rows[1:]] == ["study-b", "study-a"]
    assert all(len(row) == 13 for row in rows)
    assert not output_path.read_text().splitlines()[0].endswith(",")


@pytest.mark.parametrize(
    "predictions",
    [
        {"study-a": scores(0.5)},
        {"study-a": scores(0.5), "study-b": scores(0.5), "study-c": scores(0.5)},
        {"study-a": scores(float("nan")), "study-b": scores(0.5)},
        {"study-a": scores(-0.01), "study-b": scores(0.5)},
        {"study-a": np.zeros(len(TARGET_COLUMNS) - 1), "study-b": scores(0.5)},
    ],
)
def test_write_submission_rejects_incomplete_or_invalid_predictions(tmp_path, predictions) -> None:
    with pytest.raises(SubmissionError):
        write_submission(tmp_path / "submission.csv", ("study-a", "study-b"), predictions)


def test_write_submission_refuses_to_overwrite_an_existing_artifact(tmp_path) -> None:
    output_path = tmp_path / "submission.csv"
    write_submission(output_path, ("study-a",), {"study-a": scores(0.5)})

    with pytest.raises(FileExistsError):
        write_submission(output_path, ("study-a",), {"study-a": scores(0.25)})
