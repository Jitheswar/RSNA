from __future__ import annotations

import pytest

from rsna_knee.report_only_aggregate import (
    ReportOnlyAggregateError,
    aggregate_report_only_records,
)
from rsna_knee.report_labels import LabelState
from rsna_knee.targets import TARGET_COLUMNS


def _record(study: str, states: tuple[str, ...]) -> dict[str, object]:
    values = [
        1 if state == LabelState.POSITIVE.value else 0 if state == LabelState.NEGATIVE.value else None
        for state in states
    ]
    masks = [int(value is not None) for value in values]
    return {
        "artifact_kind": "admitted_training_labels",
        "study_instance_uid": study,
        "interpretation": {
            "target_order": list(TARGET_COLUMNS),
            "states": list(states),
            "values": values,
            "supervision_mask": masks,
        },
    }


def _partitions() -> tuple[list[dict[str, object]], set[str]]:
    states_for_first_target = (
        LabelState.POSITIVE.value,
        LabelState.NEGATIVE.value,
        LabelState.NOT_MENTIONED.value,
        LabelState.UNCERTAIN.value,
        LabelState.UNKNOWN.value,
        LabelState.EXTRACTION_FAILED.value,
    )
    records = {
        f"report-{index}": _record(
            f"report-{index}",
            (states_for_first_target[index],) + (LabelState.NOT_MENTIONED.value,) * (len(TARGET_COLUMNS) - 1),
        )
        for index in range(6)
    }
    return (
        [
            {"fold": 0, "partitions": {"training_labels": [records[f"report-{i}"] for i in range(3)]}},
            {"fold": 1, "partitions": {"training_labels": [records[f"report-{i}"] for i in range(6)]}},
            {"fold": 2, "partitions": {"training_labels": [records[f"report-{i}"] for i in range(3, 6)]}},
        ],
        set(records),
    )


def test_report_only_aggregate_counts_every_state_once_per_unique_report() -> None:
    partitions, expected_uids = _partitions()

    aggregate = aggregate_report_only_records(partitions, expected_uids)

    assert aggregate.report_count == 6
    assert aggregate.training_record_occurrences == 12
    assert aggregate.fold_training_report_counts == ((0, 3), (1, 6), (2, 3))
    assert aggregate.occurrence_distribution == {2: 6}

    first_target = aggregate.targets[0]
    assert first_target.state_counts == {
        "positive": 1,
        "negative": 1,
        "not-mentioned": 1,
        "uncertain": 1,
        "unknown": 1,
        "extraction-failed": 1,
    }
    assert first_target.supervised_count == 2
    assert first_target.coverage == pytest.approx(2 / 6)

    for target in aggregate.targets:
        assert sum(target.state_counts.values()) == aggregate.report_count
        assert target.supervised_count == (
            target.state_counts["positive"] + target.state_counts["negative"]
        )
        assert target.coverage == pytest.approx(target.supervised_count / aggregate.report_count)


def test_report_only_aggregate_rejects_missing_unique_report_state() -> None:
    partitions, expected_uids = _partitions()
    expected_uids.add("missing-report")

    with pytest.raises(ReportOnlyAggregateError, match="missing report-only records"):
        aggregate_report_only_records(partitions, expected_uids)
