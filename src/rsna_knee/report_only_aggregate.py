"""Aggregate fold-safe report-only labels without changing extraction semantics.

This module is deliberately separate from ``report_extraction``.  It consumes
the private, identifier-bearing fold artifact and emits only aggregate counts.
Unique report-only coverage is computed from one record per study, while fold
training occurrences remain a separate execution statistic.  Label-state keys
come directly from :class:`LabelState`, so hyphenated values such as
``not-mentioned`` cannot be silently lost through a spelling mismatch.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence, cast

from .report_labels import LabelState
from .targets import TARGET_COLUMNS


REPORT_ONLY_AGGREGATE_SCHEMA_VERSION = "rsna-knee-report-only-aggregate-v1"
_STATE_VALUES = tuple(state.value for state in LabelState)


class ReportOnlyAggregateError(ValueError):
    """Raised when a report-only aggregate cannot be reconciled safely."""


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReportOnlyAggregateError(f"{name} must be an object")
    return value


def _non_empty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReportOnlyAggregateError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReportOnlyAggregateError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class ReportOnlyTargetAggregate:
    """State and masked-supervision counts for one unique report-only target."""

    target: str
    report_count: int
    positive_count: int
    negative_count: int
    not_mentioned_count: int
    uncertain_count: int
    unknown_count: int
    extraction_failed_count: int

    def __post_init__(self) -> None:
        if self.target not in TARGET_COLUMNS:
            raise ReportOnlyAggregateError(f"unknown target {self.target!r}")
        counts = (
            self.report_count,
            self.positive_count,
            self.negative_count,
            self.not_mentioned_count,
            self.uncertain_count,
            self.unknown_count,
            self.extraction_failed_count,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise ReportOnlyAggregateError(f"{self.target} counts must be non-negative integers")
        if sum(counts[1:]) != self.report_count:
            raise ReportOnlyAggregateError(f"{self.target} state counts do not add to report_count")

    @property
    def state_counts(self) -> dict[str, int]:
        """Return all states using their canonical serialized enum values."""

        return {
            LabelState.POSITIVE.value: self.positive_count,
            LabelState.NEGATIVE.value: self.negative_count,
            LabelState.NOT_MENTIONED.value: self.not_mentioned_count,
            LabelState.UNCERTAIN.value: self.uncertain_count,
            LabelState.UNKNOWN.value: self.unknown_count,
            LabelState.EXTRACTION_FAILED.value: self.extraction_failed_count,
        }

    @property
    def supervised_count(self) -> int:
        """Return the only states admitted to masked supervision."""

        return self.positive_count + self.negative_count

    @property
    def coverage(self) -> float | None:
        """Return supervised entries divided by the unique report population."""

        return self.supervised_count / self.report_count if self.report_count else None

    def to_dict(self) -> dict[str, object]:
        """Serialize aggregate counts without identifiers or source text."""

        return {
            "target": self.target,
            "report_count": self.report_count,
            "positive_extracted_count": self.positive_count,
            "negative_extracted_count": self.negative_count,
            "not_mentioned_count": self.not_mentioned_count,
            "uncertain_count": self.uncertain_count,
            "unknown_count": self.unknown_count,
            "extraction_failed_count": self.extraction_failed_count,
            "state_counts": self.state_counts,
            "supervised_count": self.supervised_count,
            "coverage": self.coverage,
            "accuracy": None,
        }


@dataclass(frozen=True)
class ReportOnlyAggregate:
    """Unique report-only counts plus separate fold-occurrence statistics."""

    report_count: int
    training_record_occurrences: int
    fold_training_report_counts: tuple[tuple[int, int], ...]
    fold_training_supervised_entry_counts: tuple[tuple[int, int], ...]
    _occurrence_distribution: tuple[tuple[int, int], ...]
    targets: tuple[ReportOnlyTargetAggregate, ...]

    def __post_init__(self) -> None:
        if self.report_count < 0 or self.training_record_occurrences < 0:
            raise ReportOnlyAggregateError("aggregate population counts cannot be negative")
        if tuple(target.target for target in self.targets) != TARGET_COLUMNS:
            raise ReportOnlyAggregateError("aggregate targets must use competition order")
        if any(target.report_count != self.report_count for target in self.targets):
            raise ReportOnlyAggregateError("target report populations disagree")
        if sum(count for _, count in self._occurrence_distribution) != self.report_count:
            raise ReportOnlyAggregateError("occurrence distribution does not cover unique reports")
        if sum(occurrences * count for occurrences, count in self._occurrence_distribution) != (
            self.training_record_occurrences
        ):
            raise ReportOnlyAggregateError("occurrence distribution does not cover fold records")

    @property
    def occurrence_distribution(self) -> dict[int, int]:
        """Return number of unique reports seen in each number of folds."""

        return dict(self._occurrence_distribution)

    def to_dict(self) -> dict[str, object]:
        """Serialize text-free report-only evidence."""

        return {
            "schema_version": REPORT_ONLY_AGGREGATE_SCHEMA_VERSION,
            "corpus": {
                "report_only_study_count": self.report_count,
                "training_partition_record_occurrences": self.training_record_occurrences,
                "occurrence_distribution": {
                    str(key): value for key, value in self.occurrence_distribution.items()
                },
                "accuracy": None,
            },
            "folds": [
                {
                    "fold": fold,
                    "eligible_training_report_count": report_count,
                    "training_supervised_entry_count": supervised_count,
                }
                for (fold, report_count), (_, supervised_count) in zip(
                    self.fold_training_report_counts,
                    self.fold_training_supervised_entry_counts,
                )
            ],
            "targets": [target.to_dict() for target in self.targets],
            "accuracy": None,
        }


def _interpretation_signature(interpretation: Mapping[str, object]) -> tuple[object, ...]:
    """Return state-bearing fields used to compare repeated fold records."""

    return (
        tuple(cast(list[object], interpretation["target_order"])),
        tuple(cast(list[object], interpretation["states"])),
        tuple(cast(list[object], interpretation["values"])),
        tuple(cast(list[object], interpretation["supervision_mask"])),
    )


def _validated_interpretation(record: Mapping[str, object], name: str) -> Mapping[str, object]:
    interpretation = _mapping(record.get("interpretation"), f"{name}.interpretation")
    target_order = interpretation.get("target_order")
    states = interpretation.get("states")
    values = interpretation.get("values")
    masks = interpretation.get("supervision_mask")
    if target_order != list(TARGET_COLUMNS):
        raise ReportOnlyAggregateError(f"{name} target order is not canonical")
    if not all(isinstance(value, list) for value in (states, values, masks)):
        raise ReportOnlyAggregateError(f"{name} interpretation vectors must be arrays")
    states = cast(list[object], states)
    values = cast(list[object], values)
    masks = cast(list[object], masks)
    if not (len(states) == len(values) == len(masks) == len(TARGET_COLUMNS)):
        raise ReportOnlyAggregateError(f"{name} interpretation vectors have the wrong length")
    for index, (state, value, mask) in enumerate(zip(states, values, masks)):
        if state not in _STATE_VALUES:
            raise ReportOnlyAggregateError(f"{name} has an unknown state at target {index}")
        if isinstance(mask, bool) or mask not in (0, 1):
            raise ReportOnlyAggregateError(f"{name} has an invalid supervision mask at target {index}")
        if state == LabelState.POSITIVE.value:
            valid = value == 1 and mask == 1
        elif state == LabelState.NEGATIVE.value:
            valid = value == 0 and mask == 1
        else:
            valid = value is None and mask == 0
        if not valid:
            raise ReportOnlyAggregateError(f"{name} has inconsistent state/value/mask at target {index}")
    return interpretation


def aggregate_report_only_records(
    fold_partitions: Sequence[Mapping[str, object]],
    expected_report_only_uids: set[str],
) -> ReportOnlyAggregate:
    """Aggregate serialized training-label records by unique report-only study.

    ``fold_partitions`` is the private ``fold_partitions`` payload emitted by
    Ticket 15.  The expected UID set is used only for reconciliation and is
    never returned.  Every unique report must have one consistent state vector;
    repeated fold occurrences affect execution counts only.
    """

    if not isinstance(expected_report_only_uids, set):
        raise ReportOnlyAggregateError("expected_report_only_uids must be a set")
    if any(not isinstance(uid, str) or not uid for uid in expected_report_only_uids):
        raise ReportOnlyAggregateError("expected report-only identifiers must be non-empty strings")

    record_by_uid: dict[str, Mapping[str, object]] = {}
    occurrence_counts: Counter[str] = Counter()
    fold_report_counts: list[tuple[int, int]] = []
    fold_supervised_counts: list[tuple[int, int]] = []
    seen_folds: set[int] = set()

    for partition_index, raw_partition in enumerate(fold_partitions):
        partition = _mapping(raw_partition, f"fold_partitions[{partition_index}]")
        fold = partition.get("fold")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
            raise ReportOnlyAggregateError(f"fold_partitions[{partition_index}].fold is invalid")
        if fold in seen_folds:
            raise ReportOnlyAggregateError(f"duplicate fold {fold}")
        seen_folds.add(fold)
        partitions = _mapping(partition.get("partitions"), f"fold {fold}.partitions")
        raw_records = partitions.get("training_labels")
        if not isinstance(raw_records, list):
            raise ReportOnlyAggregateError(f"fold {fold}.training_labels must be an array")
        fold_seen: set[str] = set()
        supervised_entries = 0
        for record_index, raw_record in enumerate(raw_records):
            record = _mapping(raw_record, f"fold {fold}.training_labels[{record_index}]")
            uid = _non_empty_text(record.get("study_instance_uid"), "training record study_instance_uid")
            if uid in fold_seen:
                raise ReportOnlyAggregateError(f"fold {fold} contains duplicate training report {uid}")
            fold_seen.add(uid)
            if record.get("artifact_kind") not in (None, "admitted_training_labels"):
                raise ReportOnlyAggregateError(f"fold {fold} contains a non-training artifact")
            interpretation = _validated_interpretation(record, f"fold {fold} record {record_index}")
            if uid in record_by_uid:
                previous = _mapping(record_by_uid[uid].get("interpretation"), "previous interpretation")
                if _interpretation_signature(previous) != _interpretation_signature(interpretation):
                    raise ReportOnlyAggregateError(f"report {uid} changed interpretation across folds")
            else:
                record_by_uid[uid] = record
            occurrence_counts[uid] += 1
            masks = cast(list[object], interpretation["supervision_mask"])
            supervised_entries += sum(cast(int, mask) for mask in masks)
        fold_report_counts.append((fold, len(raw_records)))
        fold_supervised_counts.append((fold, supervised_entries))

    actual_uids = set(record_by_uid)
    missing = expected_report_only_uids - actual_uids
    extra = actual_uids - expected_report_only_uids
    if missing:
        raise ReportOnlyAggregateError(f"missing report-only records for {len(missing)} studies")
    if extra:
        raise ReportOnlyAggregateError(f"unexpected non-report-only records for {len(extra)} studies")

    counts: list[Counter[object]] = [Counter() for _ in TARGET_COLUMNS]
    for record in record_by_uid.values():
        interpretation = _mapping(record.get("interpretation"), "report interpretation")
        states = cast(list[object], interpretation["states"])
        for index, state in enumerate(states):
            counts[index][state] += 1

    targets = tuple(
        ReportOnlyTargetAggregate(
            target=target,
            report_count=len(actual_uids),
            positive_count=counts[index][LabelState.POSITIVE.value],
            negative_count=counts[index][LabelState.NEGATIVE.value],
            not_mentioned_count=counts[index][LabelState.NOT_MENTIONED.value],
            uncertain_count=counts[index][LabelState.UNCERTAIN.value],
            unknown_count=counts[index][LabelState.UNKNOWN.value],
            extraction_failed_count=counts[index][LabelState.EXTRACTION_FAILED.value],
        )
        for index, target in enumerate(TARGET_COLUMNS)
    )
    return ReportOnlyAggregate(
        report_count=len(actual_uids),
        training_record_occurrences=sum(occurrence_counts.values()),
        fold_training_report_counts=tuple(sorted(fold_report_counts)),
        fold_training_supervised_entry_counts=tuple(sorted(fold_supervised_counts)),
        _occurrence_distribution=tuple(sorted(Counter(occurrence_counts.values()).items())),
        targets=targets,
    )


__all__ = [
    "REPORT_ONLY_AGGREGATE_SCHEMA_VERSION",
    "ReportOnlyAggregate",
    "ReportOnlyAggregateError",
    "ReportOnlyTargetAggregate",
    "aggregate_report_only_records",
]
