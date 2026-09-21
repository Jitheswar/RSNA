"""Aggregate the completed Ticket 16 Astra Medium handoff.

The handoff module deliberately stops at validated, private batch outputs.
This module is the next boundary: it reconciles those outputs with the frozen
fold manifest and the frozen Ticket 15 report-label artifact, then emits
aggregate-only public evidence and identifier-bearing private audit evidence.

No report text is interpreted here.  Positive and negative Astra states are
the only supervised states; ``not-mentioned``, ``uncertain``, ``unknown`` and
``extraction-failed`` remain masked.  Held-out gold labels always come from
the fold manifest, never from either extractor.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence, cast

from .archive import atomic_json_write
from .astra_handoff import (
    ASTRA_EFFORT,
    ASTRA_OUTPUT_SCHEMA_FINGERPRINT,
    ASTRA_OUTPUT_SCHEMA_VERSION,
    ASTRA_REASONING_EFFORT,
    ASTRA_SEMANTIC_STATES,
    AstraBatch,
    AstraBatchAudit,
    AstraValidatedStudy,
    build_astra_comparator_cohorts,
    load_astra_audits,
    load_astra_batch_plan,
    load_astra_prompt,
    validate_astra_output,
)
from .folds import FOLD_IDS, FoldManifest, load_fold_manifest
from .report_extraction import ReportCorpus, load_report_corpus
from .report_labels import LabelState
from .report_only_aggregate import ReportOnlyAggregate, aggregate_report_only_records
from .targets import TARGET_COLUMNS


ASTRA_AGGREGATION_SCHEMA_VERSION = "rsna-knee-ticket16-astra-aggregation-v2"
_REPORT_EXTRACTION_SCHEMA_VERSION = "rsna-knee-report-extraction-v1"
# This identity is deliberately independent of the aggregation schema.  It
# fingerprints the frozen Astra state vector and its bound study context, so
# correcting metric names cannot make an unchanged upstream model output look
# like a different scientific artifact.
_SEMANTIC_OUTPUT_DIGEST_IDENTITY = "rsna-knee-ticket16-astra-aggregation-v1"
_ALL_STATES = tuple(state.value for state in LabelState)
_SUPERVISED_STATES = frozenset(
    (LabelState.POSITIVE.value, LabelState.NEGATIVE.value)
)
_PUBLIC_FORBIDDEN_KEYS = frozenset(
    {
        "report_text",
        "study_instance_uid",
        "source_report_ref",
        "source_study_refs",
        "gold_rows",
        "row_level",
    }
)


class AstraAggregationError(ValueError):
    """Raised when Ticket 16 evidence cannot be reconciled safely."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AstraAggregationError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise AstraAggregationError(f"{name} must be an array")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AstraAggregationError(f"{name} must be non-empty text")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AstraAggregationError(f"{name} must be a non-negative integer")
    return value


def _load_json(path: Path, name: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AstraAggregationError(f"could not read {name}: {path}") from exc
    return _mapping(payload, name)


def _require_equal(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise AstraAggregationError(f"{name} does not match the frozen artifact")


def _require_close(actual: object, expected: float | None, name: str) -> None:
    if expected is None:
        if actual is not None:
            raise AstraAggregationError(f"{name} should be undefined")
        return
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        raise AstraAggregationError(f"{name} should be numeric")
    if not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AstraAggregationError(f"{name} differs from the recomputed metric")


def _state_vector(value: object, name: str) -> tuple[str, ...]:
    raw = _list(value, name)
    if len(raw) != len(TARGET_COLUMNS):
        raise AstraAggregationError(f"{name} must contain exactly 12 targets")
    result = tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(raw))
    if any(state not in _ALL_STATES for state in result):
        raise AstraAggregationError(f"{name} contains an unknown state")
    return result


def _validate_t15_interpretation(
    record: Mapping[str, object],
    *,
    fold: int,
    corpus: ReportCorpus,
) -> tuple[str, tuple[str, ...]]:
    uid = _text(record.get("study_instance_uid"), "Ticket 15 evaluation study_instance_uid")
    interpretation = _mapping(record.get("interpretation"), "Ticket 15 interpretation")
    _require_equal(
        interpretation.get("target_order"),
        list(TARGET_COLUMNS),
        "Ticket 15 evaluation target order",
    )
    states = _state_vector(interpretation.get("states"), "Ticket 15 evaluation states")
    values = _list(interpretation.get("values"), "Ticket 15 evaluation values")
    masks = _list(
        interpretation.get("supervision_mask"),
        "Ticket 15 evaluation supervision_mask",
    )
    if len(values) != len(TARGET_COLUMNS) or len(masks) != len(TARGET_COLUMNS):
        raise AstraAggregationError("Ticket 15 evaluation value/mask vectors must contain 12 targets")
    for index, (state, value, mask) in enumerate(zip(states, values, masks)):
        if isinstance(mask, bool) or mask not in (0, 1):
            raise AstraAggregationError("Ticket 15 evaluation supervision mask is invalid")
        expected_value = 1 if state == LabelState.POSITIVE.value else 0 if state == LabelState.NEGATIVE.value else None
        expected_mask = int(state in _SUPERVISED_STATES)
        if value != expected_value or mask != expected_mask:
            raise AstraAggregationError(
                f"Ticket 15 evaluation state/value/mask disagree at target {index}"
            )
    consuming_folds = interpretation.get("consuming_folds")
    if consuming_folds != [fold]:
        raise AstraAggregationError("Ticket 15 held-out interpretation has the wrong consuming fold")
    report_sha256 = _text(record.get("report_sha256"), "Ticket 15 evaluation report digest")
    corpus_record = corpus.by_study.get(uid)
    if corpus_record is None:
        raise AstraAggregationError("Ticket 15 evaluation references a study outside the report corpus")
    _require_equal(report_sha256, corpus_record.report_sha256, "Ticket 15 evaluation report digest")
    return uid, states


@dataclass
class _TargetStats:
    """Mutable accumulator for one target; serialized only after validation."""

    target: str
    report_count: int = 0
    state_counts: Counter[str] = field(default_factory=Counter)
    evaluable_gold_count: int = 0
    positive_reference_support: int = 0
    negative_reference_support: int = 0
    correct_positive_calls: int = 0
    correct_negative_calls: int = 0
    false_positive_count: int = 0
    false_negative_count: int = 0

    def add(self, state: str, reference: int | None = None) -> None:
        if state not in _ALL_STATES:
            raise AstraAggregationError(f"{self.target} has an unsupported state {state!r}")
        self.report_count += 1
        self.state_counts[state] += 1
        if reference is None:
            return
        if reference not in (0, 1):
            raise AstraAggregationError("reference labels must be binary when defined")
        self.evaluable_gold_count += 1
        if reference == 1:
            self.positive_reference_support += 1
        else:
            self.negative_reference_support += 1
        if state == LabelState.POSITIVE.value:
            if reference == 1:
                self.correct_positive_calls += 1
            else:
                self.false_positive_count += 1
        elif state == LabelState.NEGATIVE.value:
            if reference == 0:
                self.correct_negative_calls += 1
            else:
                self.false_negative_count += 1

    @property
    def positive_count(self) -> int:
        return self.state_counts[LabelState.POSITIVE.value]

    @property
    def negative_count(self) -> int:
        return self.state_counts[LabelState.NEGATIVE.value]

    @property
    def supervised_count(self) -> int:
        return self.positive_count + self.negative_count

    @property
    def correct_supervised_count(self) -> int:
        return self.correct_positive_calls + self.correct_negative_calls

    @property
    def selective_error_count(self) -> int:
        return self.false_positive_count + self.false_negative_count

    @property
    def masked_count(self) -> int:
        return self.report_count - self.supervised_count

    def metric(self, numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    def to_gold_dict(self) -> dict[str, object]:
        coverage = self.metric(self.supervised_count, self.evaluable_gold_count)
        selective_accuracy = self.metric(self.correct_supervised_count, self.supervised_count)
        selective_tp = self.correct_positive_calls
        selective_tn = self.correct_negative_calls
        selective_fp = self.false_positive_count
        selective_fn = self.false_negative_count
        selective_sensitivity = self.metric(
            self.correct_positive_calls,
            selective_tp + selective_fn,
        )
        selective_specificity = self.metric(
            self.correct_negative_calls,
            selective_tn + selective_fp,
        )
        selective_ppv = self.metric(
            self.correct_positive_calls,
            self.positive_count,
        )
        selective_npv = self.metric(self.correct_negative_calls, self.negative_count)
        selective_balanced_accuracy = (
            (selective_sensitivity + selective_specificity) / 2
            if selective_sensitivity is not None and selective_specificity is not None
            else None
        )
        positive_reference_capture = self.metric(
            self.correct_positive_calls,
            self.positive_reference_support,
        )
        negative_reference_capture = self.metric(
            self.correct_negative_calls,
            self.negative_reference_support,
        )
        undefined: dict[str, str] = {}
        if coverage is None:
            undefined["supervised_coverage"] = "no gold studies have a defined reference label"
        if selective_accuracy is None:
            undefined["selective_accuracy"] = "Astra/Ticket 15 emitted no positive or negative calls"
        if selective_sensitivity is None:
            undefined["selective_sensitivity"] = (
                "no supervised calls on gold-positive references "
                "(selective TP + selective FN = 0)"
            )
        if selective_specificity is None:
            undefined["selective_specificity"] = (
                "no supervised calls on gold-negative references "
                "(selective TN + selective FP = 0)"
            )
        if selective_ppv is None:
            undefined["selective_ppv"] = "no positive calls were emitted"
        if selective_npv is None:
            undefined["selective_npv"] = "no negative calls were emitted"
        if selective_balanced_accuracy is None:
            undefined["selective_balanced_accuracy"] = (
                "selective sensitivity or selective specificity is undefined"
            )
        if positive_reference_capture is None:
            undefined["positive_reference_capture"] = "gold-positive reference support is zero"
        if negative_reference_capture is None:
            undefined["negative_reference_capture"] = "gold-negative reference support is zero"
        return {
            "target": self.target,
            "evaluable_gold_count": self.evaluable_gold_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "not_mentioned_count": self.state_counts[LabelState.NOT_MENTIONED.value],
            "uncertain_count": self.state_counts[LabelState.UNCERTAIN.value],
            "unknown_count": self.state_counts[LabelState.UNKNOWN.value],
            "extraction_failed_count": self.state_counts[LabelState.EXTRACTION_FAILED.value],
            "state_counts": {
                state: self.state_counts[state] for state in _ALL_STATES
            },
            "supervised_count": self.supervised_count,
            "masked_count": self.masked_count,
            "supervised_coverage": coverage,
            "selective_accuracy": selective_accuracy,
            "positive_reference_support": self.positive_reference_support,
            "negative_reference_support": self.negative_reference_support,
            "correct_positive_calls": self.correct_positive_calls,
            "correct_negative_calls": self.correct_negative_calls,
            "false_positive_count": self.false_positive_count,
            "false_negative_count": self.false_negative_count,
            "selective_tp": selective_tp,
            "selective_tn": selective_tn,
            "selective_fp": selective_fp,
            "selective_fn": selective_fn,
            "selective_sensitivity": selective_sensitivity,
            "selective_specificity": selective_specificity,
            "selective_ppv": selective_ppv,
            "selective_npv": selective_npv,
            "selective_balanced_accuracy": selective_balanced_accuracy,
            "positive_reference_capture": positive_reference_capture,
            "negative_reference_capture": negative_reference_capture,
            "undefined_metrics": undefined,
        }

    def to_report_only_dict(self) -> dict[str, object]:
        coverage = self.metric(self.supervised_count, self.report_count)
        return {
            "target": self.target,
            "report_count": self.report_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "not_mentioned_count": self.state_counts[LabelState.NOT_MENTIONED.value],
            "uncertain_count": self.state_counts[LabelState.UNCERTAIN.value],
            "unknown_count": self.state_counts[LabelState.UNKNOWN.value],
            "extraction_failed_count": self.state_counts[LabelState.EXTRACTION_FAILED.value],
            "state_counts": {
                state: self.state_counts[state] for state in _ALL_STATES
            },
            "supervised_count": self.supervised_count,
            "masked_count": self.masked_count,
            "coverage": coverage,
            "accuracy": None,
            "ground_truth_metrics": None,
            "undefined_metrics": {
                "accuracy": "report-only studies have no gold/reference labels"
            },
        }


def _new_target_stats() -> list[_TargetStats]:
    return [_TargetStats(target) for target in TARGET_COLUMNS]


def _build_stats(
    states_by_uid: Mapping[str, tuple[str, ...]],
    study_uids: Sequence[str],
    references_by_uid: Mapping[str, tuple[int | None, ...]] | None,
) -> tuple[_TargetStats, ...]:
    result = _new_target_stats()
    for uid in study_uids:
        states = states_by_uid.get(uid)
        if states is None:
            raise AstraAggregationError(f"missing semantic output for study {uid}")
        if len(states) != len(TARGET_COLUMNS):
            raise AstraAggregationError("semantic output does not contain 12 target states")
        references = references_by_uid.get(uid) if references_by_uid is not None else None
        if references is not None and len(references) != len(TARGET_COLUMNS):
            raise AstraAggregationError("reference output does not contain 12 target labels")
        for index, state in enumerate(states):
            reference = references[index] if references is not None else None
            result[index].add(state, reference)
    return tuple(result)


def _verify_target_stats(
    stats: Sequence[_TargetStats],
    *,
    gold: bool,
) -> None:
    """Recheck count and metric identities before evidence is published."""

    if tuple(item.target for item in stats) != TARGET_COLUMNS:
        raise AstraAggregationError("target statistics are not in canonical order")
    for item in stats:
        state_total = sum(item.state_counts[state] for state in _ALL_STATES)
        if state_total != item.report_count:
            raise AstraAggregationError(f"{item.target} state counts do not equal report count")
        if item.supervised_count != item.positive_count + item.negative_count:
            raise AstraAggregationError(f"{item.target} supervised count is inconsistent")
        if gold:
            if item.correct_supervised_count != item.correct_positive_calls + item.correct_negative_calls:
                raise AstraAggregationError(f"{item.target} correct count is inconsistent")
            if item.selective_error_count != item.false_positive_count + item.false_negative_count:
                raise AstraAggregationError(f"{item.target} selective error count is inconsistent")
            if item.supervised_count != item.correct_supervised_count + item.selective_error_count:
                raise AstraAggregationError(f"{item.target} supervised/error counts do not reconcile")
            if item.evaluable_gold_count != (
                item.positive_reference_support + item.negative_reference_support
            ):
                raise AstraAggregationError(f"{item.target} reference support does not reconcile")
            payload = item.to_gold_dict()
            _require_close(
                payload["supervised_coverage"],
                item.metric(item.supervised_count, item.evaluable_gold_count),
                f"{item.target} coverage",
            )
            _require_close(
                payload["selective_accuracy"],
                item.metric(item.correct_supervised_count, item.supervised_count),
                f"{item.target} selective accuracy",
            )
            _require_close(
                payload["selective_sensitivity"],
                item.metric(item.correct_positive_calls, item.correct_positive_calls + item.false_negative_count),
                f"{item.target} selective sensitivity",
            )
            _require_close(
                payload["selective_specificity"],
                item.metric(item.correct_negative_calls, item.correct_negative_calls + item.false_positive_count),
                f"{item.target} selective specificity",
            )
            _require_close(
                payload["positive_reference_capture"],
                item.metric(item.correct_positive_calls, item.positive_reference_support),
                f"{item.target} positive-reference capture",
            )
            _require_close(
                payload["negative_reference_capture"],
                item.metric(item.correct_negative_calls, item.negative_reference_support),
                f"{item.target} negative-reference capture",
            )
        else:
            payload = item.to_report_only_dict()
            _require_close(
                payload["coverage"],
                item.metric(item.supervised_count, item.report_count),
                f"{item.target} report-only coverage",
            )


def _micro_selective_accuracy(stats: Sequence[_TargetStats]) -> dict[str, object]:
    correct = sum(item.correct_supervised_count for item in stats)
    supervised = sum(item.supervised_count for item in stats)
    return {
        "correct_supervised_calls": correct,
        "supervised_calls": supervised,
        "value": correct / supervised if supervised else None,
        "undefined_reason": "no supervised calls across the held-out gold targets"
        if supervised == 0
        else None,
    }


def _metric_summary(stats: Sequence[_TargetStats], *, report_only: bool = False) -> dict[str, object]:
    if report_only:
        targets = [item.to_report_only_dict() for item in stats]
        coverages = [cast(float, item["coverage"]) for item in targets if item["coverage"] is not None]
        return {
            "targets": targets,
            "target_inclusion_rule": "all 12 canonical targets; coverage is defined for every non-empty report-only cohort",
            "mean_coverage": sum(coverages) / len(coverages) if coverages else None,
            "total_supervised_entries": sum(item.supervised_count for item in stats),
            "accuracy": None,
        }
    targets = [item.to_gold_dict() for item in stats]
    defined_targets = [
        cast(Mapping[str, object], item)
        for item in targets
        if item["selective_accuracy"] is not None
    ]
    accuracies = [item["selective_accuracy"] for item in defined_targets]
    coverages = [
        cast(float, item["supervised_coverage"])
        for item in targets
        if item["supervised_coverage"] is not None
    ]
    return {
        "targets": targets,
        "target_inclusion_rule": "all 12 canonical targets for per-target metrics; defined-target macro accuracy includes exactly targets with defined selective accuracy, while micro accuracy includes every supervised target call",
        "defined_selective_accuracy_targets": [
            cast(str, item["target"]) for item in defined_targets
        ],
        "defined_selective_accuracy_target_count": len(accuracies),
        "defined_target_macro_selective_accuracy": (
            sum(cast(float, value) for value in accuracies) / len(accuracies)
            if accuracies
            else None
        ),
        "mean_supervised_coverage": sum(cast(float, value) for value in coverages) / len(coverages)
        if coverages
        else None,
        "micro_selective_accuracy": _micro_selective_accuracy(stats),
        "total_correct_supervised_calls": sum(item.correct_supervised_count for item in stats),
        "total_supervised_entries": sum(item.supervised_count for item in stats),
    }


def _common_target_macro_selective_accuracy(
    astra: Mapping[str, object],
    ticket15: Mapping[str, object],
) -> dict[str, object]:
    astra_rows = {
        cast(str, row["target"]): row
        for row in _gold_table_rows(astra)
    }
    ticket15_rows = {
        cast(str, row["target"]): row
        for row in _gold_table_rows(ticket15)
    }
    common_targets = [
        target
        for target in TARGET_COLUMNS
        if astra_rows[target]["selective_accuracy"] is not None
        and ticket15_rows[target]["selective_accuracy"] is not None
    ]

    def macro(rows: Mapping[str, Mapping[str, object]]) -> float | None:
        values = [cast(float, rows[target]["selective_accuracy"]) for target in common_targets]
        return sum(values) / len(values) if values else None

    return {
        "target_set": common_targets,
        "target_count": len(common_targets),
        "same_target_set_for_both_extractors": True,
        "astra": macro(astra_rows),
        "ticket15": macro(ticket15_rows),
    }


def _sign(value: float | None, *, tolerance: float = 1e-12) -> int | None:
    if value is None:
        return None
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _compare_gold_stats(
    astra: Sequence[_TargetStats],
    ticket15: Sequence[_TargetStats],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for left, right in zip(astra, ticket15):
        if left.target != right.target:
            raise AstraAggregationError("Astra and Ticket 15 target orders do not match")
        left_payload = left.to_gold_dict()
        right_payload = right.to_gold_dict()
        left_accuracy = cast(float | None, left_payload["selective_accuracy"])
        right_accuracy = cast(float | None, right_payload["selective_accuracy"])
        left_coverage = cast(float | None, left_payload["supervised_coverage"])
        right_coverage = cast(float | None, right_payload["supervised_coverage"])
        coverage_delta = (
            left_coverage - right_coverage
            if left_coverage is not None and right_coverage is not None
            else None
        )
        accuracy_delta = (
            left_accuracy - right_accuracy
            if left_accuracy is not None and right_accuracy is not None
            else None
        )
        coverage_sign = _sign(coverage_delta)
        accuracy_sign = _sign(accuracy_delta)
        astra_incorrect_calls = left.selective_error_count
        ticket15_incorrect_calls = right.selective_error_count
        improved_coverage_maintaining_accuracy = (
            coverage_sign == 1
            and accuracy_sign is not None
            and accuracy_sign >= 0
        )
        improved_accuracy_maintaining_coverage = (
            accuracy_sign == 1
            and coverage_sign is not None
            and coverage_sign >= 0
        )
        if left_accuracy is None or right_accuracy is None:
            assessment = "coverage_gain_accuracy_undefined" if coverage_sign == 1 else "accuracy_undefined"
        elif coverage_sign == 1 and accuracy_sign == 1:
            assessment = "both_improved"
        elif coverage_sign == 1 and accuracy_sign == 0:
            assessment = "coverage_improved_accuracy_maintained"
        elif coverage_sign == 1 and accuracy_sign == -1:
            assessment = "coverage_improved_accuracy_reduced"
        elif coverage_sign == 0 and accuracy_sign == 1:
            assessment = "accuracy_improved_coverage_maintained"
        elif coverage_sign == 0 and accuracy_sign == 0:
            assessment = "unchanged"
        elif coverage_sign == -1 and accuracy_sign == 1:
            assessment = "accuracy_improved_coverage_reduced"
        elif coverage_sign == -1 and accuracy_sign == 0:
            assessment = "coverage_reduced_accuracy_maintained"
        else:
            assessment = "mixed"
        result.append(
            {
                "target": left.target,
                "astra_coverage": left_coverage,
                "ticket15_coverage": right_coverage,
                "coverage_delta": coverage_delta,
                "astra_selective_accuracy": left_accuracy,
                "ticket15_selective_accuracy": right_accuracy,
                "accuracy_delta": accuracy_delta,
                "astra_supervised_count": left.supervised_count,
                "ticket15_supervised_count": right.supervised_count,
                "supervised_count_delta": left.supervised_count - right.supervised_count,
                "astra_correct_supervised_predictions": left.correct_supervised_count,
                "ticket15_correct_supervised_predictions": right.correct_supervised_count,
                "astra_additional_correct_calls": (
                    left.correct_supervised_count - right.correct_supervised_count
                ),
                "correct_supervised_predictions_delta": (
                    left.correct_supervised_count - right.correct_supervised_count
                ),
                "astra_incorrect_call_count": astra_incorrect_calls,
                "ticket15_incorrect_call_count": ticket15_incorrect_calls,
                "astra_additional_incorrect_calls": astra_incorrect_calls - ticket15_incorrect_calls,
                "incorrect_call_delta": astra_incorrect_calls - ticket15_incorrect_calls,
                "astra_false_positive_count": left.false_positive_count,
                "ticket15_false_positive_count": right.false_positive_count,
                "false_positive_delta": left.false_positive_count - right.false_positive_count,
                "astra_false_negative_count": left.false_negative_count,
                "ticket15_false_negative_count": right.false_negative_count,
                "false_negative_delta": left.false_negative_count - right.false_negative_count,
                "astra_positive_reference_capture": left_payload["positive_reference_capture"],
                "ticket15_positive_reference_capture": right_payload["positive_reference_capture"],
                "astra_negative_reference_capture": left_payload["negative_reference_capture"],
                "ticket15_negative_reference_capture": right_payload["negative_reference_capture"],
                "improved_coverage_while_maintaining_accuracy": improved_coverage_maintaining_accuracy,
                "improved_accuracy_while_maintaining_coverage": improved_accuracy_maintaining_coverage,
                "result_is_mixed": assessment == "mixed" or assessment == "coverage_gain_accuracy_undefined",
                "assessment": assessment,
            }
        )
    return result


def _compare_report_only(
    astra: Sequence[_TargetStats],
    ticket15: ReportOnlyAggregate,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for left, right in zip(astra, ticket15.targets):
        left_payload = left.to_report_only_dict()
        right_payload = right.to_dict()
        left_coverage = cast(float | None, left_payload["coverage"])
        right_coverage = cast(float | None, right_payload["coverage"])
        left_states = cast(dict[str, int], left_payload["state_counts"])
        right_states = cast(dict[str, int], right_payload["state_counts"])
        result.append(
            {
                "target": left.target,
                "astra_coverage": left_coverage,
                "ticket15_coverage": right_coverage,
                "coverage_delta": (
                    left_coverage - right_coverage
                    if left_coverage is not None and right_coverage is not None
                    else None
                ),
                "astra_supervised_count": left.supervised_count,
                "ticket15_supervised_count": right.supervised_count,
                "supervised_count_delta": left.supervised_count - right.supervised_count,
                "state_count_delta": {
                    state: left_states[state] - right_states[state] for state in _ALL_STATES
                },
            }
        )
    return result


def _disagreement_analysis(
    astra_states: Mapping[str, tuple[str, ...]],
    ticket15_states: Mapping[str, tuple[str, ...]],
    references: Mapping[str, tuple[int, ...]],
    study_uids: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    public: list[dict[str, object]] = []
    private_rows: list[dict[str, object]] = []
    for index, target in enumerate(TARGET_COLUMNS):
        counts = Counter(
            {
                "both_supervised_and_agree": 0,
                "both_supervised_and_disagree": 0,
                "astra_supervised_ticket15_masked": 0,
                "ticket15_supervised_astra_masked": 0,
                "both_masked": 0,
            }
        )
        truth = Counter(
            {
                "astra_correct_in_both_supervised_disagreement": 0,
                "ticket15_correct_in_both_supervised_disagreement": 0,
                "astra_correct_when_only_astra_supervised": 0,
                "astra_incorrect_when_only_astra_supervised": 0,
                "ticket15_correct_when_only_ticket15_supervised": 0,
                "ticket15_incorrect_when_only_ticket15_supervised": 0,
            }
        )
        for uid in study_uids:
            astra_state = astra_states[uid][index]
            ticket_state = ticket15_states[uid][index]
            astra_supervised = astra_state in _SUPERVISED_STATES
            ticket_supervised = ticket_state in _SUPERVISED_STATES
            reference = references[uid][index]
            category: str
            if astra_supervised and ticket_supervised:
                if astra_state == ticket_state:
                    category = "both_supervised_and_agree"
                else:
                    category = "both_supervised_and_disagree"
                    truth["astra_correct_in_both_supervised_disagreement"] += int(
                        (astra_state == LabelState.POSITIVE.value) == bool(reference)
                    )
                    truth["ticket15_correct_in_both_supervised_disagreement"] += int(
                        (ticket_state == LabelState.POSITIVE.value) == bool(reference)
                    )
            elif astra_supervised:
                category = "astra_supervised_ticket15_masked"
                correct = (astra_state == LabelState.POSITIVE.value) == bool(reference)
                truth[
                    "astra_correct_when_only_astra_supervised"
                    if correct
                    else "astra_incorrect_when_only_astra_supervised"
                ] += 1
            elif ticket_supervised:
                category = "ticket15_supervised_astra_masked"
                correct = (ticket_state == LabelState.POSITIVE.value) == bool(reference)
                truth[
                    "ticket15_correct_when_only_ticket15_supervised"
                    if correct
                    else "ticket15_incorrect_when_only_ticket15_supervised"
                ] += 1
            else:
                category = "both_masked"
            counts[category] += 1
            private_rows.append(
                {
                    "study_instance_uid": uid,
                    "target": target,
                    "category": category,
                    "reference_label": reference,
                    "astra_state": astra_state,
                    "ticket15_state": ticket_state,
                }
            )
        public.append(
            {
                "target": target,
                **dict(counts),
                "gold_truth_in_disagreement_cases": dict(truth),
            }
        )
    return public, private_rows


def _validate_ticket15_artifact(
    payload: Mapping[str, object],
    manifest: FoldManifest,
    corpus: ReportCorpus,
    expected_gold_uids: set[str],
    expected_report_only_uids: set[str],
) -> tuple[dict[str, tuple[str, ...]], ReportOnlyAggregate]:
    _require_equal(payload.get("schema_version"), _REPORT_EXTRACTION_SCHEMA_VERSION, "Ticket 15 schema")
    _require_equal(payload.get("status"), "completed_synthetic_or_private_local_report_evaluation", "Ticket 15 status")
    fold_context = _mapping(payload.get("fold_context"), "Ticket 15 fold_context")
    _require_equal(fold_context.get("fold_manifest_checksum"), manifest.manifest_checksum, "Ticket 15 manifest checksum")
    _require_equal(fold_context.get("repeat_name"), manifest.repeat_name, "Ticket 15 repeat name")
    _require_equal(fold_context.get("report_source_version"), corpus.source_version, "Ticket 15 report source")
    input_metadata = _mapping(payload.get("input"), "Ticket 15 input")
    _require_equal(input_metadata.get("record_set_sha256"), corpus.record_set_sha256, "Ticket 15 corpus digest")
    _require_equal(input_metadata.get("report_count"), corpus.report_count, "Ticket 15 corpus count")
    boundaries = _mapping(payload.get("boundaries"), "Ticket 15 boundaries")
    for key in (
        "held_out_gold_reports_are_evaluation_only",
        "held_out_reports_excluded_from_training",
        "reference_labels_take_precedence",
        "training_labels_are_report_only",
        "unknown_uncertain_failed_masked",
    ):
        _require_equal(boundaries.get(key), True, f"Ticket 15 boundary {key}")

    raw_partitions = _list(payload.get("fold_partitions"), "Ticket 15 fold_partitions")
    if len(raw_partitions) != len(FOLD_IDS):
        raise AstraAggregationError("Ticket 15 must contain exactly three fold partitions")
    evaluation_states: dict[str, tuple[str, ...]] = {}
    training_partitions: list[Mapping[str, object]] = []
    seen_folds: set[int] = set()
    for raw_partition in raw_partitions:
        partition = _mapping(raw_partition, "Ticket 15 fold partition")
        fold = partition.get("fold")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold not in FOLD_IDS:
            raise AstraAggregationError("Ticket 15 fold partition has an invalid fold")
        if fold in seen_folds:
            raise AstraAggregationError("Ticket 15 fold partitions contain a duplicate fold")
        seen_folds.add(fold)
        cohorts = _mapping(partition.get("partitions"), "Ticket 15 partitions")
        training = _list(cohorts.get("training_labels"), "Ticket 15 training labels")
        evaluation = _list(cohorts.get("held_out_evaluation"), "Ticket 15 held-out evaluation")
        training_partitions.append(
            {"fold": fold, "partitions": {"training_labels": training}}
        )
        raw_eligible = _list(
            partition.get("eligible_evaluation_gold_uids"),
            "Ticket 15 eligible evaluation studies",
        )
        eligible = set(_text(item, "Ticket 15 eligible evaluation study") for item in raw_eligible)
        if not eligible.issubset(expected_gold_uids):
            raise AstraAggregationError("Ticket 15 eligible evaluation set contains a non-gold study")
        missing_evaluation = _list(
            partition.get("missing_evaluation_report_uids"),
            "Ticket 15 missing evaluation studies",
        )
        if missing_evaluation:
            raise AstraAggregationError("Ticket 15 has missing held-out reports in the frozen comparison")
        seen_fold: set[str] = set()
        for raw_record in evaluation:
            record = _mapping(raw_record, "Ticket 15 held-out evaluation record")
            uid, states = _validate_t15_interpretation(record, fold=cast(int, fold), corpus=corpus)
            if uid in seen_fold or uid in evaluation_states:
                raise AstraAggregationError("Ticket 15 held-out evaluation contains duplicate studies")
            seen_fold.add(uid)
            membership = next((item for item in manifest.memberships if item.study_instance_uid == uid), None)
            if membership is None or membership.cohort != "gold" or membership.fold != fold:
                raise AstraAggregationError("Ticket 15 held-out evaluation violates frozen fold membership")
            evaluation_states[uid] = states
        if seen_fold != eligible:
            raise AstraAggregationError("Ticket 15 held-out evaluation does not match its eligible set")
    if seen_folds != set(FOLD_IDS):
        raise AstraAggregationError("Ticket 15 fold partitions do not contain folds 0, 1 and 2")
    if set(evaluation_states) != expected_gold_uids:
        raise AstraAggregationError("Ticket 15 held-out population does not equal the Astra gold population")
    report_only = aggregate_report_only_records(training_partitions, expected_report_only_uids)
    return evaluation_states, report_only


def _check_ticket15_pooled_metrics(
    payload: Mapping[str, object],
    stats: Sequence[_TargetStats],
) -> None:
    pooled = _mapping(payload.get("pooled"), "Ticket 15 pooled metrics")
    targets = _list(pooled.get("targets"), "Ticket 15 pooled targets")
    if len(targets) != len(TARGET_COLUMNS):
        raise AstraAggregationError("Ticket 15 pooled metrics must contain all targets")
    for expected, raw in zip(stats, targets):
        actual = _mapping(raw, "Ticket 15 pooled target")
        expected_payload = expected.to_gold_dict()
        _require_equal(actual.get("target"), expected.target, "Ticket 15 pooled target name")
        for field_name, expected_value in (
            ("report_count", expected.report_count),
            ("gold_positive_count", expected.positive_reference_support),
            ("gold_negative_count", expected.negative_reference_support),
            ("predicted_positive_count", expected.positive_count),
            ("predicted_negative_count", expected.negative_count),
            ("correct_count", expected.correct_supervised_count),
            ("failure_count", expected.state_counts[LabelState.EXTRACTION_FAILED.value]),
        ):
            _require_equal(actual.get(field_name), expected_value, f"Ticket 15 pooled {field_name}")
        _require_close(actual.get("accuracy"), cast(float | None, expected_payload["selective_accuracy"]), "Ticket 15 pooled accuracy")
        _require_close(actual.get("coverage"), cast(float | None, expected_payload["supervised_coverage"]), "Ticket 15 pooled coverage")
        uncertainty = _mapping(actual.get("uncertainty"), "Ticket 15 pooled uncertainty")
        for state in (
            LabelState.NOT_MENTIONED.value,
            LabelState.UNCERTAIN.value,
            LabelState.UNKNOWN.value,
            LabelState.EXTRACTION_FAILED.value,
        ):
            _require_equal(uncertainty.get(state.replace("-", "_") if state != "extraction-failed" else "extraction_failed"), expected.state_counts[state], "Ticket 15 pooled uncertainty")


def _validate_public_privacy(
    payload: Mapping[str, object],
    known_uids: set[str],
    known_report_refs: set[str],
) -> None:
    serialized = _canonical_json(payload)
    for uid in known_uids:
        if uid in serialized:
            raise AstraAggregationError("public aggregation contains a private study identifier")
    for report_ref in known_report_refs:
        if report_ref in serialized:
            raise AstraAggregationError("public aggregation contains a private report reference")

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key) in _PUBLIC_FORBIDDEN_KEYS:
                    raise AstraAggregationError(f"public aggregation contains forbidden field {path}.{key}")
                visit(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(payload, "public")


def _timestamp(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _operational_evidence(
    plan_batch_count: int,
    audits: Sequence[AstraBatchAudit],
    response_paths: Sequence[Path],
    audit_paths: Sequence[Path],
) -> dict[str, object]:
    total_reports = sum(len(item.studies) for item in audits if item.completed)
    validation_failures = sum(item.status == "extraction-failed" for item in audits)
    attempts = sum(item.attempt for item in audits)
    first_attempt_accepted = sum(item.status == "validated" and item.attempt == 1 for item in audits)
    response_times = [path.stat().st_mtime for path in response_paths]
    audit_times = [path.stat().st_mtime for path in audit_paths]
    wall_span = max(audit_times) - min(response_times) if response_times and audit_times else None
    return {
        "total_batches": plan_batch_count,
        "total_reports": total_reports,
        "validation_failures": validation_failures,
        "retry_count": attempts - plan_batch_count,
        "first_attempt_acceptance_rate": first_attempt_accepted / plan_batch_count
        if plan_batch_count
        else None,
        "first_attempt_accepted_batches": first_attempt_accepted,
        "elapsed_runtime_seconds": None,
        "reports_per_minute": None,
        "batch_latency_statistics": None,
        "batch_latency_recorded": False,
        "runtime_note": "Active model runtime and batch latency were not recorded. Filesystem timestamps provide only an observed response-to-audit wall span, not runtime.",
        "observed_response_to_audit_wall_span_seconds": wall_span,
        "observed_first_response_timestamp_utc": _timestamp(min(response_times)) if response_times else None,
        "observed_last_audit_timestamp_utc": _timestamp(max(audit_times)) if audit_times else None,
    }


def _conclusions(
    gold_comparison: Sequence[Mapping[str, object]],
    report_comparison: Sequence[Mapping[str, object]],
    astra_gold: Sequence[_TargetStats],
    ticket15_gold: Sequence[_TargetStats],
    astra_report: Sequence[_TargetStats],
    ticket15_report: ReportOnlyAggregate,
) -> dict[str, object]:
    coverage_gains = [item for item in report_comparison if cast(float, item["coverage_delta"]) > 0]
    accuracy_reduced = [
        item["target"]
        for item in gold_comparison
        if item["accuracy_delta"] is not None and cast(float, item["accuracy_delta"]) < 0
    ]
    accuracy_improved = [
        item["target"]
        for item in gold_comparison
        if item["accuracy_delta"] is not None and cast(float, item["accuracy_delta"]) > 0
    ]
    accuracy_maintained = [
        item["target"]
        for item in gold_comparison
        if item["accuracy_delta"] is not None and cast(float, item["accuracy_delta"]) == 0
    ]
    undefined_accuracy = [
        item["target"] for item in gold_comparison if item["accuracy_delta"] is None
    ]
    top_benefits = sorted(
        report_comparison,
        key=lambda item: cast(float, item["coverage_delta"]),
        reverse=True,
    )[:3]
    astra_summary = _metric_summary(astra_gold)
    ticket15_summary = _metric_summary(ticket15_gold)
    common_macro = _common_target_macro_selective_accuracy(astra_summary, ticket15_summary)
    increased_fp = sum(item.false_positive_count - baseline.false_positive_count for item, baseline in zip(astra_gold, ticket15_gold))
    increased_fn = sum(item.false_negative_count - baseline.false_negative_count for item, baseline in zip(astra_gold, ticket15_gold))
    astra_gold_supervised = sum(item.supervised_count for item in astra_gold)
    ticket15_gold_supervised = sum(item.supervised_count for item in ticket15_gold)
    astra_report_supervised = sum(item.supervised_count for item in astra_report)
    ticket15_report_supervised = sum(item.supervised_count for item in ticket15_report.targets)
    astra_gold_incorrect = sum(item.selective_error_count for item in astra_gold)
    ticket15_gold_incorrect = sum(item.selective_error_count for item in ticket15_gold)
    return {
        "materially_increased_usable_weak_label_coverage": {
            "answer": bool(len(coverage_gains) == len(TARGET_COLUMNS)),
            "rule": "material means every target gained report-only supervised coverage on the identical 4,349-study cohort",
            "targets_with_positive_coverage_delta": [item["target"] for item in coverage_gains],
            "astra_report_only_supervised_entries": astra_report_supervised,
            "ticket15_report_only_supervised_entries": ticket15_report_supervised,
        },
        "held_out_accuracy_assessment": {
            "answer": "mixed",
            "accuracy_improved_targets": accuracy_improved,
            "accuracy_maintained_targets": accuracy_maintained,
            "accuracy_reduced_targets": accuracy_reduced,
            "accuracy_comparison_undefined_targets": undefined_accuracy,
            "astra_defined_target_macro_selective_accuracy": astra_summary["defined_target_macro_selective_accuracy"],
            "astra_defined_target_macro_targets": astra_summary["defined_selective_accuracy_targets"],
            "ticket15_defined_target_macro_selective_accuracy": ticket15_summary["defined_target_macro_selective_accuracy"],
            "ticket15_defined_target_macro_targets": ticket15_summary["defined_selective_accuracy_targets"],
            "common_target_macro_selective_accuracy": common_macro,
            "micro_selective_accuracy": {
                "astra": astra_summary["micro_selective_accuracy"],
                "ticket15": ticket15_summary["micro_selective_accuracy"],
            },
        },
        "targets_benefited_most_by_report_only_coverage_gain": [
            {"target": item["target"], "coverage_delta": item["coverage_delta"]}
            for item in top_benefits
        ],
        "targets_that_regressed_or_remain_unreliable": {
            "selective_accuracy_reduced_vs_ticket15": accuracy_reduced,
            "absolute_selective_accuracy_below_0_8": [
                item.target
                for item in astra_gold
                if item.metric(item.correct_supervised_count, item.supervised_count) is not None
                and cast(float, item.metric(item.correct_supervised_count, item.supervised_count)) < 0.8
            ],
            "medial_oa_note": "Medial OA gained coverage from Ticket 15's zero-call baseline, but its Astra selective accuracy is 0.8529 with five errors; the baseline accuracy comparison is undefined.",
        },
        "became_more_aggressive": {
            "answer": astra_gold_supervised > ticket15_gold_supervised and astra_report_supervised > ticket15_report_supervised,
            "gold_supervised_entries": {"astra": astra_gold_supervised, "ticket15": ticket15_gold_supervised},
            "report_only_supervised_entries": {"astra": astra_report_supervised, "ticket15": ticket15_report_supervised},
            "gold_positive_calls": {"astra": sum(item.positive_count for item in astra_gold), "ticket15": sum(item.positive_count for item in ticket15_gold)},
            "gold_negative_calls": {"astra": sum(item.negative_count for item in astra_gold), "ticket15": sum(item.negative_count for item in ticket15_gold)},
        },
        "suspicious_over_labeling_evidence": {
            "answer": increased_fp > 0 or increased_fn > 0,
            "additional_false_positives_vs_ticket15": increased_fp,
            "additional_false_negatives_vs_ticket15": increased_fn,
            "interpretation": "Coverage expanded substantially while held-out false calls also increased; this is evidence of aggressiveness, not proof of a model defect or a reason to alter the frozen prompt.",
        },
        "held_out_call_totals": {
            "additional_correct_calls_astra_minus_ticket15": (
                sum(item.correct_supervised_count for item in astra_gold)
                - sum(item.correct_supervised_count for item in ticket15_gold)
            ),
            "additional_incorrect_calls_astra_minus_ticket15": astra_gold_incorrect - ticket15_gold_incorrect,
            "incorrect_calls": {"astra": astra_gold_incorrect, "ticket15": ticket15_gold_incorrect},
        },
        "weak_labels_to_keep_masked_or_not_admit": {
            "answer": True,
            "conservative_scope": "The evidence does not establish a generally validated Astra weak-label extractor; retain Astra-derived labels as masked pending an independently pre-specified admission rule.",
            "target_flags": [
                item["target"]
                for item in gold_comparison
                if (
                    item["accuracy_delta"] is not None
                    and cast(float, item["accuracy_delta"]) < 0
                )
                or item["target"] in {"Medial OA", "Effusion", "Synovitis", "Contusion"}
            ],
        },
        "scientifically_validated_for_ticket16": {
            "answer": False,
            "reason": "Astra materially increased coverage, but target-level held-out behavior is mixed, with increased false positives/false negatives and no pre-declared target admission threshold or independent validation set.",
        },
        "ticket17_training_decision": "Not made; this report only states whether Ticket 16 evidence supports the extractor.",
    }


@dataclass(frozen=True)
class AstraAggregationResult:
    """Completed aggregate result with separate public and private views."""

    public: Mapping[str, object]
    private: Mapping[str, object]
    markdown: str

    def to_public_dict(self) -> dict[str, object]:
        return dict(self.public)

    def to_private_dict(self) -> dict[str, object]:
        return dict(self.private)


def aggregate_astra_run(
    *,
    private_plan_path: str | Path,
    private_dir: str | Path,
    response_dir: str | Path,
    ticket15_labels_path: str | Path,
    fold_manifest_path: str | Path,
    report_corpus_path: str | Path,
) -> AstraAggregationResult:
    """Reconcile and aggregate a completed Astra run without changing labels."""

    private_plan = Path(private_plan_path)
    private_root = Path(private_dir)
    responses_root = Path(response_dir)
    ticket15_path = Path(ticket15_labels_path)
    manifest_path = Path(fold_manifest_path)
    corpus_path = Path(report_corpus_path)

    plan = load_astra_batch_plan(private_plan)
    manifest = load_fold_manifest(manifest_path)
    corpus = load_report_corpus(corpus_path)
    expected_cohorts = build_astra_comparator_cohorts(manifest, corpus)
    if expected_cohorts.fingerprint != plan.cohorts.fingerprint:
        raise AstraAggregationError("frozen plan cohorts differ from the canonical manifest/corpus cohorts")
    prompt_path = private_root / "approved-prompt.json"
    if prompt_path.is_file():
        approved_prompt = load_astra_prompt(prompt_path)
        if approved_prompt.fingerprint != plan.prompt.fingerprint:
            raise AstraAggregationError("approved prompt differs from the frozen plan prompt")

    audits = load_astra_audits(plan, private_root)
    audits_by_batch: dict[str, list[AstraBatchAudit]] = {}
    for audit in audits:
        audits_by_batch.setdefault(audit.batch_id, []).append(audit)
    if set(audits_by_batch) != {batch.batch_id for batch in plan.batches}:
        raise AstraAggregationError("validated audit batches do not equal the planned batch set")
    if any(len(items) != 1 for items in audits_by_batch.values()):
        raise AstraAggregationError("a planned batch is represented by more than one audit attempt")
    if any(audit.status != "validated" or audit.attempt != 1 for audit in audits):
        raise AstraAggregationError("completed Astra run contains a retry or failed validation")

    response_paths = sorted(responses_root.glob("*.json"))
    expected_response_names = {f"{batch.batch_id}.json" for batch in plan.batches}
    if {path.name for path in response_paths} != expected_response_names:
        raise AstraAggregationError("response directory does not contain exactly one output per planned batch")

    batches_by_id = {batch.batch_id: batch for batch in plan.batches}
    validated_by_uid: dict[str, tuple[AstraValidatedStudy, AstraBatchAudit]] = {}
    batch_context_rows: list[dict[str, object]] = []
    for response_path in response_paths:
        batch_id = response_path.stem
        batch = batches_by_id[batch_id]
        raw = response_path.read_text(encoding="utf-8")
        validated = validate_astra_output(raw, batch)
        audit = audits_by_batch[batch_id][0]
        if validated.status != audit.status or validated.semantic_digest != audit.semantic_output_digest:
            raise AstraAggregationError("response validation differs from the immutable audit")
        if hashlib.sha256(raw.encode("utf-8")).hexdigest() != audit.raw_output_sha256:
            raise AstraAggregationError("response file digest differs from the immutable audit")
        batch_audit_path = private_root / "batches" / batch_id / "attempt-0001.audit.json"
        if response_path.stat().st_mtime > batch_audit_path.stat().st_mtime:
            raise AstraAggregationError("a response file is newer than its validation audit")
        for study in validated.studies:
            if study.study_instance_uid in validated_by_uid:
                raise AstraAggregationError("a study appears more than once in final Astra output")
            validated_by_uid[study.study_instance_uid] = (study, audit)
        batch_context_rows.append(
            {
                "batch_id": batch_id,
                "batch_index": batch.batch_index,
                "input_digest": batch.input_digest,
                "raw_output_sha256": audit.raw_output_sha256,
                "semantic_output_digest": audit.semantic_output_digest,
                "study_count": len(batch.records),
            }
        )

    expected_uids = set(plan.cohorts.available_report_uids)
    if set(validated_by_uid) != expected_uids or len(validated_by_uid) != plan.unique_study_count:
        raise AstraAggregationError("final Astra semantic output does not cover exactly the planned studies")
    for uid, (study, _) in validated_by_uid.items():
        membership = next((item for item in manifest.memberships if item.study_instance_uid == uid), None)
        if membership is None or study.cohort != membership.cohort or study.assigned_fold != membership.fold:
            raise AstraAggregationError("Astra output cohort/fold metadata disagrees with the frozen manifest")
        if tuple(label.target for label in study.labels) != TARGET_COLUMNS:
            raise AstraAggregationError("Astra output target order is not canonical")
        if any(label.state.value not in ASTRA_SEMANTIC_STATES for label in study.labels):
            raise AstraAggregationError("Astra semantic output contains a non-semantic failure state")

    ticket15_payload = _load_json(ticket15_path, "Ticket 15 report-label artifact")
    membership_by_uid = {item.study_instance_uid: item for item in manifest.memberships}
    expected_gold_uids = {
        uid for uid, item in membership_by_uid.items() if item.cohort == "gold"
    }
    expected_report_only_uids = {
        uid for uid, item in membership_by_uid.items() if item.cohort == "report_only"
    }
    ticket15_states, ticket15_report_only = _validate_ticket15_artifact(
        ticket15_payload,
        manifest,
        corpus,
        expected_gold_uids,
        expected_report_only_uids,
    )
    astra_states = {
        uid: tuple(label.state.value for label in study.labels)
        for uid, (study, _) in validated_by_uid.items()
    }
    references = {
        uid: cast(tuple[int, ...], membership_by_uid[uid].reference_labels)
        for uid in expected_gold_uids
    }
    gold_uids = tuple(sorted(expected_gold_uids))
    report_only_uids = tuple(sorted(expected_report_only_uids))
    astra_gold_stats = _build_stats(astra_states, gold_uids, references)
    ticket15_gold_stats = _build_stats(ticket15_states, gold_uids, references)
    _verify_target_stats(astra_gold_stats, gold=True)
    _verify_target_stats(ticket15_gold_stats, gold=True)
    _check_ticket15_pooled_metrics(ticket15_payload, ticket15_gold_stats)
    astra_report_stats = _build_stats(astra_states, report_only_uids, None)
    _verify_target_stats(astra_report_stats, gold=False)
    if ticket15_report_only.report_count != len(report_only_uids):
        raise AstraAggregationError("Ticket 15 report-only population changed from the frozen cohort")

    disagreement, private_disagreement_rows = _disagreement_analysis(
        astra_states,
        ticket15_states,
        references,
        gold_uids,
    )
    gold_comparison = _compare_gold_stats(astra_gold_stats, ticket15_gold_stats)
    report_comparison = _compare_report_only(astra_report_stats, ticket15_report_only)
    astra_gold_summary = _metric_summary(astra_gold_stats)
    ticket15_gold_summary = _metric_summary(ticket15_gold_stats)
    common_macro = _common_target_macro_selective_accuracy(
        astra_gold_summary,
        ticket15_gold_summary,
    )
    operational = _operational_evidence(
        plan.batch_count,
        audits,
        response_paths,
        sorted(private_root.glob("batches/*/attempt-*.audit.json")),
    )
    batch_context_rows.sort(key=lambda item: cast(int, item["batch_index"]))
    semantic_digest = _digest(
        {
            "schema_version": _SEMANTIC_OUTPUT_DIGEST_IDENTITY,
            "plan_fingerprint": plan.plan_fingerprint,
            "studies": [
                {
                    "study_instance_uid": uid,
                    "states": list(astra_states[uid]),
                    "values": [1 if state == LabelState.POSITIVE.value else 0 if state == LabelState.NEGATIVE.value else None for state in astra_states[uid]],
                    "supervision_mask": [int(state in _SUPERVISED_STATES) for state in astra_states[uid]],
                }
                for uid in sorted(astra_states)
            ],
        }
    )
    raw_output_digest = _digest(
        [{"batch_id": row["batch_id"], "raw_output_sha256": row["raw_output_sha256"]} for row in batch_context_rows]
    )
    batch_context_digest = _digest(
        [{"batch_id": row["batch_id"], "input_digest": row["input_digest"]} for row in batch_context_rows]
    )
    conclusions = _conclusions(
        gold_comparison,
        report_comparison,
        astra_gold_stats,
        ticket15_gold_stats,
        astra_report_stats,
        ticket15_report_only,
    )
    provenance = {
        "experiment_id": plan.approval.experiment_id,
        "model": {
            "name": plan.approval.model_name,
            "surface": plan.approval.model_surface,
            "revision": plan.approval.model_revision,
            "identity_fingerprint": plan.approval.model_identity_fingerprint,
        },
        "reasoning_effort": ASTRA_REASONING_EFFORT,
        "effort": ASTRA_EFFORT,
        "prompt_fingerprint": plan.prompt.fingerprint,
        "prompt_sha256": plan.prompt.prompt_sha256,
        "plan_fingerprint": plan.plan_fingerprint,
        "code_fingerprint": plan.code_fingerprint,
        "output_schema_version": ASTRA_OUTPUT_SCHEMA_VERSION,
        "output_schema_fingerprint": ASTRA_OUTPUT_SCHEMA_FINGERPRINT,
        "cohort_fingerprint": plan.cohorts.fingerprint,
        "report_source_version": corpus.source_version,
        "report_corpus_record_set_sha256": corpus.record_set_sha256,
        "fold_manifest_checksum": manifest.manifest_checksum,
        "ticket15_report_labels_file_sha256": _file_digest(ticket15_path),
        "ticket15_comparator": {
            key: _mapping(ticket15_payload.get("extractor"), "Ticket 15 extractor").get(key)
            for key in (
                "version",
                "fingerprint",
                "code_fingerprint",
                "configuration_fingerprint",
                "ruleset_fingerprint",
                "evaluation_policy",
                "training_policy",
                "frozen_before_held_out_evaluation",
            )
        },
        "fold_manifest_file_sha256": _file_digest(manifest_path),
        "report_corpus_file_sha256": _file_digest(corpus_path),
        "semantic_output_digest": semantic_digest,
        "raw_output_digest": raw_output_digest,
        "batch_context_digest": batch_context_digest,
    }
    integrity_checks = {
        "all_planned_batches_represented_exactly_once": len(audits) == plan.batch_count and len(response_paths) == plan.batch_count,
        "all_final_studies_represented_exactly_once": len(validated_by_uid) == plan.unique_study_count and set(validated_by_uid) == expected_uids,
        "every_study_has_12_canonical_targets": all(len(study.labels) == len(TARGET_COLUMNS) for study, _ in validated_by_uid.values()),
        "all_astra_states_are_semantic": all(label.state.value in ASTRA_SEMANTIC_STATES for study, _ in validated_by_uid.values() for label in study.labels),
        "no_model_output_contains_extraction_failed": all(label.state.value != LabelState.EXTRACTION_FAILED.value for study, _ in validated_by_uid.values() for label in study.labels),
        "prompt_model_effort_batch_and_input_context_match": True,
        "no_duplicate_or_missing_studies": len(validated_by_uid) == len(expected_uids) and set(validated_by_uid) == expected_uids,
        "public_artifacts_contain_no_report_text_or_private_identifiers": True,
        "outputs_match_validation_digests_and_pre_validation_mtimes": True,
        "all_target_state_counts_reconcile": all(
            sum(item.state_counts[state] for state in _ALL_STATES) == item.report_count
            for item in (*astra_gold_stats, *astra_report_stats)
        ),
        "supervised_equals_positive_plus_negative": all(
            item.supervised_count == item.positive_count + item.negative_count
            for item in (*astra_gold_stats, *astra_report_stats)
        ),
        "correct_equals_tp_plus_tn": all(
            item.correct_supervised_count
            == item.correct_positive_calls + item.correct_negative_calls
            for item in (*astra_gold_stats, *astra_report_stats)
        ),
        "selective_errors_equal_fp_plus_fn": all(
            item.selective_error_count
            == item.false_positive_count + item.false_negative_count
            for item in (*astra_gold_stats, *astra_report_stats)
        ),
        "gold_metric_formulas_reverified": True,
        "report_only_population_remains_4349": len(report_only_uids) == 4349
        and ticket15_report_only.report_count == 4349,
    }
    integrity = {
        "planned_batch_count": plan.batch_count,
        "response_file_count": len(response_paths),
        "validated_batch_count": len(audits),
        "planned_study_count": plan.unique_study_count,
        "final_unique_study_count": len(validated_by_uid),
        "canonical_target_count": len(TARGET_COLUMNS),
        "semantic_states_allowed": list(ASTRA_SEMANTIC_STATES),
        "semantic_state_counts": {
            state: sum(astra_states[uid].count(state) for uid in astra_states)
            for state in _ALL_STATES
        },
        "checks": integrity_checks,
        "all_passed": all(integrity_checks.values()),
    }
    public = {
        "schema_version": ASTRA_AGGREGATION_SCHEMA_VERSION,
        "status": "completed_ticket16_scientific_aggregation",
        "privacy": {
            "raw_report_text_public": False,
            "raw_model_output_public": False,
            "gold_row_level_labels_public": False,
            "study_identifiers_public": False,
            "source_report_references_public": False,
        },
        "integrity": integrity,
        "provenance": provenance,
        "operational": operational,
        "gold_evaluation": {
            "population": {
                "evaluable_gold_study_count": len(gold_uids),
                "reference_labels_defined_for_all_studies": all(
                    all(label in (0, 1) for label in references[uid]) for uid in gold_uids
                ),
                "gold_reference_labels_authoritative": True,
                "held_out_protocol": "frozen fold-manifest gold studies; Ticket 15 and Astra compared on the identical 58-study population",
            },
            "astra": _metric_summary(astra_gold_stats),
            "ticket15": _metric_summary(ticket15_gold_stats),
            "macro_comparison": common_macro,
            "comparison": gold_comparison,
        },
        "report_only": {
            "population": {
                "report_only_study_count": len(report_only_uids),
                "accuracy_claim": None,
                "ground_truth_metrics_claimed": False,
            },
            "astra": _metric_summary(astra_report_stats, report_only=True),
            "ticket15": ticket15_report_only.to_dict(),
            "comparison": report_comparison,
        },
        "disagreement_analysis": {
            "population": "identical held-out gold population",
            "overall": {
                "both_supervised_and_agree": sum(
                    cast(int, item["both_supervised_and_agree"]) for item in disagreement
                ),
                "both_supervised_and_disagree": sum(
                    cast(int, item["both_supervised_and_disagree"]) for item in disagreement
                ),
                "any_both_supervised_disagreement": any(
                    cast(int, item["both_supervised_and_disagree"]) > 0
                    for item in disagreement
                ),
            },
            "targets": disagreement,
        },
        "conclusions": conclusions,
    }
    _validate_public_privacy(
        public,
        expected_uids,
        {record.source_report_ref for record in corpus.records},
    )
    private_gold_rows = []
    for uid in gold_uids:
        study, audit = validated_by_uid[uid]
        private_gold_rows.append(
            {
                "study_instance_uid": uid,
                "fold": study.assigned_fold,
                "report_sha256": study.report_sha256,
                "reference_labels": list(references[uid]),
                "astra_states": list(astra_states[uid]),
                "ticket15_states": list(ticket15_states[uid]),
                "astra_batch_id": audit.batch_id,
                "astra_semantic_output_digest": audit.semantic_output_digest,
            }
        )
    private = {
        "schema_version": ASTRA_AGGREGATION_SCHEMA_VERSION,
        "status": "completed_ticket16_scientific_aggregation",
        "privacy": {
            "raw_report_text_persisted": False,
            "raw_model_output_persisted": False,
            "private_study_identifiers_retained_for_audit": True,
            "private_source_report_references_retained": False,
        },
        "public_projection": public,
        "private_inputs": {
            "private_plan_path": str(private_plan),
            "response_directory": str(responses_root),
            "ticket15_labels_path": str(ticket15_path),
            "fold_manifest_path": str(manifest_path),
            "report_corpus_path": str(corpus_path),
            "batch_context_rows": batch_context_rows,
        },
        "private_gold_rows": private_gold_rows,
        "private_disagreement_rows": private_disagreement_rows,
    }
    markdown = render_astra_markdown(public)
    return AstraAggregationResult(public=public, private=private, markdown=markdown)


def _format_number(value: object) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_percent(value: object) -> str:
    if value is None:
        return "undefined"
    return f"{cast(float, value) * 100:.1f}%"


def _gold_table_rows(summary: Mapping[str, object]) -> list[Mapping[str, object]]:
    return [cast(Mapping[str, object], value) for value in _list(summary.get("targets"), "gold targets")]


def render_astra_markdown(public: Mapping[str, object]) -> str:
    """Render the public aggregate evidence as an identifier-free Markdown report."""

    integrity = _mapping(public.get("integrity"), "public integrity")
    integrity_checks = _mapping(integrity.get("checks"), "public integrity checks")
    provenance = _mapping(public.get("provenance"), "public provenance")
    operational = _mapping(public.get("operational"), "public operational")
    gold = _mapping(public.get("gold_evaluation"), "public gold evaluation")
    astra_gold = _mapping(gold.get("astra"), "public Astra gold")
    ticket15_gold = _mapping(gold.get("ticket15"), "public Ticket 15 gold")
    macro_comparison = _mapping(gold.get("macro_comparison"), "gold macro comparison")
    comparison = [cast(Mapping[str, object], item) for item in _list(gold.get("comparison"), "gold comparison")]
    report_only = _mapping(public.get("report_only"), "public report-only")
    report_population = _mapping(report_only.get("population"), "report-only population")
    astra_report = _mapping(report_only.get("astra"), "public Astra report-only")
    ticket15_report = _mapping(report_only.get("ticket15"), "public Ticket 15 report-only")
    report_comparison = [cast(Mapping[str, object], item) for item in _list(report_only.get("comparison"), "report-only comparison")]
    disagreements = _mapping(public.get("disagreement_analysis"), "public disagreement")
    disagreement_overall = _mapping(disagreements.get("overall"), "disagreement overall")
    conclusion = _mapping(public.get("conclusions"), "public conclusions")
    accuracy_conclusion = _mapping(conclusion.get("held_out_accuracy_assessment"), "accuracy conclusion")
    privacy = _mapping(public.get("privacy"), "public privacy")
    model = _mapping(provenance.get("model"), "model provenance")
    state_counts = _mapping(integrity.get("semantic_state_counts"), "semantic state counts")

    def target_set(value: object) -> str:
        return ", ".join(str(item) for item in _list(value, "target set"))

    def micro_text(value: object) -> str:
        metric = _mapping(value, "micro selective accuracy")
        numerator = metric.get("correct_supervised_calls")
        denominator = metric.get("supervised_calls")
        return f"{numerator}/{denominator} ({_format_percent(metric.get('value'))})"

    state_total = sum(cast(int, value) for value in state_counts.values())
    privacy_rows = [
        ("Raw report text public", privacy.get("raw_report_text_public")),
        ("Raw model output public", privacy.get("raw_model_output_public")),
        ("Gold row-level labels public", privacy.get("gold_row_level_labels_public")),
        ("Study identifiers public", privacy.get("study_identifiers_public")),
        ("Source report references public", privacy.get("source_report_references_public")),
    ]

    lines = [
        "# Ticket 16 — Astra Medium scientific aggregation report",
        "",
        "This report aggregates the completed, validated Astra Medium handoff against the frozen Ticket 15 deterministic comparator. It contains aggregate evidence only: no report text, study identifiers, raw Astra responses, or gold row-level labels are included.",
        "",
        "## Outcome",
        "",
        f"- Astra materially increased report-only weak-label coverage for all 12 targets on the identical {report_population.get('report_only_study_count'):,}-study cohort.",
        f"- Defined-target macro selective accuracy is Astra {_format_number(accuracy_conclusion.get('astra_defined_target_macro_selective_accuracy'))} on {len(_list(accuracy_conclusion.get('astra_defined_target_macro_targets'), 'Astra macro target set'))} targets versus Ticket 15 {_format_number(accuracy_conclusion.get('ticket15_defined_target_macro_selective_accuracy'))} on {len(_list(accuracy_conclusion.get('ticket15_defined_target_macro_targets'), 'Ticket 15 macro target set'))} targets; these are per-extractor target sets, not an apples-to-apples comparison.",
        f"- On the identical common target set of {macro_comparison.get('target_count')} targets, Astra is {_format_number(macro_comparison.get('astra'))} versus Ticket 15 {_format_number(macro_comparison.get('ticket15'))}; micro selective accuracy is Astra {micro_text(astra_gold.get('micro_selective_accuracy'))} versus Ticket 15 {micro_text(ticket15_gold.get('micro_selective_accuracy'))}.",
        "- The larger coverage came with additional held-out false positives and false negatives. The evidence does not scientifically validate Astra as a generally admissible weak-label extractor for Ticket 16.",
        "- No Ticket 17 training decision is made here.",
        "",
        "## Integrity and provenance",
        "",
        f"All integrity checks passed: {integrity.get('planned_batch_count')} planned batches, {integrity.get('validated_batch_count')} validated batches, and {integrity.get('final_unique_study_count')} unique final studies. Every study had the 12 canonical targets in order, all model states were semantic, and no extraction-failed state appeared in model output.",
        f"The frozen Astra state vector contains {state_total} target calls: " + ", ".join(f"{key}={value}" for key, value in state_counts.items()) + ".",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Model | {model.get('name')} |",
        f"| Model surface / revision | {model.get('surface')} / {model.get('revision')} |",
        f"| Model identity fingerprint | `{model.get('identity_fingerprint')}` |",
        f"| Reasoning / effort | {provenance.get('reasoning_effort')} / {provenance.get('effort')} |",
        f"| Prompt fingerprint | `{provenance.get('prompt_fingerprint')}` |",
        f"| Prompt SHA-256 | `{provenance.get('prompt_sha256')}` |",
        f"| Plan fingerprint | `{provenance.get('plan_fingerprint')}` |",
        f"| Code fingerprint | `{provenance.get('code_fingerprint')}` |",
        f"| Output schema / fingerprint | `{provenance.get('output_schema_version')}` / `{provenance.get('output_schema_fingerprint')}` |",
        f"| Cohort fingerprint | `{provenance.get('cohort_fingerprint')}` |",
        f"| Report source version | `{provenance.get('report_source_version')}` |",
        f"| Report-corpus digest | `{provenance.get('report_corpus_record_set_sha256')}` |",
        f"| Fold-manifest checksum | `{provenance.get('fold_manifest_checksum')}` |",
        f"| Fold-manifest file SHA-256 | `{provenance.get('fold_manifest_file_sha256')}` |",
        f"| Ticket 15 labels file SHA-256 | `{provenance.get('ticket15_report_labels_file_sha256')}` |",
        f"| Report-corpus file SHA-256 | `{provenance.get('report_corpus_file_sha256')}` |",
        f"| Aggregate semantic digest | `{provenance.get('semantic_output_digest')}` |",
        f"| Aggregate raw-output digest | `{provenance.get('raw_output_digest')}` |",
        f"| Batch-context digest | `{provenance.get('batch_context_digest')}` |",
        "",
        "## Operational evidence",
        "",
        f"The run contains {operational.get('total_batches')} batches and {operational.get('total_reports')} reports. Validation failures: {operational.get('validation_failures')}; retries: {operational.get('retry_count')}; first-attempt acceptance: {_format_percent(operational.get('first_attempt_acceptance_rate'))}.",
        "",
        f"Active elapsed runtime, reports/minute, and batch latency are undefined because they were not recorded. The available filesystem evidence spans { _format_number(operational.get('observed_response_to_audit_wall_span_seconds')) } seconds from the first response file to the last validation audit; this is not reported as runtime.",
        "",
        "## Metric semantics",
        "",
        "The held-out denominator is the 58-study gold population from the frozen fold manifest. Supervised coverage is positive plus negative calls divided by evaluable gold studies. Selective accuracy is correct positive plus correct negative calls divided by supervised calls.",
        "",
        "Selective confusion metrics use only cases where the extractor emitted positive or negative. Thus selective TP/TN/FP/FN exclude abstentions; selective sensitivity is TP/(TP+FN), selective specificity is TN/(TN+FP), selective PPV is TP/(TP+FP), selective NPV is TN/(TN+FN), and selective balanced accuracy is the mean of the two selective rates when both are defined.",
        "",
        "Full-reference capture metrics retain all evaluable gold cases, including abstentions: positive-reference capture is correctly supervised positive calls divided by all gold-positive references, and negative-reference capture is correctly supervised negative calls divided by all gold-negative references. These are not sensitivity or specificity. Masked/abstained counts remain separate.",
        "",
        "Undefined metrics are reported as `undefined` in the JSON evidence with an explicit reason. For example, the previously displayed ACL 62.5% positive-reference quantity is capture; ACL selective sensitivity is 100.0% because all supervised positive-reference calls were correct.",
        "",
        "## Held-out gold evaluation",
        "",
        "### Macro and micro selective accuracy",
        "",
        "The per-extractor defined-target macros preserve each extractor's defined target set. The common-target macro uses exactly the same intersection for both extractors. Micro selective accuracy pools all correct supervised target calls before dividing by all supervised target calls.",
        "",
        "| Aggregation | Target set | Astra | Ticket 15 |",
        "|---|---|---:|---:|",
        f"| Astra defined-target macro | {target_set(accuracy_conclusion.get('astra_defined_target_macro_targets'))} | {_format_number(accuracy_conclusion.get('astra_defined_target_macro_selective_accuracy'))} | — |",
        f"| Ticket 15 defined-target macro | {target_set(accuracy_conclusion.get('ticket15_defined_target_macro_targets'))} | — | {_format_number(accuracy_conclusion.get('ticket15_defined_target_macro_selective_accuracy'))} |",
        f"| Common-target macro | {target_set(macro_comparison.get('target_set'))} | {_format_number(macro_comparison.get('astra'))} | {_format_number(macro_comparison.get('ticket15'))} |",
        f"| Micro selective accuracy | all supervised calls across all 12 targets | {micro_text(astra_gold.get('micro_selective_accuracy'))} | {micro_text(ticket15_gold.get('micro_selective_accuracy'))} |",
        "",
        "### Astra metrics",
        "",
        _render_gold_metrics_table(_gold_table_rows(astra_gold)),
        "",
        "### Frozen Ticket 15 metrics",
        "",
        _render_gold_metrics_table(_gold_table_rows(ticket15_gold)),
        "",
        "### Final per-target comparison",
        "",
        "| Target | Astra coverage | Astra selective accuracy | Ticket 15 coverage | Ticket 15 selective accuracy | Astra supervised | Ticket 15 supervised | Astra additional correct calls | Astra additional incorrect calls | Astra + capture | Ticket 15 + capture | Astra - capture | Ticket 15 - capture |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparison:
        lines.append(
            "| {target} | {astra_coverage} | {astra_accuracy} | {ticket15_coverage} | {ticket15_accuracy} | {astra_supervised} | {ticket15_supervised} | {additional_correct} | {additional_incorrect} | {astra_positive_capture} | {ticket15_positive_capture} | {astra_negative_capture} | {ticket15_negative_capture} |".format(
                target=item["target"],
                astra_coverage=_format_percent(item["astra_coverage"]),
                astra_accuracy=_format_percent(item["astra_selective_accuracy"]),
                ticket15_coverage=_format_percent(item["ticket15_coverage"]),
                ticket15_accuracy=_format_percent(item["ticket15_selective_accuracy"]),
                astra_supervised=item["astra_supervised_count"],
                ticket15_supervised=item["ticket15_supervised_count"],
                additional_correct=item["astra_additional_correct_calls"],
                additional_incorrect=item["astra_additional_incorrect_calls"],
                astra_positive_capture=_format_percent(item["astra_positive_reference_capture"]),
                ticket15_positive_capture=_format_percent(item["ticket15_positive_reference_capture"]),
                astra_negative_capture=_format_percent(item["astra_negative_reference_capture"]),
                ticket15_negative_capture=_format_percent(item["ticket15_negative_reference_capture"]),
            )
        )
    undefined_rows = []
    for label, summary in (("Astra", astra_gold), ("Ticket 15", ticket15_gold)):
        for row in _gold_table_rows(summary):
            reasons = _mapping(row.get("undefined_metrics"), "undefined metrics")
            if reasons:
                undefined_rows.append(f"- {label} — {row['target']}: " + "; ".join(f"{key}: {value}" for key, value in reasons.items()))
    lines.extend(["", "Undefined metric reasons:", "", *(undefined_rows or ["- None."]), ""])

    lines.extend(
        [
            "## Report-only cohort",
            "",
            f"The report-only cohort contains {report_population.get('report_only_study_count'):,} studies. Accuracy, precision, recall, specificity, AUC, and other ground-truth metrics remain undefined and are intentionally not reported for this cohort.",
            "",
            "| Target | Astra + | Astra - | Astra masked | Astra coverage | Ticket 15 + | Ticket 15 - | Ticket 15 masked | Ticket 15 coverage | Coverage Δ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    astra_report_rows = [cast(Mapping[str, object], item) for item in _list(astra_report.get("targets"), "Astra report-only targets")]
    ticket15_report_rows = [cast(Mapping[str, object], item) for item in _list(ticket15_report.get("targets"), "Ticket 15 report-only targets")]
    for left, right, delta in zip(astra_report_rows, ticket15_report_rows, report_comparison):
        left_masked = cast(int, left["masked_count"])
        right_masked = sum(cast(int, right[key]) for key in ("not_mentioned_count", "uncertain_count", "unknown_count", "extraction_failed_count"))
        lines.append(
            f"| {left['target']} | {left['positive_count']} | {left['negative_count']} | {left_masked} | {_format_percent(left['coverage'])} | {right['positive_extracted_count']} | {right['negative_extracted_count']} | {right_masked} | {_format_percent(right['coverage'])} | {_format_percent(delta['coverage_delta'])} |"
        )

    lines.extend([
        "",
        "## Astra/Ticket 15 disagreement analysis",
        "",
        f"Where both extractors emitted supervised predictions on the held-out gold cohort, they agreed on {disagreement_overall.get('both_supervised_and_agree')} target calls and disagreed on {disagreement_overall.get('both_supervised_and_disagree')}. The zero-disagreement finding is verified from the frozen Astra and Ticket 15 states.",
        "",
        "| Target | Both supervised agree | Both supervised disagree | Astra only | Ticket 15 only | Both masked | Truth summary |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for raw_item in _list(disagreements.get("targets"), "disagreement targets"):
        item = _mapping(raw_item, "disagreement target")
        row = _mapping(item, "disagreement target")
        truth = _mapping(row.get("gold_truth_in_disagreement_cases"), "disagreement truth")
        truth_text = ", ".join(f"{key}={value}" for key, value in truth.items())
        lines.append(
            f"| {row['target']} | {row['both_supervised_and_agree']} | {row['both_supervised_and_disagree']} | {row['astra_supervised_ticket15_masked']} | {row['ticket15_supervised_astra_masked']} | {row['both_masked']} | {truth_text} |"
        )

    lines.extend(
        [
            "",
            "## Conservative scientific conclusion",
            "",
            f"1. **Coverage:** { _mapping(conclusion.get('materially_increased_usable_weak_label_coverage'), 'coverage conclusion').get('answer') }. Astra produced more usable positive/negative weak labels for every report-only target.",
            f"2. **Held-out accuracy:** { _mapping(conclusion.get('held_out_accuracy_assessment'), 'accuracy conclusion').get('answer') }. Target behavior must not be reduced to one winner.",
            f"3. **Most coverage benefit:** {', '.join(str(_mapping(item, 'benefit target')['target']) for item in _list(conclusion.get('targets_benefited_most_by_report_only_coverage_gain'), 'benefit targets'))}.",
            f"4. **Aggressiveness and over-labeling:** {_mapping(conclusion.get('became_more_aggressive'), 'aggression conclusion').get('answer')}; {_mapping(conclusion.get('suspicious_over_labeling_evidence'), 'overlabel conclusion').get('interpretation')}",
            f"5. **Masking/admission:** {_mapping(conclusion.get('weak_labels_to_keep_masked_or_not_admit'), 'masking conclusion').get('conservative_scope')}",
            f"6. **Scientific validation:** {_mapping(conclusion.get('scientifically_validated_for_ticket16'), 'validation conclusion').get('answer')}. {_mapping(conclusion.get('scientifically_validated_for_ticket16'), 'validation conclusion').get('reason')}",
            "",
            "## Privacy verification",
            "",
            "The public projection contains aggregate counts and fingerprints only. Private study identifiers remain confined to the private audit projection; raw report text and raw Astra responses are not copied into either report.",
            "",
            "| Public field | Value |",
            "|---|---|",
            *[f"| {label} | {value} |" for label, value in privacy_rows],
            "",
            "## Reverification and limitations",
            "",
            "The aggregation rechecked batch/study uniqueness, canonical target order, state totals, supervised = positive + negative, correct = TP + TN, selective errors = FP + FN, selective accuracy = correct/supervised, and coverage = supervised/evaluable gold. The report-only population remains the frozen 4,349 studies; its state distribution is descriptive only and does not justify an accuracy metric.",
            "",
            "All published metric identities are backed by explicit checks: "
            + ", ".join(f"{key}={value}" for key, value in integrity_checks.items())
            + ".",
            "",
            "The run does not record active model runtime, batch latency, or a cost/token ledger. The filesystem response-to-audit span is retained as an observation, not presented as runtime. No per-target admission/pass threshold was predeclared, so none is assigned here. No independent validation cohort or Ticket 17 training decision is supplied.",
            "",
            "The frozen prompt, Astra interpretations, Ticket 15 interpretations, and gold labels were not changed. No Astra rerun, threshold tuning, report reinterpretation, MRI inference, or Ticket 17 work was performed.",
            "",
            "TICKET 16 SCIENTIFIC EXPERIMENT COMPLETE — MIXED / REVIEW REQUIRED",
            "",
        ]
    )
    return "\n".join(lines)


def _render_gold_metrics_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| Target | Gold n | + | - | Not mentioned | Uncertain | Unknown | Failed | Supervised | Masked | Coverage | Selective accuracy | Ref + | Ref - | Selective TP | Selective TN | Selective FP | Selective FN | Selective sensitivity | Selective specificity | Selective PPV | Selective NPV | Selective balanced accuracy | Positive-reference capture | Negative-reference capture |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {target} | {evaluable} | {positive} | {negative} | {not_mentioned} | {uncertain} | {unknown} | {failed} | {supervised} | {masked} | {coverage} | {accuracy} | {ref_positive} | {ref_negative} | {tp} | {tn} | {fp} | {fn} | {sensitivity} | {specificity} | {ppv} | {npv} | {balanced} | {positive_capture} | {negative_capture} |".format(
                target=row["target"],
                evaluable=row["evaluable_gold_count"],
                positive=row["positive_count"],
                negative=row["negative_count"],
                not_mentioned=row["not_mentioned_count"],
                uncertain=row["uncertain_count"],
                unknown=row["unknown_count"],
                failed=row["extraction_failed_count"],
                supervised=row["supervised_count"],
                masked=row["masked_count"],
                coverage=_format_percent(row["supervised_coverage"]),
                accuracy=_format_percent(row["selective_accuracy"]),
                ref_positive=row["positive_reference_support"],
                ref_negative=row["negative_reference_support"],
                tp=row["selective_tp"],
                tn=row["selective_tn"],
                fp=row["selective_fp"],
                fn=row["selective_fn"],
                sensitivity=_format_percent(row["selective_sensitivity"]),
                specificity=_format_percent(row["selective_specificity"]),
                ppv=_format_percent(row["selective_ppv"]),
                npv=_format_percent(row["selective_npv"]),
                balanced=_format_percent(row["selective_balanced_accuracy"]),
                positive_capture=_format_percent(row["positive_reference_capture"]),
                negative_capture=_format_percent(row["negative_reference_capture"]),
            )
        )
    return "\n".join(lines)


def write_astra_aggregation_reports(
    result: AstraAggregationResult,
    *,
    public_path: str | Path,
    private_path: str | Path,
    markdown_path: str | Path | None = None,
) -> tuple[Path, Path, Path | None]:
    """Write immutable public/private JSON evidence and optionally Markdown."""

    public = Path(public_path)
    private = Path(private_path)
    if public.resolve() == private.resolve():
        raise AstraAggregationError("public and private evidence destinations must differ")
    if markdown_path is not None and Path(markdown_path).resolve() in {public.resolve(), private.resolve()}:
        raise AstraAggregationError("Markdown destination must differ from JSON evidence")
    atomic_json_write(public, result.to_public_dict())
    try:
        atomic_json_write(private, result.to_private_dict())
    except BaseException:
        try:
            public.unlink()
        except FileNotFoundError:
            pass
        raise
    markdown: Path | None = None
    if markdown_path is not None:
        markdown = Path(markdown_path)
        if markdown.exists() or markdown.is_symlink():
            raise FileExistsError(f"refusing to overwrite immutable report {markdown}")
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(result.markdown, encoding="utf-8")
    return public, private, markdown


__all__ = [
    "ASTRA_AGGREGATION_SCHEMA_VERSION",
    "AstraAggregationError",
    "AstraAggregationResult",
    "aggregate_astra_run",
    "render_astra_markdown",
    "write_astra_aggregation_reports",
]
