"""Fold-clean audit and admission of existing report-derived labels.

This module consumes only immutable Ticket 15/16/17 artifacts. It never reads
report text or calls an extractor. For each consuming fold, it estimates the
positive and negative precision of each source on the other folds' gold
reports, then admits a binary state only when its one-sided 95% Wilson lower
bound exceeds the matching training-side class prevalence. The consuming
fold's own gold reports are used only after that gate is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from statistics import NormalDist
from typing import Mapping, Sequence, cast

from .astra_handoff import (
    AstraBatchPlan,
    ASTRA_HANDOFF_CODE_FINGERPRINT,
    ASTRA_OUTPUT_SCHEMA_VERSION,
    AstraBatchAudit,
    AstraValidatedStudy,
    load_astra_audits,
    load_astra_batch_plan,
)
from .folds import FOLD_IDS, FoldManifest, load_fold_manifest
from .report_labels import LabelState
from .targets import NUM_TARGETS, TARGET_COLUMNS


AUDIT_SCHEMA_VERSION = "rsna-knee-report-label-audit-v1"
POLICY_SCHEMA_VERSION = "rsna-knee-report-admission-policy-v1"
POLICY_VERSION = "training-side-wilson-lower-above-class-prevalence-v1"
_VALID_STATES = frozenset(state.value for state in LabelState)
_BINARY_STATES = (LabelState.POSITIVE.value, LabelState.NEGATIVE.value)
_PRIVATE_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "study_instance_uid",
        "source_report_ref",
        "report_text",
        "source_report_refs",
        "source_study_refs",
        "private_inputs",
        "private_paths",
        "gold_rows",
        "row_level",
    }
)


class ReportLabelAuditError(ValueError):
    """Raised when source, cohort, policy, or artifact provenance is unsafe."""


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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ReportLabelAuditError(f"could not hash artifact: {path}") from exc
    return digest.hexdigest()


def _json_file(path: str | Path, field: str) -> Mapping[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportLabelAuditError(f"could not read {field}: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ReportLabelAuditError(f"{field} must contain a JSON object")
    return cast(Mapping[str, object], payload)


def _object(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReportLabelAuditError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ReportLabelAuditError(f"{field} must be an array")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReportLabelAuditError(f"{field} must be non-empty text")
    return value


def _fold(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in FOLD_IDS:
        raise ReportLabelAuditError(f"{field} must be one of {FOLD_IDS}")
    return value


def _states(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ReportLabelAuditError(f"{field} must be an array or state tuple")
    items = value
    if len(items) != NUM_TARGETS:
        raise ReportLabelAuditError(f"{field} must contain exactly {NUM_TARGETS} states")
    result = tuple(_text(item, f"{field}[{index}]") for index, item in enumerate(items))
    if any(item not in _VALID_STATES for item in result):
        raise ReportLabelAuditError(f"{field} contains an unsupported state")
    return result


def _references(value: object, field: str) -> tuple[int, ...]:
    items = _array(value, field)
    if len(items) != NUM_TARGETS or any(
        isinstance(item, bool) or item not in (0, 1) for item in items
    ):
        raise ReportLabelAuditError(f"{field} must contain {NUM_TARGETS} binary reference labels")
    return tuple(cast(int, item) for item in items)


def _state_value(state: str) -> int | None:
    if state == LabelState.POSITIVE.value:
        return 1
    if state == LabelState.NEGATIVE.value:
        return 0
    return None


def make_policy_payload(*, frozen_at_utc: str | None = None) -> dict[str, object]:
    """Return the predeclared fold-specific policy with a content digest."""

    timestamp = frozen_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload: dict[str, object] = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "frozen_at_utc": timestamp,
        "fit_scope": "gold reports assigned outside the consuming fold only",
        "calibration_unit": "study-target call",
        "confidence_interval": {
            "method": "one-sided Wilson score lower bound",
            "confidence_level": 0.95,
            "success": "a binary report call equals its frozen reference label",
        },
        "admission_rule": (
            "admit a positive or negative state for a target and consuming fold only when "
            "its training-side Wilson lower bound for precision is strictly greater than "
            "the training-side prevalence of the called class; otherwise reject that state"
        ),
        "called_class_baseline": "reference-positive prevalence for positive calls; reference-negative prevalence for negative calls",
        "states_always_masked": [
            LabelState.NOT_MENTIONED.value,
            LabelState.UNCERTAIN.value,
            LabelState.UNKNOWN.value,
            LabelState.EXTRACTION_FAILED.value,
        ],
        "reference_precedence": True,
        "prompt_or_rule_adaptation": False,
        "confidence_claim": "model-stated confidence is ignored and is not calibrated reliability",
        "held_out_evaluation_used_to_fit_policy": False,
        "target_order": list(TARGET_COLUMNS),
    }
    payload["policy_sha256"] = _digest(payload)
    return payload


def validate_frozen_policy(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate and copy a policy artifact frozen before outcome evaluation."""

    raw = dict(payload)
    declared = raw.pop("policy_sha256", None)
    if declared != _digest(raw):
        raise ReportLabelAuditError("frozen policy checksum does not match its contents")
    if raw.get("schema_version") != POLICY_SCHEMA_VERSION or raw.get("policy_version") != POLICY_VERSION:
        raise ReportLabelAuditError("unsupported report admission policy")
    if raw.get("target_order") != list(TARGET_COLUMNS):
        raise ReportLabelAuditError("frozen policy target order is not canonical")
    if raw.get("held_out_evaluation_used_to_fit_policy") is not False:
        raise ReportLabelAuditError("frozen policy may not use held-out evaluation")
    timestamp = raw.get("frozen_at_utc")
    if not isinstance(timestamp, str) or make_policy_payload(frozen_at_utc=timestamp) != {
        **raw,
        "policy_sha256": declared,
    }:
        raise ReportLabelAuditError("frozen policy fields differ from the implemented prospective rule")
    raw["policy_sha256"] = declared
    return raw


@dataclass(frozen=True)
class LabelObservation:
    """One text-free study result tied to exact fold and report identities."""

    study_uid: str
    cohort: str
    assigned_fold: int
    report_sha256: str
    states: tuple[str, ...]
    consuming_folds: tuple[int, ...]
    reference_labels: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _text(self.study_uid, "study identity")
        if self.cohort not in {"gold", "report_only"}:
            raise ReportLabelAuditError("observation cohort is invalid")
        _fold(self.assigned_fold, "assigned fold")
        if len(self.report_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.report_sha256
        ):
            raise ReportLabelAuditError("report identity must be a lowercase SHA-256 digest")
        _states(self.states, "observation states")
        if tuple(sorted(self.consuming_folds)) != self.consuming_folds:
            raise ReportLabelAuditError("consuming folds must be sorted")
        if len(set(self.consuming_folds)) != len(self.consuming_folds):
            raise ReportLabelAuditError("consuming folds must be unique")
        if any(fold not in FOLD_IDS for fold in self.consuming_folds):
            raise ReportLabelAuditError("consuming folds are outside the canonical split")
        if self.cohort == "gold":
            if self.reference_labels is None or len(self.reference_labels) != NUM_TARGETS:
                raise ReportLabelAuditError("gold observations require a complete reference vector")
            if any(
                isinstance(value, bool) or value not in (0, 1)
                for value in cast(tuple[object, ...], self.reference_labels)
            ):
                raise ReportLabelAuditError("gold references must be binary")
            if self.consuming_folds != (self.assigned_fold,):
                raise ReportLabelAuditError("gold reports may be consumed only by their held-out fold")
        elif self.reference_labels is not None:
            raise ReportLabelAuditError("report-only observations cannot carry reference labels")
        expected_consumers = tuple(fold for fold in FOLD_IDS if fold != self.assigned_fold)
        if self.cohort == "report_only" and self.consuming_folds != expected_consumers:
            raise ReportLabelAuditError("report-only calls must exclude their assigned fold")


@dataclass(frozen=True)
class LabelSource:
    source_id: str
    identity: Mapping[str, object]
    gold: Mapping[str, LabelObservation]
    report_only: Mapping[str, LabelObservation]


@dataclass(frozen=True)
class Gate:
    target: str
    state: str
    calibration_support: int
    class_support: int
    call_support: int
    correct_calls: int
    reliability: float | None
    lower_95: float | None
    upper_95: float | None
    class_prevalence: float | None
    admitted: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "state": self.state,
            "calibration_support": self.calibration_support,
            "class_support": self.class_support,
            "call_support": self.call_support,
            "correct_calls": self.correct_calls,
            "reliability": self.reliability,
            "one_sided_wilson_95_lower": self.lower_95,
            "one_sided_wilson_95_upper": self.upper_95,
            "called_class_prevalence": self.class_prevalence,
            "admitted": self.admitted,
            "reason": self.reason,
        }


def wilson_interval(successes: int, total: int, *, confidence_level: float = 0.95) -> tuple[float, float] | None:
    """Return the two-sided Wilson interval; None means no estimable calls."""

    if total == 0:
        return None
    if successes < 0 or successes > total:
        raise ReportLabelAuditError("Wilson successes must be within the call support")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _one_sided_wilson(successes: int, total: int, *, confidence_level: float = 0.95) -> tuple[float, float] | None:
    if total == 0:
        return None
    z = NormalDist().inv_cdf(confidence_level)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _fit_gates(source: LabelSource, consuming_fold: int) -> dict[tuple[int, str], Gate]:
    training_gold = tuple(
        row for row in source.gold.values() if row.assigned_fold != consuming_fold
    )
    if any(row.assigned_fold == consuming_fold for row in training_gold):
        raise ReportLabelAuditError("consuming-fold gold entered the policy fit")
    if not training_gold:
        raise ReportLabelAuditError("policy fit has no training-side gold reports")
    gates: dict[tuple[int, str], Gate] = {}
    for target_index, target in enumerate(TARGET_COLUMNS):
        reference_values = tuple(cast(tuple[int, ...], row.reference_labels)[target_index] for row in training_gold)
        for state in _BINARY_STATES:
            call_value = _state_value(state)
            class_support = sum(value == call_value for value in reference_values)
            call_rows = tuple(row for row in training_gold if row.states[target_index] == state)
            correct = sum(cast(tuple[int, ...], row.reference_labels)[target_index] == call_value for row in call_rows)
            support = len(training_gold)
            prevalence = class_support / support if support else None
            interval = _one_sided_wilson(correct, len(call_rows))
            reliability = correct / len(call_rows) if call_rows else None
            if not call_rows:
                admitted, reason = False, "no_training_side_calls"
            elif not class_support:
                admitted, reason = False, "no_training_side_reference_support_for_called_class"
            elif interval is None or prevalence is None or interval[0] <= prevalence:
                admitted, reason = False, "lower_bound_not_above_training_side_prevalence"
            else:
                admitted, reason = True, "lower_bound_above_training_side_prevalence"
            gates[(target_index, state)] = Gate(
                target=target,
                state=state,
                calibration_support=support,
                class_support=class_support,
                call_support=len(call_rows),
                correct_calls=correct,
                reliability=reliability,
                lower_95=interval[0] if interval else None,
                upper_95=interval[1] if interval else None,
                class_prevalence=prevalence,
                admitted=admitted,
                reason=reason,
            )
    return gates


def _admitted_state(state: str, gate: Gate) -> bool:
    return state in _BINARY_STATES and gate.admitted


def _target_metrics(
    rows: Sequence[LabelObservation],
    target_index: int,
    gates: Mapping[tuple[int, str], Gate] | None,
) -> dict[str, object]:
    states = tuple(row.states[target_index] for row in rows)
    if any(row.reference_labels is None for row in rows):
        raise ReportLabelAuditError("target metrics require gold reference rows")
    references = tuple(cast(tuple[int, ...], row.reference_labels)[target_index] for row in rows)
    state_counts = {state: states.count(state) for state in sorted(_VALID_STATES)}
    positive_indices = [index for index, state in enumerate(states) if state == LabelState.POSITIVE.value]
    negative_indices = [index for index, state in enumerate(states) if state == LabelState.NEGATIVE.value]
    call_indices = positive_indices + negative_indices
    correct_positive = sum(references[index] == 1 for index in positive_indices)
    correct_negative = sum(references[index] == 0 for index in negative_indices)
    positive_ci = wilson_interval(correct_positive, len(positive_indices))
    negative_ci = wilson_interval(correct_negative, len(negative_indices))
    positive_gate = gates.get((target_index, LabelState.POSITIVE.value)) if gates is not None else None
    negative_gate = gates.get((target_index, LabelState.NEGATIVE.value)) if gates is not None else None
    admitted_positive_support = len(positive_indices) if positive_gate and positive_gate.admitted else 0
    admitted_negative_support = len(negative_indices) if negative_gate and negative_gate.admitted else 0
    admitted_positive_correct = correct_positive if admitted_positive_support else 0
    admitted_negative_correct = correct_negative if admitted_negative_support else 0
    admitted_positive_ci = wilson_interval(admitted_positive_correct, admitted_positive_support)
    admitted_negative_ci = wilson_interval(admitted_negative_correct, admitted_negative_support)
    admitted_indices = []
    if gates is not None:
        admitted_indices = [
            index
            for index, state in enumerate(states)
            if state in _BINARY_STATES and _admitted_state(state, gates[(target_index, state)])
        ]
    support = len(rows)
    return {
        "target": TARGET_COLUMNS[target_index],
        "reference_support": support,
        "reference_positive_support": sum(value == 1 for value in references),
        "reference_negative_support": sum(value == 0 for value in references),
        "state_counts": state_counts,
        "positive_reliability": correct_positive / len(positive_indices) if positive_indices else None,
        "positive_reliability_95_ci": list(positive_ci) if positive_ci else None,
        "positive_call_support": len(positive_indices),
        "positive_state_admitted_by_policy": positive_gate.admitted if positive_gate else None,
        "admitted_positive_call_support": admitted_positive_support if gates is not None else None,
        "admitted_positive_reliability": (
            admitted_positive_correct / admitted_positive_support if admitted_positive_support else None
        ),
        "admitted_positive_reliability_95_ci": list(admitted_positive_ci) if admitted_positive_ci else None,
        "negative_reliability": correct_negative / len(negative_indices) if negative_indices else None,
        "negative_reliability_95_ci": list(negative_ci) if negative_ci else None,
        "negative_call_support": len(negative_indices),
        "negative_state_admitted_by_policy": negative_gate.admitted if negative_gate else None,
        "admitted_negative_call_support": admitted_negative_support if gates is not None else None,
        "admitted_negative_reliability": (
            admitted_negative_correct / admitted_negative_support if admitted_negative_support else None
        ),
        "admitted_negative_reliability_95_ci": list(admitted_negative_ci) if admitted_negative_ci else None,
        "selective_accuracy": (correct_positive + correct_negative) / len(call_indices) if call_indices else None,
        "selective_call_support": len(call_indices),
        "binary_call_coverage": len(call_indices) / support if support else None,
        "not_mentioned_count": state_counts[LabelState.NOT_MENTIONED.value],
        "uncertain_count": state_counts[LabelState.UNCERTAIN.value],
        "unknown_count": state_counts[LabelState.UNKNOWN.value],
        "extraction_failed_count": state_counts[LabelState.EXTRACTION_FAILED.value],
        "admitted_call_support": len(admitted_indices) if gates is not None else None,
        "admitted_call_coverage": len(admitted_indices) / support if support and gates is not None else None,
        "observable_reference_disagreements": (len(positive_indices) - correct_positive) + (len(negative_indices) - correct_negative),
    }


def _gate_summary(gates: Mapping[tuple[int, str], Gate]) -> list[dict[str, object]]:
    return [gates[(target_index, state)].to_dict() for target_index in range(NUM_TARGETS) for state in _BINARY_STATES]


def _target_gate_decision(gates: Mapping[tuple[int, str], Gate], target_index: int) -> str:
    positive = gates[(target_index, LabelState.POSITIVE.value)].admitted
    negative = gates[(target_index, LabelState.NEGATIVE.value)].admitted
    if positive and negative:
        return "both_binary_states_admitted"
    if positive:
        return "positive_only_admitted"
    if negative:
        return "negative_only_admitted"
    return "no_binary_state_admitted"


def _report_only_target_metrics(
    rows: Sequence[LabelObservation],
    target_index: int,
    gates: Mapping[tuple[int, str], Gate],
) -> dict[str, object]:
    states = tuple(row.states[target_index] for row in rows)
    accepted = [
        state
        for state in states
        if state in _BINARY_STATES and _admitted_state(state, gates[(target_index, state)])
    ]
    return {
        "target": TARGET_COLUMNS[target_index],
        "eligible_report_support": len(rows),
        "state_counts": {state: states.count(state) for state in sorted(_VALID_STATES)},
        "binary_call_support": sum(state in _BINARY_STATES for state in states),
        "binary_call_coverage": sum(state in _BINARY_STATES for state in states) / len(rows) if rows else None,
        "admitted_positive_calls": accepted.count(LabelState.POSITIVE.value),
        "admitted_negative_calls": accepted.count(LabelState.NEGATIVE.value),
        "admitted_call_support": len(accepted),
        "rejected_binary_call_support": sum(state in _BINARY_STATES for state in states) - len(accepted),
        "usable_call_coverage": len(accepted) / len(rows) if rows else None,
        "not_mentioned_count": states.count(LabelState.NOT_MENTIONED.value),
        "uncertain_count": states.count(LabelState.UNCERTAIN.value),
        "unknown_count": states.count(LabelState.UNKNOWN.value),
        "extraction_failed_count": states.count(LabelState.EXTRACTION_FAILED.value),
    }


def _cohort_admission_summary(
    rows: Sequence[LabelObservation],
    gates: Mapping[tuple[int, str], Gate],
) -> dict[str, int]:
    """Count accepted calls and unique studies for one consuming-fold cohort."""

    admitted_studies = admitted_calls = rejected_calls = 0
    for row in rows:
        has_admitted_call = False
        for target_index, state in enumerate(row.states):
            if state not in _BINARY_STATES:
                continue
            gate = gates[(target_index, state)]
            if gate.admitted:
                admitted_calls += 1
                has_admitted_call = True
            else:
                rejected_calls += 1
        admitted_studies += int(has_admitted_call)
    return {
        "eligible_report_only_studies": len(rows),
        "admitted_studies_with_any_call": admitted_studies,
        "admitted_target_calls": admitted_calls,
        "rejected_binary_calls": rejected_calls,
    }


def _paired_coverage(
    ticket15_rows: Sequence[LabelObservation],
    learned_rows: Sequence[LabelObservation],
    target_index: int,
    ticket15_gates: Mapping[tuple[int, str], Gate],
    learned_gates: Mapping[tuple[int, str], Gate],
    *,
    with_reference: bool,
) -> dict[str, object]:
    by_ticket15 = {row.study_uid: row for row in ticket15_rows}
    by_learned = {row.study_uid: row for row in learned_rows}
    if set(by_ticket15) != set(by_learned):
        raise ReportLabelAuditError("paired extractor cohorts differ")
    ticket15_only = learned_only = shared = neither = agreement = disagreement = 0
    learned_only_correct = learned_only_support = 0
    ticket15_only_correct = ticket15_only_support = 0
    admitted_learned_only = admitted_ticket15_only = 0
    shared_admitted_by_ticket15 = shared_admitted_by_learned = 0
    shared_admitted_by_both = shared_admitted_by_either = 0
    for uid in sorted(by_ticket15):
        left = by_ticket15[uid]
        right = by_learned[uid]
        if left.report_sha256 != right.report_sha256:
            raise ReportLabelAuditError("paired extractor report digests differ")
        left_state = left.states[target_index]
        right_state = right.states[target_index]
        left_call = left_state in _BINARY_STATES
        right_call = right_state in _BINARY_STATES
        if left_call and right_call:
            shared += 1
            agreement += left_state == right_state
            disagreement += left_state != right_state
            left_admitted = _admitted_state(left_state, ticket15_gates[(target_index, left_state)])
            right_admitted = _admitted_state(right_state, learned_gates[(target_index, right_state)])
            shared_admitted_by_ticket15 += int(left_admitted)
            shared_admitted_by_learned += int(right_admitted)
            shared_admitted_by_both += int(left_admitted and right_admitted)
            shared_admitted_by_either += int(left_admitted or right_admitted)
        elif left_call:
            ticket15_only += 1
            admitted_ticket15_only += int(_admitted_state(left_state, ticket15_gates[(target_index, left_state)]))
            if with_reference:
                assert left.reference_labels is not None
                ticket15_only_support += 1
                ticket15_only_correct += _state_value(left_state) == left.reference_labels[target_index]
        elif right_call:
            learned_only += 1
            admitted_learned_only += int(_admitted_state(right_state, learned_gates[(target_index, right_state)]))
            if with_reference:
                assert right.reference_labels is not None
                learned_only_support += 1
                learned_only_correct += _state_value(right_state) == right.reference_labels[target_index]
        else:
            neither += 1
    return {
        "ticket15_only_calls": ticket15_only,
        "learned_only_new_calls": learned_only,
        "shared_calls": shared,
        "neither_source_called": neither,
        "shared_state_agreements": agreement,
        "shared_state_disagreements": disagreement,
        "learned_only_admitted_calls": admitted_learned_only,
        "ticket15_only_admitted_calls": admitted_ticket15_only,
        "shared_admitted_by_ticket15_calls": shared_admitted_by_ticket15,
        "shared_admitted_by_learned_calls": shared_admitted_by_learned,
        "shared_admitted_by_both_calls": shared_admitted_by_both,
        "shared_admitted_by_either_calls": shared_admitted_by_either,
        "learned_only_reference_correct": learned_only_correct if with_reference else None,
        "learned_only_reference_support": learned_only_support if with_reference else None,
        "learned_only_reference_reliability": (
            learned_only_correct / learned_only_support if with_reference and learned_only_support else None
        ),
        "ticket15_only_reference_correct": ticket15_only_correct if with_reference else None,
        "ticket15_only_reference_support": ticket15_only_support if with_reference else None,
        "ticket15_only_reference_reliability": (
            ticket15_only_correct / ticket15_only_support if with_reference and ticket15_only_support else None
        ),
    }


def _make_private_admissions(
    source: LabelSource,
    gates_by_fold: Mapping[int, Mapping[tuple[int, str], Gate]],
) -> dict[str, object]:
    folds: list[dict[str, object]] = []
    for fold in FOLD_IDS:
        rows = sorted(
            (row for row in source.report_only.values() if fold in row.consuming_folds),
            key=lambda row: row.study_uid,
        )
        gates = gates_by_fold[fold]
        admitted_rows: list[dict[str, object]] = []
        for row in rows:
            target_rows: list[dict[str, object]] = []
            for index, (target, state) in enumerate(zip(TARGET_COLUMNS, row.states)):
                gate = gates.get((index, state)) if state in _BINARY_STATES else None
                accepted = bool(gate and gate.admitted)
                target_rows.append(
                    {
                        "target": target,
                        "state": state,
                        "value": _state_value(state) if accepted else None,
                        "supervision_mask": int(accepted),
                        "admitted": accepted,
                        "rejection_reason": None if accepted else (
                            gate.reason if gate is not None else "state_is_always_masked"
                        ),
                    }
                )
            admitted_rows.append(
                {
                    "study_instance_uid": row.study_uid,
                    "assigned_fold": row.assigned_fold,
                    "consuming_fold": fold,
                    "report_sha256": row.report_sha256,
                    "targets": target_rows,
                }
            )
        summary = _cohort_admission_summary(rows, gates)
        folds.append(
            {
                "fold": fold,
                "calibration_folds": [item for item in FOLD_IDS if item != fold],
                "eligible_report_only_study_count": summary["eligible_report_only_studies"],
                "admitted_studies_with_any_call": summary["admitted_studies_with_any_call"],
                "admitted_target_call_count": summary["admitted_target_calls"],
                "studies": admitted_rows,
            }
        )
    return {"source_id": source.source_id, "folds": folds}


def _ensure_public_safe(value: object, known_private_values: Sequence[str] = ()) -> None:
    encoded = _canonical_json(value)
    for private_value in known_private_values:
        if private_value and private_value in encoded:
            raise ReportLabelAuditError("public report contains a private identity")

    def visit(current: object, path: str) -> None:
        if isinstance(current, Mapping):
            for key, item in current.items():
                if str(key) in _PRIVATE_FORBIDDEN_PUBLIC_KEYS:
                    raise ReportLabelAuditError(f"public report contains private field {path}.{key}")
                visit(item, f"{path}.{key}")
        elif isinstance(current, list):
            for index, item in enumerate(current):
                visit(item, f"{path}[{index}]")

    visit(value, "public")


def build_audit_artifacts(
    *,
    frozen_policy: Mapping[str, object],
    manifest: FoldManifest,
    ticket15_labels: Mapping[str, object],
    ticket15_labels_sha256: str,
    ticket15_repeat_labels: Mapping[str, object] | None,
    ticket15_repeat_labels_sha256: str | None,
    ticket16_plan: object,
    ticket16_audits: Sequence[AstraBatchAudit],
    ticket16_private_aggregation: Mapping[str, object],
    ticket16_public_aggregation: Mapping[str, object],
    ticket16_private_aggregation_sha256: str,
    ticket16_public_aggregation_sha256: str,
    ticket16_plan_sha256: str,
    ticket16_plan_metadata: Mapping[str, object],
    ticket16_prompt_sha256: str,
    ticket16_response_manifest: Mapping[str, object],
    ticket17_private_supervision: Mapping[str, object],
    ticket17_public_supervision: Mapping[str, object],
    ticket17_private_sha256: str,
    ticket17_public_sha256: str,
    ticket04_manifest_sha256: str,
    ticket16_report_corpus_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate existing artifact lineage, freeze gates, and emit both views."""

    policy = validate_frozen_policy(frozen_policy)
    if manifest.superseded or manifest.fold_ids != FOLD_IDS:
        raise ReportLabelAuditError("report audit requires the canonical nonsuperseded three-fold manifest")
    if ticket15_labels.get("schema_version") != "rsna-knee-report-extraction-v1":
        raise ReportLabelAuditError("Ticket 15 artifact schema is unsupported")
    if ticket15_labels.get("status") != "completed_synthetic_or_private_local_report_evaluation":
        raise ReportLabelAuditError("Ticket 15 artifact is not a completed extraction record")
    t15_fold_context = _object(ticket15_labels.get("fold_context"), "Ticket 15 fold context")
    if t15_fold_context.get("fold_manifest_checksum") != manifest.manifest_checksum:
        raise ReportLabelAuditError("Ticket 15 fold identity differs from the canonical manifest")
    t15_input = _object(ticket15_labels.get("input"), "Ticket 15 report source")
    t15_provenance = _object(ticket15_labels.get("provenance"), "Ticket 15 provenance")
    t15_extractor = _object(ticket15_labels.get("extractor"), "Ticket 15 extractor")
    t15_boundaries = _object(ticket15_labels.get("boundaries"), "Ticket 15 boundaries")
    if t15_input.get("raw_report_text_persisted") is not False:
        raise ReportLabelAuditError("Ticket 15 artifact must not persist raw report text")
    t15_privacy = _object(ticket15_labels.get("privacy"), "Ticket 15 privacy")
    if t15_privacy.get("raw_report_text_persisted") is not False:
        raise ReportLabelAuditError("Ticket 15 privacy boundary is not recorded")
    if _object(ticket15_labels.get("reproducibility"), "Ticket 15 reproducibility").get("verified") is not True:
        raise ReportLabelAuditError("Ticket 15 output is not reproducibility-verified")
    if t15_extractor.get("frozen_before_held_out_evaluation") is not True:
        raise ReportLabelAuditError("Ticket 15 extractor was not frozen before held-out evaluation")
    current_t15_code = hashlib.sha256(Path(__file__).with_name("report_extraction.py").read_bytes()).hexdigest()
    if t15_extractor.get("code_fingerprint") != current_t15_code:
        raise ReportLabelAuditError("Ticket 15 extraction code no longer matches its frozen artifact")
    if ticket15_repeat_labels is not None:
        repeat_reproducibility = _object(ticket15_repeat_labels.get("reproducibility"), "Ticket 15 repeat reproducibility")
        repeat_input = _object(ticket15_repeat_labels.get("input"), "Ticket 15 repeat input")
        repeat_context = _object(ticket15_repeat_labels.get("fold_context"), "Ticket 15 repeat fold context")
        repeat_extractor = _object(ticket15_repeat_labels.get("extractor"), "Ticket 15 repeat extractor")
        if (
            ticket15_repeat_labels.get("schema_version") != ticket15_labels.get("schema_version")
            or repeat_reproducibility.get("verified") is not True
            or repeat_reproducibility.get("semantic_digest")
            != _object(ticket15_labels.get("reproducibility"), "Ticket 15 reproducibility").get("semantic_digest")
            or repeat_input.get("record_set_sha256") != t15_input.get("record_set_sha256")
            or repeat_context.get("fold_manifest_checksum") != manifest.manifest_checksum
            or repeat_extractor.get("fingerprint") != t15_extractor.get("fingerprint")
        ):
            raise ReportLabelAuditError("Ticket 15 repeat artifact does not reproduce the frozen extraction")
    for key in (
        "held_out_gold_reports_are_evaluation_only",
        "held_out_reports_excluded_from_training",
        "reference_labels_take_precedence",
        "report_derived_rules_cross_fold",
        "training_labels_are_report_only",
        "unknown_uncertain_failed_masked",
    ):
        expected = False if key == "report_derived_rules_cross_fold" else True
        if t15_boundaries.get(key) is not expected:
            raise ReportLabelAuditError(f"Ticket 15 boundary {key} is not satisfied")

    plan = cast(AstraBatchPlan, ticket16_plan)
    cohorts = getattr(plan, "cohorts", None)
    approval = getattr(plan, "approval", None)
    prompt = getattr(plan, "prompt", None)
    batches = getattr(plan, "batches", None)
    if cohorts is None or approval is None or prompt is None or batches is None:
        raise ReportLabelAuditError("Ticket 16 plan did not load as a complete frozen plan")
    if cohorts.fold_manifest_checksum != manifest.manifest_checksum:
        raise ReportLabelAuditError("Ticket 16 fold identity differs from the canonical manifest")
    if cohorts.report_record_set_sha256 != t15_input.get("record_set_sha256"):
        raise ReportLabelAuditError("Ticket 15 and 16 report source digests differ")
    if cohorts.report_source_version != t15_input.get("source_version"):
        raise ReportLabelAuditError("Ticket 15 and 16 report source versions differ")
    if len(cohorts.folds) != len(FOLD_IDS):
        raise ReportLabelAuditError("Ticket 16 plan does not contain the three canonical fold cohorts")
    for fold, plan_fold in zip(FOLD_IDS, cohorts.folds):
        expected_train_studies = set(manifest.training_study_uids(fold))
        expected_held_out_studies = set(manifest.held_out_study_uids(fold))
        expected_training_reports = {
            item.study_instance_uid
            for item in manifest.report_only_memberships
            if item.fold != fold
        }
        expected_held_out_reports = expected_held_out_studies
        expected_evaluation_gold = {
            item.study_instance_uid
            for item in manifest.gold_memberships
            if item.fold == fold
        }
        if (
            plan_fold.fold != fold
            or set(plan_fold.fold_training_study_uids) != expected_train_studies
            or set(plan_fold.fold_held_out_study_uids) != expected_held_out_studies
            or set(plan_fold.eligible_training_report_uids) != expected_training_reports
            or set(plan_fold.eligible_evaluation_gold_uids) != expected_evaluation_gold
            or plan_fold.missing_training_report_uids
            or plan_fold.missing_evaluation_report_uids
            or set(plan_fold.excluded_held_out_report_uids) != expected_held_out_reports
        ):
            raise ReportLabelAuditError(f"Ticket 16 fold {fold} eligibility differs from the canonical manifest")
    if not approval.private_report_execution_approved:
        raise ReportLabelAuditError("Ticket 16 private report execution lacks explicit owner approval")
    if approval.data_transfer_policy != "private-reports-remain-in-user-controlled-codex-session":
        raise ReportLabelAuditError("Ticket 16 report usage policy is not the approved private handoff")
    if not prompt.approved or not prompt.is_frozen:
        raise ReportLabelAuditError("Ticket 16 prompt is not approved and frozen")
    if getattr(approval, "experiment_id", None) != "ticket16-astra-medium-codex-subscription-v1":
        raise ReportLabelAuditError("Ticket 16 model identity is not the approved experiment")
    if getattr(plan, "code_fingerprint", None) is None:
        raise ReportLabelAuditError("Ticket 16 plan lacks its output validator identity")
    if plan.code_fingerprint != ASTRA_HANDOFF_CODE_FINGERPRINT:
        raise ReportLabelAuditError("Ticket 16 output validator code differs from the frozen plan")
    protocol = _object(ticket16_plan_metadata.get("protocol"), "Ticket 16 protocol")
    plan_experiment = _object(ticket16_plan_metadata.get("experiment"), "Ticket 16 approved experiment")
    approval_scope = _object(plan_experiment.get("approval_scope"), "Ticket 16 approval scope")
    if approval_scope.get("identity_approved") is not True or approval_scope.get("private_report_execution_approved") is not True:
        raise ReportLabelAuditError("Ticket 16 model identity or private report use lacks exact approval")
    raw_schema = _object(ticket16_plan_metadata.get("output_schema"), "Ticket 16 output schema")
    if raw_schema.get("schema_version") != ASTRA_OUTPUT_SCHEMA_VERSION or raw_schema.get("target_order") != list(TARGET_COLUMNS):
        raise ReportLabelAuditError("Ticket 16 frozen output schema or target order is inconsistent")
    for key in (
        "canonical_three_fold_isolation",
        "gold_labels_from_frozen_manifest_take_precedence",
        "held_out_reports_never_admitted_to_training",
        "prompt_tuning_or_adaptation",
        "weak_label_training",
        "raw_report_text_public",
        "raw_model_output_public",
        "study_identifiers_public",
    ):
        expected = False if key in {"prompt_tuning_or_adaptation", "weak_label_training", "raw_report_text_public", "raw_model_output_public", "study_identifiers_public"} else True
        if protocol.get(key) is not expected:
            raise ReportLabelAuditError(f"Ticket 16 protocol boundary {key} is not satisfied")

    embedded_public = _object(ticket16_private_aggregation.get("public_projection"), "Ticket 16 public projection")
    if embedded_public != ticket16_public_aggregation:
        raise ReportLabelAuditError("Ticket 16 public aggregation differs from the private artifact projection")
    if ticket16_private_aggregation.get("schema_version") != "rsna-knee-ticket16-astra-aggregation-v2":
        raise ReportLabelAuditError("Ticket 16 corrected aggregation schema is unsupported")
    aggregate_inputs = _object(ticket16_private_aggregation.get("private_inputs"), "Ticket 16 aggregation inputs")
    aggregate_provenance = _object(ticket16_public_aggregation.get("provenance"), "Ticket 16 aggregation provenance")
    plan_fingerprint = getattr(plan, "plan_fingerprint", None)
    if aggregate_provenance.get("plan_fingerprint") != plan_fingerprint:
        raise ReportLabelAuditError("Ticket 16 aggregation is bound to another plan")
    if aggregate_provenance.get("fold_manifest_checksum") != manifest.manifest_checksum:
        raise ReportLabelAuditError("Ticket 16 aggregate fold checksum differs")
    if aggregate_provenance.get("prompt_fingerprint") != prompt.fingerprint:
        raise ReportLabelAuditError("Ticket 16 aggregation is bound to another prompt")
    if aggregate_provenance.get("output_schema_version") != ASTRA_OUTPUT_SCHEMA_VERSION:
        raise ReportLabelAuditError("Ticket 16 output schema differs from the fixed handoff contract")
    if aggregate_provenance.get("ticket15_report_labels_file_sha256") != ticket15_labels_sha256:
        raise ReportLabelAuditError("Ticket 16 aggregation is bound to another Ticket 15 label artifact")
    if aggregate_provenance.get("report_corpus_file_sha256") != ticket16_report_corpus_sha256:
        raise ReportLabelAuditError("Ticket 16 report source file hash differs from its aggregation")
    if aggregate_provenance.get("report_corpus_record_set_sha256") != cohorts.report_record_set_sha256:
        raise ReportLabelAuditError("Ticket 16 aggregate report-set digest differs from the frozen plan")
    if ticket16_public_aggregation.get("status") != "completed_ticket16_scientific_aggregation":
        raise ReportLabelAuditError("Ticket 16 aggregation is not complete")
    aggregation_batch_rows = _array(aggregate_inputs.get("batch_context_rows"), "Ticket 16 batch context")
    response_manifest_rows = _array(ticket16_response_manifest.get("responses"), "Ticket 16 response manifest")
    if len(aggregation_batch_rows) != len(response_manifest_rows):
        raise ReportLabelAuditError("Ticket 16 aggregate batch context is incomplete")
    aggregate_batches = {
        _text(_object(row, "Ticket 16 aggregate batch").get("batch_id"), "Ticket 16 aggregate batch id"):
        _object(row, "Ticket 16 aggregate batch")
        for row in aggregation_batch_rows
    }
    response_batches: dict[str, Mapping[str, object]] = {}
    for raw_response in response_manifest_rows:
        response = _object(raw_response, "Ticket 16 response manifest row")
        batch_id = _text(response.get("batch_id"), "Ticket 16 response manifest batch id")
        if batch_id in response_batches:
            raise ReportLabelAuditError("Ticket 16 response manifest contains a duplicate batch")
        response_batches[batch_id] = response
    if set(aggregate_batches) != set(response_batches):
        raise ReportLabelAuditError("Ticket 16 aggregate output batch set differs from immutable responses")
    for batch_id, response in response_batches.items():
        aggregate_batch = aggregate_batches[batch_id]
        for field, expected_value in (
            ("input_digest", response["input_digest"]),
            ("raw_output_sha256", response["raw_output_sha256"]),
            ("semantic_output_digest", response["semantic_output_digest"]),
            ("study_count", response["study_count"]),
        ):
            if aggregate_batch.get(field) != expected_value:
                raise ReportLabelAuditError("Ticket 16 aggregate batch identity differs from immutable response")

    audit_by_batch: dict[str, AstraBatchAudit] = {}
    for audit in ticket16_audits:
        if audit.batch_id in audit_by_batch:
            raise ReportLabelAuditError("Ticket 16 plan has duplicate immutable audits")
        audit_by_batch[audit.batch_id] = audit
    expected_batch_ids = {batch.batch_id for batch in batches}
    if set(audit_by_batch) != expected_batch_ids:
        raise ReportLabelAuditError("Ticket 16 audits do not cover the exact frozen batch set")
    if any(audit.status not in {"validated", "extraction-failed"} or audit.attempt != 1 for audit in ticket16_audits):
        raise ReportLabelAuditError("Ticket 16 has a retried or context-rejected output batch")
    for audit in ticket16_audits:
        if audit.status == "extraction-failed" and any(
            label.state is not LabelState.EXTRACTION_FAILED
            for study in audit.studies
            for label in study.labels
        ):
            raise ReportLabelAuditError("failed Ticket 16 batches must preserve extraction-failed masks")
    if ticket16_response_manifest.get("batch_count") != len(batches):
        raise ReportLabelAuditError("Ticket 16 raw output responses do not cover every batch")
    if ticket16_response_manifest.get("validated_response_count") != len(batches):
        raise ReportLabelAuditError("Ticket 16 raw output response count is incomplete")

    memberships = {item.study_instance_uid: item for item in manifest.memberships}
    t15_training: dict[int, dict[str, LabelObservation]] = {fold: {} for fold in FOLD_IDS}
    t15_gold: dict[str, LabelObservation] = {}
    raw_folds = _array(ticket15_labels.get("fold_partitions"), "Ticket 15 fold partitions")
    if len(raw_folds) != len(FOLD_IDS):
        raise ReportLabelAuditError("Ticket 15 artifact must contain all three folds")
    for raw_fold in raw_folds:
        fold_record = _object(raw_fold, "Ticket 15 fold record")
        fold = _fold(fold_record.get("fold"), "Ticket 15 fold")
        if fold in t15_training and t15_training[fold]:
            raise ReportLabelAuditError("Ticket 15 contains a duplicate fold")
        partitions = _object(fold_record.get("partitions"), "Ticket 15 fold partitions")
        for kind, destination, expected_cohort in (
            ("training_labels", t15_training[fold], "report_only"),
            ("held_out_evaluation", {}, "gold"),
        ):
            records = _array(partitions.get(kind), f"Ticket 15 {kind}")
            for raw_row in records:
                raw_mapping = _object(raw_row, "Ticket 15 report record")
                uid = _text(raw_mapping.get("study_instance_uid"), "Ticket 15 study identity")
                membership = memberships.get(uid)
                if membership is None or membership.cohort != expected_cohort:
                    raise ReportLabelAuditError("Ticket 15 report row has an unknown fold cohort")
                row = _parse_t15_observation(
                    raw_row,
                    fold=fold,
                    assigned_fold=membership.fold,
                    expected_cohort=expected_cohort,
                    reference_labels=membership.reference_labels,
                )
                if row.assigned_fold != membership.fold:
                    raise ReportLabelAuditError("Ticket 15 report row assigned fold differs from manifest")
                if expected_cohort == "report_only":
                    if membership.fold == fold:
                        raise ReportLabelAuditError("Ticket 15 admitted a report from the consuming fold's holdout")
                    if row.study_uid in destination:
                        raise ReportLabelAuditError("Ticket 15 has duplicate training rows within a fold")
                    destination[row.study_uid] = row
                else:
                    if membership.fold != fold or row.study_uid in t15_gold:
                        raise ReportLabelAuditError("Ticket 15 gold reports violate held-out fold isolation")
                    t15_gold[row.study_uid] = row
        eligible_train = set(_array(fold_record.get("eligible_training_report_uids"), "Ticket 15 eligible train IDs"))
        eligible_eval = set(_array(fold_record.get("eligible_evaluation_gold_uids"), "Ticket 15 eligible gold IDs"))
        if set(_array(fold_record.get("fold_training_study_uids"), "Ticket 15 training study IDs")) != set(manifest.training_study_uids(fold)):
            raise ReportLabelAuditError("Ticket 15 training study cohort differs from the manifest")
        if set(_array(fold_record.get("fold_held_out_study_uids"), "Ticket 15 held-out study IDs")) != set(manifest.held_out_study_uids(fold)):
            raise ReportLabelAuditError("Ticket 15 held-out study cohort differs from the manifest")
        expected_excluded = set(manifest.held_out_study_uids(fold))
        if set(_array(fold_record.get("excluded_held_out_report_uids"), "Ticket 15 excluded held-out report IDs")) != expected_excluded:
            raise ReportLabelAuditError("Ticket 15 held-out report exclusion differs from the manifest")
        if eligible_train != set(t15_training[fold]):
            raise ReportLabelAuditError("Ticket 15 training output does not cover its full eligible report cohort")
        actual_gold = {uid for uid, row in t15_gold.items() if row.assigned_fold == fold}
        if eligible_eval != actual_gold:
            raise ReportLabelAuditError("Ticket 15 held-out output does not cover its eligible gold cohort")
        if _array(fold_record.get("missing_training_report_uids"), "Ticket 15 missing train IDs"):
            raise ReportLabelAuditError("Ticket 15 training cohort has missing reports")
        if _array(fold_record.get("missing_evaluation_report_uids"), "Ticket 15 missing gold IDs"):
            raise ReportLabelAuditError("Ticket 15 gold evaluation cohort has missing reports")

    expected_gold_uids = {item.study_instance_uid for item in manifest.gold_memberships}
    expected_report_only_uids = {item.study_instance_uid for item in manifest.report_only_memberships}
    if set(t15_gold) != expected_gold_uids:
        raise ReportLabelAuditError("Ticket 15 held-out report cohort does not match all gold studies")
    for fold in FOLD_IDS:
        expected_train = {
            item.study_instance_uid for item in manifest.report_only_memberships if item.fold != fold
        }
        if set(t15_training[fold]) != expected_train:
            raise ReportLabelAuditError("Ticket 15 training report membership does not match the manifest")

    astra_gold: dict[str, LabelObservation] = {}
    astra_report_only: dict[str, LabelObservation] = {}
    for audit in ticket16_audits:
        for study in audit.studies:
            membership = memberships.get(study.study_instance_uid)
            if membership is None:
                raise ReportLabelAuditError("Ticket 16 output contains a study outside the frozen manifest")
            if membership.cohort != study.cohort or membership.fold != study.assigned_fold:
                raise ReportLabelAuditError("Ticket 16 output cohort or assigned fold differs from manifest")
            observation = _astra_observation(study, membership.reference_labels)
            destination = astra_gold if observation.cohort == "gold" else astra_report_only
            if observation.study_uid in destination:
                raise ReportLabelAuditError("Ticket 16 outputs contain a duplicate study")
            destination[observation.study_uid] = observation
    if set(astra_gold) != expected_gold_uids:
        raise ReportLabelAuditError("Ticket 16 gold output cohort does not match the manifest")
    if set(astra_report_only) != expected_report_only_uids:
        raise ReportLabelAuditError("Ticket 16 report-only output cohort does not match the manifest")

    aggregate_gold_rows = _array(ticket16_private_aggregation.get("private_gold_rows"), "Ticket 16 gold rows")
    if len(aggregate_gold_rows) != len(astra_gold):
        raise ReportLabelAuditError("Ticket 16 private aggregate has incomplete gold rows")
    aggregate_gold_uids: set[str] = set()
    for raw_gold in aggregate_gold_rows:
        aggregate_row = _object(raw_gold, "Ticket 16 aggregate gold row")
        uid = _text(aggregate_row.get("study_instance_uid"), "Ticket 16 gold study identity")
        if uid in aggregate_gold_uids:
            raise ReportLabelAuditError("Ticket 16 aggregate contains duplicate gold rows")
        aggregate_gold_uids.add(uid)
        astra = astra_gold.get(uid)
        t15 = t15_gold.get(uid)
        membership = memberships.get(uid)
        if astra is None or t15 is None or membership is None:
            raise ReportLabelAuditError("Ticket 16 aggregate gold row has no matching fold output")
        if _fold(aggregate_row.get("fold"), "Ticket 16 aggregate gold fold") != membership.fold:
            raise ReportLabelAuditError("Ticket 16 aggregate gold fold differs from manifest")
        if _references(aggregate_row.get("reference_labels"), "Ticket 16 aggregate references") != cast(tuple[int, ...], membership.reference_labels):
            raise ReportLabelAuditError("Ticket 16 aggregate reference labels differ from manifest")
        if _states(aggregate_row.get("astra_states"), "Ticket 16 aggregate Astra states") != astra.states:
            raise ReportLabelAuditError("Ticket 16 aggregate Astra states differ from immutable output")
        if _states(aggregate_row.get("ticket15_states"), "Ticket 16 aggregate Ticket 15 states") != t15.states:
            raise ReportLabelAuditError("Ticket 16 aggregate Ticket 15 states differ from frozen extraction")
        if aggregate_row.get("report_sha256") != astra.report_sha256 or astra.report_sha256 != t15.report_sha256:
            raise ReportLabelAuditError("Ticket 16 output report digest differs across artifacts")
    if aggregate_gold_uids != expected_gold_uids:
        raise ReportLabelAuditError("Ticket 16 aggregate gold rows do not match the manifest")

    for fold in FOLD_IDS:
        t15_ids = set(t15_training[fold])
        astra_ids = {
            uid for uid, row in astra_report_only.items() if fold in row.consuming_folds
        }
        if t15_ids != astra_ids:
            raise ReportLabelAuditError("Ticket 15 and Ticket 16 report-only training cohorts differ")
        for uid in t15_ids:
            if t15_training[fold][uid].report_sha256 != astra_report_only[uid].report_sha256:
                raise ReportLabelAuditError("Ticket 15 and Ticket 16 report-only source digests differ")

    ticket17_private_schema = ticket17_private_supervision.get("schema_version")
    if ticket17_private_schema != "rsna-knee-supervision-admission-v1":
        raise ReportLabelAuditError("Ticket 17 admitted supervision schema is unsupported")
    ticket17_public_schema = ticket17_public_supervision.get("schema_version")
    if ticket17_public_schema != ticket17_private_schema:
        raise ReportLabelAuditError("Ticket 17 public/private admission schemas differ")
    if ticket17_private_supervision.get("fold_manifest_checksum") != manifest.manifest_checksum:
        raise ReportLabelAuditError("Ticket 17 fold identity differs from the canonical manifest")
    t15_reproducibility = _object(ticket15_labels.get("reproducibility"), "Ticket 15 reproducibility")
    if ticket17_private_supervision.get("source_artifact_digest") != t15_reproducibility.get("semantic_digest"):
        raise ReportLabelAuditError("Ticket 17 admission points to another deterministic extraction")
    ticket17_counts = _object(ticket17_private_supervision.get("counts"), "Ticket 17 private counts")
    if ticket17_counts != ticket17_public_supervision.get("counts"):
        raise ReportLabelAuditError("Ticket 17 public admission counts differ from its private artifact")

    source_identity = {
        "fold_manifest_checksum": manifest.manifest_checksum,
        "fold_manifest_file_sha256": ticket04_manifest_sha256,
        "fold_count": len(FOLD_IDS),
        "study_count": len(manifest.memberships),
        "gold_study_count": len(manifest.gold_memberships),
        "report_only_study_count": len(manifest.report_only_memberships),
        "study_level_grouping": not manifest.grouping.reliable,
        "source_split": manifest.source_split,
        "source_version": manifest.source_version,
        "report_set_sha256": cohorts.report_record_set_sha256,
        "report_source_version": cohorts.report_source_version,
        "report_corpus_file_sha256": ticket16_report_corpus_sha256,
    }
    deterministic = LabelSource(
        source_id="ticket15-deterministic",
        identity={
            "extractor_version": t15_extractor.get("version"),
            "configuration_fingerprint": t15_extractor.get("configuration_fingerprint"),
            "ruleset_fingerprint": t15_extractor.get("ruleset_fingerprint"),
            "code_fingerprint": t15_extractor.get("code_fingerprint"),
            "semantic_digest": _object(ticket15_labels.get("reproducibility"), "Ticket 15 reproducibility").get("semantic_digest"),
            "artifact_sha256": ticket15_labels_sha256,
            "repeat_artifact_sha256": ticket15_repeat_labels_sha256,
            "repeat_semantic_digest": _object(ticket15_repeat_labels.get("reproducibility"), "Ticket 15 repeat reproducibility").get("semantic_digest") if ticket15_repeat_labels is not None else None,
            "report_set_sha256": t15_input.get("record_set_sha256"),
            "fold_manifest_checksum": t15_fold_context.get("fold_manifest_checksum"),
            "reproducibility_verified": True,
            "frozen_before_held_out_evaluation": True,
        },
        gold=t15_gold,
        report_only={
            uid: LabelObservation(
                study_uid=uid,
                cohort="report_only",
                assigned_fold=source_row.assigned_fold,
                report_sha256=source_row.report_sha256,
                states=source_row.states,
                consuming_folds=tuple(fold for fold in FOLD_IDS if uid in t15_training[fold]),
            )
            for uid, source_row in _unique_report_rows(t15_training).items()
        },
    )
    learned = LabelSource(
        source_id="ticket16-astra-medium",
        identity={
            "experiment_id": approval.experiment_id,
            "model_name": approval.model_name,
            "model_revision": approval.model_revision,
            "model_identity_fingerprint": approval.model_identity_fingerprint,
            "data_transfer_policy": approval.data_transfer_policy,
            "private_report_execution_approved": approval.private_report_execution_approved,
            "prompt_fingerprint": prompt.fingerprint,
            "prompt_sha256": ticket16_prompt_sha256,
            "prompt_approved": prompt.approved,
            "prompt_frozen": prompt.is_frozen,
            "plan_fingerprint": plan_fingerprint,
            "plan_file_sha256": ticket16_plan_sha256,
            "output_schema_version": ASTRA_OUTPUT_SCHEMA_VERSION,
            "output_validator_code_fingerprint": plan.code_fingerprint,
            "output_artifact_sha256": _digest(ticket16_response_manifest),
            "validated_response_count": sum(audit.status == "validated" for audit in ticket16_audits),
            "first_attempt_validated_count": sum(audit.status == "validated" for audit in ticket16_audits),
            "validation_failure_count": sum(audit.status == "extraction-failed" for audit in ticket16_audits),
            "prompt_tuning_or_adaptation": False,
            "fold_manifest_checksum": manifest.manifest_checksum,
            "report_set_sha256": cohorts.report_record_set_sha256,
            "aggregation_sha256": ticket16_private_aggregation_sha256,
            "public_aggregation_sha256": ticket16_public_aggregation_sha256,
        },
        gold=astra_gold,
        report_only=astra_report_only,
    )

    gates: dict[str, dict[int, dict[tuple[int, str], Gate]]] = {
        deterministic.source_id: {},
        learned.source_id: {},
    }
    for source in (deterministic, learned):
        for fold in FOLD_IDS:
            gates[source.source_id][fold] = _fit_gates(source, fold)

    evaluation: list[dict[str, object]] = []
    report_only_coverage: list[dict[str, object]] = []
    target_status: dict[str, dict[str, object]] = {target: {} for target in TARGET_COLUMNS}
    for fold in FOLD_IDS:
        heldout = tuple(row for row in deterministic.gold.values() if row.assigned_fold == fold)
        if not heldout or {row.study_uid for row in heldout} != {
            row.study_uid for row in learned.gold.values() if row.assigned_fold == fold
        }:
            raise ReportLabelAuditError("fold-specific gold evaluation cohorts differ")
        fold_sources: dict[str, object] = {}
        for source in (deterministic, learned):
            source_gold = tuple(row for row in source.gold.values() if row.assigned_fold == fold)
            source_gates = gates[source.source_id][fold]
            fold_sources[source.source_id] = {
                "policy_fit": _gate_summary(source_gates),
                "held_out_gold": [
                    _target_metrics(source_gold, target_index, source_gates)
                    for target_index in range(NUM_TARGETS)
                ],
            }
        fold_metrics: dict[str, object] = {"fold": fold, "sources": fold_sources}
        paired_gold: list[dict[str, object]] = []
        t15_eval = tuple(row for row in deterministic.gold.values() if row.assigned_fold == fold)
        astra_eval = tuple(row for row in learned.gold.values() if row.assigned_fold == fold)
        for target_index, target in enumerate(TARGET_COLUMNS):
            paired = _paired_coverage(
                t15_eval,
                astra_eval,
                target_index,
                gates[deterministic.source_id][fold],
                gates[learned.source_id][fold],
                with_reference=True,
            )
            paired["target"] = target
            paired_gold.append(paired)
        fold_metrics["paired_held_out_coverage"] = paired_gold
        evaluation.append(fold_metrics)

        training_records = tuple(
            row for row in deterministic.report_only.values() if fold in row.consuming_folds
        )
        learned_records = tuple(
            row for row in learned.report_only.values() if fold in row.consuming_folds
        )
        per_target: list[dict[str, object]] = []
        for target_index, target in enumerate(TARGET_COLUMNS):
            t15_metrics = _report_only_target_metrics(
                training_records,
                target_index,
                gates[deterministic.source_id][fold],
            )
            learned_metrics = _report_only_target_metrics(
                learned_records,
                target_index,
                gates[learned.source_id][fold],
            )
            paired = _paired_coverage(
                training_records,
                learned_records,
                target_index,
                gates[deterministic.source_id][fold],
                gates[learned.source_id][fold],
                with_reference=False,
            )
            per_target.append(
                {
                    "target": target,
                    "ticket15": t15_metrics,
                    "learned": learned_metrics,
                    "coverage_partition": paired,
                }
            )
            t15_accepted = cast(int, t15_metrics["admitted_call_support"])
            learned_accepted = cast(int, learned_metrics["admitted_call_support"])
            target_status[target][str(fold)] = {
                "ticket15_admitted_calls": t15_accepted,
                "learned_admitted_calls": learned_accepted,
                "ticket15_rejected_binary_calls": t15_metrics["rejected_binary_call_support"],
                "learned_rejected_binary_calls": learned_metrics["rejected_binary_call_support"],
                "ticket15_positive_gate": gates[deterministic.source_id][fold][(target_index, LabelState.POSITIVE.value)].admitted,
                "ticket15_negative_gate": gates[deterministic.source_id][fold][(target_index, LabelState.NEGATIVE.value)].admitted,
                "learned_positive_gate": gates[learned.source_id][fold][(target_index, LabelState.POSITIVE.value)].admitted,
                "learned_negative_gate": gates[learned.source_id][fold][(target_index, LabelState.NEGATIVE.value)].admitted,
                "ticket15_target_decision": _target_gate_decision(gates[deterministic.source_id][fold], target_index),
                "learned_target_decision": _target_gate_decision(gates[learned.source_id][fold], target_index),
            }
        report_only_coverage.append(
            {
                "fold": fold,
                "eligible_report_only_studies": len(training_records),
                "description": "reports in this fold's training partition; image readiness is not established here",
                "targets": per_target,
            }
        )
    usable_state_calls: dict[tuple[str, str], int] = {
        (target, state): 0 for target in TARGET_COLUMNS for state in _BINARY_STATES
    }
    for source in (deterministic, learned):
        for fold in FOLD_IDS:
            source_gates = gates[source.source_id][fold]
            for target_index, target in enumerate(TARGET_COLUMNS):
                for state in _BINARY_STATES:
                    if not source_gates[(target_index, state)].admitted:
                        continue
                    usable_state_calls[(target, state)] += sum(
                        row.states[target_index] == state
                        for row in source.report_only.values()
                        if fold in row.consuming_folds
                    )
    coverage_gaps = [
        {"target": target, "state": state}
        for target in TARGET_COLUMNS
        for state in _BINARY_STATES
        if usable_state_calls[(target, state)] == 0
    ]
    cohort_decisions = []
    for fold_record in report_only_coverage:
        fold = cast(int, fold_record["fold"])
        source_decisions = {}
        source_objects = {"ticket15": deterministic, "learned": learned}
        for source_key in ("ticket15", "learned"):
            source = source_objects[source_key]
            source_fold_gates = gates[source.source_id][fold]
            source_rows = tuple(
                row for row in source.report_only.values() if fold in row.consuming_folds
            )
            source_summary = _cohort_admission_summary(source_rows, source_fold_gates)
            source_decisions[source_key] = {
                **source_summary,
                "admission_scope": "target and binary state, evaluated separately for each consuming fold",
            }
        cohort_decisions.append({"fold": fold, "sources": source_decisions})

    private: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "completed_existing_artifact_audit",
        "policy": policy,
        "lineage": {
            "fold_identity": source_identity,
            "ticket15": dict(deterministic.identity),
            "ticket16": dict(learned.identity),
            "ticket17_existing_fold0_admission": {
                "private_artifact_sha256": ticket17_private_sha256,
                "public_artifact_sha256": ticket17_public_sha256,
                "source_artifact_digest": ticket17_private_supervision.get("source_artifact_digest"),
                "fold_manifest_checksum": ticket17_private_supervision.get("fold_manifest_checksum"),
                "counts": ticket17_counts,
                "new_policy_applied_to_this_historical_training_run": False,
            },
            "ticket16_public_aggregation_embedded_and_matched": True,
            "raw_report_text_decoded_or_interpreted": False,
            "existing_source_artifact_bytes_rehashed": True,
            "new_remote_report_processing": False,
            "image_training_performed": False,
            "held_out_outputs_used_in_their_own_fold_policy_fit": False,
        },
        "gate_fits": {
            source.source_id: {
                str(fold): _gate_summary(gates[source.source_id][fold]) for fold in FOLD_IDS
            }
            for source in (deterministic, learned)
        },
        "fold_specific_training_admissions": [
            _make_private_admissions(source, gates[source.source_id])
            for source in (deterministic, learned)
        ],
    }

    public: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "completed_existing_artifact_audit",
        "policy": {
            "schema_version": policy["schema_version"],
            "policy_version": policy["policy_version"],
            "frozen_at_utc": policy["frozen_at_utc"],
            "policy_sha256": policy["policy_sha256"],
            "admission_rule": policy["admission_rule"],
            "confidence_interval": policy["confidence_interval"],
            "states_always_masked": policy["states_always_masked"],
            "held_out_evaluation_used_to_fit_policy": False,
        },
        "provenance": {
            "fold_manifest_checksum": manifest.manifest_checksum,
            "fold_count": len(FOLD_IDS),
            "gold_study_count": len(manifest.gold_memberships),
            "report_only_study_count": len(manifest.report_only_memberships),
            "report_source_version": cohorts.report_source_version,
            "report_set_sha256": cohorts.report_record_set_sha256,
            "ticket15": {
                key: deterministic.identity[key]
                for key in (
                    "extractor_version",
                    "configuration_fingerprint",
                    "ruleset_fingerprint",
                    "code_fingerprint",
                    "semantic_digest",
                    "artifact_sha256",
                    "repeat_artifact_sha256",
                    "repeat_semantic_digest",
                )
            },
            "ticket16": {
                key: learned.identity[key]
                for key in (
                    "experiment_id",
                    "model_name",
                    "model_revision",
                    "model_identity_fingerprint",
                    "prompt_fingerprint",
                    "prompt_sha256",
                    "plan_fingerprint",
                    "output_schema_version",
                    "output_validator_code_fingerprint",
                    "validated_response_count",
                    "validation_failure_count",
                    "output_artifact_sha256",
                    "aggregation_sha256",
                    "public_aggregation_sha256",
                )
            },
            "ticket17_historical_fold0": {
                "private_artifact_sha256": ticket17_private_sha256,
                "source_extraction_semantic_digest": ticket17_private_supervision.get("source_artifact_digest"),
                "counts": ticket17_counts,
                "new_policy_used_in_historical_training": False,
            },
            "raw_report_text_decoded_or_interpreted": False,
            "existing_source_artifact_bytes_rehashed": True,
            "new_remote_report_processing": False,
            "image_training_performed": False,
            "held_out_outputs_used_in_their_own_fold_policy_fit": False,
        },
        "folds": evaluation,
        "report_only_training_coverage": report_only_coverage,
        "target_admission_status": target_status,
        "decisions": {
            "admission_scope": "target, binary state, source, and consuming fold",
            "cohorts": {
                "report_only_training_cohorts": cohort_decisions,
                "gold_reports": "held-out evaluation only; never included in the admitted training artifacts",
            },
            "reference_precedence": True,
            "observable_disagreements_only": True,
            "disagreement_cause_adjudicated": False,
            "local_extractor_branch": {
                "status": "warranted_for_target_state_coverage_gaps" if coverage_gaps else "not_warranted_by_target_state_coverage",
                "target_states_without_usable_report_only_calls": coverage_gaps,
                "basis": "at least one target/binary state has no fold-specific admitted report-only calls across the existing deterministic and learned sources",
                "new_extractor_run": False,
                "approval_required_before_model_or_report_use": True,
            },
            "future_local_extractor_must_use_same_policy_contract": True,
        },
        "limitations": [
            "The 58 gold studies are a small study-level cohort; no reliable patient grouping key was available.",
            "No qualified clinical reviewer adjudicated report/reference disagreements.",
            "Confidence intervals do not account for multiple target/state policy fits or the prior adaptive use of these development folds.",
            "Report-only call coverage does not establish image readiness or measured training benefit.",
            "The existing Ticket 17 fold-0 run used its original deterministic admission policy, not this new cross-fit policy.",
        ],
    }
    known_private_values = [row.study_uid for source in (deterministic, learned) for row in (*source.gold.values(), *source.report_only.values())]
    _ensure_public_safe(public, known_private_values)
    return private, public


def _unique_report_rows(
    rows_by_fold: Mapping[int, Mapping[str, LabelObservation]],
) -> dict[str, LabelObservation]:
    result: dict[str, LabelObservation] = {}
    for fold in FOLD_IDS:
        for uid, row in rows_by_fold[fold].items():
            existing = result.get(uid)
            if existing is not None and (
                existing.states != row.states or existing.report_sha256 != row.report_sha256
            ):
                raise ReportLabelAuditError("deterministic outputs differ across consuming folds")
            result[uid] = row
    return result


def _parse_t15_observation(
    raw_value: object,
    *,
    fold: int,
    assigned_fold: int,
    expected_cohort: str,
    reference_labels: Sequence[int | None] | None,
) -> LabelObservation:
    raw = _object(raw_value, "Ticket 15 report record")
    if raw.get("fold") != fold:
        raise ReportLabelAuditError("Ticket 15 record fold differs from its containing partition")
    expected_kind = "admitted_training_labels" if expected_cohort == "report_only" else "held_out_evaluation"
    if raw.get("artifact_kind") != expected_kind:
        raise ReportLabelAuditError("Ticket 15 record partition kind is inconsistent")
    interpretation = _object(raw.get("interpretation"), "Ticket 15 interpretation")
    if interpretation.get("target_order") != list(TARGET_COLUMNS):
        raise ReportLabelAuditError("Ticket 15 interpretation target order is not canonical")
    if interpretation.get("consuming_folds") != [fold]:
        raise ReportLabelAuditError("Ticket 15 interpretation consuming fold is ambiguous")
    states = _states(interpretation.get("states"), "Ticket 15 states")
    if interpretation.get("report_sha256") != raw.get("report_sha256"):
        raise ReportLabelAuditError("Ticket 15 interpretation report digest differs from its record")
    if interpretation.get("source_report_ref") != raw.get("source_report_ref"):
        raise ReportLabelAuditError("Ticket 15 interpretation report reference differs from its record")
    values = _array(interpretation.get("values"), "Ticket 15 values")
    masks = _array(interpretation.get("supervision_mask"), "Ticket 15 masks")
    if len(values) != NUM_TARGETS or len(masks) != NUM_TARGETS:
        raise ReportLabelAuditError("Ticket 15 values and masks must contain twelve targets")
    labels = _array(interpretation.get("labels"), "Ticket 15 labels")
    if len(labels) != NUM_TARGETS:
        raise ReportLabelAuditError("Ticket 15 label provenance must contain twelve targets")
    for index, (state, value, mask, raw_label) in enumerate(zip(states, values, masks, labels)):
        expected_value = _state_value(state)
        expected_mask = int(expected_value is not None)
        if value != expected_value or mask != expected_mask:
            raise ReportLabelAuditError(f"Ticket 15 state/value/mask mismatch at target {index}")
        label = _object(raw_label, "Ticket 15 target provenance")
        if label.get("target") != TARGET_COLUMNS[index] or label.get("state") != state:
            raise ReportLabelAuditError("Ticket 15 label provenance target order differs")
        if expected_mask:
            origin = _object(label.get("rule_origin"), "Ticket 15 rule origin")
            if (
                label.get("confidence") != 1.0
                or origin.get("kind") != "official-competition-definition"
                or origin.get("derived_from_report") is not False
                or origin.get("participant_interpretation") is not False
                or origin.get("shared_across_folds") is not True
            ):
                raise ReportLabelAuditError("Ticket 15 binary call lacks authoritative frozen rule provenance")
    consumers: tuple[int, ...]
    if expected_cohort == "gold":
        if reference_labels is None:
            raise ReportLabelAuditError("Ticket 15 gold observation lacks manifest references")
        refs = _references(list(reference_labels), "Ticket 15 gold references")
        consumers = (assigned_fold,)
    else:
        refs = None
        consumers = tuple(candidate for candidate in FOLD_IDS if candidate != assigned_fold)
    return LabelObservation(
        study_uid=_text(raw.get("study_instance_uid"), "Ticket 15 study identity"),
        cohort=expected_cohort,
        assigned_fold=assigned_fold,
        report_sha256=_text(raw.get("report_sha256"), "Ticket 15 report digest"),
        states=states,
        consuming_folds=consumers,
        reference_labels=refs,
    )


def _astra_observation(
    study: AstraValidatedStudy,
    reference_labels: Sequence[int | None],
) -> LabelObservation:
    states = tuple(item.state.value for item in study.labels)
    if tuple(item.target for item in study.labels) != TARGET_COLUMNS:
        raise ReportLabelAuditError("Astra target order is not canonical")
    cohort = study.cohort
    refs: tuple[int, ...] | None = None
    if cohort == "gold":
        refs = _references(list(reference_labels), "Astra gold references")
    return LabelObservation(
        study_uid=study.study_instance_uid,
        cohort=cohort,
        assigned_fold=study.assigned_fold,
        report_sha256=study.report_sha256,
        states=states,
        consuming_folds=study.consuming_folds,
        reference_labels=refs,
    )


def load_existing_artifacts(
    *,
    ticket15_labels_path: str | Path,
    ticket15_repeat_labels_path: str | Path | None,
    ticket16_plan_path: str | Path,
    ticket16_private_dir: str | Path,
    ticket16_private_aggregation_path: str | Path,
    ticket16_public_aggregation_path: str | Path,
    ticket16_response_dir: str | Path,
    ticket17_private_supervision_path: str | Path,
    ticket17_public_supervision_path: str | Path,
    fold_manifest_path: str | Path,
) -> dict[str, object]:
    """Read only stored output artifacts, never raw report text or responses."""

    manifest_path = Path(fold_manifest_path)
    manifest = load_fold_manifest(manifest_path)
    ticket15_path = Path(ticket15_labels_path)
    ticket15 = _json_file(ticket15_path, "Ticket 15 labels")
    ticket15_repeat_path = Path(ticket15_repeat_labels_path) if ticket15_repeat_labels_path is not None else None
    ticket15_repeat = _json_file(ticket15_repeat_path, "Ticket 15 repeat labels") if ticket15_repeat_path else None
    plan_path = Path(ticket16_plan_path)
    plan = load_astra_batch_plan(plan_path)
    plan_metadata = _json_file(plan_path, "Ticket 16 plan metadata")
    private_dir = Path(ticket16_private_dir)
    audits = load_astra_audits(plan, private_dir)
    private_aggregation_path = Path(ticket16_private_aggregation_path)
    private_aggregation = _json_file(private_aggregation_path, "Ticket 16 private aggregation")
    public_aggregation_path = Path(ticket16_public_aggregation_path)
    public_aggregation = _json_file(public_aggregation_path, "Ticket 16 public aggregation")
    ticket17_private_path = Path(ticket17_private_supervision_path)
    ticket17_private = _json_file(ticket17_private_path, "Ticket 17 private supervision")
    ticket17_public_path = Path(ticket17_public_supervision_path)
    ticket17_public = _json_file(ticket17_public_path, "Ticket 17 public supervision")
    prompt_path = private_dir / "approved-prompt.json"
    if not prompt_path.is_file():
        raise ReportLabelAuditError("approved Ticket 16 prompt artifact is missing")
    from .astra_handoff import load_astra_prompt

    prompt = load_astra_prompt(prompt_path)
    response_root = Path(ticket16_response_dir)
    response_paths = sorted(response_root.glob("*.json"))
    expected_audits = {audit.batch_id: audit for audit in audits}
    expected_names = {f"{batch_id}.json" for batch_id in expected_audits}
    if {path.name for path in response_paths} != expected_names:
        raise ReportLabelAuditError("Ticket 16 response directory differs from the audited batch set")
    response_rows = []
    for path in response_paths:
        batch_id = path.stem
        audit = expected_audits[batch_id]
        raw_hash = file_sha256(path)
        if raw_hash != audit.raw_output_sha256:
            raise ReportLabelAuditError("Ticket 16 response hash differs from its immutable validation audit")
        response_rows.append(
            {
                "batch_id": batch_id,
                "raw_output_sha256": raw_hash,
                "semantic_output_digest": audit.semantic_output_digest,
                "input_digest": audit.input_digest,
                "study_count": len(audit.studies),
            }
        )
    response_manifest = {
        "batch_count": len(expected_audits),
        "validated_response_count": len(response_rows),
        "responses": sorted(response_rows, key=lambda item: cast(str, item["batch_id"])),
    }
    aggregate_inputs = _object(private_aggregation.get("private_inputs"), "Ticket 16 private inputs")
    expected_t15_path = Path(_text(aggregate_inputs.get("ticket15_labels_path"), "Ticket 16 Ticket 15 path"))
    expected_manifest_path = Path(_text(aggregate_inputs.get("fold_manifest_path"), "Ticket 16 manifest path"))
    expected_plan_path = Path(_text(aggregate_inputs.get("private_plan_path"), "Ticket 16 plan path"))
    expected_response_dir = Path(_text(aggregate_inputs.get("response_directory"), "Ticket 16 response path"))
    expected_report_corpus = Path(_text(aggregate_inputs.get("report_corpus_path"), "Ticket 16 report corpus path"))
    if expected_t15_path.resolve() != ticket15_path.resolve():
        raise ReportLabelAuditError("Ticket 16 aggregation references another Ticket 15 artifact")
    if expected_manifest_path.resolve() != manifest_path.resolve():
        raise ReportLabelAuditError("Ticket 16 aggregation references another fold manifest")
    if expected_plan_path.resolve() != plan_path.resolve():
        raise ReportLabelAuditError("Ticket 16 aggregation references another plan")
    if expected_response_dir.resolve() != response_root.resolve():
        raise ReportLabelAuditError("Ticket 16 aggregation references another response directory")
    if not expected_report_corpus.is_file():
        raise ReportLabelAuditError("Ticket 16 source report artifact is unavailable")
    ticket15_public_path = ticket15_path.parent.parent / "public" / "report-evidence.json"
    ticket17_public_path = Path(ticket17_public_supervision_path)
    prompt_sha = file_sha256(prompt_path)
    return {
        "manifest": manifest,
        "manifest_sha256": file_sha256(manifest_path),
        "ticket15_labels": ticket15,
        "ticket15_labels_sha256": file_sha256(ticket15_path),
        "ticket15_repeat_labels": ticket15_repeat,
        "ticket15_repeat_labels_sha256": file_sha256(ticket15_repeat_path) if ticket15_repeat_path else None,
        "ticket15_public_sha256": file_sha256(ticket15_public_path) if ticket15_public_path.is_file() else None,
        "ticket16_plan": plan,
        "ticket16_plan_metadata": plan_metadata,
        "ticket16_plan_sha256": file_sha256(plan_path),
        "ticket16_audits": audits,
        "ticket16_private_aggregation": private_aggregation,
        "ticket16_public_aggregation": public_aggregation,
        "ticket16_private_aggregation_sha256": file_sha256(private_aggregation_path),
        "ticket16_public_aggregation_sha256": file_sha256(public_aggregation_path),
        "ticket16_prompt_sha256": prompt_sha,
        "ticket16_response_manifest": response_manifest,
        "ticket16_report_corpus_sha256": file_sha256(expected_report_corpus),
        "ticket17_private_supervision": ticket17_private,
        "ticket17_public_supervision": ticket17_public,
        "ticket17_private_sha256": file_sha256(ticket17_private_path),
        "ticket17_public_sha256": file_sha256(ticket17_public_path),
    }


def write_immutable_json(path: str | Path, payload: object, *, mode: int = 0o600) -> Path:
    """Create a JSON artifact once, with restrictive default permissions."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(destination, flags, mode)
    except FileExistsError as exc:
        raise ReportLabelAuditError(f"immutable artifact already exists: {destination}") from exc
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return destination


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
