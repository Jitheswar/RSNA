from __future__ import annotations

import pytest

from rsna_knee.report_extraction import ReportLabelRecord, ReportPartition
from rsna_knee.report_label_audit import (
    LabelObservation,
    LabelSource,
    Gate,
    ReportLabelAuditError,
    _cohort_admission_summary,
    _fit_gates,
    _paired_coverage,
    _states,
    _target_metrics,
    make_policy_payload,
    validate_frozen_policy,
    wilson_interval,
)
from rsna_knee.report_labels import interpret_report
from rsna_knee.supervision import merge_reference_and_weak_labels
from rsna_knee.targets import NUM_TARGETS, TARGET_COLUMNS


def _gold_row(uid: str, fold: int, reference: int, state: str) -> LabelObservation:
    return LabelObservation(
        study_uid=uid,
        cohort="gold",
        assigned_fold=fold,
        report_sha256=(str(fold + 1) * 64),
        states=(state,) + ("unknown",) * (NUM_TARGETS - 1),
        consuming_folds=(fold,),
        reference_labels=(reference,) + (0,) * (NUM_TARGETS - 1),
    )


def test_frozen_policy_is_content_bound_and_contains_no_evaluation_rows() -> None:
    policy = make_policy_payload(frozen_at_utc="2026-09-25T00:00:00Z")

    assert validate_frozen_policy(policy) == policy
    assert "observations" not in policy
    tampered = dict(policy)
    tampered["admission_rule"] = "fit on all folds"
    with pytest.raises(ReportLabelAuditError, match="checksum"):
        validate_frozen_policy(tampered)


def test_policy_fit_uses_other_folds_and_ignores_the_consuming_holdout() -> None:
    rows = {}
    for index in range(10):
        row = _gold_row(f"train-positive-{index}", 1, 1, "positive")
        rows[row.study_uid] = row
    for index in range(10):
        row = _gold_row(f"train-negative-{index}", 2, 0, "negative")
        rows[row.study_uid] = row
    for index in range(12):
        row = _gold_row(f"heldout-{index}", 0, 0, "positive")
        rows[row.study_uid] = row
    source = LabelSource("synthetic", {}, rows, {})

    gates = _fit_gates(source, 0)
    positive = gates[(0, "positive")]
    negative = gates[(0, "negative")]

    assert positive.calibration_support == 20
    assert positive.call_support == 10
    assert positive.class_prevalence == 0.5
    assert positive.admitted is True
    assert negative.call_support == 10
    assert negative.admitted is True


def test_gate_rejects_a_state_when_its_lower_bound_does_not_exceed_class_prevalence() -> None:
    rows = {}
    for index in range(20):
        row = _gold_row(f"positive-{index}", 1 if index < 10 else 2, 1, "positive")
        rows[row.study_uid] = row
    source = LabelSource("synthetic", {}, rows, {})

    gate = _fit_gates(source, 0)[(0, "positive")]

    assert gate.class_prevalence == 1.0
    assert gate.lower_95 is not None and gate.lower_95 < 1.0
    assert gate.admitted is False
    assert gate.reason == "lower_bound_not_above_training_side_prevalence"


def test_paired_coverage_separates_new_calls_from_shared_admitted_calls() -> None:
    gold = {
        row.study_uid: row
        for row in (
            *(_gold_row(f"positive-{index}", 1, 1, "positive") for index in range(10)),
            *(_gold_row(f"negative-{index}", 2, 0, "negative") for index in range(10)),
        )
    }
    t15_source = LabelSource("t15", {}, gold, {})
    learned_source = LabelSource("learned", {}, gold, {})
    consumers = (0, 1)

    def report_row(uid: str, state: str) -> LabelObservation:
        return LabelObservation(
            study_uid=uid,
            cohort="report_only",
            assigned_fold=2,
            report_sha256="f" * 64,
            states=(state,) + ("unknown",) * (NUM_TARGETS - 1),
            consuming_folds=consumers,
        )

    t15_rows = {
        row.study_uid: row
        for row in (
            report_row("shared", "positive"),
            report_row("t15-only", "negative"),
            report_row("learned-only", "unknown"),
        )
    }
    learned_rows = {
        row.study_uid: row
        for row in (
            report_row("shared", "positive"),
            report_row("t15-only", "unknown"),
            report_row("learned-only", "negative"),
        )
    }
    t15_gates = _fit_gates(t15_source, 0)
    learned_gates = _fit_gates(learned_source, 0)

    paired = _paired_coverage(
        tuple(t15_rows.values()),
        tuple(learned_rows.values()),
        0,
        t15_gates,
        learned_gates,
        with_reference=False,
    )

    assert paired["ticket15_only_calls"] == 1
    assert paired["learned_only_new_calls"] == 1
    assert paired["shared_calls"] == 1
    assert paired["learned_only_admitted_calls"] == 1
    assert paired["shared_admitted_by_both_calls"] == 1
    assert paired["shared_admitted_by_either_calls"] == 1
    assert paired["learned_only_reference_reliability_95_ci"] is None
    assert paired["shared_learned_reference_support"] is None
    assert paired["shared_learned_reference_reliability_95_ci"] is None


def test_paired_gold_metrics_include_wilson_intervals_for_new_and_shared_calls() -> None:
    def gold_pair(uid: str, reference: int, ticket15_state: str, learned_state: str) -> tuple[LabelObservation, LabelObservation]:
        def row(state: str) -> LabelObservation:
            return LabelObservation(
                study_uid=uid,
                cohort="gold",
                assigned_fold=0,
                report_sha256="d" * 64,
                states=(state,) + ("unknown",) * (NUM_TARGETS - 1),
                consuming_folds=(0,),
                reference_labels=(reference,) + (0,) * (NUM_TARGETS - 1),
            )

        return row(ticket15_state), row(learned_state)

    pairs = (
        gold_pair("learned-new-correct", 1, "unknown", "positive"),
        gold_pair("learned-new-wrong", 0, "not-mentioned", "positive"),
        gold_pair("shared-correct", 1, "positive", "negative"),
        gold_pair("shared-mixed", 0, "negative", "negative"),
        gold_pair("ticket15-only", 0, "negative", "unknown"),
    )
    ticket15_gates = {
        (0, "positive"): Gate("ACL", "positive", 20, 10, 10, 10, 1.0, 0.8, 1.0, 0.5, True, "admitted"),
        (0, "negative"): Gate("ACL", "negative", 20, 10, 10, 10, 1.0, 0.8, 1.0, 0.5, True, "admitted"),
    }
    learned_gates = {
        (0, "positive"): Gate("ACL", "positive", 20, 10, 10, 10, 1.0, 0.8, 1.0, 0.5, True, "admitted"),
        (0, "negative"): Gate("ACL", "negative", 20, 10, 10, 10, 1.0, 0.8, 1.0, 0.5, True, "admitted"),
    }

    paired = _paired_coverage(
        tuple(left for left, _ in pairs),
        tuple(right for _, right in pairs),
        0,
        ticket15_gates,
        learned_gates,
        with_reference=True,
    )

    assert paired["learned_only_new_calls"] == 2
    assert paired["learned_only_reference_correct"] == 1
    assert paired["learned_only_reference_support"] == 2
    assert paired["learned_only_reference_reliability"] == 0.5
    assert paired["learned_only_reference_reliability_95_ci"] == list(wilson_interval(1, 2))
    assert paired["shared_calls"] == 2
    assert paired["shared_ticket15_reference_correct"] == 2
    assert paired["shared_ticket15_reference_support"] == 2
    assert paired["shared_ticket15_reference_reliability"] == 1.0
    assert paired["shared_ticket15_reference_reliability_95_ci"] == list(wilson_interval(2, 2))
    assert paired["shared_learned_reference_correct"] == 1
    assert paired["shared_learned_reference_support"] == 2
    assert paired["shared_learned_reference_reliability"] == 0.5
    assert paired["shared_learned_reference_reliability_95_ci"] == list(wilson_interval(1, 2))
    assert paired["ticket15_only_reference_correct"] == 1
    assert paired["ticket15_only_reference_support"] == 1
    assert paired["ticket15_only_reference_reliability_95_ci"] == list(wilson_interval(1, 1))


def test_heldout_metrics_separate_raw_and_policy_admitted_reliability() -> None:
    rows = (
        _gold_row("positive", 0, 1, "positive"),
        _gold_row("negative", 0, 0, "negative"),
    )
    gates = {
        (0, "positive"): Gate("ACL", "positive", 20, 10, 10, 10, 1.0, 0.8, 1.0, 0.5, True, "admitted"),
        (0, "negative"): Gate("ACL", "negative", 20, 10, 10, 8, 0.8, 0.6, 0.95, 0.5, False, "rejected"),
    }

    metrics = _target_metrics(rows, 0, gates)

    assert metrics["positive_reliability"] == 1.0
    assert metrics["positive_call_support"] == 1
    assert metrics["positive_state_admitted_by_policy"] is True
    assert metrics["admitted_positive_reliability"] == 1.0
    assert metrics["admitted_positive_call_support"] == 1
    assert metrics["negative_reliability"] == 1.0
    assert metrics["negative_call_support"] == 1
    assert metrics["negative_state_admitted_by_policy"] is False
    assert metrics["admitted_negative_reliability"] is None
    assert metrics["admitted_negative_call_support"] == 0


def test_cohort_aggregate_counts_unique_studies_not_target_cells() -> None:
    row_with_two_admitted_targets = LabelObservation(
        study_uid="two-admitted-targets",
        cohort="report_only",
        assigned_fold=2,
        report_sha256="a" * 64,
        states=("positive", "negative", "unknown") + ("unknown",) * (NUM_TARGETS - 3),
        consuming_folds=(0, 1),
    )
    row_with_one_rejected_target = LabelObservation(
        study_uid="one-rejected-target",
        cohort="report_only",
        assigned_fold=2,
        report_sha256="b" * 64,
        states=("unknown", "unknown", "positive") + ("unknown",) * (NUM_TARGETS - 3),
        consuming_folds=(0, 1),
    )
    gates = {
        (0, "positive"): Gate("ACL", "positive", 20, 10, 10, 10, 1.0, 0.8, 1.0, 0.5, True, "admitted"),
        (1, "negative"): Gate("MCL", "negative", 20, 10, 10, 10, 1.0, 0.8, 1.0, 0.5, True, "admitted"),
        (2, "positive"): Gate("Medial Meniscus", "positive", 20, 10, 10, 4, 0.4, 0.2, 0.6, 0.5, False, "rejected"),
    }

    summary = _cohort_admission_summary(
        (row_with_two_admitted_targets, row_with_one_rejected_target),
        gates,
    )

    assert summary == {
        "eligible_report_only_studies": 2,
        "admitted_studies_with_any_call": 1,
        "admitted_target_calls": 2,
        "rejected_binary_calls": 1,
    }


def test_malformed_states_and_leaking_report_only_fold_are_rejected() -> None:
    with pytest.raises(ReportLabelAuditError, match="exactly 12"):
        _states(["unknown"] * (NUM_TARGETS - 1), "bad vector")
    with pytest.raises(ReportLabelAuditError, match="unsupported state"):
        _states(["unknown"] * (NUM_TARGETS - 1) + ["not-a-state"], "bad vector")
    with pytest.raises(ReportLabelAuditError, match="exclude their assigned fold"):
        LabelObservation(
            study_uid="leaked",
            cohort="report_only",
            assigned_fold=0,
            report_sha256="a" * 64,
            states=("positive",) + ("unknown",) * (NUM_TARGETS - 1),
            consuming_folds=(0, 1),
        )


def test_existing_supervision_boundary_keeps_reference_precedence() -> None:
    interpretation = interpret_report(
        "A high-grade partial tear of the ACL is present.",
        source_report_ref="private://synthetic-report",
        consuming_folds=(0,),
    )
    weak = ReportLabelRecord(
        study_instance_uid="synthetic-gold",
        fold=0,
        partition=ReportPartition.TRAINING_LABELS,
        interpretation=interpretation,
    )
    reference = (0,) + (None,) * (NUM_TARGETS - 1)

    merged = merge_reference_and_weak_labels(reference, weak)

    assert merged.targets[0] == 0
    assert merged.supervision_mask[0] == 1
    assert merged.disagreements[0].target == TARGET_COLUMNS[0]
    assert merged.disagreements[0].reference_value == 0
    assert merged.disagreements[0].weak_value == 1
