from __future__ import annotations

import pytest

from rsna_knee.astra_aggregation import (
    _TargetStats,
    _common_target_macro_selective_accuracy,
    _disagreement_analysis,
    _metric_summary,
)
from rsna_knee.report_labels import LabelState
from rsna_knee.targets import TARGET_COLUMNS


def test_target_stats_use_only_binary_states_for_supervision_and_metrics() -> None:
    stats = _TargetStats("ACL")
    stats.add(LabelState.POSITIVE.value, 1)
    stats.add(LabelState.NEGATIVE.value, 0)
    stats.add(LabelState.POSITIVE.value, 0)
    stats.add(LabelState.NEGATIVE.value, 1)
    stats.add(LabelState.UNKNOWN.value, 1)

    payload = stats.to_gold_dict()

    assert payload["evaluable_gold_count"] == 5
    assert payload["supervised_count"] == 4
    assert payload["supervised_coverage"] == pytest.approx(0.8)
    assert payload["correct_positive_calls"] == 1
    assert payload["correct_negative_calls"] == 1
    assert payload["false_positive_count"] == 1
    assert payload["false_negative_count"] == 1
    assert payload["selective_tp"] == 1
    assert payload["selective_tn"] == 1
    assert payload["selective_fp"] == 1
    assert payload["selective_fn"] == 1
    assert payload["selective_sensitivity"] == pytest.approx(0.5)
    assert payload["selective_specificity"] == pytest.approx(0.5)
    assert payload["selective_ppv"] == pytest.approx(0.5)
    assert payload["selective_npv"] == pytest.approx(0.5)
    assert payload["selective_balanced_accuracy"] == pytest.approx(0.5)
    assert payload["positive_reference_capture"] == pytest.approx(1 / 3)
    assert payload["negative_reference_capture"] == pytest.approx(0.5)
    assert "sensitivity" not in payload
    assert "specificity" not in payload


def test_target_stats_explain_undefined_metrics_without_inventing_values() -> None:
    stats = _TargetStats("Medial OA")
    stats.add(LabelState.UNKNOWN.value, 1)
    stats.add(LabelState.NOT_MENTIONED.value, 0)

    payload = stats.to_gold_dict()

    assert payload["selective_accuracy"] is None
    assert payload["selective_sensitivity"] is None
    assert payload["selective_specificity"] is None
    assert payload["selective_ppv"] is None
    assert payload["selective_npv"] is None
    assert payload["selective_balanced_accuracy"] is None
    assert payload["positive_reference_capture"] == 0
    assert payload["negative_reference_capture"] == 0
    assert payload["undefined_metrics"]["selective_accuracy"]
    assert payload["undefined_metrics"]["selective_sensitivity"]
    assert payload["undefined_metrics"]["selective_specificity"]
    assert payload["undefined_metrics"]["selective_ppv"]
    assert payload["undefined_metrics"]["selective_npv"]


def test_reference_capture_does_not_count_abstentions_as_selective_sensitivity() -> None:
    stats = _TargetStats("ACL")
    stats.add(LabelState.UNKNOWN.value, 1)
    stats.add(LabelState.POSITIVE.value, 1)
    stats.add(LabelState.NEGATIVE.value, 0)
    stats.add(LabelState.NOT_MENTIONED.value, 0)

    payload = stats.to_gold_dict()

    assert payload["selective_sensitivity"] == 1.0
    assert payload["positive_reference_capture"] == pytest.approx(0.5)
    assert payload["selective_specificity"] == 1.0
    assert payload["negative_reference_capture"] == pytest.approx(0.5)


def test_common_target_macro_uses_one_identical_target_set() -> None:
    astra_stats = []
    ticket15_stats = []
    for target in TARGET_COLUMNS:
        astra = _TargetStats(target)
        astra.add(LabelState.POSITIVE.value, 1)
        astra_stats.append(astra)

        ticket15 = _TargetStats(target)
        if target != "Medial OA":
            ticket15.add(LabelState.POSITIVE.value, 1)
        else:
            ticket15.add(LabelState.UNKNOWN.value, 1)
        ticket15_stats.append(ticket15)

    astra_summary = _metric_summary(astra_stats)
    ticket15_summary = _metric_summary(ticket15_stats)
    common = _common_target_macro_selective_accuracy(astra_summary, ticket15_summary)

    assert astra_summary["defined_selective_accuracy_targets"] == list(TARGET_COLUMNS)
    assert ticket15_summary["defined_selective_accuracy_targets"] == [
        target for target in TARGET_COLUMNS if target != "Medial OA"
    ]
    assert common["target_set"] == ticket15_summary["defined_selective_accuracy_targets"]
    assert common["target_count"] == 11
    assert common["same_target_set_for_both_extractors"] is True
    assert common["astra"] == pytest.approx(1.0)
    assert common["ticket15"] == pytest.approx(1.0)


def test_metric_summary_reports_micro_numerator_and_denominator() -> None:
    stats = []
    for target in TARGET_COLUMNS:
        item = _TargetStats(target)
        item.add(LabelState.POSITIVE.value, 1)
        item.add(LabelState.POSITIVE.value, 0)
        stats.append(item)

    summary = _metric_summary(stats)

    assert summary["micro_selective_accuracy"] == {
        "correct_supervised_calls": 12,
        "supervised_calls": 24,
        "value": pytest.approx(0.5),
        "undefined_reason": None,
    }


def test_disagreement_analysis_reports_only_the_supervision_relationships() -> None:
    uid = "gold-1"
    astra = {uid: (LabelState.POSITIVE.value,) + (LabelState.UNKNOWN.value,) * 11}
    ticket15 = {uid: (LabelState.NEGATIVE.value,) + (LabelState.UNKNOWN.value,) * 11}
    references = {uid: (1,) + (0,) * 11}

    public, private = _disagreement_analysis(astra, ticket15, references, (uid,))

    first = public[0]
    assert first["both_supervised_and_disagree"] == 1
    assert first["gold_truth_in_disagreement_cases"][
        "astra_correct_in_both_supervised_disagreement"
    ] == 1
    assert first["gold_truth_in_disagreement_cases"][
        "ticket15_correct_in_both_supervised_disagreement"
    ] == 0
    assert private[0]["study_instance_uid"] == uid
    assert len(public) == len(TARGET_COLUMNS)
