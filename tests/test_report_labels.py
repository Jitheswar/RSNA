from __future__ import annotations

import pytest

from rsna_knee.report_labels import (
    LabelState,
    ReportInterpretationError,
    RuleOrigin,
    RuleOriginKind,
    interpret_report,
    target_definitions,
    validate_rule_origin_for_folds,
)
from rsna_knee.targets import TARGET_COLUMNS


def test_authority_register_covers_every_competition_target() -> None:
    definitions = target_definitions()

    assert tuple(item.target for item in definitions) == TARGET_COLUMNS
    assert all(item.sources for item in definitions)
    assert all(item.positive_criterion for item in definitions)
    assert all(item.negative_criterion for item in definitions)
    assert all(
        source.retrieved_on == "2026-09-14"
        for item in definitions
        for source in item.sources
    )


def test_explicit_positive_and_negative_report_evidence_is_supervisable() -> None:
    interpretation = interpret_report(
        "A high-grade partial tear of the ACL is present. The MCL is intact.",
        source_report_ref="synthetic-positive-negative-01",
        consuming_folds=(0, 1, 2),
    )

    labels = {item.target: item for item in interpretation.labels}

    assert labels["ACL"].state is LabelState.POSITIVE
    assert labels["ACL"].value == 1
    assert labels["ACL"].supervision_mask == 1
    assert labels["MCL"].state is LabelState.NEGATIVE
    assert labels["MCL"].value == 0
    assert labels["MCL"].supervision_mask == 1
    assert labels["Medial Meniscus"].state is LabelState.NOT_MENTIONED
    assert labels["Medial Meniscus"].supervision_mask == 0
    assert interpretation.source_report_ref == "synthetic-positive-negative-01"


@pytest.mark.parametrize(
    ("report", "target"),
    [
        ("There is a possible high-grade ACL tear.", "ACL"),
        ("There is a history of an ACL tear.", "ACL"),
        ("There is an osteochondral fracture.", "Fracture"),
    ],
)
def test_uncertain_historical_and_unsupported_findings_remain_masked(
    report: str, target: str
) -> None:
    interpretation = interpret_report(report, source_report_ref="synthetic-unknown-01")
    label = next(item for item in interpretation.labels if item.target == target)

    assert label.state in {LabelState.UNCERTAIN, LabelState.UNKNOWN}
    assert label.value is None
    assert label.supervision_mask == 0


def test_empty_report_is_an_extraction_failure_not_a_negative() -> None:
    interpretation = interpret_report("", source_report_ref="synthetic-empty-01")

    assert all(item.state is LabelState.EXTRACTION_FAILED for item in interpretation.labels)
    assert all(item.supervision_mask == 0 for item in interpretation.labels)


def test_report_derived_rule_cannot_be_shared_across_folds() -> None:
    origin = RuleOrigin(
        rule_id="participant-rule-01",
        kind=RuleOriginKind.PATIENT_REPORT,
        source_id="patient-report-example-01",
        source_url="private://report-example-01",
        retrieved_on="2026-09-14",
        derived_from_report=True,
        shared_across_folds=True,
    )

    with pytest.raises(ReportInterpretationError, match="report-derived"):
        validate_rule_origin_for_folds(origin, (0, 1, 2))


@pytest.mark.parametrize(
    ("target", "report"),
    [
        ("ACL", "MRI shows a high-grade partial tear of the anterior cruciate ligament."),
        ("MCL", "There is a complete acute tear of the medial collateral ligament."),
        ("Medial Meniscus", "A medial meniscus tear reaches the articular surface on two images."),
        ("Lateral Meniscus", "A lateral meniscus tear reaches the articular surface on two images."),
        ("Medial OA", "Moderate medial compartment osteoarthritis with high-grade cartilage loss."),
        ("Lateral OA", "Large lateral tibiofemoral compartment osteoarthritis."),
        ("PF OA", "There is moderate patellofemoral compartment osteoarthritis."),
        ("Effusion", "A large joint effusion distends the joint."),
        ("Synovitis", "The synovial lining is thickened, consistent with synovitis."),
        ("Baker's", "A large popliteal cyst is present."),
        (
            "Contusion",
            "Bone marrow edema-like signal from impact represents a bone contusion, without a discrete fracture line.",
        ),
        ("Fracture", "An acute fracture line is visible in the tibial cortex."),
    ],
)
def test_each_target_requires_its_authoritative_positive_criterion(target: str, report: str) -> None:
    interpretation = interpret_report(report, source_report_ref=f"synthetic-positive-{target}")
    label = next(item for item in interpretation.labels if item.target == target)

    assert label.state is LabelState.POSITIVE
    assert label.value == 1
    assert label.supervision_mask == 1


@pytest.mark.parametrize(
    ("target", "report"),
    [
        ("ACL", "The anterior cruciate ligament is intact."),
        ("MCL", "Only a low-grade sprain of the medial collateral ligament is present."),
        ("Medial Meniscus", "Medial meniscus intrasubstance degeneration does not reach the surface."),
        ("Lateral Meniscus", "Lateral meniscus intrasubstance degeneration does not reach the surface."),
        ("Medial OA", "Only mild medial compartment cartilage wear is present."),
        ("Lateral OA", "Only mild lateral compartment cartilage wear is present."),
        ("PF OA", "Mild patellofemoral chondropathy only."),
        ("Effusion", "Only a trace joint effusion is seen."),
        ("Synovitis", "The synovial lining is normal and not thickened."),
        ("Baker's", "There is only a small Baker's cyst."),
        ("Contusion", "No bone contusion is identified."),
        ("Fracture", "No acute fracture is identified."),
    ],
)
def test_each_target_requires_explicit_negative_criterion(target: str, report: str) -> None:
    interpretation = interpret_report(report, source_report_ref=f"synthetic-negative-{target}")
    label = next(item for item in interpretation.labels if item.target == target)

    assert label.state is LabelState.NEGATIVE
    assert label.value == 0
    assert label.supervision_mask == 1


def test_conflicting_target_evidence_is_failed_and_masked() -> None:
    interpretation = interpret_report(
        "The ACL is intact. A high-grade partial tear of the ACL is also described.",
        source_report_ref="synthetic-conflict-01",
    )
    label = next(item for item in interpretation.labels if item.target == "ACL")

    assert label.state is LabelState.EXTRACTION_FAILED
    assert label.value is None
    assert label.supervision_mask == 0


def test_public_interpretation_evidence_does_not_include_raw_report_or_private_reference() -> None:
    report = "A high-grade partial tear of the ACL is present."
    interpretation = interpret_report(report, source_report_ref="private-report-ref")
    public = interpretation.to_public_dict()

    assert "private-report-ref" not in str(public)
    assert report not in str(public)
    assert interpretation.report_sha256
    assert interpretation.to_private_dict()["source_report_ref"] == "private-report-ref"


def test_fold_scoped_report_rule_is_allowed_only_for_its_declared_fold() -> None:
    origin = RuleOrigin(
        rule_id="fold-rule-01",
        kind=RuleOriginKind.PATIENT_REPORT,
        source_id="patient-report-example-01",
        source_url="private://report-example-01",
        retrieved_on="2026-09-14",
        derived_from_report=True,
        shared_across_folds=False,
        allowed_folds=(1,),
        source_report_refs=("report-example-01",),
    )

    validate_rule_origin_for_folds(origin, (1,))
    with pytest.raises(ReportInterpretationError, match="not clean"):
        validate_rule_origin_for_folds(origin, (0,))
