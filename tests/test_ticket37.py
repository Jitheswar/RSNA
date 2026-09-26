from __future__ import annotations

from dataclasses import replace
import json
import stat
from typing import Mapping, cast

import pytest

from rsna_knee.folds import FoldMembership, FoldManifest, freeze_folds
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.report_extraction import ReportCorpus, ReportRecord
from rsna_knee.report_label_audit import ReportLabelAuditError, make_policy_payload
from rsna_knee.targets import TARGET_COLUMNS
from rsna_knee.ticket37 import (
    TICKET37_OUTPUT_SCHEMA_VERSION,
    Ticket37Error,
    Ticket37Observation,
    build_ticket37_admission,
    compare_ticket37_admissions_with_ticket30,
    freeze_ticket37_prompt,
    load_ticket37_approval,
    make_ticket37_observation,
    merge_ticket37_reference_labels,
    parse_ticket37_output,
    plan_ticket37_sample,
    write_ticket37_json,
    ticket37_approval_record,
)


def _model_output(*, first_state: str = "positive", first_evidence: str = "") -> str:
    targets = []
    for index, target in enumerate(TARGET_COLUMNS):
        state = first_state if index == 0 else "not-mentioned"
        evidence = first_evidence if index == 0 else ""
        targets.append({"target": target, "state": state, "evidence": evidence})
    return json.dumps(
        {
            "schema_version": TICKET37_OUTPUT_SCHEMA_VERSION,
            "language_code": "es",
            "targets": targets,
        }
    )


def _manifest() -> FoldManifest:
    gold = [
        SourceStudy(
            f"gold-{index:02d}",
            True,
            tuple((index + target_index) % 2 for target_index in range(len(TARGET_COLUMNS))),
        )
        for index in range(9)
    ]
    report_only = [
        SourceStudy(f"report-{index:02d}", True, (None,) * len(TARGET_COLUMNS))
        for index in range(18)
    ]
    return freeze_folds(
        SourceManifest("train", tuple(gold + report_only), ()),
        source_version="ticket37-fixture-v1",
        seed=17,
    )


def test_ticket37_output_requires_exact_source_evidence_and_records_language() -> None:
    report = ReportRecord("study-es", "private://study-es", "Rotura completa del ligamento cruzado anterior.")

    result = parse_ticket37_output(
        _model_output(first_evidence="Rotura completa del ligamento cruzado anterior."),
        report=report,
    )

    assert result.language_code == "es"
    assert result.labels[0].state == "positive"
    assert result.labels[0].evidence == "Rotura completa del ligamento cruzado anterior."
    assert result.labels[0].supervision_mask == 1
    assert result.labels[1].supervision_mask == 0


def test_ticket37_unverifiable_evidence_is_masked_and_malformed_schema_fails_closed() -> None:
    report = ReportRecord("study-es", "private://study-es", "Rotura completa del ligamento cruzado anterior.")

    bad_span = parse_ticket37_output(
        _model_output(first_evidence="No aparece en el informe."),
        report=report,
    )
    malformed = parse_ticket37_output("not json", report=report)

    assert bad_span.labels[0].state == "extraction-failed"
    assert bad_span.labels[0].supervision_mask == 0
    assert bad_span.labels[1].state == "not-mentioned"
    assert all(label.state == "extraction-failed" for label in malformed.labels)
    assert all(label.supervision_mask == 0 for label in malformed.labels)


def test_ticket37_admission_rejects_fold_contamination_and_keeps_full_gold_cohort() -> None:
    manifest = _manifest()
    gold = []
    for membership in manifest.gold_memberships:
        report = ReportRecord(
            membership.study_instance_uid,
            f"private://{membership.study_instance_uid}",
            "ACL finding documented.",
        )
        result = parse_ticket37_output(
            _model_output(first_state="positive", first_evidence="ACL finding documented."),
            report=report,
        )
        gold.append(make_ticket37_observation(membership, result))
    weak = []
    for membership in manifest.report_only_memberships:
        report = ReportRecord(
            membership.study_instance_uid,
            f"private://{membership.study_instance_uid}",
            "No ACL tear is identified.",
        )
        result = parse_ticket37_output(
            _model_output(first_state="negative", first_evidence="No ACL tear is identified."),
            report=report,
        )
        weak.append(make_ticket37_observation(membership, result))

    private, public = build_ticket37_admission(
        manifest=manifest,
        observations=(*gold, *weak),
        frozen_policy=make_policy_payload(frozen_at_utc="2026-09-26T00:00:00Z"),
    )

    admissions = cast(Mapping[str, object], private["admissions"])
    assert admissions["source_id"] == "ticket37-qwen3-8b"
    assert len(cast(list[object], admissions["folds"])) == 3
    assert public["gold_study_count"] == len(manifest.gold_memberships)
    assert public["report_only_sample_count"] == len(weak)
    assert "study_instance_uid" not in json.dumps(public)
    language_reliability = cast(list[Mapping[str, object]], public["language_reliability"])
    assert all(
        "fold" in metric
        for language in language_reliability
        for metric in cast(list[Mapping[str, object]], language.get("metrics", []))
    )

    contaminated = replace(
        gold[0],
        assigned_fold=(gold[0].assigned_fold + 1) % 3,
        consuming_folds=((gold[0].assigned_fold + 1) % 3,),
    )
    with pytest.raises(Ticket37Error, match="fold manifest"):
        build_ticket37_admission(
            manifest=manifest,
            observations=(contaminated, *gold[1:], *weak),
            frozen_policy=make_policy_payload(frozen_at_utc="2026-09-26T00:00:00Z"),
        )


def test_ticket37_reference_labels_win_conflicting_admitted_report_values() -> None:
    reference: list[int | None] = [None] * len(TARGET_COLUMNS)
    weak: list[int | None] = [None] * len(TARGET_COLUMNS)
    masks = [0] * len(TARGET_COLUMNS)
    reference[:3] = [1, None, 0]
    weak[:3] = [0, 1, 1]
    masks[:3] = [1, 1, 1]
    resolved = merge_ticket37_reference_labels(
        reference_labels=reference,
        report_values=weak,
        report_masks=masks,
    )

    assert resolved[:3] == (1, 1, 0)

    bad_reference = list(reference)
    bad_reference[0] = True  # type: ignore[assignment]
    with pytest.raises(Ticket37Error, match="non-binary"):
        merge_ticket37_reference_labels(
            reference_labels=bad_reference,
            report_values=weak,
            report_masks=masks,
        )


def test_ticket37_approval_pins_exact_candidate_and_cpu_scope(tmp_path) -> None:
    path = tmp_path / "approval.json"
    payload = ticket37_approval_record(
        approval_reference="fixture user directive",
        approved_by="project owner",
        approved_on="2026-09-26",
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_ticket37_approval(path)["status"] == "approved-for-bounded-pilot"

    cast(dict[str, object], payload["candidate"])["license"] = "MIT"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Ticket37Error, match="exact reviewed model"):
        load_ticket37_approval(path)


def test_ticket37_frozen_prompt_is_stable_and_contains_valid_schema() -> None:
    prompt, fingerprint = freeze_ticket37_prompt()
    rendered = prompt.replace("{report_text}", "sample report")

    assert fingerprint == freeze_ticket37_prompt()[1]
    assert '"schema_version":"rsna-knee-ticket37-qwen3-output-v1"' in rendered
    assert "language_code" in rendered
    assert "confidence" in rendered
    assert "sample report" in rendered


def test_ticket37_pilot_plan_contains_all_gold_and_a_stable_bounded_weak_sample() -> None:
    manifest = _manifest()
    corpus = ReportCorpus(
        "ticket37-corpus-v1",
        tuple(
            ReportRecord(item.study_instance_uid, f"private://{item.study_instance_uid}", "Fixture report")
            for item in manifest.memberships
        ),
    )

    first = plan_ticket37_sample(manifest=manifest, corpus=corpus, report_only_per_fold=2)
    second = plan_ticket37_sample(manifest=manifest, corpus=corpus, report_only_per_fold=2)

    assert [(membership.study_instance_uid, report.report_sha256) for membership, report in first] == [
        (membership.study_instance_uid, report.report_sha256) for membership, report in second
    ]
    assert sum(membership.cohort == "gold" for membership, _ in first) == len(manifest.gold_memberships)
    assert sum(membership.cohort == "report_only" for membership, _ in first) == 6
    assert all(
        membership.cohort == "gold" or len(tuple(fold for fold in (0, 1, 2) if fold != membership.fold)) == 2
        for membership, _ in first
    )


def test_ticket37_private_json_is_restrictive_and_immutable(tmp_path) -> None:
    path = tmp_path / "private" / "output.json"

    write_ticket37_json(path, {"schema_version": "fixture"})

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    with pytest.raises(ReportLabelAuditError, match="immutable artifact already exists"):
        write_ticket37_json(path, {"schema_version": "changed"})


def test_ticket37_private_json_refuses_symlinks_and_permissive_existing_directories(tmp_path) -> None:
    protected = tmp_path / "protected"
    protected.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(protected, target_is_directory=True)
    with pytest.raises(Ticket37Error, match="symlink"):
        write_ticket37_json(alias / "evidence.json", {"schema_version": "fixture"})
    assert stat.S_IMODE(protected.stat().st_mode) == 0o700

    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    with pytest.raises(Ticket37Error, match="group or world access"):
        write_ticket37_json(shared / "evidence.json", {"schema_version": "fixture"})
    assert stat.S_IMODE(shared.stat().st_mode) == 0o755


def test_ticket37_public_model_manifest_can_live_in_a_shared_snapshot_directory(tmp_path) -> None:
    model_directory = tmp_path / "public-model"
    model_directory.mkdir(mode=0o755)
    model_directory.chmod(0o755)
    manifest_path = model_directory / "ticket37-model-manifest.json"

    write_ticket37_json(manifest_path, {"schema_version": "public-model"}, mode=0o644)

    assert stat.S_IMODE(model_directory.stat().st_mode) == 0o755
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o644


def test_ticket37_comparison_counts_new_calls_only_against_matching_prior_admissions() -> None:
    manifest = _manifest()
    # Restrict this public-boundary comparison to two selected studies per
    # assigned fold, matching the bounded pilot default.
    sample_uids = {
        item.study_instance_uid
        for fold in (0, 1, 2)
        for item in [candidate for candidate in manifest.report_only_memberships if candidate.fold == fold][:2]
    }
    gold: list[Ticket37Observation] = []
    report_only: list[Ticket37Observation] = []
    for membership in manifest.memberships:
        if membership.cohort == "report_only" and membership.study_instance_uid not in sample_uids:
            continue
        text = "ACL finding documented." if membership.cohort == "gold" else "No ACL tear is identified."
        state = "positive" if membership.cohort == "gold" else "negative"
        report = ReportRecord(membership.study_instance_uid, f"private://{membership.study_instance_uid}", text)
        result = parse_ticket37_output(
            _model_output(first_state=state, first_evidence=text),
            report=report,
        )
        item = make_ticket37_observation(membership, result)
        (gold if membership.cohort == "gold" else report_only).append(item)
    private, _ = build_ticket37_admission(
        manifest=manifest,
        observations=(*gold, *report_only),
        frozen_policy=make_policy_payload(frozen_at_utc="2026-09-26T00:00:00Z"),
    )
    prior_sources = []
    for source_id in ("ticket15-deterministic", "ticket16-astra-medium"):
        folds = []
        for fold in (0, 1, 2):
            studies = []
            for observation in report_only:
                if fold not in observation.consuming_folds:
                    continue
                studies.append(
                    {
                        "study_instance_uid": observation.study_uid,
                        "consuming_fold": fold,
                        "report_sha256": observation.report_sha256,
                        "targets": [
                            {
                                "target": target,
                                "state": "not-mentioned",
                                "value": None,
                                "supervision_mask": 0,
                                "admitted": False,
                                "rejection_reason": "state_is_always_masked",
                            }
                            for target in TARGET_COLUMNS
                        ],
                    }
                )
            folds.append({"fold": fold, "studies": studies})
        prior_sources.append({"source_id": source_id, "folds": folds})
    prior = {
        "lineage": {"fold_identity": {"fold_manifest_checksum": manifest.manifest_checksum}},
        "policy": make_policy_payload(frozen_at_utc="2026-09-26T00:00:00Z"),
        "fold_specific_training_admissions": prior_sources,
    }

    comparison = compare_ticket37_admissions_with_ticket30(
        ticket37_private=private,
        ticket30_private=prior,
    )

    target_counts = cast(Mapping[str, Mapping[str, object]], comparison["target_counts"])
    acl = target_counts["ACL"]
    assert acl["local_binary_call_exposures"] == 12
    assert acl["new_to_existing_admitted_calls"] == 12
    assert acl["existing_any_admitted_call_exposures"] == 0
    assert comparison["study_identifiers_public"] is False
