from __future__ import annotations

import json

import pytest

from rsna_knee.folds import (
    assert_comparable_fold_manifests,
    FoldManifestIntegrityError,
    ImmutableArtifactExistsError,
    FoldFreezeError,
    GroupingAssessment,
    freeze_folds,
    load_fold_manifest,
    load_inventory_source,
    run_fold_freeze,
    validate_fold_manifest_against_source,
    write_fold_reports,
)
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.targets import TARGET_COLUMNS


def _source_manifest() -> SourceManifest:
    studies: list[SourceStudy] = []
    for index in range(9):
        labels = tuple((index + target_index) % 2 for target_index in range(len(TARGET_COLUMNS)))
        studies.append(SourceStudy(f"study-{index:02d}", True, labels))
    for index in range(6):
        studies.append(SourceStudy(f"report-{index:02d}", True, (None,) * len(TARGET_COLUMNS)))
    return SourceManifest("train", tuple(studies), ())


def test_freeze_folds_assigns_every_training_study_to_one_declared_fold() -> None:
    manifest = freeze_folds(_source_manifest(), source_version="fixture-v1", seed=17)

    assert manifest.fold_ids == (0, 1, 2)
    assert len(manifest.memberships) == 15
    assert {item.study_instance_uid for item in manifest.memberships} == {
        *(f"study-{index:02d}" for index in range(9)),
        *(f"report-{index:02d}" for index in range(6)),
    }
    assert {item.fold for item in manifest.memberships} <= set(manifest.fold_ids)
    assert manifest.grouping.repeated_patient_leakage_cannot_be_ruled_out is True
    assert manifest.manifest_checksum


def test_freeze_folds_balances_gold_targets_without_searching_model_scores() -> None:
    first = freeze_folds(_source_manifest(), source_version="fixture-v1", seed=17)
    repeated = freeze_folds(_source_manifest(), source_version="fixture-v1", seed=17)

    assert first.memberships == repeated.memberships
    assert first.manifest_checksum == repeated.manifest_checksum
    for summary in first.fold_summaries:
        assert tuple(summary.target_counts_by_name) == TARGET_COLUMNS
        assert all(
            counts.positive > 0
            and counts.negative > 0
            and counts.support_status == "both_classes"
            for counts in summary.target_counts_by_name.values()
        )
    assert first.pooled_gold_target_counts["ACL"] == {
        "positive": 4,
        "negative": 5,
        "positive_group_count": 4,
        "negative_group_count": 5,
        "support_status": "both_classes",
    }


def test_freeze_folds_keeps_an_attested_patient_group_in_one_fold_across_cohorts() -> None:
    source = _source_manifest()
    study_to_group = {
        study.study_instance_uid: f"patient-{index}"
        for index, study in enumerate(source.studies)
    }
    for index in range(6):
        study_to_group[f"report-{index:02d}"] = f"patient-{index}"
    grouping = GroupingAssessment.reliable_key(
        study_to_group,
        field_name="PatientID",
        scope="fixture-site",
        provenance="fixture grouping audit v1",
        examined_fields=("PatientID", "IssuerOfPatientID"),
    )

    manifest = freeze_folds(source, source_version="fixture-v1", grouping=grouping, seed=17)
    by_uid = {item.study_instance_uid: item for item in manifest.memberships}

    assert manifest.grouping.repeated_patient_leakage_cannot_be_ruled_out is False
    assert all(item.group_key for item in manifest.memberships)
    for index in range(6):
        assert by_uid[f"study-{index:02d}"].fold == by_uid[f"report-{index:02d}"].fold

    with pytest.raises(FoldFreezeError, match="cover every training study"):
        freeze_folds(
            source,
            source_version="fixture-v1",
            grouping=GroupingAssessment.reliable_key(
                {"study-00": "patient-0"},
                field_name="PatientID",
                scope="fixture-site",
                provenance="fixture grouping audit v1",
            ),
        )


def test_freeze_folds_records_unavoidable_one_class_support_instead_of_inventing_support() -> None:
    labels = [0] * len(TARGET_COLUMNS)
    labels[0] = 1
    source = SourceManifest(
        "train",
        (
            SourceStudy("gold-positive", True, tuple(labels)),
            SourceStudy("gold-negative", True, (0,) * len(TARGET_COLUMNS)),
            SourceStudy("report-only", True, (None,) * len(TARGET_COLUMNS)),
        ),
        (),
    )

    manifest = freeze_folds(source, source_version="fixture-v1")

    acl_counts = [summary.target_counts_by_name["ACL"] for summary in manifest.fold_summaries]
    assert all(item.support_status == "one_class_unavoidable" for item in acl_counts)
    assert sum(item.positive for item in acl_counts) == 1
    assert sum(item.negative for item in acl_counts) == 1
    assert all("mathematically unavailable" in item.support_reason for item in acl_counts)


def test_freeze_folds_rejects_partial_labels_and_untrustworthy_grouping() -> None:
    partial = SourceManifest(
        "train",
        (SourceStudy("partial", True, (1,) + (None,) * (len(TARGET_COLUMNS) - 1)),),
        (),
    )
    with pytest.raises(FoldFreezeError, match="partially reference-labeled"):
        freeze_folds(partial, source_version="fixture-v1")

    grouping = GroupingAssessment.untrustworthy_key(
        field_name="PatientID",
        examined_fields=("PatientID",),
        reason="fixture contradiction",
    )
    with pytest.raises(FoldFreezeError, match="untrustworthy grouping"):
        freeze_folds(_source_manifest(), source_version="fixture-v1", grouping=grouping)


def test_fold_reports_are_checksum_verified_private_and_immutable(tmp_path) -> None:
    manifest = freeze_folds(_source_manifest(), source_version="fixture-v1", seed=17)
    private_path = tmp_path / "private" / "fold-manifest.json"
    public_path = tmp_path / "public" / "fold-evidence.json"

    write_fold_reports(manifest, private_path=private_path, public_path=public_path)

    assert load_fold_manifest(private_path) == manifest
    private_text = private_path.read_text(encoding="utf-8")
    public_text = public_path.read_text(encoding="utf-8")
    assert "study-00" in private_text
    assert "study-00" not in public_text
    assert "report-00" not in public_text
    assert '"manifest_checksum"' in public_text
    assert '"gold_target_counts"' in public_text

    with pytest.raises(ImmutableArtifactExistsError):
        write_fold_reports(manifest, private_path=private_path, public_path=public_path)

    private_path.write_text(private_text.replace(manifest.manifest_checksum, "0" * 64), encoding="utf-8")
    with pytest.raises(FoldManifestIntegrityError):
        load_fold_manifest(private_path)


def test_fold_manifest_exposes_training_and_held_out_partitions_for_adaptation() -> None:
    manifest = freeze_folds(_source_manifest(), source_version="fixture-v1", seed=17)

    for fold in manifest.fold_ids:
        held_out = set(manifest.held_out_study_uids(fold))
        training = set(manifest.training_study_uids(fold))
        assert held_out.isdisjoint(training)
        assert held_out | training == set(manifest.study_to_fold)
        with pytest.raises(FoldFreezeError, match="held-out"):
            manifest.assert_adaptation_input_is_training_only(fold, tuple(held_out))
        with pytest.raises(FoldFreezeError, match="unknown"):
            manifest.assert_adaptation_input_is_training_only(fold, ("unknown-study",))
        manifest.assert_adaptation_input_is_training_only(fold, tuple(training))


def test_fold_manifest_validation_checks_source_membership_and_label_joins() -> None:
    source = _source_manifest()
    manifest = freeze_folds(source, source_version="fixture-v1", seed=17)
    validate_fold_manifest_against_source(manifest, source)

    changed_labels = list(source.studies)
    changed_labels[0] = SourceStudy(
        changed_labels[0].study_instance_uid,
        changed_labels[0].report_present,
        (1,) * len(TARGET_COLUMNS),
    )
    changed_source = SourceManifest(source.split, tuple(changed_labels), source.series)
    with pytest.raises(FoldManifestIntegrityError, match="source checksum"):
        validate_fold_manifest_against_source(manifest, changed_source)


def test_reliable_grouping_audit_round_trips_without_publishing_group_values(tmp_path) -> None:
    source = _source_manifest()
    study_to_group = {
        study.study_instance_uid: f"patient-{index}"
        for index, study in enumerate(source.studies)
    }
    grouping = GroupingAssessment.reliable_key(
        study_to_group,
        field_name="PatientID",
        scope="fixture-site",
        provenance="fixture grouping audit v1",
        examined_fields=("PatientID",),
        missing_key_count=0,
        within_study_consistent=True,
        cross_study_reuse_count=6,
        issuer_scope_verified=True,
        contradiction_count=0,
    )
    manifest = freeze_folds(source, source_version="fixture-v1", grouping=grouping)
    private_path = tmp_path / "private.json"
    public_path = tmp_path / "public.json"

    write_fold_reports(manifest, private_path=private_path, public_path=public_path)

    assert load_fold_manifest(private_path) == manifest
    public_text = public_path.read_text(encoding="utf-8")
    assert "patient-0" not in public_text
    assert '"issuer_scope_verified": true' in public_text


def test_additional_repeat_requires_its_own_justification() -> None:
    with pytest.raises(FoldFreezeError, match="additional repeats"):
        freeze_folds(_source_manifest(), source_version="fixture-v1", repeat_name="repeat-2")

    repeated = freeze_folds(
        _source_manifest(),
        source_version="fixture-v1",
        repeat_name="repeat-2",
        repeat_justification="fixed seed repeat for variance reporting",
    )
    assert repeated.repeat_name == "repeat-2"
    assert repeated.repeat_justification == "fixed seed repeat for variance reporting"


def test_changed_fold_manifest_invalidates_comparison_identity() -> None:
    original = freeze_folds(_source_manifest(), source_version="fixture-v1", seed=17)
    same = freeze_folds(_source_manifest(), source_version="fixture-v1", seed=17)
    changed = freeze_folds(_source_manifest(), source_version="fixture-v1", seed=18)

    assert_comparable_fold_manifests(original, same)
    with pytest.raises(FoldFreezeError, match="invalidate affected comparisons"):
        assert_comparable_fold_manifests(original, changed)


def test_run_fold_freeze_consumes_ticket03_private_inventory_without_publishing_uids(tmp_path) -> None:
    source = _source_manifest()
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": "rsna-knee-inventory-v1",
                "source_version": "fixture-v1",
                "split": "train",
                "archive": {"fingerprint": "archive-fingerprint"},
                "source": {
                    "studies": [
                        {
                            "StudyInstanceUID": study.study_instance_uid,
                            "report_present": study.report_present,
                            "reference_labels": list(study.reference_labels),
                        }
                        for study in source.studies
                    ],
                    "series": [],
                },
                "reconciliation": {"full_corpus_reconciled": True},
            }
        ),
        encoding="utf-8",
    )

    grouping_path = tmp_path / "grouping.json"
    grouping_path.write_text(
        json.dumps(
            GroupingAssessment.no_reliable_key(
                examined_fields=("ticket-03-source-manifest",),
                reason="The source manifest exposes no authoritative patient key; use study-level folds.",
            ).to_private_dict()
        ),
        encoding="utf-8",
    )
    loaded = load_inventory_source(inventory_path)
    summary = run_fold_freeze(
        inventory_path,
        tmp_path / "folds",
        grouping_path=grouping_path,
    )

    with pytest.raises(FoldFreezeError, match="explicit grouping evidence"):
        run_fold_freeze(inventory_path, tmp_path / "missing-grouping")

    unreconciled_payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    unreconciled_payload["reconciliation"]["full_corpus_reconciled"] = False
    unreconciled_path = tmp_path / "unreconciled-inventory.json"
    unreconciled_path.write_text(json.dumps(unreconciled_payload), encoding="utf-8")
    with pytest.raises(FoldFreezeError, match="unreconciled"):
        run_fold_freeze(
            unreconciled_path,
            tmp_path / "unreconciled-folds",
            grouping_path=grouping_path,
        )

    assert loaded.source_manifest == source
    assert summary["input_inventory"] == {
        "source_studies": 15,
        "source_gold_studies": 9,
        "source_report_only_studies": 6,
        "source_manifest_checksum": summary["source"]["source_manifest_checksum"],
        "full_corpus_reconciled": True,
    }
    public_text = (tmp_path / "folds/public/fold-evidence.json").read_text(encoding="utf-8")
    assert "study-00" not in public_text
    assert "report-00" not in public_text
