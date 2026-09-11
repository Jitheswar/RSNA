from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

import rsna_knee.folds as folds_module
from rsna_knee.folds import (
    assert_comparable_fold_manifests,
    FOLD_GENERATION_METHOD,
    FoldManifestIntegrityError,
    ImmutableArtifactExistsError,
    FoldFreezeError,
    GoldFoldSizePolicy,
    GroupingAssessment,
    LEGACY_FOLD_GENERATION_METHOD,
    LEGACY_FOLD_SCHEMA_VERSION,
    evaluate_fold_validation_gates,
    freeze_folds,
    load_fold_manifest,
    load_inventory_source,
    run_fold_freeze,
    validate_fold_manifest_against_source,
    write_fold_reports,
)
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.targets import TARGET_COLUMNS

# Public pooled positive counts of the 58 reference studies; synthetic cohorts
# reuse only these margins, never real label vectors.
_REFERENCE_POSITIVE_COUNTS = (24, 9, 26, 23, 15, 11, 21, 35, 27, 12, 19, 18)


def _reference_margin_source(study_count: int, seed: int) -> SourceManifest:
    rng = np.random.default_rng(seed)
    labels = np.zeros((study_count, len(TARGET_COLUMNS)), dtype=int)
    for target_index, positives in enumerate(_REFERENCE_POSITIVE_COUNTS):
        count = min(study_count - 3, max(3, round(positives * study_count / 58)))
        labels[rng.permutation(study_count)[:count], target_index] = 1
    return SourceManifest(
        "train",
        tuple(
            SourceStudy(f"gold-{index:03d}", True, tuple(int(value) for value in labels[index]))
            for index in range(study_count)
        ),
        (),
    )


def _random_multilabel_source(study_count: int, seed: int) -> SourceManifest:
    rng = np.random.default_rng(seed)
    prevalence = np.linspace(0.1, 0.65, len(TARGET_COLUMNS))
    labels = (rng.random((study_count, len(TARGET_COLUMNS))) < prevalence).astype(int)
    studies = [
        SourceStudy(f"gold-{index:03d}", True, tuple(int(value) for value in labels[index]))
        for index in range(study_count)
    ]
    studies.extend(
        SourceStudy(f"report-{index:03d}", True, (None,) * len(TARGET_COLUMNS))
        for index in range(study_count // 3)
    )
    return SourceManifest("train", tuple(studies), ())


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


def _assert_folds_pass_every_gate(manifest) -> None:
    gates = manifest.validation_gates
    gold_count = len(manifest.gold_memberships)
    sizes = manifest.fold_gold_sizes

    assert manifest.generation_method == FOLD_GENERATION_METHOD
    assert gates is not None and gates.passed and not gates.failed_gates
    assert sorted(sizes) == sorted(
        [gold_count // 3 + (1 if fold < gold_count % 3 else 0) for fold in range(3)]
    )
    assert max(sizes) - min(sizes) <= 1
    assert max(sizes) <= manifest.gold_fold_size_policy.cap
    assert all(check.deviation <= check.tolerance for check in gates.class_balance)
    for index, target in enumerate(TARGET_COLUMNS):
        if target in gates.feasible_targets:
            for fold in range(3):
                labels = {
                    item.reference_labels[index]
                    for item in manifest.gold_memberships
                    if item.fold == fold
                }
                assert labels == {0, 1}, (target, fold)


def test_v1_seed_only_fold_failure_is_reproduced_and_corrected_by_v2() -> None:
    # A 58-study cohort with the reference cohort's positive counts reproduces
    # the frozen v1 failure: phase 2 never selects one fold, which keeps only
    # the studies placed there to seed both-class support.
    source = _reference_margin_source(58, seed=1)

    legacy = freeze_folds(source, source_version="synthetic-regression", generation_version=1)
    starved = min(range(3), key=lambda fold: legacy.fold_gold_sizes[fold])
    starved_rows = [item for item in legacy.gold_memberships if item.fold == starved]
    legacy_gates = evaluate_fold_validation_gates(
        legacy.memberships, legacy.grouping, deterministic_regeneration=True
    )

    assert legacy.schema_version == LEGACY_FOLD_SCHEMA_VERSION
    assert legacy.generation_method == LEGACY_FOLD_GENERATION_METHOD
    assert legacy.superseded is True
    assert sorted(legacy.fold_gold_sizes) == [4, 27, 27]
    assert all(
        {item.reference_labels[index] for item in starved_rows} == {0, 1}
        for index in range(len(TARGET_COLUMNS))
    )
    assert "gold_fold_size_balance" in legacy_gates.failed_gates
    assert "per_target_class_balance" in legacy_gates.failed_gates

    corrected = freeze_folds(source, source_version="synthetic-regression")

    assert corrected.superseded is False
    assert sorted(corrected.fold_gold_sizes) == [19, 19, 20]
    _assert_folds_pass_every_gate(corrected)


@pytest.mark.parametrize("study_count", [36, 58, 90, 300])
def test_random_multilabel_cohorts_pass_every_v2_gate(study_count: int) -> None:
    source = _random_multilabel_source(study_count, seed=20260911 + study_count)

    manifest = freeze_folds(source, source_version="synthetic-random", seed=5)
    repeated = freeze_folds(source, source_version="synthetic-random", seed=5)
    legacy = freeze_folds(source, source_version="synthetic-random", seed=5, generation_version=1)
    legacy_gates = evaluate_fold_validation_gates(
        legacy.memberships, legacy.grouping, deterministic_regeneration=True
    )

    _assert_folds_pass_every_gate(manifest)
    assert manifest.validation_gates.deterministic_regeneration is True
    assert repeated.manifest_checksum == manifest.manifest_checksum
    assert repeated.memberships == manifest.memberships
    assert len(manifest.report_only_memberships) == study_count // 3
    assert "gold_fold_size_balance" in legacy_gates.failed_gates


@pytest.mark.parametrize("study_count", [36, 58, 90, 300])
def test_reference_margin_cohorts_pass_every_v2_gate(study_count: int) -> None:
    manifest = freeze_folds(
        _reference_margin_source(study_count, seed=study_count),
        source_version="synthetic-margins",
    )

    _assert_folds_pass_every_gate(manifest)
    assert manifest.validation_gates.max_class_deviation < 3.0


def test_study_level_size_policy_caps_folds_at_ceil_share() -> None:
    policy = GoldFoldSizePolicy.for_group_sizes((1,) * 58)

    assert (policy.cap, policy.floor, policy.max_folds_above_floor) == (20, 19, 1)
    assert policy.singleton_groups is True
    assert policy.admits([19, 19, 19], 0, 1) is True
    assert policy.admits([20, 19, 18], 1, 1) is False
    assert policy.admits([20, 19, 18], 2, 1) is True
    assert policy.sizes_conform((20, 19, 19)) is True
    assert policy.sizes_conform((20, 20, 18)) is False
    assert GoldFoldSizePolicy.for_group_sizes((1,) * 36).cap == 12
    grouped = GoldFoldSizePolicy.for_group_sizes((3, 2, 1, 1, 1, 1))
    assert (grouped.cap, grouped.floor) == (5, None)


def test_reliable_multi_study_groups_stay_unsplit_under_the_widened_cap() -> None:
    rng = np.random.default_rng(17)
    studies: list[SourceStudy] = []
    study_to_group: dict[str, str] = {}
    for group_index in range(40):
        for member in range(int(rng.integers(1, 4))):
            uid = f"gold-{group_index:02d}-{member}"
            labels = tuple(int(value) for value in rng.random(len(TARGET_COLUMNS)) < 0.4)
            studies.append(SourceStudy(uid, True, labels))
            study_to_group[uid] = f"patient-{group_index:02d}"
    grouping = GroupingAssessment.reliable_key(
        study_to_group,
        field_name="PatientID",
        scope="fixture-site",
        provenance="fixture grouping audit v1",
    )

    manifest = freeze_folds(
        SourceManifest("train", tuple(studies), ()),
        source_version="synthetic-grouped",
        grouping=grouping,
    )
    folds_by_group: dict[str, set[int]] = {}
    for item in manifest.memberships:
        folds_by_group.setdefault(item.group_key, set()).add(item.fold)

    assert manifest.gold_fold_size_policy.singleton_groups is False
    assert max(manifest.fold_gold_sizes) <= manifest.gold_fold_size_policy.cap
    assert all(len(value) == 1 for value in folds_by_group.values())
    assert manifest.validation_gates.passed is True


def test_failed_gates_refuse_to_produce_a_manifest(monkeypatch) -> None:
    monkeypatch.setattr(folds_module, "CLASS_BALANCE_ABSOLUTE_TOLERANCE", 0.0)
    monkeypatch.setattr(folds_module, "CLASS_BALANCE_RELATIVE_TOLERANCE", 0.0)

    with pytest.raises(FoldFreezeError, match="validation gates failed.*no other seed"):
        freeze_folds(_reference_margin_source(58, seed=1), source_version="synthetic-refusal")


def test_loaded_v2_manifest_recomputes_its_gates() -> None:
    manifest = freeze_folds(_reference_margin_source(36, seed=3), source_version="synthetic-gates")
    tampered_gates = dataclasses.replace(
        manifest.validation_gates,
        fold_gold_sizes=(13, 12, 11),
    )

    with pytest.raises(FoldManifestIntegrityError, match="gates"):
        dataclasses.replace(manifest, validation_gates=tampered_gates)
    with pytest.raises(FoldManifestIntegrityError, match="version-2"):
        dataclasses.replace(manifest, generation_method=LEGACY_FOLD_GENERATION_METHOD)


def test_legacy_v1_manifest_round_trips_unchanged_and_is_marked_superseded(tmp_path) -> None:
    legacy = freeze_folds(_source_manifest(), source_version="fixture-v1", seed=17, generation_version=1)
    private_path = tmp_path / "private" / "fold-manifest.json"
    write_fold_reports(legacy, private_path=private_path, public_path=tmp_path / "public.json")

    payload = json.loads(private_path.read_text(encoding="utf-8"))
    loaded = load_fold_manifest(private_path)

    assert loaded == legacy
    assert loaded.superseded is True
    assert loaded.generation_version == 1
    assert payload["schema_version"] == LEGACY_FOLD_SCHEMA_VERSION
    assert not {"generation_version", "validation_gates", "gold_fold_size_policy"} & set(payload)
    public = json.loads((tmp_path / "public.json").read_text(encoding="utf-8"))
    assert public["superseded"]["superseded"] is True


def _write_inventory(path, source: SourceManifest) -> None:
    path.write_text(
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


def test_v2_correction_supersedes_v1_without_changing_it_or_its_seed(tmp_path) -> None:
    source = _reference_margin_source(58, seed=1)
    inventory_path = tmp_path / "inventory.json"
    _write_inventory(inventory_path, source)
    grouping = GroupingAssessment.no_reliable_key(examined_fields=("PatientID",))
    grouping_path = tmp_path / "grouping.json"
    grouping_path.write_text(json.dumps(grouping.to_private_dict()), encoding="utf-8")
    loaded = load_inventory_source(inventory_path)
    legacy = freeze_folds(
        loaded.source_manifest,
        source_version=loaded.source_version,
        source_fingerprint=loaded.source_fingerprint,
        input_inventory_reconciled=True,
        grouping=grouping,
        seed_name="fixture-real-seed",
        seed=99,
        generation_version=1,
    )
    legacy_private = tmp_path / "v1" / "private" / "fold-manifest.json"
    write_fold_reports(legacy, private_path=legacy_private, public_path=tmp_path / "v1" / "public.json")
    legacy_bytes = legacy_private.read_bytes()

    with pytest.raises(FoldFreezeError, match="seed search is not permitted"):
        run_fold_freeze(
            inventory_path,
            tmp_path / "wrong-seed",
            grouping_path=grouping_path,
            seed=100,
            supersedes_manifest_path=legacy_private,
        )
    with pytest.raises(ImmutableArtifactExistsError):
        run_fold_freeze(
            inventory_path,
            tmp_path / "v1",
            grouping_path=grouping_path,
            supersedes_manifest_path=legacy_private,
        )
    summary = run_fold_freeze(
        inventory_path,
        tmp_path / "v2",
        grouping_path=grouping_path,
        supersedes_manifest_path=legacy_private,
    )
    corrected = load_fold_manifest(tmp_path / "v2" / "private" / "fold-manifest.json")

    assert legacy_private.read_bytes() == legacy_bytes
    assert (corrected.seed_name, corrected.seed) == ("fixture-real-seed", 99)
    assert corrected.supersedes is not None
    assert corrected.supersedes.manifest_checksum == legacy.manifest_checksum
    assert corrected.manifest_checksum != legacy.manifest_checksum
    assert summary["supersedes"]["manifest_checksum"] == legacy.manifest_checksum
    assert summary["validation_gates"]["passed"] is True
    assert sorted(summary["fold_gold_sizes"]) == [19, 19, 20]
    _assert_folds_pass_every_gate(corrected)
