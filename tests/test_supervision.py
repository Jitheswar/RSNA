from __future__ import annotations

from dataclasses import replace

import pytest

from rsna_knee.folds import freeze_folds
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.report_extraction import ReportCorpus, ReportRecord, run_report_extraction
from rsna_knee.supervision import (
    ADMISSION_SCHEMA_VERSION,
    MIXTURE_POLICY_VERSION,
    WEAK_LABEL_VERSION,
    SupervisionAdmissionError,
    admit_fold_supervision,
    merge_reference_and_weak_labels,
)
from rsna_knee.synthetic import make_synthetic_study
from rsna_knee.targets import NUM_TARGETS, TARGET_COLUMNS
from rsna_knee.training import TrainingConfig, train_one_fold


POSITIVE_REPORTS = (
    "A high-grade partial tear of the ACL is present.",
    "There is a complete acute tear of the medial collateral ligament.",
    "A medial meniscus tear reaches the articular surface.",
    "A lateral meniscus tear reaches the articular surface.",
    "Moderate medial compartment osteoarthritis is present.",
    "Large lateral tibiofemoral osteoarthritis is present.",
    "Moderate patellofemoral compartment osteoarthritis is present.",
    "A large joint effusion distends the joint.",
    "Synovitis is present.",
    "A large Baker's cyst is present.",
    "A bone contusion is present.",
    "An acute fracture line is visible.",
)


def _manifest() -> tuple[SourceManifest, tuple[str, ...]]:
    gold = tuple(
        SourceStudy(
            f"gold-{index}",
            True,
            tuple((index + target_index) % 2 for target_index in range(NUM_TARGETS)),
        )
        for index in range(9)
    )
    report_only = tuple(
        SourceStudy(f"report-{index}", True, (None,) * NUM_TARGETS)
        for index in range(3)
    )
    return SourceManifest("train", gold + report_only, ()), tuple(item.study_instance_uid for item in report_only)


def _report_corpus(report_only_uids: tuple[str, ...], gold_uids: tuple[str, ...]) -> ReportCorpus:
    records = [
        ReportRecord(uid, f"private://{uid}", "The ACL is intact.")
        for uid in report_only_uids
    ]
    records.extend(
        ReportRecord(uid, f"private://{uid}", " ".join(POSITIVE_REPORTS))
        for uid in gold_uids
    )
    return ReportCorpus("synthetic-ticket17-reports-v1", tuple(records))


def test_admission_is_fold_clean_and_keeps_unknown_weak_entries_masked() -> None:
    source, report_only_uids = _manifest()
    manifest = freeze_folds(source, source_version="synthetic-ticket17-v1", seed=17)
    gold_uids = tuple(item.study_instance_uid for item in manifest.gold_memberships)
    extraction = run_report_extraction(
        manifest,
        _report_corpus(report_only_uids, gold_uids),
        verify_reproducibility=True,
    )

    admitted = admit_fold_supervision(manifest, 0, extraction)

    assert admitted.schema_version == ADMISSION_SCHEMA_VERSION
    assert admitted.weak_label_version == WEAK_LABEL_VERSION
    assert admitted.mixture_policy == MIXTURE_POLICY_VERSION
    assert set(admitted.training_study_uids) == set(manifest.training_study_uids(0))
    assert set(admitted.held_out_study_uids) == set(manifest.held_out_study_uids(0))
    assert set(admitted.admitted_report_only_study_uids).issubset(
        set(manifest.training_study_uids(0))
    )
    assert not set(admitted.admitted_report_only_study_uids).intersection(
        manifest.held_out_study_uids(0)
    )

    weak_uid = admitted.admitted_report_only_study_uids[0]
    weak = admitted.by_study[weak_uid]
    assert weak.source_kind == "weak"
    assert weak.targets[0] == 0
    assert weak.supervision_mask[0] == 1
    assert weak.supervision_mask[1:] == (0,) * (NUM_TARGETS - 1)
    assert weak.supervised_entry_count == 1

    gold_uid = next(
        uid for uid in admitted.training_study_uids if uid.startswith("gold-")
    )
    assert admitted.by_study[gold_uid].source_kind == "reference"
    assert admitted.by_study[gold_uid].supervision_mask == (1,) * NUM_TARGETS
    assert admitted.to_public_dict()["boundaries"]["fallback_source"] == (
        "trusted_reference_labels_only"
    )
    assert "study_instance_uid" not in str(admitted.to_public_dict())


def test_reference_labels_override_a_conflicting_weak_label_and_record_it() -> None:
    source, report_only_uids = _manifest()
    manifest = freeze_folds(source, source_version="synthetic-ticket17-v1", seed=19)
    gold_uids = tuple(item.study_instance_uid for item in manifest.gold_memberships)
    extraction = run_report_extraction(
        manifest,
        _report_corpus(report_only_uids, gold_uids),
        verify_reproducibility=True,
    )
    weak_record = extraction.folds[0].training_labels[0]
    reference = (1,) + (0,) * (NUM_TARGETS - 1)

    resolved = merge_reference_and_weak_labels(reference, weak_record)

    assert resolved.targets[0] == 1
    assert resolved.supervision_mask[0] == 1
    assert len(resolved.disagreements) == 1
    assert resolved.disagreements[0].target == TARGET_COLUMNS[0]
    assert resolved.disagreements[0].reference_value == 1
    assert resolved.disagreements[0].weak_value == 0
    private = resolved.to_private_dict()
    disagreement = private["disagreements"][0]
    assert disagreement["source_report_ref"].startswith("private://")
    assert len(disagreement["source_report_sha256"]) == 64


def test_admission_rejects_incomplete_upstream_lineage() -> None:
    source, report_only_uids = _manifest()
    manifest = freeze_folds(source, source_version="synthetic-ticket17-v1", seed=23)
    gold_uids = tuple(item.study_instance_uid for item in manifest.gold_memberships)
    extraction = run_report_extraction(
        manifest,
        _report_corpus(report_only_uids, gold_uids),
        verify_reproducibility=True,
    )
    broken_evaluation = replace(
        extraction.evaluation,
        upstream_input_manifest={"fold_manifest": {"manifest_checksum": "missing"}},
    )
    broken = replace(extraction, evaluation=broken_evaluation)

    with pytest.raises(SupervisionAdmissionError, match="lineage"):
        admit_fold_supervision(manifest, 0, broken)


def test_training_consumes_admitted_weak_labels_without_changing_image_baseline(tmp_path) -> None:
    source_labels, report_labels = _manifest()
    study_rows = tuple(
        (
            make_synthetic_study(
                item.study_instance_uid,
                series_count=1,
                slices_per_series=2,
                seed=100 + index,
            ),
            item,
        )
        for index, item in enumerate(source_labels.studies)
    )
    source = SourceManifest(
        "train",
        tuple(
            SourceStudy(
                study.study_instance_uid,
                source_row.has_reference_labels,
                source_row.reference_labels,
            )
            for study, source_row in study_rows
        ),
        (),
    )
    manifest = freeze_folds(source, source_version="synthetic-ticket17-v1", seed=29)
    studies = {study.study_instance_uid: study for study, _ in study_rows}
    report_only_uids = tuple(
        study.study_instance_uid
        for study, source_row in study_rows
        if not source_row.has_reference_labels
    )
    gold_uids = tuple(
        study.study_instance_uid
        for study, source_row in study_rows
        if source_row.has_reference_labels
    )
    extraction = run_report_extraction(
        manifest,
        _report_corpus(report_only_uids, gold_uids),
        verify_reproducibility=True,
    )
    admitted = admit_fold_supervision(manifest, 0, extraction)
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=2,
        max_series=1,
        successful_optimizer_steps=2,
        checkpoint_interval_steps=1,
    )

    result = train_one_fold(
        studies,
        manifest,
        0,
        tmp_path / "weak-fold-0",
        config=config,
        source_version="synthetic-ticket17-v1",
        require_preflight=False,
        allow_local_smoke=True,
        supervision=admitted,
    )

    assert result.evaluation is not None
    assert set(admitted.training_study_uids) == set(result.training_study_uids)
    assert result.sample_exposures == result.attempted_batches * config.batch_size
    assert result.supervised_entry_exposures > 0
    assert result.to_public_dict()["reference_supervision_only"] is False
    private = result.to_private_dict()
    assert private["supervision"]["weak_label_version"] == WEAK_LABEL_VERSION
    assert private["supervision"]["source_disagreements"] == []
