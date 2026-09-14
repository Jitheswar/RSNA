from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsna_knee.folds import FoldManifest, freeze_folds
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.report_labels import authority_register
from rsna_knee.report_extraction import (
    REPORT_EXTRACTION_SCHEMA_VERSION,
    ReportCorpus,
    ReportExtractionError,
    ReportPartition,
    ReportRecord,
    ReportSourceError,
    freeze_deterministic_extractor,
    load_report_corpus,
    run_report_extraction,
    validate_fold_report_boundaries,
    write_report_extraction_reports,
)
from rsna_knee.targets import TARGET_COLUMNS


POSITIVE_REPORTS = (
    "A high-grade partial tear of the ACL is present.",
    "There is a complete acute tear of the medial collateral ligament.",
    "A medial meniscus tear reaches the articular surface.",
    "A lateral meniscus tear reaches the articular surface.",
    "Moderate medial compartment osteoarthritis is present.",
    "Large lateral tibiofemoral compartment osteoarthritis is present.",
    "Moderate patellofemoral compartment osteoarthritis is present.",
    "A large joint effusion distends the joint.",
    "Synovitis is present.",
    "A large Baker's cyst is present.",
    "A bone contusion is present.",
    "An acute fracture line is visible.",
)

NEGATIVE_REPORTS = (
    "The ACL is intact.",
    "Only a low-grade sprain of the medial collateral ligament is present.",
    "Medial meniscus intrasubstance degeneration does not reach the surface.",
    "Lateral meniscus intrasubstance degeneration does not reach the surface.",
    "Only mild medial compartment cartilage wear is present.",
    "Only mild lateral compartment cartilage wear is present.",
    "Mild patellofemoral osteoarthritis only.",
    "Only a trace joint effusion is seen.",
    "The synovial lining is normal and not thickened.",
    "There is only a small Baker's cyst.",
    "No bone contusion is identified.",
    "No acute fracture is identified.",
)


def _manifest() -> FoldManifest:
    studies = [
        SourceStudy(
            f"gold-{index:02d}",
            True,
            tuple((index + target_index) % 2 for target_index in range(len(TARGET_COLUMNS))),
        )
        for index in range(12)
    ]
    studies.extend(
        SourceStudy(f"report-{index:02d}", True, (None,) * len(TARGET_COLUMNS))
        for index in range(6)
    )
    return freeze_folds(
        SourceManifest("train", tuple(studies), ()),
        source_version="report-extraction-fixture-v1",
        seed=17,
    )


def _report_for_labels(labels: tuple[int | None, ...]) -> str:
    return " ".join(
        (POSITIVE_REPORTS[index] if label == 1 else NEGATIVE_REPORTS[index])
        for index, label in enumerate(labels)
    )


def _corpus(manifest: FoldManifest) -> ReportCorpus:
    reports = [
        ReportRecord(
            item.study_instance_uid,
            f"private://{item.study_instance_uid}",
            _report_for_labels(item.reference_labels),
        )
        for item in manifest.gold_memberships
    ]
    reports.extend(
        ReportRecord(
            item.study_instance_uid,
            f"private://{item.study_instance_uid}",
            "The ACL is intact.",
        )
        for item in manifest.report_only_memberships
    )
    return ReportCorpus("synthetic-reports-v1", tuple(reports))


def test_frozen_extractor_is_reproducible_and_public_safe() -> None:
    extractor = freeze_deterministic_extractor()
    report = ReportRecord("gold-00", "private://gold-00", POSITIVE_REPORTS[0])

    first = extractor.extract(report, fold=0)
    second = extractor.extract(report, fold=0)

    assert extractor.is_frozen
    assert first.to_private_dict() == second.to_private_dict()
    assert first.to_public_dict()["extractor_version"] == extractor.extractor_version
    assert "private://gold-00" not in str(first.to_public_dict())
    assert POSITIVE_REPORTS[0] not in str(first.to_public_dict())


def test_run_partitions_training_labels_and_held_out_evaluation_cleanly() -> None:
    manifest = _manifest()
    result = run_report_extraction(manifest, _corpus(manifest), verify_reproducibility=True)

    assert result.schema_version == REPORT_EXTRACTION_SCHEMA_VERSION
    assert result.evaluation.reproducibility_verified
    assert result.evaluation.semantic_digest == result.semantic_digest
    assert tuple(item.fold for item in result.folds) == (0, 1, 2)
    assert result.evaluation.pooled_report_count == len(manifest.gold_memberships)
    assert result.evaluation.macro_accuracy == 1.0
    assert result.evaluation.mean_coverage == 1.0
    assert result.evaluation.latency.report_count == (
        len(manifest.gold_memberships) + 2 * len(manifest.report_only_memberships)
    )

    for fold_result in result.folds:
        assert all(item.partition is ReportPartition.TRAINING_LABELS for item in fold_result.training_labels)
        assert all(item.partition is ReportPartition.HELD_OUT_EVALUATION for item in fold_result.evaluation_labels)
        assert all(
            item.study_instance_uid in set(manifest.training_study_uids(fold_result.fold))
            for item in fold_result.training_labels
        )
        assert all(
            item.study_instance_uid in set(manifest.held_out_study_uids(fold_result.fold))
            for item in fold_result.evaluation_labels
        )
        assert not (
            {item.study_instance_uid for item in fold_result.training_labels}
            & {item.study_instance_uid for item in fold_result.evaluation_labels}
        )
        assert fold_result.evaluation_metrics is not None
        assert all(item.accuracy == 1.0 for item in fold_result.evaluation_metrics.targets)


def test_fold_boundary_validator_rejects_held_out_training_report() -> None:
    manifest = _manifest()
    fold = 0
    held_out = manifest.held_out_study_uids(fold)

    with pytest.raises(ReportExtractionError, match="held-out"):
        validate_fold_report_boundaries(
            manifest,
            fold,
            training_study_uids=(held_out[0],),
            evaluation_study_uids=(),
        )


def test_unknown_report_study_is_not_silently_dropped() -> None:
    manifest = _manifest()
    corpus = ReportCorpus(
        "synthetic-reports-v1",
        (ReportRecord("not-in-folds", "private://unknown", "The ACL is intact."),),
    )

    with pytest.raises(ReportSourceError, match="unknown studies"):
        run_report_extraction(manifest, corpus)


def test_unsupported_and_missing_evaluation_reports_are_explicitly_masked() -> None:
    manifest = _manifest()
    held_out_gold = manifest.held_out_study_uids(0)
    corpus = ReportCorpus(
        "synthetic-reports-v1",
        tuple(
            ReportRecord(uid, f"private://{uid}", "Possible marrow edema is present.")
            for uid in held_out_gold[:1]
        ),
    )
    result = run_report_extraction(manifest, corpus, verify_reproducibility=False)
    fold = result.folds[0]
    contusion = next(item for item in fold.evaluation_metrics.targets if item.target == "Contusion")

    assert fold.evaluation_labels[0].interpretation.labels[8].state.value == "not-mentioned"
    assert contusion.coverage == 0.0
    assert contusion.accuracy is None
    assert contusion.uncertain_count == 1
    assert contusion.abstention_count == 1
    expected_gold_count = sum(
        item.cohort == "gold" and item.fold == 0 for item in manifest.memberships
    )
    assert len(fold.missing_evaluation_report_uids) == expected_gold_count - 1

    no_report = result.folds[1].evaluation_metrics.targets[0]
    assert no_report.report_count == 0
    assert no_report.coverage is None
    assert no_report.accuracy is None


def test_report_artifacts_separate_private_labels_from_public_aggregate(tmp_path: Path) -> None:
    manifest = _manifest()
    result = run_report_extraction(manifest, _corpus(manifest), verify_reproducibility=False)
    public_path = tmp_path / "public" / "evidence.json"
    private_path = tmp_path / "private" / "labels.json"

    written = write_report_extraction_reports(
        result,
        public_path=public_path,
        private_path=private_path,
    )

    assert written == (public_path, private_path)
    public_text = public_path.read_text(encoding="utf-8")
    private_text = private_path.read_text(encoding="utf-8")
    assert "gold-00" not in public_text
    assert "private://gold-00" not in public_text
    assert POSITIVE_REPORTS[0] not in public_text
    assert "gold-00" in private_text
    assert "private://gold-00" in private_text
    assert POSITIVE_REPORTS[0] not in private_text
    public_payload = json.loads(public_text)
    private_payload = json.loads(private_text)
    assert public_payload["provenance"]["code_fingerprint"]
    assert "execution_identity" not in public_payload["provenance"]
    assert "execution_identity" in private_payload["provenance"]
    assert private_payload["fold_partitions"][0]["fold_training_study_uids"]
    assert private_payload["fold_partitions"][0]["fold_held_out_study_uids"]
    assert "pattern" in str(authority_register())

    with pytest.raises(FileExistsError):
        write_report_extraction_reports(
            result,
            public_path=public_path,
            private_path=tmp_path / "private" / "labels-2.json",
        )


def test_report_corpus_loader_requires_private_report_text(tmp_path: Path) -> None:
    path = tmp_path / "reports.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "rsna-knee-report-corpus-v1",
                "source_version": "fixture-v1",
                "reports": [
                    {
                        "study_instance_uid": "study-1",
                        "source_report_ref": "private://study-1",
                        "report_text": "The ACL is intact.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    corpus = load_report_corpus(path)

    assert corpus.records[0].report_sha256
    assert corpus.to_public_dict()["report_count"] == 1
    assert "The ACL is intact." not in str(corpus.to_private_dict())
