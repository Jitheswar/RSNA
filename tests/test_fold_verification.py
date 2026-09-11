from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rsna_knee.fold_verification import verify_v14
from rsna_knee.folds import (
    GroupingAssessment,
    freeze_folds,
    load_inventory_source,
    run_fold_freeze,
    write_fold_reports,
)
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.targets import TARGET_COLUMNS


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(23)
    labels = (rng.random((45, len(TARGET_COLUMNS))) < 0.35).astype(int)
    labels[:3] = 1
    labels[3:6] = 0
    studies = [
        SourceStudy(f"gold-{index:02d}", True, tuple(int(value) for value in labels[index]))
        for index in range(45)
    ]
    studies.extend(SourceStudy(f"report-{index:02d}", True, (None,) * len(TARGET_COLUMNS)) for index in range(20))
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
                            "report_present": True,
                            "reference_labels": list(study.reference_labels),
                        }
                        for study in studies
                    ],
                    "series": [],
                },
                "reconciliation": {"full_corpus_reconciled": True},
            }
        ),
        encoding="utf-8",
    )
    grouping = GroupingAssessment.no_reliable_key(examined_fields=("PatientID",))
    grouping_path = tmp_path / "grouping.json"
    grouping_path.write_text(json.dumps(grouping.to_private_dict()), encoding="utf-8")
    source = load_inventory_source(inventory_path)
    legacy = freeze_folds(
        source.source_manifest,
        source_version=source.source_version,
        source_fingerprint=source.source_fingerprint,
        input_inventory_reconciled=True,
        grouping=grouping,
        seed_name="fixture-seed",
        seed=7,
        generation_version=1,
    )
    legacy_path = tmp_path / "v1" / "private" / "fold-manifest.json"
    write_fold_reports(legacy, private_path=legacy_path, public_path=tmp_path / "v1" / "public.json")
    return inventory_path, grouping_path, legacy_path


def test_v14_passes_the_corrected_manifest_and_records_balance_evidence(tmp_path: Path) -> None:
    inventory_path, grouping_path, legacy_path = _write_fixture(tmp_path)
    run_fold_freeze(
        inventory_path,
        tmp_path / "v2",
        grouping_path=grouping_path,
        supersedes_manifest_path=legacy_path,
    )

    evidence = verify_v14(
        tmp_path / "v2" / "private" / "fold-manifest.json",
        inventory_path,
        grouping_path,
        public_evidence_path=tmp_path / "v2" / "public" / "fold-evidence.json",
        superseded_manifest_path=legacy_path,
        expected_gold_count=45,
    )
    statuses = {item["name"]: item["status"] for item in evidence["checks"]}
    text = json.dumps(evidence)

    assert evidence["status"] == "passed"
    assert statuses["deterministic_generation"] == "passed"
    assert statuses["no_seed_search_single_correction"] == "passed"
    assert statuses["zero_reliable_group_overlap"] == "not_applicable"
    assert evidence["manifest"]["fold_gold_sizes"] == [15, 15, 15]
    assert len(evidence["manifest"]["gold_class_counts_by_fold"]) == len(TARGET_COLUMNS)
    assert evidence["superseded_manifest"]["superseded"] is True
    assert "gold-0" not in text and "report-0" not in text


def test_v14_fails_the_superseded_manifest_and_tampered_public_evidence(tmp_path: Path) -> None:
    inventory_path, grouping_path, legacy_path = _write_fixture(tmp_path)

    legacy_evidence = verify_v14(legacy_path, inventory_path, grouping_path)
    legacy_statuses = {item["name"]: item["status"] for item in legacy_evidence["checks"]}

    assert legacy_evidence["status"] == "failed"
    assert legacy_statuses["current_generation_method"] == "failed"
    assert legacy_statuses["gold_fold_size_balance"] == "failed"
    assert legacy_statuses["deterministic_generation"] == "passed"

    run_fold_freeze(
        inventory_path,
        tmp_path / "v2",
        grouping_path=grouping_path,
        supersedes_manifest_path=legacy_path,
    )
    public_path = tmp_path / "v2" / "public" / "fold-evidence.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    public["fold_gold_sizes"] = [16, 15, 14]
    tampered_path = tmp_path / "tampered-public.json"
    tampered_path.write_text(json.dumps(public), encoding="utf-8")

    evidence = verify_v14(
        tmp_path / "v2" / "private" / "fold-manifest.json",
        inventory_path,
        grouping_path,
        public_evidence_path=tampered_path,
    )
    statuses = {item["name"]: item["status"] for item in evidence["checks"]}

    assert evidence["status"] == "failed"
    assert statuses["public_evidence_matches_manifest"] == "failed"
