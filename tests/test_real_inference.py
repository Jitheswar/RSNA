from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, UID

from rsna_knee.cache_benchmark import run_cache_benchmark
from rsna_knee.dicom import DecoderPolicy
from rsna_knee.folds import freeze_folds, write_fold_reports
from rsna_knee.geometry import GeometryPolicy
from rsna_knee.model import HierarchicalStudyModel, ModelConfig
from rsna_knee.preprocessing import PreprocessConfig
from rsna_knee.inventory import (
    SourceManifest,
    SourceStudy,
    inventory_archive,
    select_representative_batch,
    write_inventory_reports,
    write_selection_reports,
)
from rsna_knee.series_cache import SeriesCacheError
from rsna_knee.targets import NUM_TARGETS
from rsna_knee.real_inference import (
    IntegrationManifest,
    IntegrationSeries,
    IntegrationStudy,
    RealIntegrationError,
    load_integration_manifest,
    load_real_study,
    run_real_inference,
    build_integration_manifest,
    main as real_inference_main,
    write_integration_manifest,
)


def _write_slice(
    root: Path,
    study_uid: str,
    series_uid: str,
    sop_uid: str,
    position: float,
    *,
    image_type: tuple[str, ...] = ("ORIGINAL", "PRIMARY"),
) -> str:
    relative = Path("train_series") / study_uid / series_uid / f"{sop_uid}.dcm"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = UID(sop_uid)
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = UID(sop_uid)
    dataset.StudyInstanceUID = UID(study_uid)
    dataset.SeriesInstanceUID = UID(series_uid)
    dataset.Modality = "MR"
    dataset.ImagePositionPatient = [0.0, 0.0, position]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.ImageType = list(image_type)
    dataset.SeriesDescription = "diagnostic sequence"
    dataset.Rows = 2
    dataset.Columns = 3
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint16).tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return relative.as_posix()


def _manifest(study_uid: str, series: tuple[IntegrationSeries, ...]) -> IntegrationManifest:
    return _manifest_for_studies((IntegrationStudy(study_uid, series),))


def _manifest_for_studies(studies: tuple[IntegrationStudy, ...]) -> IntegrationManifest:
    geometry = GeometryPolicy(minimum_usable_slices=3)
    return IntegrationManifest(
        source_version="fixture-real-v1",
        archive_fingerprint="fixture-fingerprint",
        geometry_policy=geometry,
        decoder_policy=DecoderPolicy(require_modality=True),
        preprocess_config=PreprocessConfig(
            image_size=(8, 8),
            max_slices=8,
            max_series=4,
            geometry_policy=geometry,
        ),
        studies=studies,
    )


def test_private_manifest_round_trip_and_loader_preserve_source_mapping(tmp_path: Path) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.1"
    series_uid = "1.2.826.0.1.3680043.8.498.2"
    entries = tuple(
        _write_slice(tmp_path, study_uid, series_uid, f"1.2.826.0.1.3680043.8.498.{index}", float(index))
        for index in (3, 1, 2)
    )
    manifest = _manifest(
        study_uid,
        (
            IntegrationSeries(
                study_uid,
                series_uid,
                entries=entries,
                anatomical_plane="Sagittal",
                fluid_sensitive=True,
                fat_suppression=True,
            ),
        ),
    )
    manifest_path = tmp_path / "private" / "manifest.json"
    write_integration_manifest(manifest, manifest_path)

    loaded_manifest = load_integration_manifest(manifest_path)
    loaded = load_real_study(
        loaded_manifest,
        loaded_manifest.studies[0],
        tmp_path,
        allow_unverified_directory=True,
    )

    assert loaded.study.series[0].anatomical_plane == "Sagittal"
    assert loaded.study.series[0].fluid_sensitive is True
    assert loaded.study.series[0].fat_suppression is True
    assert loaded.study.series[0].source_entry_names == entries
    assert loaded.series_results[0].decoded_slice_count == 3
    assert loaded.series_results[0].complete is True
    assert loaded.study.series[0].slices[0].filename.endswith(".dcm")
    assert loaded.study.series[0].slices[0].source_content_sha256 == hashlib.sha256(
        (tmp_path / entries[0]).read_bytes()
    ).hexdigest()
    assert loaded_manifest.manifest_checksum

    with pytest.raises(RealIntegrationError, match="source ZIP"):
        load_real_study(loaded_manifest, loaded_manifest.studies[0], tmp_path)

    altered = IntegrationStudy(
        study_uid,
        (
            IntegrationSeries(
                study_uid,
                series_uid,
                entries=tuple(reversed(entries)),
                anatomical_plane="Sagittal",
                fluid_sensitive=True,
                fat_suppression=True,
            ),
        ),
    )
    with pytest.raises(RealIntegrationError, match="checksum-covered"):
        load_real_study(
            loaded_manifest,
            altered,
            tmp_path,
            allow_unverified_directory=True,
        )


def test_real_inference_uses_shared_model_path_and_writes_public_safe_evidence(tmp_path: Path) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.11"
    series_uid = "1.2.826.0.1.3680043.8.498.12"
    entries = tuple(
        _write_slice(tmp_path, study_uid, series_uid, f"1.2.826.0.1.3680043.8.498.{index}", float(index))
        for index in range(3)
    )
    manifest_path = tmp_path / "manifest.json"
    write_integration_manifest(
        _manifest(
            study_uid,
            (IntegrationSeries(study_uid, series_uid, entries=entries),),
        ),
        manifest_path,
    )
    output_path = tmp_path / "submission.csv"
    evidence_path = tmp_path / "public" / "evidence.json"

    evidence = run_real_inference(
        manifest_path,
        tmp_path,
        output_path,
        evidence_path=evidence_path,
        cache_root=tmp_path / "cache",
        allow_local_smoke=True,
    )

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 2
    assert evidence["status"] == "completed_real_integration"
    assert evidence["counts"]["input_studies"] == 1
    assert evidence["counts"]["fallback_studies"] == 0
    assert evidence["counts"]["scores_finite"] is True
    assert evidence["counts"]["scores_bounded"] is True
    assert evidence["cache_boundary"]["series_cache_enabled"] is True
    public_text = evidence_path.read_text(encoding="utf-8")
    assert study_uid not in public_text
    assert series_uid not in public_text
    assert "train_series/" not in public_text
    assert "/home/" not in public_text
    assert "slurm_job_id" not in public_text
    assert json.loads(public_text)["policies"]["decoder"]["supported_transfer_syntaxes"]


def test_real_loader_excludes_reliably_identified_localizer_but_keeps_diagnostic_series(
    tmp_path: Path,
) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.21"
    diagnostic_uid = "1.2.826.0.1.3680043.8.498.22"
    localizer_uid = "1.2.826.0.1.3680043.8.498.23"
    diagnostic_entries = tuple(
        _write_slice(tmp_path, study_uid, diagnostic_uid, f"1.2.826.0.1.3680043.8.498.{index}", float(index))
        for index in range(3)
    )
    localizer_entries = tuple(
        _write_slice(
            tmp_path,
            study_uid,
            localizer_uid,
            f"1.2.826.0.1.3680043.8.498.{index}",
            float(index),
            image_type=("LOCALIZER",),
        )
        for index in range(3)
    )
    loaded = load_real_study(
        _manifest(
            study_uid,
            (
                IntegrationSeries(study_uid, diagnostic_uid, entries=diagnostic_entries),
                IntegrationSeries(study_uid, localizer_uid, entries=localizer_entries),
            ),
        ),
        IntegrationStudy(
            study_uid,
            (
                IntegrationSeries(study_uid, diagnostic_uid, entries=diagnostic_entries),
                IntegrationSeries(study_uid, localizer_uid, entries=localizer_entries),
            ),
        ),
        tmp_path,
        allow_unverified_directory=True,
    )

    assert tuple(item.series_instance_uid for item in loaded.study.series) == (diagnostic_uid,)
    assert loaded.series_results[1].status == "excluded_non_diagnostic"
    assert loaded.series_results[1].warnings == ("excluded_non_diagnostic",)


def test_manifest_builder_rechecks_decoder_support_against_ticket03_selection(tmp_path: Path) -> None:
    from test_inventory import _write_fixture_archive

    archive_path = tmp_path / "source.zip"
    _write_fixture_archive(archive_path, report_only_count=60)
    inventory = inventory_archive(
        archive_path,
        source_version="fixture-real-v1",
        expected_study_count=62,
        expected_gold_count=2,
    )
    selection = select_representative_batch(inventory, report_only_count=50)
    inventory_path = tmp_path / "private" / "inventory.json"
    selection_path = tmp_path / "private" / "selection.json"
    write_inventory_reports(
        inventory,
        private_path=inventory_path,
        public_path=tmp_path / "public" / "inventory.json",
    )
    inventory_payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory_payload["reconciliation"]["full_corpus_reconciled"] = True
    inventory_path.write_text(json.dumps(inventory_payload), encoding="utf-8")
    write_selection_reports(
        selection,
        private_path=selection_path,
        public_path=tmp_path / "public" / "selection.json",
    )

    manifest = build_integration_manifest(
        inventory_path,
        selection_path,
        tmp_path / "private" / "manifest.json",
        geometry_policy=GeometryPolicy(minimum_usable_slices=1),
        decoder_policy=DecoderPolicy(),
    )

    assert manifest.source_version == "fixture-real-v1"
    assert len(manifest.studies) == 52
    assert manifest.series_count == 52
    assert manifest.decoder_policy.supported_transfer_syntaxes


def test_real_loader_rejects_a_partial_series_with_typed_file_diagnostics(tmp_path: Path) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.31"
    series_uid = "1.2.826.0.1.3680043.8.498.32"
    entries = tuple(
        _write_slice(tmp_path, study_uid, series_uid, f"1.2.826.0.1.3680043.8.498.{index}", float(index))
        for index in range(3)
    )
    (tmp_path / entries[-1]).unlink()
    manifest = _manifest(study_uid, (IntegrationSeries(study_uid, series_uid, entries=entries),))

    loaded = load_real_study(
        manifest,
        manifest.studies[0],
        tmp_path,
        allow_unverified_directory=True,
    )

    assert loaded.study.series == ()
    assert loaded.series_results[0].status == "rejected"
    assert loaded.series_results[0].complete is False
    assert loaded.series_results[0].failures[0].reason == "damaged_input"


def test_real_inference_records_geometry_rejection_and_keeps_the_study_row(tmp_path: Path) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.41"
    series_uid = "1.2.826.0.1.3680043.8.498.42"
    entries = tuple(
        _write_slice(tmp_path, study_uid, series_uid, f"1.2.826.0.1.3680043.8.498.{index}", position)
        for index, position in enumerate((0.0, 1.0, 3.0))
    )
    manifest_path = tmp_path / "manifest.json"
    write_integration_manifest(
        _manifest(study_uid, (IntegrationSeries(study_uid, series_uid, entries=entries),)),
        manifest_path,
    )

    private_evidence_path = tmp_path / "private" / "geometry.json"
    evidence = run_real_inference(
        manifest_path,
        tmp_path,
        tmp_path / "submission.csv",
        private_evidence_path=private_evidence_path,
        evidence_ticket="07",
        allow_local_smoke=True,
    )

    assert evidence["counts"]["fallback_studies"] == 1
    assert evidence["counts"]["prediction_rejected_series"] == 1
    assert evidence["counts"]["output_rows"] == 1
    assert evidence["recovery"]["failure_reasons"]["preprocessing_or_geometry"] == 1
    private = json.loads(private_evidence_path.read_text(encoding="utf-8"))
    rejection = private["studies"][0]["recovery"]["series_rejections"][0]
    assert rejection["affected_files"] == list(entries)
    assert rejection["decoder"] == "preprocessing_or_geometry"
    assert rejection["error"] == "inconsistent slice spacing"
    assert rejection["recovery"] == "study_fallback_reference_prior"


def test_real_inference_recovers_a_partially_corrupt_series_and_logs_each_attempt_privately(
    tmp_path: Path,
) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.61"
    series_uid = "1.2.826.0.1.3680043.8.498.62"
    entries = tuple(
        _write_slice(tmp_path, study_uid, series_uid, f"1.2.826.0.1.3680043.8.498.{index}", float(index))
        for index in range(4)
    )
    (tmp_path / entries[-1]).write_bytes(b"truncated-dicom")
    manifest_path = tmp_path / "manifest.json"
    write_integration_manifest(
        _manifest(study_uid, (IntegrationSeries(study_uid, series_uid, entries=entries),)),
        manifest_path,
    )
    private_evidence_path = tmp_path / "private" / "recovery.json"

    evidence = run_real_inference(
        manifest_path,
        tmp_path,
        tmp_path / "submission.csv",
        private_evidence_path=private_evidence_path,
        evidence_ticket="07",
        allow_local_smoke=True,
    )

    assert evidence["ticket"] == "07"
    assert evidence["counts"]["fallback_studies"] == 0
    assert evidence["counts"]["output_rows"] == 1
    assert evidence["recovery"]["full_study_fallback"]["numerator"] == 0
    assert evidence["recovery"]["full_study_fallback"]["denominator"] == 1
    assert evidence["recovery"]["slice_loss"]["expected"] == 4
    assert evidence["recovery"]["slice_loss"]["decoded"] == 3
    assert evidence["recovery"]["series_loss"]["usable"] == 1

    private = json.loads(private_evidence_path.read_text(encoding="utf-8"))
    series = private["studies"][0]["series"][0]
    assert series["status"] == "accepted_partial"
    assert series["decoded_slice_count"] == 3
    assert series["failures"][0]["source_entry_name"] == entries[-1]
    assert series["failures"][0]["decoder"] == "pydicom"
    assert series["failures"][0]["attempts"] == ["pydicom"]
    assert series["failures"][0]["recovery"] == "skip_damaged_input_continue_with_survivors"
    assert private["studies"][0]["recovery"]["remaining_usable_series"] == 1
    assert private["studies"][0]["recovery"]["used_fallback"] is False


def test_ticket05_real_inference_keeps_strict_series_admission_by_default(tmp_path: Path) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.63"
    series_uid = "1.2.826.0.1.3680043.8.498.64"
    entries = tuple(
        _write_slice(tmp_path, study_uid, series_uid, f"1.2.826.0.1.3680043.8.498.{index}", float(index))
        for index in range(3)
    )
    (tmp_path / entries[-1]).write_bytes(b"truncated-dicom")
    manifest_path = tmp_path / "manifest.json"
    write_integration_manifest(
        _manifest(study_uid, (IntegrationSeries(study_uid, series_uid, entries=entries),)),
        manifest_path,
    )

    evidence = run_real_inference(
        manifest_path,
        tmp_path,
        tmp_path / "submission.csv",
        allow_local_smoke=True,
    )

    assert evidence["ticket"] == "05"
    assert evidence["counts"]["fallback_studies"] == 1
    assert evidence["counts"]["usable_series"] == 0
    assert evidence["counts"]["series_with_decode_failures"] == 1


def test_real_inference_keeps_rows_for_missing_series_and_all_invalid_studies_using_reference_priors(
    tmp_path: Path,
) -> None:
    valid_uid = "1.2.826.0.1.3680043.8.498.71"
    missing_uid = "1.2.826.0.1.3680043.8.498.72"
    empty_uid = "1.2.826.0.1.3680043.8.498.73"
    series_uid = "1.2.826.0.1.3680043.8.498.74"
    entries = tuple(
        _write_slice(tmp_path, valid_uid, series_uid, f"1.2.826.0.1.3680043.8.498.{index}", float(index))
        for index in range(3)
    )
    missing_series_uid = "1.2.826.0.1.3680043.8.498.75"
    missing_entries = tuple(
        f"train_series/{missing_uid}/{missing_series_uid}/missing-{index}.dcm"
        for index in range(3)
    )
    studies = (
        IntegrationStudy(valid_uid, (IntegrationSeries(valid_uid, series_uid, entries=entries),)),
        IntegrationStudy(
            missing_uid,
            (IntegrationSeries(missing_uid, missing_series_uid, entries=missing_entries),),
        ),
        IntegrationStudy(empty_uid, ()),
    )
    manifest_path = tmp_path / "manifest.json"
    write_integration_manifest(_manifest_for_studies(studies), manifest_path)
    fallback_reference_labels = {
        "gold": (1,) + (0,) * (NUM_TARGETS - 1),
        "weak-report": (0,) * NUM_TARGETS,
    }
    output_path = tmp_path / "submission.csv"
    private_evidence_path = tmp_path / "private" / "recovery.json"

    evidence = run_real_inference(
        manifest_path,
        tmp_path,
        output_path,
        private_evidence_path=private_evidence_path,
        fallback_reference_labels=fallback_reference_labels,
        fallback_eligible_study_uids=("gold",),
        evidence_ticket="07",
        allow_local_smoke=True,
    )

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 4
    assert evidence["counts"]["fallback_studies"] == 2
    assert evidence["recovery"]["full_study_fallback"]["numerator"] == 2
    assert evidence["recovery"]["full_study_fallback"]["denominator"] == 3
    assert evidence["recovery"]["full_study_fallback"]["rate"] == pytest.approx(2 / 3)
    assert evidence["recovery"]["series_loss"]["expected"] == 2
    assert evidence["recovery"]["series_loss"]["usable"] == 1
    assert evidence["recovery"]["failure_reasons"]["damaged_input"] == 3
    assert evidence["recovery"]["failure_reasons"]["insufficient_slices"] == 1
    assert "preprocessing_or_geometry" not in evidence["recovery"]["failure_reasons"]
    assert evidence["fallback_prior"]["targets"]["ACL"]["score"] == pytest.approx(2 / 3)
    assert evidence["fallback_prior"]["targets"]["MCL"]["score"] == pytest.approx(1 / 3)

    private = json.loads(private_evidence_path.read_text(encoding="utf-8"))
    assert private["fallback_prior"]["eligible_study_uids"] == ["gold"]
    by_uid = {item["StudyInstanceUID"]: item for item in private["studies"]}
    assert by_uid[missing_uid]["recovery"]["used_fallback"] is True
    assert by_uid[missing_uid]["recovery"]["remaining_usable_series"] == 0
    assert by_uid[empty_uid]["recovery"]["used_fallback"] is True
    assert by_uid[empty_uid]["recovery"]["manifest_series_count"] == 0
    missing_failures = by_uid[missing_uid]["series"][0]["failures"]
    assert [item["affected_files"] for item in missing_failures[:3]] == [
        [entry] for entry in missing_entries
    ]
    minimum_failure = next(item for item in missing_failures if item["reason"] == "insufficient_slices")
    assert minimum_failure["affected_files"] == list(missing_entries)
    assert evidence["recovery"]["fallback_investigation"]["complete"] is False
    assert evidence["recovery"]["fallback_investigation"]["promotion_blocked_until_complete"] is True
    assert by_uid[missing_uid]["recovery"]["fallback_investigation"]["status"] == "pending_operator_review"


def test_unexpected_decoder_programming_failure_is_not_converted_to_study_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.81"
    series_uid = "1.2.826.0.1.3680043.8.498.82"
    entries = tuple(
        _write_slice(tmp_path, study_uid, series_uid, f"1.2.826.0.1.3680043.8.498.{index}", float(index))
        for index in range(3)
    )
    manifest_path = tmp_path / "manifest.json"
    write_integration_manifest(
        _manifest(study_uid, (IntegrationSeries(study_uid, series_uid, entries=entries),)),
        manifest_path,
    )

    def fail_unexpectedly(*args: object, **kwargs: object) -> object:
        raise RuntimeError("programming failure")

    monkeypatch.setattr("rsna_knee.real_inference.decode_stream", fail_unexpectedly)
    with pytest.raises(RuntimeError, match="programming failure"):
        run_real_inference(
            manifest_path,
            tmp_path,
            tmp_path / "submission.csv",
            allow_local_smoke=True,
        )


def test_recovery_cli_extracts_only_gold_rows_from_a_fold_manifest(tmp_path: Path) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.91"
    series_uid = "1.2.826.0.1.3680043.8.498.92"
    entries = tuple(
        _write_slice(tmp_path, study_uid, series_uid, f"1.2.826.0.1.3680043.8.498.{index}", float(index))
        for index in range(3)
    )
    manifest_path = tmp_path / "manifest.json"
    write_integration_manifest(
        _manifest(study_uid, (IntegrationSeries(study_uid, series_uid, entries=entries),)),
        manifest_path,
    )
    label_source = SourceManifest(
        "train",
        tuple(
            SourceStudy(
                f"reference-{index:02d}",
                True,
                tuple((index + target_index) % 2 for target_index in range(NUM_TARGETS)),
            )
            for index in range(9)
        )
        + tuple(
            SourceStudy(f"report-{index:02d}", True, (None,) * NUM_TARGETS)
            for index in range(6)
        ),
        (),
    )
    fold_manifest = freeze_folds(label_source, source_version="fixture-v1", seed=17)
    labels_path = tmp_path / "private" / "fold-manifest.json"
    write_fold_reports(
        fold_manifest,
        private_path=labels_path,
        public_path=tmp_path / "public" / "fold-evidence.json",
    )
    private_evidence_path = tmp_path / "private" / "evidence.json"

    exit_code = real_inference_main(
        [
            "--manifest",
            str(manifest_path),
            "--source",
            str(tmp_path),
            "--output",
            str(tmp_path / "submission.csv"),
            "--private-evidence",
            str(private_evidence_path),
            "--evidence-ticket",
            "07",
            "--fallback-reference-labels",
            str(labels_path),
            "--fallback-held-out-fold",
            "1",
            "--allow-local-smoke",
        ]
    )

    assert exit_code == 0
    private = json.loads(private_evidence_path.read_text(encoding="utf-8"))
    expected_eligible = sorted(
        item.study_instance_uid
        for item in fold_manifest.gold_memberships
        if item.fold != 1
    )
    assert private["fallback_prior"]["eligible_study_uids"] == expected_eligible
    assert not set(private["fallback_prior"]["eligible_study_uids"]).intersection(
        item.study_instance_uid
        for item in fold_manifest.gold_memberships
        if item.fold == 1
    )


def test_real_inference_fails_on_model_configuration_errors_instead_of_using_fallback(
    tmp_path: Path,
) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.101"
    series_uid = "1.2.826.0.1.3680043.8.498.102"
    entries = tuple(
        _write_slice(tmp_path, study_uid, series_uid, f"1.2.826.0.1.3680043.8.498.{index}", float(index))
        for index in range(3)
    )
    manifest_path = tmp_path / "manifest.json"
    write_integration_manifest(
        _manifest(study_uid, (IntegrationSeries(study_uid, series_uid, entries=entries),)),
        manifest_path,
    )

    with pytest.raises(ValueError, match="image sizes must agree"):
        run_real_inference(
            manifest_path,
            tmp_path,
            tmp_path / "submission.csv",
            model=HierarchicalStudyModel(ModelConfig(image_size=(16, 16))),
            allow_local_smoke=True,
        )


def test_real_inference_fails_on_cache_root_configuration_errors_instead_of_using_fallback(
    tmp_path: Path,
) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.111"
    series_uid = "1.2.826.0.1.3680043.8.498.112"
    entries = tuple(
        _write_slice(tmp_path, study_uid, series_uid, f"1.2.826.0.1.3680043.8.498.{index}", float(index))
        for index in range(3)
    )
    manifest_path = tmp_path / "manifest.json"
    write_integration_manifest(
        _manifest(study_uid, (IntegrationSeries(study_uid, series_uid, entries=entries),)),
        manifest_path,
    )
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    cache_root.chmod(0o700)
    (cache_root / "unrelated-file").write_text("not a disposable cache", encoding="utf-8")

    with pytest.raises(SeriesCacheError, match="new, empty"):
        run_real_inference(
            manifest_path,
            tmp_path,
            tmp_path / "submission.csv",
            cache_root=cache_root,
            allow_local_smoke=True,
        )


def test_cache_benchmark_proves_sampling_reuses_materialized_series(tmp_path: Path) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.51"
    series_uid = "1.2.826.0.1.3680043.8.498.52"
    entries = tuple(
        _write_slice(tmp_path, study_uid, series_uid, f"1.2.826.0.1.3680043.8.498.{index}", float(index))
        for index in range(3)
    )
    manifest_path = tmp_path / "manifest.json"
    write_integration_manifest(
        _manifest(study_uid, (IntegrationSeries(study_uid, series_uid, entries=entries),)),
        manifest_path,
    )

    evidence = run_cache_benchmark(
        manifest_path,
        tmp_path,
        tmp_path / "cache",
        tmp_path / "public" / "cache-evidence.json",
        workers=2,
        allow_local_smoke=True,
    )

    assert evidence["counts"]["accepted_series"] == 1
    assert evidence["counts"]["sampling_key_equivalence"] == 1
    assert evidence["runs"]["eight_slice_materialization"]["cache_misses"] == 1
    assert evidence["runs"]["sixteen_slice_reuse"]["cache_hits"] == 1
