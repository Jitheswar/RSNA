from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

from rsna_knee.inventory import load_source_manifest


TARGETS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)


def _dicom_bytes(
    study_uid: str,
    series_uid: str,
    sop_uid: str,
    *,
    position: float = 0.0,
    rows: int = 2,
    columns: int = 2,
    series_description: str | None = None,
    image_type: tuple[str, ...] = ("ORIGINAL", "PRIMARY", "M"),
) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset("fixture.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.Modality = "MR"
    dataset.ImagePositionPatient = [0.0, 0.0, position]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.ImageType = list(image_type)
    dataset.PixelSpacing = [1.0, 1.0]
    dataset.SliceThickness = 1.0
    dataset.SpacingBetweenSlices = 1.0
    if series_description is not None:
        dataset.SeriesDescription = series_description
    dataset.InstanceNumber = 1
    dataset.Rows = rows
    dataset.Columns = columns
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = b"\0" * (rows * columns * 2)
    buffer = io.BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def _write_fixture_archive(path: Path, *, report_only_count: int = 1) -> None:
    studies = ("2.25.100", "2.25.200") + tuple(
        f"2.25.{300 + index}" for index in range(report_only_count)
    )
    series = tuple(f"2.25.{101 + index}" for index in range(len(studies)))
    labels = {study: {target: "" for target in TARGETS} for study in studies}
    labels[studies[0]] = {target: "0" for target in TARGETS}
    labels[studies[0]]["ACL"] = "1"
    labels[studies[1]] = {target: "0" for target in TARGETS}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=("StudyInstanceUID", "Report", *TARGETS),
            lineterminator="\n",
        )
        writer.writeheader()
        for study in studies:
            writer.writerow({"StudyInstanceUID": study, "Report": "private report", **labels[study]})
        archive.writestr("train.csv", output.getvalue())

        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "Fluid_Sensitive",
                "Fat_Suppression",
                "Anatomical_Plane",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        planes = ("Axial", "Coronal", "Sagittal")
        for index, (study, series_uid) in enumerate(zip(studies, series)):
            writer.writerow(
                {
                    "StudyInstanceUID": study,
                    "SeriesInstanceUID": series_uid,
                    "Fluid_Sensitive": str(index % 2),
                    "Fat_Suppression": str((index + 1) % 2),
                    "Anatomical_Plane": planes[index % len(planes)],
                }
            )
            archive.writestr(
                f"train_series/{study}/{series_uid}/{2.25}.{400 + index}.dcm",
                _dicom_bytes(study, series_uid, f"2.25.{400 + index}"),
            )
        archive.writestr("train_series.csv", output.getvalue())
        archive.writestr("train_series/ignored.txt", "not a DICOM file")


def test_load_source_manifest_preserves_descriptors_and_reference_split(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.zip"
    _write_fixture_archive(archive_path)

    manifest = load_source_manifest(archive_path)

    assert manifest.study_count == 3
    assert manifest.series_count == 3
    assert manifest.gold_study_uids == ("2.25.100", "2.25.200")
    assert manifest.report_only_study_uids == ("2.25.300",)
    descriptor = manifest.series_by_uid["2.25.103"]
    assert descriptor.anatomical_plane == "Sagittal"
    assert descriptor.fluid_sensitive is False
    assert descriptor.fat_suppression is True


def test_inventory_reads_headers_without_pixels_and_keeps_unreadable_files(tmp_path: Path) -> None:
    from rsna_knee.inventory import inventory_archive

    archive_path = tmp_path / "source.zip"
    _write_fixture_archive(archive_path)
    with zipfile.ZipFile(archive_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("train_series/2.25.300/2.25.103/broken.dcm", b"not DICOM")

    report = inventory_archive(
        archive_path,
        source_version="fixture-2026-09-11",
        expected_study_count=3,
        expected_gold_count=2,
    )

    assert report.archive_data_file_count == 5
    assert report.archive_dicom_file_count == 4
    assert report.readable_header_count == 3
    assert report.unreadable_header_count == 1
    assert report.source_counts_match_expectations is True
    assert report.full_corpus_reconciled is False
    assert report.malformed_archive_entries == ("train_series/ignored.txt",)
    series = report.series_by_uid["2.25.101"]
    assert series.transfer_syntax_counts == {"1.2.840.10008.1.2.1": 1}
    assert series.dimension_counts == {"2x2": 1}
    assert series.frame_count_counts == {1: 1}
    assert series.geometry_status == "complete"
    assert report.unreadable_headers[0].reason == "invalid_dicom"


def test_representative_selection_is_deterministic_and_includes_all_gold_studies(tmp_path: Path) -> None:
    from rsna_knee.inventory import inventory_archive, select_representative_batch

    archive_path = tmp_path / "selection-source.zip"
    _write_fixture_archive(archive_path, report_only_count=60)
    report = inventory_archive(
        archive_path,
        source_version="fixture-2026-09-11",
        expected_study_count=62,
        expected_gold_count=2,
    )

    selected = select_representative_batch(
        report,
        report_only_count=50,
        seed=17,
        supported_transfer_syntaxes=("1.2.840.10008.1.2.1",),
    )
    repeated = select_representative_batch(
        report,
        report_only_count=50,
        seed=17,
        supported_transfer_syntaxes=("1.2.840.10008.1.2.1",),
    )

    assert selected.gold_study_uids == ("2.25.100", "2.25.200")
    assert len(selected.report_only_study_uids) == 50
    assert len(selected.study_uids) == 52
    assert selected.study_uids == repeated.study_uids
    assert selected.entry_names == repeated.entry_names
    assert selected.unsupported_transfer_syntaxes == ()
    assert selected.storage_estimate.extracted_bytes > 0


def test_storage_budget_requires_explicit_approval_for_an_over_budget_batch(tmp_path: Path) -> None:
    from rsna_knee.inventory import enforce_storage_budget, estimate_storage, inventory_archive

    archive_path = tmp_path / "budget-source.zip"
    _write_fixture_archive(archive_path, report_only_count=60)
    report = inventory_archive(archive_path, source_version="fixture-2026-09-11")
    estimate = estimate_storage(report, ("2.25.100",), soft_budget_bytes=1)

    assert estimate.within_soft_budget is False
    try:
        enforce_storage_budget(estimate)
    except ValueError as exc:
        assert "soft budget" in str(exc)
    else:
        raise AssertionError("over-budget extraction should require approval")
    enforce_storage_budget(estimate, approved=True)


def test_public_inventory_summary_omits_identifier_bearing_details(tmp_path: Path) -> None:
    from rsna_knee.inventory import inventory_archive, write_inventory_reports

    archive_path = tmp_path / "source.zip"
    _write_fixture_archive(archive_path)
    report = inventory_archive(archive_path, source_version="fixture-2026-09-11")
    private_path = tmp_path / "private" / "inventory.json"
    public_path = tmp_path / "public" / "inventory.json"

    write_inventory_reports(report, private_path=private_path, public_path=public_path)

    public_text = public_path.read_text(encoding="utf-8")
    public = json.loads(public_text)
    assert "2.25.100" not in public_text
    assert "train_series/" not in public_text
    assert public["schema_version"] == "rsna-knee-inventory-v1"
    assert public["counts"]["source_studies"] == 3
    assert public["descriptor_counts"]["anatomical_plane"]["Axial"] == 1
    assert public["descriptor_counts"]["fluid_sensitive"]["false"] == 2
    assert json.loads(private_path.read_text(encoding="utf-8"))["source"]["gold_studies"] == 2


def test_run_inventory_can_publish_reports_and_admit_the_selected_fixture_batch(tmp_path: Path) -> None:
    from rsna_knee.inventory import run_inventory

    archive_path = tmp_path / "source.zip"
    _write_fixture_archive(archive_path, report_only_count=60)
    output_dir = tmp_path / "reports"
    extraction_path = tmp_path / "private-extraction"

    summary = run_inventory(
        archive_path,
        output_dir,
        source_version="fixture-2026-09-11",
        report_only_count=50,
        expected_study_count=62,
        expected_gold_count=2,
        supported_transfer_syntaxes=("1.2.840.10008.1.2.1",),
        extraction_path=extraction_path,
    )

    assert summary["ticket"] == "03"
    assert summary["extraction"]["verified"] is True
    assert (extraction_path / "train_series/2.25.100/2.25.101/2.25.400.dcm").exists()


def test_inventory_marks_duplicate_geometry_and_inconsistent_dimensions_as_blocked(tmp_path: Path) -> None:
    from rsna_knee.inventory import inventory_archive

    archive_path = tmp_path / "geometry-source.zip"
    _write_fixture_archive(archive_path)
    with zipfile.ZipFile(archive_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "train_series/2.25.100/2.25.101/2.25.401.dcm",
            _dicom_bytes("2.25.100", "2.25.101", "2.25.401", rows=3, columns=2),
        )

    report = inventory_archive(archive_path, source_version="fixture-2026-09-11")
    series = report.series_by_uid["2.25.101"]

    assert series.geometry_status == "duplicate"
    assert series.duplicate_position_count == 1
    assert series.dimension_counts == {"2x2": 1, "3x2": 1}
    assert series.is_complete is False


def test_inventory_rejects_inconsistent_slice_spacing(tmp_path: Path) -> None:
    from rsna_knee.inventory import inventory_archive

    archive_path = tmp_path / "spacing-source.zip"
    _write_fixture_archive(archive_path)
    with zipfile.ZipFile(archive_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "train_series/2.25.100/2.25.101/2.25.401.dcm",
            _dicom_bytes("2.25.100", "2.25.101", "2.25.401", position=1.0),
        )
        archive.writestr(
            "train_series/2.25.100/2.25.101/2.25.402.dcm",
            _dicom_bytes("2.25.100", "2.25.101", "2.25.402", position=3.0),
        )

    report = inventory_archive(archive_path, source_version="fixture-2026-09-11")
    series = report.series_by_uid["2.25.101"]

    assert series.spacing_inconsistent_count == 1
    assert series.geometry_status == "inconsistent"
    assert series.is_complete is False


def test_inventory_excludes_reliably_named_localizer_series(tmp_path: Path) -> None:
    from rsna_knee.inventory import inventory_archive

    archive_path = tmp_path / "localizer-source.zip"
    _write_fixture_archive(archive_path)
    with zipfile.ZipFile(archive_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "train_series/2.25.100/2.25.101/2.25.401.dcm",
            _dicom_bytes(
                "2.25.100",
                "2.25.101",
                "2.25.401",
                position=1.0,
                series_description="3 plane localizer",
                image_type=("LOCALIZER",),
            ),
        )

    report = inventory_archive(archive_path, source_version="fixture-2026-09-11")
    series = report.series_by_uid["2.25.101"]

    assert series.diagnostic_status == "excluded_non_diagnostic"
    assert series.is_complete is False


def test_partial_reference_labels_are_neither_gold_nor_report_only() -> None:
    from rsna_knee.inventory import SourceStudy

    study = SourceStudy("2.25.partial", True, (1,) + (None,) * 11)

    assert study.has_reference_labels is False
    assert study.has_partial_reference_labels is True


def test_run_inventory_refuses_extraction_without_explicit_decoder_support(tmp_path: Path) -> None:
    from rsna_knee.inventory import SelectionError, run_inventory

    archive_path = tmp_path / "unsupported-source.zip"
    _write_fixture_archive(archive_path, report_only_count=50)

    with pytest.raises(SelectionError, match="transfer syntaxes"):
        run_inventory(
            archive_path,
            tmp_path / "reports",
            source_version="fixture-2026-09-11",
            report_only_count=50,
            expected_study_count=52,
            expected_gold_count=2,
            extraction_path=tmp_path / "extraction",
        )

    assert not (tmp_path / "extraction").exists()


def test_run_inventory_publishes_a_failed_completion_record_on_extraction_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rsna_knee.archive import ArchiveIntegrityError
    import rsna_knee.inventory as inventory_module

    archive_path = tmp_path / "failed-extraction-source.zip"
    _write_fixture_archive(archive_path, report_only_count=50)

    def fail_extraction(*args: object, **kwargs: object) -> object:
        raise ArchiveIntegrityError("simulated extraction precondition failure")

    monkeypatch.setattr(inventory_module, "extract_batch", fail_extraction)

    with pytest.raises(ArchiveIntegrityError, match="simulated"):
        inventory_module.run_inventory(
            archive_path,
            tmp_path / "reports",
            source_version="fixture-2026-09-11",
            report_only_count=50,
            expected_study_count=52,
            expected_gold_count=2,
            supported_transfer_syntaxes=("1.2.840.10008.1.2.1",),
            extraction_path=tmp_path / "extraction",
        )

    private = json.loads((tmp_path / "reports/private/extraction.json").read_text(encoding="utf-8"))
    public = json.loads((tmp_path / "reports/public/extraction.json").read_text(encoding="utf-8"))
    assert private["verified"] is False
    assert public["verified"] is False


def test_manifest_rejects_rows_with_missing_columns(tmp_path: Path) -> None:
    archive_path = tmp_path / "malformed-source.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("train.csv", "StudyInstanceUID,Report,ACL\n2.25.1,report\n")
        archive.writestr(
            "train_series.csv",
            "StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,Anatomical_Plane\n",
        )

    with pytest.raises(ValueError, match="wrong number of columns"):
        load_source_manifest(archive_path)


def test_supported_transfer_syntaxes_are_trimmed_before_admission(tmp_path: Path) -> None:
    from rsna_knee.inventory import inventory_archive, select_representative_batch

    archive_path = tmp_path / "syntax-source.zip"
    _write_fixture_archive(archive_path, report_only_count=50)
    report = inventory_archive(archive_path, source_version="fixture-2026-09-11")

    selection = select_representative_batch(
        report,
        report_only_count=50,
        supported_transfer_syntaxes=(" 1.2.840.10008.1.2.1 ",),
    )

    assert selection.unsupported_transfer_syntaxes == ()
