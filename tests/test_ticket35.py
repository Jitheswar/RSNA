from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, UID

from rsna_knee.dicom import DecoderPolicy
from rsna_knee.geometry import GeometryPolicy
from rsna_knee.image_variant import TICKET31_RESNET18_VARIANT, image_model_training_config
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.model import build_study_model
from rsna_knee.preprocessing import PackingConfig, PreprocessConfig
from rsna_knee.real_inference import (
    IntegrationManifest,
    IntegrationSeries,
    IntegrationStudy,
)
from rsna_knee.resnet18 import RESNET18_FIXTURE_ENCODER_ID
from rsna_knee.series_cache import SeriesCache
from rsna_knee.targets import TARGET_COLUMNS
from rsna_knee.ticket35 import (
    Ticket35Error,
    Ticket35StorageError,
    build_ticket35_membership,
    estimate_batch_storage,
    process_ticket35_studies,
    validate_ticket35_membership,
)
from rsna_knee.ticket35_cli import _approve_cpu_batch


def _target_vector(*values: tuple[int, int]) -> tuple[int | None, ...]:
    result: list[int | None] = [None] * len(TARGET_COLUMNS)
    for index, value in values:
        result[index] = value
    return tuple(result)


def _series_row(
    study_uid: str,
    series_uid: str,
    *,
    complete: bool = True,
    frames: int = 3,
) -> dict[str, object]:
    syntax = str(ExplicitVRLittleEndian)
    return {
        "StudyInstanceUID": study_uid,
        "SeriesInstanceUID": series_uid,
        "complete": complete,
        "entry_names": [
            f"train_series/{study_uid}/{series_uid}/slice-{index}.dcm"
            for index in range(frames)
        ],
        "entry_count": frames,
        "effective_frame_count": frames,
        "uncompressed_bytes": frames * 1024,
        "compressed_bytes": frames * 512,
        "transfer_syntax_counts": {syntax: frames},
        "Anatomical_Plane": "sagittal",
        "Fluid_Sensitive": True,
        "Fat_Suppression": None,
    }


def test_membership_freezes_fold_safe_union_target_coverage_and_image_exclusions(tmp_path: Path) -> None:
    assignments = {
        "report-a": 0,
        "report-b": 1,
        "report-c": 2,
        "report-no-call": 1,
    }
    admissions = {
        0: {
            "report-b": _target_vector((0, 1)),
            "report-c": _target_vector((0, 0), (1, 1)),
        },
        1: {
            "report-a": _target_vector((0, 1)),
            "report-c": _target_vector((1, 0)),
        },
        2: {
            "report-a": _target_vector((0, 0)),
            "report-b": _target_vector((0, 1)),
        },
    }
    inventory_rows = (
        _series_row("report-a", "series-a"),
        _series_row("report-b", "series-b"),
        _series_row("report-c", "series-c", complete=False),
    )

    manifest = build_ticket35_membership(
        report_fold_by_uid=assignments,
        admitted_targets_by_fold=admissions,
        inventory_series=inventory_rows,
        existing_study_uids=("report-a", "gold-0"),
        held_out_gold_by_fold={0: ("gold-0",), 1: ("gold-1",), 2: ("gold-2",)},
        source_identity={
            "source_version": "ticket35-fixture-v1",
            "archive_fingerprint": "a" * 64,
            "archive_size_bytes": 100,
            "source_manifest_checksum": "b" * 64,
            "inventory_sha256": "c" * 64,
            "integration_manifest_sha256": "d" * 64,
            "fold_manifest_checksum": "e" * 64,
            "label_artifact_sha256": "f" * 64,
        },
        batch_size=1,
    )

    assert manifest["schema_version"] == "rsna-ticket35-expanded-image-membership-v2"
    assert manifest["cohort"]["admitted_report_study_count"] == 3
    assert manifest["cohort"]["image_ready_report_study_count"] == 2
    assert manifest["cohort"]["additional_vs_existing_manifest_count"] == 1
    assert manifest["cohort"]["no_admitted_call_report_study_count"] == 1
    assert manifest["cohort"]["image_excluded_report_study_uids"] == ["report-c"]
    assert [item["study_uids"] for item in manifest["batches"]] == [["report-a"], ["report-b"]]
    assert manifest["folds"][0]["held_out_gold_study_uids"] == ["gold-0"]
    fold_zero_coverage = manifest["folds"][0]["target_coverage"]
    assert fold_zero_coverage[0] == {
        "target": TARGET_COLUMNS[0],
        "supervised_entries": 1,
        "positive_entries": 1,
        "negative_entries": 0,
    }
    assert fold_zero_coverage[1]["supervised_entries"] == 0
    assert manifest["folds"][0]["supervised_target_entry_count"] == 1
    assert manifest["folds"][0]["admitted_supervised_target_entry_count_pre_image_exclusion"] == 3
    assert manifest["folds"][0]["admitted_target_coverage_pre_image_exclusion"][0] == {
        "target": TARGET_COLUMNS[0],
        "supervised_entries": 2,
        "positive_entries": 1,
        "negative_entries": 1,
    }
    assert manifest["folds"][0]["admitted_target_coverage_pre_image_exclusion"][1][
        "supervised_entries"
    ] == 1
    assert validate_ticket35_membership(manifest)["manifest_checksum"] == manifest["manifest_checksum"]
    changed = dict(manifest)
    changed["cohort"] = {**manifest["cohort"], "admitted_report_study_count": 99}
    with pytest.raises(Ticket35Error, match="checksum"):
        validate_ticket35_membership(changed)

    held_out_admission = {fold: dict(rows) for fold, rows in admissions.items()}
    held_out_admission[0]["report-a"] = _target_vector((0, 1))
    with pytest.raises(Ticket35Error, match="held-out fold"):
        build_ticket35_membership(
            report_fold_by_uid=assignments,
            admitted_targets_by_fold=held_out_admission,
            inventory_series=inventory_rows,
            existing_study_uids=("report-a", "gold-0"),
            held_out_gold_by_fold={0: ("gold-0",), 1: ("gold-1",), 2: ("gold-2",)},
            source_identity={
                "source_version": "ticket35-fixture-v1",
                "archive_fingerprint": "a" * 64,
                "archive_size_bytes": 100,
                "source_manifest_checksum": "b" * 64,
                "inventory_sha256": "c" * 64,
                "integration_manifest_sha256": "d" * 64,
                "fold_manifest_checksum": "e" * 64,
                "label_artifact_sha256": "f" * 64,
            },
            batch_size=1,
        )

    first_batch = manifest["batches"][0]
    assert isinstance(first_batch, dict)
    pending_approval = {
        "schema_version": "rsna-ticket35-cpu-preparation-approval-v1",
        "decision": "pending_owner_review",
        "approval_status": "pending_owner_decision",
        "scope": "ticket35-image-preparation-single-batch",
        "membership_checksum": manifest["manifest_checksum"],
        "batch_indices": [0],
        "batch_summary": {
            "report_only_studies": len(first_batch["study_uids"]),
            "series": first_batch["series_count"],
            "source_entries_and_effective_slices": first_batch["effective_frame_count"],
            "uncompressed_source_bytes": first_batch["uncompressed_source_bytes"],
            "compressed_source_bytes": first_batch["compressed_source_bytes"],
            "224px_float32_pixel_cache_bytes": first_batch["pixel_cache_bytes"],
        },
        "inventory_path": str((tmp_path / "inventory.json").resolve()),
        "reference_integration_manifest_path": str((tmp_path / "integration.json").resolve()),
        "source_archive_path": str((tmp_path / "source.zip").resolve()),
        "cache_root": str((tmp_path / "cache").resolve()),
        "source_version": "ticket35-fixture-v1",
        "archive_fingerprint": "a" * 64,
        "archive_size_bytes": 100,
        "external_spend_usd": 0,
        "gpu_count": 0,
        "model_use": False,
        "feature_extraction_or_prediction": False,
        "training": False,
        "storage_gate": {"soft_budget_bytes": 2_000_000_000_000},
        "resources": {
            "partition": "cluster1-cpu",
            "nodes": 1,
            "cpus_per_task": 16,
            "memory_gib": 64,
            "walltime_seconds": 21600,
            "maximum_concurrent_jobs": 1,
            "gpu_count": 0,
        },
    }
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(pending_approval), encoding="utf-8")
    with pytest.raises(Ticket35Error, match="exact Ticket 35 preparation request"):
        _approve_cpu_batch(
            approval_path,
            manifest,
            0,
            inventory_path=tmp_path / "inventory.json",
            reference_integration_path=tmp_path / "integration.json",
            source_path=tmp_path / "source.zip",
            cache_root=tmp_path / "cache",
        )

    approved = {
        **pending_approval,
        "decision": "approved",
        "approval_status": "approved_by_project_owner",
        "approved_by": "Project owner",
        "approved_at": "2026-09-26T11:46:40+05:30",
    }
    approval_path.write_text(json.dumps(approved), encoding="utf-8")
    accepted = _approve_cpu_batch(
        approval_path,
        manifest,
        0,
        inventory_path=tmp_path / "inventory.json",
        reference_integration_path=tmp_path / "integration.json",
        source_path=tmp_path / "source.zip",
        cache_root=tmp_path / "cache",
    )
    assert accepted["decision"] == "approved"


def test_batch_storage_estimate_counts_archive_once_and_rejects_over_budget() -> None:
    estimate = estimate_batch_storage(
        current_measured_bytes=1_000,
        source_archive_bytes=200,
        uncompressed_source_bytes=10,
        effective_frame_count=3,
        image_size=(2, 2),
        soft_budget_bytes=2_000,
    )

    assert estimate["current_project_bytes"] == 800
    assert estimate["source_archive_bytes"] == 200
    assert estimate["pixel_cache_bytes"] == 48
    assert estimate["temporary_bytes"] == 58
    assert estimate["projected_peak_bytes"] == 1_116
    assert estimate["within_soft_budget"] is True
    with pytest.raises(Ticket35StorageError, match="2 TB soft budget"):
        estimate_batch_storage(
            current_measured_bytes=1_000,
            source_archive_bytes=200,
            uncompressed_source_bytes=10,
            effective_frame_count=3,
            image_size=(2, 2),
            soft_budget_bytes=1_115,
        )


def _write_dicom(path: Path, study_uid: str, series_uid: str, sop_uid: str, position: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = FileMetaDataset()
    metadata.MediaStorageSOPClassUID = MRImageStorage
    metadata.MediaStorageSOPInstanceUID = UID(sop_uid)
    metadata.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=metadata, preamble=b"\0" * 128)
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = UID(sop_uid)
    dataset.StudyInstanceUID = UID(study_uid)
    dataset.SeriesInstanceUID = UID(series_uid)
    dataset.Modality = "MR"
    dataset.ImagePositionPatient = [0.0, 0.0, float(position)]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.ImageType = ["ORIGINAL", "PRIMARY"]
    dataset.SeriesDescription = "diagnostic sequence"
    dataset.Rows = 32
    dataset.Columns = 32
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    pixels = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32) + position
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)


def _real_fixture(tmp_path: Path) -> tuple[IntegrationManifest, Path, tuple[str, str]]:
    source = tmp_path / "source"
    study_uids = (
        "1.2.826.0.1.3680043.8.498.1001",
        "1.2.826.0.1.3680043.8.498.1002",
    )
    integration_studies = []
    for study_index, study_uid in enumerate(study_uids):
        series_uid = f"1.2.826.0.1.3680043.8.498.{2001 + study_index}"
        entries = []
        for slice_index in range(3):
            sop_uid = f"1.2.826.0.1.3680043.8.498.{3001 + study_index * 10 + slice_index}"
            entry = f"train_series/{study_uid}/{series_uid}/{sop_uid}.dcm"
            _write_dicom(source / entry, study_uid, series_uid, sop_uid, slice_index)
            entries.append(entry)
        integration_studies.append(
            IntegrationStudy(
                study_uid,
                (
                    IntegrationSeries(
                        study_instance_uid=study_uid,
                        series_instance_uid=series_uid,
                        entries=tuple(entries),
                        anatomical_plane="sagittal",
                        fluid_sensitive=True,
                        expected_transfer_syntaxes=(str(ExplicitVRLittleEndian),),
                        expected_dimensions=("32x32",),
                        expected_frame_counts=(1,),
                        expected_entry_count=3,
                    ),
                ),
            )
        )
    geometry = GeometryPolicy(minimum_usable_slices=3)
    manifest = IntegrationManifest(
        source_version="ticket35-fixture-v1",
        archive_fingerprint="fixture-archive",
        geometry_policy=geometry,
        decoder_policy=DecoderPolicy(require_modality=True),
        preprocess_config=PreprocessConfig(
            image_size=(224, 224),
            max_slices=8,
            max_series=1,
            representation_id="ticket31-resnet18-224-whole-field-v1",
            geometry_policy=geometry,
        ),
        studies=tuple(integration_studies),
        packing_config=PackingConfig(max_slices=8, max_series=1),
    )
    return manifest, source, study_uids


def test_real_study_batch_resumes_reuses_and_rebuilds_through_public_inference_boundary(
    tmp_path: Path,
) -> None:
    manifest, source, study_uids = _real_fixture(tmp_path)
    cache_root = tmp_path / "series-cache"
    configuration = image_model_training_config(TICKET31_RESNET18_VARIANT)
    configuration = replace(
        configuration,
        encoder_id=RESNET18_FIXTURE_ENCODER_ID,
        encoder_weights_sha256=None,
        preprocessing_config=manifest.preprocess_config,
        max_slices=8,
        max_series=1,
    )
    model = build_study_model(configuration.model_config())

    interrupted = process_ticket35_studies(
        manifest,
        source,
        cache_root,
        model=model,
        study_uids=(study_uids[0],),
        allow_local_smoke=True,
    )
    assert interrupted["prediction_count"] == 1
    assert interrupted["cache"]["misses"] == 1
    assert interrupted["studies"][0]["series_mask_count"] == 1
    assert interrupted["studies"][0]["slice_mask_count"] == 3
    assert len(interrupted["studies"][0]["prediction_scores"]) == 12

    resumed = process_ticket35_studies(
        manifest,
        source,
        cache_root,
        model=model,
        study_uids=study_uids,
        allow_local_smoke=True,
    )
    assert resumed["prediction_count"] == 2
    assert resumed["cache"]["hits"] == 1
    assert resumed["cache"]["misses"] == 1
    assert {item["study_instance_uid"] for item in resumed["studies"]} == set(study_uids)

    first_event = next(
        item
        for item in resumed["cache"]["events"]
        if item["study_instance_uid"] == study_uids[0]
    )
    cache = SeriesCache(cache_root)
    cache.entry_path(first_event["cache_key"]).joinpath("images.npy").write_bytes(b"incomplete")
    repaired = process_ticket35_studies(
        manifest,
        source,
        cache_root,
        model=model,
        study_uids=(study_uids[0],),
        allow_local_smoke=True,
    )
    assert repaired["cache"]["rebuilds"] == 1
    assert repaired["cache"]["events"][0]["reason"] == "rebuild"

    first_series = manifest.studies[0].series[0]
    _write_dicom(
        source / first_series.entries[0],
        study_uids[0],
        first_series.series_instance_uid,
        first_series.entries[0].rsplit("/", 1)[-1].removesuffix(".dcm"),
        3,
    )
    changed_source = process_ticket35_studies(
        manifest,
        source,
        cache_root,
        model=model,
        study_uids=(study_uids[0],),
        allow_local_smoke=True,
    )
    assert changed_source["cache"]["misses"] == 1
    assert changed_source["cache"]["events"][0]["cache_key"] != first_event["cache_key"]
