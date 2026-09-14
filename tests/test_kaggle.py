from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, UID

import rsna_knee.kaggle as kaggle_module
from rsna_knee.candidate import freeze_candidate
from rsna_knee.kaggle import (
    KaggleDeliveryError,
    audit_kaggle_runtime_wheel,
    discover_kaggle_inputs,
    load_kaggle_delivery,
    load_kaggle_promotion,
    prepare_kaggle_delivery,
    promote_kaggle_candidate,
    record_kaggle_verification,
    run_kaggle_baseline,
    validate_kaggle_runtime_pins,
)
from rsna_knee.model import HierarchicalStudyModel, ModelConfig


def _write_checkpoint(path: Path) -> None:
    model = HierarchicalStudyModel(ModelConfig(image_size=(16, 16), channels=(2, 4, 8)))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "rsna-knee-checkpoint-v2",
            "checkpoint_kind": "final_selected",
            "checkpoint_reason": "final_selection",
            "fold": 0,
            "run_id": "run-0",
            "config_fingerprint": "config-0",
            "config": {
                "model": {
                    "image_size": [16, 16],
                    "channels": [2, 4, 8],
                    "input_channels": 1,
                    "num_targets": 12,
                },
                "packing": {"max_slices": 8, "max_series": 16},
                "preprocessing": {
                    "schema_version": "rsna-knee-preprocess-v1",
                    "image_size": [16, 16],
                    "geometry_policy": {
                        "policy_version": "rsna-knee-geometry-v2",
                        "minimum_usable_slices": 3,
                        "orientation_tolerance": 0.001,
                        "position_tolerance": 0.0001,
                        "spacing_tolerance": 0.001,
                        "require_consistent_dimensions": True,
                        "require_consistent_orientation": True,
                    },
                    "normalization_domain": "valid_slice",
                    "normalization_percentiles": [0, 100],
                    "use_voi_lut": False,
                    "interpolation": "nearest",
                    "padding_fill_value": 0,
                    "output_dtype": "float32",
                },
            },
            "model_state_dict": model.state_dict(),
            "identity": {
                "source_version": "synthetic-source-v1",
                "source_fingerprint": "synthetic-source-fingerprint",
                "fold_manifest_checksum": "synthetic-fold-checksum",
                "integration_manifest_checksum": "synthetic-integration-checksum",
            },
        },
        path,
    )


def _write_notebook(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": ["Write submission.csv from the offline candidate."],
                    },
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": ["# candidate\n", "# submission.csv\n"],
                    },
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )


def _make_candidate(tmp_path: Path) -> tuple[Path, dict[str, object], Path, Path]:
    checkpoint = tmp_path / "fold-0.pt"
    _write_checkpoint(checkpoint)
    oof = tmp_path / "oof-evidence.json"
    oof.write_text(
        json.dumps(
            {
                "schema_version": "rsna-knee-oof-evaluation-v2",
                "fold_manifest_checksum": "synthetic-fold-checksum",
                "fold_manifest_superseded": False,
                "prediction_count": 12,
                "run_ids": ["run-0"],
            }
        ),
        encoding="utf-8",
    )
    requirements = tmp_path / "requirements.lock"
    requirements.write_text(
        "numpy==2.2.6\npydicom==3.0.1\ntorch==2.7.1+cpu\n",
        encoding="utf-8",
    )
    candidate_notebook = tmp_path / "candidate.ipynb"
    _write_notebook(candidate_notebook)
    bundle = freeze_candidate(
        tmp_path / "candidate",
        model_checkpoints=(checkpoint,),
        oof_evidence_path=oof,
        code_root=Path("src/rsna_knee"),
        cluster_requirements_path=requirements,
        offline_requirements_path=requirements,
        notebook_path=candidate_notebook,
        candidate_id="candidate-kaggle-test",
        code_revision="test-revision",
    )
    evidence = {
        "schema_version": "rsna-knee-real-integration-evidence-v1",
        "ticket": "11",
        "status": "completed_real_integration",
        "candidate": {
            "candidate_id": bundle.candidate_id,
            "candidate_manifest_checksum": bundle.manifest_checksum,
        },
        "counts": {
            "input_studies": 1,
            "output_rows": 1,
            "output_columns": 13,
            "scores_finite": True,
            "scores_bounded": True,
            "full_study_fallback_numerator": 0,
            "full_study_fallback_denominator": 1,
            "full_study_fallback_rate": 0.0,
        },
        "submission_validation": {
            "row_count": 1,
            "column_count": 13,
            "exact_header": True,
            "exact_uid_order": True,
            "scores_finite": True,
            "scores_bounded": True,
        },
        "recovery": {
            "full_study_fallback": {"numerator": 0, "denominator": 1, "rate": 0.0},
            "fallback_investigation": {
                "complete": True,
                "investigated_cases": 0,
                "unavoidable_cases": 0,
            },
        },
        "candidate_execution": {
            "whole_pipeline_seconds": 5.0,
            "runtime_network_download": False,
            "reports_required": False,
        },
        "runtime_benchmark": {
            "whole_pipeline_seconds_observed": 5.0,
            "projected_seconds_at_target": 6500.0,
            "engineering_target_seconds": 25920,
            "hard_limit_seconds": 32400,
            "within_engineering_target": True,
            "within_hard_limit_projection": True,
            "hidden_execution_unknown": True,
            "full_scale_compliance_claim": False,
        },
    }
    evidence_path = tmp_path / "local-evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    submission_path = tmp_path / "local-submission.csv"
    submission_path.write_text(
        "StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,"
        "Effusion,Synovitis,Baker's,Contusion,Fracture\n"
        "study-a," + ",".join(["0.5"] * 12) + "\n",
        encoding="utf-8",
    )
    return bundle.root, evidence, evidence_path, submission_path


def _approval(path: Path, request_checksum: str, *, engineering_margin_approved: bool = False) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "rsna-knee-kaggle-owner-approval-v1",
                "request_checksum": request_checksum,
                "approval_scope": "ticket-12-exact-bundle-upload-and-execution",
                "approved": True,
                "rules_rechecked": True,
                "rules_accepted_by_owner": True,
                "account_eligible": True,
                "upload_allowed": True,
                "competition_submission_allowed": True,
                "engineering_margin_approved": engineering_margin_approved,
                "rules_evidence": {
                    "checked_at_utc": "2026-09-11T17:00:00+00:00",
                    "sources": ["ticket-01 official competition-rule record"],
                    "material_changes_reconciled": True,
                },
                "approver": "owner-test",
                "approved_at_utc": "2026-09-11T17:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return path


def _execution_payload(request: object, *, fallback_status: str = "measured") -> dict[str, object]:
    payload = request.payload  # type: ignore[attr-defined]
    candidate = payload["candidate"]
    fallback: dict[str, object]
    if fallback_status == "unknown":
        fallback = {
            "status": "unknown",
            "numerator": None,
            "denominator": 1,
            "rate": None,
            "unknown_hidden_failure_rate": True,
        }
    else:
        fallback = {
            "status": "measured",
            "numerator": 0,
            "denominator": 1,
            "rate": 0.0,
            "unknown_hidden_failure_rate": False,
            "investigation": {"complete": True, "investigated_cases": 0, "unavoidable_cases": 0},
        }
    return {
        "schema_version": "rsna-knee-kaggle-execution-v1",
        "status": "accepted_by_kaggle",
        "bundle": {
            "request_checksum": payload["request_checksum"],
            "candidate_manifest_checksum": candidate["manifest_checksum"],
            "notebook_sha256": payload["delivery_notebook"]["sha256"],
            "adapter_sha256": payload["delivery_adapter"]["sha256"],
            "model_checkpoints": candidate["model_checkpoints"],
            "model_size_bytes": candidate["model_size_bytes"],
            "candidate_bundle_size_bytes": candidate["candidate_bundle_size_bytes"],
            "environment": candidate["environment"],
        },
        "environment": {
            "runtime": "kaggle",
            "internet_enabled": False,
            "runtime_network_download": False,
            "reports_required": False,
            "python_version": "3.11.16",
            "torch_version": "2.7.1+cpu",
            "accelerator": "CPU",
            "installed_runtime_pins": candidate["environment"]["runtime_pins"],
        },
        "resources": {
            "model_size_bytes": candidate["model_size_bytes"],
            "candidate_bundle_size_bytes": candidate["candidate_bundle_size_bytes"],
            "peak_vram_bytes": 0,
            "ensemble_model_count": len(candidate["model_checkpoints"]),
        },
        "workload": {
            "study_count": 1,
            "series_count": 1,
            "slice_count": 3,
            "actual_workload_recorded": True,
            "hidden_test": True,
            "example_only": False,
        },
        "timing": {
            "whole_notebook_seconds": 100.0,
            "stages": {
                "startup_setup": 1.0,
                "imports": 1.0,
                "model_metadata_loading": 2.0,
                "dicom_read_decode": 50.0,
                "preprocessing": 20.0,
                "inference_ensembling": 20.0,
                "csv_creation": 6.0,
            },
            "stage_timing_source": "notebook_wall_clock",
            "projection": {
                "target_studies": 1300,
                "projected_seconds_at_target": 1300.0,
                "method": "linear_study_count_scaling",
            },
        },
        "output": {
            "filename": "submission.csv",
            "accepted_by_kaggle": True,
            "row_count": 1,
            "column_count": 13,
            "exact_header": True,
            "exact_uid_order": True,
            "scores_finite": True,
            "scores_bounded": True,
        },
        "fallback": fallback,
        "recovery": {
            "save_reload_verified": True,
            "interruption_recovery_verified": True,
            "rollback_seconds": 900,
        },
    }


def _write_private_execution(path: Path, request: object) -> Path:
    payload = request.payload  # type: ignore[attr-defined]
    path.write_text(
        json.dumps(
            {
                "schema_version": "rsna-knee-kaggle-execution-v1",
                "candidate_manifest_checksum": payload["candidate"]["manifest_checksum"],
                "studies": [
                    {
                        "StudyInstanceUID": "private-study-a",
                        "prediction": {
                            "used_full_study_fallback": False,
                            "valid_series_count": 1,
                            "rejected_series": [],
                            "warnings": [],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_kaggle_input(root: Path) -> tuple[str, ...]:
    study_uid = "1.2.826.0.1.3680043.8.498.3001"
    series_uid = "1.2.826.0.1.3680043.8.498.3002"
    study_dir = root / "test_series" / study_uid / series_uid
    study_dir.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        path = study_dir / f"slice-{index}.dcm"
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = UID(
            f"1.2.826.0.1.3680043.8.498.{3000 + index}"
        )
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = MRImageStorage
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.StudyInstanceUID = UID(study_uid)
        dataset.SeriesInstanceUID = UID(series_uid)
        dataset.Modality = "MR"
        dataset.ImagePositionPatient = [0.0, 0.0, float(index)]
        dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        dataset.Rows = 2
        dataset.Columns = 3
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.PixelData = np.arange(6, dtype=np.uint16).tobytes()
        dataset.save_as(path, enforce_file_format=True)
    (root / "test.csv").write_text(
        "StudyInstanceUID\n" + study_uid + "\n",
        encoding="utf-8",
    )
    (root / "test_series.csv").write_text(
        "StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,Anatomical_Plane\n"
        f"{study_uid},{series_uid},1,1,Sagittal\n",
        encoding="utf-8",
    )
    return (study_uid,)


def test_discover_kaggle_inputs_resolves_namespaced_mounts_and_mirrors_candidate_code(
    tmp_path: Path,
) -> None:
    candidate, _evidence, evidence_path, _submission_path = _make_candidate(tmp_path)
    notebook = tmp_path / "delivery.ipynb"
    _write_notebook(notebook)
    request = prepare_kaggle_delivery(
        candidate,
        evidence_path,
        tmp_path / "request",
        delivery_notebook_path=notebook,
        request_id="ticket-12-namespaced-inputs",
    )

    input_root = tmp_path / "kaggle-input"
    delivery_root = input_root / "datasets" / "owner" / "delivery"
    delivery_root.mkdir(parents=True)
    shutil.copytree(candidate, delivery_root / "candidate")
    shutil.copytree(candidate / "code", delivery_root / "code")
    shutil.copy2(request.root / "request.json", delivery_root / "request.json")
    shutil.copy2(notebook, delivery_root / notebook.name)

    approval_path = _approval(tmp_path / "approval.json", request.request_checksum)
    approval_root = input_root / "datasets" / "owner" / "approval"
    approval_root.mkdir(parents=True)
    shutil.copy2(approval_path, approval_root / "owner-approval.json")
    competition_root = input_root / "competitions" / "rsna-knee-abnormality-detection"
    _write_kaggle_input(competition_root)

    layout = discover_kaggle_inputs(
        input_root,
        expected_request_checksum=request.request_checksum,
        expected_candidate_manifest_checksum=request.payload["candidate"]["manifest_checksum"],
    )

    assert layout.delivery_root == delivery_root
    assert layout.candidate_root == delivery_root / "candidate"
    assert layout.code_root == delivery_root / "code"
    assert layout.request_path == delivery_root / "request.json"
    assert layout.approval_path == approval_root / "owner-approval.json"
    assert layout.competition_root == competition_root
    assert layout.request_checksum == request.request_checksum
    assert layout.validation is not None

    mirrored_module = delivery_root / "code" / "rsna_knee" / "model.py"
    mirrored_module.chmod(0o644)
    mirrored_module.write_text(mirrored_module.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(KaggleDeliveryError, match="immutable candidate code"):
        discover_kaggle_inputs(
            input_root,
            expected_request_checksum=request.request_checksum,
            expected_candidate_manifest_checksum=request.payload["candidate"]["manifest_checksum"],
        )


def test_prepare_kaggle_delivery_records_an_exact_offline_runtime_wheel(tmp_path: Path) -> None:
    candidate, _evidence, evidence_path, _submission_path = _make_candidate(tmp_path)
    notebook = tmp_path / "delivery.ipynb"
    _write_notebook(notebook)
    wheel = tmp_path / "pydicom-3.0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "pydicom-3.0.1.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: pydicom\nVersion: 3.0.1\n",
        )
        archive.writestr(
            "pydicom-3.0.1.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )

    request = prepare_kaggle_delivery(
        candidate,
        evidence_path,
        tmp_path / "request",
        delivery_notebook_path=notebook,
        delivery_runtime_wheel_path=wheel,
        request_id="ticket-12-runtime-wheel",
    )

    runtime = request.payload["delivery_runtime"]["offline_wheel"]
    assert runtime["filename"] == wheel.name
    assert runtime["package"] == "pydicom"
    assert runtime["version"] == "3.0.1"
    assert len(runtime["sha256"]) == 64


def test_prepare_kaggle_delivery_records_all_offline_runtime_wheels(tmp_path: Path) -> None:
    candidate, _evidence, evidence_path, _submission_path = _make_candidate(tmp_path)
    notebook = tmp_path / "delivery.ipynb"
    _write_notebook(notebook)
    wheels: list[Path] = []
    for package, version in (("numpy", "2.2.6"), ("pydicom", "3.0.1"), ("torch", "2.7.1+cpu")):
        wheel = tmp_path / f"{package}-{version}-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(
                f"{package}-{version}.dist-info/METADATA",
                f"Metadata-Version: 2.1\nName: {package}\nVersion: {version}\n",
            )
            archive.writestr(
                f"{package}-{version}.dist-info/WHEEL",
                "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            )
        wheels.append(wheel)

    request = prepare_kaggle_delivery(
        candidate,
        evidence_path,
        tmp_path / "request",
        delivery_notebook_path=notebook,
        delivery_runtime_wheel_paths=wheels,
        request_id="ticket-12-runtime-wheelhouse",
    )

    runtime = request.payload["delivery_runtime"]
    assert [item["package"] for item in runtime["offline_wheels"]] == ["numpy", "pydicom", "torch"]
    assert [item["version"] for item in runtime["offline_wheels"]] == ["2.2.6", "3.0.1", "2.7.1+cpu"]
    loaded = load_kaggle_delivery(request.root, delivery_runtime_wheel_paths=wheels)
    assert loaded.request_checksum == request.request_checksum


def test_prepare_kaggle_delivery_binds_execution_input_to_request_checksum(tmp_path: Path) -> None:
    candidate, _evidence, evidence_path, _submission_path = _make_candidate(tmp_path)
    notebook = tmp_path / "delivery.ipynb"
    _write_notebook(notebook)

    request = prepare_kaggle_delivery(
        candidate,
        evidence_path,
        tmp_path / "request",
        delivery_notebook_path=notebook,
        execution_input={
            "mount_slug": "representative-input-v12",
            "workload_kind": "representative",
            "manifest_filename": "input-manifest.json",
            "manifest_sha256": "a" * 64,
            "test_csv_sha256": "b" * 64,
            "test_series_csv_sha256": "c" * 64,
        },
        request_id="ticket-12-execution-input-binding",
    )

    unsigned = dict(request.payload)
    unsigned.pop("request_checksum")
    assert request.request_checksum == kaggle_module._json_sha256(unsigned)
    assert request.payload["execution_input"]["workload_kind"] == "representative"


def test_fixed_plus_scaling_projection_keeps_setup_fixed() -> None:
    projection = kaggle_module._fixed_plus_scaling_projection(
        whole_seconds=100.0,
        study_count=100,
        target_studies=1300,
        stages={
            "startup_setup": 10.0,
            "imports": 5.0,
            "input_discovery": 15.0,
            "adapter_preflight": 5.0,
            "model_metadata_loading": 5.0,
            "dicom_read_decode": 30.0,
            "preprocessing": 10.0,
            "inference_ensembling": 15.0,
            "csv_creation": 5.0,
        },
    )

    assert projection["method"] == "fixed_plus_measured_scaling"
    assert projection["fixed_seconds_observed"] == 40.0
    assert projection["scaling_seconds_observed"] == 60.0
    assert projection["projected_seconds_at_target"] == 820.0


def test_kaggle_runtime_pin_gate_checks_metadata_and_imported_versions() -> None:
    with pytest.raises(KaggleDeliveryError, match="pydicom"):
        validate_kaggle_runtime_pins(
            {
                "numpy": "2.2.6",
                "pydicom": "not-installed-version",
                "torch": "2.7.1+cpu",
            }
        )


def test_kaggle_runtime_wheel_audit_rejects_a_different_package(tmp_path: Path) -> None:
    wheel = tmp_path / "other-3.0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "other-3.0.1.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: other\nVersion: 3.0.1\n",
        )
        archive.writestr(
            "other-3.0.1.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )

    with pytest.raises(KaggleDeliveryError, match="package"):
        audit_kaggle_runtime_wheel(wheel)


def test_kaggle_runtime_wheel_audit_reads_the_wheel_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel = tmp_path / "pydicom-3.0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "pydicom-3.0.1.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: pydicom\nVersion: 3.0.1\n",
        )
        archive.writestr(
            "pydicom-3.0.1.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )

    original_read_bytes = Path.read_bytes
    read_paths: list[Path] = []

    def tracked_read_bytes(path: Path) -> bytes:
        if path == wheel:
            read_paths.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)

    audit = audit_kaggle_runtime_wheel(wheel)

    assert audit["package"] == "pydicom"
    assert audit["version"] == "3.0.1"
    assert read_paths == [wheel]


def test_ticket12_notebook_runs_frozen_adapter_in_a_fresh_runtime_process() -> None:
    notebook = Path("notebooks/ticket12_kaggle_baseline.ipynb")
    source = "".join(
        "".join(cell["source"])
        for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    )
    assert "runtime-site" in source
    assert "run-ticket12-inference.py" in source
    assert "subprocess.run([sys.executable, str(inference_script)]" in source
    assert "'code_root': str(bootstrap_delivery_root / 'code')" in source
    assert "sys.path.insert(0, config['code_root'])" in source
    assert "offline_wheel_install_seconds" in source
    assert "offline_runtime_probe_seconds" in source
    assert "input_discovery_seconds" in source
    assert "notebook_timing_details" in source
    assert "validated_layout=layout" in source


def test_kaggle_baseline_adapter_writes_exact_local_smoke_output_and_private_diagnostics(tmp_path: Path) -> None:
    candidate, _evidence, evidence_path, _submission_path = _make_candidate(tmp_path)
    input_root = tmp_path / "kaggle-input"
    expected_uids = _write_kaggle_input(input_root)

    result = run_kaggle_baseline(
        candidate,
        input_root,
        tmp_path / "run" / "submission.csv",
        evidence_path=tmp_path / "run" / "execution.json",
        private_evidence_path=tmp_path / "run" / "private.json",
        allow_local_smoke=True,
    )

    assert result.evidence["schema_version"] == "rsna-knee-kaggle-execution-v1"
    assert result.evidence["status"] == "completed_kaggle_notebook"
    assert result.evidence["workload"]["study_count"] == len(expected_uids)
    assert result.evidence["workload"]["example_only"] is True
    assert result.evidence["output"]["accepted_by_local_contract"] is True
    assert result.evidence["output"]["accepted_by_kaggle"] is False
    assert result.evidence["fallback"]["denominator"] == len(expected_uids)
    detailed = result.evidence["timing"]["detailed"]
    assert detailed["total_runtime_seconds"] == result.evidence["timing"]["whole_notebook_seconds"]
    assert detailed["input_discovery_seconds"] == 0.0
    assert detailed["offline_wheel_install_seconds"] == 0.0
    assert detailed["dicom_archive_index_discovery_seconds"] >= 0.0
    assert detailed["pixel_decode_seconds"] >= 0.0
    assert detailed["dicom_file_open_count"] == 3
    assert detailed["dicom_file_reopen_count"] == 0
    private = json.loads((tmp_path / "run" / "private.json").read_text(encoding="utf-8"))
    assert private["studies"][0]["StudyInstanceUID"] == expected_uids[0]
    assert "prediction" in private["studies"][0]
    assert private["studies"][0]["timing"]["series"][0]["file_open_count"] == 3


def test_actual_kaggle_execution_requires_request_and_owner_approval(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")

    with pytest.raises(KaggleDeliveryError, match="actual Kaggle execution requires"):
        run_kaggle_baseline(
            tmp_path / "candidate",
            tmp_path / "input",
            tmp_path / "run" / "submission.csv",
        )


def test_actual_kaggle_execution_rechecks_approval_and_delivery_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    candidate, _evidence, evidence_path, _submission_path = _make_candidate(tmp_path)
    notebook = tmp_path / "delivery.ipynb"
    _write_notebook(notebook)
    request = prepare_kaggle_delivery(
        candidate,
        evidence_path,
        tmp_path / "request",
        delivery_notebook_path=notebook,
        request_id="ticket-12-actual-gate",
    )
    approval = _approval(tmp_path / "approval.json", request.request_checksum)
    input_root = tmp_path / "kaggle-input"
    _write_kaggle_input(input_root)

    with pytest.raises(KaggleDeliveryError, match="adapter checksum"):
        run_kaggle_baseline(
            candidate,
            input_root,
            tmp_path / "bad" / "submission.csv",
            delivery_request_checksum=request.request_checksum,
            delivery_adapter_sha256="wrong-adapter-hash",
            delivery_request_root=request.root,
            owner_approval_path=approval,
            delivery_notebook_path=notebook,
            delivery_adapter_path=Path("src/rsna_knee/kaggle.py"),
            notebook_stage_timings={"startup_setup": 0.5, "imports": 1.0},
            notebook_started_at=time.perf_counter(),
        )

    result = run_kaggle_baseline(
        candidate,
        input_root,
        tmp_path / "actual" / "submission.csv",
        evidence_path=tmp_path / "actual" / "execution.json",
        delivery_request_checksum=request.request_checksum,
        delivery_request_root=request.root,
        owner_approval_path=approval,
        delivery_notebook_path=notebook,
        delivery_adapter_path=Path("src/rsna_knee/kaggle.py"),
        notebook_stage_timings={"startup_setup": 0.5, "imports": 1.0},
        notebook_started_at=time.perf_counter(),
    )

    assert result.evidence["workload"]["hidden_test"] is True
    assert result.evidence["timing"]["stage_timing_source"] == "notebook_wall_clock"
    assert result.evidence["approval"]["sha256"]


def test_actual_kaggle_execution_reuses_discovered_integrity_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The discovered proof is reused without weakening the actual-run gates."""

    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    candidate, _evidence, evidence_path, _submission_path = _make_candidate(tmp_path)
    notebook = tmp_path / "delivery.ipynb"
    _write_notebook(notebook)
    request = prepare_kaggle_delivery(
        candidate,
        evidence_path,
        tmp_path / "request",
        delivery_notebook_path=notebook,
        request_id="ticket-12-reuse-discovered-validation",
    )

    input_root = tmp_path / "kaggle-input"
    delivery_root = input_root / "datasets" / "owner" / "delivery"
    delivery_root.mkdir(parents=True)
    shutil.copytree(candidate, delivery_root / "candidate")
    shutil.copytree(candidate / "code", delivery_root / "code")
    shutil.copy2(request.root / "request.json", delivery_root / "request.json")
    shutil.copy2(notebook, delivery_root / notebook.name)
    approval_path = _approval(tmp_path / "approval.json", request.request_checksum)
    approval_root = input_root / "datasets" / "owner" / "approval"
    approval_root.mkdir(parents=True)
    shutil.copy2(approval_path, approval_root / "owner-approval.json")
    competition_root = input_root / "competitions" / "rsna-knee-abnormality-detection"
    _write_kaggle_input(competition_root)

    layout = discover_kaggle_inputs(
        input_root,
        expected_request_checksum=request.request_checksum,
        expected_candidate_manifest_checksum=request.payload["candidate"]["manifest_checksum"],
    )

    with pytest.raises(KaggleDeliveryError, match="competition input root"):
        kaggle_module._reuse_discovered_kaggle_inputs(
            layout,
            candidate_root=layout.candidate_root,
            input_root=input_root,
            delivery_request_root=layout.delivery_root,
            owner_approval_path=layout.approval_path,
            delivery_notebook_path=layout.delivery_notebook_path,
            delivery_adapter_path=layout.delivery_adapter_path,
            delivery_request_checksum=layout.request_checksum,
            delivery_notebook_sha256=request.payload["delivery_notebook"]["sha256"],
            delivery_adapter_sha256=request.payload["delivery_adapter"]["sha256"],
            delivery_runtime_wheel_path=None,
            delivery_runtime_wheel_paths=layout.runtime_wheel_paths,
        )

    def unexpected_revalidation(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("discovered delivery validation was redundantly repeated")

    monkeypatch.setattr(kaggle_module, "load_kaggle_delivery", unexpected_revalidation)
    monkeypatch.setattr(kaggle_module, "load_candidate", unexpected_revalidation)
    monkeypatch.setattr(kaggle_module, "_approval_payload", unexpected_revalidation)

    result = run_kaggle_baseline(
        layout.candidate_root,
        layout.competition_root,
        tmp_path / "run" / "submission.csv",
        evidence_path=tmp_path / "run" / "execution.json",
        delivery_request_checksum=layout.request_checksum,
        delivery_notebook_sha256=request.payload["delivery_notebook"]["sha256"],
        delivery_adapter_sha256=request.payload["delivery_adapter"]["sha256"],
        delivery_request_root=layout.delivery_root,
        owner_approval_path=layout.approval_path,
        delivery_notebook_path=layout.delivery_notebook_path,
        delivery_adapter_path=layout.delivery_adapter_path,
        delivery_runtime_wheel_paths=layout.runtime_wheel_paths,
        notebook_stage_timings={"startup_setup": 0.5, "imports": 1.0},
        notebook_started_at=time.perf_counter(),
        validated_layout=layout,
    )

    detailed = result.evidence["timing"]["detailed"]
    assert detailed["delivery_validation_mode"] == "discovered_layout_reused"
    assert detailed["delivery_validation_reused"] is True
    assert detailed["delivery_request_validation_seconds"] == 0.0
    assert detailed["candidate_hashes_reused"] is True
    assert result.evidence["output"]["accepted_by_local_contract"] is True


def test_prepare_kaggle_delivery_references_exact_candidate_without_mutating_it(tmp_path: Path) -> None:
    candidate, _evidence, evidence_path, submission_path = _make_candidate(tmp_path)
    before = (candidate / "manifest.json").read_bytes()
    notebook = tmp_path / "delivery.ipynb"
    _write_notebook(notebook)

    request = prepare_kaggle_delivery(
        candidate,
        evidence_path,
        tmp_path / "request",
        delivery_notebook_path=notebook,
        local_submission_path=submission_path,
        request_id="ticket-12-request-001",
    )

    assert request.payload["status"] == "owner_approval_required"
    assert request.payload["approval"]["approved"] is False
    assert request.payload["upload"]["allowed"] is False
    assert request.payload["candidate"]["manifest_checksum"]
    assert request.payload["candidate"]["model_checkpoints"]
    assert (candidate / "manifest.json").read_bytes() == before
    assert request.root != candidate

    with pytest.raises(FileExistsError):
        prepare_kaggle_delivery(
            candidate,
            evidence_path,
            request.root,
            delivery_notebook_path=notebook,
            request_id="ticket-12-request-001-retry",
        )


def test_prepare_kaggle_delivery_rejects_local_evidence_for_a_different_candidate(tmp_path: Path) -> None:
    candidate, evidence, evidence_path, _submission_path = _make_candidate(tmp_path)
    evidence["candidate"]["candidate_manifest_checksum"] = "different"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    notebook = tmp_path / "delivery.ipynb"
    _write_notebook(notebook)

    with pytest.raises(KaggleDeliveryError, match="candidate manifest checksum"):
        prepare_kaggle_delivery(
            candidate,
            evidence_path,
            tmp_path / "request",
            delivery_notebook_path=notebook,
            request_id="ticket-12-request-002",
        )


def test_kaggle_verification_requires_owner_approval_for_the_exact_request(tmp_path: Path) -> None:
    candidate, _evidence, evidence_path, submission_path = _make_candidate(tmp_path)
    notebook = tmp_path / "delivery.ipynb"
    _write_notebook(notebook)
    request = prepare_kaggle_delivery(
        candidate,
        evidence_path,
        tmp_path / "request",
        delivery_notebook_path=notebook,
        request_id="ticket-12-request-003",
    )
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(json.dumps(_execution_payload(request)), encoding="utf-8")
    approval_path = tmp_path / "approval.json"
    _approval(approval_path, "not-the-request")

    with pytest.raises(KaggleDeliveryError, match="approval.*request"):
        record_kaggle_verification(
            request.root,
            candidate,
            approval_path,
            execution_path,
            tmp_path / "verification",
            submission_path=submission_path,
        )


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (("workload", "hidden_test"), False, "example-only"),
        (("workload", "example_only"), True, "example-only"),
        (("timing", "whole_notebook_seconds"), 32_401.0, "hard limit"),
    ],
)
def test_kaggle_verification_rejects_example_runs_and_hard_limit(
    tmp_path: Path,
    field_path: tuple[str, str],
    value: object,
    message: str,
) -> None:
    candidate, _evidence, evidence_path, submission_path = _make_candidate(tmp_path)
    notebook = tmp_path / "delivery.ipynb"
    _write_notebook(notebook)
    request = prepare_kaggle_delivery(
        candidate,
        evidence_path,
        tmp_path / "request",
        delivery_notebook_path=notebook,
        request_id="ticket-12-rejection",
    )
    execution = _execution_payload(request)
    execution[field_path[0]][field_path[1]] = value  # type: ignore[index]
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    approval_path = _approval(tmp_path / "approval.json", request.request_checksum)

    with pytest.raises(KaggleDeliveryError, match=message):
        record_kaggle_verification(
            request.root,
            candidate,
            approval_path,
            execution_path,
            tmp_path / "verification",
            submission_path=submission_path,
        )


def test_unknown_hidden_fallback_rate_is_preserved_and_blocks_promotion(tmp_path: Path) -> None:
    candidate, _evidence, evidence_path, submission_path = _make_candidate(tmp_path)
    notebook = tmp_path / "delivery.ipynb"
    _write_notebook(notebook)
    request = prepare_kaggle_delivery(
        candidate,
        evidence_path,
        tmp_path / "request",
        delivery_notebook_path=notebook,
        request_id="ticket-12-request-004",
    )
    execution_path = tmp_path / "execution-unknown.json"
    execution_path.write_text(
        json.dumps(_execution_payload(request, fallback_status="unknown")),
        encoding="utf-8",
    )
    approval_path = _approval(tmp_path / "approval.json", request.request_checksum)
    verification = record_kaggle_verification(
        request.root,
        candidate,
        approval_path,
        execution_path,
        tmp_path / "verification-unknown",
        submission_path=submission_path,
    )

    assert verification.payload["fallback"]["status"] == "unknown"
    assert verification.payload["fallback"]["rate"] is None
    assert verification.payload["promotion"]["eligible"] is False
    with pytest.raises(KaggleDeliveryError, match="promotion"):
        promote_kaggle_candidate(
            request.root,
            candidate,
            verification.root,
            tmp_path / "promotion-unknown",
        )


def test_measured_zero_fallback_can_promote_without_mutating_the_local_candidate(tmp_path: Path) -> None:
    candidate, _evidence, evidence_path, submission_path = _make_candidate(tmp_path)
    notebook = tmp_path / "delivery.ipynb"
    _write_notebook(notebook)
    request = prepare_kaggle_delivery(
        candidate,
        evidence_path,
        tmp_path / "request",
        delivery_notebook_path=notebook,
        request_id="ticket-12-request-005",
    )
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(json.dumps(_execution_payload(request)), encoding="utf-8")
    approval_path = _approval(tmp_path / "approval.json", request.request_checksum)
    private_path = _write_private_execution(tmp_path / "private-execution.json", request)
    verification = record_kaggle_verification(
        request.root,
        candidate,
        approval_path,
        execution_path,
        tmp_path / "verification",
        submission_path=submission_path,
        private_evidence_path=private_path,
    )
    before = (candidate / "manifest.json").read_bytes()

    promotion = promote_kaggle_candidate(
        request.root,
        candidate,
        verification.root,
        tmp_path / "promotion",
    )
    loaded = load_kaggle_promotion(promotion.root)

    assert loaded.payload["promotion"]["level"] == "kaggle_verified"
    assert loaded.payload["promotion"]["final_candidate"] is False
    assert loaded.payload["promotion"]["owner_selection_required"] is True
    assert loaded.payload["candidate"]["manifest_checksum"] == request.payload["candidate"]["manifest_checksum"]
    assert (candidate / "manifest.json").read_bytes() == before
