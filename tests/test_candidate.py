from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, UID

from rsna_knee.candidate import (
    CandidateBundle,
    CandidateIntegrityError,
    benchmark_candidate_execution,
    freeze_candidate,
    load_candidate,
    run_candidate_inference,
)
from rsna_knee.dicom import DecoderPolicy
from rsna_knee.geometry import GeometryPolicy
from rsna_knee.model import HierarchicalStudyModel, ModelConfig
from rsna_knee.preprocessing import PreprocessConfig
from rsna_knee.real_inference import IntegrationManifest, IntegrationSeries, IntegrationStudy, write_integration_manifest
from rsna_knee.submission import SubmissionError, validate_submission_csv


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


def _write_fixture_slice(root: Path, study_uid: str, series_uid: str, index: int) -> str:
    relative = Path("train_series") / study_uid / series_uid / f"slice-{index}.dcm"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = UID(f"1.2.826.0.1.3680043.8.498.{index + 1000}")
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
    return relative.as_posix()


def _write_local_integration_manifest(root: Path) -> Path:
    study_uid = "1.2.826.0.1.3680043.8.498.2001"
    series_uid = "1.2.826.0.1.3680043.8.498.2002"
    entries = tuple(_write_fixture_slice(root, study_uid, series_uid, index) for index in range(3))
    geometry = GeometryPolicy(minimum_usable_slices=3)
    manifest = IntegrationManifest(
        source_version="fixture-candidate-v1",
        archive_fingerprint="fixture-candidate-fingerprint",
        geometry_policy=geometry,
        decoder_policy=DecoderPolicy(require_modality=True),
        preprocess_config=PreprocessConfig(
            image_size=(16, 16),
            max_slices=8,
            max_series=16,
            geometry_policy=geometry,
        ),
        studies=(
            IntegrationStudy(
                study_uid,
                (IntegrationSeries(study_uid, series_uid, entries=entries),),
            ),
        ),
    )
    path = root / "integration-manifest.json"
    write_integration_manifest(manifest, path)
    return path


def test_freeze_candidate_publishes_a_checksum_verified_immutable_bundle(tmp_path: Path) -> None:
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
    cluster_requirements = tmp_path / "cluster.lock"
    cluster_requirements.write_text(
        "numpy==2.2.6\npydicom==3.0.1\ntorch==2.7.1+cpu\n",
        encoding="utf-8",
    )
    offline_requirements = tmp_path / "offline.lock"
    offline_requirements.write_text(cluster_requirements.read_text(encoding="utf-8"), encoding="utf-8")
    notebook = tmp_path / "candidate.ipynb"
    notebook.write_text('{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}\n', encoding="utf-8")

    bundle = freeze_candidate(
        tmp_path / "candidate",
        model_checkpoints=(checkpoint,),
        oof_evidence_path=oof,
        code_root=Path("src/rsna_knee"),
        cluster_requirements_path=cluster_requirements,
        offline_requirements_path=offline_requirements,
        notebook_path=notebook,
        candidate_id="candidate-001",
        code_revision="test-revision",
    )

    assert isinstance(bundle, CandidateBundle)
    assert bundle.manifest_path.exists()
    loaded = load_candidate(bundle.root)
    assert loaded.manifest_checksum == bundle.manifest_checksum
    assert (bundle.root / "models" / "fold-0.pt").exists()
    assert not (bundle.root / "code" / "rsna_knee" / "training.py").exists()

    with pytest.raises(FileExistsError):
        freeze_candidate(
            bundle.root,
            model_checkpoints=(checkpoint,),
            oof_evidence_path=oof,
            code_root=Path("src/rsna_knee"),
            cluster_requirements_path=cluster_requirements,
            offline_requirements_path=offline_requirements,
            notebook_path=notebook,
            candidate_id="candidate-001",
            code_revision="test-revision",
        )


def test_validate_submission_csv_accepts_exact_contract_and_order(tmp_path: Path) -> None:
    output = tmp_path / "submission.csv"
    header = (
        "StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,"
        "Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture\n"
    )
    row = "study-a," + ",".join(["0.5"] * 12) + "\n"
    output.write_text(header + row, encoding="utf-8")

    result = validate_submission_csv(output, ("study-a",))

    assert result == {
        "row_count": 1,
        "column_count": 13,
        "exact_header": True,
        "exact_uid_order": True,
        "scores_finite": True,
        "scores_bounded": True,
    }

    output.write_text(header + "study-a," + ",".join(["nan"] * 12) + "\n", encoding="utf-8")
    with pytest.raises(SubmissionError, match="not finite"):
        validate_submission_csv(output, ("study-a",))


def test_load_candidate_rejects_a_changed_frozen_artifact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "fold-0.pt"
    _write_checkpoint(checkpoint)
    oof = tmp_path / "oof.json"
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
    requirements.write_text("numpy==2.2.6\npydicom==3.0.1\ntorch==2.7.1+cpu\n", encoding="utf-8")
    notebook = tmp_path / "candidate.ipynb"
    notebook.write_text('{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}\n', encoding="utf-8")
    bundle = freeze_candidate(
        tmp_path / "candidate",
        model_checkpoints=(checkpoint,),
        oof_evidence_path=oof,
        code_root=Path("src/rsna_knee"),
        cluster_requirements_path=requirements,
        offline_requirements_path=requirements,
        notebook_path=notebook,
        candidate_id="candidate-002",
        code_revision="test-revision",
    )

    changed = bundle.root / "models" / "fold-0.pt"
    changed.chmod(0o644)
    with changed.open("ab") as handle:
        handle.write(b"changed")
    changed.chmod(0o444)

    with pytest.raises(CandidateIntegrityError, match="changed"):
        load_candidate(bundle.root)


def test_run_candidate_inference_uses_frozen_model_and_validates_real_csv(tmp_path: Path) -> None:
    checkpoint = tmp_path / "fold-0.pt"
    _write_checkpoint(checkpoint)
    oof = tmp_path / "oof.json"
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
    requirements.write_text("numpy==2.2.6\npydicom==3.0.1\ntorch==2.7.1+cpu\n", encoding="utf-8")
    notebook = tmp_path / "candidate.ipynb"
    notebook.write_text('{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}\n', encoding="utf-8")
    bundle = freeze_candidate(
        tmp_path / "candidate",
        model_checkpoints=(checkpoint,),
        oof_evidence_path=oof,
        code_root=Path("src/rsna_knee"),
        cluster_requirements_path=requirements,
        offline_requirements_path=requirements,
        notebook_path=notebook,
        candidate_id="candidate-003",
        code_revision="test-revision",
    )
    integration_manifest = _write_local_integration_manifest(tmp_path)

    execution = run_candidate_inference(
        bundle.root,
        integration_manifest,
        tmp_path,
        tmp_path / "execution" / "submission.csv",
        evidence_path=tmp_path / "execution" / "evidence.json",
        allow_local_smoke=True,
    )

    assert execution.evidence["candidate"]["candidate_id"] == "candidate-003"
    assert execution.submission_validation["exact_uid_order"] is True
    assert execution.output_path.exists()
    assert execution.output_path.parent != bundle.root


def test_benchmark_candidate_records_measured_scope_and_target_projection(tmp_path: Path) -> None:
    checkpoint = tmp_path / "fold-0.pt"
    _write_checkpoint(checkpoint)
    oof = tmp_path / "oof.json"
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
    requirements.write_text("numpy==2.2.6\npydicom==3.0.1\ntorch==2.7.1+cpu\n", encoding="utf-8")
    notebook = tmp_path / "candidate.ipynb"
    notebook.write_text('{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}\n', encoding="utf-8")
    bundle = freeze_candidate(
        tmp_path / "candidate",
        model_checkpoints=(checkpoint,),
        oof_evidence_path=oof,
        code_root=Path("src/rsna_knee"),
        cluster_requirements_path=requirements,
        offline_requirements_path=requirements,
        notebook_path=notebook,
        candidate_id="candidate-004",
        code_revision="test-revision",
    )
    integration_manifest = _write_local_integration_manifest(tmp_path)

    execution = benchmark_candidate_execution(
        bundle.root,
        integration_manifest,
        tmp_path,
        tmp_path / "benchmark" / "submission.csv",
        evidence_path=tmp_path / "benchmark" / "evidence.json",
        expected_study_count=1300,
        long_tail_studies=1,
        allow_local_smoke=True,
    )

    benchmark = execution.evidence["runtime_benchmark"]
    assert benchmark["measured_scope"]["studies"] == 1
    assert benchmark["target_scope"]["studies"] == 1300
    assert benchmark["target_scope"]["long_tail_included"] is True
    assert benchmark["hidden_execution_unknown"] is True
    assert benchmark["full_scale_compliance_claim"] is False


def test_candidate_code_snapshot_imports_without_training_or_network_modules(tmp_path: Path) -> None:
    checkpoint = tmp_path / "fold-0.pt"
    _write_checkpoint(checkpoint)
    oof = tmp_path / "oof.json"
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
    requirements.write_text("numpy==2.2.6\npydicom==3.0.1\ntorch==2.7.1+cpu\n", encoding="utf-8")
    notebook = tmp_path / "candidate.ipynb"
    notebook.write_text('{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}\n', encoding="utf-8")
    bundle = freeze_candidate(
        tmp_path / "candidate",
        model_checkpoints=(checkpoint,),
        oof_evidence_path=oof,
        code_root=Path("src/rsna_knee"),
        cluster_requirements_path=requirements,
        offline_requirements_path=requirements,
        notebook_path=notebook,
        candidate_id="candidate-005",
        code_revision="test-revision",
    )

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(bundle.root / "code")
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, rsna_knee; print('rsna_knee.training' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert check.stdout.strip() == "False"
