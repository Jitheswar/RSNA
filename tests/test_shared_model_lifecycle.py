from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, UID

from rsna_knee.candidate import freeze_candidate, run_candidate_inference
from rsna_knee.folds import freeze_folds
from rsna_knee.model import HierarchicalStudyModel, ModelConfig
from rsna_knee.oof import aggregate_oof_run_records, write_oof_reports
from rsna_knee.preprocessing import pack_study, prepare_study
from rsna_knee.real_inference import (
    IntegrationManifest,
    IntegrationSeries,
    IntegrationStudy,
    DEFAULT_REAL_DECODER_POLICY,
    iter_real_studies,
    write_integration_manifest,
)
from rsna_knee.targets import TARGET_COLUMNS as MODEL_TARGET_COLUMNS
from rsna_knee.synthetic import make_synthetic_study
from rsna_knee.training import TrainingConfig, run_cpu_preflight, train_one_fold
from rsna_knee.inventory import SourceManifest, SourceStudy


def _write_fixture_slice(root: Path, study_uid: str, series_uid: str, index: int) -> str:
    sop_uid = f"1.2.826.0.1.3680043.8.498.{50000 + index}"
    relative = Path("train_series") / study_uid / series_uid / f"{sop_uid}.dcm"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = UID(sop_uid)
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
    pixels = (np.arange(6, dtype=np.uint16) + index * 6).astype(np.uint16)
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return relative.as_posix()


def test_compact_model_completes_training_oof_candidate_and_submission_lifecycle(
    tmp_path: Path,
) -> None:
    study_uids = tuple(
        f"1.2.826.0.1.3680043.8.498.{10000 + index}" for index in range(9)
    )
    synthetic_studies = tuple(
        make_synthetic_study(uid, series_count=1, slices_per_series=2, seed=100 + index)
        for index, uid in enumerate(study_uids)
    )
    source_manifest = SourceManifest(
        "train",
        tuple(
            SourceStudy(
                study.study_instance_uid,
                True,
                tuple(
                    (index + target_index) % 2
                    for target_index in range(len(MODEL_TARGET_COLUMNS))
                ),
            )
            for index, study in enumerate(synthetic_studies)
        ),
        (),
    )
    fold_manifest = freeze_folds(
        source_manifest,
        source_version="synthetic-ticket28-v1",
        seed=17,
    )
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        successful_optimizer_steps=2,
        checkpoint_interval_steps=1,
        seed=2801,
    )
    preflight = run_cpu_preflight(require_isolated_python=False)
    assert preflight.passed

    results = tuple(
        train_one_fold(
            {study.study_instance_uid: study for study in synthetic_studies},
            fold_manifest,
            fold,
            tmp_path / f"fold-{fold}",
            config=config,
            source_version="synthetic-ticket28-v1",
            source_fingerprint="synthetic-ticket28-source-fingerprint",
            integration_manifest_checksum="synthetic-ticket28-training-manifest-checksum",
            preflight=preflight,
            require_preflight=False,
            allow_local_smoke=True,
        )
        for fold in range(3)
    )
    assert all(result.final_checkpoint_path is not None for result in results)
    assert all(result.evaluation is not None for result in results)
    assert all(result.evaluation.targets[0].target == MODEL_TARGET_COLUMNS[0] for result in results)

    run_paths = tuple(tmp_path / f"fold-{fold}" for fold in range(3))
    oof = aggregate_oof_run_records(
        run_paths,
        fold_manifest,
        bootstrap_resamples=8,
        bootstrap_seed=28,
    )
    public_oof = tmp_path / "oof-evidence.json"
    write_oof_reports(oof, public_path=public_oof, source_run_paths=run_paths)
    assert oof.prediction_count == len(study_uids)
    assert oof.run_ids == tuple(result.run_id for result in results)
    assert tuple(target.target for target in oof.pooled_targets) == MODEL_TARGET_COLUMNS

    candidate = freeze_candidate(
        tmp_path / "candidate",
        model_checkpoints=tuple(result.final_checkpoint_path for result in results if result.final_checkpoint_path),
        oof_evidence_path=public_oof,
        candidate_id="ticket28-synthetic-lifecycle",
        code_revision="ticket28-test",
        source_version="synthetic-ticket28-v1",
        fold_manifest_checksum=fold_manifest.manifest_checksum,
    )
    assert candidate.model_config.to_dict()["schema_version"] == "rsna-knee-model-config-v1"

    source = tmp_path / "dicom-source"
    inference_uid = study_uids[0]
    series_uid = "1.2.826.0.1.3680043.8.498.20001"
    entries = tuple(_write_fixture_slice(source, inference_uid, series_uid, index) for index in range(3))
    preprocess = config.preprocess_config()
    packing = config.packing_config()
    geometry = preprocess.geometry_policy
    manifest = IntegrationManifest(
        source_version="synthetic-ticket28-input-v1",
        archive_fingerprint="synthetic-ticket28-input-fingerprint",
        geometry_policy=geometry,
        decoder_policy=DEFAULT_REAL_DECODER_POLICY,
        preprocess_config=preprocess,
        studies=(
            IntegrationStudy(
                inference_uid,
                (IntegrationSeries(inference_uid, series_uid, entries=entries),),
            ),
        ),
        packing_config=packing,
    )
    manifest_path = tmp_path / "integration-manifest.json"
    write_integration_manifest(manifest, manifest_path)

    # The direct legacy constructor is the before-prefactor compact model
    # implementation.  Its scores define the expected frozen-candidate row.
    loaded = next(iter_real_studies(manifest, source, allow_unverified_directory=True))
    prepared = prepare_study(loaded.study, preprocess)
    packed = pack_study(loaded.study, prepared.series, preprocess, packing_config=packing)
    model_scores = []
    for result in results:
        checkpoint = torch.load(
            result.final_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        model = HierarchicalStudyModel(config.model_config())
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        with torch.inference_mode():
            model_scores.append(
                model.predict_proba(
                    torch.from_numpy(packed.images).unsqueeze(0),
                    torch.from_numpy(packed.slice_mask).unsqueeze(0),
                    torch.from_numpy(packed.series_mask).unsqueeze(0),
                )
                .squeeze(0)
                .numpy()
            )
    expected_scores = np.mean(np.stack(model_scores, axis=0), axis=0)

    submission_path = tmp_path / "submission.csv"
    execution = run_candidate_inference(
        candidate.root,
        manifest_path,
        source,
        submission_path,
        allow_local_smoke=True,
    )
    with submission_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == ("StudyInstanceUID", *MODEL_TARGET_COLUMNS)
    assert len(rows) == 2
    assert rows[1][0] == inference_uid
    np.testing.assert_allclose(
        np.asarray(rows[1][1:], dtype=np.float64), expected_scores, rtol=0.0, atol=1e-7
    )
    assert execution.submission_validation["exact_uid_order"] is True
    assert execution.submission_validation["scores_finite"] is True
    assert execution.submission_validation["scores_bounded"] is True
