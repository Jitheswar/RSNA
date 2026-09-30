from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rsna_knee.evaluation import TARGET_COLUMNS
from rsna_knee.folds import freeze_folds
from rsna_knee.image_variant import TICKET31_RESNET18_VARIANT, image_model_training_config
from rsna_knee.inference import InferenceEngine, infer_studies
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.model import build_study_model
from rsna_knee.preprocessing import PackingConfig
from rsna_knee.resnet18 import RESNET18_FIXTURE_ENCODER_ID
from rsna_knee.synthetic import make_synthetic_study
from rsna_knee.training import load_training_checkpoint, train_one_fold


def test_ticket31_synthetic_fold_reloads_predicts_held_out_and_runs_one_study_offline(
    tmp_path: Path,
) -> None:
    studies = tuple(
        make_synthetic_study(
            f"ticket31-synthetic-{index}",
            series_count=1,
            slices_per_series=2,
            seed=3100 + index,
        )
        for index in range(9)
    )
    source = SourceManifest(
        "train",
        tuple(
            SourceStudy(
                study.study_instance_uid,
                True,
                tuple((index + target_index) % 2 for target_index in range(len(TARGET_COLUMNS))),
            )
            for index, study in enumerate(studies)
        ),
        (),
    )
    folds = freeze_folds(source, source_version="synthetic-ticket31-v1", seed=3101)

    config = image_model_training_config(
        TICKET31_RESNET18_VARIANT,
        packing_config=PackingConfig(max_slices=2, max_series=1),
    )
    config = replace(
        config,
        encoder_id=RESNET18_FIXTURE_ENCODER_ID,
        encoder_weights_sha256=None,
        successful_optimizer_steps=1,
        checkpoint_interval_steps=1,
        seed=3102,
    )
    assert config.encoder_id == RESNET18_FIXTURE_ENCODER_ID
    assert config.encoder_weights_sha256 is None

    previous_thread_count = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = train_one_fold(
            studies,
            folds,
            0,
            tmp_path / "ticket31-fold-0",
            config=config,
            source_version="synthetic-ticket31-v1",
            source_fingerprint="synthetic-ticket31-source-v1",
            require_preflight=False,
            allow_local_smoke=True,
        )

        assert result.successful_optimizer_steps == 1
        assert result.final_checkpoint_path is not None
        assert result.evaluation is not None
        assert set(result.training_study_uids).isdisjoint(result.held_out_study_uids)
        assert set(result.predictions) == set(result.held_out_study_uids)
        assert len(result.predictions) == 3
        assert all(
            np.isfinite(scores).all() and ((scores >= 0.0) & (scores <= 1.0)).all()
            for scores in result.predictions.values()
        )

        restored_model = build_study_model(config.model_config())
        checkpoint = load_training_checkpoint(
            result.final_checkpoint_path,
            model=restored_model,
            expected_config=config,
            expected_fold=0,
            expected_checkpoint_kind="final_selected",
        )
        assert checkpoint["checkpoint_kind"] == "final_selected"
        assert restored_model.config.encoder_id == RESNET18_FIXTURE_ENCODER_ID

        study_by_uid = {study.study_instance_uid: study for study in studies}
        held_out_studies = tuple(study_by_uid[uid] for uid in result.held_out_study_uids)
        replay_path = tmp_path / "reloaded-held-out-predictions.csv"
        replay = infer_studies(
            InferenceEngine(
                restored_model,
                config.preprocess_config(),
                packing_config=config.packing_config(),
            ),
            held_out_studies,
            replay_path,
        )
        assert replay.output_path == replay_path
        for prediction in replay.results:
            np.testing.assert_allclose(
                prediction.scores,
                result.predictions[prediction.study_instance_uid],
                rtol=0.0,
                atol=1e-6,
            )

        assert result.execution_attempt_id is not None
        attempt_dir = (
            tmp_path / "ticket31-fold-0" / "attempts" / result.execution_attempt_id
        )
        offline_csv = attempt_dir / "offline-study-submission.csv"
        offline_evidence_path = attempt_dir / "offline-study-evidence.json"
        assert offline_csv.is_file()
        assert offline_evidence_path.is_file()
        evidence = json.loads(offline_evidence_path.read_text(encoding="utf-8"))
        assert evidence["scope"] == "one-held-out-study-from-one-fold;not-full-oof"
        assert evidence["prediction_count"] == 1
        assert evidence["runtime_network_download"] is False
        offline_uid = evidence["study_instance_uid"]
        assert offline_uid in result.held_out_study_uids

        with offline_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert tuple(rows[0]) == ("StudyInstanceUID", *TARGET_COLUMNS)
        assert len(rows) == 2
        assert rows[1][0] == offline_uid
        np.testing.assert_allclose(
            np.asarray(rows[1][1:], dtype=np.float64),
            result.predictions[offline_uid],
            rtol=0.0,
            atol=1e-6,
        )
    finally:
        torch.set_num_threads(previous_thread_count)
