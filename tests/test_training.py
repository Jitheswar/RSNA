from __future__ import annotations

import numpy as np
import pytest
import torch

from rsna_knee.training import (
    TrainingConfig,
    TrainingConfigurationError,
    TrainingExample,
    TrainingProgressError,
    load_training_checkpoint,
    masked_bce_loss,
    run_cpu_preflight,
    train_fixed_budget,
    train_one_fold,
)
from rsna_knee.folds import freeze_folds
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.synthetic import make_synthetic_study
from rsna_knee.targets import TARGET_COLUMNS


def test_masked_bce_averages_only_supervised_entries() -> None:
    logits = torch.zeros((1, 12))
    logits[0, 2] = 2.0
    targets = torch.zeros((1, 12))
    targets[0, 0] = 1.0
    targets[0, 2] = 1.0
    supervision_mask = torch.zeros((1, 12), dtype=torch.bool)
    supervision_mask[0, (0, 2)] = True

    loss = masked_bce_loss(logits, targets, supervision_mask)

    assert loss is not None
    expected = (torch.nn.functional.softplus(torch.tensor(0.0)) + torch.nn.functional.softplus(torch.tensor(-2.0))) / 2
    assert torch.allclose(loss, expected)


def test_fully_masked_batches_make_no_optimizer_progress() -> None:
    example = TrainingExample(
        study_instance_uid="masked-study",
        images=np.zeros((1, 1, 1, 16, 16), dtype=np.float32),
        slice_mask=np.ones((1, 1), dtype=bool),
        series_mask=np.ones(1, dtype=bool),
        targets=np.zeros(12, dtype=np.float32),
        supervision_mask=np.zeros(12, dtype=bool),
    )
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=1,
        max_series=1,
        successful_optimizer_steps=1,
        persistent_no_progress_limit=2,
    )

    with pytest.raises(TrainingProgressError, match="no supervised progress"):
        train_fixed_budget((example,), config)


def _tiny_example(study_uid: str, target: int) -> TrainingExample:
    return TrainingExample(
        study_instance_uid=study_uid,
        images=np.full((1, 1, 1, 16, 16), 0.25 + target * 0.5, dtype=np.float32),
        slice_mask=np.ones((1, 1), dtype=bool),
        series_mask=np.ones(1, dtype=bool),
        targets=np.full(12, target, dtype=np.float32),
        supervision_mask=np.ones(12, dtype=bool),
    )


def test_fixed_budget_makes_learning_progress_and_records_each_step(tmp_path) -> None:
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=1,
        max_series=1,
        successful_optimizer_steps=12,
        checkpoint_interval_steps=4,
    )
    result = train_fixed_budget(
        (_tiny_example("tiny-negative", 0), _tiny_example("tiny-positive", 1)),
        config,
        artifact_dir=tmp_path / "run",
    )

    assert result.successful_optimizer_steps == 12
    assert result.scheduler_steps == 12
    assert len(result.loss_history) == 12
    assert result.loss_history[-1] < result.loss_history[0]
    assert result.final_checkpoint_path is not None
    assert result.final_checkpoint_path.exists()
    assert result.latest_checkpoint_path is not None
    assert result.latest_checkpoint_path.exists()

    payload = load_training_checkpoint(
        result.final_checkpoint_path,
        expected_config=config,
    )
    assert payload["selection_policy"] == "final_checkpoint"
    assert payload["progress"]["successful_optimizer_steps"] == 12
    with pytest.raises(TrainingConfigurationError, match="scientific configuration"):
        load_training_checkpoint(
            result.final_checkpoint_path,
            expected_config=TrainingConfig(
                image_size=(16, 16),
                channels=(2, 4, 8),
                max_slices=1,
                max_series=1,
                successful_optimizer_steps=13,
            ),
        )


def test_one_fold_uses_reference_gold_only_and_evaluates_final_checkpoint(tmp_path) -> None:
    studies = tuple(
        make_synthetic_study(f"gold-{index}", series_count=1, slices_per_series=2, seed=100 + index)
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
        )
        + (SourceStudy("report-only", True, (None,) * len(TARGET_COLUMNS)),),
        (),
    )
    fold_manifest = freeze_folds(source, source_version="synthetic-training-v1", seed=17)
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=2,
        max_series=1,
        successful_optimizer_steps=2,
        checkpoint_interval_steps=1,
    )

    result = train_one_fold(
        {study.study_instance_uid: study for study in studies},
        fold_manifest,
        0,
        tmp_path / "fold-0",
        config=config,
        source_version="synthetic-training-v1",
        require_preflight=False,
        allow_local_smoke=True,
    )

    assert result.successful_optimizer_steps == 2
    assert result.evaluation is not None
    assert set(result.training_study_uids).isdisjoint(result.held_out_study_uids)
    assert "report-only" not in result.training_study_uids
    assert (tmp_path / "fold-0" / "configuration.json").exists()
    assert (tmp_path / "fold-0" / "metrics.json").exists()
    assert (tmp_path / "fold-0" / "run.json").exists()
    assert result.evaluation.fold == 0


def test_cpu_preflight_records_the_required_runtime_checks(tmp_path) -> None:
    result = run_cpu_preflight(
        output_path=tmp_path / "preflight.json",
        require_isolated_python=True,
    )

    assert result.passed is True
    assert all(result.checks.values())
    assert result.environment["python_version"].startswith("3.11")
    assert (tmp_path / "preflight.json").exists()
