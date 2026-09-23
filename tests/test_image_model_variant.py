from __future__ import annotations

import torch

from rsna_knee.image_variant import (
    TICKET19_VARIANT,
    image_model_training_config,
)
from rsna_knee.model import HierarchicalStudyModel
from rsna_knee.training import TrainingConfig


def test_ticket19_variant_changes_only_the_recorded_encoder_width() -> None:
    baseline = image_model_training_config("mechanics-baseline")
    candidate = image_model_training_config(TICKET19_VARIANT)

    assert baseline.channels == (8, 16, 32)
    assert candidate.channels == (16, 32, 64)
    assert candidate.image_size == baseline.image_size == (128, 128)
    assert candidate.max_slices == baseline.max_slices == 8
    assert candidate.max_series == baseline.max_series == 16
    assert candidate.batch_size == baseline.batch_size == 2
    assert candidate.successful_optimizer_steps == baseline.successful_optimizer_steps == 100
    assert candidate.seed == baseline.seed
    assert candidate.batch_order_seed == baseline.batch_order_seed
    assert candidate.optimizer == baseline.optimizer == "Adam"
    assert candidate.learning_rate == baseline.learning_rate == 0.001
    assert candidate.schedule == baseline.schedule == "constant"
    assert candidate.loss == baseline.loss == "masked_bce"
    assert candidate.augmentation == baseline.augmentation == "none"
    assert candidate.config_fingerprint != baseline.config_fingerprint


def test_ticket19_encoder_keeps_the_masked_twelve_score_model_contract() -> None:
    config: TrainingConfig = image_model_training_config(TICKET19_VARIANT)
    model = HierarchicalStudyModel(config.model_config()).eval()
    images = torch.rand((1, 1, 2, 1, 128, 128))
    slice_mask = torch.tensor([[[True, False]]])
    series_mask = torch.tensor([[True]])

    with torch.no_grad():
        original = model.predict_proba(images, slice_mask, series_mask)
        changed_padding = images.clone()
        changed_padding[:, :, 1] = 10_000.0
        updated = model.predict_proba(changed_padding, slice_mask, series_mask)

    assert original.shape == (1, 12)
    assert torch.isfinite(original).all()
    assert ((original >= 0.0) & (original <= 1.0)).all()
    assert torch.allclose(original, updated, atol=1e-6)
