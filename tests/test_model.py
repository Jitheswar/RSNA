from __future__ import annotations

import numpy as np
import pytest
import torch

from rsna_knee.model import (
    COMPACT_MODEL_ID,
    COMPACT_POOLING_ID,
    MODEL_CONFIG_SCHEMA_VERSION,
    HierarchicalStudyModel,
    ModelConfig,
    ModelConfigurationError,
    build_study_model,
    restore_study_model,
)


def test_model_has_twelve_outputs_and_ignores_masked_padding() -> None:
    torch.manual_seed(11)
    model = HierarchicalStudyModel(ModelConfig(channels=(2, 4, 8)))
    model.eval()
    images = torch.rand((1, 2, 4, 1, 128, 128))
    slice_mask = torch.tensor([[[True, True, False, False], [True, False, False, False]]])
    series_mask = torch.tensor([[True, True]])

    with torch.no_grad():
        original = model(images, slice_mask, series_mask)
        changed_padding = images.clone()
        changed_padding[:, 0, 2:] = 1000.0
        changed_padding[:, 1, 1:] = -1000.0
        updated = model(changed_padding, slice_mask, series_mask)

    assert original.shape == (1, 12)
    assert torch.allclose(original, updated, atol=1e-6)


def test_model_produces_finite_gradients_for_a_masked_study() -> None:
    torch.manual_seed(12)
    model = HierarchicalStudyModel(ModelConfig(channels=(2, 4, 8)))
    images = torch.rand((1, 1, 2, 1, 128, 128))
    slice_mask = torch.tensor([[[True, True]]])
    series_mask = torch.tensor([[True]])

    loss = model(images, slice_mask, series_mask).square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_model_returns_finite_logits_for_an_all_invalid_study() -> None:
    model = HierarchicalStudyModel(ModelConfig(channels=(2, 4, 8)))

    logits = model(
        torch.zeros((1, 2, 4, 1, 128, 128)),
        torch.zeros((1, 2, 4), dtype=torch.bool),
        torch.zeros((1, 2), dtype=torch.bool),
    )

    assert logits.shape == (1, 12)
    assert np.isfinite(logits.detach().numpy()).all()


def test_versioned_model_configuration_reads_legacy_shape_and_rejects_unknown_identity() -> None:
    legacy = {
        "image_size": [16, 16],
        "channels": [2, 4, 8],
        "input_channels": 1,
        "num_targets": 12,
    }
    config = ModelConfig.from_dict(legacy)

    assert config == ModelConfig(image_size=(16, 16), channels=(2, 4, 8))
    assert ModelConfig.from_dict(config.to_dict()) == config
    assert config.to_dict() == {
        "schema_version": MODEL_CONFIG_SCHEMA_VERSION,
        "model_id": COMPACT_MODEL_ID,
        "pooling_id": COMPACT_POOLING_ID,
        **legacy,
    }
    with pytest.raises(ModelConfigurationError, match="unsupported model identity"):
        ModelConfig.from_dict({**config.to_dict(), "model_id": "unregistered-model-v99"})
    with pytest.raises(ModelConfigurationError, match="unsupported pooling identity"):
        ModelConfig.from_dict({**config.to_dict(), "pooling_id": "unknown-pooling-v1"})
    with pytest.raises(ModelConfigurationError, match="unsupported fields"):
        ModelConfig.from_dict({**config.to_dict(), "weights_path": "/tmp/unreviewed.pt"})


def test_shared_reconstruction_preserves_the_compact_baseline_probabilities() -> None:
    torch.manual_seed(20260925)
    config = ModelConfig(image_size=(16, 16), channels=(2, 4, 8))
    baseline = HierarchicalStudyModel(config)
    baseline.eval()
    images = torch.linspace(0.0, 1.0, 2 * 2 * 16 * 16, dtype=torch.float32).reshape(
        1, 2, 2, 1, 16, 16
    )
    slice_mask = torch.tensor([[[True, True], [True, False]]])
    series_mask = torch.tensor([[True, True]])
    expected = torch.tensor(
        [
            0.48816239833831787,
            0.46565741300582886,
            0.5453450083732605,
            0.4579584300518036,
            0.5125989317893982,
            0.5154812335968018,
            0.43772292137145996,
            0.5449249744415283,
            0.4948369264602661,
            0.47319990396499634,
            0.5446000099182129,
            0.4299467206001282,
        ]
    )
    reconstructed = restore_study_model(config.to_dict(), baseline.state_dict())
    direct = build_study_model(config)
    direct.load_state_dict(baseline.state_dict(), strict=True)
    reconstructed.eval()
    direct.eval()

    with torch.inference_mode():
        baseline_scores = baseline.predict_proba(images, slice_mask, series_mask).squeeze(0)
        reconstructed_scores = reconstructed.predict_proba(images, slice_mask, series_mask).squeeze(0)
        direct_scores = direct.predict_proba(images, slice_mask, series_mask).squeeze(0)

    torch.testing.assert_close(baseline_scores, expected, rtol=0.0, atol=1e-7)
    assert torch.equal(baseline_scores, reconstructed_scores)
    assert torch.equal(baseline_scores, direct_scores)
