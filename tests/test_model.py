from __future__ import annotations

import numpy as np
import torch

from rsna_knee.model import HierarchicalStudyModel, ModelConfig


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
