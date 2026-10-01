"""The adaptation seam must preserve native packing and actual gradients."""

import numpy as np
import pytest
import torch

from rsna_knee.model import ModelConfig, build_study_model
from rsna_knee.prefix_adaptation import prefix_logits, prefix_pooled_features
from rsna_knee.prefix_cache import PrefixEntry
from rsna_knee.resnet18 import RESNET18_FINETUNE_MODEL_ID, RESNET18_FIXTURE_ENCODER_ID


@pytest.fixture
def real_prefix_batch():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = ModelConfig(image_size=(224, 224), model_id=RESNET18_FINETUNE_MODEL_ID,
                         pooling_id="plane-stratified-series-mean-v1",
                         encoder_id=RESNET18_FIXTURE_ENCODER_ID)
    model = build_study_model(config)
    model.train()
    images = torch.zeros((2, 16, 8, 1, 224, 224))
    slices = torch.zeros((2, 16, 8), dtype=torch.bool)
    series = torch.zeros((2, 16), dtype=torch.bool)
    descriptors = torch.zeros((2, 16, 10))
    slices[0, 1, [0, 4]] = True
    slices[1, 2, [1, 7]] = True
    series[0, 1] = True
    series[1, 2] = True
    descriptors[0, 1, 0] = 1
    descriptors[1, 2, 2] = 1
    descriptors[:, :, 4:] = 0.25
    images[slices] = torch.rand((4, 1, 224, 224))
    entries = []
    with torch.no_grad():
        for index in range(2):
            flat_indices = torch.nonzero(slices[index].flatten()).flatten()
            pixels = images[index].reshape(-1, 1, 224, 224)[flat_indices].expand(-1, 3, -1, -1)
            prefix = model.encoder.forward_before_layer4((pixels - model.imagenet_mean) / model.imagenet_std)
            entries.append(PrefixEntry(prefix.numpy(), slices[index].numpy(), series[index].numpy(),
                                       descriptors[index].numpy(), flat_indices.numpy(), {}))
    return model, images, slices, series, descriptors, entries


def test_cached_batch_matches_native_logits_gradients_and_fixed_buffers(real_prefix_batch):
    model, images, slices, series, descriptors, entries = real_prefix_batch
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    native = model(images, slices, series, descriptors)
    native.square().mean().backward()
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                 if parameter.requires_grad}
    model.zero_grad(set_to_none=True)
    cached = prefix_logits(model, entries)
    torch.testing.assert_close(cached, native.detach(), rtol=0, atol=1e-6)
    cached.square().mean().backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=1e-6)
        else:
            assert parameter.grad is None
    assert all(torch.equal(value, initial[key]) for key, value in model.state_dict().items())
    assert any(gradient.abs().sum() > 0 for name, gradient in gradients.items()
               if name.startswith("encoder.layer4."))


def test_zero_image_control_retains_native_masks_and_descriptors(real_prefix_batch):
    model, _, slices, series, descriptors, entries = real_prefix_batch
    expected = model._pool_features(torch.zeros((2, 16, 8, 512)), slices, series, descriptors)
    actual = prefix_pooled_features(model, entries, zero_images=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert np.any(actual.numpy() != 0)


def test_empty_prefix_batch_is_rejected(real_prefix_batch):
    model, *_ = real_prefix_batch
    with pytest.raises(ValueError, match="nonempty"):
        prefix_logits(model, [])
