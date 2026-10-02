"""Learning and delivery boundaries for the static whole-encoder route."""

import copy

import pytest
import torch

from rsna_knee.feature_pooling import ANATOMICAL_POOLING_ID
from rsna_knee.full_backbone import fixed_shape_masked_bce
from rsna_knee.model import ModelConfig, build_study_model, restore_study_model
from rsna_knee.resnet18 import (
    RESNET18_FIXTURE_ENCODER_ID,
    RESNET18_FULL_FINETUNE_MODEL_ID,
    RESNET18_MODEL_ID,
)
from rsna_knee.training import masked_bce_loss


@pytest.fixture(autouse=True)
def bounded_test_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def config(model_id):
    return ModelConfig(image_size=(32, 32), model_id=model_id,
                       pooling_id=ANATOMICAL_POOLING_ID,
                       encoder_id=RESNET18_FIXTURE_ENCODER_ID, head_hidden_size=8)


def inputs():
    torch.manual_seed(617)
    images = torch.rand(2, 2, 2, 1, 32, 32)
    slices = torch.tensor([[[True, False], [True, True]],
                           [[True, False], [False, False]]])
    series = torch.tensor([[True, True], [True, False]])
    descriptors = torch.zeros(2, 2, 10)
    descriptors[:, 0, 0] = 1
    descriptors[:, 1, 1] = 1
    return images, slices, series, descriptors


def test_whole_encoder_updates_and_reloads_with_padding_isolated(tmp_path):
    torch.manual_seed(60)
    frozen = build_study_model(config(RESNET18_MODEL_ID))
    model = restore_study_model(config(RESNET18_FULL_FINETUNE_MODEL_ID).to_dict(), frozen.state_dict())
    assert all(p.requires_grad for p in model.encoder.parameters())
    batch = inputs()
    frozen.eval(); model.train()
    assert torch.allclose(frozen(*batch), model(*batch), atol=1e-6, rtol=1e-5)
    assert all(not m.training for m in model.encoder.modules() if isinstance(m, torch.nn.BatchNorm2d))
    poisoned = batch[0].clone()
    valid = batch[1] & batch[2].unsqueeze(-1)
    poisoned[~valid] = float('nan')
    assert torch.equal(model(*batch), model(poisoned, *batch[1:]))
    state = copy.deepcopy(model.state_dict())
    buffers = {k: v.clone() for k, v in model.encoder.named_buffers()}
    targets = torch.rand(2, 12)
    mask = torch.ones_like(targets, dtype=torch.bool)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fixed_shape_masked_bce(model(poisoned, *batch[1:]), targets, mask).backward()
    for name in ('conv1.weight', 'layer1.0.conv1.weight', 'layer4.1.conv2.weight'):
        grad = dict(model.encoder.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    optimizer.step()
    for name in ('encoder.conv1.weight', 'encoder.layer1.0.conv1.weight', 'encoder.layer4.1.conv2.weight'):
        assert not torch.equal(state[name], model.state_dict()[name])
    assert all(torch.equal(value, buffers[name]) for name, value in model.encoder.named_buffers())
    model.eval()
    expected = model(*batch).detach()
    path = tmp_path / 'whole-backbone.pt'
    torch.save({'model_config': model.config.to_dict(), 'state_dict': model.state_dict()}, path)
    checkpoint = torch.load(path, weights_only=True)
    restored = restore_study_model(checkpoint['model_config'], checkpoint['state_dict']).eval()
    assert restored.config.model_id == RESNET18_FULL_FINETUNE_MODEL_ID
    assert torch.equal(restored(*batch), expected)


def test_static_model_does_not_select_variable_length_valid_indices(monkeypatch):
    model = build_study_model(config(RESNET18_FULL_FINETUNE_MODEL_ID))
    def forbidden(*args, **kwargs):
        raise AssertionError('data-dependent valid index selection reached')
    monkeypatch.setattr(torch, 'nonzero', forbidden)
    output = model(*inputs())
    assert output.shape == (2, 12) and torch.isfinite(output).all()


def test_encoder_chunk_shapes_stay_constant_when_valid_counts_change():
    model = build_study_model(config(RESNET18_FULL_FINETUNE_MODEL_ID)).eval()
    images = torch.rand(1, 3, 8, 1, 32, 32)
    slices = torch.ones(1, 3, 8, dtype=torch.bool)
    series = torch.ones(1, 3, dtype=torch.bool)
    shapes = []
    hook = model.encoder.register_forward_pre_hook(lambda _, args: shapes.append(tuple(args[0].shape)))
    try:
        model.encode_packed(images, slices, series)
        dense_shapes = shapes[:]
        shapes.clear()
        slices[:, :, 1:] = False
        model.encode_packed(images, slices, series)
        assert shapes == dense_shapes == [(16, 3, 32, 32), (8, 3, 32, 32)]
    finally:
        hook.remove()


def test_masked_loss_matches_selected_entries_and_ignores_missing_nan_targets():
    torch.manual_seed(61)
    logits = torch.randn(3, 12, requires_grad=True)
    targets = torch.rand(3, 12)
    mask = torch.rand(3, 12) > 0.4
    targets[~mask] = float('nan')
    actual = fixed_shape_masked_bce(logits, targets, mask)
    expected = masked_bce_loss(logits, targets, mask)
    assert expected is not None and torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    assert torch.allclose(actual_grad, expected_grad, atol=1e-7, rtol=1e-6)
    assert torch.equal(actual_grad[~mask], torch.zeros_like(actual_grad[~mask]))


def test_empty_mask_produces_zero_loss_and_zero_gradient_without_host_branch():
    logits = torch.full((1, 12), float('nan'), requires_grad=True)
    loss = fixed_shape_masked_bce(logits, torch.full_like(logits, float('nan')), torch.zeros_like(logits, dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_masked_loss_rejects_wrong_target_width():
    with pytest.raises(ValueError):
        fixed_shape_masked_bce(torch.zeros(2, 11), torch.zeros(2, 11), torch.ones(2, 11, dtype=torch.bool))
