"""Synthetic CPU mechanics; these checks do not prove TPU execution or MRI gain."""
from copy import deepcopy

import pytest
import torch

from rsna_knee.replicated_masked_loss import admit_global_supervised_entries, rank_supervision, replica_masked_bce
from rsna_knee.training import masked_bce_loss


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def batch(seed=75):
    rng = torch.Generator().manual_seed(seed)
    features = torch.randn(4, 8, generator=rng)
    targets = torch.randint(0, 2, (4, 12), generator=rng).float()
    mask = torch.zeros(4, 12, dtype=torch.bool)
    offset = seed - 75
    for rank, count in enumerate((12, 7 + offset % 5, 3 + offset % 7, 1 + offset % 9)):
        mask[rank, :count] = True
    targets[~mask] = float('nan')
    return features, targets, mask


def test_unequal_supervision_and_four_empty_ranks_match_global_loss_and_gradients():
    _, targets, mask = batch()
    logits = torch.randn(4, 12, generator=torch.Generator().manual_seed(175), requires_grad=True)
    expected = masked_bce_loss(logits, targets, mask)
    assert expected is not None
    denominator = admit_global_supervised_entries(targets, mask)
    replica_losses = []
    for rank in range(8):
        t, m, d = rank_supervision(targets, mask, rank)
        z = logits[rank:rank + 1] if rank < 4 else torch.full((1, 12), float('nan'), requires_grad=True)
        replica_losses.append(replica_masked_bce(z, t, m, d))
    actual = torch.stack(replica_losses).mean()
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    a = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    e = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(a, e, atol=1e-7, rtol=1e-6)
    assert denominator.item() == 23
    assert all(loss.item() == 0 for loss in replica_losses[4:])
    assert torch.equal(a[~mask], torch.zeros_like(a[~mask]))
    # Incorrect averaging of four local means is measurably different here.
    local_means = torch.stack([masked_bce_loss(logits[r:r+1], targets[r:r+1], mask[r:r+1]) for r in range(4)]).mean()
    assert not torch.allclose(local_means, expected, atol=1e-5, rtol=1e-5)


def test_replica_loss_has_no_device_scalar_read_or_variable_index_selection(monkeypatch):
    _, targets, mask = batch()
    t, m, d = rank_supervision(targets, mask, 0)
    def forbidden(*args, **kwargs):
        raise AssertionError('Device scalar/index read reached')
    monkeypatch.setattr(torch.Tensor, 'item', forbidden)
    monkeypatch.setattr(torch, 'nonzero', forbidden)
    loss = replica_masked_bce(torch.zeros(1, 12, requires_grad=True), t, m, d)
    assert loss.shape == () and loss.requires_grad
    loss.backward()


@pytest.mark.parametrize('defect', ['empty', 'nan', 'range', 'shape', 'dtype', 'non_cpu'])
def test_bad_global_admission_rejected(defect):
    _, targets, mask = batch()
    if defect == 'empty': mask.zero_()
    if defect == 'nan': targets[0, 0] = float('nan')
    if defect == 'range': targets[0, 0] = 2
    if defect == 'shape': targets = targets[:3];mask = mask[:3]
    if defect == 'dtype': mask = mask.float()
    if defect == 'non_cpu': targets = targets.to('meta');mask = mask.to('meta')
    with pytest.raises(ValueError): admit_global_supervised_entries(targets, mask)


@pytest.mark.parametrize('rank', [-1, 8, True])
def test_unregistered_replica_never_gets_an_input(rank):
    _, targets, mask = batch()
    with pytest.raises(ValueError): rank_supervision(targets, mask, rank)


def model():
    torch.manual_seed(20261002)
    return torch.nn.Sequential(torch.nn.Linear(8, 7), torch.nn.ReLU(), torch.nn.Linear(7, 32), torch.nn.ReLU(), torch.nn.Linear(32, 12))


def optimizer_for(m):
    return torch.optim.Adam([{'params': m[0].parameters(), 'lr': 1e-5},
        {'params': list(m[2].parameters()) + list(m[4].parameters()), 'lr': 1e-4}],
        betas=(.9, .999), eps=1e-8, weight_decay=0, foreach=False)


def step(m, optimizer, values, replicated):
    x, targets, mask = values
    optimizer.zero_grad(set_to_none=True)
    if replicated:
        for rank in range(8):
            t, supervision, denominator = rank_supervision(targets, mask, rank)
            features = x[rank:rank + 1] if rank < 4 else torch.zeros(1, 8)
            replica_masked_bce(m(features), t, supervision, denominator).backward()
        for p in m.parameters():
            p.grad.div_(8)
    else:
        masked_bce_loss(m(x), targets, mask).backward()
    optimizer.step()


def test_ordinary_adam_sixteen_updates_and_fresh_checkpoint_replay(tmp_path):
    batches = [batch(75 + index) for index in range(16)]
    assert len({int(values[2].sum()) for values in batches}) > 1
    original, replicated = model(), model()
    a, b = optimizer_for(original), optimizer_for(replicated)
    for values in batches:
        step(original, a, values, False)
        step(replicated, b, values, True)
    for x, y in zip(original.parameters(), replicated.parameters(), strict=True):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-4)
    for x, y in zip(a.state.values(), b.state.values(), strict=True):
        assert x['step'].item() == y['step'].item() == 16
        for key in ('exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(x[key], y[key], atol=1e-6, rtol=1e-4)
    staged = model();c = optimizer_for(staged)
    for values in batches[:8]: step(staged, c, values, True)
    path = tmp_path / 'pilot-synthetic-checkpoint.pt'
    torch.save(dict(model=staged.state_dict(), optimizer=c.state_dict(), next_batch=8), path)
    loaded = torch.load(path, weights_only=True)
    resumed = model();d = optimizer_for(resumed)
    resumed.load_state_dict(loaded['model']);d.load_state_dict(loaded['optimizer'])
    for values in batches[loaded['next_batch']:]: step(resumed, d, values, True)
    for x, y in zip(replicated.parameters(), resumed.parameters(), strict=True): assert torch.equal(x, y)
    for x, y in zip(b.state.values(), d.state.values(), strict=True):
        assert all(torch.equal(x[key], y[key]) for key in ('step', 'exp_avg', 'exp_avg_sq'))
    assert [g['lr'] for g in d.param_groups] == [1e-5, 1e-4]


def test_whole_backbone_gradients_adam_and_fixed_normalization_match_four_study_mean():
    from rsna_knee.feature_pooling import ANATOMICAL_POOLING_ID
    from rsna_knee.model import ModelConfig, build_study_model
    from rsna_knee.resnet18 import RESNET18_FIXTURE_ENCODER_ID, RESNET18_FULL_FINETUNE_MODEL_ID
    torch.manual_seed(20261002)
    config = ModelConfig(image_size=(32, 32), model_id=RESNET18_FULL_FINETUNE_MODEL_ID,
        pooling_id=ANATOMICAL_POOLING_ID, encoder_id=RESNET18_FIXTURE_ENCODER_ID, head_hidden_size=32)
    original = build_study_model(config).train()
    replicated = deepcopy(original).train()
    images = torch.rand(4, 2, 2, 1, 32, 32)
    slices = torch.ones(4, 2, 2, dtype=torch.bool)
    slices[1:, 1, 1] = False
    series = torch.ones(4, 2, dtype=torch.bool)
    descriptors = torch.zeros(4, 2, 10)
    descriptors[:, 0, 0] = 1;descriptors[:, 1, 1] = 1
    images[~slices] = float('nan')
    _, targets, mask = batch()
    buffers = {k: v.clone() for k, v in original.named_buffers()}
    def optimizer(m):
        return torch.optim.Adam([{'params': m.encoder.parameters(), 'lr': 1e-5}, {'params': m.head.parameters(), 'lr': 1e-4}],
            betas=(.9, .999), eps=1e-8, weight_decay=0, foreach=False)
    a, b = optimizer(original), optimizer(replicated)
    expected = masked_bce_loss(original(images, slices, series, descriptors), targets, mask)
    expected.backward()
    for rank in range(8):
        t, supervision, denominator = rank_supervision(targets, mask, rank)
        inputs = tuple(x[rank:rank + 1] for x in (images, slices, series, descriptors)) if rank < 4 else (
            torch.zeros_like(images[:1]), torch.zeros_like(slices[:1]), torch.zeros_like(series[:1]), torch.zeros_like(descriptors[:1]))
        replica_masked_bce(replicated(*inputs), t, supervision, denominator).backward()
    for p in replicated.parameters(): p.grad.div_(8)
    for x, y in zip(original.parameters(), replicated.parameters(), strict=True):
        torch.testing.assert_close(x.grad, y.grad, atol=1e-6, rtol=1e-4)
    a.step();b.step()
    for x, y in zip(original.parameters(), replicated.parameters(), strict=True):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-4)
    for x, y in zip(a.state.values(), b.state.values(), strict=True):
        assert x['step'].item() == y['step'].item() == 1
        for key in ('exp_avg', 'exp_avg_sq'): torch.testing.assert_close(x[key], y[key], atol=1e-6, rtol=1e-4)
    for m in (original, replicated):
        assert all(torch.equal(value, buffers[name]) for name, value in m.named_buffers())


@pytest.mark.parametrize('defect', ['world_size', 'logit_shape', 'target_dtype', 'denominator_shape', 'device'])
def test_invalid_replica_contract_rejected(defect):
    _, targets, mask = batch()
    t, m, d = rank_supervision(targets, mask, 0)
    z = torch.zeros(1, 12)
    world_size = 8
    if defect == 'world_size': world_size = 4
    if defect == 'logit_shape': z = torch.zeros(2, 12)
    if defect == 'target_dtype': t = t.double()
    if defect == 'denominator_shape': d = d.reshape(1)
    if defect == 'device': d = d.to('meta')
    with pytest.raises(ValueError): replica_masked_bce(z, t, m, d, world_size=world_size)


def test_padding_replica_with_absent_nan_logits_has_exact_zero_gradient():
    _, targets, mask = batch()
    t, m, d = rank_supervision(targets, mask, 7)
    logits = torch.full((1, 12), float('nan'), requires_grad=True)
    loss = replica_masked_bce(logits, t, m, d)
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))
