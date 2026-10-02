"""Portable state retains exact Adam, scheduler, sampler, metadata and RNG."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import random

import numpy as np
import pytest
import torch

from rsna_knee.tpu_pilot_state import (capture_host_rng, decode_host_rng, encode_host_rng,
    load_cpu_reference, portable_cpu_reference, restore_host_rng, unpack_cpu_reference)

SHA = 'a' * 64


def model_and_optimizer():
    torch.manual_seed(75)
    model = torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.ReLU(), torch.nn.Linear(7, 12))
    optimizer = torch.optim.Adam([dict(params=model[0].parameters(), lr=1e-5), dict(params=model[2].parameters(), lr=1e-4)])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    return model, optimizer, scheduler


def update(model, optimizer, scheduler):
    # Consume each RNG family, including the cached Gaussian fields.
    scalar = random.random() + float(np.random.normal())
    x = torch.randn(4, 4) * scalar
    y = torch.rand(4, 12)
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.binary_cross_entropy_with_logits(model(x), y).backward()
    optimizer.step();scheduler.step()


def payload(model, optimizer, scheduler, step):
    return dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(), config=dict(model=dict(id='synthetic')),
        identity=dict(disposable=True, original_fit_updates_credited=False),
        sampler_state=dict(kind='deterministic-cyclic', seed=20260912, next_batch_index=step),
        progress=dict(successful_optimizer_steps=step, attempted_batches=step, scheduler_steps=step,
            next_batch_index=step, sample_exposures=4*step, skipped_all_masked_batches=0),
        rng_state=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state()))


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
        if isinstance(a, OrderedDict) and hasattr(a, '_metadata'):
            equal(a._metadata, b._metadata)
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x,y in zip(a,b,strict=True): equal(x,y)
    else:
        assert a == b


def test_rng_admission_does_not_consume_rng_and_restores_all_cached_draws():
    random.seed(75);np.random.seed(75);torch.manual_seed(75)
    random.gauss(0,1);np.random.normal()
    before = capture_host_rng()
    equal(before, capture_host_rng())
    expected = (random.gauss(0,1), random.random(), np.random.normal(size=4), torch.randn(4))
    restore_host_rng(before)
    actual = (random.gauss(0,1), random.random(), np.random.normal(size=4), torch.randn(4))
    equal(expected, actual)
    equal(encode_host_rng(decode_host_rng(before)), before)


def test_restricted_packet_replays_all_state_to_sixteen_without_original_mutation(tmp_path):
    random.seed(75);np.random.seed(75)
    model, optimizer, scheduler = model_and_optimizer()
    for _ in range(8): update(model, optimizer, scheduler)
    original = payload(model, optimizer, scheduler, 8)
    before = deepcopy(original)
    packet = portable_cpu_reference(original, parent_sha256=SHA, step=8)
    equal(original, before)
    path = tmp_path / 'portable.pt';torch.save(packet, path)
    # No added NumPy safe globals and no weights_only=False remote load.
    restricted = torch.load(path, weights_only=True)
    restored = unpack_cpu_reference(restricted, parent_sha256=SHA, step=8)
    equal(original, restored)
    packet_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    equal(restored, load_cpu_reference(path, packet_sha256=packet_sha, parent_sha256=SHA, step=8))
    for _ in range(8): update(model, optimizer, scheduler)
    expected = payload(model, optimizer, scheduler, 16)
    fresh, new_optimizer, new_scheduler = model_and_optimizer()
    fresh.load_state_dict(restored['model_state_dict'], strict=True)
    new_optimizer.load_state_dict(restored['optimizer_state_dict'])
    new_scheduler.load_state_dict(restored['scheduler_state_dict'])
    restore_host_rng(restricted['payload']['rng_state'])
    for _ in range(8): update(fresh, new_optimizer, new_scheduler)
    equal(expected, payload(fresh, new_optimizer, new_scheduler, 16))
    with pytest.raises(ValueError, match='file changed'):
        load_cpu_reference(path, packet_sha256='b'*64, parent_sha256=SHA, step=8)


def test_initial_model_packet_has_no_unearned_optimizer_steps():
    model, _, _ = model_and_optimizer()
    initial = dict(model_state_dict=model.state_dict(), config={}, disposable=True, initial_model_state_sha256=SHA,
        rng_state=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state()))
    packet = portable_cpu_reference(initial, parent_sha256=SHA, step=0)
    assert packet['actual_TPU_execution'] is False
    equal(initial, unpack_cpu_reference(packet, parent_sha256=SHA, step=0))
    initial['progress'] = {}
    with pytest.raises(ValueError, match='learned updates'):
        portable_cpu_reference(initial, parent_sha256=SHA, step=0)


@pytest.mark.parametrize('defect', ['word_range', 'word_dtype', 'word_shape', 'position', 'algorithm', 'torch_state'])
def test_bad_rng_state_rejected(defect):
    packet = capture_host_rng()
    if defect == 'word_range': packet['numpy']['keys'][0] = 2**32
    if defect == 'word_dtype': packet['numpy']['keys'] = packet['numpy']['keys'].float()
    if defect == 'word_shape': packet['numpy']['keys'] = packet['numpy']['keys'][:-1]
    if defect == 'position': packet['numpy']['position'] = 625
    if defect == 'algorithm': packet['numpy']['algorithm'] = 'changed'
    if defect == 'torch_state': packet['torch'] = torch.zeros(1, dtype=torch.uint8)
    with pytest.raises(ValueError): decode_host_rng(packet)


@pytest.mark.parametrize('defect', ['source_sha', 'steps', 'exposures', 'missing_adam', 'credited_fit', 'non_cpu', 'object'])
def test_invalid_reference_never_becomes_portable(defect):
    model, optimizer, scheduler = model_and_optimizer()
    for _ in range(3): update(model, optimizer, scheduler)
    value = payload(model, optimizer, scheduler, 3)
    sha = SHA
    if defect == 'source_sha': sha = 'bad'
    if defect == 'steps': value['progress']['scheduler_steps'] = 2
    if defect == 'exposures': value['progress']['sample_exposures'] = 24
    if defect == 'missing_adam': value.pop('optimizer_state_dict')
    if defect == 'credited_fit': value['identity']['original_fit_updates_credited'] = True
    if defect == 'non_cpu': value['model_state_dict']['0.weight'] = value['model_state_dict']['0.weight'].to('meta')
    if defect == 'object': value['unsupported'] = object()
    with pytest.raises(ValueError): portable_cpu_reference(value, parent_sha256=sha, step=3)


def test_remote_packet_identity_cannot_claim_TPU_or_relabel_source():
    model, optimizer, scheduler = model_and_optimizer()
    for _ in range(3): update(model, optimizer, scheduler)
    packet = portable_cpu_reference(payload(model, optimizer, scheduler, 3), parent_sha256=SHA, step=3)
    for key in ('actual_TPU_execution', 'original_fit_updates_credited', 'candidate_promotion', 'competition_submission'):
        bad = deepcopy(packet);bad[key] = True
        with pytest.raises(ValueError): unpack_cpu_reference(bad, parent_sha256=SHA, step=3)
    with pytest.raises(ValueError): unpack_cpu_reference(packet, parent_sha256='b'*64, step=3)


def test_original_numpy_pickle_requires_conversion_before_restricted_loading(tmp_path):
    import pickle
    model, optimizer, scheduler = model_and_optimizer()
    for _ in range(3): update(model, optimizer, scheduler)
    original = payload(model, optimizer, scheduler, 3)
    path = tmp_path / "original.pt"
    torch.save(original, path)
    with pytest.raises(pickle.UnpicklingError):
        torch.load(path, weights_only=True)
    original_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    converted = portable_cpu_reference(original, parent_sha256=original_sha, step=3)
    target = tmp_path / "converted.pt";torch.save(converted, target)
    equal(original, unpack_cpu_reference(torch.load(target, weights_only=True), parent_sha256=original_sha, step=3))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_sha
