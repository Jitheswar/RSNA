"""CPU input contracts; not original source MRI or actual TPU collectives."""
from copy import deepcopy

import pytest
import torch

from rsna_knee.tpu_pilot_batches import global_source_exposures, map_global_batch_to_replica
from rsna_knee.replicated_masked_loss import replica_masked_bce
from rsna_knee.training import masked_bce_loss


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def batch():
    # Expand constant original224 geometry rather than spend memory on a
    # synthetic decoded archive. The production mapping preserves these values.
    images = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1, 1, 1).expand(4, 16, 8, 1, 224, 224)
    descriptors = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1).expand(4, 16, 10)
    masks = torch.arange(12)[None] < torch.tensor([12, 8, 3, 1])[:, None]
    targets = torch.ones(4, 12);targets[~masks] = float('nan')
    return dict(images=images, slice_mask=torch.ones(4, 16, 8, dtype=torch.bool), series_mask=torch.ones(4, 16, dtype=torch.bool),
        series_descriptors=descriptors, targets=targets, supervision_mask=masks)


def test_all_eight_ranks_keep_original_order_global_denominator_and_real_exposures():
    values = batch()
    replicas = [map_global_batch_to_replica(values, r) for r in range(8)]
    assert [r.role for r in replicas] == ['reference', 'reference', 'weak', 'weak', 'padding', 'padding', 'padding', 'padding']
    assert sum(r.sample_exposures for r in replicas) == 4
    assert sum(r.supervised_entry_exposures for r in replicas) == 24
    for rank, replica in enumerate(replicas):
        assert replica.global_supervised_entries.item() == 24
        assert replica.model_inputs[0].shape == (1, 16, 8, 1, 224, 224)
        if rank < 4:
            for i, name in enumerate(('images', 'slice_mask', 'series_mask', 'series_descriptors')):
                assert torch.equal(replica.model_inputs[i], values[name][rank:rank + 1])
            torch.testing.assert_close(replica.targets, values['targets'][rank:rank + 1], equal_nan=True)
            assert torch.equal(replica.supervision_mask, values['supervision_mask'][rank:rank + 1])
        else:
            assert all(not bool(v.any()) for v in replica.model_inputs)
            assert not bool(replica.supervision_mask.any()) and not bool(replica.targets.any())
            assert replica.sample_exposures == replica.supervised_entry_exposures == 0
    assert bool(torch.isnan(values['targets'][~values['supervision_mask']]).all())


def test_mapped_losses_preserve_combined_entry_mean_including_empty_ranks():
    values = batch()
    logits = torch.randn(4, 12, generator=torch.Generator().manual_seed(75), requires_grad=True)
    expected = masked_bce_loss(logits, values['targets'], values['supervision_mask'])
    losses = []
    for rank in range(8):
        replica = map_global_batch_to_replica(values, rank)
        z = logits[rank:rank + 1] if rank < 4 else torch.ones(1,12,requires_grad=True)
        losses.append(replica_masked_bce(z, replica.targets, replica.supervision_mask, replica.global_supervised_entries))
    actual = torch.stack(losses).mean()
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0], atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('defect', ['image_size', 'pixel_dtype', 'mask_dtype', 'non_cpu', 'empty_MRI', 'empty_real_target', 'supervised_nan', 'missing_field', 'rank'])
def test_corrupt_or_changed_original_input_rejected(defect):
    value = batch();rank = 0
    if defect == 'image_size': value['images'] = value['images'][..., :32, :32]
    if defect == 'pixel_dtype': value['images'] = value['images'].to(torch.float16)
    if defect == 'mask_dtype': value['series_mask'] = value['series_mask'].float()
    if defect == 'non_cpu': value['images'] = value['images'].to('meta')
    if defect == 'empty_MRI': value['slice_mask'][2].zero_()
    if defect == 'empty_real_target': value['supervision_mask'][2].zero_()
    if defect == 'supervised_nan': value['targets'][2, 0] = float('nan')
    if defect == 'missing_field': value.pop('series_descriptors')
    if defect == 'rank': rank = True
    with pytest.raises(ValueError): map_global_batch_to_replica(value, rank)


def test_prefix_source_counts_match_unchanged_four_real_draws_after_resume():
    masks = []
    for step in range(16):
        mask = torch.arange(12)[None] < torch.tensor([12,12,1+step%7,1+step%5])[:,None]
        masks.append(mask)
    before = deepcopy(masks)
    for boundary in (0,3,8,16):
        exposure = global_source_exposures(masks, boundary)
        assert exposure['reference'] == dict(sample_exposures=2*boundary, supervised_entry_exposures=24*boundary)
        assert exposure['weak'] == dict(sample_exposures=2*boundary, supervised_entry_exposures=sum(int(m[2:].sum()) for m in masks[:boundary]))
        assert exposure['mixed'] == exposure['pseudo'] == dict(sample_exposures=0, supervised_entry_exposures=0)
        assert sum(v['sample_exposures'] for v in exposure.values()) == 4*boundary
    assert all(torch.equal(a,b) for a,b in zip(before,masks,strict=True))


@pytest.mark.parametrize('defect', ['subset', 'boundary', 'bool_boundary', 'padding_mask'])
def test_subset_resampling_or_padding_credit_rejected(defect):
    masks = [batch()['supervision_mask'] for _ in range(16)];boundary = 16
    if defect == 'subset': masks = masks[:8]
    if defect == 'boundary': boundary = 17
    if defect == 'bool_boundary': boundary = True
    if defect == 'padding_mask': masks[8][3].zero_()
    with pytest.raises(ValueError): global_source_exposures(masks, boundary)
