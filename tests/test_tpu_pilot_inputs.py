"""Synthetic metadata contracts; not original-MRI or accelerator proof."""
import numpy as np
import pytest

from rsna_knee.tpu_pilot_inputs import attachment_upper_bound, select_original_pilot_batches, validate_pilot_study_arrays
from rsna_knee.training import _batch_indices


def pools():
    return [f'reference-{i}' for i in range(38)], [f'report-{i}' for i in range(2731)]


def test_original_pool_order_and_first_sixteen_draws_are_preserved():
    reference, report = pools()
    selection = select_original_pilot_batches(reference, report, ['held-out'])
    assert len(selection.study_order) == 64
    for step, batch in enumerate(selection.batch_study_indices):
        expected = [reference[i] for i in _batch_indices(38, 2, step, 20260912)] + [report[i] for i in _batch_indices(2731, 2, step, 20260912 + 34034034)]
        assert [selection.study_order[i] for i in batch] == expected
    assert selection == select_original_pilot_batches(reference, report, ['held-out'])
    reverse = select_original_pilot_batches(reference[::-1], report, ['held-out'])
    assert reverse.reference_pool_sha256 != selection.reference_pool_sha256
    assert reverse.study_order != selection.study_order
    assert selection.to_private_dict()['original_MRI_verified'] is False


@pytest.mark.parametrize('defect', ['subset', 'duplicates', 'overlap', 'held_reference', 'held_report'])
def test_reindexed_or_leaking_source_pools_rejected(defect):
    reference, report = pools();held = []
    if defect == 'subset': reference = reference[:32]
    if defect == 'duplicates': report[-1] = report[0]
    if defect == 'overlap': report[-1] = reference[0]
    if defect == 'held_reference': held = [reference[0]]
    if defect == 'held_report': held = [report[0]]
    with pytest.raises(ValueError): select_original_pilot_batches(reference, report, held)


def arrays():
    return dict(images=np.zeros((16, 8, 1, 224, 224), dtype=np.float32), slice_mask=np.ones((16, 8), dtype=np.bool_),
        series_mask=np.ones(16, dtype=np.bool_), series_descriptors=np.zeros((16, 10), dtype=np.float32),
        targets=np.zeros(12, dtype=np.float32), supervision_mask=np.ones(12, dtype=np.bool_))


def test_original_array_geometry_and_absent_nan_targets_admitted_without_conversion():
    value = arrays();value['targets'][0] = np.nan;value['supervision_mask'][0] = False
    validate_pilot_study_arrays(value)
    assert np.isnan(value['targets'][0])
    assert value['images'].dtype == np.float32


@pytest.mark.parametrize('defect', ['resolution', 'pixel_dtype', 'pixel_nan', 'empty_images', 'empty_labels', 'present_nan', 'mask_dtype', 'descriptor_shape', 'extra'])
def test_changed_or_invalid_original_packing_rejected(defect):
    value = arrays()
    if defect == 'resolution': value['images'] = value['images'][..., :128, :128]
    if defect == 'pixel_dtype': value['images'] = value['images'].astype(np.float16)
    if defect == 'pixel_nan': value['images'][0, 0, 0, 0, 0] = np.nan
    if defect == 'empty_images': value['series_mask'][:] = False
    if defect == 'empty_labels': value['supervision_mask'][:] = False
    if defect == 'present_nan': value['targets'][0] = np.nan
    if defect == 'mask_dtype': value['supervision_mask'] = value['supervision_mask'].astype(np.float32)
    if defect == 'descriptor_shape': value['series_descriptors'] = value['series_descriptors'][:, :4]
    if defect == 'extra': value['extra'] = np.array(0)
    with pytest.raises(ValueError): validate_pilot_study_arrays(value)


def test_attachment_bound_counts_full_raw_geometry_and_both_states():
    assert attachment_upper_bound(initial_state_bytes=135071169, step3_state_bytes=135071169, code_and_metadata_bytes=0) == 1914309506
    with pytest.raises(ValueError, match='2GB'):
        attachment_upper_bound(initial_state_bytes=135071169, step3_state_bytes=135071169, code_and_metadata_bytes=100_000_000)
    with pytest.raises(ValueError): attachment_upper_bound(initial_state_bytes=0, step3_state_bytes=1, code_and_metadata_bytes=0)
