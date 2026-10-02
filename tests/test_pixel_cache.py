"""Exact MRI reconstruction and corruption boundaries for training pixels."""

import json

import numpy as np
import pytest

from rsna_knee.pixel_cache import PixelCacheError, load_pixel_entry, write_pixel_entry
from rsna_knee.prefix_cache import file_digest


def fixture_entry(tmp_path):
    slices = np.zeros((16, 8), dtype=np.bool_)
    slices[0, [0, 3]] = True
    slices[4, 2] = True
    slices[5, 7] = True  # Excluded by the series mask.
    series = np.zeros(16, dtype=np.bool_)
    series[[0, 4]] = True
    images = np.zeros((16, 8, 1, 224, 224), dtype=np.float32)
    images[0, 0] = 0.1; images[0, 3] = 0.7; images[4, 2] = 0.9
    descriptors = np.zeros((16, 10), dtype=np.float32)
    descriptors[0, 0] = 1; descriptors[4, 2] = 1
    metadata = {'study_instance_uid':'study-a','pixel_identity_sha256':'pixels',
                'source_identity_sha256':'source','materialization_plan_sha256':'plan',
                'job_id':'123','source_kind':'ticket35-prepared-series-cache-read-only'}
    path = tmp_path / 'pixels.npz'
    checksum = write_pixel_entry(path, pixels=images[slices & series[:, None]],
        slice_mask=slices, series_mask=series, series_descriptors=descriptors, metadata=metadata)
    expected = dict(expected_sha256=checksum, expected_study_uid='study-a',
                    expected_identity_sha256='pixels', expected_source_identity_sha256='source',
                    expected_plan_sha256='plan')
    return path, expected, images


def rewrite(path, key, mutate):
    with np.load(path, allow_pickle=False) as source:
        values = {k:source[k].copy() for k in source.files}
    values[key] = mutate(values[key])
    np.savez(path, **values)


def test_exact_pixels_mask_indices_and_zero_padding_survive_round_trip(tmp_path):
    path, expected, images = fixture_entry(tmp_path)
    entry = load_pixel_entry(path, **expected)
    assert np.array_equal(entry.valid_flat_indices, [0, 3, 34])
    assert np.array_equal(entry.packed_images(), images)
    assert entry.slice_mask[5, 7] and not entry.series_mask[5]
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_pixel_entry(path, pixels=entry.pixels, slice_mask=entry.slice_mask,
            series_mask=entry.series_mask, series_descriptors=entry.series_descriptors, metadata=entry.metadata)
    assert path.read_bytes() == original


@pytest.mark.parametrize('field,value', [('expected_study_uid','other-study'),
    ('expected_identity_sha256','other-representation'), ('expected_source_identity_sha256','other-source'),
    ('expected_plan_sha256','other-plan')])
def test_wrong_source_or_representation_cannot_enter_training(tmp_path, field, value):
    path, expected, _ = fixture_entry(tmp_path)
    expected[field] = value
    with pytest.raises(PixelCacheError):
        load_pixel_entry(path, **expected)


def test_changed_pixel_bytes_rejected_even_if_file_checksum_is_replaced(tmp_path):
    path, expected, _ = fixture_entry(tmp_path)
    def corrupt(pixels):
        pixels[0, 0, 0, 0] = 0.5
        return pixels
    rewrite(path, 'pixels', corrupt)
    expected['expected_sha256'] = file_digest(path)
    with pytest.raises(PixelCacheError):
        load_pixel_entry(path, **expected)


def test_reordered_slice_mapping_rejected_even_with_matching_array_hash(tmp_path):
    path, expected, _ = fixture_entry(tmp_path)
    with np.load(path, allow_pickle=False) as source:
        values = {k:source[k].copy() for k in source.files}
    values['valid_flat_indices'] = values['valid_flat_indices'][::-1]
    from rsna_knee.prefix_cache import array_digest
    record = json.loads(str(values['metadata'].item()))
    record['array_sha256']['valid_flat_indices'] = array_digest(values['valid_flat_indices'])
    values['metadata'] = np.asarray(json.dumps(record))
    np.savez(path, **values)
    expected['expected_sha256'] = file_digest(path)
    with pytest.raises(PixelCacheError):
        load_pixel_entry(path, **expected)


def test_nonfinite_valid_pixels_fail_before_publishing(tmp_path):
    path, expected, _ = fixture_entry(tmp_path)
    entry = load_pixel_entry(path, **expected)
    entry.pixels[0, 0, 0, 0] = np.nan
    target = tmp_path / 'invalid.npz'
    with pytest.raises(PixelCacheError):
        write_pixel_entry(target, pixels=entry.pixels, slice_mask=entry.slice_mask,
            series_mask=entry.series_mask, series_descriptors=entry.series_descriptors, metadata=entry.metadata)
    assert not target.exists()
