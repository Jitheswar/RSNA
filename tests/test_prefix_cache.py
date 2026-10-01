"""Protect immutable prefix inputs at the extraction-to-adaptation boundary."""

from pathlib import Path

import numpy as np
import pytest

from rsna_knee.prefix_cache import PrefixCacheError, file_digest, load_prefix_entry, write_prefix_entry


def inputs():
    slices = np.zeros((16, 8), dtype=bool)
    slices[0, (1, 4)] = True
    slices[2, 7] = True
    series = np.zeros(16, dtype=bool)
    series[[0, 2]] = True
    return {"prefix": np.arange(3 * 256 * 14 * 14, dtype=np.float32).reshape(3, 256, 14, 14),
            "slice_mask": slices, "series_mask": series,
            "series_descriptors": np.zeros((16, 10), dtype=np.float32),
            "metadata": {"study_instance_uid": "fixture-a", "feature_identity_sha256": "a" * 64,
                         "source_identity_sha256": "b" * 64, "extraction_plan_sha256": "c" * 64,
                         "job_id": "123", "source_kind": "frozen-rsna-archive"}}


def load(path: Path, **overrides):
    args = {"expected_sha256": file_digest(path), "expected_study_uid": "fixture-a",
            "expected_identity_sha256": "a" * 64, "expected_source_identity_sha256": "b" * 64,
            "expected_plan_sha256": "c" * 64}
    return load_prefix_entry(path, **{**args, **overrides})


def test_entry_preserves_noncontiguous_valid_slice_mapping_and_arrays(tmp_path):
    path = tmp_path / "entry.npz"
    original = inputs()
    checksum = write_prefix_entry(path, **original)
    actual = load(path, expected_sha256=checksum)
    np.testing.assert_array_equal(actual.valid_flat_indices, [1, 4, 23])
    np.testing.assert_array_equal(actual.prefix, original["prefix"])
    np.testing.assert_array_equal(actual.slice_mask, original["slice_mask"])
    np.testing.assert_array_equal(actual.series_descriptors, original["series_descriptors"])


@pytest.mark.parametrize("key", ["expected_study_uid", "expected_identity_sha256",
                                 "expected_source_identity_sha256", "expected_plan_sha256"])
def test_entry_rejects_wrong_cohort_or_provenance(tmp_path, key):
    path = tmp_path / "entry.npz"
    write_prefix_entry(path, **inputs())
    with pytest.raises(PrefixCacheError):
        load(path, **{key: "different"})


def test_entry_rejects_changed_bytes_and_preserves_existing_output(tmp_path):
    path = tmp_path / "entry.npz"
    checksum = write_prefix_entry(path, **inputs())
    with pytest.raises(FileExistsError):
        write_prefix_entry(path, **inputs())
    assert file_digest(path) == checksum
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(PrefixCacheError):
        load(path, expected_sha256=checksum)


@pytest.mark.parametrize("corruption", ["nan", "wrong_precision", "wrong_count"])
def test_invalid_activations_are_not_published(tmp_path, corruption):
    original = inputs()
    if corruption == "nan":
        original["prefix"][0, 0, 0, 0] = np.nan
    elif corruption == "wrong_precision":
        original["prefix"] = original["prefix"].astype(np.float64)
    else:
        original["prefix"] = original["prefix"][:2]
    path = tmp_path / "entry.npz"
    with pytest.raises(PrefixCacheError):
        write_prefix_entry(path, **original)
    assert not path.exists()


def test_recomputed_file_hash_cannot_hide_reordered_slice_indices(tmp_path):
    path = tmp_path / "entry.npz"
    write_prefix_entry(path, **inputs())
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    arrays["valid_flat_indices"] = arrays["valid_flat_indices"][::-1]
    np.savez(path, **arrays)
    with pytest.raises(PrefixCacheError):
        load(path)


def test_recomputed_file_hash_cannot_hide_changed_activation_array(tmp_path):
    path = tmp_path / "entry.npz"
    write_prefix_entry(path, **inputs())
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    arrays["prefix"][0, 0, 0, 0] += 1
    np.savez(path, **arrays)
    with pytest.raises(PrefixCacheError):
        load(path)
