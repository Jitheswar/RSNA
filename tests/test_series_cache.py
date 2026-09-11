from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from rsna_knee.domain import DicomSlice, SeriesRecord
from rsna_knee.preprocessing import PreprocessConfig
from rsna_knee.series_cache import SeriesCache, SeriesCacheError


ORIENTATION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def make_series(
    series_uid: str = "series-1",
    *,
    pixel_offset: int = 0,
    source_names: tuple[str, ...] = (),
) -> SeriesRecord:
    slices = tuple(
        DicomSlice(
            sop_instance_uid=f"slice-{index}",
            image_position=(0.0, 0.0, float(index)),
            image_orientation=ORIENTATION,
            pixel_array=np.arange(12, dtype=np.int16).reshape(3, 4) + pixel_offset + index,
            filename=(source_names[position] if source_names else ""),
        )
        for position, index in enumerate((2, 0, 1))
    )
    return SeriesRecord(
        "study-1",
        series_uid,
        slices,
        source_entry_names=source_names,
    )


def test_cache_persists_complete_slice_provenance_and_reuses_sampling_variants(tmp_path: Path) -> None:
    cache = SeriesCache(tmp_path / "cache")
    series = make_series(source_names=("source-2.dcm", "source-0.dcm", "source-1.dcm"))
    eight = PreprocessConfig(image_size=(8, 8), max_slices=8, max_series=8)
    sixteen = PreprocessConfig(image_size=(8, 8), max_slices=16, max_series=16)

    first = cache.get_or_prepare(series, eight, source_content_hashes=("hash-2", "hash-0", "hash-1"))
    second = cache.get_or_prepare(series, sixteen, source_content_hashes=("hash-2", "hash-0", "hash-1"))

    assert first.hit is False
    assert second.hit is True
    assert second.cache_key == first.cache_key
    np.testing.assert_array_equal(first.prepared.images, second.prepared.images)

    metadata = json.loads((cache.entry_path(first.cache_key) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["state"] == "complete"
    assert metadata["source"]["study_instance_uid"] == "study-1"
    assert metadata["source"]["series_instance_uid"] == "series-1"
    assert [item["sop_instance_uid"] for item in metadata["source"]["slices"]] == [
        "slice-0",
        "slice-1",
        "slice-2",
    ]
    assert [item["content_sha256"] for item in metadata["source"]["slices"]] == [
        "hash-0",
        "hash-1",
        "hash-2",
    ]
    assert metadata["prepared"]["shape"] == [3, 1, 8, 8]
    assert metadata["prepared"]["order"] == ["slice-0", "slice-1", "slice-2"]
    assert metadata["array"]["sha256"]
    assert metadata["metadata_sha256"]


def test_source_or_materialization_change_rebuilds_only_affected_series(tmp_path: Path) -> None:
    cache = SeriesCache(tmp_path / "cache")
    config = PreprocessConfig(image_size=(8, 8))
    first_series = make_series("series-1")
    second_series = make_series("series-2")

    first = cache.get_or_prepare(first_series, config)
    unaffected = cache.get_or_prepare(second_series, config)
    changed_source = cache.get_or_prepare(
        first_series,
        config,
        source_content_hashes=("changed-2", "changed-0", "changed-1"),
    )
    changed_pixels = cache.get_or_prepare(make_series("series-1", pixel_offset=1), config)
    unchanged = cache.get_or_prepare(second_series, config)
    changed_policy = cache.get_or_prepare(first_series, PreprocessConfig(image_size=(4, 8)))

    assert first.hit is False
    assert unaffected.hit is False
    assert changed_source.hit is False
    assert changed_source.cache_key != first.cache_key
    assert changed_pixels.hit is False
    assert changed_pixels.cache_key != first.cache_key
    assert unchanged.hit is True
    assert unchanged.cache_key == unaffected.cache_key
    assert changed_policy.hit is False
    assert changed_policy.cache_key != first.cache_key


def test_checksum_mismatch_is_a_miss_and_is_repaired_atomically(tmp_path: Path) -> None:
    cache = SeriesCache(tmp_path / "cache")
    series = make_series()
    config = PreprocessConfig(image_size=(8, 8))
    first = cache.get_or_prepare(series, config)
    array_path = cache.entry_path(first.cache_key) / "images.npy"
    array_path.write_bytes(array_path.read_bytes() + b"corruption")

    repaired = cache.get_or_prepare(series, config)

    assert repaired.hit is False
    assert repaired.cache_key == first.cache_key
    assert np.isfinite(repaired.prepared.images).all()
    assert not list((cache.entry_path(first.cache_key)).parent.glob(f".{first.cache_key}.partial-*"))


def test_incomplete_entry_and_interrupted_staging_are_never_cache_hits(tmp_path: Path) -> None:
    cache = SeriesCache(tmp_path / "cache")
    series = make_series()
    config = PreprocessConfig(image_size=(8, 8))
    key = cache.cache_key(series, config)
    entry = cache.entry_path(key)
    entry.mkdir(parents=True)
    (entry / "metadata.json").write_text(
        json.dumps({"schema_version": "rsna-knee-series-cache-v1", "state": "writing"}),
        encoding="utf-8",
    )
    partial = entry.parent / f".{key}.partial-interrupted"
    partial.mkdir()
    (partial / "images.npy").write_bytes(b"partial")

    result = cache.get_or_prepare(series, config)

    assert result.hit is False
    assert result.prepared.source_slice_uids == ("slice-0", "slice-1", "slice-2")
    assert partial.exists()
    assert json.loads((entry / "metadata.json").read_text(encoding="utf-8"))["state"] == "complete"

    retained_report = cache.cleanup_disposable(referenced_keys={result.cache_key})
    assert retained_report.removed_partial_paths == ()
    assert partial.exists()
    removed_report = cache.cleanup_disposable(
        referenced_keys={result.cache_key},
        remove_partial=True,
    )
    assert removed_report.removed_partial_paths
    assert not partial.exists()


def test_same_key_writers_are_serialized_and_one_published_entry_wins(tmp_path: Path) -> None:
    cache = SeriesCache(tmp_path / "cache")
    series = make_series()
    config = PreprocessConfig(image_size=(8, 8))

    def prepare() -> bool:
        return cache.get_or_prepare(series, config).hit

    with ThreadPoolExecutor(max_workers=2) as executor:
        hits = list(executor.map(lambda _: prepare(), range(2)))

    assert sorted(hits) == [False, True]
    assert len(cache.list_complete_keys()) == 1


def test_cleanup_removes_only_explicitly_unreferenced_cache_entries(tmp_path: Path) -> None:
    cache = SeriesCache(tmp_path / "cache")
    config = PreprocessConfig(image_size=(8, 8))
    retained = cache.get_or_prepare(make_series("series-retained"), config)
    disposable = cache.get_or_prepare(make_series("series-disposable"), config)

    report = cache.cleanup_disposable(referenced_keys={retained.cache_key})

    assert report.removed_keys == (disposable.cache_key,)
    assert cache.entry_path(retained.cache_key).exists()
    assert not cache.entry_path(disposable.cache_key).exists()


def test_benchmark_reports_declared_concurrency_and_atomic_write_space(tmp_path: Path) -> None:
    cache = SeriesCache(tmp_path / "cache")
    config = PreprocessConfig(image_size=(8, 8))
    studies = tuple(make_series(f"series-{index}") for index in range(3))

    cold = cache.benchmark(studies, config, workers=2)
    warm = cache.benchmark(studies, config, workers=2)

    assert cold.series_count == 3
    assert cold.worker_count == 2
    assert cold.cache_misses == 3
    assert cold.cache_hits == 0
    assert cold.bytes_written > 0
    assert cold.peak_temporary_bytes > 0
    assert warm.cache_hits == 3
    assert warm.cache_misses == 0
    assert warm.bytes_read > 0


def test_cache_requires_a_new_or_explicitly_marked_disposable_root(tmp_path: Path) -> None:
    populated = tmp_path / "raw-data"
    populated.mkdir()
    populated.chmod(0o700)
    (populated / "protected.dcm").write_bytes(b"do not remove")

    with pytest.raises(SeriesCacheError, match="new, empty, or previously marked"):
        SeriesCache(populated)

    assert (populated / "protected.dcm").exists()
