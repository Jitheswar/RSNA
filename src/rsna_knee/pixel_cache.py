"""Exact unfitted MRI pixels in explicit packed-slice order."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .prefix_cache import array_digest, file_digest


SCHEMA_VERSION = "rsna-mri-pixel-entry-v1"
ARRAY_KEYS = ("pixels", "slice_mask", "series_mask", "series_descriptors", "valid_flat_indices")


class PixelCacheError(ValueError):
    """The pixel entry is incomplete, incompatible or changed."""


@dataclass(frozen=True)
class PixelEntry:
    pixels: np.ndarray
    slice_mask: np.ndarray
    series_mask: np.ndarray
    series_descriptors: np.ndarray
    valid_flat_indices: np.ndarray
    metadata: dict[str, object]

    def packed_images(self) -> np.ndarray:
        """Restore the fixed bag exactly; omitted invalid slots contain zero."""
        images = np.zeros((128, 1, 224, 224), dtype=np.float32)
        images[self.valid_flat_indices] = self.pixels
        return images.reshape(16, 8, 1, 224, 224)


@dataclass(frozen=True)
class PackedPixelArray:
    """NumPy-compatible verified pixels without retaining every packed bag.

    The shared trainer may retain all TrainingExample objects. This object
    retains only immutable lookup identity, reading and reconstructing a bag
    when NumPy requests its values. Every access rechecks the cache bytes.
    """

    path: Path
    expected_sha256: str
    expected_study_uid: str
    expected_identity_sha256: str
    expected_source_identity_sha256: str
    expected_plan_sha256: str

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return (16, 8, 1, 224, 224)

    def __array__(self, dtype: np.dtype[Any] | None = None,
                  copy: bool | None = None) -> np.ndarray:
        if copy is False:
            raise ValueError("Exact packed pixels require materialization from their verified file")
        entry = load_pixel_entry(self.path, expected_sha256=self.expected_sha256,
            expected_study_uid=self.expected_study_uid, expected_identity_sha256=self.expected_identity_sha256,
            expected_source_identity_sha256=self.expected_source_identity_sha256,
            expected_plan_sha256=self.expected_plan_sha256)
        images = entry.packed_images()
        if dtype is not None:
            images = images.astype(dtype, copy=False)
        return images.copy() if copy else images


def _validate(arrays: Mapping[str, np.ndarray]) -> None:
    pixels, slices, series, descriptors, indices = (arrays[key] for key in ARRAY_KEYS)
    if (slices.shape != (16, 8) or slices.dtype != np.bool_
            or series.shape != (16,) or series.dtype != np.bool_
            or descriptors.shape != (16, 10) or descriptors.dtype != np.float32
            or not np.isfinite(descriptors).all()):
        raise PixelCacheError("Packed masks or anatomical descriptors are incompatible")
    expected = np.flatnonzero((slices & series[:, None]).reshape(-1))
    if (not len(expected) or pixels.shape != (len(expected), 1, 224, 224)
            or pixels.dtype != np.float32 or not np.isfinite(pixels).all()
            or indices.dtype != np.int64 or not np.array_equal(indices, expected)):
        raise PixelCacheError("Pixel numerics, dimensions or valid-slice order are incompatible")


def write_pixel_entry(path: Path, *, pixels: np.ndarray, slice_mask: np.ndarray,
                      series_mask: np.ndarray, series_descriptors: np.ndarray,
                      metadata: Mapping[str, object]) -> str:
    """Publish a complete entry, preserving any existing bytes."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    if slice_mask.shape != (16, 8) or series_mask.shape != (16,):
        raise PixelCacheError("Packed masks are incompatible")
    arrays = {"pixels": pixels, "slice_mask": slice_mask, "series_mask": series_mask,
              "series_descriptors": series_descriptors,
              "valid_flat_indices": np.flatnonzero((slice_mask & series_mask[:, None]).reshape(-1))}
    _validate(arrays)
    if (any(not isinstance(metadata.get(key), str) or not metadata[key] for key in
            ("study_instance_uid", "pixel_identity_sha256", "source_identity_sha256",
             "materialization_plan_sha256", "job_id"))
            or metadata.get("source_kind") not in
            {"frozen-rsna-archive", "ticket35-prepared-series-cache-read-only"}):
        raise PixelCacheError("Required source and materialization identity is missing")
    record = {**metadata, "schema_version": SCHEMA_VERSION,
              "array_sha256": {key: array_digest(value) for key, value in arrays.items()}}
    descriptor, name = tempfile.mkstemp(prefix=".pixel-", suffix=".npz", dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, allow_pickle=False, pixels=arrays["pixels"],
                     slice_mask=arrays["slice_mask"], series_mask=arrays["series_mask"],
                     series_descriptors=arrays["series_descriptors"],
                     valid_flat_indices=arrays["valid_flat_indices"],
                     metadata=np.asarray(json.dumps(record, sort_keys=True, allow_nan=False)))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        return file_digest(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_pixel_entry(path: Path, *, expected_sha256: str, expected_study_uid: str,
                     expected_identity_sha256: str, expected_source_identity_sha256: str,
                     expected_plan_sha256: str) -> PixelEntry:
    """Verify complete bytes, array identities, source and ordered membership."""
    if not path.is_file() or path.is_symlink() or file_digest(path) != expected_sha256:
        raise PixelCacheError("Pixel file is missing, unsafe or changed")
    try:
        with np.load(path, allow_pickle=False) as source:
            if set(source.files) != {*ARRAY_KEYS, "metadata"}:
                raise PixelCacheError("Pixel payload is incomplete or has extra fields")
            arrays = {key: source[key] for key in ARRAY_KEYS}
            metadata = json.loads(str(source["metadata"].item()))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise PixelCacheError("Pixel payload could not be decoded") from exc
    _validate(arrays)
    expected = {"schema_version": SCHEMA_VERSION, "study_instance_uid": expected_study_uid,
                "pixel_identity_sha256": expected_identity_sha256,
                "source_identity_sha256": expected_source_identity_sha256,
                "materialization_plan_sha256": expected_plan_sha256,
                "array_sha256": {key: array_digest(value) for key, value in arrays.items()}}
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
        raise PixelCacheError("Pixel metadata, source or array identity differs")
    return PixelEntry(pixels=arrays["pixels"], slice_mask=arrays["slice_mask"],
                      series_mask=arrays["series_mask"], series_descriptors=arrays["series_descriptors"],
                      valid_flat_indices=arrays["valid_flat_indices"], metadata=metadata)


__all__ = ["PixelCacheError", "PixelEntry", "PackedPixelArray", "write_pixel_entry", "load_pixel_entry"]
