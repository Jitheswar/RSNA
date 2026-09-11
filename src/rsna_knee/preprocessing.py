"""Fixed, deterministic image preparation for the mechanics baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .domain import DicomSlice, SeriesRecord, StudyRecord
from .geometry import GeometryError, order_slices


class PreprocessingError(ValueError):
    """Raised when a slice or series cannot be prepared deterministically."""


class StudyPackingError(ValueError):
    """Raised when a study cannot fit the explicitly configured tensor shape."""


@dataclass(frozen=True)
class PreprocessConfig:
    """Versionable fixed preprocessing and baseline tensor dimensions."""

    image_size: tuple[int, int] = (128, 128)
    max_slices: int = 8
    max_series: int = 16

    def __post_init__(self) -> None:
        if len(self.image_size) != 2 or any(size <= 0 for size in self.image_size):
            raise ValueError("image_size must contain two positive dimensions")
        if self.max_slices <= 0:
            raise ValueError("max_slices must be positive")
        if self.max_series <= 0:
            raise ValueError("max_series must be positive")


@dataclass(frozen=True)
class PreparedSeries:
    """All accepted, preprocessed slices for one source series."""

    study_instance_uid: str
    series_instance_uid: str
    images: np.ndarray
    source_slice_uids: tuple[str, ...]


@dataclass(frozen=True)
class SeriesFailure:
    """Private diagnostic describing why one series was not admitted."""

    series_instance_uid: str
    reason: str


@dataclass(frozen=True)
class PreparedStudy:
    """Prepared valid series plus explicit failures for the same study."""

    study_instance_uid: str
    series: tuple[PreparedSeries, ...]
    failures: tuple[SeriesFailure, ...]


@dataclass(frozen=True)
class PackedStudy:
    """Fixed-shape study tensor and masks consumed by the model."""

    study_instance_uid: str
    images: np.ndarray
    slice_mask: np.ndarray
    series_mask: np.ndarray
    sampled_source_indices: tuple[tuple[int, ...], ...]


def _resize_aspect_preserving(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_height, target_width = target_size
    height, width = image.shape
    scale = min(target_height / height, target_width / width)
    resized_height = max(1, min(target_height, int(round(height * scale))))
    resized_width = max(1, min(target_width, int(round(width * scale))))

    row_indices = np.minimum(
        height - 1,
        np.floor(np.arange(resized_height) * height / resized_height).astype(np.int64),
    )
    column_indices = np.minimum(
        width - 1,
        np.floor(np.arange(resized_width) * width / resized_width).astype(np.int64),
    )
    resized = image[row_indices[:, None], column_indices[None, :]]
    canvas = np.zeros((target_height, target_width), dtype=np.float32)
    top = (target_height - resized_height) // 2
    left = (target_width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def _preprocess_pixels(item: DicomSlice, config: PreprocessConfig) -> np.ndarray:
    try:
        raw = np.asarray(item.pixel_array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PreprocessingError(f"pixel array is not numeric for slice {item.sop_instance_uid}") from exc
    if raw.ndim != 2 or 0 in raw.shape:
        raise PreprocessingError(f"pixel array must be a non-empty 2D image for slice {item.sop_instance_uid}")

    valid = np.isfinite(raw)
    if item.padding_value is not None:
        if not np.isfinite(item.padding_value):
            raise PreprocessingError(f"padding value is non-finite for slice {item.sop_instance_uid}")
        valid &= raw != item.padding_value
    if not valid.any():
        raise PreprocessingError(f"slice {item.sop_instance_uid} contains no valid pixels")

    if not np.isfinite(item.rescale_slope) or not np.isfinite(item.rescale_intercept):
        raise PreprocessingError(f"modality transform is non-finite for slice {item.sop_instance_uid}")
    if item.rescale_slope == 0:
        raise PreprocessingError(f"modality slope is zero for slice {item.sop_instance_uid}")

    transformed = np.zeros_like(raw, dtype=np.float64)
    transformed[valid] = raw[valid] * item.rescale_slope + item.rescale_intercept
    values = transformed[valid]
    if not np.isfinite(values).all():
        raise PreprocessingError(f"modality transform produced non-finite pixels for slice {item.sop_instance_uid}")

    photometric = item.photometric_interpretation.upper()
    if photometric == "MONOCHROME1":
        low = float(values.min())
        high = float(values.max())
        transformed[valid] = low + high - values
    elif photometric != "MONOCHROME2":
        raise PreprocessingError(f"unsupported photometric interpretation {photometric!r}")

    values = transformed[valid]
    low = float(values.min())
    high = float(values.max())
    if high > low:
        transformed[valid] = (values - low) / (high - low)
    else:
        transformed[valid] = 0.0
    transformed[~valid] = 0.0
    return _resize_aspect_preserving(transformed.astype(np.float32), config.image_size)[None, ...]


def prepare_series(series: SeriesRecord, config: PreprocessConfig = PreprocessConfig()) -> PreparedSeries:
    """Validate, spatially order and preprocess every accepted slice in a series."""

    if not series.slices:
        raise PreprocessingError(f"series {series.series_instance_uid} contains no slices")
    ordered = order_slices(series.slices)
    images = []
    for item in ordered:
        images.append(_preprocess_pixels(item, config))
    return PreparedSeries(
        study_instance_uid=series.study_instance_uid,
        series_instance_uid=series.series_instance_uid,
        images=np.stack(images).astype(np.float32, copy=False),
        source_slice_uids=tuple(item.sop_instance_uid for item in ordered),
    )


def prepare_study(study: StudyRecord, config: PreprocessConfig = PreprocessConfig()) -> PreparedStudy:
    """Prepare each series independently, retaining failures for diagnostics."""

    prepared: list[PreparedSeries] = []
    failures: list[SeriesFailure] = []
    for series in sorted(study.series, key=lambda item: item.series_instance_uid):
        try:
            prepared.append(prepare_series(series, config))
        except (GeometryError, PreprocessingError) as exc:
            failures.append(SeriesFailure(series.series_instance_uid, str(exc)))
    return PreparedStudy(study.study_instance_uid, tuple(prepared), tuple(failures))


def _sample_indices(count: int, max_slices: int) -> tuple[int, ...]:
    if count <= max_slices:
        return tuple(range(count))
    indices = np.rint(np.linspace(0, count - 1, num=max_slices)).astype(np.int64)
    return tuple(int(index) for index in indices)


def pack_study(
    study: StudyRecord,
    prepared_series: Sequence[PreparedSeries],
    config: PreprocessConfig = PreprocessConfig(),
) -> PackedStudy:
    """Pack valid series into a fixed tensor without hiding missing evidence."""

    if len(prepared_series) > config.max_series:
        raise StudyPackingError(
            f"study {study.study_instance_uid} has {len(prepared_series)} valid series, "
            f"exceeding configured max_series={config.max_series}"
        )
    study_series_ids = {item.series_instance_uid for item in study.series}
    ordered_series = sorted(prepared_series, key=lambda item: item.series_instance_uid)
    if len({item.series_instance_uid for item in ordered_series}) != len(ordered_series):
        raise StudyPackingError("prepared series identifiers must be unique")
    if any(item.study_instance_uid != study.study_instance_uid for item in ordered_series):
        raise StudyPackingError("prepared series and study identifiers do not agree")
    if any(item.series_instance_uid not in study_series_ids for item in ordered_series):
        raise StudyPackingError("prepared series is not present in the source study")

    image_height, image_width = config.image_size
    images = np.zeros(
        (config.max_series, config.max_slices, 1, image_height, image_width),
        dtype=np.float32,
    )
    slice_mask = np.zeros((config.max_series, config.max_slices), dtype=bool)
    series_mask = np.zeros(config.max_series, dtype=bool)
    sampled_source_indices: list[tuple[int, ...]] = []

    for series_index, item in enumerate(ordered_series):
        if item.images.ndim != 4 or item.images.shape[1:] != (1, image_height, image_width):
            raise StudyPackingError(f"prepared series {item.series_instance_uid} has an incompatible image shape")
        indices = _sample_indices(item.images.shape[0], config.max_slices)
        if indices:
            images[series_index, : len(indices)] = item.images[list(indices)]
            slice_mask[series_index, : len(indices)] = True
            series_mask[series_index] = True
        sampled_source_indices.append(indices)

    while len(sampled_source_indices) < config.max_series:
        sampled_source_indices.append(())
    return PackedStudy(
        study_instance_uid=study.study_instance_uid,
        images=images,
        slice_mask=slice_mask,
        series_mask=series_mask,
        sampled_source_indices=tuple(sampled_source_indices),
    )
