"""Fixed, deterministic image preparation for the mechanics baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .domain import DicomSlice, SeriesRecord, StudyRecord
from .geometry import GeometryError, GeometryPolicy, order_slices


PREPROCESS_SCHEMA_VERSION = "rsna-knee-preprocess-v1"
SERIES_DESCRIPTOR_COLUMNS = (
    "plane_axial",
    "plane_coronal",
    "plane_sagittal",
    "plane_unknown",
    "fluid_sensitive_true",
    "fluid_sensitive_false",
    "fluid_sensitive_unknown",
    "fat_suppression_true",
    "fat_suppression_false",
    "fat_suppression_unknown",
)


class PreprocessingError(ValueError):
    """Raised when a slice or series cannot be prepared deterministically."""


class StudyPackingError(ValueError):
    """Raised when a study cannot fit the explicitly configured tensor shape."""


@dataclass(frozen=True)
class PackingConfig:
    """Model-input sampling controls applied after series materialization."""

    max_slices: int = 8
    max_series: int = 16

    def __post_init__(self) -> None:
        if self.max_slices <= 0:
            raise ValueError("max_slices must be positive")
        if self.max_series <= 0:
            raise ValueError("max_series must be positive")

    def to_dict(self) -> dict[str, int]:
        return {"max_slices": self.max_slices, "max_series": self.max_series}


@dataclass(frozen=True)
class PreprocessConfig:
    """Versionable, model-independent series materialization policy.

    ``max_slices`` and ``max_series`` remain as compatibility conveniences for
    the original mechanics API.  They are packing controls and are excluded
    from :meth:`cache_config`; changing them cannot invalidate a materialized
    series cache.
    """

    image_size: tuple[int, int] = (128, 128)
    max_slices: int = 8
    max_series: int = 16
    geometry_policy: GeometryPolicy = GeometryPolicy()
    normalization_domain: str = "valid_slice"
    normalization_percentiles: tuple[float, float] = (0.0, 100.0)
    use_voi_lut: bool = False
    interpolation: str = "nearest"
    padding_fill_value: float = 0.0
    output_dtype: str = "float32"
    representation_id: str | None = None

    def __post_init__(self) -> None:
        if len(self.image_size) != 2 or any(size <= 0 for size in self.image_size):
            raise ValueError("image_size must contain two positive dimensions")
        if self.max_slices <= 0:
            raise ValueError("max_slices must be positive")
        if self.max_series <= 0:
            raise ValueError("max_series must be positive")
        if not isinstance(self.geometry_policy, GeometryPolicy):
            raise ValueError("geometry_policy must be a GeometryPolicy")
        if self.normalization_domain != "valid_slice":
            raise ValueError("normalization_domain must be 'valid_slice'")
        if len(self.normalization_percentiles) != 2:
            raise ValueError("normalization_percentiles must contain low and high values")
        low, high = self.normalization_percentiles
        if not np.isfinite((low, high)).all() or not 0.0 <= low <= high <= 100.0:
            raise ValueError("normalization_percentiles must be ordered values within [0, 100]")
        if self.use_voi_lut:
            raise ValueError("VOI/window transforms are deliberately omitted by this policy")
        if self.interpolation != "nearest":
            raise ValueError("only deterministic nearest-neighbor interpolation is supported")
        if not np.isfinite(self.padding_fill_value):
            raise ValueError("padding_fill_value must be finite")
        if self.output_dtype != "float32":
            raise ValueError("the preprocessing output dtype must be float32")
        if self.representation_id is not None and not self.representation_id.strip():
            raise ValueError("representation_id must be non-empty")

    def packing_config(self) -> PackingConfig:
        return PackingConfig(max_slices=self.max_slices, max_series=self.max_series)

    def cache_config(self) -> dict[str, object]:
        """Return only choices that affect materialized series pixels."""

        payload: dict[str, object] = {
            "schema_version": PREPROCESS_SCHEMA_VERSION,
            "image_size": list(self.image_size),
            "geometry_policy": self.geometry_policy.to_dict(),
            "normalization_domain": self.normalization_domain,
            "normalization_percentiles": list(self.normalization_percentiles),
            "use_voi_lut": self.use_voi_lut,
            "interpolation": self.interpolation,
            "padding_fill_value": self.padding_fill_value,
            "output_dtype": self.output_dtype,
        }
        if self.representation_id is not None:
            payload["representation_id"] = self.representation_id
        return payload


@dataclass(frozen=True)
class PreparedSeries:
    """All accepted, preprocessed slices for one source series."""

    study_instance_uid: str
    series_instance_uid: str
    images: np.ndarray
    source_slice_uids: tuple[str, ...]
    warnings: tuple[str, ...] = ()


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
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PackedStudy:
    """Fixed-shape study tensor and masks consumed by the model."""

    study_instance_uid: str
    images: np.ndarray
    slice_mask: np.ndarray
    series_mask: np.ndarray
    sampled_source_indices: tuple[tuple[int, ...], ...]
    series_descriptors: np.ndarray | None = None


def _series_descriptor(series: SeriesRecord) -> np.ndarray:
    """Encode declared anatomy and uncertainty without inventing missing values."""

    descriptor = np.zeros(len(SERIES_DESCRIPTOR_COLUMNS), dtype=np.float32)
    plane = (series.anatomical_plane or "").strip().casefold()
    plane_index = {"axial": 0, "coronal": 1, "sagittal": 2}.get(plane, 3)
    descriptor[plane_index] = 1.0
    descriptor[4 + (0 if series.fluid_sensitive is True else 1 if series.fluid_sensitive is False else 2)] = 1.0
    descriptor[7 + (0 if series.fat_suppression is True else 1 if series.fat_suppression is False else 2)] = 1.0
    return descriptor


def _resize_aspect_preserving(
    image: np.ndarray,
    target_size: tuple[int, int],
    *,
    padding_fill_value: float,
) -> np.ndarray:
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
    canvas = np.full((target_height, target_width), padding_fill_value, dtype=np.float32)
    top = (target_height - resized_height) // 2
    left = (target_width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def _preprocess_pixels(item: DicomSlice, config: PreprocessConfig) -> tuple[np.ndarray, str | None]:
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
        if item.padding_range_limit is not None:
            if not np.isfinite(item.padding_range_limit):
                raise PreprocessingError(
                    f"padding range limit is non-finite for slice {item.sop_instance_uid}"
                )
            lower = min(item.padding_value, item.padding_range_limit)
            upper = max(item.padding_value, item.padding_range_limit)
            valid &= ~((raw >= lower) & (raw <= upper))
    elif item.padding_range_limit is not None:
        raise PreprocessingError(
            f"padding range limit has no PixelPaddingValue for slice {item.sop_instance_uid}"
        )
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
    percentile_low, percentile_high = config.normalization_percentiles
    low = float(np.percentile(values, percentile_low))
    high = float(np.percentile(values, percentile_high))
    warning: str | None = None
    if high > low:
        transformed[valid] = (values - low) / (high - low)
    else:
        transformed[valid] = 0.0
        warning = f"{item.sop_instance_uid}:constant_valid_intensity"
    transformed[~valid] = 0.0
    return (
        _resize_aspect_preserving(
            transformed.astype(np.float32),
            config.image_size,
            padding_fill_value=config.padding_fill_value,
        )[None, ...],
        warning,
    )


def prepare_series(series: SeriesRecord, config: PreprocessConfig = PreprocessConfig()) -> PreparedSeries:
    """Validate, spatially order and preprocess every accepted slice in a series."""

    if not series.slices:
        raise PreprocessingError(f"series {series.series_instance_uid} contains no slices")
    ordered = order_slices(series.slices, config.geometry_policy)
    images = []
    warnings: list[str] = []
    for item in ordered:
        image, warning = _preprocess_pixels(item, config)
        images.append(image)
        if warning is not None:
            warnings.append(warning)
    return PreparedSeries(
        study_instance_uid=series.study_instance_uid,
        series_instance_uid=series.series_instance_uid,
        images=np.stack(images).astype(np.float32, copy=False),
        source_slice_uids=tuple(item.sop_instance_uid for item in ordered),
        warnings=tuple(warnings),
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
    return PreparedStudy(
        study.study_instance_uid,
        tuple(prepared),
        tuple(failures),
        tuple(warning for item in prepared for warning in item.warnings),
    )


def _sample_indices(count: int, max_slices: int) -> tuple[int, ...]:
    if count <= max_slices:
        return tuple(range(count))
    indices = np.rint(np.linspace(0, count - 1, num=max_slices)).astype(np.int64)
    return tuple(int(index) for index in indices)


def pack_study(
    study: StudyRecord,
    prepared_series: Sequence[PreparedSeries],
    config: PreprocessConfig | None = None,
    *,
    packing_config: PackingConfig | None = None,
) -> PackedStudy:
    """Pack valid series into a fixed tensor without hiding missing evidence."""

    materialization_config = config or PreprocessConfig()
    packing = packing_config or materialization_config.packing_config()
    if len(prepared_series) > packing.max_series:
        raise StudyPackingError(
            f"study {study.study_instance_uid} has {len(prepared_series)} valid series, "
            f"exceeding configured max_series={packing.max_series}"
        )
    study_series_ids = {item.series_instance_uid for item in study.series}
    ordered_series = sorted(prepared_series, key=lambda item: item.series_instance_uid)
    if len({item.series_instance_uid for item in ordered_series}) != len(ordered_series):
        raise StudyPackingError("prepared series identifiers must be unique")
    if any(item.study_instance_uid != study.study_instance_uid for item in ordered_series):
        raise StudyPackingError("prepared series and study identifiers do not agree")
    if any(item.series_instance_uid not in study_series_ids for item in ordered_series):
        raise StudyPackingError("prepared series is not present in the source study")

    image_height, image_width = materialization_config.image_size
    images = np.zeros(
        (packing.max_series, packing.max_slices, 1, image_height, image_width),
        dtype=np.float32,
    )
    if materialization_config.padding_fill_value != 0.0:
        images.fill(materialization_config.padding_fill_value)
    slice_mask = np.zeros((packing.max_series, packing.max_slices), dtype=bool)
    series_mask = np.zeros(packing.max_series, dtype=bool)
    series_descriptors = np.zeros(
        (packing.max_series, len(SERIES_DESCRIPTOR_COLUMNS)), dtype=np.float32
    )
    descriptors_by_series = {
        item.series_instance_uid: _series_descriptor(item) for item in study.series
    }
    sampled_source_indices: list[tuple[int, ...]] = []

    for series_index, item in enumerate(ordered_series):
        if item.images.ndim != 4 or item.images.shape[1:] != (1, image_height, image_width):
            raise StudyPackingError(f"prepared series {item.series_instance_uid} has an incompatible image shape")
        indices = _sample_indices(item.images.shape[0], packing.max_slices)
        if indices:
            images[series_index, : len(indices)] = item.images[list(indices)]
            slice_mask[series_index, : len(indices)] = True
            series_mask[series_index] = True
        sampled_source_indices.append(indices)
        series_descriptors[series_index] = descriptors_by_series[item.series_instance_uid]

    while len(sampled_source_indices) < packing.max_series:
        sampled_source_indices.append(())
    return PackedStudy(
        study_instance_uid=study.study_instance_uid,
        images=images,
        slice_mask=slice_mask,
        series_mask=series_mask,
        sampled_source_indices=tuple(sampled_source_indices),
        series_descriptors=series_descriptors,
    )
