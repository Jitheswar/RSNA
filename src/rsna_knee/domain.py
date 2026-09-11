"""Small, dependency-light records shared by the inference path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DicomSlice:
    """Decoded DICOM slice metadata and pixels.

    The inference core consumes this record rather than importing a DICOM
    reader. A future decoder adapter can map its library-specific dataset to
    this stable interface, while synthetic tests can construct it directly.
    """

    sop_instance_uid: str
    image_position: Optional[tuple[float, float, float]]
    image_orientation: Optional[tuple[float, float, float, float, float, float]]
    pixel_array: np.ndarray
    study_instance_uid: Optional[str] = None
    series_instance_uid: Optional[str] = None
    filename: str = ""
    instance_number: Optional[int] = None
    rescale_slope: float = 1.0
    rescale_intercept: float = 0.0
    photometric_interpretation: str = "MONOCHROME2"
    padding_value: Optional[float] = None
    padding_range_limit: Optional[float] = None
    transfer_syntax_uid: Optional[str] = None
    modality: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    number_of_frames: int = 1
    samples_per_pixel: int = 1
    bits_allocated: Optional[int] = None
    bits_stored: Optional[int] = None
    high_bit: Optional[int] = None
    pixel_representation: int = 0
    image_type_values: tuple[str, ...] = ()
    series_description: str = ""
    source_content_sha256: Optional[str] = None


@dataclass(frozen=True)
class SeriesRecord:
    """A source series with its parent study identity preserved."""

    study_instance_uid: str
    series_instance_uid: str
    slices: tuple[DicomSlice, ...]
    description: str = ""
    anatomical_plane: Optional[str] = None
    fluid_sensitive: Optional[bool] = None
    fat_suppression: Optional[bool] = None
    diagnostic_status: str = "unknown"
    source_entry_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for item in self.slices:
            if item.study_instance_uid not in (None, self.study_instance_uid):
                raise ValueError(f"slice {item.sop_instance_uid} belongs to another study")
            if item.series_instance_uid not in (None, self.series_instance_uid):
                raise ValueError(f"slice {item.sop_instance_uid} belongs to another series")


@dataclass(frozen=True)
class StudyRecord:
    """A study-level inference input containing distinct source series."""

    study_instance_uid: str
    series: tuple[SeriesRecord, ...]

    def __post_init__(self) -> None:
        for item in self.series:
            if item.study_instance_uid != self.study_instance_uid:
                raise ValueError(f"series {item.series_instance_uid} belongs to another study")
