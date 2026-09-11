"""Small adapter from pydicom datasets to the inference core record."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError

from .domain import DicomSlice


class DicomDecodeError(ValueError):
    """Raised when a DICOM dataset cannot become one supported slice record."""


def _required(dataset: Dataset, name: str) -> object:
    value = getattr(dataset, name, None)
    if value is None:
        raise DicomDecodeError(f"DICOM attribute {name} is missing")
    return value


def _float_tuple(value: object, length: int, name: str) -> tuple[float, ...]:
    try:
        values_array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DicomDecodeError(f"DICOM attribute {name} is invalid") from exc
    if values_array.shape != (length,) or not np.isfinite(values_array).all():
        raise DicomDecodeError(f"DICOM attribute {name} is invalid")
    return tuple(float(item) for item in values_array)


def decode_dataset(dataset: Dataset, *, filename: str = "") -> DicomSlice:
    """Decode one pydicom dataset into the stable, training-free core record."""

    study_instance_uid = str(_required(dataset, "StudyInstanceUID"))
    series_instance_uid = str(_required(dataset, "SeriesInstanceUID"))
    sop_instance_uid = str(_required(dataset, "SOPInstanceUID"))
    image_position = _float_tuple(_required(dataset, "ImagePositionPatient"), 3, "ImagePositionPatient")
    image_orientation = _float_tuple(
        _required(dataset, "ImageOrientationPatient"),
        6,
        "ImageOrientationPatient",
    )
    photometric = str(_required(dataset, "PhotometricInterpretation")).upper()
    try:
        pixel_array = np.asarray(dataset.pixel_array)
    except (
        AttributeError,
        IndexError,
        InvalidDicomError,
        KeyError,
        NotImplementedError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise DicomDecodeError(f"pixel decoding failed for slice {sop_instance_uid}") from exc
    if pixel_array.ndim != 2 or 0 in pixel_array.shape:
        raise DicomDecodeError(f"slice {sop_instance_uid} is not a non-empty two-dimensional image")

    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    padding = getattr(dataset, "PixelPaddingValue", None)
    padding_value = float(padding) if padding is not None else None
    return DicomSlice(
        sop_instance_uid=sop_instance_uid,
        image_position=image_position,  # type: ignore[arg-type]
        image_orientation=image_orientation,  # type: ignore[arg-type]
        pixel_array=pixel_array,
        study_instance_uid=study_instance_uid,
        series_instance_uid=series_instance_uid,
        filename=filename or str(getattr(dataset, "filename", "") or ""),
        instance_number=(int(dataset.InstanceNumber) if hasattr(dataset, "InstanceNumber") else None),
        rescale_slope=slope,
        rescale_intercept=intercept,
        photometric_interpretation=photometric,
        padding_value=padding_value,
    )


def read_dicom_slice(path: str | Path) -> DicomSlice:
    """Read one file through pydicom; broader corpus support is a later gate."""

    source = Path(path)
    try:
        dataset = pydicom.dcmread(source, force=False)
    except (EOFError, InvalidDicomError, OSError, TypeError, ValueError) as exc:
        raise DicomDecodeError(f"could not read DICOM file {source}") from exc
    return decode_dataset(dataset, filename=str(source))
