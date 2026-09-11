"""Pinned DICOM decoding adapters for the shared MRI inference core.

The decoder deliberately stops at one validated, single-frame grayscale slice.
Archive-backed callers can use the stream and bytes entry points without first
extracting private competition files to a second location.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.uid import ExplicitVRLittleEndian

from .domain import DicomSlice


DICOM_POLICY_VERSION = "rsna-knee-dicom-v1"
DEFAULT_SUPPORTED_TRANSFER_SYNTAXES = (str(ExplicitVRLittleEndian),)


class DicomDecodeError(ValueError):
    """Base class for a dataset that cannot become one supported slice."""


class UnsupportedTransferSyntaxError(DicomDecodeError):
    """Raised when the dataset uses a syntax outside the frozen decoder set."""


class DicomValidationError(DicomDecodeError):
    """Raised when declared DICOM semantics are inconsistent or unsupported."""


class DamagedDicomError(DicomDecodeError):
    """Raised when a file or pixel payload is damaged or truncated."""


class DecoderInfrastructureError(DicomDecodeError):
    """Raised when a known syntax needs an unavailable or broken decoder backend."""


@dataclass(frozen=True)
class DecoderPolicy:
    """Versioned decoder and pixel-semantics admission policy.

    Only explicit VR little endian is admitted by the initial CPU environment.
    Adding a compressed syntax requires a new policy, a pinned pixel handler,
    and representative verification; it is never an implicit pydicom fallback.
    """

    policy_version: str = DICOM_POLICY_VERSION
    supported_transfer_syntaxes: tuple[str, ...] = DEFAULT_SUPPORTED_TRANSFER_SYNTAXES
    attempt_order: tuple[str, ...] = ("pydicom",)
    require_file_meta: bool = True
    require_modality: bool = False
    allow_multiframe: bool = False

    def __post_init__(self) -> None:
        if self.policy_version != DICOM_POLICY_VERSION:
            raise ValueError(f"unsupported decoder policy version {self.policy_version!r}")
        syntaxes = tuple(str(value).strip() for value in self.supported_transfer_syntaxes)
        if not syntaxes or any(not value for value in syntaxes):
            raise ValueError("supported_transfer_syntaxes must contain non-empty UIDs")
        if len(set(syntaxes)) != len(syntaxes):
            raise ValueError("supported_transfer_syntaxes must be unique")
        attempts = tuple(value.strip().lower() for value in self.attempt_order)
        if not attempts or any(not value for value in attempts):
            raise ValueError("attempt_order must contain non-empty backend names")
        if len(set(attempts)) != len(attempts):
            raise ValueError("attempt_order must not contain duplicate backends")
        if attempts != ("pydicom",):
            raise ValueError("the pinned decoder currently supports only the pydicom backend")
        object.__setattr__(self, "supported_transfer_syntaxes", syntaxes)
        object.__setattr__(self, "attempt_order", attempts)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "supported_transfer_syntaxes": list(self.supported_transfer_syntaxes),
            "attempt_order": list(self.attempt_order),
            "require_file_meta": self.require_file_meta,
            "require_modality": self.require_modality,
            "allow_multiframe": self.allow_multiframe,
        }

    @classmethod
    def from_dict(cls, value: object) -> "DecoderPolicy":
        if not isinstance(value, dict):
            raise ValueError("decoder policy must be an object")
        syntaxes = value.get("supported_transfer_syntaxes")
        attempts = value.get("attempt_order")
        if not isinstance(syntaxes, (list, tuple)) or not isinstance(attempts, (list, tuple)):
            raise ValueError("decoder policy syntax and attempt fields must be arrays")
        return cls(
            policy_version=str(value.get("policy_version", "")),
            supported_transfer_syntaxes=tuple(str(item) for item in syntaxes),
            attempt_order=tuple(str(item) for item in attempts),
            require_file_meta=bool(value.get("require_file_meta", True)),
            require_modality=bool(value.get("require_modality", False)),
            allow_multiframe=bool(value.get("allow_multiframe", False)),
        )


def _required(dataset: Dataset, name: str) -> object:
    value = getattr(dataset, name, None)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise DicomValidationError(f"DICOM attribute {name} is missing")
    return value


def _required_text(dataset: Dataset, name: str) -> str:
    return str(_required(dataset, name)).strip()


def _float_value(value: object, name: str, *, allow_missing: bool = False, default: float = 0.0) -> float:
    if value is None:
        if allow_missing:
            return default
        raise DicomValidationError(f"DICOM attribute {name} is missing")
    try:
        parsed = float(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise DicomValidationError(f"DICOM attribute {name} is invalid") from exc
    if not np.isfinite(parsed):
        raise DicomValidationError(f"DICOM attribute {name} is invalid")
    return parsed


def _float_tuple(value: object, length: int, name: str) -> tuple[float, ...]:
    try:
        values_array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DicomValidationError(f"DICOM attribute {name} is invalid") from exc
    if values_array.shape != (length,) or not np.isfinite(values_array).all():
        raise DicomValidationError(f"DICOM attribute {name} is invalid")
    return tuple(float(item) for item in values_array)


def _positive_int(dataset: Dataset, name: str, *, default: int | None = None) -> int:
    value = getattr(dataset, name, default)
    if value is None:
        raise DicomValidationError(f"DICOM attribute {name} is missing")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DicomValidationError(f"DICOM attribute {name} is invalid") from exc
    if parsed <= 0:
        raise DicomValidationError(f"DICOM attribute {name} must be positive")
    return parsed


def _nonnegative_int(dataset: Dataset, name: str, *, default: int = 0) -> int:
    value = getattr(dataset, name, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DicomValidationError(f"DICOM attribute {name} is invalid") from exc
    if parsed < 0:
        raise DicomValidationError(f"DICOM attribute {name} must be non-negative")
    return parsed


def _image_type_values(dataset: Dataset) -> tuple[str, ...]:
    value = getattr(dataset, "ImageType", None)
    if value is None:
        return ()
    if isinstance(value, str):
        values: Iterable[object] = (value,)
    else:
        values = value
    return tuple(normalized for item in values if (normalized := str(item).strip().upper()))


def _transfer_syntax(dataset: Dataset, policy: DecoderPolicy) -> str:
    file_meta = getattr(dataset, "file_meta", None)
    transfer_syntax_value = getattr(file_meta, "TransferSyntaxUID", None)
    if transfer_syntax_value is None:
        if policy.require_file_meta:
            raise UnsupportedTransferSyntaxError("DICOM file has no declared transfer syntax")
        return "unknown"
    transfer_syntax = str(transfer_syntax_value).strip()
    if transfer_syntax not in policy.supported_transfer_syntaxes:
        raise UnsupportedTransferSyntaxError(
            f"transfer syntax {transfer_syntax!r} is not admitted by decoder policy "
            f"{policy.policy_version}"
        )
    return transfer_syntax


def _read_pixel_array(dataset: Dataset, sop_instance_uid: str) -> np.ndarray:
    try:
        pixel_array = np.asarray(dataset.pixel_array)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise DamagedDicomError(f"pixel decoding failed for slice {sop_instance_uid}") from exc
    except (ImportError, ModuleNotFoundError, NotImplementedError, RuntimeError) as exc:
        raise DecoderInfrastructureError(
            f"decoder backend could not materialize pixels for slice {sop_instance_uid}"
        ) from exc
    if pixel_array.ndim == 0 or not np.isfinite(pixel_array).all():
        raise DamagedDicomError(f"decoded pixels are invalid for slice {sop_instance_uid}")
    return pixel_array


def decode_dataset(
    dataset: Dataset,
    *,
    filename: str = "",
    policy: DecoderPolicy = DecoderPolicy(),
) -> DicomSlice:
    """Validate and decode one DICOM dataset into the stable core record."""

    transfer_syntax_uid = _transfer_syntax(dataset, policy)
    study_instance_uid = _required_text(dataset, "StudyInstanceUID")
    series_instance_uid = _required_text(dataset, "SeriesInstanceUID")
    sop_instance_uid = _required_text(dataset, "SOPInstanceUID")
    modality_value = getattr(dataset, "Modality", None)
    modality = str(modality_value).strip().upper() if modality_value is not None else None
    if modality is not None and modality != "MR":
        raise DicomValidationError(f"Modality {modality!r} is not supported; expected 'MR'")
    if policy.require_modality and modality is None:
        raise DicomValidationError("DICOM attribute Modality is missing")

    image_position = _float_tuple(_required(dataset, "ImagePositionPatient"), 3, "ImagePositionPatient")
    image_orientation = _float_tuple(
        _required(dataset, "ImageOrientationPatient"),
        6,
        "ImageOrientationPatient",
    )
    photometric = _required_text(dataset, "PhotometricInterpretation").upper()
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise DicomValidationError(f"PhotometricInterpretation {photometric!r} is not supported")

    rows = _positive_int(dataset, "Rows")
    columns = _positive_int(dataset, "Columns")
    samples_per_pixel = _positive_int(dataset, "SamplesPerPixel", default=1)
    if samples_per_pixel != 1:
        raise DicomValidationError("SamplesPerPixel must be 1 for the grayscale inference path")
    frame_count = _positive_int(dataset, "NumberOfFrames", default=1)
    if frame_count != 1 and not policy.allow_multiframe:
        raise DicomValidationError(
            f"NumberOfFrames={frame_count} is unsupported by the single-frame decoder policy"
        )

    bits_allocated = _positive_int(dataset, "BitsAllocated")
    bits_stored = _positive_int(dataset, "BitsStored")
    high_bit = _nonnegative_int(dataset, "HighBit")
    pixel_representation = _nonnegative_int(dataset, "PixelRepresentation")
    if bits_allocated not in {8, 16, 32}:
        raise DicomValidationError(f"BitsAllocated={bits_allocated} is unsupported")
    if bits_stored > bits_allocated or high_bit != bits_stored - 1:
        raise DicomValidationError("BitsStored and HighBit are inconsistent with BitsAllocated")
    if pixel_representation not in {0, 1}:
        raise DicomValidationError("PixelRepresentation must be 0 (unsigned) or 1 (signed)")

    pixel_array = _read_pixel_array(dataset, sop_instance_uid)
    if pixel_array.shape != (rows, columns):
        raise DicomValidationError(
            f"decoded pixel shape {pixel_array.shape} does not match declared {(rows, columns)}"
        )
    if not np.issubdtype(pixel_array.dtype, np.integer):
        raise DicomValidationError("decoded grayscale pixels must use an integer dtype")
    if pixel_representation == 1 and not np.issubdtype(pixel_array.dtype, np.signedinteger):
        raise DicomValidationError("signed PixelRepresentation decoded to a non-signed dtype")
    if pixel_representation == 0 and not np.issubdtype(pixel_array.dtype, np.unsignedinteger):
        raise DicomValidationError("unsigned PixelRepresentation decoded to a signed dtype")

    slope = _float_value(
        getattr(dataset, "RescaleSlope", None),
        "RescaleSlope",
        allow_missing=True,
        default=1.0,
    )
    intercept = _float_value(
        getattr(dataset, "RescaleIntercept", None),
        "RescaleIntercept",
        allow_missing=True,
        default=0.0,
    )
    if slope == 0.0:
        raise DicomValidationError("RescaleSlope must not be zero")
    padding_raw = getattr(dataset, "PixelPaddingValue", None)
    padding_range_raw = getattr(dataset, "PixelPaddingRangeLimit", None)
    padding = _float_value(padding_raw, "PixelPaddingValue") if padding_raw is not None else None
    padding_range = (
        _float_value(padding_range_raw, "PixelPaddingRangeLimit")
        if padding_range_raw is not None
        else None
    )

    instance_number_value = getattr(dataset, "InstanceNumber", None)
    instance_number: int | None
    if instance_number_value is None:
        instance_number = None
    else:
        try:
            instance_number = int(instance_number_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DicomValidationError("InstanceNumber is invalid") from exc

    return DicomSlice(
        sop_instance_uid=sop_instance_uid,
        image_position=image_position,  # type: ignore[arg-type]
        image_orientation=image_orientation,  # type: ignore[arg-type]
        pixel_array=pixel_array,
        study_instance_uid=study_instance_uid,
        series_instance_uid=series_instance_uid,
        filename=filename or str(getattr(dataset, "filename", "") or ""),
        instance_number=instance_number,
        rescale_slope=slope,
        rescale_intercept=intercept,
        photometric_interpretation=photometric,
        padding_value=padding,
        padding_range_limit=padding_range,
        transfer_syntax_uid=transfer_syntax_uid,
        modality=modality,
        rows=rows,
        columns=columns,
        number_of_frames=frame_count,
        samples_per_pixel=samples_per_pixel,
        bits_allocated=bits_allocated,
        bits_stored=bits_stored,
        high_bit=high_bit,
        pixel_representation=pixel_representation,
        image_type_values=_image_type_values(dataset),
        series_description=str(getattr(dataset, "SeriesDescription", "") or "").strip(),
    )


def decode_stream(
    stream: BinaryIO,
    *,
    filename: str = "",
    policy: DecoderPolicy = DecoderPolicy(),
) -> DicomSlice:
    """Read one DICOM object from an open binary stream and validate it."""

    try:
        dataset = pydicom.dcmread(stream, force=False)
    except (EOFError, InvalidDicomError, OSError, TypeError, ValueError, IndexError, KeyError) as exc:
        raise DamagedDicomError(f"could not read DICOM file {filename or '<stream>'}") from exc
    except (ImportError, ModuleNotFoundError, NotImplementedError, RuntimeError) as exc:
        raise DecoderInfrastructureError(
            f"decoder backend could not read DICOM file {filename or '<stream>'}"
        ) from exc
    return decode_dataset(dataset, filename=filename, policy=policy)


def decode_bytes(
    payload: bytes,
    *,
    filename: str = "",
    policy: DecoderPolicy = DecoderPolicy(),
) -> DicomSlice:
    """Decode one in-memory DICOM object through the same stream adapter."""

    if not isinstance(payload, bytes) or not payload:
        raise DamagedDicomError("DICOM payload must be non-empty bytes")
    return decode_stream(BytesIO(payload), filename=filename, policy=policy)


def read_dicom_slice(
    path: str | Path,
    *,
    policy: DecoderPolicy = DecoderPolicy(),
) -> DicomSlice:
    """Read and validate one filesystem DICOM file."""

    source = Path(path)
    try:
        with source.open("rb") as stream:
            return decode_stream(stream, filename=str(source), policy=policy)
    except FileNotFoundError as exc:
        raise DamagedDicomError(f"could not read DICOM file {source}") from exc
    except (OSError, TypeError) as exc:
        raise DamagedDicomError(f"could not read DICOM file {source}") from exc


__all__ = [
    "DICOM_POLICY_VERSION",
    "DEFAULT_SUPPORTED_TRANSFER_SYNTAXES",
    "DecoderPolicy",
    "DicomDecodeError",
    "UnsupportedTransferSyntaxError",
    "DicomValidationError",
    "DamagedDicomError",
    "DecoderInfrastructureError",
    "decode_dataset",
    "decode_stream",
    "decode_bytes",
    "read_dicom_slice",
]
