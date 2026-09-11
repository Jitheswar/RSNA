"""Deterministic in-memory pydicom fixtures for public mechanics verification."""

from __future__ import annotations

import hashlib

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, UID

from .dicom import decode_dataset
from .domain import DicomSlice, SeriesRecord, StudyRecord


ORIENTATION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _stable_uid(*parts: object) -> str:
    digest = hashlib.sha256("/".join(str(part) for part in parts).encode("utf-8")).digest()
    return "2.25." + str(int.from_bytes(digest[:16], byteorder="big"))


def make_synthetic_study(
    study_instance_uid: str,
    *,
    series_count: int = 2,
    slices_per_series: int = 6,
    seed: int = 20260911,
) -> StudyRecord:
    """Create a reproducible multi-series study with shuffled source slices.

    Each slice is assembled as an explicit-little-endian pydicom dataset and
    passed through the decoder adapter. No private competition data or files
    are used. Slice filenames and instance numbers are deliberately unhelpful
    so geometry ordering is exercised.
    """

    if series_count <= 0 or slices_per_series <= 0:
        raise ValueError("series_count and slices_per_series must be positive")
    random = np.random.default_rng(seed)
    dicom_study_uid = _stable_uid(study_instance_uid, "study")
    series: list[SeriesRecord] = []
    for series_index in range(series_count):
        source_slices: list[DicomSlice] = []
        dicom_series_uid = _stable_uid(study_instance_uid, "series", series_index)
        for slice_index in range(slices_per_series):
            height, width = 24 + series_index * 2, 32 + series_index * 3
            row_gradient = np.linspace(0.0, 1.0, num=height, dtype=np.float32)[:, None]
            column_gradient = np.linspace(0.0, 1.0, num=width, dtype=np.float32)[None, :]
            noise = random.normal(0.0, 0.02, size=(height, width)).astype(np.float32)
            pixels = row_gradient + column_gradient + slice_index * 0.1 + noise
            raw_pixels = np.clip(np.rint((pixels - pixels.min()) * 1000.0), 0, 65535).astype(np.uint16)
            sop_uid = UID(_stable_uid(study_instance_uid, "slice", series_index, slice_index))
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = MRImageStorage
            file_meta.MediaStorageSOPInstanceUID = sop_uid
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            dataset = FileDataset(
                f"not-spatial-{slices_per_series - slice_index:04d}.dcm",
                {},
                file_meta=file_meta,
                preamble=b"\0" * 128,
            )
            dataset.SOPClassUID = MRImageStorage
            dataset.SOPInstanceUID = sop_uid
            dataset.StudyInstanceUID = UID(dicom_study_uid)
            dataset.SeriesInstanceUID = UID(dicom_series_uid)
            dataset.Modality = "MR"
            dataset.ImagePositionPatient = [0.0, 0.0, float(slice_index)]
            dataset.ImageOrientationPatient = list(ORIENTATION)
            dataset.InstanceNumber = slices_per_series - slice_index
            dataset.Rows, dataset.Columns = raw_pixels.shape
            dataset.SamplesPerPixel = 1
            dataset.PhotometricInterpretation = "MONOCHROME2"
            dataset.BitsAllocated = 16
            dataset.BitsStored = 16
            dataset.HighBit = 15
            dataset.PixelRepresentation = 0
            dataset.PixelData = raw_pixels.tobytes()
            source_slices.append(
                decode_dataset(
                    dataset,
                    filename=f"not-spatial-{slices_per_series - slice_index:04d}.dcm",
                )
            )
        source_slices.reverse()
        series.append(
            SeriesRecord(
                study_instance_uid=dicom_study_uid,
                series_instance_uid=dicom_series_uid,
                slices=tuple(source_slices),
            )
        )
    return StudyRecord(dicom_study_uid, tuple(reversed(series)))
