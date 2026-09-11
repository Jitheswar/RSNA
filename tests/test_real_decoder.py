from __future__ import annotations

import io

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, MRImageStorage

from rsna_knee.dicom import (
    DicomValidationError,
    DecoderPolicy,
    UnsupportedTransferSyntaxError,
    decode_bytes,
    decode_stream,
)


def _dicom_bytes(
    *,
    transfer_syntax: str = ExplicitVRLittleEndian,
    pixel_representation: int = 0,
    photometric: str = "MONOCHROME2",
    padding_value: int | None = None,
    padding_range_limit: int | None = None,
    modality: str = "MR",
) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.8.498.100"
    file_meta.TransferSyntaxUID = transfer_syntax
    dataset = FileDataset("fixture.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = "1.2.826.0.1.3680043.8.498.100"
    dataset.StudyInstanceUID = "1.2.826.0.1.3680043.8.498.1"
    dataset.SeriesInstanceUID = "1.2.826.0.1.3680043.8.498.2"
    dataset.Modality = modality
    dataset.ImagePositionPatient = [0.0, 0.0, 2.0]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.InstanceNumber = 1
    dataset.Rows = 2
    dataset.Columns = 3
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = pixel_representation
    if padding_value is not None:
        dataset.PixelPaddingValue = padding_value
    if padding_range_limit is not None:
        dataset.PixelPaddingRangeLimit = padding_range_limit
    pixels = np.array([[padding_value or 0, 2, 3], [4, 5, 6]], dtype=np.int16 if pixel_representation else np.uint16)
    dataset.PixelData = pixels.tobytes()
    buffer = io.BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def test_decoder_maps_a_realistic_signed_slice_and_padding_range() -> None:
    decoded = decode_bytes(
        _dicom_bytes(pixel_representation=1, padding_value=-1, padding_range_limit=0),
        policy=DecoderPolicy(),
        filename="fixture.dcm",
    )

    assert decoded.transfer_syntax_uid == str(ExplicitVRLittleEndian)
    assert decoded.modality == "MR"
    assert decoded.rows == 2
    assert decoded.columns == 3
    assert decoded.padding_value == -1.0
    assert decoded.padding_range_limit == 0.0
    assert decoded.filename == "fixture.dcm"


def test_decoder_rejects_transfer_syntax_outside_the_frozen_policy() -> None:
    with pytest.raises(UnsupportedTransferSyntaxError, match="transfer syntax"):
        decode_bytes(
            _dicom_bytes(transfer_syntax=ImplicitVRLittleEndian),
            policy=DecoderPolicy(),
        )


def test_decoder_rejects_non_mr_and_multichannel_pixel_semantics() -> None:
    with pytest.raises(DicomValidationError, match="Modality"):
        decode_bytes(_dicom_bytes(modality="CT"), policy=DecoderPolicy())


def test_decoder_stream_and_bytes_paths_share_the_same_policy() -> None:
    payload = _dicom_bytes()

    from_bytes = decode_bytes(payload, policy=DecoderPolicy())
    from_stream = decode_stream(io.BytesIO(payload), policy=DecoderPolicy())

    np.testing.assert_array_equal(from_bytes.pixel_array, from_stream.pixel_array)
    assert from_bytes.sop_instance_uid == from_stream.sop_instance_uid
