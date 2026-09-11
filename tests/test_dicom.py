from __future__ import annotations

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

from rsna_knee.dicom import decode_dataset


def test_decode_dataset_maps_a_synthetic_mr_slice_to_the_core_record() -> None:
    sop_uid = "1.2.826.0.1.3680043.8.498.100"
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset("synthetic.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPInstanceUID = sop_uid
    dataset.StudyInstanceUID = "1.2.826.0.1.3680043.8.498.1"
    dataset.SeriesInstanceUID = "1.2.826.0.1.3680043.8.498.2"
    dataset.ImagePositionPatient = [0.0, 0.0, 4.0]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.Rows = 2
    dataset.Columns = 2
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.array([[1, 2], [3, 4]], dtype=np.uint16).tobytes()

    decoded = decode_dataset(dataset)

    assert decoded.study_instance_uid == "1.2.826.0.1.3680043.8.498.1"
    assert decoded.series_instance_uid == "1.2.826.0.1.3680043.8.498.2"
    assert decoded.sop_instance_uid == sop_uid
    assert decoded.image_position == (0.0, 0.0, 4.0)
    assert decoded.image_orientation == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    np.testing.assert_array_equal(decoded.pixel_array, np.array([[1, 2], [3, 4]], dtype=np.uint16))
