from __future__ import annotations

import numpy as np

from rsna_knee.domain import DicomSlice, SeriesRecord, StudyRecord
from rsna_knee.preprocessing import (
    PreprocessConfig,
    pack_study,
    prepare_series,
)


ORIENTATION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def make_series() -> SeriesRecord:
    slices = tuple(
        DicomSlice(
            sop_instance_uid=f"slice-{index}",
            image_position=(0.0, 0.0, float(index)),
            image_orientation=ORIENTATION,
            pixel_array=np.arange(32, dtype=np.float32).reshape(4, 8) + index,
            filename=f"misleading-{10 - index}.dcm",
            instance_number=10 - index,
        )
        for index in (2, 0, 1)
    )
    return SeriesRecord("study-1", "series-1", slices)


def test_prepare_series_is_fixed_shape_and_retains_spatial_source_order() -> None:
    config = PreprocessConfig(image_size=(8, 8), max_slices=4, max_series=2)

    prepared = prepare_series(make_series(), config)

    assert prepared.images.shape == (3, 1, 8, 8)
    assert prepared.source_slice_uids == ("slice-0", "slice-1", "slice-2")
    assert np.isfinite(prepared.images).all()
    assert float(prepared.images.min()) >= 0.0
    assert float(prepared.images.max()) <= 1.0


def test_pack_study_marks_padding_without_repeating_evidence() -> None:
    config = PreprocessConfig(image_size=(8, 8), max_slices=4, max_series=2)
    study = StudyRecord("study-1", (make_series(),))
    prepared = prepare_series(make_series(), config)

    packed = pack_study(study, (prepared,), config)

    assert packed.images.shape == (2, 4, 1, 8, 8)
    assert packed.slice_mask.tolist() == [[True, True, True, False], [False] * 4]
    assert packed.series_mask.tolist() == [True, False]
    assert packed.sampled_source_indices == ((0, 1, 2), ())
    assert np.all(packed.images[0, 3] == 0.0)
    assert np.all(packed.images[1] == 0.0)
