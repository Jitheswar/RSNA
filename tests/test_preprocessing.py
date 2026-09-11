from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.domain import DicomSlice, SeriesRecord, StudyRecord
from rsna_knee.preprocessing import (
    PreprocessConfig,
    PreprocessingError,
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


def test_preprocessing_masks_stored_padding_before_modality_normalization() -> None:
    item = DicomSlice(
        sop_instance_uid="slice-padding",
        image_position=(0.0, 0.0, 0.0),
        image_orientation=ORIENTATION,
        pixel_array=np.array([[-1, 0], [10, 20]], dtype=np.int16),
        padding_value=-1.0,
        padding_range_limit=0.0,
        rescale_slope=2.0,
        rescale_intercept=-10.0,
    )
    series = SeriesRecord("study-padding", "series-padding", (item,))

    prepared = prepare_series(
        series,
        PreprocessConfig(image_size=(2, 2), max_slices=8, max_series=1),
    )

    np.testing.assert_allclose(prepared.images[0, 0], [[0.0, 0.0], [0.0, 1.0]])
    assert prepared.warnings == ()


def test_preprocessing_records_a_warning_for_constant_valid_intensity() -> None:
    item = DicomSlice(
        sop_instance_uid="slice-constant",
        image_position=(0.0, 0.0, 0.0),
        image_orientation=ORIENTATION,
        pixel_array=np.array([[-1, 5], [-1, 5]], dtype=np.int16),
        padding_value=-1.0,
    )
    prepared = prepare_series(
        SeriesRecord("study-constant", "series-constant", (item,)),
        PreprocessConfig(image_size=(2, 2), max_slices=8, max_series=1),
    )

    assert np.all(prepared.images == 0.0)
    assert prepared.warnings == ("slice-constant:constant_valid_intensity",)


def test_cache_configuration_does_not_change_when_only_sampling_changes() -> None:
    first = PreprocessConfig(image_size=(8, 8), max_slices=8, max_series=4)
    second = PreprocessConfig(image_size=(8, 8), max_slices=16, max_series=12)

    assert first.cache_config() == second.cache_config()


def test_preprocessing_rejects_an_all_padding_slice() -> None:
    item = DicomSlice(
        sop_instance_uid="slice-all-padding",
        image_position=(0.0, 0.0, 0.0),
        image_orientation=ORIENTATION,
        pixel_array=np.full((2, 2), -1, dtype=np.int16),
        padding_value=-1.0,
    )

    with pytest.raises(PreprocessingError, match="no valid pixels"):
        prepare_series(
            SeriesRecord("study-all-padding", "series-all-padding", (item,)),
            PreprocessConfig(image_size=(2, 2), max_slices=8, max_series=1),
        )


def test_preprocessing_applies_monochrome1_polarity_once() -> None:
    item = DicomSlice(
        sop_instance_uid="slice-monochrome1",
        image_position=(0.0, 0.0, 0.0),
        image_orientation=ORIENTATION,
        pixel_array=np.array([[0, 10]], dtype=np.uint16),
        photometric_interpretation="MONOCHROME1",
    )

    prepared = prepare_series(
        SeriesRecord("study-monochrome1", "series-monochrome1", (item,)),
        PreprocessConfig(image_size=(1, 2), max_slices=8, max_series=1),
    )

    np.testing.assert_allclose(prepared.images[0, 0], [[1.0, 0.0]])
