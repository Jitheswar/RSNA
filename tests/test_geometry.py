from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.domain import DicomSlice
from rsna_knee.geometry import GeometryError, order_slices


ORIENTATION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def make_slice(uid: str, z: float, *, filename: str, instance_number: int) -> DicomSlice:
    return DicomSlice(
        sop_instance_uid=uid,
        filename=filename,
        instance_number=instance_number,
        image_position=(0.0, 0.0, z),
        image_orientation=ORIENTATION,
        pixel_array=np.ones((4, 4), dtype=np.float32),
    )


def test_order_slices_uses_spatial_position_not_enumeration_or_filename() -> None:
    slices = [
        make_slice("s2", 2.0, filename="0001.dcm", instance_number=1),
        make_slice("s0", 0.0, filename="9999.dcm", instance_number=99),
        make_slice("s1", 1.0, filename="0100.dcm", instance_number=2),
    ]

    ordered = order_slices(slices)

    assert [item.sop_instance_uid for item in ordered] == ["s0", "s1", "s2"]


@pytest.mark.parametrize(
    "change",
    [
        {"image_position": None},
        {"image_orientation": None},
        {"image_position": (0.0, 0.0, float("nan"))},
    ],
)
def test_order_slices_rejects_uncertain_geometry(change: dict[str, object]) -> None:
    item = make_slice("s0", 0.0, filename="slice.dcm", instance_number=1)
    for field, value in change.items():
        object.__setattr__(item, field, value)

    with pytest.raises(GeometryError):
        order_slices([item])


def test_order_slices_rejects_duplicate_spatial_locations() -> None:
    slices = [
        make_slice("s0", 0.0, filename="a.dcm", instance_number=1),
        make_slice("s1", 0.0, filename="b.dcm", instance_number=2),
    ]

    with pytest.raises(GeometryError, match="duplicate"):
        order_slices(slices)
