"""Deterministic geometry validation and anatomical slice ordering."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .domain import DicomSlice


class GeometryError(ValueError):
    """Raised when a series cannot be ordered from reliable DICOM geometry."""


GEOMETRY_POLICY_VERSION = "rsna-knee-geometry-v2"


@dataclass(frozen=True)
class GeometryPolicy:
    """Numerical policy used by :func:`order_slices`.

    The policy is deliberately explicit so a future data audit can version
    and change tolerances or the minimum usable-slice count without silently
    changing an existing cache.
    """

    policy_version: str = GEOMETRY_POLICY_VERSION
    position_tolerance: float = 1e-4
    orientation_tolerance: float = 1e-3
    spacing_tolerance: float = 1e-3
    minimum_usable_slices: int = 1
    require_consistent_dimensions: bool = True
    require_consistent_orientation: bool = True

    def __post_init__(self) -> None:
        if self.policy_version != GEOMETRY_POLICY_VERSION:
            raise ValueError(f"unsupported geometry policy version {self.policy_version!r}")
        if self.position_tolerance <= 0:
            raise ValueError("position_tolerance must be positive")
        if self.orientation_tolerance <= 0:
            raise ValueError("orientation_tolerance must be positive")
        if self.spacing_tolerance <= 0:
            raise ValueError("spacing_tolerance must be positive")
        if self.minimum_usable_slices <= 0:
            raise ValueError("minimum_usable_slices must be positive")
        if not isinstance(self.require_consistent_dimensions, bool):
            raise ValueError("require_consistent_dimensions must be boolean")
        if not isinstance(self.require_consistent_orientation, bool):
            raise ValueError("require_consistent_orientation must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "position_tolerance": self.position_tolerance,
            "orientation_tolerance": self.orientation_tolerance,
            "spacing_tolerance": self.spacing_tolerance,
            "minimum_usable_slices": self.minimum_usable_slices,
            "require_consistent_dimensions": self.require_consistent_dimensions,
            "require_consistent_orientation": self.require_consistent_orientation,
        }

    @classmethod
    def from_dict(cls, value: object) -> "GeometryPolicy":
        if not isinstance(value, dict):
            raise ValueError("geometry policy must be an object")
        return cls(
            policy_version=str(value.get("policy_version", "")),
            position_tolerance=float(value.get("position_tolerance", 1e-4)),
            orientation_tolerance=float(value.get("orientation_tolerance", 1e-3)),
            spacing_tolerance=float(value.get("spacing_tolerance", 1e-3)),
            minimum_usable_slices=int(value.get("minimum_usable_slices", 1)),
            require_consistent_dimensions=bool(value.get("require_consistent_dimensions", True)),
            require_consistent_orientation=bool(value.get("require_consistent_orientation", True)),
        )


def _finite_vector(
    value: object,
    *,
    length: int,
    field_name: str,
    uid: str,
) -> np.ndarray:
    if value is None:
        raise GeometryError(f"{field_name} is missing for slice {uid}")
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise GeometryError(f"{field_name} is invalid for slice {uid}")
    return vector


def _normal_for(item: DicomSlice, policy: GeometryPolicy) -> np.ndarray:
    orientation = _finite_vector(
        item.image_orientation,
        length=6,
        field_name="image orientation",
        uid=item.sop_instance_uid,
    )
    row = orientation[:3]
    column = orientation[3:]
    row_norm = np.linalg.norm(row)
    column_norm = np.linalg.norm(column)
    if not np.isclose(row_norm, 1.0, atol=policy.orientation_tolerance):
        raise GeometryError(f"row direction is not unit length for slice {item.sop_instance_uid}")
    if not np.isclose(column_norm, 1.0, atol=policy.orientation_tolerance):
        raise GeometryError(f"column direction is not unit length for slice {item.sop_instance_uid}")
    if not np.isclose(float(np.dot(row, column)), 0.0, atol=policy.orientation_tolerance):
        raise GeometryError(f"row and column directions are not orthogonal for slice {item.sop_instance_uid}")
    normal = np.cross(row, column)
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= policy.orientation_tolerance:
        raise GeometryError(f"slice normal is undefined for slice {item.sop_instance_uid}")
    return normal / normal_norm


def order_slices(
    slices: Sequence[DicomSlice],
    policy: GeometryPolicy = GeometryPolicy(),
) -> tuple[DicomSlice, ...]:
    """Return slices ordered by their position projected onto the series normal.

    File names and ``InstanceNumber`` are intentionally never consulted. Every
    slice must carry finite position and orientation metadata, and every
    orientation must agree with the first slice. Duplicate projected
    positions are rejected because their order would be ambiguous.
    """

    if not slices:
        raise GeometryError("series contains no slices")
    if len(slices) < policy.minimum_usable_slices:
        raise GeometryError(
            f"series contains {len(slices)} slices; at least "
            f"{policy.minimum_usable_slices} usable slices are required"
        )

    reference_normal: np.ndarray | None = None
    reference_orientation: np.ndarray | None = None
    positioned: list[tuple[float, DicomSlice]] = []
    reference_shape: tuple[int, int] | None = None
    for item in slices:
        position = _finite_vector(
            item.image_position,
            length=3,
            field_name="image position",
            uid=item.sop_instance_uid,
        )
        orientation = _finite_vector(
            item.image_orientation,
            length=6,
            field_name="image orientation",
            uid=item.sop_instance_uid,
        )
        normal = _normal_for(item, policy)
        try:
            shape = tuple(np.asarray(item.pixel_array).shape)
        except (TypeError, ValueError) as exc:
            raise GeometryError(f"pixel dimensions are invalid for slice {item.sop_instance_uid}") from exc
        if len(shape) != 2 or 0 in shape:
            raise GeometryError(f"pixel dimensions are invalid for slice {item.sop_instance_uid}")
        if reference_shape is None:
            reference_shape = (int(shape[0]), int(shape[1]))
        elif policy.require_consistent_dimensions and shape != reference_shape:
            raise GeometryError(
                f"inconsistent pixel dimensions for slice {item.sop_instance_uid}: "
                f"{shape} does not match {reference_shape}"
            )
        if reference_normal is None:
            reference_normal = normal
            reference_orientation = orientation
        elif float(np.dot(reference_normal, normal)) < 1.0 - policy.orientation_tolerance:
            raise GeometryError(f"inconsistent slice orientation for slice {item.sop_instance_uid}")
        elif (
            policy.require_consistent_orientation
            and reference_orientation is not None
            and not np.allclose(orientation, reference_orientation, atol=policy.orientation_tolerance, rtol=0.0)
        ):
            raise GeometryError(f"inconsistent slice orientation for slice {item.sop_instance_uid}")
        positioned.append((float(np.dot(position, reference_normal)), item))

    positioned.sort(key=lambda entry: (entry[0], entry[1].sop_instance_uid))
    for previous, current in zip(positioned, positioned[1:]):
        if current[0] - previous[0] <= policy.position_tolerance:
            raise GeometryError(
                "duplicate or indistinguishable spatial location for "
                f"slices {previous[1].sop_instance_uid} and {current[1].sop_instance_uid}"
            )
    if len(positioned) > 2:
        spacings = [current[0] - previous[0] for previous, current in zip(positioned, positioned[1:])]
        reference_spacing = float(np.median(np.asarray(spacings, dtype=np.float64)))
        if reference_spacing <= policy.position_tolerance:
            raise GeometryError("series has no positive spatial spacing")
        if any(
            not math.isclose(
                spacing,
                reference_spacing,
                rel_tol=policy.spacing_tolerance,
                abs_tol=policy.spacing_tolerance,
            )
            for spacing in spacings
        ):
            raise GeometryError("inconsistent slice spacing")
    return tuple(item for _, item in positioned)


__all__ = ["GEOMETRY_POLICY_VERSION", "GeometryError", "GeometryPolicy", "order_slices"]
