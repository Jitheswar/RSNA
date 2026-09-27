"""Plane-aware pooling shared by frozen slice representations."""

from __future__ import annotations

import torch

from .preprocessing import SERIES_DESCRIPTOR_COLUMNS


ANATOMICAL_POOLING_ID = "plane-stratified-series-mean-v1"
SIMPLE_POOLING_ID = "flat-series-mean-v1"


def pool_frozen_features(
    features: torch.Tensor,
    slice_mask: torch.Tensor,
    series_mask: torch.Tensor,
    series_descriptors: torch.Tensor,
    *,
    feature_width: int,
    pooling_id: str,
) -> torch.Tensor:
    """Mean-pool valid slice features while preserving plane and descriptor groups."""

    if features.ndim != 4 or features.shape[-1] != feature_width:
        raise ValueError(f"features must have shape (batch, series, slices, {feature_width})")
    batch, series, slices, _ = features.shape
    if slice_mask.shape != (batch, series, slices):
        raise ValueError("slice_mask must match the feature dimensions")
    if series_mask.shape != (batch, series):
        raise ValueError("series_mask must match the feature dimensions")
    if series_descriptors.shape != (batch, series, len(SERIES_DESCRIPTOR_COLUMNS)):
        raise ValueError("series descriptors must preserve the ten declared metadata states")
    if pooling_id not in {ANATOMICAL_POOLING_ID, SIMPLE_POOLING_ID}:
        raise ValueError(f"unsupported frozen-feature pooling identity {pooling_id!r}")

    valid_slices = slice_mask.to(device=features.device, dtype=torch.bool)
    valid_series = series_mask.to(device=features.device, dtype=torch.bool)
    valid = valid_slices & valid_series.unsqueeze(-1)
    weights = valid.to(dtype=features.dtype).unsqueeze(-1)
    series_features = (features * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
    admitted_series = valid.any(dim=2)
    descriptors = series_descriptors.to(device=features.device, dtype=features.dtype)
    descriptor_mask = admitted_series.to(dtype=features.dtype).unsqueeze(-1)

    if pooling_id == SIMPLE_POOLING_ID:
        series_weights = descriptor_mask
        pooled_features = (series_features * series_weights).sum(dim=1)
        pooled_features = pooled_features / series_weights.sum(dim=1).clamp_min(1.0)
        pooled_descriptors = (descriptors * series_weights).sum(dim=1)
        pooled_descriptors = pooled_descriptors / series_weights.sum(dim=1).clamp_min(1.0)
        present = series_weights.sum(dim=1).clamp_max(1.0)
        return torch.cat((pooled_features, pooled_descriptors, present), dim=1)

    plane_groups: list[torch.Tensor] = []
    for plane_index in range(4):
        group_weights = (
            admitted_series & (descriptors[:, :, plane_index] > 0.5)
        ).to(dtype=features.dtype)
        expanded = group_weights.unsqueeze(-1)
        count = group_weights.sum(dim=1, keepdim=True)
        pooled_features = (series_features * expanded).sum(dim=1) / count.clamp_min(1.0)
        pooled_metadata = (descriptors[:, :, 4:] * expanded).sum(dim=1) / count.clamp_min(1.0)
        present = (count > 0).to(dtype=features.dtype)
        plane_groups.append(torch.cat((pooled_features, pooled_metadata, present), dim=1))
    return torch.cat(plane_groups, dim=1)


__all__ = ["ANATOMICAL_POOLING_ID", "SIMPLE_POOLING_ID", "pool_frozen_features"]
