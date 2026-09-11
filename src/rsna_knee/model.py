"""Tiny image-only hierarchical model for the mechanics baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .targets import NUM_TARGETS


@dataclass(frozen=True)
class ModelConfig:
    """Architecture values that are part of a scientific configuration."""

    image_size: tuple[int, int] = (128, 128)
    channels: tuple[int, int, int] = (8, 16, 32)
    input_channels: int = 1
    num_targets: int = NUM_TARGETS

    def __post_init__(self) -> None:
        if len(self.image_size) != 2 or any(size <= 0 for size in self.image_size):
            raise ValueError("image_size must contain two positive dimensions")
        if len(self.channels) != 3 or any(channel <= 0 for channel in self.channels):
            raise ValueError("channels must contain three positive widths")
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if self.num_targets != NUM_TARGETS:
            raise ValueError(f"the mechanics head must have exactly {NUM_TARGETS} outputs")


class HierarchicalStudyModel(nn.Module):
    """Shared 2D encoder followed by masked series and study pooling.

    The public forward interface accepts a fixed-shape batch with dimensions
    ``(batch, series, slices, channels, height, width)`` plus a slice mask and
    a series mask. Masked pixels never reach either mean pool, so changing
    padded tensor contents cannot change a study prediction.
    """

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        first, second, third = config.channels
        self.slice_encoder = nn.Sequential(
            nn.Conv2d(config.input_channels, first, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(first, second, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(second, third, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(third, config.num_targets)

    def forward(
        self,
        images: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 6:
            raise ValueError("images must have shape (batch, series, slices, channels, height, width)")
        batch_size, series_count, slice_count, channels, height, width = images.shape
        if (height, width) != self.config.image_size:
            raise ValueError(
                f"images have spatial shape {(height, width)}, expected {self.config.image_size}"
            )
        if channels != self.config.input_channels:
            raise ValueError(f"images have {channels} channels, expected {self.config.input_channels}")
        if slice_mask.shape != (batch_size, series_count, slice_count):
            raise ValueError("slice_mask must match the first three image dimensions")
        if series_mask.shape != (batch_size, series_count):
            raise ValueError("series_mask must match the batch and series dimensions")

        flat_images = images.to(dtype=torch.float32).reshape(
            batch_size * series_count * slice_count,
            channels,
            height,
            width,
        )
        encoded = self.slice_encoder(flat_images).flatten(start_dim=1)
        embedding_size = encoded.shape[-1]
        encoded = encoded.reshape(batch_size, series_count, slice_count, embedding_size)

        valid_slices = slice_mask.to(device=images.device, dtype=torch.bool)
        valid_series = series_mask.to(device=images.device, dtype=torch.bool)
        effective_slice_mask = valid_slices & valid_series.unsqueeze(-1)
        slice_weights = effective_slice_mask.to(dtype=encoded.dtype).unsqueeze(-1)
        series_sums = (encoded * slice_weights).sum(dim=2)
        slice_counts = slice_weights.sum(dim=2).clamp_min(1.0)
        series_embeddings = series_sums / slice_counts

        series_has_slices = effective_slice_mask.any(dim=2)
        effective_series_mask = valid_series & series_has_slices
        series_weights = effective_series_mask.to(dtype=encoded.dtype).unsqueeze(-1)
        study_sum = (series_embeddings * series_weights).sum(dim=1)
        series_counts = series_weights.sum(dim=1).clamp_min(1.0)
        study_embedding = study_sum / series_counts
        return self.head(study_embedding)

    def predict_proba(
        self,
        images: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the twelve bounded confidence scores for a packed batch."""

        return torch.sigmoid(self(images, slice_mask, series_mask)).clamp(0.0, 1.0)


# A short name for callers that only need the mechanics baseline.
MechanicsModel = HierarchicalStudyModel
