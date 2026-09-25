"""Tiny image-only hierarchical model for the mechanics baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Mapping, cast

import torch
from torch import nn

from .targets import NUM_TARGETS


MODEL_CONFIG_SCHEMA_VERSION = "rsna-knee-model-config-v1"
COMPACT_MODEL_ID = "compact-hierarchical-cnn-v1"
COMPACT_POOLING_ID = "masked-series-study-mean-v1"


class ModelConfigurationError(ValueError):
    """Raised when a versioned model configuration cannot be reconstructed."""


@dataclass(frozen=True)
class ModelConfig:
    """Versioned architecture values for the compact mechanics model.

    ``to_dict`` writes explicit identity metadata.  ``from_dict`` also accepts
    the pre-versioned shape used by historical checkpoints and candidates.
    Keeping the identity constants outside the dataclass fields leaves the
    compact model's legacy ``__dict__`` and tensor names unchanged.
    """

    schema_version: ClassVar[str] = MODEL_CONFIG_SCHEMA_VERSION
    model_id: ClassVar[str] = COMPACT_MODEL_ID
    pooling_id: ClassVar[str] = COMPACT_POOLING_ID

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

    def to_dict(self) -> dict[str, object]:
        """Return the canonical, explicitly identified model configuration."""

        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "pooling_id": self.pooling_id,
            "image_size": list(self.image_size),
            "channels": list(self.channels),
            "input_channels": self.input_channels,
            "num_targets": self.num_targets,
        }

    @classmethod
    def from_dict(cls, value: object) -> ModelConfig:
        """Parse either the current identity-bearing or historical config."""

        if not isinstance(value, Mapping):
            raise ModelConfigurationError("model configuration must be an object")
        payload = cast(Mapping[str, object], value)
        version = payload.get("schema_version")
        if version is None:
            if any(key in payload for key in ("model_id", "pooling_id")):
                raise ModelConfigurationError(
                    "model identity metadata requires a versioned configuration"
                )
            allowed = {"image_size", "channels", "input_channels", "num_targets"}
        else:
            if version != MODEL_CONFIG_SCHEMA_VERSION:
                raise ModelConfigurationError(f"unsupported model configuration schema {version!r}")
            if payload.get("model_id") != COMPACT_MODEL_ID:
                raise ModelConfigurationError(
                    f"unsupported model identity {payload.get('model_id')!r}"
                )
            if payload.get("pooling_id") != COMPACT_POOLING_ID:
                raise ModelConfigurationError(
                    f"unsupported pooling identity {payload.get('pooling_id')!r}"
                )
            allowed = {
                "schema_version",
                "model_id",
                "pooling_id",
                "image_size",
                "channels",
                "input_channels",
                "num_targets",
            }
        unexpected = set(payload) - allowed
        if unexpected:
            raise ModelConfigurationError(
                f"model configuration contains unsupported fields: {sorted(unexpected)}"
            )
        image_size = payload.get("image_size")
        channels = payload.get("channels")
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
            raise ModelConfigurationError("model image_size must contain two values")
        if not isinstance(channels, (list, tuple)) or len(channels) != 3:
            raise ModelConfigurationError("model channels must contain three values")
        try:
            config = cls(
                image_size=(int(str(image_size[0])), int(str(image_size[1]))),
                channels=(
                    int(str(channels[0])),
                    int(str(channels[1])),
                    int(str(channels[2])),
                ),
                input_channels=int(str(payload.get("input_channels", 1))),
                num_targets=int(str(payload.get("num_targets", NUM_TARGETS))),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelConfigurationError("model configuration values are invalid") from exc
        return config


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


def build_study_model(config: ModelConfig) -> HierarchicalStudyModel:
    """Reconstruct the supported model implementation from its configuration."""

    if not isinstance(config, ModelConfig):
        raise ModelConfigurationError("build_study_model requires a ModelConfig")
    if config.model_id != COMPACT_MODEL_ID or config.pooling_id != COMPACT_POOLING_ID:
        raise ModelConfigurationError("model configuration selects an unsupported implementation")
    return HierarchicalStudyModel(config)


def restore_study_model(config_value: object, state_dict: object) -> HierarchicalStudyModel:
    """Build a model and strictly load its checkpoint state."""

    config = config_value if isinstance(config_value, ModelConfig) else ModelConfig.from_dict(config_value)
    if not isinstance(state_dict, Mapping):
        raise ModelConfigurationError("model state must be an object")
    model = build_study_model(config)
    try:
        model.load_state_dict(cast(Mapping[str, torch.Tensor], state_dict), strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ModelConfigurationError("model state is incompatible with its configuration") from exc
    return model


# A short name for callers that only need the mechanics baseline.
MechanicsModel = HierarchicalStudyModel


__all__ = [
    "COMPACT_MODEL_ID",
    "COMPACT_POOLING_ID",
    "MODEL_CONFIG_SCHEMA_VERSION",
    "HierarchicalStudyModel",
    "MechanicsModel",
    "ModelConfig",
    "ModelConfigurationError",
    "build_study_model",
    "restore_study_model",
]
