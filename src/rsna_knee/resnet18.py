"""Frozen slice-level ResNet-18 features with deterministic study pooling."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .feature_pooling import (
    ANATOMICAL_POOLING_ID,
    SIMPLE_POOLING_ID,
    pool_frozen_features,
)
from .preprocessing import SERIES_DESCRIPTOR_COLUMNS


RESNET18_MODEL_ID = "resnet18-frozen-slice-features-v1"
RESNET18_FINETUNE_MODEL_ID = "resnet18-layer4-finetune-v1"
RESNET18_ENCODER_ID = "torchvision-resnet18-imagenet1k-v1"
RESNET18_FIXTURE_ENCODER_ID = "ticket31-random-fixture-resnet18-v1"
RESNET18_FEATURE_DIM = 512
RESNET18_FEATURE_BATCH_SIZE = 16


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != channels:
            self.downsample = nn.Sequential(
                _conv1x1(in_channels, channels, stride),
                nn.BatchNorm2d(channels),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value
        result = self.conv1(value)
        result = self.bn1(result)
        result = self.relu(result)
        result = self.conv2(result)
        result = self.bn2(result)
        if self.downsample is not None:
            identity = self.downsample(value)
        return self.relu(result + identity)


class _ResNet18Backbone(nn.Module):
    """TorchVision-compatible ResNet-18 feature trunk without its ImageNet head."""

    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=2)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, channels: int, *, blocks: int, stride: int = 1) -> nn.Sequential:
        layers: list[nn.Module] = [_BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers.extend(_BasicBlock(self.in_channels, channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward_before_layer4(self, images: torch.Tensor) -> torch.Tensor:
        result = self.conv1(images)
        result = self.bn1(result)
        result = self.relu(result)
        result = self.maxpool(result)
        result = self.layer1(result)
        result = self.layer2(result)
        return self.layer3(result)

    def forward_from_layer4(self, features: torch.Tensor) -> torch.Tensor:
        result = self.layer4(features)
        return torch.flatten(self.avgpool(result), start_dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_from_layer4(self.forward_before_layer4(images))


class ResNet18StudyModel(nn.Module):
    """Frozen ResNet-18 slice features and a fold-specific trainable study head."""

    imagenet_mean: torch.Tensor
    imagenet_std: torch.Tensor

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        if config.input_channels != 1:
            raise ValueError("the ResNet-18 knee representation expects grayscale slices")
        if len(config.image_size) != 2 or min(config.image_size) < 32:
            raise ValueError("ResNet-18 images must be at least 32 pixels in each dimension")
        if config.encoder_id not in {RESNET18_ENCODER_ID, RESNET18_FIXTURE_ENCODER_ID}:
            raise ValueError(f"unsupported ResNet-18 encoder identity {config.encoder_id!r}")
        if config.pooling_id not in {ANATOMICAL_POOLING_ID, SIMPLE_POOLING_ID}:
            raise ValueError(f"unsupported ResNet-18 pooling identity {config.pooling_id!r}")
        self.encoder = _ResNet18Backbone()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        if config.pooling_id == ANATOMICAL_POOLING_ID:
            pooled_width = 4 * (RESNET18_FEATURE_DIM + 6 + 1)
        else:
            pooled_width = RESNET18_FEATURE_DIM + len(SERIES_DESCRIPTOR_COLUMNS) + 1
        self.head = nn.Sequential(
            nn.Linear(pooled_width, config.head_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.head_hidden_size, config.num_targets),
        )
        self.register_buffer(
            "imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )

    def train(self, mode: bool = True) -> ResNet18StudyModel:
        super().train(mode)
        self.encoder.eval()
        return self

    def load_pretrained_state(self, state: Mapping[str, torch.Tensor]) -> None:
        """Load the torchvision state mapping after its artifact has been admitted."""

        source = dict(state)
        source.pop("fc.weight", None)
        source.pop("fc.bias", None)
        try:
            self.encoder.load_state_dict(source, strict=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("pretrained checkpoint is not compatible with ResNet-18") from exc

    def encode_packed(
        self,
        images: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode only valid slices, returning zero features for padded slots."""

        if images.ndim != 6:
            raise ValueError("images must have shape (batch, series, slices, channels, height, width)")
        batch, series, slices, channels, height, width = images.shape
        if (height, width) != tuple(self.config.image_size) or channels != 1:
            raise ValueError("image shape does not match the ResNet-18 configuration")
        if slice_mask.shape != (batch, series, slices) or series_mask.shape != (batch, series):
            raise ValueError("slice and series masks do not match the packed images")
        valid = slice_mask.to(dtype=torch.bool) & series_mask.to(dtype=torch.bool).unsqueeze(-1)
        flat = images.to(dtype=torch.float32).reshape(-1, 1, height, width)
        valid_indices = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
        features = flat.new_zeros((flat.shape[0], RESNET18_FEATURE_DIM))
        if valid_indices.numel():
            with torch.no_grad():
                for start in range(0, valid_indices.numel(), RESNET18_FEATURE_BATCH_SIZE):
                    indices = valid_indices[start : start + RESNET18_FEATURE_BATCH_SIZE]
                    grayscale = flat.index_select(0, indices)
                    rgb = grayscale.expand(-1, 3, -1, -1)
                    normalized = (rgb - self.imagenet_mean) / self.imagenet_std
                    encoded = self.encoder(normalized)
                    features.index_copy_(0, indices, encoded)
        return features.reshape(batch, series, slices, RESNET18_FEATURE_DIM)

    def _pool_features(
        self,
        features: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
        series_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        return pool_frozen_features(
            features,
            slice_mask,
            series_mask,
            series_descriptors,
            feature_width=RESNET18_FEATURE_DIM,
            pooling_id=self.config.pooling_id,
        )

    def forward_from_features(
        self,
        features: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
        series_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self._pool_features(features, slice_mask, series_mask, series_descriptors))

    def forward(
        self,
        images: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
        series_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        features = self.encode_packed(images, slice_mask, series_mask)
        return self.forward_from_features(features, slice_mask, series_mask, series_descriptors)

    def predict_proba(
        self,
        images: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
        series_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self(images, slice_mask, series_mask, series_descriptors)).clamp(0.0, 1.0)

    def predict_proba_from_features(
        self,
        features: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
        series_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.forward_from_features(features, slice_mask, series_mask, series_descriptors)
        ).clamp(0.0, 1.0)

    def encode_one_study(
        self,
        images: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode one packed study for the immutable feature cache."""

        if images.ndim != 5 or slice_mask.ndim != 2 or series_mask.ndim != 1:
            raise ValueError("one-study feature extraction requires unbatched packed inputs")
        encoded = self.encode_packed(
            images.unsqueeze(0), slice_mask.unsqueeze(0), series_mask.unsqueeze(0)
        )
        return encoded.squeeze(0)


class ResNet18FineTunedStudyModel(ResNet18StudyModel):
    """Adapt only ResNet-18 layer4 while fitting a fold-specific study head."""

    def __init__(self, config: Any) -> None:
        if config.model_id != RESNET18_FINETUNE_MODEL_ID:
            raise ValueError("limited fine-tuning requires its versioned model identity")
        super().__init__(config)
        for parameter in self.encoder.layer4.parameters():
            parameter.requires_grad_(True)

    def encode_packed(
        self,
        images: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode valid slices with gradients only through the final residual stage."""

        if images.ndim != 6:
            raise ValueError("images must have shape (batch, series, slices, channels, height, width)")
        batch, series, slices, channels, height, width = images.shape
        if (height, width) != tuple(self.config.image_size) or channels != 1:
            raise ValueError("image shape does not match the ResNet-18 configuration")
        if slice_mask.shape != (batch, series, slices) or series_mask.shape != (batch, series):
            raise ValueError("slice and series masks do not match the packed images")
        valid = slice_mask.to(dtype=torch.bool) & series_mask.to(dtype=torch.bool).unsqueeze(-1)
        flat = images.to(dtype=torch.float32).reshape(-1, 1, height, width)
        valid_indices = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
        if not valid_indices.numel():
            return flat.new_zeros((batch, series, slices, RESNET18_FEATURE_DIM))

        encoded_chunks: list[torch.Tensor] = []
        for start in range(0, valid_indices.numel(), RESNET18_FEATURE_BATCH_SIZE):
            indices = valid_indices[start : start + RESNET18_FEATURE_BATCH_SIZE]
            grayscale = flat.index_select(0, indices)
            rgb = grayscale.expand(-1, 3, -1, -1)
            normalized = (rgb - self.imagenet_mean) / self.imagenet_std
            with torch.no_grad():
                frozen_features = self.encoder.forward_before_layer4(normalized)
            encoded_chunks.append(self.encoder.forward_from_layer4(frozen_features))

        encoded = torch.cat(encoded_chunks, dim=0)
        features = flat.new_zeros((flat.shape[0], RESNET18_FEATURE_DIM)).index_copy(
            0, valid_indices, encoded
        )
        return features.reshape(batch, series, slices, RESNET18_FEATURE_DIM)


__all__ = [
    "ANATOMICAL_POOLING_ID",
    "RESNET18_ENCODER_ID",
    "RESNET18_FINETUNE_MODEL_ID",
    "RESNET18_FEATURE_DIM",
    "RESNET18_FIXTURE_ENCODER_ID",
    "RESNET18_MODEL_ID",
    "ResNet18StudyModel",
    "ResNet18FineTunedStudyModel",
    "SIMPLE_POOLING_ID",
]
