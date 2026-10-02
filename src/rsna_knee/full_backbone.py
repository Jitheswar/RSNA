"""Fixed-shape whole-backbone MRI learning for accelerator admission."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from .resnet18 import (
    RESNET18_FEATURE_BATCH_SIZE,
    RESNET18_FEATURE_DIM,
    RESNET18_FULL_FINETUNE_MODEL_ID,
    ResNet18StudyModel,
)
from .targets import NUM_TARGETS


class ResNet18FullBackboneStudyModel(ResNet18StudyModel):
    """Train every encoder stage without data-dependent slice tensor shapes.

    Encoder normalization retains admitted running statistics. Encoding padded
    slots therefore cannot change valid slice features or normalization buffers.
    Fixed encoder chunks bound individual operation workspace; training retains
    all chunks' activations, so total memory still requires an actual device
    pilot. The padded bag dimensions determine the graph independently of how
    many slices are valid in a particular batch.
    """

    def __init__(self, config: Any) -> None:
        if config.model_id != RESNET18_FULL_FINETUNE_MODEL_ID:
            raise ValueError("whole-backbone adaptation requires its versioned identity")
        super().__init__(config)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)

    def encode_packed(
        self,
        images: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 6:
            raise ValueError("images must have shape (batch, series, slices, channels, height, width)")
        batch, series, slices, channels, height, width = images.shape
        if (height, width) != tuple(self.config.image_size) or channels != 1:
            raise ValueError("image shape does not match the full-backbone configuration")
        if slice_mask.shape != (batch, series, slices) or series_mask.shape != (batch, series):
            raise ValueError("slice and series masks do not match the packed images")
        if batch * series * slices == 0:
            raise ValueError("fixed-shape bags must have nonempty dimensions")
        valid = slice_mask.to(device=images.device, dtype=torch.bool)
        valid = valid & series_mask.to(device=images.device, dtype=torch.bool).unsqueeze(-1)
        flat = images.to(dtype=torch.float32).reshape(-1, 1, height, width)
        admitted = valid.reshape(-1, 1, 1, 1)
        flat = torch.where(admitted, flat, torch.zeros_like(flat))
        chunks = []
        for start in range(0, flat.shape[0], RESNET18_FEATURE_BATCH_SIZE):
            rgb = flat[start : start + RESNET18_FEATURE_BATCH_SIZE].expand(-1, 3, -1, -1)
            chunks.append(self.encoder((rgb - self.imagenet_mean) / self.imagenet_std))
        encoded = torch.cat(chunks, dim=0)
        encoded = torch.where(valid.reshape(-1, 1), encoded, torch.zeros_like(encoded))
        return encoded.reshape(batch, series, slices, RESNET18_FEATURE_DIM)


def fixed_shape_masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> torch.Tensor:
    """Average over supervised entries without a scalar read or masked indexing.

    Absent logits/targets are replaced before BCE, including NaNs. Empty masks
    yield differentiable zero; the training-data admission must separately
    reject empty supervision before transferring a batch to the accelerator.
    """

    if logits.shape != targets.shape or logits.shape != supervision_mask.shape:
        raise ValueError("logits, targets and supervision mask must have identical shapes")
    if logits.ndim != 2 or logits.shape[1] != NUM_TARGETS:
        raise ValueError(f"training tensors must have shape (batch, {NUM_TARGETS})")
    mask = supervision_mask.to(device=logits.device, dtype=torch.bool)
    safe_logits = torch.where(mask, logits, torch.zeros_like(logits))
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
    entries = F.binary_cross_entropy_with_logits(safe_logits, safe_targets, reduction="none")
    weights = mask.to(dtype=logits.dtype)
    return (entries * weights).sum() / weights.sum().clamp_min(1.0)


__all__ = ["ResNet18FullBackboneStudyModel", "fixed_shape_masked_bce"]
