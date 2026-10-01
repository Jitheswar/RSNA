"""Run the native final ResNet stage on verified packed prefix activations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .prefix_cache import PrefixEntry
from .resnet18 import RESNET18_FEATURE_BATCH_SIZE, RESNET18_FEATURE_DIM, ResNet18StudyModel


def prefix_pooled_features(
    model: ResNet18StudyModel, entries: Sequence[PrefixEntry], *, zero_images: bool = False,
) -> torch.Tensor:
    """Preserve slice order, anatomical pooling and gradients through layer4.

    Entries have already passed the immutable prefix-cache loader. Chunking
    matches native MRI inference, including chunks that span study boundaries.
    """
    if not entries:
        raise ValueError("A nonempty verified prefix batch is required")
    if model.imagenet_mean.device.type != "cpu":
        raise ValueError("This prefix-adaptation experiment is CPU only")
    slices = torch.from_numpy(np.stack([entry.slice_mask for entry in entries]))
    series = torch.from_numpy(np.stack([entry.series_mask for entry in entries]))
    descriptors = torch.from_numpy(np.stack([entry.series_descriptors for entry in entries]))
    packed_count = entries[0].slice_mask.size
    if any(entry.slice_mask.shape != entries[0].slice_mask.shape for entry in entries):
        raise ValueError("Prefix batches must share the frozen packing shape")
    features = torch.zeros((len(entries) * packed_count, RESNET18_FEATURE_DIM))
    if not zero_images:
        prefix = torch.from_numpy(np.concatenate([entry.prefix for entry in entries], axis=0))
        indices = torch.from_numpy(np.concatenate([
            entry.valid_flat_indices + index * packed_count for index, entry in enumerate(entries)
        ]))
        values = torch.cat([
            model.encoder.forward_from_layer4(prefix[start:start + RESNET18_FEATURE_BATCH_SIZE])
            for start in range(0, len(prefix), RESNET18_FEATURE_BATCH_SIZE)
        ])
        features = features.index_copy(0, indices, values)
    return model._pool_features(
        features.reshape(len(entries), *entries[0].slice_mask.shape, RESNET18_FEATURE_DIM),
        slices, series, descriptors,
    )


def prefix_logits(
    model: ResNet18StudyModel, entries: Sequence[PrefixEntry], *, zero_images: bool = False,
) -> torch.Tensor:
    """Use the same head as native study inference."""
    return model.head(prefix_pooled_features(model, entries, zero_images=zero_images))
