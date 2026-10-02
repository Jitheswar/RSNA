"""Fixed CPU input mapping for four real studies and four TPU padding ranks.

All admission and exposure accounting happen before device transfer. This module
cannot acquire a device or imply original-MRI verification or TPU execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .replicated_masked_loss import RANK_ROLES, WORLD_SIZE, rank_supervision
from .targets import NUM_TARGETS

INPUT_SHAPES = dict(images=(4, 16, 8, 1, 224, 224), slice_mask=(4, 16, 8), series_mask=(4, 16),
    series_descriptors=(4, 16, 10), targets=(4, NUM_TARGETS), supervision_mask=(4, NUM_TARGETS))
BOOL_FIELDS = frozenset(('slice_mask', 'series_mask', 'supervision_mask'))
MODEL_FIELDS = ('images', 'slice_mask', 'series_mask', 'series_descriptors')


@dataclass(frozen=True)
class PilotReplicaBatch:
    rank: int
    role: str
    model_inputs: tuple[torch.Tensor, ...]
    targets: torch.Tensor
    supervision_mask: torch.Tensor
    global_supervised_entries: torch.Tensor
    sample_exposures: int
    supervised_entry_exposures: int


def map_global_batch_to_replica(batch: Mapping[str, torch.Tensor], rank: int) -> PilotReplicaBatch:
    """Preserve the ordered reference/reference/weak/weak batch without resampling."""
    if type(rank) is not int or rank not in range(WORLD_SIZE):
        raise ValueError('One of the eight fixed global ranks required')
    if set(batch) != set(INPUT_SHAPES):
        raise ValueError('All original input fields required')
    for name, shape in INPUT_SHAPES.items():
        value = batch[name]
        dtype = torch.bool if name in BOOL_FIELDS else torch.float32
        if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f'Original CPU shape and dtype required for {name}')
    if not bool(torch.isfinite(batch['images']).all()) or not bool(torch.isfinite(batch['series_descriptors']).all()):
        raise ValueError('Finite original CPU pixels/descriptors required')
    valid = batch['slice_mask'] & batch['series_mask'].unsqueeze(-1)
    if not bool(valid.reshape(4, -1).any(dim=1).all()) or not bool(batch['supervision_mask'].any(dim=1).all()):
        raise ValueError('Every real draw needs original valid MRI and nonempty supervision')
    targets, mask, denominator = rank_supervision(batch['targets'], batch['supervision_mask'], rank)
    inputs = tuple(batch[name][rank:rank + 1] if rank < 4 else torch.zeros_like(batch[name][:1]) for name in MODEL_FIELDS)
    return PilotReplicaBatch(rank, RANK_ROLES[rank], inputs, targets, mask, denominator,
        int(rank < 4), int(mask.sum()))


def global_source_exposures(ordered_masks: Sequence[torch.Tensor], successful_steps: int) -> dict[str, dict[str, int]]:
    """Count only the fixed real draws; resumed paths count the original prefix."""
    if len(ordered_masks) != 16 or type(successful_steps) is not int or successful_steps not in range(17):
        raise ValueError('Complete original16-mask sequence and exact0..16 boundary required')
    for mask in ordered_masks:
        if not isinstance(mask, torch.Tensor) or mask.device.type != 'cpu' or mask.dtype != torch.bool or mask.shape != (4, NUM_TARGETS):
            raise ValueError('Original four-study CPU supervision masks required')
        if not bool(mask.any(dim=1).all()):
            raise ValueError('Padding cannot replace an original real draw')
    result = {role: dict(sample_exposures=0, supervised_entry_exposures=0) for role in ('reference', 'weak', 'mixed', 'pseudo')}
    for mask in ordered_masks[:successful_steps]:
        for role, start in (('reference', 0), ('weak', 2)):
            result[role]['sample_exposures'] += 2
            result[role]['supervised_entry_exposures'] += int(mask[start:start + 2].sum())
    return result
