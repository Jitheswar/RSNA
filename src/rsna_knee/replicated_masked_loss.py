"""Eight-rank pilot loss preserving the original four-study masked BCE mean.

This is a separate mechanics seam, not an admitted TPU trainer. Input admission
runs on CPU before transfer; the replica loss never reads a device scalar.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .targets import NUM_TARGETS

WORLD_SIZE = 8
REAL_STUDIES = 4
RANK_ROLES = ("reference", "reference", "weak", "weak", "padding", "padding", "padding", "padding")


def admit_global_supervised_entries(targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Validate the four original CPU draws and freeze their common denominator."""
    if targets.device.type != "cpu" or mask.device.type != "cpu":
        raise ValueError("Global supervision must be admitted on CPU before transfer")
    if targets.shape != (REAL_STUDIES, NUM_TARGETS) or mask.shape != targets.shape:
        raise ValueError("Exactly four original twelve-target studies required")
    if targets.dtype != torch.float32 or mask.dtype != torch.bool:
        raise ValueError("Original FP32 targets and boolean supervision required")
    supervised = targets[mask]
    if not supervised.numel() or not bool(torch.isfinite(supervised).all()) or not bool(((supervised >= 0) & (supervised <= 1)).all()):
        raise ValueError("Nonempty finite bounded global supervision required")
    return mask.sum().to(dtype=torch.float32)


def rank_supervision(targets: torch.Tensor, mask: torch.Tensor, rank: int
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map fixed real draws to ranks0–3; ranks4–7 add no supervision/exposure."""
    if type(rank) is not int or rank not in range(WORLD_SIZE):
        raise ValueError("Exactly one of eight fixed replica ranks required")
    denominator = admit_global_supervised_entries(targets, mask)
    if rank < REAL_STUDIES:
        return targets[rank:rank + 1].clone(), mask[rank:rank + 1].clone(), denominator
    return torch.zeros((1, NUM_TARGETS), dtype=torch.float32), torch.zeros((1, NUM_TARGETS), dtype=torch.bool), denominator


def replica_masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                       global_supervised_entries: torch.Tensor, *, world_size: int = WORLD_SIZE
                       ) -> torch.Tensor:
    """Return8*N_rank/D_global for an ensuing eight-rank MEAN gradient reduction.

    The positive denominator must be admitted from all four CPU masks before
    transfer. Never use a per-rank mean, a SUM collective, or count padding as
    real examples. Ordinary Adam and finite-state checks remain the caller's
    responsibility. No TPU backend or throughput assertion is made here.
    """
    if type(world_size) is not int or world_size != WORLD_SIZE:
        raise ValueError("This fixed pilot requires eight mean-reduced ranks")
    if logits.shape != (1, NUM_TARGETS) or targets.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError("One twelve-target study per replica required")
    if logits.dtype != torch.float32 or targets.dtype != torch.float32 or mask.dtype != torch.bool:
        raise ValueError("FP32 logits/targets and boolean masks required")
    if global_supervised_entries.shape != () or global_supervised_entries.dtype != torch.float32:
        raise ValueError("Admitted scalar FP32 global denominator required")
    if any(value.device != logits.device for value in (targets, mask, global_supervised_entries)):
        raise ValueError("All replica tensors must be transferred to the same device")
    safe_logits = torch.where(mask, logits, torch.zeros_like(logits))
    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
    entries = F.binary_cross_entropy_with_logits(safe_logits, safe_targets, reduction="none")
    numerator = (entries * mask.to(dtype=logits.dtype)).sum()
    return world_size * numerator / global_supervised_entries
