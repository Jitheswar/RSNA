"""Train an explicitly identified head on immutable pooled slice features.

This experiment module keeps the existing ResNet and DINO run contracts intact.
It reuses their pooling, masked loss and deterministic additive sampler.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from .feature_pooling import ANATOMICAL_POOLING_ID, pool_frozen_features
from .targets import NUM_TARGETS
from .training import _batch_indices, _model_state_sha256, masked_bce_loss, seed_everything


FEATURE_WIDTH = 768
POOLED_WIDTH = 4 * (FEATURE_WIDTH + 6 + 1)
HEAD_ID = "convnext-tiny-frozen-plane-pooled-mlp32-v1"


def pool_study(features: np.ndarray, slice_mask: np.ndarray,
               series_mask: np.ndarray, descriptors: np.ndarray) -> torch.Tensor:
    """Apply the existing anatomical pooling once, without fitting anything."""
    with torch.inference_mode():
        value = pool_frozen_features(
            torch.from_numpy(features).unsqueeze(0),
            torch.from_numpy(slice_mask).unsqueeze(0),
            torch.from_numpy(series_mask).unsqueeze(0),
            torch.from_numpy(descriptors).unsqueeze(0),
            feature_width=FEATURE_WIDTH, pooling_id=ANATOMICAL_POOLING_ID,
        ).squeeze(0)
    # Clone outside inference mode so the head can save its input for backward.
    return value.clone()


def zero_image_features(pooled: torch.Tensor) -> torch.Tensor:
    if pooled.ndim != 2 or pooled.shape[1] != POOLED_WIDTH:
        raise ValueError("invalid pooled feature shape")
    result = pooled.clone().reshape(-1, 4, FEATURE_WIDTH + 7)
    result[:, :, :FEATURE_WIDTH] = 0
    return result.reshape(-1, POOLED_WIDTH)


def build_head(seed: int) -> nn.Sequential:
    seed_everything(seed)
    return nn.Sequential(nn.Linear(POOLED_WIDTH, 32), nn.ReLU(inplace=True),
                         nn.Linear(32, NUM_TARGETS))


def sequence_digest(uids: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(uids), separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def train_head(
    *, pooled: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
    study_uids: Sequence[str], reference_indices: Sequence[int], weak_indices: Sequence[int],
    successful_steps: int, batch_size: int, initialization_seed: int, batch_order_seed: int,
    weak_seed_offset: int, learning_rate: float,
    checkpoint: Callable[[nn.Module, torch.optim.Optimizer, dict], None] | None = None,
    checkpoint_interval_steps: int = 15,
) -> tuple[nn.Module, dict]:
    """Fit final-step heads with matched gold exposure and optional report batches."""
    if (pooled.ndim != 2 or pooled.shape[1] != POOLED_WIDTH
            or targets.shape != (len(study_uids), NUM_TARGETS) or mask.shape != targets.shape
            or pooled.shape[0] != len(study_uids)):
        raise ValueError("head input shapes differ")
    if (len(set(study_uids)) != len(study_uids) or not reference_indices
            or successful_steps <= 0 or batch_size <= 0 or checkpoint_interval_steps <= 0
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("invalid head experiment configuration")
    for indices in [reference_indices, weak_indices]:
        if len(set(indices)) != len(indices) or any(index < 0 or index >= len(study_uids) for index in indices):
            raise ValueError("training partition indices differ")
    if set(reference_indices) & set(weak_indices):
        raise ValueError("gold and weak partitions overlap")
    if not (torch.isfinite(pooled).all() and torch.isfinite(targets).all()):
        raise ValueError("head input contains nonfinite values")
    if torch.any((targets[mask.bool()] < 0) | (targets[mask.bool()] > 1)):
        raise ValueError("supervised targets must be probabilities")
    if pooled.device.type != "cpu" or targets.device.type != "cpu" or mask.device.type != "cpu":
        raise ValueError("head experiment is CPU only")
    head = build_head(initialization_seed)
    initial_hash = _model_state_sha256(head)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    gold_sequence, weak_sequence, losses = [], [], []
    source_counts = {kind: {"sample_exposures": 0, "supervised_entry_exposures": 0}
                     for kind in ["reference", "weak"]}
    attempts = skipped = 0
    started = time.monotonic()
    state = {}
    while len(losses) < successful_steps:
        gold = [reference_indices[index] for index in
                _batch_indices(len(reference_indices), batch_size, attempts, batch_order_seed)]
        weak = [weak_indices[index] for index in
                _batch_indices(len(weak_indices), batch_size, attempts,
                               batch_order_seed + weak_seed_offset)] if weak_indices else []
        attempts += 1
        gold_sequence.extend(study_uids[index] for index in gold)
        weak_sequence.extend(study_uids[index] for index in weak)
        for kind, indices in [("reference", gold), ("weak", weak)]:
            source_counts[kind]["sample_exposures"] += len(indices)
            if indices:
                source_counts[kind]["supervised_entry_exposures"] += int(mask[indices].sum())
        indices = gold + weak
        optimizer.zero_grad(set_to_none=True)
        loss = masked_bce_loss(head(pooled[indices]), targets[indices], mask[indices])
        if loss is None:
            skipped += 1
            if skipped >= 100:
                raise ValueError("persistent entirely masked batches")
            continue
        if not torch.isfinite(loss):
            raise ValueError("head loss is nonfinite")
        loss.backward()
        if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
               for parameter in head.parameters()):
            raise ValueError("head gradients are missing or nonfinite")
        optimizer.step()
        losses.append(float(loss.detach()))
        state = {
            "head_id": HEAD_ID, "successful_optimizer_steps": len(losses),
            "attempted_batches": attempts, "skipped_batches": skipped,
            "initial_head_state_sha256": initial_hash,
            "gold_exposure_sequence_sha256": sequence_digest(gold_sequence),
            "weak_exposure_sequence_sha256": sequence_digest(weak_sequence),
            "distinct_weak_studies_exposed": len(set(weak_sequence)),
            "weak_training_pool_count": len(weak_indices), "exposures_by_source": source_counts,
            "loss_history": losses, "elapsed_seconds": time.monotonic() - started,
        }
        if checkpoint is not None and (len(losses) % checkpoint_interval_steps == 0
                                       or len(losses) == successful_steps):
            checkpoint(head, optimizer, state)
    head.eval()
    return head, state


def save_head_checkpoint(path: Path, head: nn.Module, optimizer: torch.optim.Optimizer,
                         state: dict, identity: dict, *, final: bool) -> None:
    """Save exact head/optimizer state, with explicit external-feature provenance."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".partial")
    if final and path.exists():
        raise FileExistsError(path)
    torch.save({"schema_version": "rsna-frozen-head-checkpoint-v1", "head_id": HEAD_ID,
                "head_state_dict": head.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "identity": identity, "counts": state,
                "checkpoint_kind": "final_selected" if final else "latest_resumable",
                "rng_state": torch.get_rng_state()}, temporary)
    if final:
        temporary.chmod(0o600)
        import os
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
    else:
        temporary.replace(path)


def reload_head(path: Path, expected_identity: dict) -> tuple[nn.Module, dict]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (value.get("schema_version") != "rsna-frozen-head-checkpoint-v1"
            or value.get("head_id") != HEAD_ID or value.get("identity") != expected_identity
            or value.get("checkpoint_kind") != "final_selected"):
        raise ValueError("head checkpoint contract differs")
    head = build_head(0)
    head.load_state_dict(value["head_state_dict"], strict=True)
    if any(not torch.isfinite(parameter).all() for parameter in head.parameters()):
        raise ValueError("head checkpoint contains nonfinite parameters")
    head.eval()
    return head, value["counts"]
