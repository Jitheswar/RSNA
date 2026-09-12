"""Reference-only training and experiment recording for the mechanics baseline.

The module deliberately owns the scientific training boundary: image inputs
are already represented by the shared ``StudyRecord``/packing interfaces,
labels come from a frozen reference fold manifest, and report-derived
supervision is not an accepted input.  The Slurm adapter is a thin caller of
the functions here; it does not contain a second training implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence, cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .archive import atomic_json_write
from .domain import StudyRecord
from .folds import FOLD_IDS, FoldManifest
from .model import HierarchicalStudyModel, ModelConfig
from .preprocessing import (
    PackedStudy,
    PackingConfig,
    PreprocessConfig,
    SeriesFailure,
    pack_study,
    prepare_study,
)
from .series_cache import SeriesCache
from .targets import NUM_TARGETS

if TYPE_CHECKING:
    from .evaluation import FoldEvaluation


TRAINING_SCHEMA_VERSION = "rsna-knee-training-v1"
LEGACY_CHECKPOINT_SCHEMA_VERSION = "rsna-knee-checkpoint-v1"
CHECKPOINT_SCHEMA_VERSION = "rsna-knee-checkpoint-v2"
PREFLIGHT_SCHEMA_VERSION = "rsna-knee-cpu-preflight-v1"
DEFAULT_TRAINING_SEED = 20260911
DEFAULT_BATCH_ORDER_SEED = 20260912
DEFAULT_SOURCE_VERSION = "synthetic-development"


class TrainingError(RuntimeError):
    """Base class for fixed-budget training failures."""


class TrainingConfigurationError(ValueError):
    """Raised when a scientific training configuration is unsafe or invalid."""


class TrainingDataError(ValueError):
    """Raised when a study or supervision input cannot be admitted."""


class TrainingProgressError(TrainingError):
    """Raised when a run cannot make supervised optimizer progress."""


class TrainingNumericsError(TrainingError):
    """Raised when loss, gradients, or parameters become non-finite."""


class CheckpointError(TrainingError):
    """Raised when a checkpoint cannot be safely written or reloaded."""


class TrainingInterrupted(TrainingError):
    """Raised by the deterministic test hook after a durable checkpoint."""

    def __init__(self, message: str, *, checkpoint_path: Path | None = None) -> None:
        super().__init__(message)
        self.checkpoint_path = checkpoint_path


# Capability returned only to the approval-gated GPU orchestration module.
_GPU_EXECUTION_CAPABILITY = object()


def _approved_gpu_execution_capability() -> object:
    return _GPU_EXECUTION_CAPABILITY


def _require_gpu_slurm_context() -> None:
    """Keep CUDA training inside the scheduler-owned execution boundary."""

    if not os.environ.get("SLURM_JOB_ID", "").strip():
        raise TrainingConfigurationError(
            "CUDA training requires an active Slurm job; use the approval-gated GPU runner"
        )


@dataclass(frozen=True)
class RetryClassification:
    """Conservative classification for the single allowed infrastructure retry."""

    retryable: bool
    category: str
    reason: str
    retry_number_allowed: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "retryable": self.retryable,
            "category": self.category,
            "reason": self.reason,
            "retry_number_allowed": self.retry_number_allowed,
        }


def classify_retry_failure(error: BaseException | str) -> RetryClassification:
    """Classify failures without automatically submitting another allocation.

    Only explicit infrastructure/interruption signals qualify for the one
    identical retry described by the project specification.  User cancellation,
    resource exhaustion, data/configuration errors, and numerical failures are
    deliberately non-retryable.
    """

    message = str(error).lower()
    if isinstance(
        error,
        (
            TrainingConfigurationError,
            TrainingDataError,
            TrainingProgressError,
            TrainingNumericsError,
            CheckpointError,
        ),
    ):
        return RetryClassification(False, "deterministic_or_semantic", str(error))
    non_retryable_tokens = (
        "out of memory",
        "oom",
        "user cancel",
        "cancelled by user",
        "canceled by user",
        "nan",
        "non-finite",
        "incompatible",
        "deterministic loader",
    )
    if any(token in message for token in non_retryable_tokens):
        return RetryClassification(False, "deterministic_or_resource", str(error))
    transient_tokens = (
        "node failure",
        "externally interrupted",
        "preempt",
        "gpu reset",
        "filesystem",
        "nfs",
        "network",
        "connection reset",
        "temporarily unavailable",
        "slurmstepd",
    )
    if isinstance(error, TrainingInterrupted) or any(token in message for token in transient_tokens):
        return RetryClassification(True, "transient_infrastructure", str(error))
    return RetryClassification(False, "unknown", str(error))


def retry_is_allowed(error: BaseException | str, *, previous_retries: int = 0) -> bool:
    """Return whether one identical retry remains permitted for ``error``."""

    if previous_retries < 0:
        raise ValueError("previous_retries cannot be negative")
    classification = classify_retry_failure(error)
    return classification.retryable and previous_retries < classification.retry_number_allowed


class PreflightError(TrainingError):
    """Raised when a real training run has not passed the CPU preflight."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainingConfig:
    """Scientific and bounded execution defaults for one mechanics run."""

    image_size: tuple[int, int] = (128, 128)
    channels: tuple[int, int, int] = (8, 16, 32)
    input_channels: int = 1
    max_slices: int = 8
    max_series: int = 16
    batch_size: int = 2
    successful_optimizer_steps: int = 100
    optimizer: str = "Adam"
    learning_rate: float = 0.001
    schedule: str = "constant"
    seed: int = DEFAULT_TRAINING_SEED
    batch_order_seed: int = DEFAULT_BATCH_ORDER_SEED
    persistent_no_progress_limit: int = 100
    checkpoint_interval_steps: int = 15
    checkpoint_interval_seconds: float = 15.0 * 60.0
    loss: str = "masked_bce"
    augmentation: str = "none"
    device: str = "cpu"
    selection_policy: str = "final_checkpoint"
    preprocessing_config: PreprocessConfig | None = None

    def __post_init__(self) -> None:
        if len(self.image_size) != 2 or any(size <= 0 for size in self.image_size):
            raise TrainingConfigurationError("image_size must contain two positive dimensions")
        if len(self.channels) != 3 or any(channel <= 0 for channel in self.channels):
            raise TrainingConfigurationError("channels must contain three positive widths")
        if self.input_channels != 1:
            raise TrainingConfigurationError("the mechanics baseline uses one grayscale channel")
        if self.max_slices <= 0 or self.max_series <= 0:
            raise TrainingConfigurationError("max_slices and max_series must be positive")
        if self.batch_size != 2:
            raise TrainingConfigurationError("the mechanics baseline requires batch_size=2")
        if self.successful_optimizer_steps <= 0:
            raise TrainingConfigurationError("successful_optimizer_steps must be positive")
        if self.optimizer != "Adam":
            raise TrainingConfigurationError("the mechanics baseline requires Adam")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise TrainingConfigurationError("learning_rate must be a positive finite value")
        if self.schedule != "constant":
            raise TrainingConfigurationError("the mechanics baseline requires a constant schedule")
        if self.persistent_no_progress_limit <= 0:
            raise TrainingConfigurationError("persistent_no_progress_limit must be positive")
        if self.checkpoint_interval_steps <= 0:
            raise TrainingConfigurationError("checkpoint_interval_steps must be positive")
        if not np.isfinite(self.checkpoint_interval_seconds) or self.checkpoint_interval_seconds <= 0:
            raise TrainingConfigurationError(
                "checkpoint_interval_seconds must be a positive finite value"
            )
        if self.loss != "masked_bce":
            raise TrainingConfigurationError("the mechanics baseline requires masked_bce loss")
        if self.augmentation != "none":
            raise TrainingConfigurationError("the mechanics baseline has no augmentation")
        if not self.device.strip():
            raise TrainingConfigurationError("device must be non-empty")
        if self.selection_policy != "final_checkpoint":
            raise TrainingConfigurationError("the mechanics baseline evaluates the final checkpoint")
        if self.preprocessing_config is not None:
            if not isinstance(self.preprocessing_config, PreprocessConfig):
                raise TrainingConfigurationError("preprocessing_config must be a PreprocessConfig")
            if self.preprocessing_config.image_size != self.image_size:
                raise TrainingConfigurationError(
                    "preprocessing and model image sizes must agree"
                )
            if self.preprocessing_config.max_slices != self.max_slices:
                raise TrainingConfigurationError(
                    "preprocessing and packing max_slices must agree"
                )
            if self.preprocessing_config.max_series != self.max_series:
                raise TrainingConfigurationError(
                    "preprocessing and packing max_series must agree"
                )

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            image_size=self.image_size,
            channels=self.channels,
            input_channels=self.input_channels,
        )

    def preprocess_config(self) -> PreprocessConfig:
        if self.preprocessing_config is not None:
            return self.preprocessing_config
        return PreprocessConfig(
            image_size=self.image_size,
            max_slices=self.max_slices,
            max_series=self.max_series,
        )

    def packing_config(self) -> PackingConfig:
        return PackingConfig(max_slices=self.max_slices, max_series=self.max_series)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "model": {
                "image_size": list(self.image_size),
                "channels": list(self.channels),
                "input_channels": self.input_channels,
                "num_targets": NUM_TARGETS,
            },
            "packing": self.packing_config().to_dict(),
            "preprocessing": self.preprocess_config().cache_config(),
            "optimization": {
                "batch_size": self.batch_size,
                "successful_optimizer_steps": self.successful_optimizer_steps,
                "optimizer": self.optimizer,
                "learning_rate": self.learning_rate,
                "schedule": self.schedule,
            },
            "seeds": {
                "initialization": self.seed,
                "batch_order": self.batch_order_seed,
            },
            "progress": {
                "persistent_no_progress_limit": self.persistent_no_progress_limit,
                "checkpoint_interval_steps": self.checkpoint_interval_steps,
                "checkpoint_interval_seconds": self.checkpoint_interval_seconds,
            },
            "loss": self.loss,
            "augmentation": self.augmentation,
            "device": self.device,
            "selection_policy": self.selection_policy,
        }

    def scientific_dict(self) -> dict[str, object]:
        """Return fields whose changes require a new scientific run identity."""

        payload = self.to_dict()
        progress = payload.get("progress")
        if isinstance(progress, dict):
            progress = dict(progress)
            progress.pop("checkpoint_interval_steps", None)
            progress.pop("persistent_no_progress_limit", None)
            progress.pop("checkpoint_interval_seconds", None)
            payload["progress"] = progress
        payload.pop("device", None)
        return payload

    @property
    def config_fingerprint(self) -> str:
        return _json_sha256(self.scientific_dict())


_OPERATIONAL_IDENTITY_KEYS = frozenset(
    {
        "attempt_id",
        "execution_attempt_id",
        "parent_attempt_id",
        "started_at",
        "finished_at",
        "timestamp",
        "hostname",
        "environment",
        "resources",
        "slurm_attempt_id",
        "slurm_job_id",
        "slurm_job_name",
        "slurm_partition",
        "slurm_node_list",
        "slurm_cpus_per_task",
        "slurm_mem_per_node",
        "slurm_mem_per_cpu",
        "slurm_submit_dir",
        "logs",
        "checkpoint_interval_steps",
        "checkpoint_interval_seconds",
    }
)


def _scientific_identity_value(value: object, *, key: str | None = None) -> object:
    """Remove execution-only metadata before hashing scientific identity."""

    if key is not None and key in _OPERATIONAL_IDENTITY_KEYS:
        return None
    if isinstance(value, Mapping):
        cleaned: dict[str, object] = {}
        for child_key, child_value in sorted(value.items(), key=lambda item: str(item[0])):
            child_name = str(child_key)
            if (
                child_name in _OPERATIONAL_IDENTITY_KEYS
                or child_name.startswith("parent_")
                or child_name == "semantic_fingerprint"
            ):
                continue
            cleaned[child_name] = _scientific_identity_value(child_value, key=child_name)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_scientific_identity_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def semantic_fingerprint(
    config: TrainingConfig,
    identity: Mapping[str, object],
) -> str:
    """Hash the complete scientific identity used to decide resume compatibility.

    Operational details such as the Slurm allocation, host, execution attempt,
    and checkpoint cadence are deliberately excluded.  Data, split, labels,
    preprocessing, architecture, initialization, seeds, optimization, loss,
    augmentation, and budget remain in the hash through the configuration and
    identity payload.
    """

    payload = {
        "configuration": _scientific_identity_value(config.scientific_dict()),
        "identity": _scientific_identity_value(dict(identity)),
    }
    return _json_sha256(payload)


@dataclass(frozen=True)
class SupervisionProvenance:
    """Provenance for one target vector admitted to mechanics training."""

    source_kind: str = "reference"
    source_id: str = "frozen-fold-manifest"
    fold: int | None = None
    held_out_excluded: bool = True
    eligible_study_uids: tuple[str, ...] = ()
    excluded_study_uids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source_kind != "reference":
            raise TrainingDataError("ticket 08 accepts reference supervision only")
        if not self.source_id.strip():
            raise TrainingDataError("supervision provenance requires a source identity")
        if self.fold is not None and self.fold not in FOLD_IDS:
            raise TrainingDataError(f"supervision fold must be one of {FOLD_IDS}")
        if not self.held_out_excluded:
            raise TrainingDataError("held-out studies must be excluded from supervision provenance")
        if len(set(self.eligible_study_uids)) != len(self.eligible_study_uids):
            raise TrainingDataError("eligible supervision studies must be unique")
        if len(set(self.excluded_study_uids)) != len(self.excluded_study_uids):
            raise TrainingDataError("excluded supervision studies must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "fold": self.fold,
            "held_out_excluded": self.held_out_excluded,
            "eligible_study_uids": list(self.eligible_study_uids),
            "excluded_study_uids": list(self.excluded_study_uids),
        }


@dataclass(frozen=True)
class TrainingExample:
    """One fixed-shape packed study and its reference target entries."""

    study_instance_uid: str
    images: np.ndarray
    slice_mask: np.ndarray
    series_mask: np.ndarray
    targets: np.ndarray
    supervision_mask: np.ndarray
    provenance: SupervisionProvenance = SupervisionProvenance()
    rejected_series: tuple[SeriesFailure, ...] = ()

    def __post_init__(self) -> None:
        if not self.study_instance_uid.strip():
            raise TrainingDataError("training examples require a study identifier")
        images = np.asarray(self.images)
        slice_mask = np.asarray(self.slice_mask)
        series_mask = np.asarray(self.series_mask)
        targets = np.asarray(self.targets)
        supervision_mask = np.asarray(self.supervision_mask)
        if images.ndim != 5 or images.shape[2] != 1:
            raise TrainingDataError(
                "training images must have shape (series, slices, 1, height, width)"
            )
        if slice_mask.shape != images.shape[:2] or series_mask.shape != (images.shape[0],):
            raise TrainingDataError("training image and mask shapes do not agree")
        if targets.shape != (NUM_TARGETS,) or supervision_mask.shape != (NUM_TARGETS,):
            raise TrainingDataError(f"training targets must have shape ({NUM_TARGETS},)")
        if images.dtype.kind not in "fiu" or not np.isfinite(images).all():
            raise TrainingDataError("training images must be finite numeric arrays")
        if not np.asarray(slice_mask, dtype=bool).any() or not np.asarray(series_mask, dtype=bool).any():
            raise TrainingDataError("training examples must contain at least one valid slice and series")
        supervised = np.asarray(supervision_mask, dtype=bool)
        if not np.isfinite(targets[supervised]).all():
            raise TrainingDataError("supervised target entries must be finite")
        if np.any((targets[supervised] < 0.0) | (targets[supervised] > 1.0)):
            raise TrainingDataError("supervised target entries must be binary values")
        if self.provenance.source_kind != "reference":
            raise TrainingDataError("ticket 08 accepts reference supervision only")

    @classmethod
    def from_packed(
        cls,
        packed: PackedStudy,
        targets: Sequence[float | int | None],
        *,
        provenance: SupervisionProvenance = SupervisionProvenance(),
        rejected_series: Sequence[SeriesFailure] = (),
    ) -> "TrainingExample":
        if len(targets) != NUM_TARGETS:
            raise TrainingDataError(f"targets must contain {NUM_TARGETS} entries")
        target_array = np.asarray(
            [0.0 if value is None else float(value) for value in targets],
            dtype=np.float32,
        )
        mask = np.asarray([value is not None for value in targets], dtype=bool)
        return cls(
            study_instance_uid=packed.study_instance_uid,
            images=np.asarray(packed.images, dtype=np.float32),
            slice_mask=np.asarray(packed.slice_mask, dtype=bool),
            series_mask=np.asarray(packed.series_mask, dtype=bool),
            targets=target_array,
            supervision_mask=mask,
            provenance=provenance,
            rejected_series=tuple(rejected_series),
        )


@dataclass(frozen=True)
class CpuPreflightResult:
    """Machine-readable result of the bounded CPU training preflight."""

    passed: bool
    checks: Mapping[str, bool]
    environment: Mapping[str, object]
    details: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "passed": self.passed,
            "checks": dict(self.checks),
            "environment": dict(self.environment),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class TrainingResult:
    """Fixed-budget outcome and, when requested, held-out evaluation."""

    model: HierarchicalStudyModel
    config: TrainingConfig
    successful_optimizer_steps: int
    scheduler_steps: int
    attempted_batches: int
    skipped_batches: int
    loss_history: tuple[float, ...]
    final_checkpoint_path: Path | None = None
    latest_checkpoint_path: Path | None = None
    run_id: str | None = None
    fold: int | None = None
    evaluation: FoldEvaluation | None = None
    predictions: Mapping[str, np.ndarray] = field(default_factory=dict)
    training_study_uids: tuple[str, ...] = ()
    held_out_study_uids: tuple[str, ...] = ()
    data_fingerprint: str | None = None
    best_metric: float | None = None
    epoch: int = 0
    execution_attempt_id: str | None = None
    resumed_from_checkpoint: str | None = None
    replayed_batches: int = 0
    checkpoint_events: tuple[Mapping[str, object], ...] = ()
    semantic_fingerprint_value: str | None = None

    @property
    def final_loss(self) -> float | None:
        return self.loss_history[-1] if self.loss_history else None

    def to_public_dict(self) -> dict[str, object]:
        evaluation: object = None
        if self.evaluation is not None:
            evaluation = self.evaluation.to_public_dict()
        return {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "status": "completed_fixed_budget_training",
            "run_id": self.run_id,
            "fold": self.fold,
            "config_fingerprint": self.config.config_fingerprint,
            "semantic_fingerprint": self.semantic_fingerprint_value,
            "config": self.config.to_dict(),
            "counts": {
                "training_studies": len(self.training_study_uids),
                "held_out_studies": len(self.held_out_study_uids),
                "successful_optimizer_steps": self.successful_optimizer_steps,
                "scheduler_steps": self.scheduler_steps,
                "attempted_batches": self.attempted_batches,
                "skipped_all_masked_batches": self.skipped_batches,
                "supervised_entries_per_step": NUM_TARGETS,
            },
            "loss": {
                "first": self.loss_history[0] if self.loss_history else None,
                "final": self.final_loss,
                "finite": bool(np.isfinite(self.loss_history).all()) if self.loss_history else False,
            },
            "best_metric": self.best_metric,
            "checkpoint": {
                "final_present": self.final_checkpoint_path is not None,
                "latest_present": self.latest_checkpoint_path is not None,
                "final_sha256": (
                    checkpoint_sha256(self.final_checkpoint_path)
                    if self.final_checkpoint_path is not None and self.final_checkpoint_path.exists()
                    else None
                ),
            },
            "execution": {
                "attempt_id": self.execution_attempt_id,
                "resumed_from_checkpoint": self.resumed_from_checkpoint is not None,
                "replayed_batches": self.replayed_batches,
                "checkpoint_events": len(self.checkpoint_events),
            },
            "evaluation": evaluation,
            "reference_supervision_only": True,
            "held_out_inputs_excluded_from_fitting": True,
            "learned_normalization": False,
        }

    def to_private_dict(self) -> dict[str, object]:
        payload = self.to_public_dict()
        payload["training_study_uids"] = list(self.training_study_uids)
        payload["held_out_study_uids"] = list(self.held_out_study_uids)
        payload["data_fingerprint"] = self.data_fingerprint
        payload["best_metric"] = self.best_metric
        payload["epoch"] = self.epoch
        payload["execution_attempt_id"] = self.execution_attempt_id
        payload["resumed_from_checkpoint"] = self.resumed_from_checkpoint
        payload["replayed_batches"] = self.replayed_batches
        payload["checkpoint_events"] = [dict(event) for event in self.checkpoint_events]
        payload["predictions"] = {
            uid: np.asarray(scores, dtype=float).tolist() for uid, scores in self.predictions.items()
        }
        return payload


def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Return BCE averaged over supervised entries, or ``None`` when empty."""

    if logits.shape != targets.shape or logits.shape != supervision_mask.shape:
        raise ValueError("logits, targets, and supervision_mask must have identical shapes")
    if logits.ndim != 2 or logits.shape[1] != NUM_TARGETS:
        raise ValueError(f"training tensors must have shape (batch, {NUM_TARGETS})")
    mask = supervision_mask.to(dtype=torch.bool, device=logits.device)
    if not bool(mask.any()):
        return None
    selected_logits = logits[mask]
    selected_targets = targets.to(dtype=logits.dtype, device=logits.device)[mask]
    return F.binary_cross_entropy_with_logits(selected_logits, selected_targets, reduction="mean")


masked_binary_cross_entropy_with_logits = masked_bce_loss


def seed_everything(seed: int) -> None:
    """Set every local RNG used by the mechanics run."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fingerprint_training_examples(examples: Sequence[TrainingExample]) -> str:
    """Fingerprint packed pixels, masks, reference labels, and provenance."""

    rows: list[dict[str, object]] = []
    for example in sorted(examples, key=lambda item: item.study_instance_uid):
        rows.append(
            {
                "study_instance_uid": example.study_instance_uid,
                "images": _array_sha256(example.images),
                "slice_mask": _array_sha256(np.asarray(example.slice_mask, dtype=np.uint8)),
                "series_mask": _array_sha256(np.asarray(example.series_mask, dtype=np.uint8)),
                "targets": _array_sha256(np.asarray(example.targets, dtype=np.float32)),
                "supervision_mask": _array_sha256(
                    np.asarray(example.supervision_mask, dtype=np.uint8)
                ),
                "provenance": example.provenance.to_dict(),
            }
        )
    return _json_sha256(rows)


def _study_mapping(studies: Mapping[str, StudyRecord] | Sequence[StudyRecord]) -> dict[str, StudyRecord]:
    if isinstance(studies, Mapping):
        values = dict(studies)
    else:
        values = {}
        for study in studies:
            if study.study_instance_uid in values:
                raise TrainingDataError(f"duplicate study input {study.study_instance_uid}")
            values[study.study_instance_uid] = study
    if any(key != study.study_instance_uid for key, study in values.items()):
        raise TrainingDataError("study mapping keys must match StudyInstanceUID values")
    return values


def build_fold_examples(
    studies: Mapping[str, StudyRecord] | Sequence[StudyRecord],
    fold_manifest: FoldManifest,
    fold: int,
    *,
    partition: str = "training",
    preprocess_config: PreprocessConfig = PreprocessConfig(),
    packing_config: PackingConfig | None = None,
    series_cache: SeriesCache | None = None,
    allow_partial_inputs: bool = False,
) -> tuple[TrainingExample, ...]:
    """Prepare only gold studies from one declared training or held-out partition.

    Preprocessing is fixed and series-local; no statistics are fitted from the
    selected studies.  Report-only memberships and arbitrary label mappings
    cannot enter this ticket's training path.
    """

    if fold not in FOLD_IDS:
        raise TrainingDataError(f"fold must be one of {FOLD_IDS}")
    if partition not in {"training", "held_out"}:
        raise TrainingDataError("partition must be 'training' or 'held_out'")
    source = _study_mapping(studies)
    memberships = tuple(
        item
        for item in fold_manifest.gold_memberships
        if (item.fold != fold if partition == "training" else item.fold == fold)
    )
    if not memberships:
        raise TrainingDataError(f"fold {fold} has no gold studies in the {partition} partition")
    selected_uids = tuple(sorted(item.study_instance_uid for item in memberships))
    missing = sorted(set(selected_uids) - set(source))
    if missing:
        raise TrainingDataError(f"{partition} study inputs are incomplete; missing {missing[:3]}")
    if partition == "training":
        fold_manifest.assert_adaptation_input_is_training_only(fold, selected_uids)
    membership_by_uid = {item.study_instance_uid: item for item in memberships}
    held_out_uids = tuple(sorted(fold_manifest.held_out_study_uids(fold)))
    provenance = SupervisionProvenance(
        source_kind="reference",
        source_id=f"fold-manifest:{fold_manifest.manifest_checksum}",
        fold=fold,
        held_out_excluded=True,
        eligible_study_uids=selected_uids,
        excluded_study_uids=held_out_uids,
    )
    packing = packing_config or preprocess_config.packing_config()
    examples: list[TrainingExample] = []
    for study_uid in selected_uids:
        study = source[study_uid]
        prepared = (
            series_cache.prepare_study(study, preprocess_config)
            if series_cache is not None
            else prepare_study(study, preprocess_config)
        )
        if prepared.failures and not allow_partial_inputs:
            details = "; ".join(
                f"{failure.series_instance_uid}: {failure.reason}" for failure in prepared.failures[:3]
            )
            raise TrainingDataError(
                f"fold {fold} {partition} study {study_uid} has rejected input series: {details}"
            )
        if not prepared.series:
            raise TrainingDataError(
                f"fold {fold} {partition} study {study_uid} has no usable image series"
            )
        packed = pack_study(study, prepared.series, preprocess_config, packing_config=packing)
        examples.append(
            TrainingExample.from_packed(
                packed,
                membership_by_uid[study_uid].reference_labels,
                provenance=provenance,
                rejected_series=prepared.failures,
            )
        )
    return tuple(examples)


def build_fold_training_examples(
    studies: Mapping[str, StudyRecord] | Sequence[StudyRecord],
    fold_manifest: FoldManifest,
    fold: int,
    *,
    preprocess_config: PreprocessConfig = PreprocessConfig(),
    packing_config: PackingConfig | None = None,
    series_cache: SeriesCache | None = None,
    allow_partial_inputs: bool = False,
) -> tuple[TrainingExample, ...]:
    """Named convenience wrapper for the fitting partition."""

    return build_fold_examples(
        studies,
        fold_manifest,
        fold,
        partition="training",
        preprocess_config=preprocess_config,
        packing_config=packing_config,
        series_cache=series_cache,
        allow_partial_inputs=allow_partial_inputs,
    )


def build_fold_held_out_examples(
    studies: Mapping[str, StudyRecord] | Sequence[StudyRecord],
    fold_manifest: FoldManifest,
    fold: int,
    *,
    preprocess_config: PreprocessConfig = PreprocessConfig(),
    packing_config: PackingConfig | None = None,
    series_cache: SeriesCache | None = None,
    allow_partial_inputs: bool = False,
) -> tuple[TrainingExample, ...]:
    """Named convenience wrapper for final-checkpoint evaluation inputs."""

    return build_fold_examples(
        studies,
        fold_manifest,
        fold,
        partition="held_out",
        preprocess_config=preprocess_config,
        packing_config=packing_config,
        series_cache=series_cache,
        allow_partial_inputs=allow_partial_inputs,
    )


def _validate_examples(examples: Sequence[TrainingExample], config: TrainingConfig) -> None:
    if not examples:
        raise TrainingDataError("at least one training example is required")
    expected_shape = (
        config.max_series,
        config.max_slices,
        config.input_channels,
        config.image_size[0],
        config.image_size[1],
    )
    seen: set[str] = set()
    for example in examples:
        if example.study_instance_uid in seen:
            raise TrainingDataError(f"duplicate training study {example.study_instance_uid}")
        seen.add(example.study_instance_uid)
        if tuple(example.images.shape) != expected_shape:
            raise TrainingDataError(
                f"study {example.study_instance_uid} has shape {example.images.shape}; expected {expected_shape}"
            )
        if example.provenance.source_kind != "reference":
            raise TrainingDataError("ticket 08 accepts reference supervision only")
        if not example.provenance.held_out_excluded:
            raise TrainingDataError("training provenance does not exclude held-out inputs")


def _rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, object]) -> None:
    """Restore all RNG streams captured by a checkpoint."""

    try:
        random.setstate(cast(tuple[Any, ...], state["python"]))
        np.random.set_state(cast(tuple[Any, ...], state["numpy"]))
        torch_state = state["torch"]
        torch.set_rng_state(torch.as_tensor(torch_state, dtype=torch.uint8, device="cpu"))
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [
                    torch.as_tensor(item, dtype=torch.uint8, device="cpu")
                    for item in cast(Sequence[Any], cuda_state)
                ]
            )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise CheckpointError("checkpoint RNG state is invalid") from exc


def atomic_torch_save(
    path: str | Path,
    payload: object,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a torch artifact atomically, leaving an old complete file intact on failure."""

    destination = Path(path)
    if not overwrite and (destination.exists() or destination.is_symlink()):
        raise FileExistsError(f"refusing to overwrite protected checkpoint {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except BaseException as exc:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        if isinstance(exc, (OSError, RuntimeError, TypeError, ValueError)):
            raise CheckpointError(f"could not publish checkpoint {destination}") from exc
        raise
    return destination


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_training_checkpoint(
    path: str | Path,
    model: HierarchicalStudyModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainingConfig,
    *,
    successful_optimizer_steps: int,
    attempted_batches: int,
    skipped_batches: int,
    loss_history: Sequence[float],
    fold: int | None = None,
    run_id: str | None = None,
    identity: Mapping[str, object] | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
    batch_in_epoch: int = 0,
    next_batch_index: int | None = None,
    best_metric: float | None = None,
    scaler: Any | None = None,
    amp_scaler: Any | None = None,
    checkpoint_kind: str = "latest_resumable",
    checkpoint_reason: str = "manual",
    checkpoint_timing: Mapping[str, object] | None = None,
    checkpoint_events: Sequence[Mapping[str, object]] = (),
    sampler_state: Mapping[str, object] | None = None,
    recovery: Mapping[str, object] | None = None,
    parent_provenance: Mapping[str, object] | None = None,
    execution_attempt_id: str | None = None,
    scheduler_steps: int | None = None,
    consecutive_no_progress: int = 0,
    overwrite: bool = True,
) -> Path:
    """Persist a complete training state atomically.

    ``latest_resumable`` is the only checkpoint kind accepted by the resume
    path.  ``final_selected`` is intentionally immutable and records the
    selected final state separately from the operational recovery pointer.
    """

    if successful_optimizer_steps < 0 or attempted_batches < successful_optimizer_steps:
        raise CheckpointError("checkpoint progress counters are inconsistent")
    if fold is not None and fold not in FOLD_IDS:
        raise CheckpointError(f"checkpoint fold must be one of {FOLD_IDS}")
    if checkpoint_kind not in {"latest_resumable", "final_selected"}:
        raise CheckpointError("checkpoint kind must be latest_resumable or final_selected")
    if checkpoint_kind == "final_selected":
        # A selected result is evidence, not an operational pointer.  Enforce
        # this here so callers cannot accidentally replace it by passing
        # overwrite=True.
        overwrite = False
    if epoch is not None and epoch < 0:
        raise CheckpointError("checkpoint epoch cannot be negative")
    if batch_in_epoch < 0:
        raise CheckpointError("checkpoint batch_in_epoch cannot be negative")
    if next_batch_index is not None and next_batch_index < 0:
        raise CheckpointError("checkpoint next_batch_index cannot be negative")
    if global_step is not None and global_step < 0:
        raise CheckpointError("checkpoint global_step cannot be negative")
    if scheduler_steps is None:
        scheduler_steps = successful_optimizer_steps
    if scheduler_steps < 0:
        raise CheckpointError("checkpoint scheduler steps cannot be negative")
    if consecutive_no_progress < 0:
        raise CheckpointError("checkpoint consecutive no-progress count cannot be negative")
    if best_metric is not None and not np.isfinite(best_metric):
        raise CheckpointError("checkpoint best metric must be finite when present")
    if scaler is not None and amp_scaler is not None:
        raise CheckpointError("pass only one of scaler or amp_scaler")
    if amp_scaler is not None:
        scaler = amp_scaler
    losses = tuple(loss_history)
    if not np.isfinite(np.asarray(losses, dtype=np.float64)).all():
        raise CheckpointError("checkpoint loss history contains non-finite values")
    identity_payload = dict(identity or {})
    scaler_state: object = None
    if scaler is not None:
        state_dict = getattr(scaler, "state_dict", None)
        if not callable(state_dict):
            raise CheckpointError("AMP scaler does not expose state_dict()")
        scaler_state = state_dict()
        if not isinstance(scaler_state, Mapping):
            raise CheckpointError("AMP scaler state is not a mapping")
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_reason": checkpoint_reason,
        "config": config.to_dict(),
        "config_fingerprint": config.config_fingerprint,
        "semantic_fingerprint": semantic_fingerprint(config, identity_payload),
        "identity": identity_payload,
        "fold": fold,
        "run_id": run_id,
        "execution_attempt_id": execution_attempt_id,
        "epoch": 0 if epoch is None else epoch,
        "global_step": successful_optimizer_steps if global_step is None else global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "amp_scaler_state_dict": dict(scaler_state) if isinstance(scaler_state, Mapping) else None,
        "scaler_state_dict": dict(scaler_state) if isinstance(scaler_state, Mapping) else None,
        "progress": {
            "epoch": 0 if epoch is None else epoch,
            "global_step": successful_optimizer_steps if global_step is None else global_step,
            "successful_optimizer_steps": successful_optimizer_steps,
            "scheduler_steps": scheduler_steps,
            "attempted_batches": attempted_batches,
            "skipped_all_masked_batches": skipped_batches,
            "loss_history": list(losses),
            "batch_in_epoch": batch_in_epoch,
            "next_batch_index": (
                attempted_batches if next_batch_index is None else next_batch_index
            ),
            "consecutive_no_progress": consecutive_no_progress,
            "best_metric": best_metric,
        },
        "rng_state": _rng_state(),
        "best_metric": best_metric,
        "checkpoint_timing": dict(checkpoint_timing or {}),
        "checkpoint_events": [dict(event) for event in checkpoint_events],
        "sampler_state": dict(sampler_state or {}),
        "recovery": dict(recovery or {}),
        "parent_provenance": dict(parent_provenance or {}),
        "selection_policy": config.selection_policy,
    }
    return atomic_torch_save(path, payload, overwrite=overwrite)


def _legacy_scientific_config_payload(
    config_payload: Mapping[str, object],
) -> dict[str, object]:
    """Recreate the ticket-08 v1 scientific configuration representation."""

    payload = dict(config_payload)
    progress = payload.get("progress")
    if isinstance(progress, Mapping):
        progress_payload = dict(progress)
        progress_payload.pop("checkpoint_interval_steps", None)
        progress_payload.pop("checkpoint_interval_seconds", None)
        progress_payload.pop("persistent_no_progress_limit", None)
        payload["progress"] = progress_payload
    else:
        payload["progress"] = {}
    payload.pop("loss", None)
    payload.pop("augmentation", None)
    payload.pop("device", None)
    return payload


def _legacy_config_fingerprint(config: TrainingConfig) -> str:
    return _json_sha256(_legacy_scientific_config_payload(config.to_dict()))


def _legacy_config_fingerprint_from_payload(
    config_payload: Mapping[str, object],
) -> str | None:
    try:
        return _json_sha256(_legacy_scientific_config_payload(config_payload))
    except (TypeError, ValueError):
        return None


def _canonical_identity_equal(left: object, right: object) -> bool:
    try:
        return _canonical_json(_scientific_identity_value(left)) == _canonical_json(
            _scientific_identity_value(right)
        )
    except (TypeError, ValueError):
        return False


def _legacy_identity_matches(
    legacy_identity: Mapping[str, object],
    expected_identity: Mapping[str, object],
    *,
    expected_config: TrainingConfig,
    legacy_config_fingerprint: str,
) -> bool:
    """Compare every identity field recorded by the v1 checkpoint."""

    recorded_config_fingerprint = legacy_identity.get("configuration_fingerprint")
    if (
        recorded_config_fingerprint is not None
        and recorded_config_fingerprint != legacy_config_fingerprint
    ):
        return False
    for key, value in legacy_identity.items():
        if key == "configuration_fingerprint":
            continue
        if key not in expected_identity or not _canonical_identity_equal(
            value, expected_identity[key]
        ):
            return False
    return expected_config.loss == "masked_bce" and expected_config.augmentation == "none"


def _legacy_configuration_matches(
    saved_configuration: Mapping[str, object],
    *,
    expected_config: TrainingConfig,
    expected_identity: Mapping[str, object],
) -> bool:
    """Validate an immutable ticket-08 configuration before v2 migration."""

    saved_config = saved_configuration.get("config")
    saved_fingerprint = saved_configuration.get("scientific_configuration_fingerprint")
    if not isinstance(saved_config, Mapping) or not isinstance(saved_fingerprint, str):
        return False
    computed_fingerprint = _legacy_config_fingerprint_from_payload(saved_config)
    if computed_fingerprint != saved_fingerprint or saved_fingerprint != _legacy_config_fingerprint(
        expected_config
    ):
        return False
    saved_identity = saved_configuration.get("identity")
    return isinstance(saved_identity, Mapping) and _legacy_identity_matches(
        saved_identity,
        expected_identity,
        expected_config=expected_config,
        legacy_config_fingerprint=saved_fingerprint,
    )


def _migrate_legacy_checkpoint(
    payload: Mapping[str, object],
    *,
    source: Path,
    expected_config: TrainingConfig | None,
    expected_identity: Mapping[str, object] | None,
) -> dict[str, object]:
    """Normalize a complete ticket-08 v1 checkpoint to the v2 load shape."""

    checkpoint_config = payload.get("config")
    checkpoint_fingerprint = payload.get("config_fingerprint")
    identity = payload.get("identity")
    progress = payload.get("progress")
    if not isinstance(checkpoint_config, Mapping) or not isinstance(checkpoint_fingerprint, str):
        raise CheckpointError("legacy checkpoint configuration metadata is incomplete")
    if not isinstance(identity, Mapping) or not isinstance(progress, dict):
        raise CheckpointError("legacy checkpoint identity or progress metadata is incomplete")
    migrated = dict(payload)
    migrated["migrated_from_schema_version"] = LEGACY_CHECKPOINT_SCHEMA_VERSION
    migrated["legacy_config"] = dict(checkpoint_config)
    migrated["legacy_config_fingerprint"] = checkpoint_fingerprint
    migrated["schema_version"] = CHECKPOINT_SCHEMA_VERSION
    migrated["checkpoint_version"] = CHECKPOINT_SCHEMA_VERSION
    migrated["checkpoint_kind"] = (
        "final_selected" if source.name in {"final.pt", "final_selected.pt"} else "latest_resumable"
    )
    migrated["checkpoint_reason"] = "legacy_v1_migrated"

    migrated_progress = dict(progress)
    successful_steps = migrated_progress.get("successful_optimizer_steps")
    attempted_batches = migrated_progress.get("attempted_batches")
    if not isinstance(successful_steps, int) or not isinstance(attempted_batches, int):
        raise CheckpointError("legacy checkpoint progress metadata is incomplete")
    migrated_progress.setdefault("scheduler_steps", successful_steps)
    migrated_progress.setdefault("batch_in_epoch", 0)
    migrated_progress.setdefault("next_batch_index", attempted_batches)
    migrated_progress.setdefault("consecutive_no_progress", 0)
    migrated_progress.setdefault("best_metric", None)
    migrated["progress"] = migrated_progress
    migrated["epoch"] = migrated_progress.get("epoch", 0)
    migrated["global_step"] = migrated_progress.get("global_step", successful_steps)
    migrated.setdefault("amp_scaler_state_dict", None)
    migrated.setdefault("scaler_state_dict", None)
    migrated.setdefault("best_metric", migrated_progress.get("best_metric"))
    migrated.setdefault("checkpoint_timing", {})
    migrated.setdefault("checkpoint_events", [])
    migrated.setdefault(
        "sampler_state",
        {"kind": "legacy-v1", "next_batch_index": attempted_batches},
    )
    migrated.setdefault(
        "recovery",
        {
            "legacy_checkpoint": True,
            "exact_interrupted_batch_replay": False,
        },
    )
    migrated.setdefault("parent_provenance", {})
    if expected_config is not None:
        identity_for_hash = dict(expected_identity or identity)
        migrated["config"] = expected_config.to_dict()
        migrated["config_fingerprint"] = expected_config.config_fingerprint
        migrated["semantic_fingerprint"] = semantic_fingerprint(
            expected_config, identity_for_hash
        )
    else:
        migrated["semantic_fingerprint"] = _json_sha256(
            {
                "legacy_checkpoint_schema": LEGACY_CHECKPOINT_SCHEMA_VERSION,
                "config_fingerprint": checkpoint_fingerprint,
                "identity": dict(identity),
            }
        )
    return migrated


def load_training_checkpoint(
    path: str | Path,
    *,
    model: HierarchicalStudyModel | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: Any | None = None,
    expected_config: TrainingConfig | None = None,
    expected_fold: int | None = None,
    expected_identity: Mapping[str, object] | None = None,
    expected_semantic_fingerprint: str | None = None,
    expected_checkpoint_kind: str | None = None,
    restore_rng: bool = False,
) -> dict[str, object]:
    """Load and validate a complete checkpoint, optionally restoring modules."""

    source = Path(path)
    try:
        map_location: str | torch.device = "cpu"
        if model is not None:
            map_location = next(model.parameters()).device
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"could not load checkpoint {source}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint schema is unsupported")
    checkpoint_schema = payload.get("schema_version")
    is_legacy_checkpoint = checkpoint_schema == LEGACY_CHECKPOINT_SCHEMA_VERSION
    if checkpoint_schema not in {LEGACY_CHECKPOINT_SCHEMA_VERSION, CHECKPOINT_SCHEMA_VERSION}:
        raise CheckpointError("checkpoint schema is unsupported")
    checkpoint_config = payload.get("config")
    checkpoint_fingerprint = payload.get("config_fingerprint")
    if not isinstance(checkpoint_config, Mapping) or not isinstance(checkpoint_fingerprint, str):
        raise CheckpointError("checkpoint configuration metadata is incomplete")
    if is_legacy_checkpoint:
        if _legacy_config_fingerprint_from_payload(checkpoint_config) != checkpoint_fingerprint:
            raise CheckpointError("legacy checkpoint configuration fingerprint is invalid")
        if (
            expected_config is not None
            and checkpoint_fingerprint != _legacy_config_fingerprint(expected_config)
        ):
            raise TrainingConfigurationError(
                "checkpoint scientific configuration does not match the requested run"
            )
        legacy_identity = payload.get("identity")
        if not isinstance(legacy_identity, Mapping):
            raise CheckpointError("checkpoint scientific identity is missing")
        if expected_identity is not None:
            if expected_config is None:
                raise TrainingConfigurationError(
                    "expected_config is required when checking checkpoint identity"
                )
            if not _legacy_identity_matches(
                legacy_identity,
                expected_identity,
                expected_config=expected_config,
                legacy_config_fingerprint=checkpoint_fingerprint,
            ):
                raise TrainingConfigurationError(
                    "checkpoint semantic fingerprint does not match the requested run"
                )
        payload = _migrate_legacy_checkpoint(
            payload,
            source=source,
            expected_config=expected_config,
            expected_identity=expected_identity,
        )
        checkpoint_config = payload["config"]
        checkpoint_fingerprint = payload["config_fingerprint"]
    if expected_config is not None and checkpoint_fingerprint != expected_config.config_fingerprint:
        raise TrainingConfigurationError(
            "checkpoint scientific configuration does not match the requested run"
        )
    checkpoint_fold = payload.get("fold")
    if expected_fold is not None and checkpoint_fold != expected_fold:
        raise TrainingConfigurationError("checkpoint fold does not match the requested fold")
    checkpoint_kind = payload.get("checkpoint_kind")
    if checkpoint_kind not in {"latest_resumable", "final_selected"}:
        raise CheckpointError("checkpoint kind is missing or unsupported")
    if expected_checkpoint_kind is not None and checkpoint_kind != expected_checkpoint_kind:
        raise TrainingConfigurationError("checkpoint kind is not valid for this operation")
    actual_semantic = payload.get("semantic_fingerprint")
    if not isinstance(actual_semantic, str) or not actual_semantic:
        raise CheckpointError("checkpoint semantic fingerprint is missing")
    if not isinstance(payload.get("identity"), Mapping):
        raise CheckpointError("checkpoint scientific identity is missing")
    if expected_semantic_fingerprint is None and expected_identity is not None:
        if expected_config is None:
            raise TrainingConfigurationError(
                "expected_config is required when checking checkpoint identity"
            )
        expected_semantic_fingerprint = semantic_fingerprint(expected_config, expected_identity)
    if (
        expected_semantic_fingerprint is not None
        and actual_semantic != expected_semantic_fingerprint
    ):
        raise TrainingConfigurationError(
            "checkpoint semantic fingerprint does not match the requested run"
        )
    model_state = payload.get("model_state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    scheduler_state = payload.get("scheduler_state_dict")
    if not isinstance(model_state, Mapping) or not isinstance(optimizer_state, Mapping) or not isinstance(
        scheduler_state, Mapping
    ):
        raise CheckpointError("checkpoint is missing complete training state")
    if model is not None:
        try:
            model.load_state_dict(model_state)
        except (RuntimeError, ValueError) as exc:
            raise TrainingConfigurationError("checkpoint model state is incompatible") from exc
    if optimizer is not None:
        try:
            optimizer.load_state_dict(dict(optimizer_state))
        except (RuntimeError, ValueError, KeyError) as exc:
            raise TrainingConfigurationError("checkpoint optimizer state is incompatible") from exc
    if scheduler is not None:
        try:
            scheduler.load_state_dict(dict(scheduler_state))
        except (RuntimeError, ValueError, KeyError) as exc:
            raise TrainingConfigurationError("checkpoint scheduler state is incompatible") from exc
    scaler_state = payload.get("amp_scaler_state_dict")
    if scaler_state is None:
        scaler_state = payload.get("scaler_state_dict")
    if scaler is not None:
        if not isinstance(scaler_state, Mapping):
            raise CheckpointError("checkpoint is missing the AMP scaler state")
        load_scaler_state = getattr(scaler, "load_state_dict", None)
        if not callable(load_scaler_state):
            raise CheckpointError("AMP scaler does not expose load_state_dict()")
        try:
            load_scaler_state(dict(scaler_state))
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            raise TrainingConfigurationError("checkpoint AMP scaler state is incompatible") from exc
    if not isinstance(payload.get("rng_state"), dict):
        raise CheckpointError("checkpoint is missing RNG state")
    progress = payload.get("progress")
    if not isinstance(progress, dict):
        raise CheckpointError("checkpoint is missing progress state")
    required_progress = (
        "epoch",
        "global_step",
        "successful_optimizer_steps",
        "scheduler_steps",
        "attempted_batches",
        "skipped_all_masked_batches",
        "loss_history",
        "batch_in_epoch",
        "next_batch_index",
        "consecutive_no_progress",
    )
    if any(name not in progress for name in required_progress):
        raise CheckpointError("checkpoint progress state is incomplete")
    integer_progress = (
        "epoch",
        "global_step",
        "successful_optimizer_steps",
        "scheduler_steps",
        "attempted_batches",
        "skipped_all_masked_batches",
        "batch_in_epoch",
        "next_batch_index",
        "consecutive_no_progress",
    )
    if any(
        not isinstance(progress[name], int)
        or isinstance(progress[name], bool)
        or progress[name] < 0
        for name in integer_progress
    ):
        raise CheckpointError("checkpoint progress counters are invalid")
    if progress["attempted_batches"] < progress["successful_optimizer_steps"]:
        raise CheckpointError("checkpoint attempted and successful batch counts disagree")
    loss_values = progress["loss_history"]
    if not isinstance(loss_values, (list, tuple)) or any(
        not isinstance(value, (int, float)) or not np.isfinite(float(value))
        for value in loss_values
    ):
        raise CheckpointError("checkpoint loss history is invalid")
    if len(loss_values) != progress["successful_optimizer_steps"]:
        raise CheckpointError("checkpoint loss history and optimizer progress disagree")
    best_metric = payload.get("best_metric")
    if best_metric is None and isinstance(progress.get("best_metric"), (int, float)):
        best_metric = progress.get("best_metric")
    if best_metric is not None and (
        not isinstance(best_metric, (int, float)) or not np.isfinite(float(best_metric))
    ):
        raise CheckpointError("checkpoint best metric is invalid")
    if restore_rng:
        _restore_rng_state(cast(Mapping[str, object], payload["rng_state"]))
    return payload


def _finite_gradients(model: nn.Module) -> bool:
    found = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        found = True
        if not torch.isfinite(parameter.grad).all():
            return False
    return found


def _default_run_id(identity: Mapping[str, object], config: TrainingConfig) -> str:
    return semantic_fingerprint(config, identity)


def _checkpoint_identity(
    config: TrainingConfig,
    examples: Sequence[TrainingExample],
    *,
    source_version: str,
    source_fingerprint: str | None,
    cache_fingerprint: str | None,
    fold_manifest_checksum: str | None,
    integration_manifest_checksum: str | None,
    fold: int | None,
) -> dict[str, object]:
    return {
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "model_source": "local:HierarchicalStudyModel;weights=deterministic-initialization",
        "data": {
            "source_version": source_version,
            "source_fingerprint": source_fingerprint,
            "training_examples": fingerprint_training_examples(examples),
            "cache_fingerprint": cache_fingerprint,
            "integration_manifest_checksum": integration_manifest_checksum,
        },
        "split": {
            "fold": fold,
            "fold_manifest_checksum": fold_manifest_checksum,
        },
        "labels": {
            "source": "reference-only:frozen-fold-manifest",
            "supervision": "masked_reference_entries",
        },
        "source_version": source_version,
        "source_fingerprint": source_fingerprint,
        "data_fingerprint": fingerprint_training_examples(examples),
        "cache_fingerprint": cache_fingerprint,
        "fold_manifest_checksum": fold_manifest_checksum,
        "integration_manifest_checksum": integration_manifest_checksum,
        "fold": fold,
        "label_source": "reference-only:frozen-fold-manifest",
        "preprocessing": config.preprocess_config().cache_config(),
        "packing": config.packing_config().to_dict(),
        "architecture": config.model_config().__dict__,
        "initialization": "deterministic-initialization",
        "optimization": {
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "schedule": config.schedule,
        },
        "loss": config.loss,
        "augmentation": config.augmentation,
        "budget": {
            "successful_optimizer_steps": config.successful_optimizer_steps,
            "batch_size": config.batch_size,
        },
        "configuration_fingerprint": config.config_fingerprint,
        "seeds": {"initialization": config.seed, "batch_order": config.batch_order_seed},
    }


def _batch_indices(
    example_count: int,
    batch_size: int,
    attempted_batch: int,
    seed: int,
) -> tuple[int, ...]:
    """Return a reproducible shuffled batch without a stateful sampler."""

    batches_per_cycle = max(1, (example_count + batch_size - 1) // batch_size)
    cycle, batch_in_cycle = divmod(attempted_batch, batches_per_cycle)
    order = list(range(example_count))
    random.Random(seed + cycle).shuffle(order)
    start = batch_in_cycle * batch_size
    return tuple(order[(start + offset) % example_count] for offset in range(batch_size))


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_artifact_dir(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        if candidate.name == "history":
            return candidate.parent
        if candidate.name == "checkpoints":
            return candidate
        return candidate / "checkpoints"
    if candidate.name == "checkpoints":
        return candidate
    return candidate.parent


def _resume_candidates(path: str | Path) -> tuple[Path, ...]:
    candidate = Path(path)
    if candidate.is_dir():
        checkpoint_dir = _checkpoint_artifact_dir(candidate)
        latest = checkpoint_dir / "latest.pt"
    else:
        latest = candidate
        checkpoint_dir = candidate.parent
    history_dir = checkpoint_dir / "history"
    candidates = [latest]
    if history_dir.is_dir():
        candidates.extend(sorted(history_dir.glob("*.pt"), reverse=True))
    # Keep the explicit path first and make the fallback list deterministic.
    return tuple(dict.fromkeys(candidates))


def find_latest_resumable_checkpoint(
    path: str | Path,
    *,
    expected_config: TrainingConfig | None = None,
    expected_fold: int | None = None,
    expected_identity: Mapping[str, object] | None = None,
) -> Path:
    """Return the highest-progress complete resumable checkpoint.

    A damaged ``latest.pt`` is ignored only when a complete historical
    checkpoint is available.  Semantic/configuration mismatches are raised,
    rather than silently falling back to a checkpoint from another run.
    """

    valid: list[tuple[int, int, str, Path]] = []
    for candidate in _resume_candidates(path):
        if not candidate.is_file():
            continue
        try:
            payload = load_training_checkpoint(
                candidate,
                expected_config=expected_config,
                expected_fold=expected_fold,
                expected_identity=expected_identity,
                expected_checkpoint_kind="latest_resumable",
            )
        except CheckpointError:
            continue
        progress = cast(Mapping[str, object], payload["progress"])
        valid.append(
            (
                int(cast(Any, progress["successful_optimizer_steps"])),
                int(cast(Any, progress["attempted_batches"])),
                str(payload.get("checkpoint_reason", "")),
                candidate,
            )
        )
    if not valid:
        raise CheckpointError(f"no complete resumable checkpoint is available under {path}")
    return max(valid, key=lambda item: (item[0], item[1], item[2]))[3]


def _next_execution_attempt(
    output_dir: Path,
    *,
    run_id: str,
    semantic_identity: str,
    parent_attempt_id: str | None,
    parent_run_id: str | None,
    requested_resources: Mapping[str, object] | None,
    supplied_attempt_id: str | None,
) -> tuple[str, Path]:
    attempts_dir = output_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    if supplied_attempt_id is not None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", supplied_attempt_id):
            raise TrainingConfigurationError("execution attempt ID contains unsafe characters")
        attempt_id = supplied_attempt_id
        attempt_dir = attempts_dir / attempt_id
        try:
            attempt_dir.mkdir()
        except FileExistsError as exc:
            raise FileExistsError(f"execution attempt already exists: {attempt_id}") from exc
    else:
        numbers = [
            int(match.group(1))
            for item in attempts_dir.iterdir()
            if (match := re.fullmatch(r"attempt-(\d+)", item.name)) is not None
        ]
        next_number = max(numbers, default=0) + 1
        while True:
            attempt_id = f"attempt-{next_number:04d}"
            attempt_dir = attempts_dir / attempt_id
            try:
                attempt_dir.mkdir()
                break
            except FileExistsError:
                next_number += 1
    atomic_json_write(
        attempt_dir / "attempt.json",
        {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "run_id": run_id,
            "semantic_fingerprint": semantic_identity,
            "execution_attempt_id": attempt_id,
            "parent_execution_attempt_id": parent_attempt_id,
            "parent_run_id": parent_run_id,
            "status": "started",
            "started_at": _utc_timestamp(),
            "environment": collect_environment_record(),
            "resources": dict(requested_resources or {}),
        },
    )
    return attempt_id, attempt_dir


def _default_requested_resources() -> dict[str, object]:
    """Return the bounded CPU envelope used when no execution resource is supplied."""

    return {
        "requested_partition": "cluster1-cpu",
        "requested_nodes": 1,
        "requested_cpus": 16,
        "requested_memory_gb": 64,
        "requested_wall_time_hours": 6,
        "exclusive": False,
    }


def train_fixed_budget(
    examples: Sequence[TrainingExample],
    config: TrainingConfig = TrainingConfig(),
    *,
    model: HierarchicalStudyModel | None = None,
    artifact_dir: str | Path | None = None,
    fold: int | None = None,
    run_id: str | None = None,
    identity: Mapping[str, object] | None = None,
    resume_from: str | Path | None = None,
    execution_attempt_id: str | None = None,
    requested_resources: Mapping[str, object] | None = None,
    parent_provenance: Mapping[str, object] | None = None,
    amp_scaler: Any | None = None,
    scaler: Any | None = None,
    best_metric: float | None = None,
    checkpoint_clock: Callable[[], float] | None = None,
    interrupt_after_steps: int | None = None,
) -> TrainingResult:
    """Run a CPU fixed-budget training job.

    CUDA execution is deliberately not part of this public training seam;
    the approval-gated GPU adapter uses ``_train_fixed_budget_gpu`` instead.
    """

    if config.device != "cpu":
        raise TrainingConfigurationError(
            "CUDA fixed-budget training is available only through the approval-gated GPU runner"
        )
    return _train_fixed_budget_impl(
        examples,
        config,
        model=model,
        artifact_dir=artifact_dir,
        fold=fold,
        run_id=run_id,
        identity=identity,
        resume_from=resume_from,
        execution_attempt_id=execution_attempt_id,
        requested_resources=requested_resources,
        parent_provenance=parent_provenance,
        amp_scaler=amp_scaler,
        scaler=scaler,
        best_metric=best_metric,
        checkpoint_clock=checkpoint_clock,
        interrupt_after_steps=interrupt_after_steps,
    )


def _train_fixed_budget_impl(
    examples: Sequence[TrainingExample],
    config: TrainingConfig = TrainingConfig(),
    *,
    model: HierarchicalStudyModel | None = None,
    artifact_dir: str | Path | None = None,
    fold: int | None = None,
    run_id: str | None = None,
    identity: Mapping[str, object] | None = None,
    resume_from: str | Path | None = None,
    execution_attempt_id: str | None = None,
    requested_resources: Mapping[str, object] | None = None,
    gpu_authorization: object | None = None,
    parent_provenance: Mapping[str, object] | None = None,
    amp_scaler: Any | None = None,
    scaler: Any | None = None,
    best_metric: float | None = None,
    checkpoint_clock: Callable[[], float] | None = None,
    interrupt_after_steps: int | None = None,
) -> TrainingResult:
    """Run or resume exactly ``successful_optimizer_steps`` supervised steps.

    Batches are deterministic and cyclic.  A batch without any supervised
    target entries is recorded and skipped; it never calls either
    ``optimizer.step`` or ``scheduler.step``.  A configured consecutive skip
    limit turns persistent no-progress into an explicit failure.  A resume
    restores the model, optimizer, scheduler, optional AMP scaler, RNG streams,
    progress counters, and scientific identity from the last complete
    ``latest_resumable`` checkpoint.
    """

    if config.device != "cpu" and gpu_authorization is not _GPU_EXECUTION_CAPABILITY:
        raise TrainingConfigurationError(
            "CUDA fixed-budget training requires the approval-gated GPU execution path"
        )
    if config.device != "cpu":
        _require_gpu_slurm_context()
    examples_tuple = tuple(examples)
    _validate_examples(examples_tuple, config)
    if interrupt_after_steps is not None and interrupt_after_steps < 0:
        raise TrainingConfigurationError("interrupt_after_steps cannot be negative")
    if amp_scaler is not None and scaler is not None:
        raise TrainingConfigurationError("pass only one of amp_scaler or scaler")
    if scaler is not None:
        amp_scaler = scaler
    if best_metric is not None and not np.isfinite(best_metric):
        raise TrainingConfigurationError("best_metric must be finite when present")
    if requested_resources is not None and not isinstance(requested_resources, Mapping):
        raise TrainingConfigurationError("requested_resources must be a mapping")
    effective_resources = dict(requested_resources or _default_requested_resources())
    clock = checkpoint_clock or time.monotonic
    resume_payload: dict[str, object] | None = None
    resume_path: Path | None = None
    if resume_from is not None:
        if artifact_dir is None:
            resume_candidate = Path(resume_from)
            if resume_candidate.is_dir():
                if resume_candidate.name == "history":
                    artifact_dir = resume_candidate.parent.parent
                elif resume_candidate.name == "checkpoints":
                    artifact_dir = resume_candidate.parent
                else:
                    artifact_dir = resume_candidate
            elif resume_candidate.parent.name == "checkpoints":
                artifact_dir = resume_candidate.parent.parent
            elif resume_candidate.parent.name == "history":
                artifact_dir = resume_candidate.parent.parent.parent
            else:
                artifact_dir = resume_candidate.parent
        resume_path = Path(resume_from)
        expected_identity = identity
        if resume_path.is_dir() or resume_path.name == "latest.pt":
            resume_path = find_latest_resumable_checkpoint(
                resume_path,
                expected_config=config,
                expected_fold=fold,
                expected_identity=expected_identity,
            )
        resume_payload = load_training_checkpoint(
            resume_path,
            expected_config=config,
            expected_fold=fold,
            expected_identity=expected_identity,
            expected_checkpoint_kind="latest_resumable",
        )
        saved_identity = resume_payload.get("identity")
        if not isinstance(saved_identity, Mapping):
            raise CheckpointError("resumable checkpoint has no scientific identity")
        if identity is None:
            identity = cast(Mapping[str, object], saved_identity)
        saved_parent_provenance = resume_payload.get("parent_provenance")
        if parent_provenance is None and isinstance(saved_parent_provenance, Mapping):
            parent_provenance = cast(Mapping[str, object], saved_parent_provenance)
    seed_everything(config.seed)
    device = torch.device(config.device)
    if model is None:
        model = HierarchicalStudyModel(config.model_config())
    elif model.config != config.model_config():
        raise TrainingConfigurationError("provided model configuration does not match TrainingConfig")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    output_dir = Path(artifact_dir) if artifact_dir is not None else None
    latest_path: Path | None = None
    final_path: Path | None = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest_path = checkpoint_dir / "latest.pt"
        final_path = checkpoint_dir / "final.pt"
        if resume_payload is None and final_path.exists():
            raise FileExistsError(f"refusing to start over an existing final checkpoint {final_path}")
        if resume_payload is None and (
            latest_path.exists() or any((checkpoint_dir / "history").glob("*.pt"))
        ):
            raise FileExistsError(
                f"refusing to overwrite an existing run; provide resume_from for {output_dir}"
            )
    identity_payload = dict(identity or {})
    if not identity_payload:
        identity_payload = {
            "data_fingerprint": fingerprint_training_examples(examples_tuple),
            "label_source": "reference-only",
            "fold": fold,
        }
    effective_run_id = run_id or _default_run_id(identity_payload, config)
    semantic_identity = semantic_fingerprint(config, identity_payload)
    if resume_payload is not None:
        saved_run_id = resume_payload.get("run_id")
        if resume_payload.get("migrated_from_schema_version") == LEGACY_CHECKPOINT_SCHEMA_VERSION:
            if saved_run_id is not None:
                effective_run_id = str(saved_run_id)
        elif saved_run_id is not None and saved_run_id != effective_run_id:
            raise TrainingConfigurationError("checkpoint run identity does not match the requested run")
        saved_semantic = resume_payload.get("semantic_fingerprint")
        if saved_semantic != semantic_identity:
            raise TrainingConfigurationError(
                "checkpoint semantic fingerprint does not match the requested run"
            )
        effective_run_id = str(saved_run_id or effective_run_id)
    if output_dir is not None:
        configuration_path = output_dir / "configuration.json"
        if resume_payload is None and not configuration_path.exists():
            atomic_json_write(
                configuration_path,
                {
                    "schema_version": TRAINING_SCHEMA_VERSION,
                    "run_id": effective_run_id,
                    "fold": fold,
                    "config": config.to_dict(),
                    "scientific_configuration_fingerprint": config.config_fingerprint,
                    "semantic_fingerprint": semantic_identity,
                    "identity": identity_payload,
                    "parent_provenance": dict(parent_provenance or {}),
                    "environment": collect_environment_record(),
                    "resources": {
                        **effective_resources,
                        "slurm_attempt_id": os.environ.get("SLURM_JOB_ID"),
                    },
                },
            )
        else:
            try:
                saved_configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise CheckpointError("could not read the immutable experiment configuration") from exc
            if not isinstance(saved_configuration, Mapping):
                raise CheckpointError("experiment configuration is not an object")
            if saved_configuration.get("semantic_fingerprint") is None:
                if not _legacy_configuration_matches(
                    saved_configuration,
                    expected_config=config,
                    expected_identity=identity_payload,
                ):
                    raise TrainingConfigurationError(
                        "legacy experiment configuration does not match the requested scientific run"
                    )
                saved_configuration_run_id = saved_configuration.get("run_id")
                if isinstance(saved_configuration_run_id, str) and saved_configuration_run_id:
                    if resume_payload is not None and saved_configuration_run_id != effective_run_id:
                        raise TrainingConfigurationError(
                            "legacy experiment run identity does not match its checkpoint"
                        )
                    effective_run_id = saved_configuration_run_id
            elif (
                saved_configuration.get("run_id") != effective_run_id
                or saved_configuration.get("semantic_fingerprint") != semantic_identity
            ):
                raise TrainingConfigurationError(
                    "experiment configuration does not match the requested scientific run"
                )

    successful_steps = 0
    scheduler_steps = 0
    attempted_batches = 0
    skipped_batches = 0
    consecutive_no_progress = 0
    loss_history: list[float] = []
    epoch = 0
    checkpoint_events: list[Mapping[str, object]] = []
    replayed_batches = 0
    if resume_payload is not None:
        load_training_checkpoint(
            resume_path if resume_path is not None else "",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=amp_scaler,
            expected_config=config,
            expected_fold=fold,
            expected_identity=identity_payload,
            expected_checkpoint_kind="latest_resumable",
            restore_rng=True,
        )
        progress = cast(Mapping[str, object], resume_payload["progress"])
        successful_steps = int(cast(Any, progress["successful_optimizer_steps"]))
        scheduler_steps = int(cast(Any, progress["scheduler_steps"]))
        attempted_batches = int(cast(Any, progress["attempted_batches"]))
        skipped_batches = int(cast(Any, progress["skipped_all_masked_batches"]))
        consecutive_no_progress = int(cast(Any, progress["consecutive_no_progress"]))
        loss_history = [
            float(cast(Any, value))
            for value in cast(Sequence[object], progress["loss_history"])
        ]
        epoch = int(cast(Any, progress["epoch"]))
        saved_best_metric = resume_payload.get("best_metric")
        if saved_best_metric is not None:
            if best_metric is not None and float(cast(Any, saved_best_metric)) != best_metric:
                raise TrainingConfigurationError(
                    "requested best metric does not match the resumable checkpoint"
                )
            best_metric = float(cast(Any, saved_best_metric))
        saved_events = resume_payload.get("checkpoint_events", [])
        if isinstance(saved_events, Sequence) and not isinstance(saved_events, (str, bytes)):
            checkpoint_events = [
                dict(event)
                for event in saved_events
                if isinstance(event, Mapping)
            ]
        parent_attempt = resume_payload.get("execution_attempt_id")
        if output_dir is not None and isinstance(parent_attempt, str) and parent_attempt:
            previous_failure = output_dir / "attempts" / parent_attempt / "failure.json"
            try:
                previous_record = json.loads(previous_failure.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                previous_record = None
            if isinstance(previous_record, Mapping):
                replayed_value = previous_record.get("uncheckpointed_batches", 0)
                if isinstance(replayed_value, (int, float)) and replayed_value >= 0:
                    replayed_batches = int(replayed_value)
        if successful_steps > config.successful_optimizer_steps:
            raise TrainingConfigurationError(
                "checkpoint has more optimizer steps than the requested scientific budget"
            )

    attempt_id: str | None = None
    attempt_dir: Path | None = None
    log_handle: Any = None
    if output_dir is not None:
        attempt_id, attempt_dir = _next_execution_attempt(
            output_dir,
            run_id=effective_run_id,
            semantic_identity=semantic_identity,
            parent_attempt_id=(
                str(resume_payload.get("execution_attempt_id"))
                if resume_payload is not None and resume_payload.get("execution_attempt_id")
                else None
            ),
            parent_run_id=(
                str(resume_payload.get("run_id"))
                if resume_payload is not None and resume_payload.get("run_id")
                else (str(parent_provenance.get("parent_run_id")) if parent_provenance and parent_provenance.get("parent_run_id") else None)
            ),
            requested_resources=effective_resources,
            supplied_attempt_id=execution_attempt_id,
        )
        log_handle = (attempt_dir / "training.log").open("a", encoding="utf-8")

    def write_log(event: str, **values: object) -> None:
        if log_handle is None:
            return
        log_handle.write(json.dumps({"event": event, **values}, sort_keys=True) + "\n")
        log_handle.flush()

    write_log(
        "training_started",
        run_id=effective_run_id,
        fold=fold,
        execution_attempt_id=attempt_id,
        resumed_from_checkpoint=str(resume_path) if resume_path is not None else None,
        successful_optimizer_steps=successful_steps,
    )

    batches_per_epoch = max(1, (len(examples_tuple) + config.batch_size - 1) // config.batch_size)
    last_checkpoint_at = clock()
    last_checkpoint_steps = successful_steps
    last_checkpoint_attempted_batches = attempted_batches
    checkpoint_sequence = 0
    if output_dir is not None:
        history_dir = output_dir / "checkpoints" / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_sequence = len(tuple(history_dir.glob("*.pt")))

    def publish_checkpoint(
        *,
        reason: str,
        overwrite_latest: bool = True,
        selected_final: bool = False,
        now: float | None = None,
    ) -> None:
        nonlocal last_checkpoint_at, last_checkpoint_steps
        nonlocal last_checkpoint_attempted_batches, checkpoint_sequence
        if output_dir is None or latest_path is None or final_path is None:
            return
        checkpoint_now = clock() if now is None else now
        elapsed = max(0.0, checkpoint_now - last_checkpoint_at)
        timing = {
            "interval_seconds": config.checkpoint_interval_seconds,
            "elapsed_since_previous_seconds": elapsed,
            "overdue_seconds": max(0.0, elapsed - config.checkpoint_interval_seconds),
            "measured_at_monotonic": checkpoint_now,
        }
        event: dict[str, object] = {
            "reason": reason,
            "epoch": epoch,
            "global_step": successful_steps,
            "attempted_batches": attempted_batches,
            "uncheckpointed_batches": attempted_batches - last_checkpoint_attempted_batches,
            "replayed_batches": replayed_batches,
            "elapsed_since_previous_seconds": elapsed,
            "overdue_seconds": timing["overdue_seconds"],
        }
        event_history = [*checkpoint_events, event]
        checkpoint_path = final_path if selected_final else latest_path
        checkpoint_kind = "final_selected" if selected_final else "latest_resumable"

        def save_state(destination: Path, *, kind: str, overwrite: bool) -> None:
            save_training_checkpoint(
                destination,
                model,
                optimizer,
                scheduler,
                config,
                successful_optimizer_steps=successful_steps,
                attempted_batches=attempted_batches,
                skipped_batches=skipped_batches,
                loss_history=loss_history,
                fold=fold,
                run_id=effective_run_id,
                identity=identity_payload,
                epoch=epoch,
                global_step=successful_steps,
                batch_in_epoch=attempted_batches % batches_per_epoch,
                next_batch_index=attempted_batches,
                best_metric=best_metric,
                scaler=amp_scaler,
                checkpoint_kind=kind,
                checkpoint_reason=reason,
                checkpoint_timing=timing,
                checkpoint_events=event_history,
                sampler_state={
                    "kind": "deterministic-cyclic",
                    "seed": config.batch_order_seed,
                    "next_batch_index": attempted_batches,
                },
                recovery={
                    "max_uncheckpointed_batches": batches_per_epoch,
                    "uncheckpointed_batches": event["uncheckpointed_batches"],
                    "replayed_batches": replayed_batches,
                    "exact_interrupted_batch_replay": False,
                },
                parent_provenance=parent_provenance,
                execution_attempt_id=attempt_id,
                scheduler_steps=scheduler_steps,
                consecutive_no_progress=consecutive_no_progress,
                overwrite=overwrite,
            )

        if not selected_final:
            history_path = (
                output_dir
                / "checkpoints"
                / "history"
                / f"{attempt_id or 'attempt'}-{checkpoint_sequence:06d}.pt"
            )
            save_state(history_path, kind=checkpoint_kind, overwrite=False)
            checkpoint_sequence += 1
        save_state(
            checkpoint_path,
            kind=checkpoint_kind,
            overwrite=overwrite_latest if not selected_final else False,
        )
        checkpoint_events.append(event)
        last_checkpoint_at = checkpoint_now
        last_checkpoint_steps = successful_steps
        last_checkpoint_attempted_batches = attempted_batches
        write_log(
            "checkpoint_published",
            path=str(checkpoint_path),
            checkpoint_kind=checkpoint_kind,
            **event,
        )

    try:
        if latest_path is not None and resume_payload is None:
            publish_checkpoint(reason="initial_state", now=last_checkpoint_at)
        if interrupt_after_steps == 0:
            raise TrainingInterrupted(
                "training interrupted before the first optimizer step",
                checkpoint_path=latest_path,
            )
        while successful_steps < config.successful_optimizer_steps:
            batch_examples = tuple(
                examples_tuple[index]
                for index in _batch_indices(
                    len(examples_tuple),
                    config.batch_size,
                    attempted_batches,
                    config.batch_order_seed,
                )
            )
            attempted_batches += 1
            batch_mask = np.stack([example.supervision_mask for example in batch_examples])
            loss_value: float | None = None
            if not bool(batch_mask.any()):
                skipped_batches += 1
                consecutive_no_progress += 1
                write_log(
                    "batch_skipped_no_supervision",
                    attempted_batch=attempted_batches,
                    skipped_batches=skipped_batches,
                )
                if consecutive_no_progress >= config.persistent_no_progress_limit:
                    raise TrainingProgressError(
                        "no supervised progress after "
                        f"{consecutive_no_progress} consecutive all-masked batches"
                    )
            else:
                images = torch.from_numpy(
                    np.stack([example.images for example in batch_examples])
                ).to(device)
                slice_mask = torch.from_numpy(
                    np.stack([example.slice_mask for example in batch_examples])
                ).to(device)
                series_mask = torch.from_numpy(
                    np.stack([example.series_mask for example in batch_examples])
                ).to(device)
                targets = torch.from_numpy(
                    np.stack([example.targets for example in batch_examples])
                ).to(device)
                supervision_mask = torch.from_numpy(batch_mask).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images, slice_mask, series_mask)
                loss = masked_bce_loss(logits, targets, supervision_mask)
                if loss is None:
                    skipped_batches += 1
                    consecutive_no_progress += 1
                else:
                    if not bool(torch.isfinite(loss)):
                        raise TrainingNumericsError("training loss is non-finite")
                    if amp_scaler is None:
                        loss.backward()
                    else:
                        scale = getattr(amp_scaler, "scale", None)
                        if not callable(scale):
                            raise TrainingConfigurationError("AMP scaler does not expose scale()")
                        scale(loss).backward()
                        unscale = getattr(amp_scaler, "unscale_", None)
                        if not callable(unscale):
                            raise TrainingConfigurationError("AMP scaler does not expose unscale_()")
                        unscale(optimizer)
                    if not _finite_gradients(model):
                        raise TrainingNumericsError("training gradients are missing or non-finite")
                    if amp_scaler is None:
                        optimizer.step()
                    else:
                        scaler_step = getattr(amp_scaler, "step", None)
                        scaler_update = getattr(amp_scaler, "update", None)
                        if not callable(scaler_step) or not callable(scaler_update):
                            raise TrainingConfigurationError(
                                "AMP scaler does not expose step()/update()"
                            )
                        scaler_step(optimizer)
                        scaler_update()
                    scheduler.step()
                    successful_steps += 1
                    scheduler_steps += 1
                    consecutive_no_progress = 0
                    loss_value = float(loss.detach().cpu().item())
                    if not np.isfinite(loss_value):
                        raise TrainingNumericsError("recorded training loss is non-finite")
                    loss_history.append(loss_value)
                    write_log(
                        "optimizer_step",
                        attempted_batch=attempted_batches,
                        successful_optimizer_steps=successful_steps,
                        loss=loss_value,
                        learning_rate=float(optimizer.param_groups[0]["lr"]),
                    )

            epoch = attempted_batches // batches_per_epoch
            now = clock()
            at_epoch_boundary = attempted_batches % batches_per_epoch == 0
            step_due = (
                successful_steps > 0
                and successful_steps - last_checkpoint_steps >= config.checkpoint_interval_steps
            )
            time_due = now - last_checkpoint_at >= config.checkpoint_interval_seconds
            target_reached = successful_steps >= config.successful_optimizer_steps
            if latest_path is not None and (at_epoch_boundary or step_due or time_due or target_reached):
                reasons = [
                    name
                    for name, due in (
                        ("epoch_boundary", at_epoch_boundary),
                        ("optimizer_step_interval", step_due),
                        ("time_interval", time_due),
                        ("target_reached", target_reached),
                    )
                    if due
                ]
                publish_checkpoint(reason="+".join(reasons), now=now)
            if (
                interrupt_after_steps is not None
                and successful_steps >= interrupt_after_steps
                and successful_steps < config.successful_optimizer_steps
            ):
                raise TrainingInterrupted(
                    f"training interrupted after {successful_steps} optimizer steps",
                    checkpoint_path=latest_path,
                )
        if final_path is not None:
            publish_checkpoint(reason="final_selection", selected_final=True, now=clock())
    except BaseException as exc:
        try:
            write_log(
                "training_failed",
                run_id=effective_run_id,
                execution_attempt_id=attempt_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
        except Exception:
            # A full or unavailable log filesystem must not replace the
            # scientific failure that caused this attempt to stop.
            pass
        if attempt_dir is not None:
            try:
                classification = classify_retry_failure(exc)
                atomic_json_write(
                    attempt_dir / "failure.json",
                    {
                        "schema_version": TRAINING_SCHEMA_VERSION,
                        "run_id": effective_run_id,
                        "execution_attempt_id": attempt_id,
                        "status": "failed",
                        "finished_at": _utc_timestamp(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "retry": classification.to_dict(),
                        "last_complete_checkpoint": str(latest_path) if latest_path and latest_path.exists() else None,
                        "uncheckpointed_batches": max(
                            0, attempted_batches - last_checkpoint_attempted_batches
                        ),
                        "replayed_batches_on_next_resume": max(
                            0, attempted_batches - last_checkpoint_attempted_batches
                        ),
                    },
                )
            except (OSError, TypeError, ValueError):
                pass
        raise
    else:
        write_log(
            "training_finished",
            successful_optimizer_steps=successful_steps,
            attempted_batches=attempted_batches,
            skipped_batches=skipped_batches,
            epoch=epoch,
        )
        if attempt_dir is not None:
            atomic_json_write(
                attempt_dir / "completion.json",
                {
                    "schema_version": TRAINING_SCHEMA_VERSION,
                    "run_id": effective_run_id,
                    "execution_attempt_id": attempt_id,
                    "status": "completed",
                    "finished_at": _utc_timestamp(),
                    "successful_optimizer_steps": successful_steps,
                    "attempted_batches": attempted_batches,
                    "skipped_batches": skipped_batches,
                    "epoch": epoch,
                    "resumed_from_checkpoint": str(resume_path) if resume_path is not None else None,
                    "replayed_batches": replayed_batches,
                    "checkpoint_events": len(checkpoint_events),
                },
            )
    finally:
        if log_handle is not None:
            log_handle.close()
    return TrainingResult(
        model=model,
        config=config,
        successful_optimizer_steps=successful_steps,
        scheduler_steps=scheduler_steps,
        attempted_batches=attempted_batches,
        skipped_batches=skipped_batches,
        loss_history=tuple(loss_history),
        final_checkpoint_path=final_path,
        latest_checkpoint_path=latest_path,
        run_id=effective_run_id,
        fold=fold,
        data_fingerprint=str(identity_payload.get("data_fingerprint"))
        if identity_payload.get("data_fingerprint") is not None
        else None,
        best_metric=best_metric,
        epoch=epoch,
        execution_attempt_id=attempt_id,
        resumed_from_checkpoint=str(resume_path) if resume_path is not None else None,
        replayed_batches=replayed_batches,
        checkpoint_events=tuple(dict(event) for event in checkpoint_events),
        semantic_fingerprint_value=semantic_identity,
    )


def _train_fixed_budget_gpu(
    examples: Sequence[TrainingExample],
    config: TrainingConfig,
    *,
    model: HierarchicalStudyModel | None = None,
    artifact_dir: str | Path | None = None,
    fold: int | None = None,
    run_id: str | None = None,
    identity: Mapping[str, object] | None = None,
    resume_from: str | Path | None = None,
    execution_attempt_id: str | None = None,
    requested_resources: Mapping[str, object] | None = None,
    parent_provenance: Mapping[str, object] | None = None,
    amp_scaler: Any | None = None,
    scaler: Any | None = None,
    best_metric: float | None = None,
    checkpoint_clock: Callable[[], float] | None = None,
    interrupt_after_steps: int | None = None,
) -> TrainingResult:
    """Private CUDA seam entered only after the GPU adapter's approval gates."""

    if config.device == "cpu" or not config.device.startswith("cuda"):
        raise TrainingConfigurationError("the private GPU seam requires a CUDA device")
    return _train_fixed_budget_impl(
        examples,
        config,
        model=model,
        artifact_dir=artifact_dir,
        fold=fold,
        run_id=run_id,
        identity=identity,
        resume_from=resume_from,
        execution_attempt_id=execution_attempt_id,
        requested_resources=requested_resources,
        gpu_authorization=_GPU_EXECUTION_CAPABILITY,
        parent_provenance=parent_provenance,
        amp_scaler=amp_scaler,
        scaler=scaler,
        best_metric=best_metric,
        checkpoint_clock=checkpoint_clock,
        interrupt_after_steps=interrupt_after_steps,
    )


def resume_training(
    examples: Sequence[TrainingExample],
    checkpoint: str | Path,
    config: TrainingConfig = TrainingConfig(),
    *,
    artifact_dir: str | Path | None = None,
    **kwargs: Any,
) -> TrainingResult:
    """Resume a run from a complete latest-resumable checkpoint."""

    return train_fixed_budget(
        examples,
        config,
        artifact_dir=artifact_dir,
        resume_from=checkpoint,
        **kwargs,
    )


resume_fixed_budget = resume_training


def fork_training_run(
    examples: Sequence[TrainingExample],
    parent_checkpoint: str | Path,
    config: TrainingConfig,
    artifact_dir: str | Path,
    *,
    identity: Mapping[str, object],
    fold: int | None = None,
    execution_attempt_id: str | None = None,
    checkpoint_clock: Callable[[], float] | None = None,
) -> TrainingResult:
    """Start a new run with explicit provenance from an incompatible parent.

    A fork deliberately starts from deterministic initialization.  Reusing a
    parent checkpoint's weights after a semantic change would make the new
    initialization ambiguous, so warm starts are not implicit.
    """

    parent_path = Path(parent_checkpoint)
    if parent_path.is_dir() or parent_path.name == "latest.pt":
        parent_path = find_latest_resumable_checkpoint(parent_path)
    parent = load_training_checkpoint(
        parent_path,
        expected_checkpoint_kind="latest_resumable",
    )
    parent_identity = parent.get("semantic_fingerprint")
    child_identity = semantic_fingerprint(config, identity)
    if parent_identity == child_identity:
        raise TrainingConfigurationError(
            "an unchanged scientific identity should resume the parent run instead of forking"
        )
    parent_run_id = parent.get("run_id")
    provenance = {
        "parent_run_id": parent_run_id,
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": checkpoint_sha256(parent_path),
        "parent_semantic_fingerprint": parent_identity,
    }
    return train_fixed_budget(
        examples,
        config,
        artifact_dir=artifact_dir,
        fold=fold,
        identity=identity,
        parent_provenance=provenance,
        execution_attempt_id=execution_attempt_id,
        checkpoint_clock=checkpoint_clock,
    )


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _code_revision() -> str | None:
    configured = os.environ.get("RSNA_CODE_REVISION")
    if configured and configured.strip():
        return configured.strip()
    repository = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


def collect_environment_record() -> dict[str, object]:
    """Capture interpreter, package, OS, and Slurm attempt identity."""

    record: dict[str, object] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "python_prefix": sys.prefix,
        "python_base_prefix": sys.base_prefix,
        "python_isolated_venv": sys.prefix != sys.base_prefix,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "pydicom_version": _package_version("pydicom"),
        "pytest_version": _package_version("pytest"),
        "mypy_version": _package_version("mypy"),
        "code_revision": _code_revision(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        "slurm_mem_per_cpu": os.environ.get("SLURM_MEM_PER_CPU"),
        "slurm_submit_dir": os.environ.get("SLURM_SUBMIT_DIR"),
    }
    record["environment_fingerprint"] = _json_sha256(record)
    return record


def _inference_import_check() -> tuple[bool, str]:
    """Check the inference module in a fresh interpreter for training imports."""

    code = (
        "import sys; from rsna_knee.inference import InferenceEngine; "
        "raise SystemExit(1 if 'rsna_knee.training' in sys.modules else 0)"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    if completed.returncode != 0:
        return False, completed.stderr.strip() or "inference import loaded training orchestration"
    return True, "inference import completed without rsna_knee.training"


def run_cpu_preflight(
    *,
    output_path: str | Path | None = None,
    require_isolated_python: bool = True,
    require_slurm: bool = False,
) -> CpuPreflightResult:
    """Run the finite-loss/checkpoint/import checks that gate real training."""

    requested_threads = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    try:
        thread_count = max(1, min(16, int(requested_threads)))
    except ValueError:
        thread_count = 1
    torch.set_num_threads(thread_count)
    environment = collect_environment_record()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    checks["python_311"] = sys.version_info[:2] == (3, 11)
    checks["user_environment_python"] = bool(environment["python_isolated_venv"])
    if not require_isolated_python:
        checks["user_environment_python"] = True
    checks["cluster1_cpu_slurm"] = (
        not require_slurm
        or (bool(environment["slurm_job_id"]) and environment["slurm_partition"] == "cluster1-cpu")
    )
    checks["core_dependencies_import"] = False
    checks["known_roc_auc"] = False
    checks["finite_model_loss_and_gradients"] = False
    checks["checkpoint_save_reload"] = False
    checks["inference_import_isolated"] = False
    try:
        import pydicom  # noqa: F401

        checks["core_dependencies_import"] = all(
            item is not None
            for item in (
                environment["numpy_version"],
                environment["torch_version"],
                environment["pydicom_version"],
            )
        )
        from .evaluation import roc_auc_score

        known = (
            roc_auc_score((0, 1), (0.1, 0.9)) == 1.0
            and roc_auc_score((0, 1), (0.9, 0.1)) == 0.0
            and roc_auc_score((0, 1), (0.5, 0.5)) == 0.5
        )
        checks["known_roc_auc"] = known
        tiny_model = HierarchicalStudyModel(ModelConfig(image_size=(16, 16), channels=(2, 4, 8)))
        tiny_optimizer = torch.optim.Adam(tiny_model.parameters(), lr=0.001)
        tiny_scheduler = torch.optim.lr_scheduler.LambdaLR(tiny_optimizer, lr_lambda=lambda _: 1.0)
        tiny_images = torch.rand((2, 1, 1, 1, 16, 16))
        tiny_slice_mask = torch.ones((2, 1, 1), dtype=torch.bool)
        tiny_series_mask = torch.ones((2, 1), dtype=torch.bool)
        tiny_targets = torch.zeros((2, NUM_TARGETS))
        tiny_targets[:, 0] = 1.0
        tiny_logits = tiny_model(tiny_images, tiny_slice_mask, tiny_series_mask)
        tiny_loss = masked_bce_loss(
            tiny_logits,
            tiny_targets,
            torch.ones_like(tiny_targets, dtype=torch.bool),
        )
        if tiny_loss is not None and bool(torch.isfinite(tiny_loss)):
            tiny_loss.backward()
            checks["finite_model_loss_and_gradients"] = _finite_gradients(tiny_model)
        with tempfile.TemporaryDirectory(prefix="rsna-ticket08-preflight-") as temporary:
            checkpoint = Path(temporary) / "preflight.pt"
            preflight_config = TrainingConfig(
                image_size=(16, 16),
                channels=(2, 4, 8),
                max_slices=1,
                max_series=1,
                successful_optimizer_steps=1,
            )
            save_training_checkpoint(
                checkpoint,
                tiny_model,
                tiny_optimizer,
                tiny_scheduler,
                preflight_config,
                successful_optimizer_steps=0,
                attempted_batches=0,
                skipped_batches=0,
                loss_history=(),
                overwrite=False,
            )
            reloaded = HierarchicalStudyModel(ModelConfig(image_size=(16, 16), channels=(2, 4, 8)))
            loaded = load_training_checkpoint(checkpoint, model=reloaded)
            checks["checkpoint_save_reload"] = isinstance(loaded.get("model_state_dict"), dict)
        isolated, message = _inference_import_check()
        checks["inference_import_isolated"] = isolated
        details["inference_import"] = message
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, CheckpointError) as exc:
        details["failure"] = str(exc)
    details["required_isolated_python"] = require_isolated_python
    details["required_slurm"] = require_slurm
    passed = all(checks.values())
    result = CpuPreflightResult(passed=passed, checks=checks, environment=environment, details=details)
    if output_path is not None:
        atomic_json_write(output_path, result.to_dict())
    return result


def require_cpu_preflight(
    result: CpuPreflightResult,
    *,
    require_slurm: bool = True,
    require_isolated_python: bool = True,
) -> None:
    """Gate real training on a complete preflight result."""

    required_checks = {
        "python_311",
        "user_environment_python",
        "cluster1_cpu_slurm",
        "core_dependencies_import",
        "known_roc_auc",
        "finite_model_loss_and_gradients",
        "checkpoint_save_reload",
        "inference_import_isolated",
    }
    failed = sorted(
        name for name in required_checks if result.checks.get(name) is not True
    )
    if require_isolated_python and result.details.get("required_isolated_python") is not True:
        failed.append("required_isolated_python")
    if require_slurm and result.details.get("required_slurm") is not True:
        failed.append("required_slurm")
    if any(value is not True for value in result.checks.values()):
        failed.extend(
            name for name, passed in result.checks.items() if passed is not True and name not in failed
        )
    if not result.passed or failed:
        raise PreflightError(f"CPU preflight has not passed: {', '.join(failed)}")


def _public_run_environment() -> dict[str, object]:
    environment = collect_environment_record()
    return {
        key: value
        for key, value in environment.items()
        if key
        not in {
            "python_executable",
            "python_prefix",
            "python_base_prefix",
            "hostname",
            "slurm_submit_dir",
        }
    }


def _train_one_fold_impl(
    studies: Mapping[str, StudyRecord] | Sequence[StudyRecord],
    fold_manifest: FoldManifest,
    fold: int,
    output_dir: str | Path,
    *,
    config: TrainingConfig = TrainingConfig(),
    source_version: str = DEFAULT_SOURCE_VERSION,
    source_fingerprint: str | None = None,
    cache_fingerprint: str | None = None,
    integration_manifest_checksum: str | None = None,
    series_cache: SeriesCache | None = None,
    preprocess_config: PreprocessConfig | None = None,
    packing_config: PackingConfig | None = None,
    preflight: CpuPreflightResult | None = None,
    require_preflight: bool = True,
    allow_local_smoke: bool = False,
    allow_gpu: bool = False,
    requested_resources: Mapping[str, object] | None = None,
    gpu_authorization: object | None = None,
    device: str | torch.device | None = None,
    resume_from: str | Path | None = None,
    execution_attempt_id: str | None = None,
    interrupt_after_steps: int | None = None,
) -> TrainingResult:
    """Train the reference-only partition and evaluate the reloaded final checkpoint."""

    if fold not in FOLD_IDS:
        raise TrainingDataError(f"fold must be one of {FOLD_IDS}")
    if not source_version.strip():
        raise TrainingConfigurationError("source_version must be non-empty")
    if require_preflight and not (source_fingerprint or "").strip():
        raise TrainingConfigurationError(
            "real training requires a checksum-backed source fingerprint"
        )
    if require_preflight and not (integration_manifest_checksum or "").strip():
        raise TrainingConfigurationError(
            "real training requires the checksum of its integration manifest"
        )
    if not require_preflight and not (
        allow_local_smoke and source_version.startswith("synthetic-")
    ):
        raise PreflightError(
            "unpreflighted fold training is limited to an explicit synthetic local smoke run"
        )
    if preprocess_config is not None:
        if config.preprocessing_config is not None and config.preprocessing_config != preprocess_config:
            raise TrainingConfigurationError(
                "the supplied preprocessing policy conflicts with TrainingConfig"
            )
        config = replace(config, preprocessing_config=preprocess_config)
    if packing_config is not None and packing_config != config.packing_config():
        raise TrainingConfigurationError(
            "the supplied packing policy conflicts with TrainingConfig"
        )
    if device is not None and str(device) != config.device:
        config = replace(config, device=str(device))
    if config.device != "cpu":
        if not allow_gpu or gpu_authorization is not _GPU_EXECUTION_CAPABILITY:
            raise TrainingConfigurationError(
                "GPU fold training requires the approval-gated GPU execution path"
            )
        if not require_preflight:
            raise PreflightError("GPU fold training always requires a passed CPU preflight")
    if allow_gpu and config.device != "cuda" and not config.device.startswith("cuda:"):
        raise TrainingConfigurationError("GPU-enabled fold training requires a CUDA device")
    if require_preflight:
        if preflight is None:
            raise PreflightError("a passed CPU preflight result is required before real training")
        require_cpu_preflight(
            preflight,
            require_slurm=not allow_local_smoke,
            require_isolated_python=not allow_local_smoke,
        )
        if not allow_local_smoke and preflight.details.get("required_slurm") is not True:
            raise PreflightError(
                "real training requires a CPU preflight executed with the Slurm gate enabled"
            )
    training_examples = build_fold_training_examples(
        studies,
        fold_manifest,
        fold,
        preprocess_config=config.preprocess_config(),
        packing_config=config.packing_config(),
        series_cache=series_cache,
    )
    held_out_examples = build_fold_held_out_examples(
        studies,
        fold_manifest,
        fold,
        preprocess_config=config.preprocess_config(),
        packing_config=config.packing_config(),
        series_cache=series_cache,
    )
    held_out_uids = tuple(example.study_instance_uid for example in held_out_examples)
    training_uids = tuple(example.study_instance_uid for example in training_examples)
    if set(training_uids).intersection(held_out_uids):
        raise TrainingDataError("training and held-out study partitions overlap")
    identity = _checkpoint_identity(
        config,
        training_examples,
        source_version=source_version,
        source_fingerprint=source_fingerprint,
        cache_fingerprint=(
            cache_fingerprint
            or (
                _json_sha256({"complete_keys": list(series_cache.list_complete_keys())})
                if series_cache is not None
                else "not-used"
            )
        ),
        fold_manifest_checksum=fold_manifest.manifest_checksum,
        integration_manifest_checksum=integration_manifest_checksum,
        fold=fold,
    )
    identity["held_out_data_fingerprint"] = fingerprint_training_examples(held_out_examples)
    identity["held_out_inputs_excluded_from_fitting"] = True
    identity["learned_normalization_fit_studies"] = []
    run_id = _default_run_id(identity, config)
    scientific_identity = semantic_fingerprint(config, identity)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if resume_from is None and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing experiment directory {destination}")
    model_config = config.model_config()
    if resume_from is None:
        atomic_json_write(
            destination / "configuration.json",
            {
                "schema_version": TRAINING_SCHEMA_VERSION,
                "run_id": run_id,
                "fold": fold,
                "config": config.to_dict(),
                "scientific_configuration_fingerprint": config.config_fingerprint,
                "semantic_fingerprint": scientific_identity,
                "identity": identity,
                "environment": collect_environment_record(),
                "resources": {
                    **dict(requested_resources or _default_requested_resources()),
                    "slurm_attempt_id": os.environ.get("SLURM_JOB_ID"),
                },
                "supervision": {
                    "kind": "reference",
                    "fold_manifest_checksum": fold_manifest.manifest_checksum,
                    "training_studies": list(training_uids),
                    "held_out_studies": list(held_out_uids),
                    "held_out_inputs_excluded": True,
                    "reports_used": False,
                },
                "inference_configuration": {
                    "model": {
                        "image_size": list(model_config.image_size),
                        "channels": list(model_config.channels),
                        "input_channels": model_config.input_channels,
                        "num_targets": model_config.num_targets,
                    },
                    "preprocessing": config.preprocess_config().cache_config(),
                    "packing": config.packing_config().to_dict(),
                    "checkpoint_selection": "final_checkpoint",
                },
                "preflight": preflight.to_dict() if preflight is not None else None,
            },
        )
    else:
        configuration_path = destination / "configuration.json"
        if not configuration_path.is_file():
            raise TrainingConfigurationError(
                "cannot resume without the original immutable configuration record"
            )
        try:
            saved_configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CheckpointError("could not read the immutable experiment configuration") from exc
        if not isinstance(saved_configuration, Mapping):
            raise CheckpointError("experiment configuration is not an object")
        if saved_configuration.get("semantic_fingerprint") is None:
            if not _legacy_configuration_matches(
                saved_configuration,
                expected_config=config,
                expected_identity=identity,
            ):
                raise TrainingConfigurationError(
                    "legacy experiment configuration does not match the requested scientific run"
                )
            legacy_run_id = saved_configuration.get("run_id")
            if not isinstance(legacy_run_id, str) or not legacy_run_id:
                raise CheckpointError("legacy experiment configuration has no run identity")
            run_id = legacy_run_id
        elif saved_configuration.get("run_id") != run_id:
            raise TrainingConfigurationError("experiment run identity does not match the requested run")
        elif saved_configuration.get("semantic_fingerprint") != scientific_identity:
            raise TrainingConfigurationError(
                "experiment semantic fingerprint does not match the requested run"
            )
    result = _train_fixed_budget_impl(
        training_examples,
        config,
        artifact_dir=destination,
        fold=fold,
        run_id=run_id,
        identity=identity,
        resume_from=resume_from,
        execution_attempt_id=execution_attempt_id,
        requested_resources=requested_resources,
        gpu_authorization=gpu_authorization,
        interrupt_after_steps=interrupt_after_steps,
    )
    if result.final_checkpoint_path is None:
        raise CheckpointError("fixed-budget run did not produce a final checkpoint")
    load_training_checkpoint(
        result.final_checkpoint_path,
        model=result.model,
        expected_config=config,
        expected_fold=fold,
        expected_identity=identity,
        expected_checkpoint_kind="final_selected",
    )
    from .evaluation import evaluate_fold_predictions, predict_training_examples

    predictions = predict_training_examples(result.model, held_out_examples, device=config.device)
    evaluation = evaluate_fold_predictions(
        cast(Mapping[str, Sequence[float]], predictions), fold_manifest, fold
    )
    completed = TrainingResult(
        model=result.model,
        config=result.config,
        successful_optimizer_steps=result.successful_optimizer_steps,
        scheduler_steps=result.scheduler_steps,
        attempted_batches=result.attempted_batches,
        skipped_batches=result.skipped_batches,
        loss_history=result.loss_history,
        final_checkpoint_path=result.final_checkpoint_path,
        latest_checkpoint_path=result.latest_checkpoint_path,
        run_id=run_id,
        fold=fold,
        evaluation=evaluation,
        predictions=predictions,
        training_study_uids=training_uids,
        held_out_study_uids=held_out_uids,
        data_fingerprint=str(identity["data_fingerprint"]),
        best_metric=result.best_metric,
        epoch=result.epoch,
        execution_attempt_id=result.execution_attempt_id,
        resumed_from_checkpoint=result.resumed_from_checkpoint,
        replayed_batches=result.replayed_batches,
        checkpoint_events=result.checkpoint_events,
        semantic_fingerprint_value=scientific_identity,
    )
    atomic_json_write(destination / "metrics.json", evaluation.to_private_dict())
    atomic_json_write(destination / "run.json", completed.to_private_dict())
    return completed


def train_one_fold(
    studies: Mapping[str, StudyRecord] | Sequence[StudyRecord],
    fold_manifest: FoldManifest,
    fold: int,
    output_dir: str | Path,
    *,
    config: TrainingConfig = TrainingConfig(),
    source_version: str = DEFAULT_SOURCE_VERSION,
    source_fingerprint: str | None = None,
    cache_fingerprint: str | None = None,
    integration_manifest_checksum: str | None = None,
    series_cache: SeriesCache | None = None,
    preprocess_config: PreprocessConfig | None = None,
    packing_config: PackingConfig | None = None,
    preflight: CpuPreflightResult | None = None,
    require_preflight: bool = True,
    allow_local_smoke: bool = False,
    requested_resources: Mapping[str, object] | None = None,
    device: str | torch.device | None = None,
    resume_from: str | Path | None = None,
    execution_attempt_id: str | None = None,
    interrupt_after_steps: int | None = None,
) -> TrainingResult:
    """Train one reference fold on CPU.

    CUDA fold training is deliberately private to the approval-gated GPU
    adapter; this public seam cannot opt into it with a keyword or sentinel.
    """

    effective_device = str(device) if device is not None else config.device
    if effective_device != "cpu":
        raise TrainingConfigurationError(
            "CUDA fold training is available only through the approval-gated GPU runner"
        )
    return _train_one_fold_impl(
        studies,
        fold_manifest,
        fold,
        output_dir,
        config=config,
        source_version=source_version,
        source_fingerprint=source_fingerprint,
        cache_fingerprint=cache_fingerprint,
        integration_manifest_checksum=integration_manifest_checksum,
        series_cache=series_cache,
        preprocess_config=preprocess_config,
        packing_config=packing_config,
        preflight=preflight,
        require_preflight=require_preflight,
        allow_local_smoke=allow_local_smoke,
        allow_gpu=False,
        gpu_authorization=None,
        requested_resources=requested_resources,
        device=device,
        resume_from=resume_from,
        execution_attempt_id=execution_attempt_id,
        interrupt_after_steps=interrupt_after_steps,
    )


def _train_one_fold_gpu(
    studies: Mapping[str, StudyRecord] | Sequence[StudyRecord],
    fold_manifest: FoldManifest,
    fold: int,
    output_dir: str | Path,
    *,
    config: TrainingConfig,
    source_version: str,
    source_fingerprint: str,
    integration_manifest_checksum: str,
    series_cache: SeriesCache | None = None,
    preprocess_config: PreprocessConfig | None = None,
    packing_config: PackingConfig | None = None,
    preflight: CpuPreflightResult,
    cache_fingerprint: str | None = None,
    requested_resources: Mapping[str, object] | None = None,
    device: str | torch.device | None = None,
    resume_from: str | Path | None = None,
    execution_attempt_id: str | None = None,
    interrupt_after_steps: int | None = None,
) -> TrainingResult:
    """Private CUDA fold seam entered after request, approval, and preflight checks."""

    effective_device = str(device) if device is not None else config.device
    if effective_device == "cpu" or not effective_device.startswith("cuda"):
        raise TrainingConfigurationError("the private GPU fold seam requires a CUDA device")
    return _train_one_fold_impl(
        studies,
        fold_manifest,
        fold,
        output_dir,
        config=config,
        source_version=source_version,
        source_fingerprint=source_fingerprint,
        cache_fingerprint=cache_fingerprint,
        integration_manifest_checksum=integration_manifest_checksum,
        series_cache=series_cache,
        preprocess_config=preprocess_config,
        packing_config=packing_config,
        preflight=preflight,
        require_preflight=True,
        allow_local_smoke=False,
        allow_gpu=True,
        gpu_authorization=_GPU_EXECUTION_CAPABILITY,
        requested_resources=requested_resources,
        device=device,
        resume_from=resume_from,
        execution_attempt_id=execution_attempt_id,
        interrupt_after_steps=interrupt_after_steps,
    )


def _parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=FOLD_IDS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--source-fingerprint", default=None)
    parser.add_argument("--preflight-evidence", type=Path, default=None)
    parser.add_argument("--evidence", type=Path, default=None)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="resume the immutable experiment from latest.pt or a checkpoint history path",
    )
    parser.add_argument("--execution-attempt-id", default=None)
    parser.add_argument(
        "--allow-superseded-fold-manifest",
        action="store_true",
        help="replay a superseded version-1 fold manifest for historical evidence only",
    )
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("--allow-local-smoke", action="store_true")
    return parser


def _check_training_execution_context(*, require_slurm: bool, allow_local_smoke: bool) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if require_slurm and (not job_id or partition != "cluster1-cpu"):
        raise PreflightError("ticket 08 training requires a cluster1-cpu Slurm allocation")
    if not job_id and not allow_local_smoke:
        raise PreflightError(
            "refusing training outside Slurm; use --allow-local-smoke only for a tiny synthetic check"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the bounded ticket-08 training adapter."""

    args = _parser().parse_args(argv)
    try:
        _check_training_execution_context(
            require_slurm=args.require_slurm,
            allow_local_smoke=args.allow_local_smoke,
        )
        from .folds import load_fold_manifest
        from .real_inference import iter_real_studies, load_integration_manifest

        fold_manifest = load_fold_manifest(args.fold_manifest)
        if fold_manifest.superseded and not args.allow_superseded_fold_manifest:
            raise TrainingDataError(
                "the fold manifest was produced by the superseded version-1 generator; "
                "train against the version-2 manifest"
            )
        integration_manifest = load_integration_manifest(args.integration_manifest)
        source = args.source
        if not source.exists():
            raise TrainingDataError(f"real-DICOM source does not exist: {source}")
        if integration_manifest.source_version != args.source_version:
            raise TrainingDataError(
                "source version disagrees with the integration manifest"
            )
        if fold_manifest.source_version != args.source_version:
            raise TrainingDataError("source version disagrees with the fold manifest")
        if integration_manifest.source_manifest_checksum != fold_manifest.source_manifest_checksum:
            raise TrainingDataError(
                "integration and fold manifests have different source-manifest checksums"
            )
        if (
            args.source_fingerprint is not None
            and args.source_fingerprint != integration_manifest.archive_fingerprint
        ):
            raise TrainingDataError(
                "source fingerprint disagrees with the integration manifest"
            )
        required_gold = {
            item.study_instance_uid for item in fold_manifest.gold_memberships
        }
        if not required_gold.issubset(set(integration_manifest.study_uids)):
            raise TrainingDataError(
                "integration manifest does not cover every frozen gold study"
            )
        preflight = run_cpu_preflight(
            output_path=args.preflight_evidence,
            require_isolated_python=not args.allow_local_smoke,
            require_slurm=args.require_slurm,
        )
        require_cpu_preflight(preflight)
        studies: dict[str, StudyRecord] = {}
        for loaded in iter_real_studies(
            integration_manifest,
            source,
            require_complete_series=True,
            allow_unverified_directory=args.allow_local_smoke,
        ):
            studies[loaded.study.study_instance_uid] = loaded.study
        cache = SeriesCache(args.cache_root) if args.cache_root is not None else None
        result = train_one_fold(
            studies,
            fold_manifest,
            args.fold,
            args.output,
            source_version=args.source_version,
            source_fingerprint=args.source_fingerprint or integration_manifest.archive_fingerprint,
            integration_manifest_checksum=integration_manifest.manifest_checksum,
            series_cache=cache,
            preprocess_config=integration_manifest.preprocess_config,
            packing_config=integration_manifest.packing_config,
            preflight=preflight,
            allow_local_smoke=args.allow_local_smoke,
            resume_from=args.resume_from,
            execution_attempt_id=args.execution_attempt_id,
        )
        if args.evidence is not None:
            atomic_json_write(
                args.evidence,
                {
                    **result.to_public_dict(),
                    "environment": _public_run_environment(),
                    "source_version": args.source_version,
                    "fold_manifest_checksum": fold_manifest.manifest_checksum,
                    "preflight_passed": preflight.passed,
                },
            )
        print(json.dumps(result.to_public_dict(), indent=2, sort_keys=True))
    except (OSError, RuntimeError, ValueError, TrainingError) as exc:
        print(f"ticket-08 training failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointError",
    "LEGACY_CHECKPOINT_SCHEMA_VERSION",
    "RetryClassification",
    "CpuPreflightResult",
    "DEFAULT_BATCH_ORDER_SEED",
    "DEFAULT_SOURCE_VERSION",
    "DEFAULT_TRAINING_SEED",
    "PREFLIGHT_SCHEMA_VERSION",
    "PreflightError",
    "SupervisionProvenance",
    "TrainingConfig",
    "TrainingConfigurationError",
    "TrainingDataError",
    "TrainingError",
    "TrainingExample",
    "TrainingInterrupted",
    "TrainingNumericsError",
    "TrainingProgressError",
    "TrainingResult",
    "atomic_torch_save",
    "build_fold_examples",
    "build_fold_held_out_examples",
    "build_fold_training_examples",
    "checkpoint_sha256",
    "classify_retry_failure",
    "find_latest_resumable_checkpoint",
    "collect_environment_record",
    "fork_training_run",
    "fingerprint_training_examples",
    "load_training_checkpoint",
    "masked_bce_loss",
    "masked_binary_cross_entropy_with_logits",
    "require_cpu_preflight",
    "resume_fixed_budget",
    "resume_training",
    "retry_is_allowed",
    "run_cpu_preflight",
    "save_training_checkpoint",
    "seed_everything",
    "semantic_fingerprint",
    "train_fixed_budget",
    "train_one_fold",
]
