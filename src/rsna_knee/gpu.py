"""Approval-gated Slurm GPU experiment orchestration.

The module owns the boundary between a passed CPU preflight, an immutable
scientific request, and an explicitly approved GPU allocation.  It records the
resource facts observed by the job and delegates model training to the shared
training implementation; it does not provide a second model or data path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import fcntl
import getpass
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, cast

import torch

from .archive import atomic_json_write
from .model import COMPACT_MODEL_ID, COMPACT_POOLING_ID, RESNET18_MODEL_ID
from .pretrained import (
    PretrainedResourceError,
    RESNET18_ARTIFACT_BYTES,
    RESNET18_ARTIFACT_ID,
    RESNET18_DOWNLOAD_URL,
    RESNET18_SHA256,
    RESNET18_SOURCE_REVISION,
    TICKET33_RESOURCE_SCOPE,
    TICKET39_RESOURCE_SCOPE,
    load_resource_approval,
)
from .resnet18 import RESNET18_ENCODER_ID, RESNET18_FINETUNE_MODEL_ID
from .gpu_campaign import (
    GPU_CAMPAIGN_BUDGET_HOURS,
    GpuCampaignActiveRequestError,
    GpuCampaignBudgetError,
    GpuCampaignLedgerError,
    begin_gpu_submission_attempt,
    check_gpu_campaign_capacity,
    finish_gpu_campaign_request,
    initialize_gpu_campaign_ledger,
    record_gpu_allocation,
    record_gpu_submission_result,
    reserve_gpu_campaign_request,
)
from .training import (
    PREFLIGHT_SCHEMA_VERSION,
    CpuPreflightResult,
    PreflightError,
    RetryClassification,
    TrainingConfig,
    TrainingResult,
    checkpoint_sha256,
    classify_retry_failure,
    load_training_checkpoint,
    require_cpu_preflight,
    _train_fixed_budget_gpu,
    _train_one_fold_gpu,
    _new_gpu_execution_authorization,
)


GPU_EXPERIMENT_SCHEMA_VERSION = "rsna-knee-gpu-experiment-v2"
GPU_APPROVAL_SCHEMA_VERSION = "rsna-knee-gpu-owner-approval-v1"
GPU_EXECUTION_SCHEMA_VERSION = "rsna-knee-gpu-execution-v1"
GPU_SUBMISSION_SCHEMA_VERSION = "rsna-knee-gpu-submission-v1"
GPU_EXECUTION_RECORD_SCHEMA_VERSION = "rsna-knee-gpu-execution-record-v1"
GPU_PREFLIGHT_SCHEMA_VERSION = "rsna-knee-gpu-preflight-v1"
GPU_APPROVAL_SCOPE = "ticket-13-approved-gpu-experiment"
GPU_DEFAULT_PARTITION = "cluster1-gpu"
GPU_SMOKE_PARTITION = "cluster1-gpu-add"
GPU_ALLOWED_PARTITIONS = (GPU_DEFAULT_PARTITION, GPU_SMOKE_PARTITION)
GPU_RETRY_LIMIT = 1
GPU_SMOKE_WALL_TIME_LIMIT = 30
GPU_FULL_WALL_TIME_LIMIT = 6 * 60


class GpuExperimentError(RuntimeError):
    """Base class for unsafe or unsuccessful GPU experiment operations."""


class GpuConfigurationError(ValueError, GpuExperimentError):
    """Raised when a GPU request is malformed or exceeds the safe envelope."""


class GpuApprovalError(GpuExperimentError):
    """Raised when a CPU preflight or owner approval is missing or mismatched."""


class GpuResourceError(GpuExperimentError):
    """Raised when the observed Slurm/CUDA resource is not the approved one."""


class GpuExecutionError(GpuExperimentError):
    """Raised when an approved GPU execution cannot complete safely."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> Mapping[str, object]:
    if not path.is_file():
        raise GpuApprovalError(f"{description} is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GpuApprovalError(f"could not read {description}: {path}") from exc
    if not isinstance(payload, Mapping):
        raise GpuApprovalError(f"{description} must contain a JSON object")
    return cast(Mapping[str, object], payload)


def _nonempty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GpuConfigurationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _safe_identifier(value: object, field_name: str) -> str:
    identifier = _nonempty(value, field_name)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", identifier):
        raise GpuConfigurationError(f"{field_name} contains unsafe characters")
    return identifier


def _parse_gpu_count(gres: str) -> int:
    match = re.fullmatch(r"gpu(?::[A-Za-z0-9_.-]+)?:(\d+)", gres)
    if match is None:
        raise GpuConfigurationError("gres must use Slurm's gpu[:type]:count syntax")
    return int(match.group(1))


@dataclass(frozen=True)
class GpuResourceRequest:
    """One explicit, single-node, single-GPU Slurm resource request."""

    partition: str = GPU_DEFAULT_PARTITION
    gres: str = "gpu:1"
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int = 16
    memory_gb: int = 64
    wall_time_minutes: int = 360
    exclusive: bool = False
    smoke_test: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.partition, str) or not self.partition.strip():
            raise GpuConfigurationError("partition must be a non-empty string")
        if not isinstance(self.gres, str) or not self.gres.strip():
            raise GpuConfigurationError("gres must be a non-empty string")
        integer_fields = (
            self.nodes,
            self.ntasks,
            self.cpus_per_task,
            self.memory_gb,
            self.wall_time_minutes,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_fields):
            raise GpuConfigurationError("GPU resource integer fields must be integers")
        if self.partition not in GPU_ALLOWED_PARTITIONS:
            raise GpuConfigurationError(
                f"partition must be one of {GPU_ALLOWED_PARTITIONS}; no fallback is allowed"
            )
        count = _parse_gpu_count(self.gres)
        if count != 1:
            raise GpuConfigurationError("this ticket permits exactly one GPU; multi-GPU needs a new approval")
        if self.nodes != 1 or self.ntasks != 1:
            raise GpuConfigurationError("GPU experiments are bounded to one node and one task")
        if not 1 <= self.cpus_per_task <= 16:
            raise GpuConfigurationError("cpus_per_task must be between 1 and 16")
        if not 1 <= self.memory_gb <= 64:
            raise GpuConfigurationError("memory_gb must be between 1 and 64")
        if not 1 <= self.wall_time_minutes <= GPU_FULL_WALL_TIME_LIMIT:
            raise GpuConfigurationError("wall_time_minutes must be between 1 minute and six hours")
        if not isinstance(self.exclusive, bool) or not isinstance(self.smoke_test, bool):
            raise GpuConfigurationError("exclusive and smoke_test must be boolean")
        if self.partition == GPU_SMOKE_PARTITION and not self.smoke_test:
            raise GpuConfigurationError("cluster1-gpu-add is reserved for an approved GPU smoke test")
        if self.smoke_test and self.wall_time_minutes > GPU_SMOKE_WALL_TIME_LIMIT:
            raise GpuConfigurationError(
                f"GPU smoke tests must request at most {GPU_SMOKE_WALL_TIME_LIMIT} minutes"
            )

    @property
    def gpu_count(self) -> int:
        return _parse_gpu_count(self.gres)

    def to_dict(self) -> dict[str, object]:
        return {
            "partition": self.partition,
            "gres": self.gres,
            "gpu_count": self.gpu_count,
            "nodes": self.nodes,
            "ntasks": self.ntasks,
            "cpus_per_task": self.cpus_per_task,
            "memory_gb": self.memory_gb,
            "wall_time_minutes": self.wall_time_minutes,
            "exclusive": self.exclusive,
            "smoke_test": self.smoke_test,
        }

    @classmethod
    def from_dict(cls, value: object) -> "GpuResourceRequest":
        if not isinstance(value, Mapping):
            raise GpuConfigurationError("GPU resource request must be an object")
        integer_fields = ("nodes", "ntasks", "cpus_per_task", "memory_gb", "wall_time_minutes")
        boolean_fields = ("exclusive", "smoke_test")
        if any(
            not isinstance(value.get(field_name), int) or isinstance(value.get(field_name), bool)
            for field_name in integer_fields
        ):
            raise GpuConfigurationError("GPU resource request integer fields are invalid")
        if any(not isinstance(value.get(field_name), bool) for field_name in boolean_fields):
            raise GpuConfigurationError("GPU resource request boolean fields are invalid")
        try:
            request = cls(
                partition=value["partition"],  # type: ignore[arg-type]
                gres=value["gres"],  # type: ignore[arg-type]
                nodes=value["nodes"],  # type: ignore[arg-type]
                ntasks=value["ntasks"],  # type: ignore[arg-type]
                cpus_per_task=value["cpus_per_task"],  # type: ignore[arg-type]
                memory_gb=value["memory_gb"],  # type: ignore[arg-type]
                wall_time_minutes=value["wall_time_minutes"],  # type: ignore[arg-type]
                exclusive=value["exclusive"],  # type: ignore[arg-type]
                smoke_test=value["smoke_test"],  # type: ignore[arg-type]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GpuConfigurationError("GPU resource request fields are invalid") from exc
        if value.get("gpu_count") != request.gpu_count:
            raise GpuConfigurationError("GPU resource request count is inconsistent with gres")
        return request


@dataclass(frozen=True)
class GpuExperimentRequest:
    """An immutable GPU request that can be approved by checksum."""

    root: Path
    request_path: Path
    request_checksum: str
    payload: Mapping[str, object]

    @property
    def resources(self) -> GpuResourceRequest:
        experiment = self.payload.get("resources")
        return GpuResourceRequest.from_dict(experiment)

    @property
    def training_config(self) -> TrainingConfig:
        experiment = self.payload.get("experiment")
        if not isinstance(experiment, Mapping):
            raise GpuConfigurationError("GPU request experiment section is invalid")
        return _training_config_from_dict(experiment.get("training_config"))


@dataclass(frozen=True)
class GpuApproval:
    """A checksum-bound owner authorization for exactly one request."""

    path: Path
    approval_checksum: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class GpuSubmissionResult:
    """Scheduler outcome, including the bounded retry decision."""

    status: str
    job_ids: tuple[str, ...]
    retry_count: int
    attempts: tuple[Mapping[str, object], ...]
    last_failure: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": GPU_EXECUTION_SCHEMA_VERSION,
            "status": self.status,
            "job_ids": list(self.job_ids),
            "retry_count": self.retry_count,
            "attempts": [dict(attempt) for attempt in self.attempts],
            "last_failure": dict(self.last_failure),
        }


@dataclass(frozen=True)
class GpuExecutionResult:
    """Successful approved execution and its identifier-free evidence."""

    request: GpuExperimentRequest
    training: TrainingResult
    evidence: Mapping[str, object]
    evidence_path: Path | None
    retry_count: int

    def to_public_dict(self) -> dict[str, object]:
        return dict(self.evidence)


def _training_config_from_dict(value: object) -> TrainingConfig:
    """Rebuild a TrainingConfig from its stable serialized representation."""

    if not isinstance(value, Mapping):
        raise GpuConfigurationError("GPU request training_config must be an object")
    model = value.get("model")
    packing = value.get("packing")
    preprocessing = value.get("preprocessing")
    optimization = value.get("optimization")
    seeds = value.get("seeds")
    progress = value.get("progress")
    if not all(isinstance(item, Mapping) for item in (model, packing, preprocessing, optimization, seeds, progress)):
        raise GpuConfigurationError("GPU request training_config is incomplete")
    model = cast(Mapping[str, object], model)
    packing = cast(Mapping[str, object], packing)
    preprocessing = cast(Mapping[str, object], preprocessing)
    optimization = cast(Mapping[str, object], optimization)
    seeds = cast(Mapping[str, object], seeds)
    progress = cast(Mapping[str, object], progress)
    from .geometry import GeometryPolicy
    from .preprocessing import PreprocessConfig

    try:
        image_size_value = cast(Sequence[Any], model["image_size"])
        channels_value = cast(Sequence[Any], model["channels"])
        preprocessing_image_size = cast(Sequence[Any], preprocessing["image_size"])
        preprocessing_percentiles = cast(Sequence[Any], preprocessing["normalization_percentiles"])
        if len(image_size_value) != 2 or len(channels_value) != 3:
            raise TypeError("model dimensions are not sequences")
        if len(preprocessing_image_size) != 2:
            raise TypeError("preprocessing image size is not a sequence")
        if len(preprocessing_percentiles) != 2:
            raise TypeError("preprocessing percentiles are not a sequence")
        preprocess = PreprocessConfig(
            image_size=(int(cast(Any, preprocessing_image_size[0])), int(cast(Any, preprocessing_image_size[1]))),
            max_slices=int(cast(Any, packing["max_slices"])),
            max_series=int(cast(Any, packing["max_series"])),
            geometry_policy=GeometryPolicy.from_dict(preprocessing["geometry_policy"]),
            normalization_domain=str(preprocessing["normalization_domain"]),
            normalization_percentiles=(
                float(cast(Any, preprocessing_percentiles[0])),
                float(cast(Any, preprocessing_percentiles[1])),
            ),
            use_voi_lut=bool(preprocessing["use_voi_lut"]),
            interpolation=str(preprocessing["interpolation"]),
            padding_fill_value=float(cast(Any, preprocessing["padding_fill_value"])),
            output_dtype=str(preprocessing["output_dtype"]),
            representation_id=(
                str(preprocessing["representation_id"])
                if preprocessing.get("representation_id") is not None
                else None
            ),
        )
        config = TrainingConfig(
            image_size=(int(cast(Any, image_size_value[0])), int(cast(Any, image_size_value[1]))),
            channels=(
                int(cast(Any, channels_value[0])),
                int(cast(Any, channels_value[1])),
                int(cast(Any, channels_value[2])),
            ),
            input_channels=int(cast(Any, model["input_channels"])),
            max_slices=int(cast(Any, packing["max_slices"])),
            max_series=int(cast(Any, packing["max_series"])),
            batch_size=int(cast(Any, optimization["batch_size"])),
            successful_optimizer_steps=int(cast(Any, optimization["successful_optimizer_steps"])),
            optimizer=str(optimization["optimizer"]),
            learning_rate=float(cast(Any, optimization["learning_rate"])),
            encoder_learning_rate=(
                float(cast(Any, optimization["encoder_learning_rate"]))
                if optimization.get("encoder_learning_rate") is not None
                else None
            ),
            schedule=str(optimization["schedule"]),
            seed=int(cast(Any, seeds["initialization"])),
            batch_order_seed=int(cast(Any, seeds["batch_order"])),
            persistent_no_progress_limit=int(cast(Any, progress["persistent_no_progress_limit"])),
            checkpoint_interval_steps=int(cast(Any, progress["checkpoint_interval_steps"])),
            checkpoint_interval_seconds=float(cast(Any, progress["checkpoint_interval_seconds"])),
            loss=str(value["loss"]),
            augmentation=str(value["augmentation"]),
            device=str(value["device"]),
            selection_policy=str(value["selection_policy"]),
            preprocessing_config=preprocess,
            model_id=str(model.get("model_id", COMPACT_MODEL_ID)),
            pooling_id=str(model.get("pooling_id", COMPACT_POOLING_ID)),
            encoder_id=(
                str(model["encoder_id"])
                if model.get("encoder_id") is not None
                else None
            ),
            encoder_weights_sha256=(
                str(model["encoder_weights_sha256"])
                if model.get("encoder_weights_sha256") is not None
                else None
            ),
            head_hidden_size=int(cast(Any, model.get("head_hidden_size", 32))),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GpuConfigurationError("GPU request training_config fields are invalid") from exc
    if config.to_dict() != dict(value):
        raise GpuConfigurationError("GPU request training_config is not canonical")
    return config


def _ticket33_pretrained_resource(execution: object) -> Mapping[str, object]:
    """Validate request-bound provenance for the only pretrained GPU model."""

    if not isinstance(execution, Mapping):
        raise GpuApprovalError("GPU request execution section is invalid")
    resource = execution.get("pretrained_resource")
    if not isinstance(resource, Mapping):
        raise GpuApprovalError("GPU request has no pretrained artifact authorization scope")
    expected = {
        "artifact_id": RESNET18_ARTIFACT_ID,
        "source_revision": RESNET18_SOURCE_REVISION,
        "download_url": RESNET18_DOWNLOAD_URL,
        "sha256": RESNET18_SHA256,
        "expected_bytes": RESNET18_ARTIFACT_BYTES,
        "scope": TICKET33_RESOURCE_SCOPE,
        "encoder_frozen": True,
        "runtime_network": False,
        "external_spend": 0,
    }
    if any(resource.get(name) != value for name, value in expected.items()):
        raise GpuApprovalError("GPU request pretrained artifact identity or scope is invalid")
    if (
        not isinstance(resource.get("expected_bytes"), int)
        or isinstance(resource.get("expected_bytes"), bool)
        or not isinstance(resource.get("external_spend"), int)
        or isinstance(resource.get("external_spend"), bool)
    ):
        raise GpuApprovalError("GPU request pretrained artifact numeric fields are invalid")
    weights_path = Path(_nonempty(resource.get("weights_path"), "pretrained weights path"))
    if not weights_path.is_absolute():
        raise GpuApprovalError("GPU request pretrained weights path must be absolute")
    approval_path = Path(
        _nonempty(resource.get("resource_approval_path"), "pretrained resource approval path")
    )
    if not approval_path.is_absolute():
        raise GpuApprovalError("GPU request pretrained resource approval path must be absolute")
    return cast(Mapping[str, object], resource)


def _ticket39_pretrained_resource(execution: object) -> Mapping[str, object]:
    """Validate the exact ResNet resource scope for layer4 fine-tuning."""

    if not isinstance(execution, Mapping):
        raise GpuApprovalError("GPU request execution section is invalid")
    resource = execution.get("pretrained_resource")
    if not isinstance(resource, Mapping):
        raise GpuApprovalError("GPU request has no pretrained artifact authorization scope")
    expected = {
        "artifact_id": RESNET18_ARTIFACT_ID,
        "source_revision": RESNET18_SOURCE_REVISION,
        "download_url": RESNET18_DOWNLOAD_URL,
        "sha256": RESNET18_SHA256,
        "expected_bytes": RESNET18_ARTIFACT_BYTES,
        "scope": TICKET39_RESOURCE_SCOPE,
        "decision": "approved",
        "rights_status": "eligible",
        "rights_status_scope": TICKET39_RESOURCE_SCOPE,
        "encoder_frozen": False,
        "trainable_encoder_portion": "layer4",
        "runtime_network": False,
        "external_spend": 0,
    }
    if any(resource.get(name) != value for name, value in expected.items()):
        raise GpuApprovalError("Ticket 39 pretrained artifact identity or scope is invalid")
    if (
        not isinstance(resource.get("expected_bytes"), int)
        or isinstance(resource.get("expected_bytes"), bool)
        or not isinstance(resource.get("external_spend"), int)
        or isinstance(resource.get("external_spend"), bool)
    ):
        raise GpuApprovalError("Ticket 39 pretrained artifact numeric fields are invalid")
    for field_name in ("weights_path", "resource_approval_path"):
        value = Path(_nonempty(resource.get(field_name), field_name))
        if not value.is_absolute():
            raise GpuApprovalError(f"Ticket 39 {field_name} must be absolute")
    return cast(Mapping[str, object], resource)


def _load_cpu_preflight(path: Path) -> CpuPreflightResult:
    payload = _read_json(path, "CPU preflight evidence")
    if payload.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise GpuApprovalError("CPU preflight has an unexpected schema")
    checks = payload.get("checks")
    environment = payload.get("environment")
    details = payload.get("details")
    if not isinstance(checks, Mapping) or not isinstance(environment, Mapping) or not isinstance(details, Mapping):
        raise GpuApprovalError("CPU preflight evidence is incomplete")
    if not isinstance(payload.get("passed"), bool):
        raise GpuApprovalError("CPU preflight result does not declare passed")
    if any(not isinstance(value, bool) for value in checks.values()):
        raise GpuApprovalError("CPU preflight checks are not boolean")
    result = CpuPreflightResult(
        passed=bool(payload["passed"]),
        checks={str(key): value for key, value in checks.items()},
        environment=dict(environment),
        details=dict(details),
    )
    try:
        require_cpu_preflight(result, require_slurm=True, require_isolated_python=True)
    except PreflightError as exc:
        raise GpuApprovalError(f"CPU preflight has not passed: {exc}") from exc
    return result


def load_cpu_preflight(path: str | Path) -> CpuPreflightResult:
    """Load the exact passed CPU preflight required by GPU execution."""

    return _load_cpu_preflight(Path(path).resolve())


def load_gpu_experiment_request(path: str | Path) -> GpuExperimentRequest:
    """Load and checksum-verify an immutable GPU request directory or file."""

    candidate = Path(path).resolve()
    request_path = candidate / "request.json" if candidate.is_dir() else candidate
    payload = _read_json(request_path, "GPU experiment request")
    if payload.get("schema_version") != GPU_EXPERIMENT_SCHEMA_VERSION:
        raise GpuApprovalError("GPU experiment request has an unexpected schema")
    if payload.get("status") != "owner_approval_required":
        raise GpuApprovalError("GPU experiment request is not approval-gated")
    checksum = _nonempty(payload.get("request_checksum"), "request_checksum")
    unsigned = dict(payload)
    unsigned.pop("request_checksum", None)
    if _json_sha256(unsigned) != checksum:
        raise GpuApprovalError("GPU experiment request checksum is invalid")
    _safe_identifier(payload.get("experiment_id"), "experiment_id")
    approval_payload = payload.get("approval")
    if not isinstance(approval_payload, Mapping):
        raise GpuApprovalError("GPU experiment request has no approval policy")
    if (
        approval_payload.get("required") is not True
        or approval_payload.get("approved") is not False
        or approval_payload.get("scope") != GPU_APPROVAL_SCOPE
        or approval_payload.get("retry_limit") != GPU_RETRY_LIMIT
        or not isinstance(approval_payload.get("retry_allowed"), bool)
    ):
        raise GpuApprovalError("GPU experiment request approval policy is invalid")
    evidence_payload = payload.get("evidence")
    if not isinstance(evidence_payload, Mapping):
        raise GpuApprovalError("GPU experiment request has no evidence plan")
    if evidence_payload.get("schema_version") != GPU_EXECUTION_SCHEMA_VERSION:
        raise GpuApprovalError("GPU experiment request evidence plan has an unexpected schema")
    resources = GpuResourceRequest.from_dict(payload.get("resources"))
    preflight_reference = payload.get("cpu_preflight")
    if not isinstance(preflight_reference, Mapping):
        raise GpuApprovalError("GPU experiment request has no CPU preflight reference")
    if preflight_reference.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise GpuApprovalError("GPU request references an unexpected CPU preflight schema")
    preflight_path = Path(_nonempty(preflight_reference.get("path"), "CPU preflight path")).resolve()
    preflight_checksum = _nonempty(preflight_reference.get("sha256"), "CPU preflight checksum")
    try:
        preflight_matches = _sha256_file(preflight_path) == preflight_checksum
    except OSError as exc:
        raise GpuApprovalError("referenced CPU preflight file is unavailable") from exc
    if not preflight_matches:
        raise GpuApprovalError("referenced CPU preflight checksum is no longer valid")
    _load_cpu_preflight(preflight_path)
    experiment_payload = payload.get("experiment")
    if not isinstance(experiment_payload, Mapping):
        raise GpuConfigurationError("GPU request experiment section is invalid")
    config = _training_config_from_dict(experiment_payload.get("training_config"))
    if experiment_payload.get("scientific_config_fingerprint") != config.config_fingerprint:
        raise GpuApprovalError("GPU request scientific configuration fingerprint is invalid")
    execution_payload = payload.get("execution", {})
    if config.model_id == RESNET18_MODEL_ID:
        if (
            config.encoder_id != RESNET18_ENCODER_ID
            or config.encoder_weights_sha256 != RESNET18_SHA256
        ):
            raise GpuApprovalError("GPU ResNet request must use the admitted ImageNet encoder")
        _ticket33_pretrained_resource(execution_payload)
    elif config.model_id == RESNET18_FINETUNE_MODEL_ID:
        if (
            config.encoder_id != RESNET18_ENCODER_ID
            or config.encoder_weights_sha256 != RESNET18_SHA256
            or config.encoder_learning_rate is None
        ):
            raise GpuApprovalError("Ticket 39 must use the admitted encoder and layer4 schedule")
        _ticket39_pretrained_resource(execution_payload)
    elif isinstance(execution_payload, Mapping) and execution_payload.get("pretrained_resource") is not None:
        raise GpuApprovalError("GPU request binds pretrained weights for an unsupported model")
    inputs = payload.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise GpuConfigurationError("GPU request inputs section is invalid")
    scientific_identity = experiment_payload.get("scientific_identity")
    expected_scientific_identity = {
        "config_fingerprint": config.config_fingerprint,
        "inputs": dict(inputs),
    }
    if scientific_identity != expected_scientific_identity:
        raise GpuApprovalError("GPU request scientific identity is invalid")
    if config.device != "cuda" and not config.device.startswith("cuda:"):
        raise GpuConfigurationError("GPU request must use a CUDA device")
    if resources.smoke_test and config.successful_optimizer_steps > 2:
        raise GpuConfigurationError("GPU smoke requests must use at most two optimizer steps")
    campaign = payload.get("campaign")
    if not isinstance(campaign, Mapping):
        raise GpuApprovalError("GPU request is missing the shared campaign budget contract")
    ledger_path_value = _nonempty(campaign.get("ledger_path"), "campaign ledger path")
    ledger_path = Path(ledger_path_value)
    if not ledger_path.is_absolute():
        raise GpuApprovalError("GPU campaign ledger path must be absolute")
    budget_seconds = campaign.get("budget_seconds")
    comparison_reserve_seconds = campaign.get("reserved_comparison_seconds")
    comparison_release_seconds = campaign.get("comparison_reserve_release_seconds")
    comparison_after_request_seconds = campaign.get("comparison_reserve_after_request_seconds")
    max_attempts = campaign.get("max_attempts")
    request_reservation_seconds = campaign.get("request_reservation_seconds")
    if (
        budget_seconds != int(GPU_CAMPAIGN_BUDGET_HOURS * 3600)
        or not isinstance(comparison_reserve_seconds, int)
        or isinstance(comparison_reserve_seconds, bool)
        or not isinstance(comparison_release_seconds, int)
        or isinstance(comparison_release_seconds, bool)
        or comparison_release_seconds < 0
        or comparison_release_seconds > comparison_reserve_seconds
        or not isinstance(comparison_after_request_seconds, int)
        or isinstance(comparison_after_request_seconds, bool)
        or comparison_after_request_seconds
        != comparison_reserve_seconds - comparison_release_seconds
        or not isinstance(max_attempts, int)
        or isinstance(max_attempts, bool)
        or not isinstance(request_reservation_seconds, int)
        or isinstance(request_reservation_seconds, bool)
    ):
        raise GpuApprovalError("GPU campaign budget fields are invalid")
    expected_attempts = 2 if approval_payload["retry_allowed"] else 1
    if max_attempts != expected_attempts:
        raise GpuApprovalError("GPU campaign attempt count differs from the approval retry scope")
    if request_reservation_seconds != max_attempts * resources.wall_time_minutes * 60:
        raise GpuApprovalError("GPU campaign reservation does not match the approved wall time")
    remaining_before = campaign.get("remaining_hours_before_request")
    remaining_after = campaign.get("remaining_hours_after_request_and_comparison_reserve")
    if (
        not isinstance(remaining_before, (int, float))
        or isinstance(remaining_before, bool)
        or not isinstance(remaining_after, (int, float))
        or isinstance(remaining_after, bool)
        or remaining_after < 0
    ):
        raise GpuApprovalError("GPU request has invalid remaining campaign hours")
    expected_remaining_after = (
        float(remaining_before)
        - request_reservation_seconds / 3600.0
        - (comparison_reserve_seconds - comparison_release_seconds) / 3600.0
    )
    if abs(float(remaining_after) - expected_remaining_after) > 1e-9:
        raise GpuApprovalError("GPU request remaining hours do not match its reserve release")
    return GpuExperimentRequest(request_path.parent, request_path, checksum, payload)


def load_gpu_approval(path: str | Path, request: GpuExperimentRequest) -> GpuApproval:
    """Load owner approval and require that it names the exact request."""

    approval_path = Path(path).resolve()
    payload = _read_json(approval_path, "GPU owner approval")
    if payload.get("schema_version") != GPU_APPROVAL_SCHEMA_VERSION:
        raise GpuApprovalError("GPU owner approval has an unexpected schema")
    if payload.get("request_checksum") != request.request_checksum:
        raise GpuApprovalError("GPU owner approval does not name this request")
    if payload.get("approval_scope") != GPU_APPROVAL_SCOPE:
        raise GpuApprovalError("GPU owner approval scope is not ticket 13")
    if payload.get("approved") is not True:
        raise GpuApprovalError("GPU owner approval is not granted")
    requested_approval = request.payload.get("approval")
    requested_retry = (
        requested_approval.get("retry_allowed")
        if isinstance(requested_approval, Mapping)
        else None
    )
    if not isinstance(payload.get("retry_allowed"), bool) or payload.get("retry_allowed") != requested_retry:
        raise GpuApprovalError("GPU owner approval retry scope does not match the request")
    _nonempty(payload.get("approver"), "GPU approval approver")
    timestamp = _nonempty(payload.get("approved_at_utc"), "GPU approval timestamp")
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GpuApprovalError("GPU approval timestamp is not ISO-8601") from exc
    if request.training_config.model_id in {RESNET18_MODEL_ID, RESNET18_FINETUNE_MODEL_ID}:
        resource = (
            _ticket39_pretrained_resource(request.payload.get("execution"))
            if request.training_config.model_id == RESNET18_FINETUNE_MODEL_ID
            else _ticket33_pretrained_resource(request.payload.get("execution"))
        )
        expected_resource_values = {
            "artifact_id": resource["artifact_id"],
            "source_revision": resource["source_revision"],
            "download_url": resource["download_url"],
            "sha256": resource["sha256"],
            "expected_bytes": resource["expected_bytes"],
            "external_spend": resource["external_spend"],
            "scope": resource["scope"],
            "decision": "approved",
            "rights_status": "eligible",
            "rights_status_scope": resource["scope"],
        }
        resource_approval_path = Path(
            _nonempty(
                resource.get("resource_approval_path"),
                "pretrained resource approval path",
            )
        )
        if resource_approval_path.is_symlink() or not resource_approval_path.is_file():
            raise GpuApprovalError(
                "pretrained artifact authorization requires a regular non-symlink resource approval file"
            )
        resource_approval_path = resource_approval_path.resolve()
        try:
            resource_approval = load_resource_approval(resource_approval_path)
        except PretrainedResourceError as exc:
            raise GpuApprovalError(
                f"pretrained artifact authorization is invalid: {exc}"
            ) from exc
        if any(
            resource_approval.get(name) != value
            for name, value in expected_resource_values.items()
        ):
            raise GpuApprovalError(
                "pretrained artifact authorization does not match the requested artifact or resource scope"
            )
        if not isinstance(resource_approval.get("expected_bytes"), int) or isinstance(
            resource_approval.get("expected_bytes"), bool
        ):
            raise GpuApprovalError("pretrained artifact authorization byte count is invalid")
        if not isinstance(resource_approval.get("external_spend"), int) or isinstance(
            resource_approval.get("external_spend"), bool
        ):
            raise GpuApprovalError("pretrained artifact authorization spend is invalid")
    return GpuApproval(approval_path, _sha256_file(approval_path), payload)


def _required_slurm_int(environment: Mapping[str, str], name: str, *, default: int | None = None) -> int:
    value = environment.get(name)
    if value is None:
        if default is not None:
            return default
        raise GpuResourceError(f"Slurm allocation does not report {name}")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise GpuResourceError(f"Slurm allocation field {name} is not an integer") from exc
    if parsed <= 0:
        raise GpuResourceError(f"Slurm allocation field {name} must be positive")
    return parsed


def _visible_gpu_count(value: str | None) -> int:
    if value is None or not value.strip():
        raise GpuResourceError("Slurm allocation does not expose CUDA_VISIBLE_DEVICES")
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if not devices or any(item == "-1" for item in devices):
        raise GpuResourceError("CUDA_VISIBLE_DEVICES does not expose an allocated GPU")
    return len(devices)


def _slurm_memory_mb(value: str) -> int:
    match = re.fullmatch(r"(\d+)([KMGTP]?)", value.strip().upper())
    if match is None:
        raise GpuResourceError("Slurm memory allocation is not parseable")
    amount = int(match.group(1))
    multiplier = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024, "P": 1024 * 1024 * 1024}[match.group(2)]
    return max(1, int(amount * multiplier))


def _device_file_covers_index(device_file: str, device_index: int) -> bool:
    exact = re.fullmatch(r"/dev/nvidia(\d+)", device_file)
    if exact is not None:
        return int(exact.group(1)) == device_index
    range_match = re.fullmatch(r"/dev/nvidia\[(\d+)-(\d+)\]", device_file)
    if range_match is None:
        return False
    first = int(range_match.group(1))
    last = int(range_match.group(2))
    return first <= device_index <= last


def _slurm_node_name_matches(pattern: str, node_name: str) -> bool:
    if pattern == node_name:
        return True
    match = re.fullmatch(r"([^\[]*)\[([^]]+)\]([^\[]*)", pattern)
    if match is None:
        return False
    prefix, members, suffix = match.groups()
    if not node_name.startswith(prefix) or not node_name.endswith(suffix):
        return False
    end = len(node_name) - len(suffix) if suffix else len(node_name)
    middle = node_name[len(prefix) : end]
    for member in members.split(","):
        range_match = re.fullmatch(r"(\d+)-(\d+)", member)
        if range_match is not None:
            if not middle.isdecimal():
                continue
            first = int(range_match.group(1))
            last = int(range_match.group(2))
            candidate = int(middle)
            if first <= candidate <= last:
                return True
        elif middle == member:
            return True
    return False


def _whole_gpu_device_proof(
    resource: GpuResourceRequest,
    values: Mapping[str, str],
    *,
    allocated_gres: str,
) -> tuple[bool, dict[str, object]]:
    """Prove whole-device ownership from scheduler and active GRES evidence."""

    job_gres_detail = (
        values.get("SLURM_JOB_GRES_DETAIL")
        or values.get("SLURM_GRES_DETAIL")
        or ""
    ).strip()
    detail_match = re.fullmatch(r"gpu:1\(IDX:(\d+)\)", job_gres_detail)
    allocated_device_index = int(detail_match.group(1)) if detail_match is not None else None
    node_name = (values.get("SLURM_JOB_NODELIST") or values.get("SLURMD_NODENAME") or "").strip()
    node_gres = (values.get("SLURM_NODE_GRES") or "").strip()
    config_entry = (values.get("SLURM_GRES_CONFIG_ENTRY") or "").strip()
    device_file = (values.get("SLURM_GRES_DEVICE_FILE") or "").strip()
    device_isolation = (
        values.get("SLURM_CONSTRAIN_DEVICES")
        or values.get("SLURM_GRES_DEVICE_ISOLATION")
        or ""
    ).strip().lower()
    gres_types = (values.get("SLURM_GRES_TYPES") or "").strip()
    select_type = (values.get("SLURM_SELECT_TYPE") or "").strip().lower()
    select_type_parameters = (values.get("SLURM_SELECT_TYPE_PARAMETERS") or "").strip()
    config_scope = (values.get("SLURM_GRES_CONFIG_SCOPE") or "").strip().lower()
    sharing_mode = (values.get("SLURM_GRES_SHARING_MODE") or "").strip().lower()
    forbidden_terms = ("mps", "mig", "shard", "fraction", "slice")
    mode_text = " ".join(
        (job_gres_detail, node_gres, config_entry, gres_types, sharing_mode)
    ).lower()
    forbidden_modes = tuple(term for term in forbidden_terms if term in mode_text)

    node_gpu_match = re.fullmatch(r"gpu:(\d+)", node_gres)
    config_node_match = re.search(r"(?:^|\s)NodeName=([^\s]+)", config_entry)
    config_name_match = re.search(r"(?:^|\s)Name=([^\s]+)", config_entry)
    config_file_match = re.search(r"(?:^|\s)File=([^\s]+)", config_entry)
    configured_file = config_file_match.group(1) if config_file_match is not None else ""
    node_mapping_matches_job_node = (
        config_node_match is not None
        and bool(node_name)
        and _slurm_node_name_matches(config_node_match.group(1), node_name)
    ) or (
        config_scope == "local_node"
        and config_node_match is None
        and bool(node_name)
    )
    checks: dict[str, bool] = {
        "exact_generic_request": resource.gres == "gpu:1",
        "exact_generic_allocation": allocated_gres == "gpu:1",
        "all_reported_gres_match": all(
            value.split("(", 1)[0] == "gpu:1"
            for value in (
                values.get("SLURM_JOB_GRES"),
                values.get("SLURM_TRES_PER_NODE"),
            )
            if value is not None and value.strip()
        ),
        "concrete_allocated_gres_detail": detail_match is not None,
        "node_generic_gpu_gres": node_gpu_match is not None,
        "node_mapping_matches_job_node": node_mapping_matches_job_node,
        "generic_gres_device_file": config_name_match is not None and config_name_match.group(1) == "gpu",
        "device_file_observed": bool(device_file) and configured_file == device_file,
        "allocated_index_matches_device_file": (
            allocated_device_index is not None
            and bool(device_file)
            and _device_file_covers_index(device_file, allocated_device_index)
        ),
        "device_isolation_enabled": device_isolation in {"1", "true", "yes"},
        "generic_gpu_type_configured": "gpu" in {item.strip() for item in gres_types.split(",") if item.strip()},
        "consumable_select_type": select_type in {"select/cons_res", "select/cons_tres"},
        "select_type_parameters_observed": bool(select_type_parameters),
        "non_fractional_gpu_mode": bool(sharing_mode)
        and sharing_mode in {"none", "disabled", "no", "false"}
        and not forbidden_modes,
        "allocated_index_within_node_gpu_count": (
            allocated_device_index is not None
            and node_gpu_match is not None
            and allocated_device_index < int(node_gpu_match.group(1))
        ),
    }
    missing_or_failed = tuple(name for name, passed in checks.items() if not passed)
    proof: dict[str, object] = {
        "checks": checks,
        "allocated_gres_detail": job_gres_detail,
        "allocated_device_index": allocated_device_index,
        "node_name": node_name,
        "node_gres": node_gres,
        "device_file": device_file,
        "gres_config_entry": config_entry,
        "gres_types": gres_types,
        "select_type": select_type,
        "select_type_parameters": select_type_parameters,
        "config_scope": config_scope,
        "device_isolation": device_isolation,
        "sharing_mode": sharing_mode,
        "forbidden_modes": forbidden_modes,
        "failure_reasons": missing_or_failed,
    }
    return not missing_or_failed, proof


def verify_slurm_gpu_allocation(
    resource: GpuResourceRequest,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Verify that the current job is the exact approved Slurm allocation.

    The check records the scheduler's sharing signal when one is available but
    never treats a nominal A40 memory value as the memory assigned to the job.
    ``cluster1-gpu-add`` is admitted only for requests explicitly marked as
    smoke tests.
    """

    values = dict(os.environ if environment is None else environment)
    try:
        job_id = _nonempty(values.get("SLURM_JOB_ID"), "SLURM_JOB_ID")
        partition = _nonempty(values.get("SLURM_JOB_PARTITION"), "SLURM_JOB_PARTITION")
    except GpuConfigurationError as exc:
        raise GpuResourceError(str(exc)) from exc
    if partition != resource.partition:
        raise GpuResourceError(
            f"allocated partition {partition!r} does not match approved partition {resource.partition!r}"
        )
    if partition == GPU_SMOKE_PARTITION and not resource.smoke_test:
        raise GpuResourceError("cluster1-gpu-add is reserved for an explicitly approved GPU smoke test")
    nodes = _required_slurm_int(values, "SLURM_JOB_NUM_NODES")
    tasks = _required_slurm_int(values, "SLURM_NTASKS")
    if nodes != resource.nodes or tasks != resource.ntasks:
        raise GpuResourceError("allocated node/task count does not match the approved request")
    allocated_cpus = _required_slurm_int(values, "SLURM_CPUS_PER_TASK")
    if allocated_cpus != resource.cpus_per_task:
        raise GpuResourceError("allocated CPU count does not match the approved request")
    try:
        reported_memory = _nonempty(values.get("SLURM_MEM_PER_NODE"), "SLURM_MEM_PER_NODE")
        reported_time = _nonempty(values.get("SLURM_TIMELIMIT"), "SLURM_TIMELIMIT")
    except GpuConfigurationError as exc:
        raise GpuResourceError(str(exc)) from exc
    allocated_memory_mb = _slurm_memory_mb(reported_memory)
    if allocated_memory_mb != resource.memory_gb * 1024:
        raise GpuResourceError("allocated host memory does not match the approved request")
    allocated_wall_time = _slurm_time_minutes(reported_time)
    if allocated_wall_time != resource.wall_time_minutes:
        raise GpuResourceError("allocated wall time does not match the approved request")
    allocated_gres = values.get("SLURM_JOB_GRES") or values.get("SLURM_TRES_PER_NODE")
    if allocated_gres is None:
        raise GpuResourceError("Slurm allocation does not expose its GPU GRES")
    allocated_gres = allocated_gres.split("(", 1)[0]
    if allocated_gres != resource.gres:
        raise GpuResourceError(
            f"allocated GPU GRES {allocated_gres!r} does not match approved {resource.gres!r}"
        )
    gpus_on_node = values.get("SLURM_GPUS_ON_NODE") or values.get("SLURM_GPUS")
    if gpus_on_node is None:
        raise GpuResourceError("Slurm allocation does not expose its GPU count")
    try:
        gpu_count = int(gpus_on_node)
    except ValueError as exc:
        raise GpuResourceError("Slurm GPU count is not an integer") from exc
    if gpu_count <= 0:
        raise GpuResourceError("Slurm GPU count must be positive")
    if gpu_count != resource.gpu_count:
        raise GpuResourceError("allocated GPU count does not match the approved single-GPU request")
    visible_count = _visible_gpu_count(values.get("CUDA_VISIBLE_DEVICES"))
    if visible_count != resource.gpu_count:
        raise GpuResourceError("CUDA_VISIBLE_DEVICES exposes a different GPU count than approved")
    sharing_behavior = values.get("SLURM_JOB_SHARED") or values.get("SLURM_SHARED")
    if sharing_behavior is None or not sharing_behavior.strip():
        raise GpuResourceError("Slurm allocation does not report sharing behavior")
    if sharing_behavior.strip().lower() in {"not_reported", "unknown", "unavailable"}:
        raise GpuResourceError("Slurm sharing behavior is not verified")
    whole_gpu_device_verified, whole_gpu_device_proof = _whole_gpu_device_proof(
        resource,
        values,
        allocated_gres=allocated_gres,
    )
    return {
        "job_id": job_id,
        "partition": partition,
        "nodes": nodes,
        "ntasks": tasks,
        "allocated_gres": allocated_gres,
        "allocated_gpu_count": gpu_count,
        "visible_gpu_count": visible_count,
        "allocated_cpus_per_task": allocated_cpus,
        "allocated_memory_mb": allocated_memory_mb,
        "allocated_wall_time_minutes": allocated_wall_time,
        "sharing_behavior": sharing_behavior,
        "sharing_verified": True,
        "whole_gpu_device_verified": whole_gpu_device_verified,
        "whole_gpu_device_proof": whole_gpu_device_proof,
        "exclusive_requested": resource.exclusive,
        "nominal_memory_assumed": False,
    }


def _cuda_device_selector(device_index: int) -> str:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None or not visible_devices.strip():
        return str(device_index)
    selectors = [item.strip() for item in visible_devices.split(",") if item.strip()]
    if device_index >= len(selectors) or selectors[device_index] == "-1":
        raise GpuResourceError("CUDA_VISIBLE_DEVICES does not identify the current GPU")
    return selectors[device_index]


def _memory_totals_consistent(observed_bytes: int, expected_bytes: int) -> bool:
    return abs(observed_bytes - expected_bytes) <= max(
        256 * 1024 * 1024,
        expected_bytes // 20,
    )


def _nvidia_smi_facts(device_selector: str) -> dict[str, object]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--id",
                device_selector,
                "--query-gpu=index,name,driver_version,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuResourceError("could not query nvidia-smi for the GPU driver") from exc
    if completed.returncode != 0 or not completed.stdout.strip():
        raise GpuResourceError(
            "nvidia-smi did not return driver and memory facts: "
            f"{completed.stderr.strip() or 'unknown error'}"
        )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise GpuResourceError("nvidia-smi returned more than one GPU fact row")
    fields = [item.strip() for item in rows[0].split(",")]
    if len(fields) != 6 or not fields[0] or not fields[1] or not fields[2]:
        raise GpuResourceError("nvidia-smi returned an unparseable GPU fact row")
    facts: dict[str, object] = {
        "nvidia_smi_device_index": fields[0],
        "nvidia_smi_device_name": fields[1],
        "driver_version": fields[2],
    }
    for key, value in (
        ("nvidia_smi_memory_total_mb", fields[3]),
        ("nvidia_smi_memory_free_mb", fields[4]),
        ("nvidia_smi_memory_used_mb", fields[5]),
    ):
        try:
            facts[key] = int(float(value))
        except ValueError as exc:
            raise GpuResourceError(f"nvidia-smi returned a non-numeric {key}") from exc
    return facts


def _default_gpu_probe() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise GpuResourceError("CUDA is not available in the current Python environment")
    try:
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        device_selector = _cuda_device_selector(device_index)
        facts = _nvidia_smi_facts(device_selector)
        nvidia_total_bytes = int(cast(int, facts["nvidia_smi_memory_total_mb"])) * 1024 * 1024
        if not _memory_totals_consistent(nvidia_total_bytes, int(properties.total_memory)):
            raise GpuResourceError(
                "nvidia-smi memory facts do not match the CUDA-visible device"
            )
        return {
            "device_index": device_index,
            "device_selector": device_selector,
            "visible_device_count": torch.cuda.device_count(),
            "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "driver_version": facts["driver_version"],
            "cuda_runtime_version": torch.version.cuda,
            "device_total_memory_bytes": int(properties.total_memory),
            "cuda_total_memory_bytes": int(total_bytes),
            "cuda_free_memory_bytes": int(free_bytes),
            "available_memory_bytes": int(free_bytes),
            "runtime_visible_memory_limit_bytes": int(total_bytes),
            "available_memory_source": "torch.cuda.mem_get_info.free_bytes",
            "allocated_vram_bytes": int(torch.cuda.memory_allocated(device_index)),
            "reserved_vram_bytes": int(torch.cuda.memory_reserved(device_index)),
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device_index)),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device_index)),
            **facts,
        }
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise GpuResourceError("could not collect CUDA device and memory facts") from exc


def validate_gpu_environment(
    snapshot: Mapping[str, object],
    *,
    resource: GpuResourceRequest,
    allocation: Mapping[str, object],
) -> dict[str, object]:
    """Validate and normalize the hardware facts captured for an allocation."""

    required_text = (
        "device_name",
        "driver_version",
        "cuda_runtime_version",
        "available_memory_source",
    )
    for field_name in required_text:
        try:
            _nonempty(snapshot.get(field_name), f"GPU {field_name}")
        except GpuConfigurationError as exc:
            raise GpuResourceError(str(exc)) from exc
    integer_fields = (
        "visible_device_count",
        "device_total_memory_bytes",
        "available_memory_bytes",
        "runtime_visible_memory_limit_bytes",
        "allocated_vram_bytes",
        "reserved_vram_bytes",
        "peak_vram_bytes",
        "peak_reserved_vram_bytes",
    )
    normalized = dict(snapshot)
    for field_name in integer_fields:
        value = snapshot.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise GpuResourceError(f"GPU {field_name} must be a non-negative integer")
        normalized[field_name] = value
    visible_device_count = cast(int, normalized["visible_device_count"])
    available_memory_bytes = cast(int, normalized["available_memory_bytes"])
    runtime_visible_memory_limit_bytes = cast(
        int, normalized["runtime_visible_memory_limit_bytes"]
    )
    device_total_memory_bytes = cast(int, normalized["device_total_memory_bytes"])
    peak_vram_bytes = cast(int, normalized["peak_vram_bytes"])
    if visible_device_count != resource.gpu_count:
        raise GpuResourceError("CUDA exposes a different GPU count than the approved request")
    if available_memory_bytes <= 0:
        raise GpuResourceError("GPU allocation reports no available memory")
    if runtime_visible_memory_limit_bytes <= 0:
        raise GpuResourceError("GPU runtime reports no visible memory limit")
    if device_total_memory_bytes <= 0:
        raise GpuResourceError("GPU device reports no total memory")
    if not _memory_totals_consistent(
        runtime_visible_memory_limit_bytes,
        device_total_memory_bytes,
    ):
        raise GpuResourceError("CUDA memory totals do not match the device properties")
    nvidia_total_mb = snapshot.get("nvidia_smi_memory_total_mb")
    if nvidia_total_mb is not None:
        if not isinstance(nvidia_total_mb, int) or isinstance(nvidia_total_mb, bool) or nvidia_total_mb <= 0:
            raise GpuResourceError("nvidia-smi memory total must be a positive integer")
        nvidia_total_bytes = nvidia_total_mb * 1024 * 1024
        if not _memory_totals_consistent(nvidia_total_bytes, device_total_memory_bytes):
            raise GpuResourceError("nvidia-smi memory facts do not match the CUDA-visible device")
    if available_memory_bytes > runtime_visible_memory_limit_bytes:
        raise GpuResourceError("GPU free memory exceeds the visible memory limit")
    if peak_vram_bytes > runtime_visible_memory_limit_bytes:
        raise GpuResourceError("observed peak VRAM exceeds the runtime-visible memory limit")
    if allocation.get("partition") != resource.partition:
        raise GpuResourceError("GPU hardware facts are bound to a different Slurm partition")
    available_memory_source = str(normalized["available_memory_source"])
    if available_memory_source.lower() in {"nominal", "device_specification", "user_report"}:
        raise GpuResourceError("GPU available memory must come from an observed allocation")
    whole_device_verified = allocation.get("whole_gpu_device_verified") is True
    memory_assignment_verified = False
    assigned_memory_value = snapshot.get("assigned_memory_bytes")
    if assigned_memory_value is not None:
        if (
            not isinstance(assigned_memory_value, int)
            or isinstance(assigned_memory_value, bool)
            or assigned_memory_value <= 0
        ):
            raise GpuResourceError("GPU assigned memory must be a positive integer when reported")
        assigned_memory_source_value = snapshot.get("assigned_memory_source")
        try:
            assigned_memory_source = _nonempty(
                assigned_memory_source_value,
                "GPU assigned_memory_source",
            )
        except GpuConfigurationError as exc:
            raise GpuResourceError(str(exc)) from exc
        if assigned_memory_source.lower() in {
            "nominal",
            "device_specification",
            "user_report",
        }:
            raise GpuResourceError("GPU assigned memory must come from an observed allocation")
    if whole_device_verified:
        assigned_memory_bytes = runtime_visible_memory_limit_bytes
        assigned_memory_source = "cuda_total_on_verified_whole_gpu_gres"
        memory_assignment_verified = True
    elif assigned_memory_value is not None:
        assigned_memory_bytes = assigned_memory_value
        memory_assignment_verified = True
    else:
        assigned_memory_bytes = None
        assigned_memory_source = "not_verified"
    if assigned_memory_bytes is not None:
        if assigned_memory_bytes > runtime_visible_memory_limit_bytes:
            raise GpuResourceError("GPU assigned memory exceeds the runtime-visible memory limit")
        if available_memory_bytes > assigned_memory_bytes:
            raise GpuResourceError("GPU free memory exceeds the assigned memory")
        if peak_vram_bytes > assigned_memory_bytes:
            raise GpuResourceError("observed peak VRAM exceeds the assigned GPU memory")
    normalized["resource_request"] = resource.to_dict()
    normalized["allocation"] = dict(allocation)
    normalized["whole_gpu_device_verified"] = whole_device_verified
    normalized["nominal_device_memory_used"] = False
    normalized["assigned_memory_bytes"] = assigned_memory_bytes
    normalized["assigned_memory_source"] = assigned_memory_source
    normalized["memory_assignment_verified"] = memory_assignment_verified
    normalized["peak_vram_within_assigned_memory"] = (
        memory_assignment_verified
        and assigned_memory_bytes is not None
        and peak_vram_bytes <= assigned_memory_bytes
    )
    return normalized


def collect_gpu_environment(
    *,
    resource: GpuResourceRequest | None = None,
    slurm_environment: Mapping[str, str] | None = None,
    probe: Callable[[], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Capture actual CUDA facts, optionally binding them to a Slurm request."""

    allocation, snapshot = _collect_gpu_observation(
        resource=resource,
        slurm_environment=slurm_environment,
        probe=probe,
    )
    if resource is not None and allocation is not None:
        return validate_gpu_environment(snapshot, resource=resource, allocation=allocation)
    return snapshot


def _collect_gpu_observation(
    *,
    resource: GpuResourceRequest | None,
    slurm_environment: Mapping[str, str] | None,
    probe: Callable[[], Mapping[str, object]] | None,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    allocation = (
        verify_slurm_gpu_allocation(resource, slurm_environment)
        if resource is not None
        else None
    )
    snapshot = dict(probe() if probe is not None else _default_gpu_probe())
    return allocation, snapshot


def classify_gpu_failure(error: BaseException | str) -> RetryClassification:
    """Conservatively classify a scheduler or CUDA failure for one retry."""

    message = str(error).lower()
    non_retryable_tokens = (
        "cancelled by user",
        "canceled by user",
        "cancelled",
        "canceled",
        "scancel",
        "out of memory",
        "oom-kill",
        "oom_kill",
        "cuda out of memory",
        "nan",
        "non-finite",
        "incompatible",
        "deterministic loader",
        "code bug",
    )
    if any(token in message for token in non_retryable_tokens):
        return RetryClassification(False, "deterministic_or_resource", str(error))
    return classify_retry_failure(error)


def _slurm_time(minutes: int) -> str:
    hours, remainder = divmod(minutes, 60)
    return f"{hours:02d}:{remainder:02d}:00"


def _slurm_time_minutes(value: str) -> int:
    raw = value.strip()
    if raw.isdigit():
        minutes = int(raw)
    else:
        days = 0
        clock = raw
        if "-" in raw:
            day_text, clock = raw.split("-", 1)
            if not day_text.isdigit():
                raise GpuResourceError("Slurm time limit is not parseable")
            days = int(day_text)
        parts = clock.split(":")
        if len(parts) == 2:
            hour_text, minute_text = parts
            second_text = "0"
        elif len(parts) == 3:
            hour_text, minute_text, second_text = parts
        else:
            raise GpuResourceError("Slurm time limit is not parseable")
        if not all(part.isdigit() for part in (hour_text, minute_text, second_text)):
            raise GpuResourceError("Slurm time limit is not parseable")
        if int(second_text) != 0:
            raise GpuResourceError("Slurm time limit has sub-minute precision")
        minutes = days * 24 * 60 + int(hour_text) * 60 + int(minute_text)
    if minutes <= 0:
        raise GpuResourceError("Slurm time limit must be positive")
    return minutes


def _job_id(stdout: str) -> str | None:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    candidate = lines[-1].split(";", 1)[0].strip()
    if candidate.isdigit():
        return candidate
    match = re.fullmatch(r"Submitted batch job (\d+)", candidate)
    return match.group(1) if match is not None else None


def _query_active_gpu_jobs() -> tuple[Mapping[str, str], ...]:
    """Read this account's active GPU jobs; fail closed when Slurm cannot answer."""

    user = os.environ.get("USER", "").strip() or getpass.getuser()
    command = [
        "squeue",
        "--noheader",
        "--user",
        user,
        "--partition",
        ",".join(GPU_ALLOWED_PARTITIONS),
        "--states",
        "PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED",
        "--format=%i|%P|%j|%T",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuApprovalError("could not verify that this account has no active GPU request") from exc
    if result.returncode != 0:
        raise GpuApprovalError("Slurm could not confirm that this account has no active GPU request")
    jobs: list[Mapping[str, str]] = []
    for line in str(result.stdout or "").splitlines():
        fields = line.strip().split("|", 3)
        if len(fields) != 4:
            raise GpuApprovalError("Slurm returned an unreadable active GPU job row")
        jobs.append(
            {
                "job_id": fields[0],
                "partition": fields[1],
                "job_name": fields[2],
                "state": fields[3],
            }
        )
    return tuple(jobs)


def _require_idle_gpu_account(
    scheduler_query: Callable[[], Sequence[Mapping[str, object]]],
) -> None:
    active_states = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
    jobs = scheduler_query()
    for job in jobs:
        partition = job.get("partition")
        state = job.get("state")
        if partition in GPU_ALLOWED_PARTITIONS and state in active_states:
            raise GpuCampaignActiveRequestError(
                "this account already has an active or pending GPU request "
                f"(job {job.get('job_id', 'unknown')}, state={state})"
            )


def _recorded_gpu_failure(
    environment: Mapping[str, str],
    *,
    request_checksum: str,
) -> tuple[int, RetryClassification | None]:
    """Read durable in-job failure metadata before deciding a scheduler retry."""

    output_value = environment.get("RSNA_TICKET13_OUTPUT")
    if output_value is None or not output_value.strip():
        return 0, None
    failure_path = Path(output_value) / "gpu-failure.json"
    if not failure_path.exists():
        return 0, None
    try:
        payload = _read_json(failure_path, "GPU failure evidence")
    except GpuExperimentError as exc:
        return GPU_RETRY_LIMIT, RetryClassification(False, "recorded_failure", str(exc))
    if payload.get("schema_version") != GPU_EXECUTION_SCHEMA_VERSION:
        return GPU_RETRY_LIMIT, RetryClassification(
            False,
            "recorded_failure",
            "durable GPU failure evidence has an unexpected schema",
        )
    if payload.get("request_checksum") != request_checksum:
        return GPU_RETRY_LIMIT, RetryClassification(
            False,
            "recorded_failure",
            "durable GPU failure evidence names a different request",
        )
    recorded_retry = payload.get("retry_count")
    if not isinstance(recorded_retry, int) or isinstance(recorded_retry, bool) or recorded_retry < 0:
        return GPU_RETRY_LIMIT, RetryClassification(
            False,
            "recorded_failure",
            "durable GPU failure evidence has an invalid retry count",
        )
    retry_payload = payload.get("retry")
    if not isinstance(retry_payload, Mapping) or not isinstance(retry_payload.get("retryable"), bool):
        return min(recorded_retry, GPU_RETRY_LIMIT), RetryClassification(
            False,
            "recorded_failure",
            "durable GPU failure evidence has no valid retry classification",
        )
    retry_number_allowed = retry_payload.get("retry_number_allowed", GPU_RETRY_LIMIT)
    if (
        not isinstance(retry_number_allowed, int)
        or isinstance(retry_number_allowed, bool)
        or retry_number_allowed < 0
    ):
        return min(recorded_retry, GPU_RETRY_LIMIT), RetryClassification(
            False,
            "recorded_failure",
            "durable GPU failure evidence has an invalid retry limit",
        )
    return min(recorded_retry, GPU_RETRY_LIMIT), RetryClassification(
        bool(retry_payload["retryable"]),
        str(retry_payload.get("category", "recorded_failure")),
        str(retry_payload.get("reason", "recorded GPU failure")),
        retry_number_allowed,
    )


def _gpu_submission_record_path(
    request: GpuExperimentRequest,
    approval: GpuApproval,
) -> Path:
    """Return a checksum-keyed sidecar that consumes one approval artifact."""

    return request.root.parent / (
        ".ticket13-submission-"
        f"{request.request_checksum[:16]}-{approval.approval_checksum[:16]}.json"
    )


def _load_gpu_submission_record(
    path: Path,
    *,
    request_checksum: str,
    approval_checksum: str,
) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    payload = _read_json(path, "GPU submission record")
    if payload.get("schema_version") != GPU_SUBMISSION_SCHEMA_VERSION:
        raise GpuApprovalError("GPU submission record has an unexpected schema")
    if payload.get("request_checksum") != request_checksum:
        raise GpuApprovalError("GPU submission record names a different request")
    if payload.get("approval_sha256") != approval_checksum:
        raise GpuApprovalError("GPU submission record names a different approval artifact")
    status = payload.get("status")
    if status not in {"submitting", "failed", "submitted", "completed"}:
        raise GpuApprovalError("GPU submission record has an invalid status")
    retry_count = payload.get("retry_count")
    if not isinstance(retry_count, int) or isinstance(retry_count, bool) or retry_count < 0:
        raise GpuApprovalError("GPU submission record has an invalid retry count")
    return payload


def _write_gpu_submission_record(
    path: Path,
    *,
    request: GpuExperimentRequest,
    approval: GpuApproval,
    status: str,
    retry_count: int,
    attempts: Sequence[Mapping[str, object]],
    job_ids: Sequence[str] = (),
    last_failure: Mapping[str, object] | None = None,
) -> None:
    atomic_json_write(
        path,
        {
            "schema_version": GPU_SUBMISSION_SCHEMA_VERSION,
            "status": status,
            "request_checksum": request.request_checksum,
            "approval_sha256": approval.approval_checksum,
            "retry_count": retry_count,
            "job_ids": list(job_ids),
            "attempts": [dict(attempt) for attempt in attempts],
            "last_failure": dict(last_failure or {}),
        },
        overwrite=path.exists(),
    )


def _write_gpu_preflight_record(
    destination: Path,
    *,
    request: GpuExperimentRequest,
    approval: GpuApproval,
    preflight_path: Path,
    status: str,
    allocation: Mapping[str, object] | None = None,
    hardware: Mapping[str, object] | None = None,
    error: BaseException | None = None,
) -> Path:
    """Persist resource facts before the final GPU memory gate."""

    payload: dict[str, object] = {
        "schema_version": GPU_PREFLIGHT_SCHEMA_VERSION,
        "status": status,
        "request_checksum": request.request_checksum,
        "approval_sha256": approval.approval_checksum,
        "cpu_preflight_sha256": _sha256_file(preflight_path),
        "slurm_allocation": dict(allocation or {}),
        "hardware": dict(hardware or {}),
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)
    path = destination / "gpu-preflight.json"
    atomic_json_write(path, payload, overwrite=True)
    return path


def _gpu_execution_record_path(
    request: GpuExperimentRequest,
    approval: GpuApproval,
) -> Path:
    return request.root.parent / (
        ".ticket13-execution-"
        f"{request.request_checksum[:16]}-{approval.approval_checksum[:16]}.json"
    )


@contextmanager
def _exclusive_gpu_record_lock(path: Path) -> Iterator[None]:
    """Serialize approval consumption across processes sharing a filesystem."""

    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _lock_gpu_record(
    record_path: Callable[[GpuExperimentRequest, GpuApproval], Path],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def locked(
            request_path: str | Path,
            approval_path: str | Path,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            request = load_gpu_experiment_request(request_path)
            approval = load_gpu_approval(approval_path, request)
            with _exclusive_gpu_record_lock(record_path(request, approval)):
                return function(request_path, approval_path, *args, **kwargs)

        return locked

    return decorate


def _load_gpu_execution_record(
    path: Path,
    *,
    request_checksum: str,
    approval_checksum: str,
) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    payload = _read_json(path, "GPU execution record")
    if payload.get("schema_version") != GPU_EXECUTION_RECORD_SCHEMA_VERSION:
        raise GpuApprovalError("GPU execution record has an unexpected schema")
    if payload.get("request_checksum") != request_checksum:
        raise GpuApprovalError("GPU execution record names a different request")
    if payload.get("approval_sha256") != approval_checksum:
        raise GpuApprovalError("GPU execution record names a different approval artifact")
    if payload.get("status") not in {"running", "failed", "completed"}:
        raise GpuApprovalError("GPU execution record has an invalid status")
    retry_count = payload.get("retry_count")
    if not isinstance(retry_count, int) or isinstance(retry_count, bool) or retry_count < 0:
        raise GpuApprovalError("GPU execution record has an invalid retry count")
    _nonempty(payload.get("output_dir"), "GPU execution record output directory")
    return payload


def _write_gpu_execution_record(
    path: Path,
    *,
    request: GpuExperimentRequest,
    approval: GpuApproval,
    status: str,
    output_dir: Path,
    retry_count: int,
    attempts: Sequence[Mapping[str, object]],
    retry: Mapping[str, object] | None = None,
) -> None:
    atomic_json_write(
        path,
        {
            "schema_version": GPU_EXECUTION_RECORD_SCHEMA_VERSION,
            "status": status,
            "request_checksum": request.request_checksum,
            "approval_sha256": approval.approval_checksum,
            "output_dir": str(output_dir),
            "retry_count": retry_count,
            "attempts": [dict(attempt) for attempt in attempts],
            "retry": dict(retry or {}),
        },
        overwrite=path.exists(),
    )


@_lock_gpu_record(_gpu_submission_record_path)
def submit_gpu_experiment(
    request_path: str | Path,
    approval_path: str | Path,
    *,
    script_path: str | Path = "slurm/run_ticket13_gpu_experiment.sbatch",
    environment: Mapping[str, str] | None = None,
    sbatch_binary: str = "sbatch",
    wait: bool = True,
    runner: Callable[..., Any] | None = None,
    scheduler_query: Callable[[], Sequence[Mapping[str, object]]] | None = None,
) -> GpuSubmissionResult:
    """Submit the exact approved resource request through ``sbatch``.

    Validation happens before the scheduler command is built.  With
    ``wait=True`` a single unchanged submission is permitted after a clearly
    transient failure, provided the owner approval covers that retry.  The
    function checks the shared campaign ledger and this account's active Slurm
    jobs before submission. It never invokes ``srun`` and never changes
    resources or scientific configuration on retry.
    """

    request = load_gpu_experiment_request(request_path)
    approval = load_gpu_approval(approval_path, request)
    approval_file = Path(approval_path).resolve()
    script = Path(script_path).resolve()
    if not script.is_file():
        raise GpuExecutionError(f"GPU sbatch adapter is not a regular file: {script}")
    if not isinstance(sbatch_binary, str) or not sbatch_binary.strip():
        raise GpuConfigurationError("sbatch_binary must be non-empty")
    resource = request.resources
    campaign = cast(Mapping[str, object], request.payload["campaign"])
    ledger_path = Path(_nonempty(campaign["ledger_path"], "campaign ledger path"))
    if scheduler_query is None:
        scheduler_query = _query_active_gpu_jobs if runner is None else (lambda: ())
    _require_idle_gpu_account(scheduler_query)
    submission_environment = dict(os.environ)
    for key, value in dict(environment or {}).items():
        if not key.startswith("RSNA_TICKET13_"):
            raise GpuConfigurationError("GPU submission environment keys must use RSNA_TICKET13_ prefix")
        if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
            raise GpuConfigurationError(f"GPU submission environment value is invalid for {key}")
        submission_environment[key] = value
    preflight_reference = cast(Mapping[str, object], request.payload["cpu_preflight"])
    submission_environment.update(
        {
            "RSNA_TICKET13_REQUEST": str(request.request_path),
            "RSNA_TICKET13_APPROVAL": str(approval_file),
            "RSNA_TICKET13_PREFLIGHT": str(
                Path(_nonempty(preflight_reference["path"], "CPU preflight path")).resolve()
            ),
            "RSNA_TICKET13_PREVIOUS_RETRIES": "0",
        }
    )
    submission_record_path = _gpu_submission_record_path(request, approval)
    submission_record = _load_gpu_submission_record(
        submission_record_path,
        request_checksum=request.request_checksum,
        approval_checksum=approval.approval_checksum,
    )
    recorded_retry_count, recorded_classification = _recorded_gpu_failure(
        submission_environment,
        request_checksum=request.request_checksum,
    )
    retry_count = 0
    attempts: list[Mapping[str, object]] = []
    job_ids: list[str] = []
    last_failure: dict[str, object] = {}
    stale_failure_metadata = False
    if submission_record is not None:
        raw_attempts = submission_record.get("attempts")
        if not isinstance(raw_attempts, list) or any(
            not isinstance(attempt, Mapping) for attempt in raw_attempts
        ):
            raise GpuApprovalError("GPU submission record has invalid attempts")
        attempts = [dict(attempt) for attempt in raw_attempts]
        raw_job_ids = submission_record.get("job_ids", [])
        if not isinstance(raw_job_ids, list) or any(
            not isinstance(job_id, str) or not job_id.strip() for job_id in raw_job_ids
        ):
            raise GpuApprovalError("GPU submission record has invalid job IDs")
        job_ids = list(raw_job_ids)
        record_status = str(submission_record["status"])
        record_retry_count = cast(int, submission_record["retry_count"])
        if record_status == "submitting":
            raise GpuApprovalError(
                "a prior submission has an unknown outcome; do not reuse the approval"
            )
        if record_status in {"submitted", "completed"}:
            if recorded_classification is None:
                raise GpuApprovalError(
                    "this approval has already been consumed by a successful submission"
                )
            classification = recorded_classification
        else:
            recorded_failure = submission_record.get("last_failure")
            if not isinstance(recorded_failure, Mapping):
                raise GpuApprovalError("GPU submission record has no failure classification")
            if not isinstance(recorded_failure.get("retryable"), bool):
                raise GpuApprovalError("GPU submission record has an invalid failure classification")
            classification = RetryClassification(
                bool(recorded_failure["retryable"]),
                str(recorded_failure.get("category", "recorded_failure")),
                str(recorded_failure.get("reason", "recorded GPU submission failure")),
                int(recorded_failure.get("retry_number_allowed", GPU_RETRY_LIMIT)),
            )
        if (
            approval.payload.get("retry_allowed") is not True
            or not classification.retryable
            or classification.retry_number_allowed < GPU_RETRY_LIMIT
            or max(record_retry_count, recorded_retry_count) >= GPU_RETRY_LIMIT
        ):
            raise GpuApprovalError(
                "this approval cannot authorize another GPU submission attempt"
            )
        retry_count = max(record_retry_count, recorded_retry_count) + 1
        last_failure = classification.to_dict()
    elif recorded_classification is not None:
        if (
            recorded_classification.category == "recorded_failure"
            and "different request" in recorded_classification.reason
        ):
            # A stale output directory must not authorize a retry, but it also
            # must not prevent one fresh submission for this request.
            stale_failure_metadata = True
            retry_count = recorded_retry_count
        else:
            if (
                approval.payload.get("retry_allowed") is not True
                or not recorded_classification.retryable
                or recorded_classification.retry_number_allowed < GPU_RETRY_LIMIT
                or recorded_retry_count >= GPU_RETRY_LIMIT
            ):
                raise GpuApprovalError(
                    "existing GPU failure evidence requires a new approved request"
                )
            retry_count = recorded_retry_count + 1
            last_failure = recorded_classification.to_dict()
    reserve_gpu_campaign_request(
        ledger_path,
        request_checksum=request.request_checksum,
        experiment_id=str(request.payload["experiment_id"]),
        wall_time_minutes=resource.wall_time_minutes,
        retry_allowed=bool(approval.payload.get("retry_allowed")),
        remaining_attempts=cast(int, campaign["max_attempts"]) - retry_count,
        comparison_reserve_release_seconds=cast(
            int, campaign["comparison_reserve_release_seconds"]
        ),
    )
    submission_environment.update(
        {
            "RSNA_TICKET13_CAMPAIGN_LEDGER": str(ledger_path),
            "RSNA_TICKET13_REQUEST_CHECKSUM": request.request_checksum,
            "RSNA_TICKET13_SUBMISSION_WAIT": "1" if wait else "0",
        }
    )
    submission_environment["RSNA_TICKET13_PREVIOUS_RETRIES"] = str(retry_count)
    _write_gpu_submission_record(
        submission_record_path,
        request=request,
        approval=approval,
        status="submitting",
        retry_count=retry_count,
        attempts=attempts,
        last_failure=last_failure,
    )
    command = [
        sbatch_binary,
        "--wait" if wait else "--parsable",
        f"--job-name=rsna-ticket13-{request.payload['experiment_id']}",
        f"--partition={resource.partition}",
        f"--gres={resource.gres}",
        f"--nodes={resource.nodes}",
        f"--ntasks={resource.ntasks}",
        f"--cpus-per-task={resource.cpus_per_task}",
        f"--mem={resource.memory_gb}G",
        f"--time={_slurm_time(resource.wall_time_minutes)}",
    ]
    if resource.exclusive:
        command.append("--exclusive")
    command.append(str(script))
    execute = runner or subprocess.run
    while True:
        attempt_number = len(attempts) + 1
        submitted_epoch = int(time.time())
        begin_gpu_submission_attempt(
            ledger_path,
            request_checksum=request.request_checksum,
            attempt_number=attempt_number,
            submitted_epoch=submitted_epoch,
        )
        submission_environment["RSNA_TICKET13_SUBMITTED_EPOCH"] = str(submitted_epoch)
        submission_environment["RSNA_TICKET13_ATTEMPT_NUMBER"] = str(attempt_number)
        try:
            completed = execute(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=submission_environment,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            completed = subprocess.CompletedProcess(command, 1, stdout="", stderr=str(exc))
        stdout = str(getattr(completed, "stdout", "") or "")
        stderr = str(getattr(completed, "stderr", "") or "")
        returncode = int(getattr(completed, "returncode", 1))
        submitted_job_id = _job_id(stdout)
        if returncode == 0 and submitted_job_id is None:
            returncode = 1
            stderr = "sbatch exited successfully without a parseable job ID"
        if submitted_job_id is not None:
            job_ids.append(submitted_job_id)
        record_gpu_submission_result(
            ledger_path,
            request_checksum=request.request_checksum,
            attempt_number=attempt_number,
            job_id=submitted_job_id,
            exit_code=returncode,
            submission_wait=wait,
        )
        attempts.append(
            {
                "attempt": len(attempts) + 1,
                "returncode": returncode,
                "job_id": submitted_job_id,
                "stdout_present": bool(stdout.strip()),
                "stderr": stderr[-2000:] if stderr else "",
            }
        )
        if returncode == 0:
            _write_gpu_submission_record(
                submission_record_path,
                request=request,
                approval=approval,
                status="completed" if wait else "submitted",
                retry_count=retry_count,
                attempts=attempts,
                job_ids=job_ids,
                last_failure=last_failure,
            )
            if wait:
                finish_gpu_campaign_request(
                    ledger_path,
                    request_checksum=request.request_checksum,
                    status="completed",
                )
            return GpuSubmissionResult(
                status="completed" if wait else "submitted",
                job_ids=tuple(job_ids),
                retry_count=retry_count,
                attempts=tuple(attempts),
                last_failure=last_failure,
            )
        recorded_retry_count, recorded_classification = _recorded_gpu_failure(
            submission_environment,
            request_checksum=request.request_checksum,
        )
        retry_count = max(retry_count, recorded_retry_count)
        classification = recorded_classification or classify_gpu_failure(f"{stderr}\n{stdout}".strip())
        if stale_failure_metadata:
            classification = RetryClassification(
                False,
                "stale_request",
                "retry suppressed because the existing failure names a different request",
            )
        last_failure = classification.to_dict()
        _write_gpu_submission_record(
            submission_record_path,
            request=request,
            approval=approval,
            status="failed",
            retry_count=retry_count,
            attempts=attempts,
            job_ids=job_ids,
            last_failure=last_failure,
        )
        if (
            not wait
            or approval.payload.get("retry_allowed") is not True
            or not classification.retryable
            or retry_count >= GPU_RETRY_LIMIT
        ):
            finish_gpu_campaign_request(
                ledger_path,
                request_checksum=request.request_checksum,
                status="failed",
            )
            return GpuSubmissionResult(
                status="failed",
                job_ids=tuple(job_ids),
                retry_count=retry_count,
                attempts=tuple(attempts),
                last_failure=last_failure,
            )
        retry_count += 1
        submission_environment["RSNA_TICKET13_PREVIOUS_RETRIES"] = str(retry_count)
        _write_gpu_submission_record(
            submission_record_path,
            request=request,
            approval=approval,
            status="submitting",
            retry_count=retry_count,
            attempts=attempts,
            job_ids=job_ids,
            last_failure=last_failure,
        )


def _write_gpu_failure(
    destination: Path,
    *,
    request: GpuExperimentRequest,
    approval: GpuApproval,
    preflight_path: Path,
    retry_count: int,
    attempts: Sequence[Mapping[str, object]],
    error: BaseException,
    resource_preflight: Mapping[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema_version": GPU_EXECUTION_SCHEMA_VERSION,
        "status": "failed",
        "request_checksum": request.request_checksum,
        "approval_sha256": approval.approval_checksum,
        "cpu_preflight_sha256": _sha256_file(preflight_path),
        "retry_count": retry_count,
        "attempts": [dict(attempt) for attempt in attempts],
        "error_type": type(error).__name__,
        "error": str(error),
        "retry": classify_gpu_failure(error).to_dict(),
        "resource_preflight_file": "gpu-preflight.json",
    }
    if resource_preflight is not None:
        payload["resource_preflight"] = dict(resource_preflight)
    atomic_json_write(destination / "gpu-failure.json", payload, overwrite=True)


@_lock_gpu_record(_gpu_execution_record_path)
def run_approved_gpu_experiment(
    request_path: str | Path,
    approval_path: str | Path,
    *,
    preflight_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    evidence_path: str | Path | None = None,
    studies: Any = None,
    fold_manifest: Any = None,
    fold: int | None = None,
    source_version: str | None = None,
    source_fingerprint: str | None = None,
    fold_manifest_checksum: str | None = None,
    integration_manifest_checksum: str | None = None,
    series_cache: Any = None,
    slurm_environment: Mapping[str, str] | None = None,
    gpu_probe: Callable[[], Mapping[str, object]] | None = None,
    training_runner: Callable[[Path | None], TrainingResult] | None = None,
    synthetic_test_hooks: bool | None = None,
    previous_retries: int = 0,
) -> GpuExecutionResult:
    """Run the exact approved GPU experiment through the shared trainer.

    ``training_runner`` is an injectable process boundary for synthetic tests;
    production callers leave it unset and the function calls
    :func:`rsna_knee.training.train_one_fold`.  A transient failure can cause
    one unchanged retry, preferably from the durable latest checkpoint.  OOM,
    cancellation, numerical, semantic, and loader failures stop immediately.
    """

    request = load_gpu_experiment_request(request_path)
    approval = load_gpu_approval(approval_path, request)
    preflight_reference = cast(Mapping[str, object], request.payload["cpu_preflight"])
    referenced_preflight = Path(
        _nonempty(preflight_reference["path"], "CPU preflight path")
    ).resolve()
    if preflight_path is not None and Path(preflight_path).resolve() != referenced_preflight:
        raise GpuApprovalError("supplied CPU preflight path differs from the approved request")
    preflight = load_cpu_preflight(referenced_preflight)
    resource = request.resources
    config = request.training_config
    request_inputs = request.payload.get("inputs", {})
    if not isinstance(request_inputs, Mapping):
        raise GpuConfigurationError("GPU request inputs section is invalid")
    experiment_payload = request.payload.get("experiment")
    if not isinstance(experiment_payload, Mapping):
        raise GpuConfigurationError("GPU request experiment section is invalid")
    approved_scientific_identity = experiment_payload.get("scientific_identity")
    expected_scientific_identity = {
        "config_fingerprint": config.config_fingerprint,
        "inputs": dict(request_inputs),
    }
    if approved_scientific_identity != expected_scientific_identity:
        raise GpuApprovalError("approved GPU scientific identity does not match the request")
    for field_name, supplied_value in (
        ("fold", fold),
        ("source_version", source_version),
        ("source_fingerprint", source_fingerprint),
        ("fold_manifest_checksum", fold_manifest_checksum),
        ("integration_manifest_checksum", integration_manifest_checksum),
    ):
        expected_value = request_inputs.get(field_name)
        if expected_value is not None and supplied_value != expected_value:
            raise GpuConfigurationError(f"supplied GPU input {field_name} differs from the approved request")
    if previous_retries < 0 or previous_retries > GPU_RETRY_LIMIT:
        raise GpuConfigurationError("previous_retries must be between zero and the retry limit")
    synthetic_hooks_used = (
        training_runner is not None if synthetic_test_hooks is None else synthetic_test_hooks
    )
    destination = Path(output_dir or request.root.parent / f"{request.root.name}-execution").resolve()
    if destination == request.root or request.root in destination.parents:
        raise GpuExecutionError("GPU execution artifacts must not be written inside the immutable request")
    destination.mkdir(parents=True, exist_ok=True)
    evidence_destination = Path(evidence_path or destination / "gpu-execution.json").resolve()
    if evidence_destination == request.root or request.root in evidence_destination.parents:
        raise GpuExecutionError("GPU evidence must not be written inside the immutable request")
    execution_record_path = _gpu_execution_record_path(request, approval)
    execution_record = _load_gpu_execution_record(
        execution_record_path,
        request_checksum=request.request_checksum,
        approval_checksum=approval.approval_checksum,
    )
    if execution_record is not None:
        recorded_output = Path(
            _nonempty(execution_record["output_dir"], "GPU execution record output directory")
        ).resolve()
        if recorded_output != destination:
            raise GpuExecutionError(
                "the approved GPU execution is bound to a different output directory"
            )
        execution_status = str(execution_record["status"])
        if execution_status in {"running", "completed"}:
            raise GpuExecutionError(
                "this approval has already been consumed by a GPU execution"
            )
        execution_retry = cast(int, execution_record["retry_count"])
        execution_retry_payload = execution_record.get("retry")
        if (
            not isinstance(execution_retry_payload, Mapping)
            or not isinstance(execution_retry_payload.get("retryable"), bool)
            or approval.payload.get("retry_allowed") is not True
            or not bool(execution_retry_payload["retryable"])
            or execution_retry >= GPU_RETRY_LIMIT
        ):
            raise GpuExecutionError(
                "existing GPU execution evidence requires diagnosis and a new approved request"
            )
            previous_retries = max(previous_retries, execution_retry + 1)
    if execution_record is None and any(destination.iterdir()):
        raise GpuExecutionError(
            "refusing to reuse a non-empty GPU execution directory without an execution record"
        )
    training_destination = destination / "training"
    training_destination.mkdir(parents=True, exist_ok=True)
    recorded_retry_count, recorded_classification = _recorded_gpu_failure(
        {"RSNA_TICKET13_OUTPUT": str(destination)},
        request_checksum=request.request_checksum,
    )
    if recorded_classification is not None:
        if not recorded_classification.retryable or recorded_retry_count >= GPU_RETRY_LIMIT:
            raise GpuExecutionError(
                "existing GPU failure evidence requires diagnosis and a new approved request"
            )
        previous_retries = max(previous_retries, recorded_retry_count + 1)
    runner = training_runner
    if runner is None:
        if studies is None or fold_manifest is None or fold is None:
            raise GpuConfigurationError(
                "production GPU execution requires studies, fold manifest, and fold"
            )
        if not source_version or not source_fingerprint or not integration_manifest_checksum:
            raise GpuConfigurationError(
                "production GPU execution requires source and integration manifest identities"
            )
        required_inputs = (
            "fold",
            "source_version",
            "source_fingerprint",
            "fold_manifest_checksum",
            "integration_manifest_checksum",
        )
        if any(request_inputs.get(name) is None for name in required_inputs):
            raise GpuConfigurationError(
                "approved GPU request does not bind every training input identity"
            )
        if request_inputs.get("fold_manifest_checksum") != fold_manifest.manifest_checksum:
            raise GpuConfigurationError("fold manifest checksum differs from the approved request")
        approved_fold = cast(int, fold)
        approved_source_version = cast(str, source_version)
        approved_source_fingerprint = cast(str, source_fingerprint)
        approved_integration_manifest_checksum = cast(
            str,
            integration_manifest_checksum,
        )
        pretrained_weights_path: Path | None = None
        resource_approval_path: Path | None = None
        if config.model_id in {RESNET18_MODEL_ID, RESNET18_FINETUNE_MODEL_ID}:
            pretrained_resource = (
                _ticket39_pretrained_resource(request.payload.get("execution"))
                if config.model_id == RESNET18_FINETUNE_MODEL_ID
                else _ticket33_pretrained_resource(request.payload.get("execution"))
            )
            pretrained_weights_path = Path(
                _nonempty(pretrained_resource.get("weights_path"), "pretrained weights path")
            ).resolve()
            resource_approval_path = Path(
                _nonempty(
                    pretrained_resource.get("resource_approval_path"),
                    "pretrained resource approval path",
                )
            ).resolve()
    allocation_environment = dict(os.environ if slurm_environment is None else slurm_environment)
    attempts: list[Mapping[str, object]] = []
    retry_count = previous_retries
    allocation: dict[str, object] = {}
    raw_snapshot: dict[str, object] = {}
    before: dict[str, object] = {}
    try:
        collected_allocation, raw_snapshot = _collect_gpu_observation(
            resource=resource,
            slurm_environment=allocation_environment,
            probe=gpu_probe,
        )
        allocation = dict(collected_allocation or {})
        _write_gpu_preflight_record(
            destination,
            request=request,
            approval=approval,
            preflight_path=referenced_preflight,
            status="observed",
            allocation=allocation,
            hardware=raw_snapshot,
        )
        before = validate_gpu_environment(
            raw_snapshot,
            resource=resource,
            allocation=allocation,
        )
        if not resource.smoke_test and allocation.get("whole_gpu_device_verified") is not True:
            raise GpuResourceError(
                "full GPU training requires a verified whole-device GPU GRES allocation"
            )
        if before.get("memory_assignment_verified") is not True:
            raise GpuResourceError(
                "GPU memory assigned to this job could not be verified; stop before training"
            )
        _write_gpu_preflight_record(
            destination,
            request=request,
            approval=approval,
            preflight_path=referenced_preflight,
            status="passed",
            allocation=allocation,
            hardware=before,
        )
    except Exception as exc:
        failure_hardware: Mapping[str, object] = before or raw_snapshot
        _write_gpu_preflight_record(
            destination,
            request=request,
            approval=approval,
            preflight_path=referenced_preflight,
            status="failed",
            allocation=allocation,
            hardware=failure_hardware,
            error=exc,
        )
        classification = classify_gpu_failure(exc)
        attempts.append(
            {
                "attempt": 1,
                "error_type": type(exc).__name__,
                "retry": {
                    "retryable": classification.retryable,
                    "category": classification.category,
                    "reason": "redacted; see private GPU failure evidence",
                    "retry_number_allowed": classification.retry_number_allowed,
                },
            }
        )
        _write_gpu_execution_record(
            execution_record_path,
            request=request,
            approval=approval,
            status="failed",
            output_dir=destination,
            retry_count=retry_count,
            attempts=attempts,
            retry=classification.to_dict(),
        )
        _write_gpu_failure(
            destination,
            request=request,
            approval=approval,
            preflight_path=referenced_preflight,
            retry_count=retry_count,
            attempts=attempts,
            error=exc,
            resource_preflight={
                "status": "failed",
                "slurm_allocation": dict(allocation),
                "hardware": dict(failure_hardware),
            },
        )
        if isinstance(exc, GpuResourceError):
            raise
        raise GpuExecutionError(f"approved GPU preflight failed: {exc}") from exc
    authorization = _new_gpu_execution_authorization(
        request_checksum=request.request_checksum,
        approval_checksum=approval.approval_checksum,
        preflight_checksum=_sha256_file(referenced_preflight),
        approval_scope=GPU_APPROVAL_SCOPE,
        slurm_job_id=str(allocation["job_id"]),
        allocation_fingerprint=_json_sha256(allocation),
    )
    _write_gpu_execution_record(
        execution_record_path,
        request=request,
        approval=approval,
        status="running",
        output_dir=destination,
        retry_count=retry_count,
        attempts=attempts,
    )
    if runner is None and callable(studies):
        studies = studies()
        if studies is None:
            raise GpuConfigurationError("the approved GPU study loader returned no studies")

    if runner is None:
        def production_training_runner(resume_from: Path | None) -> TrainingResult:
            return _train_one_fold_gpu(
                studies,
                fold_manifest,
                approved_fold,
                training_destination,
                config=config,
                source_version=approved_source_version,
                source_fingerprint=approved_source_fingerprint,
                integration_manifest_checksum=approved_integration_manifest_checksum,
                series_cache=series_cache,
                preflight=preflight,
                preflight_checksum=_sha256_file(referenced_preflight),
                authorization=authorization,
                preflight_path=referenced_preflight,
                request_path=request.root,
                approval_path=approval.path,
                allocation_environment=allocation_environment,
                resource=resource.to_dict(),
                approved_scientific_identity=cast(
                    Mapping[str, object],
                    approved_scientific_identity,
                ),
                device=config.device,
                resume_from=resume_from,
                requested_resources=resource.to_dict(),
                pretrained_weights_path=pretrained_weights_path,
                resource_approval_path=resource_approval_path,
            )
        runner = production_training_runner
    while True:
        try:
            cuda_timing = config.device.startswith("cuda") and torch.cuda.is_available()
            if cuda_timing:
                torch.cuda.synchronize()
            training_started = time.perf_counter()
            training = runner(
                training_destination / "checkpoints" / "latest.pt"
                if (training_destination / "checkpoints" / "latest.pt").is_file()
                and retry_count > 0
                else None
            )
            if cuda_timing:
                torch.cuda.synchronize()
            training_elapsed_seconds = max(time.perf_counter() - training_started, 1e-9)
            if training.config.to_dict() != config.to_dict():
                raise GpuConfigurationError("GPU training changed the approved scientific configuration")
            final_checkpoint = training.final_checkpoint_path
            if final_checkpoint is None or not final_checkpoint.is_file():
                raise GpuExecutionError("approved GPU training did not produce a final checkpoint")
            try:
                load_training_checkpoint(
                    final_checkpoint,
                    model=training.model,
                    expected_config=config,
                    expected_fold=fold,
                    expected_checkpoint_kind="final_selected",
                )
            except Exception as exc:
                raise GpuExecutionError("final GPU checkpoint could not be reloaded") from exc
            after = collect_gpu_environment(
                resource=resource,
                slurm_environment=allocation_environment,
                probe=gpu_probe,
            )
            if after.get("memory_assignment_verified") is not True:
                raise GpuResourceError(
                    "GPU memory assignment was not verified after training"
                )
            if after.get("assigned_memory_bytes") != before.get("assigned_memory_bytes"):
                raise GpuResourceError(
                    "GPU assigned memory changed during the approved execution"
                )
            before_peak = int(cast(Any, before["peak_vram_bytes"]))
            after_peak = int(cast(Any, after["peak_vram_bytes"]))
            final_sha256 = checkpoint_sha256(final_checkpoint)
            latest_checkpoint = training.latest_checkpoint_path
            evidence: dict[str, object] = {
                "schema_version": GPU_EXECUTION_SCHEMA_VERSION,
                "status": "completed",
                "experiment_id": request.payload["experiment_id"],
                "request_checksum": request.request_checksum,
                "approval_sha256": approval.approval_checksum,
                "cpu_preflight_sha256": _sha256_file(referenced_preflight),
                "scientific_config_fingerprint": config.config_fingerprint,
                "scientific_identity": dict(
                    cast(Mapping[str, object], approved_scientific_identity)
                ),
                "resources": resource.to_dict(),
                "slurm_allocation": dict(allocation),
                "hardware": {
                    "device_name": after["device_name"],
                    "driver_version": after["driver_version"],
                    "cuda_runtime_version": after["cuda_runtime_version"],
                    "driver_compatibility_verified": gpu_probe is None,
                    "available_memory_source": after.get(
                        "available_memory_source", "injected_test_snapshot"
                    ),
                    "before": dict(before),
                    "after": dict(after),
                    "peak_vram_bytes": max(before_peak, after_peak),
                    "available_memory_bytes": before["available_memory_bytes"],
                    "runtime_visible_memory_limit_bytes": before[
                        "runtime_visible_memory_limit_bytes"
                    ],
                    "assigned_memory_bytes": before["assigned_memory_bytes"],
                    "assigned_memory_source": before["assigned_memory_source"],
                    "memory_assignment_verified": after["memory_assignment_verified"],
                    "peak_vram_within_assigned_memory": after[
                        "peak_vram_within_assigned_memory"
                    ],
                    "nominal_device_memory_used": False,
                },
                "training": training.to_public_dict(),
                "training_performance": {
                    "end_to_end_training_elapsed_seconds": training_elapsed_seconds,
                    "optimizer_step_compute_seconds": training.optimizer_step_compute_seconds,
                    "measured_optimizer_steps": training.measured_optimizer_steps,
                    "forward_backward_steps_per_second": (
                        training.measured_optimizer_steps / training.optimizer_step_compute_seconds
                        if training.optimizer_step_compute_seconds is not None
                        and training.optimizer_step_compute_seconds > 0
                        else None
                    ),
                    "training_study_examples_per_second": (
                        training.measured_optimizer_steps
                        * config.batch_size
                        / training.optimizer_step_compute_seconds
                        if training.optimizer_step_compute_seconds is not None
                        and training.optimizer_step_compute_seconds > 0
                        else None
                    ),
                    "measurement_scope": (
                        "CUDA event time around successful forward, backward, optimizer, and "
                        "scheduler work; excludes per-batch host input construction, checkpoint "
                        "I/O, feature materialization, and evaluation"
                    ),
                },
                "checkpoint": {
                    "final_present": True,
                    "final_sha256": final_sha256,
                    "latest_present": latest_checkpoint is not None
                    and latest_checkpoint.is_file(),
                    "final_reload_passed": True,
                    "resumed_from_checkpoint": training.resumed_from_checkpoint is not None,
                    "replayed_batches": training.replayed_batches,
                },
                "retry": {
                    "retry_count": retry_count,
                    "retry_limit": GPU_RETRY_LIMIT,
                    "unchanged_resource_and_configuration": True,
                    "attempts": [dict(attempt) for attempt in attempts],
                },
                "scope": {
                    "smoke_test": resource.smoke_test,
                    "gpu_compatibility_verified_for_this_allocation": (
                        gpu_probe is None and not synthetic_hooks_used
                    ),
                    "synthetic_test_hooks_used": gpu_probe is not None or synthetic_hooks_used,
                },
            }
            atomic_json_write(evidence_destination, evidence)
            _write_gpu_execution_record(
                execution_record_path,
                request=request,
                approval=approval,
                status="completed",
                output_dir=destination,
                retry_count=retry_count,
                attempts=attempts,
            )
            return GpuExecutionResult(
                request=request,
                training=training,
                evidence=evidence,
                evidence_path=evidence_destination,
                retry_count=retry_count,
            )
        except Exception as exc:
            classification = classify_gpu_failure(exc)
            attempts.append(
                {
                    "attempt": len(attempts) + 1,
                    "error_type": type(exc).__name__,
                    "retry": {
                        "retryable": classification.retryable,
                        "category": classification.category,
                        "reason": "redacted; see private GPU failure evidence",
                        "retry_number_allowed": classification.retry_number_allowed,
                    },
                }
            )
            _write_gpu_execution_record(
                execution_record_path,
                request=request,
                approval=approval,
                status="failed",
                output_dir=destination,
                retry_count=retry_count,
                attempts=attempts,
                retry=classification.to_dict(),
            )
            if (
                approval.payload.get("retry_allowed") is True
                and classification.retryable
                and retry_count < GPU_RETRY_LIMIT
            ):
                retry_count += 1
                _write_gpu_execution_record(
                    execution_record_path,
                    request=request,
                    approval=approval,
                    status="running",
                    output_dir=destination,
                    retry_count=retry_count,
                    attempts=attempts,
                    retry=classification.to_dict(),
                )
                continue
            _write_gpu_failure(
                destination,
                request=request,
                approval=approval,
                preflight_path=referenced_preflight,
                retry_count=retry_count,
                attempts=attempts,
                error=exc,
            )
            if isinstance(exc, GpuExecutionError):
                raise
            raise GpuExecutionError(f"approved GPU execution failed: {exc}") from exc


def run_gpu_smoke_test(
    examples: Sequence[Any],
    request_path: str | Path,
    approval_path: str | Path,
    *,
    preflight_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    evidence_path: str | Path | None = None,
    slurm_environment: Mapping[str, str] | None = None,
    gpu_probe: Callable[[], Mapping[str, object]] | None = None,
    previous_retries: int = 0,
) -> GpuExecutionResult:
    """Run the approved model shape through forward, backward, and checkpointing."""

    request = load_gpu_experiment_request(request_path)
    if not request.resources.smoke_test:
        raise GpuExecutionError("GPU smoke execution requires a smoke-test-scoped request")
    if not examples:
        raise GpuExecutionError("GPU smoke execution requires at least one example")
    config = request.training_config
    if config.model_id in {RESNET18_MODEL_ID, RESNET18_FINETUNE_MODEL_ID}:
        raise GpuExecutionError(
            "pretrained ResNet smoke runs must use the approved real-fold GPU adapter"
        )
    approval = load_gpu_approval(approval_path, request)
    preflight_reference = cast(Mapping[str, object], request.payload["cpu_preflight"])
    referenced_preflight = Path(
        _nonempty(preflight_reference["path"], "CPU preflight path")
    ).resolve()
    preflight = load_cpu_preflight(referenced_preflight)
    smoke_environment = dict(os.environ if slurm_environment is None else slurm_environment)
    smoke_allocation = verify_slurm_gpu_allocation(request.resources, smoke_environment)
    authorization = _new_gpu_execution_authorization(
        request_checksum=request.request_checksum,
        approval_checksum=approval.approval_checksum,
        preflight_checksum=_sha256_file(referenced_preflight),
        approval_scope=GPU_APPROVAL_SCOPE,
        slurm_job_id=str(smoke_allocation["job_id"]),
        allocation_fingerprint=_json_sha256(smoke_allocation),
    )
    destination = Path(
        output_dir or request.root.parent / f"{request.root.name}-execution"
    ).resolve()
    training_destination = destination / "training"

    def runner(resume_from: Path | None) -> TrainingResult:
        return _train_fixed_budget_gpu(
            examples,
            config,
            artifact_dir=training_destination,
            resume_from=resume_from,
            requested_resources=request.resources.to_dict(),
            authorization=authorization,
            preflight=preflight,
            preflight_checksum=_sha256_file(referenced_preflight),
            preflight_path=referenced_preflight,
            request_path=request.root,
            approval_path=approval.path,
            allocation_environment=smoke_environment,
            resource=request.resources.to_dict(),
        )

    return run_approved_gpu_experiment(
        request.root,
        approval_path,
        preflight_path=preflight_path,
        output_dir=destination,
        evidence_path=evidence_path,
        slurm_environment=slurm_environment,
        gpu_probe=gpu_probe,
        training_runner=runner,
        synthetic_test_hooks=False,
        previous_retries=previous_retries,
    )


def prepare_gpu_experiment(
    output_root: str | Path,
    *,
    preflight_path: str | Path,
    resource: GpuResourceRequest = GpuResourceRequest(),
    campaign_ledger_path: str | Path | None = None,
    reserved_comparison_hours: float = 0.0,
    comparison_reserve_release_minutes: int = 0,
    training_config: TrainingConfig | None = None,
    experiment_id: str | None = None,
    experiment_identity: Mapping[str, object] | None = None,
    fold: int | None = None,
    source_version: str | None = None,
    source_fingerprint: str | None = None,
    fold_manifest_checksum: str | None = None,
    integration_manifest_checksum: str | None = None,
    retry_allowed: bool = True,
    execution: Mapping[str, object] | None = None,
) -> GpuExperimentRequest:
    """Create an immutable, owner-approval-required GPU execution request."""

    preflight_file = Path(preflight_path).resolve()
    preflight = _load_cpu_preflight(preflight_file)
    del preflight
    config = training_config or TrainingConfig(device="cuda")
    if config.device != "cuda" and not config.device.startswith("cuda:"):
        raise GpuConfigurationError("GPU experiment training_config must use a CUDA device")
    execution_payload = dict(execution or {})
    if config.model_id == RESNET18_MODEL_ID:
        if (
            config.encoder_id != RESNET18_ENCODER_ID
            or config.encoder_weights_sha256 != RESNET18_SHA256
        ):
            raise GpuConfigurationError("GPU ResNet execution requires the admitted ImageNet encoder")
        _ticket33_pretrained_resource(execution_payload)
    elif config.model_id == RESNET18_FINETUNE_MODEL_ID:
        if (
            config.encoder_id != RESNET18_ENCODER_ID
            or config.encoder_weights_sha256 != RESNET18_SHA256
            or config.encoder_learning_rate is None
        ):
            raise GpuConfigurationError("Ticket 39 must use the admitted encoder and layer4 schedule")
        _ticket39_pretrained_resource(execution_payload)
    elif execution_payload.get("pretrained_resource") is not None:
        raise GpuConfigurationError("pretrained GPU resource metadata requires the ResNet-18 model")
    try:
        _canonical_json(execution_payload)
    except (TypeError, ValueError) as exc:
        raise GpuConfigurationError("execution metadata must be JSON-serializable") from exc
    if resource.smoke_test and config.successful_optimizer_steps > 2:
        raise GpuConfigurationError("GPU smoke requests must use at most two optimizer steps")
    if not isinstance(retry_allowed, bool):
        raise GpuConfigurationError("retry_allowed must be boolean")
    if fold is not None and fold not in (0, 1, 2):
        raise GpuConfigurationError("fold must be one of 0, 1, or 2")
    input_values: dict[str, object] = {}
    for field_name, field_value in (
        ("fold", fold),
        ("source_version", source_version),
        ("source_fingerprint", source_fingerprint),
        ("fold_manifest_checksum", fold_manifest_checksum),
        ("integration_manifest_checksum", integration_manifest_checksum),
    ):
        if field_value is not None:
            if field_name != "fold":
                _nonempty(field_value, f"GPU input {field_name}")
            input_values[field_name] = field_value
    scientific_identity = {
        "config_fingerprint": config.config_fingerprint,
        "inputs": dict(input_values),
    }
    identity = dict(experiment_identity or {})
    try:
        _canonical_json(identity)
    except (TypeError, ValueError) as exc:
        raise GpuConfigurationError("experiment_identity must be JSON-serializable") from exc
    request_id = _safe_identifier(
        experiment_id or f"gpu-{config.config_fingerprint[:12]}",
        "experiment_id",
    )
    root = Path(output_root).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"refusing to overwrite existing GPU request directory {root}")
    ledger_path = (
        Path(campaign_ledger_path).resolve()
        if campaign_ledger_path is not None
        else (root.parent / ".gpu-campaign-ledger.json").resolve()
    )
    ledger = initialize_gpu_campaign_ledger(
        ledger_path,
        reserved_comparison_hours=reserved_comparison_hours,
    )
    retry_limit = GPU_RETRY_LIMIT if retry_allowed else 0
    request_reservation_seconds = (retry_limit + 1) * resource.wall_time_minutes * 60
    if (
        not isinstance(comparison_reserve_release_minutes, int)
        or isinstance(comparison_reserve_release_minutes, bool)
        or comparison_reserve_release_minutes < 0
    ):
        raise GpuConfigurationError(
            "comparison_reserve_release_minutes must be a non-negative integer"
        )
    comparison_reserve_release_seconds = comparison_reserve_release_minutes * 60
    if comparison_reserve_release_seconds > resource.wall_time_minutes * 60:
        raise GpuConfigurationError(
            "comparison reserve release cannot exceed one fold's request limit"
        )
    capacity = check_gpu_campaign_capacity(
        ledger_path,
        request_reserved_seconds=request_reservation_seconds,
        comparison_reserve_release_seconds=comparison_reserve_release_seconds,
    )
    comparison_reserve_seconds = int(cast(int, ledger["reserved_comparison_seconds"]))
    remaining_after_request_and_reserve = (
        capacity["remaining_seconds"]
        - request_reservation_seconds
        - comparison_reserve_seconds
        + comparison_reserve_release_seconds
    )
    if remaining_after_request_and_reserve < 0:
        raise GpuCampaignBudgetError("GPU request would consume the protected comparison reserve")
    root.mkdir(parents=True, exist_ok=True)
    evidence_payload: dict[str, object] = {
        "schema_version": GPU_EXECUTION_SCHEMA_VERSION,
        "execution_file": "gpu-execution.json",
        "failure_file": "gpu-failure.json",
        "checkpoint_directory": "training/checkpoints",
        "required_fields": [
            "slurm_allocation",
            "hardware",
            "training",
            "checkpoint",
            "retry",
            "scope",
        ],
    }
    payload: dict[str, object] = {
        "schema_version": GPU_EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": request_id,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "status": "owner_approval_required",
        "resources": resource.to_dict(),
        "cpu_preflight": {
            "path": str(preflight_file),
            "sha256": _sha256_file(preflight_file),
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "passed": True,
        },
        "experiment": {
            "training_config": config.to_dict(),
            "scientific_config_fingerprint": config.config_fingerprint,
            "identity": identity,
            "scientific_identity": scientific_identity,
        },
        "inputs": input_values,
        "approval": {
            "required": True,
            "approved": False,
            "scope": GPU_APPROVAL_SCOPE,
            "retry_allowed": retry_allowed,
            "retry_limit": GPU_RETRY_LIMIT,
        },
        "evidence": evidence_payload,
        "campaign": {
            "ledger_path": str(ledger_path),
            "budget_seconds": int(cast(int, ledger["budget_seconds"])),
            "reserved_comparison_seconds": comparison_reserve_seconds,
            "comparison_reserve_release_seconds": comparison_reserve_release_seconds,
            "comparison_reserve_after_request_seconds": (
                comparison_reserve_seconds - comparison_reserve_release_seconds
            ),
            "remaining_hours_before_request": capacity["remaining_seconds"] / 3600.0,
            "request_reservation_seconds": request_reservation_seconds,
            "max_attempts": retry_limit + 1,
            "remaining_hours_after_request_and_comparison_reserve": (
                remaining_after_request_and_reserve / 3600.0
            ),
            "accounting": "Slurm queue wait and allocated wall time are recorded separately per attempt.",
        },
        "execution": execution_payload,
    }
    payload["request_checksum"] = _json_sha256(payload)
    request_path = root / "request.json"
    atomic_json_write(request_path, payload)
    return GpuExperimentRequest(root, request_path, str(payload["request_checksum"]), payload)


__all__ = [
    "GPU_ALLOWED_PARTITIONS",
    "GPU_APPROVAL_SCHEMA_VERSION",
    "GPU_APPROVAL_SCOPE",
    "GPU_DEFAULT_PARTITION",
    "GPU_EXECUTION_SCHEMA_VERSION",
    "GPU_EXECUTION_RECORD_SCHEMA_VERSION",
    "GPU_EXPERIMENT_SCHEMA_VERSION",
    "GPU_PREFLIGHT_SCHEMA_VERSION",
    "GPU_RETRY_LIMIT",
    "GPU_SMOKE_PARTITION",
    "GPU_SMOKE_WALL_TIME_LIMIT",
    "GPU_SUBMISSION_SCHEMA_VERSION",
    "GpuApproval",
    "GpuApprovalError",
    "GpuConfigurationError",
    "GpuExecutionError",
    "GpuExperimentError",
    "GpuExperimentRequest",
    "GpuExecutionResult",
    "GpuResourceError",
    "GpuResourceRequest",
    "GpuSubmissionResult",
    "classify_gpu_failure",
    "collect_gpu_environment",
    "load_gpu_approval",
    "load_gpu_experiment_request",
    "load_cpu_preflight",
    "prepare_gpu_experiment",
    "run_approved_gpu_experiment",
    "run_gpu_smoke_test",
    "submit_gpu_experiment",
    "validate_gpu_environment",
    "verify_slurm_gpu_allocation",
]
