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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

import torch

from .archive import atomic_json_write
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
    train_fixed_budget,
    train_one_fold,
    _approved_gpu_execution_capability,
)


GPU_EXPERIMENT_SCHEMA_VERSION = "rsna-knee-gpu-experiment-v1"
GPU_APPROVAL_SCHEMA_VERSION = "rsna-knee-gpu-owner-approval-v1"
GPU_EXECUTION_SCHEMA_VERSION = "rsna-knee-gpu-execution-v1"
GPU_APPROVAL_SCOPE = "ticket-13-approved-gpu-experiment"
GPU_DEFAULT_PARTITION = "cluster1-gpu"
GPU_SMOKE_PARTITION = "cluster1-gpu-add"
GPU_ALLOWED_PARTITIONS = (GPU_DEFAULT_PARTITION, GPU_SMOKE_PARTITION)
GPU_RETRY_LIMIT = 1
GPU_SMOKE_WALL_TIME_LIMIT = 60


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
        if not 1 <= self.wall_time_minutes <= 24 * 60:
            raise GpuConfigurationError("wall_time_minutes must be between 1 and 1440")
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
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GpuConfigurationError("GPU request training_config fields are invalid") from exc
    if config.to_dict() != dict(value):
        raise GpuConfigurationError("GPU request training_config is not canonical")
    return config


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
    inputs = payload.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise GpuConfigurationError("GPU request inputs section is invalid")
    if config.device != "cuda" and not config.device.startswith("cuda:"):
        raise GpuConfigurationError("GPU request must use a CUDA device")
    if resources.smoke_test and config.successful_optimizer_steps > 2:
        raise GpuConfigurationError("GPU smoke requests must use at most two optimizer steps")
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
        "exclusive_requested": resource.exclusive,
        "nominal_memory_assumed": False,
    }


def _nvidia_smi_facts() -> dict[str, object]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free,memory.used",
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
    fields = [item.strip() for item in completed.stdout.splitlines()[0].split(",")]
    if len(fields) != 5 or not fields[0] or not fields[1]:
        raise GpuResourceError("nvidia-smi returned an unparseable GPU fact row")
    facts: dict[str, object] = {
        "nvidia_smi_device_name": fields[0],
        "driver_version": fields[1],
    }
    for key, value in (
        ("nvidia_smi_memory_total_mb", fields[2]),
        ("nvidia_smi_memory_free_mb", fields[3]),
        ("nvidia_smi_memory_used_mb", fields[4]),
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
        facts = _nvidia_smi_facts()
        return {
            "device_index": device_index,
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
    peak_vram_bytes = cast(int, normalized["peak_vram_bytes"])
    if visible_device_count != resource.gpu_count:
        raise GpuResourceError("CUDA exposes a different GPU count than the approved request")
    if available_memory_bytes <= 0:
        raise GpuResourceError("GPU allocation reports no available memory")
    if runtime_visible_memory_limit_bytes <= 0:
        raise GpuResourceError("GPU runtime reports no visible memory limit")
    if available_memory_bytes > runtime_visible_memory_limit_bytes:
        raise GpuResourceError("GPU free memory exceeds the visible memory limit")
    if peak_vram_bytes > runtime_visible_memory_limit_bytes:
        raise GpuResourceError("observed peak VRAM exceeds the runtime-visible memory limit")
    if allocation.get("partition") != resource.partition:
        raise GpuResourceError("GPU hardware facts are bound to a different Slurm partition")
    available_memory_source = str(normalized["available_memory_source"])
    if available_memory_source.lower() in {"nominal", "device_specification", "user_report"}:
        raise GpuResourceError("GPU available memory must come from an observed allocation")
    normalized["resource_request"] = resource.to_dict()
    normalized["allocation"] = dict(allocation)
    normalized["nominal_device_memory_used"] = False
    sharing_behavior = str(allocation.get("sharing_behavior", "")).strip().lower()
    normalized["memory_assignment_verified"] = sharing_behavior in {
        "0",
        "false",
        "no",
        "exclusive",
        "unshared",
    }
    return normalized


def collect_gpu_environment(
    *,
    resource: GpuResourceRequest | None = None,
    slurm_environment: Mapping[str, str] | None = None,
    probe: Callable[[], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Capture actual CUDA facts, optionally binding them to a Slurm request."""

    allocation = (
        verify_slurm_gpu_allocation(resource, slurm_environment)
        if resource is not None
        else None
    )
    snapshot = dict(probe() if probe is not None else _default_gpu_probe())
    if resource is not None and allocation is not None:
        return validate_gpu_environment(snapshot, resource=resource, allocation=allocation)
    return snapshot


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
    return lines[-1].split(";", 1)[0]


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


def submit_gpu_experiment(
    request_path: str | Path,
    approval_path: str | Path,
    *,
    script_path: str | Path = "slurm/run_ticket13_gpu_experiment.sbatch",
    environment: Mapping[str, str] | None = None,
    sbatch_binary: str = "sbatch",
    wait: bool = True,
    runner: Callable[..., Any] | None = None,
) -> GpuSubmissionResult:
    """Submit the exact approved resource request through ``sbatch``.

    Validation happens before the scheduler command is built.  With
    ``wait=True`` a single unchanged submission is permitted after a clearly
    transient failure, provided the owner approval covers that retry.  The
    function never invokes ``srun`` and never changes resources or scientific
    configuration on retry.
    """

    request = load_gpu_experiment_request(request_path)
    approval = load_gpu_approval(approval_path, request)
    script = Path(script_path).resolve()
    if not script.is_file():
        raise GpuExecutionError(f"GPU sbatch adapter is not a regular file: {script}")
    if not isinstance(sbatch_binary, str) or not sbatch_binary.strip():
        raise GpuConfigurationError("sbatch_binary must be non-empty")
    resource = request.resources
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
            "RSNA_TICKET13_APPROVAL": str(Path(approval_path).resolve()),
            "RSNA_TICKET13_PREFLIGHT": str(
                Path(_nonempty(preflight_reference["path"], "CPU preflight path")).resolve()
            ),
            "RSNA_TICKET13_PREVIOUS_RETRIES": "0",
        }
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
    attempts: list[Mapping[str, object]] = []
    job_ids: list[str] = []
    retry_count = 0
    last_failure: dict[str, object] = {}
    while True:
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
        if submitted_job_id is not None and returncode == 0:
            job_ids.append(submitted_job_id)
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
        last_failure = classification.to_dict()
        if (
            not wait
            or approval.payload.get("retry_allowed") is not True
            or not classification.retryable
            or retry_count >= GPU_RETRY_LIMIT
        ):
            return GpuSubmissionResult(
                status="failed",
                job_ids=tuple(job_ids),
                retry_count=retry_count,
                attempts=tuple(attempts),
                last_failure=last_failure,
            )
        retry_count += 1
        submission_environment["RSNA_TICKET13_PREVIOUS_RETRIES"] = str(retry_count)


def _write_gpu_failure(
    destination: Path,
    *,
    request: GpuExperimentRequest,
    approval: GpuApproval,
    preflight_path: Path,
    retry_count: int,
    attempts: Sequence[Mapping[str, object]],
    error: BaseException,
) -> None:
    atomic_json_write(
        destination / "gpu-failure.json",
        {
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
        },
        overwrite=True,
    )


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
    allocation_environment = dict(os.environ if slurm_environment is None else slurm_environment)
    allocation = verify_slurm_gpu_allocation(resource, allocation_environment)
    if not resource.smoke_test and str(allocation["sharing_behavior"]).strip().lower() not in {
        "0",
        "false",
        "no",
        "exclusive",
        "unshared",
    }:
        raise GpuResourceError(
            "full GPU training requires a verified unshared allocation; use a smoke request for shared memory"
        )
    before = collect_gpu_environment(
        resource=resource,
        slurm_environment=allocation_environment,
        probe=gpu_probe,
    )

    if training_runner is None:
        if studies is None or fold_manifest is None or fold is None:
            raise GpuConfigurationError("production GPU execution requires studies, fold manifest, and fold")
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
            raise GpuConfigurationError("approved GPU request does not bind every training input identity")
        if request_inputs.get("fold_manifest_checksum") != fold_manifest.manifest_checksum:
            raise GpuConfigurationError("fold manifest checksum differs from the approved request")

        def training_runner(resume_from: Path | None) -> TrainingResult:
            return train_one_fold(
                studies,
                fold_manifest,
                fold,
                destination,
                config=config,
                source_version=source_version,
                source_fingerprint=source_fingerprint,
                integration_manifest_checksum=integration_manifest_checksum,
                series_cache=series_cache,
                preflight=preflight,
                require_preflight=True,
                allow_gpu=True,
                gpu_authorization=_approved_gpu_execution_capability(),
                device=config.device,
                resume_from=resume_from,
                requested_resources=resource.to_dict(),
            )

    attempts: list[Mapping[str, object]] = []
    retry_count = previous_retries
    while True:
        try:
            training = training_runner(
                destination / "checkpoints" / "latest.pt"
                if (destination / "checkpoints" / "latest.pt").is_file()
                and retry_count > 0
                else None
            )
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
                    "memory_assignment_verified": after["memory_assignment_verified"],
                    "nominal_device_memory_used": False,
                },
                "training": training.to_public_dict(),
                "checkpoint": {
                    "final_present": True,
                    "final_sha256": final_sha256,
                    "latest_present": latest_checkpoint is not None
                    and latest_checkpoint.is_file(),
                    "final_reload_passed": True,
                    "resumed_from_checkpoint": training.resumed_from_checkpoint,
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
                    "error": str(exc),
                    "retry": classification.to_dict(),
                }
            )
            if (
                approval.payload.get("retry_allowed") is True
                and classification.retryable
                and retry_count < GPU_RETRY_LIMIT
            ):
                retry_count += 1
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
    destination = Path(
        output_dir or request.root.parent / f"{request.root.name}-execution"
    ).resolve()

    def runner(resume_from: Path | None) -> TrainingResult:
        return train_fixed_budget(
            examples,
            config,
            artifact_dir=destination,
            resume_from=resume_from,
            requested_resources=request.resources.to_dict(),
            gpu_authorization=_approved_gpu_execution_capability(),
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
    identity = dict(experiment_identity or {})
    try:
        _canonical_json(identity)
    except (TypeError, ValueError) as exc:
        raise GpuConfigurationError("experiment_identity must be JSON-serializable") from exc
    request_id = _safe_identifier(
        experiment_id or f"ticket13-gpu-{config.config_fingerprint[:12]}",
        "experiment_id",
    )
    root = Path(output_root).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"refusing to overwrite existing GPU request directory {root}")
    root.mkdir(parents=True, exist_ok=True)
    execution_payload = dict(execution or {})
    try:
        _canonical_json(execution_payload)
    except (TypeError, ValueError) as exc:
        raise GpuConfigurationError("execution metadata must be JSON-serializable") from exc
    evidence_payload: dict[str, object] = {
        "schema_version": GPU_EXECUTION_SCHEMA_VERSION,
        "execution_file": "gpu-execution.json",
        "failure_file": "gpu-failure.json",
        "checkpoint_directory": "checkpoints",
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
    "GPU_EXPERIMENT_SCHEMA_VERSION",
    "GPU_RETRY_LIMIT",
    "GPU_SMOKE_PARTITION",
    "GPU_SMOKE_WALL_TIME_LIMIT",
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
