"""Ticket 33's separately approved private Kaggle T4 smoke and resume check."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, cast

import numpy as np
import torch

from .archive import atomic_json_write
from .feature_cache import FrozenFeatureCache
from .folds import FoldManifest, load_fold_manifest
from .pretrained import RESNET18_SHA256, TICKET33_RESOURCE_SCOPE, load_resource_approval, load_resnet18_weights
from .real_inference import (
    KaggleCompetitionSource,
    IntegrationManifest,
    iter_real_studies,
    load_integration_manifest,
)
from .resnet18 import RESNET18_FEATURE_DIM, ResNet18StudyModel
from .training import (
    CpuPreflightResult,
    TrainingConfig,
    TrainingDataError,
    TrainingInterrupted,
    TrainingResult,
    _attach_cached_frozen_features,
    _default_run_id,
    _model_state_sha256,
    _new_kaggle_gpu_execution_authorization,
    _train_fixed_budget_gpu,
    checkpoint_sha256,
    fingerprint_training_examples,
    load_training_checkpoint,
    require_cpu_preflight,
    run_cpu_preflight,
    build_fold_training_examples,
    seed_everything,
)
from .model import build_study_model


KAGGLE_REQUEST_SCHEMA = "rsna-ticket33-kaggle-gpu-request-v1"
KAGGLE_APPROVAL_SCHEMA = "rsna-ticket33-kaggle-gpu-approval-v1"
KAGGLE_RESOURCE_ADDENDUM_SCHEMA = "rsna-ticket33-kaggle-resource-addendum-v1"
KAGGLE_APPROVAL_SCOPE = "ticket33-kaggle-free-t4-validation"
KAGGLE_COMPETITION_SLUG = "rsna-knee-abnormality-detection"
KAGGLE_ACCELERATOR = "NvidiaTeslaT4"
KAGGLE_KERNEL_ID = "jitheswar/ticket33-kaggle-free-t4-validation"
KAGGLE_BUNDLE_DATASET_SLUG = "jitheswar/ticket33-kaggle-gpu-bundle"
KAGGLE_MAX_WALL_SECONDS = 30 * 60
KAGGLE_ALLOCATED_GPU_COUNT = 2
KAGGLE_VISIBLE_GPU_COUNT = 1
KAGGLE_MAX_CPU_COUNT = 4
KAGGLE_MAX_MEMORY_GB = 29
KAGGLE_MAX_WORKING_STORAGE_GB = 20
KAGGLE_TRAINING_STUDY_COUNT = 2
KAGGLE_OUTPUT_SCHEMA = "rsna-ticket33-kaggle-gpu-evidence-v1"


class KaggleGpuError(RuntimeError):
    """Raised when a Kaggle request, runtime or measured smoke is unsafe."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise KaggleGpuError(f"{description} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise KaggleGpuError(f"could not read {description}") from exc
    if not isinstance(value, Mapping):
        raise KaggleGpuError(f"{description} must contain a JSON object")
    return value


def _bundle_file(root: Path, record: object, description: str) -> Path:
    if not isinstance(record, Mapping):
        raise KaggleGpuError(f"request is missing {description} bundle identity")
    relative = record.get("path")
    expected_digest = record.get("sha256")
    expected_bytes = record.get("bytes")
    if not isinstance(relative, str) or not relative.strip():
        raise KaggleGpuError(f"request has an invalid {description} path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise KaggleGpuError(f"request {description} path must stay inside its immutable bundle")
    if (
        not isinstance(expected_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
        or not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
    ):
        raise KaggleGpuError(f"request has an invalid {description} file identity")
    path = root.joinpath(*pure.parts)
    try:
        resolved_root = root.resolve()
        path.resolve().relative_to(resolved_root)
    except ValueError as exc:
        raise KaggleGpuError(f"request {description} path escapes its bundle") from exc
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise KaggleGpuError(f"request {description} path crosses a symlink")
    if path.is_symlink() or not path.is_file():
        raise KaggleGpuError(f"request {description} file is unavailable")
    if path.stat().st_size != expected_bytes or _sha256(path) != expected_digest:
        raise KaggleGpuError(f"request {description} bytes do not match the approved identity")
    return path


@dataclass(frozen=True)
class KaggleGpuRequest:
    root: Path
    request_path: Path
    request_checksum: str
    payload: Mapping[str, object]
    training_config: TrainingConfig

    @property
    def resources(self) -> Mapping[str, object]:
        value = self.payload.get("resources")
        if not isinstance(value, Mapping):
            raise KaggleGpuError("Kaggle request resource envelope is invalid")
        return dict(value)

    @property
    def wall_time_limit_seconds(self) -> int:
        value = self.resources.get("wall_time_limit_seconds")
        if not isinstance(value, int) or isinstance(value, bool):
            raise KaggleGpuError("Kaggle request wall-time limit is invalid")
        return value


@dataclass(frozen=True)
class KaggleGpuApproval:
    path: Path
    approval_checksum: str
    payload: Mapping[str, object]


def load_kaggle_gpu_request(path: str | Path) -> KaggleGpuRequest:
    """Load an immutable request and recheck its exact notebook bundle."""

    candidate = Path(path).resolve()
    request_path = candidate / "request.json" if candidate.is_dir() else candidate
    payload = _read_json(request_path, "Kaggle GPU request")
    if payload.get("schema_version") != KAGGLE_REQUEST_SCHEMA:
        raise KaggleGpuError("Kaggle request has an unexpected schema")
    checksum = payload.get("request_checksum")
    unsigned = dict(payload)
    unsigned.pop("request_checksum", None)
    if not isinstance(checksum, str) or _json_sha256(unsigned) != checksum:
        raise KaggleGpuError("Kaggle request checksum is invalid")
    if payload.get("status") != "owner_approval_required":
        raise KaggleGpuError("Kaggle request is not approval-gated")
    approval_policy = payload.get("approval")
    if not isinstance(approval_policy, Mapping) or (
        approval_policy.get("required") is not True
        or approval_policy.get("approved") is not False
        or approval_policy.get("scope") != KAGGLE_APPROVAL_SCOPE
        or approval_policy.get("retry_allowed") is not False
    ):
        raise KaggleGpuError("Kaggle request owner-approval policy is invalid")
    resources = payload.get("resources")
    expected_resources = {
        "backend": "kaggle-private-notebook",
        "accelerator": KAGGLE_ACCELERATOR,
        "assigned_gpu_count": KAGGLE_ALLOCATED_GPU_COUNT,
        "visible_gpu_count": KAGGLE_VISIBLE_GPU_COUNT,
        "cpu_limit": KAGGLE_MAX_CPU_COUNT,
        "memory_limit_gb": KAGGLE_MAX_MEMORY_GB,
        "working_storage_limit_gb": KAGGLE_MAX_WORKING_STORAGE_GB,
        "internet_enabled": False,
        "is_private": True,
        "external_spend_usd": 0,
    }
    if not isinstance(resources, Mapping):
        raise KaggleGpuError("Kaggle request resource envelope is invalid")
    wall_time_limit = resources.get("wall_time_limit_seconds")
    if (
        isinstance(wall_time_limit, bool)
        or not isinstance(wall_time_limit, int)
        or not 60 <= wall_time_limit <= KAGGLE_MAX_WALL_SECONDS
        or {key: value for key, value in resources.items() if key != "wall_time_limit_seconds"}
        != expected_resources
    ):
        raise KaggleGpuError("Kaggle resource envelope differs from the approved free T4 scope")
    if payload.get("kernel_id") != KAGGLE_KERNEL_ID:
        raise KaggleGpuError("Kaggle request names a different private kernel")
    if payload.get("competition_slug") != KAGGLE_COMPETITION_SLUG:
        raise KaggleGpuError("Kaggle request names a different competition source")
    bundle = payload.get("bundle")
    if not isinstance(bundle, Mapping):
        raise KaggleGpuError("Kaggle request is missing its input bundle identity")
    if bundle.get("kaggle_dataset_slug") != KAGGLE_BUNDLE_DATASET_SLUG:
        raise KaggleGpuError("Kaggle request does not bind its private input dataset")
    file_records = bundle.get("files")
    if not isinstance(file_records, Mapping):
        raise KaggleGpuError("Kaggle request bundle file map is invalid")
    root = request_path.parent.resolve()
    for name, record in file_records.items():
        if not isinstance(name, str) or not isinstance(record, Mapping):
            raise KaggleGpuError("Kaggle request bundle file map is malformed")
        expected_path = _bundle_file(root, {**dict(record), "path": name}, f"bundle member {name}")
        if expected_path.name == "":
            raise KaggleGpuError("Kaggle request contains an invalid bundle member")
    source_files = bundle.get("source_files")
    if not isinstance(source_files, Mapping) or not source_files:
        raise KaggleGpuError("Kaggle request does not bind the uploaded source code")
    code_hash = hashlib.sha256()
    for name in sorted(source_files):
        if not isinstance(name, str):
            raise KaggleGpuError("Kaggle source file path is invalid")
        record = source_files[name]
        source_file = _bundle_file(root, {**cast(Mapping[str, object], record), "path": name}, "source code")
        code_hash.update(name.encode("utf-8"))
        code_hash.update(b"\0")
        code_hash.update(source_file.read_bytes())
        code_hash.update(b"\0")
    if bundle.get("source_tree_sha256") != code_hash.hexdigest():
        raise KaggleGpuError("Kaggle source tree differs from its request-bound checksum")
    from .gpu import _training_config_from_dict

    experiment = payload.get("experiment")
    if not isinstance(experiment, Mapping):
        raise KaggleGpuError("Kaggle request scientific configuration is missing")
    config = _training_config_from_dict(experiment.get("training_config"))
    if (
        experiment.get("scientific_config_fingerprint") != config.config_fingerprint
        or config.model_id != "resnet18-frozen-slice-features-v1"
        or config.encoder_weights_sha256 != RESNET18_SHA256
        or config.device != "cuda"
        or config.successful_optimizer_steps != 2
        or config.batch_size != 2
    ):
        raise KaggleGpuError("Kaggle request changed the frozen Ticket 33 science")
    identities = payload.get("inputs")
    if not isinstance(identities, Mapping) or identities.get("fold") != 0:
        raise KaggleGpuError("Kaggle request does not bind the Ticket 33 fold-0 inputs")
    for field_name in (
        "source_version",
        "source_fingerprint",
        "fold_manifest_checksum",
        "integration_manifest_checksum",
    ):
        value = identities.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise KaggleGpuError(f"Kaggle request has an invalid {field_name}")
        if field_name != "source_version" and re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise KaggleGpuError(f"Kaggle request has an invalid {field_name}")
    expected_scientific_identity = {
        "config_fingerprint": config.config_fingerprint,
        "inputs": {
            key: identities[key]
            for key in (
                "fold",
                "source_version",
                "source_fingerprint",
                "fold_manifest_checksum",
                "integration_manifest_checksum",
            )
        },
    }
    if payload.get("scientific_identity") != expected_scientific_identity:
        raise KaggleGpuError("Kaggle request changed the frozen scientific input identity")
    selection = payload.get("input_selection")
    if selection != {
        "partition": "reference-training-only",
        "study_count": KAGGLE_TRAINING_STUDY_COUNT,
        "policy": "first-two-study-identifiers-in-canonical-fold0-training-membership",
        "held_out_membership_read": False,
        "report_supervision": False,
        "purpose": "hardware-only smoke; no candidate evidence",
    }:
        raise KaggleGpuError("Kaggle request input selection policy is invalid")
    if payload.get("preflight_policy") != {
        "required_checks": [
            "supported_python",
            "core_dependencies_import",
            "known_roc_auc",
            "finite_model_loss_and_gradients",
            "checkpoint_save_reload",
            "inference_import_isolated",
            "real_t4_cuda",
            "assigned_vram_measured",
        ],
        "python_compatibility_policy": "ticket33-kaggle-python-3.12-compatibility-v1",
        "slurm_required": False,
        "isolated_python_required": False,
        "internet_enabled": False,
    }:
        raise KaggleGpuError("Kaggle CPU/GPU preflight policy differs from the request")
    return KaggleGpuRequest(root, request_path, checksum, payload, config)


def load_kaggle_gpu_approval(
    path: str | Path,
    request: KaggleGpuRequest,
) -> KaggleGpuApproval:
    """Require an owner decision bound to this exact Kaggle request."""

    approval_path = Path(path).resolve()
    payload = _read_json(approval_path, "Kaggle GPU owner approval")
    if payload.get("schema_version") != KAGGLE_APPROVAL_SCHEMA:
        raise KaggleGpuError("Kaggle approval has an unexpected schema")
    if (
        payload.get("approved") is not True
        or payload.get("approval_scope") != KAGGLE_APPROVAL_SCOPE
        or payload.get("request_checksum") != request.request_checksum
        or payload.get("retry_allowed") is not False
        or payload.get("external_spend_usd") != 0
    ):
        raise KaggleGpuError("Kaggle approval does not authorize this exact free private smoke")
    approver = payload.get("approved_by")
    timestamp = payload.get("approved_at_utc")
    if not isinstance(approver, str) or not approver.strip() or not isinstance(timestamp, str):
        raise KaggleGpuError("Kaggle approval is missing its owner or timestamp")
    try:
        from datetime import datetime

        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise KaggleGpuError("Kaggle approval timestamp is invalid") from exc
    return KaggleGpuApproval(approval_path, _sha256(approval_path), payload)


def _load_resource_addendum(request: KaggleGpuRequest) -> Mapping[str, object]:
    execution = request.payload.get("execution")
    if not isinstance(execution, Mapping):
        raise KaggleGpuError("Kaggle request is missing its resource decision")
    resource_record = execution.get("ticket33_resource_approval")
    addendum_record = execution.get("kaggle_resource_addendum")
    resource_path = _bundle_file(request.root, resource_record, "Ticket 33 resource approval")
    addendum_path = _bundle_file(request.root, addendum_record, "Kaggle resource addendum")
    try:
        resource = load_resource_approval(resource_path)
    except ValueError as exc:
        raise KaggleGpuError("existing Ticket 33 pretrained resource approval is invalid") from exc
    if resource.get("scope") != TICKET33_RESOURCE_SCOPE:
        raise KaggleGpuError("pretrained resource approval does not retain Ticket 33 scope")
    addendum = _read_json(addendum_path, "Kaggle resource addendum")
    if (
        addendum.get("schema_version") != KAGGLE_RESOURCE_ADDENDUM_SCHEMA
        or addendum.get("scope") != KAGGLE_APPROVAL_SCOPE
        or addendum.get("supersedes_only_processing_backend") is not True
        or addendum.get("external_spend_usd") != 0
        or addendum.get("internet_enabled") is not False
        or addendum.get("public_or_redistribution_authorized") is not False
        or addendum.get("upstream_resource_approval_sha256") != _sha256(resource_path)
        or addendum.get("pretrained_weights_sha256") != RESNET18_SHA256
    ):
        raise KaggleGpuError("Kaggle resource addendum does not preserve the existing resource gates")
    return addendum


def _validate_kernel_metadata(request: KaggleGpuRequest) -> Mapping[str, object]:
    metadata_path = _bundle_file(
        request.root,
        cast(Mapping[str, object], cast(Mapping[str, object], request.payload["execution"])["kernel_metadata"]),
        "Kaggle kernel metadata",
    )
    metadata = _read_json(metadata_path, "Kaggle kernel metadata")
    dataset_slug = cast(Mapping[str, object], request.payload["bundle"])["kaggle_dataset_slug"]
    if (
        metadata.get("id") != KAGGLE_KERNEL_ID
        or metadata.get("title") != "Ticket33 Kaggle Free T4 Validation"
        or metadata.get("kernel_type") != "notebook"
        or metadata.get("code_file") != "ticket33_gpu_validation.ipynb"
        or metadata.get("enable_gpu") is not True
        or metadata.get("enable_internet") is not False
        or metadata.get("is_private") is not True
        or metadata.get("machine_shape") != KAGGLE_ACCELERATOR
        or metadata.get("dataset_sources") != [dataset_slug]
        or metadata.get("competition_sources") != [KAGGLE_COMPETITION_SLUG]
    ):
        raise KaggleGpuError("Kaggle notebook metadata differs from the approved private T4 run")
    return metadata


def _parse_nvidia_smi() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise KaggleGpuError("nvidia-smi could not verify the assigned Kaggle devices") from exc
    devices: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        columns = [item.strip() for item in line.split(",")]
        if len(columns) != 5:
            raise KaggleGpuError("nvidia-smi returned an unexpected device record")
        try:
            devices.append(
                {
                    "index": int(columns[0]),
                    "name": columns[1],
                    "total_vram_bytes": int(columns[2]) * 1024 * 1024,
                    "used_vram_bytes": int(columns[3]) * 1024 * 1024,
                    "utilization_percent": int(columns[4]),
                }
            )
        except ValueError as exc:
            raise KaggleGpuError("nvidia-smi returned an invalid device measurement") from exc
    if len(devices) != KAGGLE_ALLOCATED_GPU_COUNT or any(
        "t4" not in str(item["name"]).casefold() for item in devices
    ):
        raise KaggleGpuError("Kaggle did not assign the approved T4 x2 environment")
    return devices


def _cgroup_memory_limit() -> int | None:
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            value = path.read_text(encoding="ascii").strip()
        except OSError:
            continue
        if value == "max":
            continue
        try:
            parsed = int(value)
        except ValueError:
            continue
        if 0 < parsed < 1 << 60:
            return parsed
    return None


def _runtime_fingerprint(runtime: Mapping[str, object]) -> str:
    volatile = {
        "initial_vram_available_bytes",
        "initial_nvidia_smi_devices",
        "working_storage_free_before_bytes",
    }
    return _json_sha256({key: value for key, value in runtime.items() if key not in volatile})


def _validate_kaggle_runtime(
    request: KaggleGpuRequest,
    *,
    metadata: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]], Path]:
    if not Path("/kaggle/working").is_dir() or not Path("/kaggle/input").is_dir():
        raise KaggleGpuError("this process is not running in the Kaggle notebook filesystem")
    run_type = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").strip()
    if run_type not in {"Batch", "Interactive"}:
        raise KaggleGpuError("Kaggle kernel run identity is unavailable")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise KaggleGpuError("the approved smoke must expose only Kaggle T4 device zero")
    if not torch.cuda.is_available() or torch.cuda.device_count() != KAGGLE_VISIBLE_GPU_COUNT:
        raise KaggleGpuError("exactly one real Kaggle CUDA device must be visible to the trainer")
    visible_name = torch.cuda.get_device_name(0)
    visible_properties = torch.cuda.get_device_properties(0)
    if "t4" not in visible_name.casefold():
        raise KaggleGpuError("visible CUDA device is not an NVIDIA T4")
    devices = _parse_nvidia_smi()
    visible_vram = int(visible_properties.total_memory)
    if not 12_000_000_000 <= visible_vram <= 20_000_000_000:
        raise KaggleGpuError("visible Kaggle T4 VRAM capacity is outside the expected T4 range")
    _, available_vram = torch.cuda.mem_get_info(0)
    if available_vram < 6 * 1024**3:
        raise KaggleGpuError("Kaggle T4 has insufficient currently available VRAM")
    affinity_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    memory_limit = _cgroup_memory_limit()
    storage = shutil.disk_usage("/kaggle/working")
    working_root = Path("/kaggle/working").resolve()
    input_root = Path("/kaggle/input")
    competition_roots = (
        input_root / KAGGLE_COMPETITION_SLUG,
        input_root / "competitions" / KAGGLE_COMPETITION_SLUG,
    )
    competition_root = next(
        (
            candidate
            for candidate in competition_roots
            if candidate.is_dir() and not candidate.is_symlink()
        ),
        None,
    )
    if competition_root is None:
        raise KaggleGpuError("the exact approved competition input is not mounted")
    runtime: dict[str, object] = {
        "kernel_id": request.payload["kernel_id"],
        "run_type": run_type,
        "kernel_metadata_private": metadata.get("is_private") in {True, "true"},
        "internet_enabled": metadata.get("enable_internet") in {True, "true"},
        "accelerator": KAGGLE_ACCELERATOR,
        "assigned_gpu_count": len(devices),
        "assigned_gpu_names": [str(item["name"]) for item in devices],
        "visible_gpu_count": torch.cuda.device_count(),
        "visible_device_name": visible_name,
        "visible_vram_bytes": visible_vram,
        "initial_vram_available_bytes": int(available_vram),
        "initial_nvidia_smi_devices": devices,
        "cpu_affinity_count": affinity_count,
        "cpu_limit": KAGGLE_MAX_CPU_COUNT,
        "memory_limit_bytes": memory_limit,
        "memory_limit_gb": KAGGLE_MAX_MEMORY_GB,
        "working_storage_total_bytes": storage.total,
        "working_storage_free_before_bytes": storage.free,
        "working_root": str(working_root),
        "competition_mount": "official competition source attached by kernel metadata",
        "external_spend_usd": 0,
        "device_exclusivity": "not independently verifiable; no exclusive assignment claimed",
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
    }
    return runtime, devices, competition_root


def _runtime_preflight_path(path: Path) -> CpuPreflightResult:
    payload = _read_json(path, "Kaggle runtime preflight")
    checks = payload.get("checks")
    environment = payload.get("environment")
    details = payload.get("details")
    if (
        payload.get("schema_version") != "rsna-knee-cpu-preflight-v1"
        or not isinstance(checks, Mapping)
        or not isinstance(environment, Mapping)
        or not isinstance(details, Mapping)
        or payload.get("passed") is not True
    ):
        raise KaggleGpuError("Kaggle runtime preflight is malformed or did not pass")
    if (
        checks.get("real_t4_cuda") is not True
        or checks.get("assigned_vram_measured") is not True
        or details.get("kaggle_gpu_preflight") != {
            "accelerator": KAGGLE_ACCELERATOR,
            "assigned_gpu_count": KAGGLE_ALLOCATED_GPU_COUNT,
            "visible_gpu_count": KAGGLE_VISIBLE_GPU_COUNT,
            "internet_enabled": False,
        }
    ):
        raise KaggleGpuError("Kaggle runtime preflight did not verify the assigned T4 resources")
    result = CpuPreflightResult(
        passed=True,
        checks={str(key): value for key, value in checks.items()},
        environment=dict(environment),
        details=dict(details),
    )
    require_cpu_preflight(
        result,
        require_slurm=False,
        require_isolated_python=False,
        allow_ticket33_kaggle_python_312=True,
    )
    return result


def _run_kaggle_cpu_preflight(
    path: Path,
    *,
    runtime: Mapping[str, object],
) -> CpuPreflightResult:
    """Run the existing checks, with an explicit Python 3.12 Kaggle allowance."""

    base = run_cpu_preflight(
        require_isolated_python=False,
        require_slurm=False,
    )
    checks = dict(base.checks)
    details = dict(base.details)
    visible_vram_bytes = runtime.get("visible_vram_bytes")
    available_vram_bytes = runtime.get("initial_vram_available_bytes")
    checks["real_t4_cuda"] = bool(
        runtime.get("accelerator") == KAGGLE_ACCELERATOR
        and runtime.get("visible_gpu_count") == KAGGLE_VISIBLE_GPU_COUNT
        and "t4" in str(runtime.get("visible_device_name", "")).casefold()
    )
    checks["assigned_vram_measured"] = bool(
        isinstance(visible_vram_bytes, int)
        and isinstance(available_vram_bytes, int)
        and available_vram_bytes > 0
    )
    details["kaggle_gpu_preflight"] = {
        "accelerator": KAGGLE_ACCELERATOR,
        "assigned_gpu_count": KAGGLE_ALLOCATED_GPU_COUNT,
        "visible_gpu_count": KAGGLE_VISIBLE_GPU_COUNT,
        "internet_enabled": False,
    }
    if sys.version_info[:2] == (3, 12):
        checks.pop("python_311", None)
        checks["python_312_compatible"] = True
        details["python_version_policy"] = "ticket33-kaggle-python-3.12-compatibility-v1"
    passed = all(checks.values())
    result = CpuPreflightResult(passed, checks, base.environment, details)
    atomic_json_write(path, result.to_dict())
    if not passed:
        print(
            json.dumps(
                {
                    "ticket33_kaggle_cpu_preflight_checks": checks,
                    "ticket33_kaggle_cpu_preflight_details": details,
                },
                sort_keys=True,
            )
        )
    require_cpu_preflight(
        result,
        require_slurm=False,
        require_isolated_python=False,
        allow_ticket33_kaggle_python_312=True,
    )
    return result


def validate_kaggle_training_authorization(
    authorization: object,
    *,
    request_path: str | Path,
    approval_path: str | Path,
    preflight_path: str | Path,
    resource: Mapping[str, object],
    preflight: CpuPreflightResult,
) -> object:
    """Recheck the private Kaggle request and runtime at the core training seam."""

    from .training import _KaggleGpuExecutionAuthorization

    if not isinstance(authorization, _KaggleGpuExecutionAuthorization):
        raise KaggleGpuError("Kaggle training authorization has an invalid private type")
    request = load_kaggle_gpu_request(request_path)
    approval = load_kaggle_gpu_approval(approval_path, request)
    if (
        request.request_checksum != authorization.request_checksum
        or approval.approval_checksum != authorization.approval_checksum
        or request.resources != dict(resource)
    ):
        raise KaggleGpuError("Kaggle training authorization does not match the request and approval")
    preflight_file = Path(preflight_path).resolve()
    if _sha256(preflight_file) != authorization.preflight_checksum:
        raise KaggleGpuError("Kaggle runtime preflight checksum changed")
    loaded_preflight = _runtime_preflight_path(preflight_file)
    if loaded_preflight != preflight:
        raise KaggleGpuError("Kaggle runtime preflight differs from the validated result")
    metadata = _validate_kernel_metadata(request)
    runtime, _, _ = _validate_kaggle_runtime(request, metadata=metadata)
    if (
        authorization.kernel_id != request.payload["kernel_id"]
        or authorization.runtime_fingerprint != _runtime_fingerprint(runtime)
    ):
        raise KaggleGpuError("Kaggle runtime identity changed after the request gate")
    _load_resource_addendum(request)
    return authorization


def _find_competition_root(
    base: Path,
    manifest: IntegrationManifest,
    selected_uids: tuple[str, ...],
) -> Path:
    selected = set(selected_uids)
    first_entry = next(
        entry
        for study in manifest.studies
        if study.study_instance_uid in selected
        for series in study.series
        for entry in series.entries
    )
    candidates = [base]
    try:
        candidates.extend(path for path in base.iterdir() if path.is_dir() and not path.is_symlink())
    except OSError as exc:
        raise KaggleGpuError("could not inspect the competition input mount") from exc
    matches = [
        candidate
        for candidate in candidates
        if (candidate / Path(*PurePosixPath(first_entry).parts)).is_file()
    ]
    if len(matches) != 1:
        raise KaggleGpuError("manifest-owned input path is absent or ambiguous in the Kaggle source")
    return matches[0].resolve()


def _select_training_membership(fold_manifest: FoldManifest, fold: int) -> tuple[str, ...]:
    memberships = sorted(
        (
            item
            for item in fold_manifest.gold_memberships
            if item.fold != fold and any(label is not None for label in item.reference_labels)
        ),
        key=lambda item: item.study_instance_uid,
    )
    if len(memberships) < KAGGLE_TRAINING_STUDY_COUNT:
        raise TrainingDataError("fold 0 has too few reference-labeled training studies for the smoke")
    selected = tuple(item.study_instance_uid for item in memberships[:KAGGLE_TRAINING_STUDY_COUNT])
    fold_manifest.assert_adaptation_input_is_training_only(fold, selected)
    return selected


def _prepare_training_examples(
    request: KaggleGpuRequest,
    fold_manifest: FoldManifest,
    integration_manifest: IntegrationManifest,
    competition_mount: Path,
) -> tuple[tuple[Any, ...], dict[str, object]]:
    selected = _select_training_membership(fold_manifest, fold=0)
    if not set(selected).issubset(set(integration_manifest.study_uids)):
        raise KaggleGpuError("approved training membership is missing from the integration manifest")
    source_root = _find_competition_root(competition_mount, integration_manifest, selected)
    attestation = KaggleCompetitionSource(
        root=source_root,
        competition_slug=KAGGLE_COMPETITION_SLUG,
        kernel_id=KAGGLE_KERNEL_ID,
        request_checksum=request.request_checksum,
    )
    loaded = tuple(
        iter_real_studies(
            integration_manifest,
            source_root,
            require_complete_series=True,
            study_uids=selected,
            kaggle_source=attestation,
        )
    )
    studies = {item.study.study_instance_uid: item.study for item in loaded}
    if set(studies) != set(selected):
        raise KaggleGpuError("Kaggle input loader changed the approved training membership")
    examples = build_fold_training_examples(
        studies,
        fold_manifest,
        0,
        preprocess_config=request.training_config.preprocess_config(),
        packing_config=request.training_config.packing_config(),
        selected_training_study_uids=selected,
    )
    if len(examples) != KAGGLE_TRAINING_STUDY_COUNT or any(
        item.provenance.source_kind != "reference"
        or item.provenance.fold != 0
        or not item.provenance.held_out_excluded
        or not bool(item.supervision_mask.any())
        for item in examples
    ):
        raise KaggleGpuError("Kaggle smoke examples failed the fold and reference-only gates")
    file_count = sum(
        len(series.entries)
        for study in integration_manifest.studies
        if study.study_instance_uid in set(selected)
        for series in study.series
    )
    source_digests = sorted(
        item.source_content_sha256
        for loaded_study in loaded
        for series in loaded_study.study.series
        for item in series.slices
        if item.source_content_sha256 is not None
    )
    decoded_slice_count = sum(
        len(series.slices)
        for loaded_study in loaded
        for series in loaded_study.study.series
    )
    if len(source_digests) != decoded_slice_count:
        raise KaggleGpuError("Kaggle DICOM loader did not retain every decoded source-file digest")
    source_evidence = {
        "competition_slug": KAGGLE_COMPETITION_SLUG,
        "source_version": integration_manifest.source_version,
        "canonical_archive_fingerprint": integration_manifest.archive_fingerprint,
        "source_membership_scope": "two canonical reference-training studies from fold 0",
        "manifest_entry_count": file_count,
        "decoded_source_file_count": decoded_slice_count,
        "loaded_study_count": len(studies),
        "selected_training_membership_sha256": _json_sha256(selected),
        "selected_file_content_sha256_manifest": _json_sha256(source_digests),
        "selected_file_digest_count": len(source_digests),
        "official_source_attestation": "competition_sources mount plus canonical manifest paths and DICOM identity checks",
        "archive_byte_equivalence_claimed": False,
        "held_out_study_count": 0,
        "held_out_images_read": False,
        "reports_used": False,
    }
    return examples, source_evidence


def _make_pretrained_model(request: KaggleGpuRequest) -> ResNet18StudyModel:
    execution = cast(Mapping[str, object], request.payload["execution"])
    weights_path = _bundle_file(request.root, execution.get("pretrained_weights"), "pretrained weights")
    approval_path = _bundle_file(
        request.root, execution.get("ticket33_resource_approval"), "Ticket 33 resource approval"
    )
    resource_approval = load_resource_approval(approval_path)
    if resource_approval.get("scope") != TICKET33_RESOURCE_SCOPE:
        raise KaggleGpuError("pretrained weights are not approved for Ticket 33")
    state = load_resnet18_weights(weights_path, approval_path)
    model = build_study_model(request.training_config.model_config())
    if not isinstance(model, ResNet18StudyModel):
        raise KaggleGpuError("Ticket 33 model builder returned an unexpected architecture")
    model.load_pretrained_state(state)
    return model


def _materialize_kaggle_frozen_features(
    request: KaggleGpuRequest,
    examples: tuple[Any, ...],
    *,
    fold_manifest: FoldManifest,
    integration_manifest: IntegrationManifest,
    source_evidence: dict[str, object],
    output_dir: Path,
) -> tuple[tuple[Any, ...], dict[str, object]]:
    """Encode the selected training studies with the approved frozen encoder."""

    if any(example.images is None or example.series_descriptors is None for example in examples):
        raise KaggleGpuError(
            "Kaggle ResNet inputs must have packed images and declared series descriptors before encoding"
        )
    config = request.training_config
    feature_identity: dict[str, object] = {
        "source_version": integration_manifest.source_version,
        "source_fingerprint": integration_manifest.archive_fingerprint,
        "integration_manifest_checksum": integration_manifest.manifest_checksum,
        "fold_manifest_checksum": fold_manifest.manifest_checksum,
        "source_membership_sha256": source_evidence["selected_training_membership_sha256"],
        "encoder_id": config.encoder_id,
        "encoder_weights_sha256": config.encoder_weights_sha256,
        "fixture_encoder_state_sha256": None,
        "preprocessing": config.preprocess_config().cache_config(),
        "packing": config.packing_config().to_dict(),
        "sampling_policy": "geometry-ordered-rounded-uniform-v1",
        "output_representation": "resnet18-adaptive-average-pool-512-float32-v1",
        "dtype": "float32",
    }
    cache = FrozenFeatureCache(
        output_dir / "frozen-feature-cache",
        feature_identity,
        feature_width=RESNET18_FEATURE_DIM,
    )
    encoder = _make_pretrained_model(request).to(torch.device(config.device))
    encoder.eval()
    feature_started = time.monotonic()
    featured_examples = _attach_cached_frozen_features(
        examples,
        model=encoder,
        cache=cache,
        device=torch.device(config.device),
    )
    del encoder
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    elapsed_seconds = time.monotonic() - feature_started
    if any(example.features is None or example.series_descriptors is None for example in featured_examples):
        raise KaggleGpuError("Kaggle ResNet feature materialization produced incomplete examples")
    evidence = {
        "identity": feature_identity,
        "identity_fingerprint": cache.identity_fingerprint,
        "read_only": cache.read_only,
        "entries": cache.hits + cache.misses,
        "cache_hits": cache.hits,
        "materialized_entries": cache.misses - cache.fallback_imports,
        "elapsed_seconds": elapsed_seconds,
        "held_out_features_materialized": False,
        "held_out_inputs_excluded": True,
        "encoder_adapted_on_held_out_data": False,
    }
    source_evidence["frozen_feature_cache"] = evidence
    return featured_examples, evidence


def _authorization(
    request: KaggleGpuRequest,
    approval: KaggleGpuApproval,
    preflight_path: Path,
    runtime: Mapping[str, object],
) -> object:
    return _new_kaggle_gpu_execution_authorization(
        request_checksum=request.request_checksum,
        approval_checksum=approval.approval_checksum,
        preflight_checksum=_sha256(preflight_path),
        approval_scope=KAGGLE_APPROVAL_SCOPE,
        kernel_id=KAGGLE_KERNEL_ID,
        runtime_fingerprint=_runtime_fingerprint(runtime),
    )


def _train_once(
    *,
    examples: tuple[Any, ...],
    request: KaggleGpuRequest,
    approval: KaggleGpuApproval,
    preflight: CpuPreflightResult,
    preflight_path: Path,
    runtime: Mapping[str, object],
    output_dir: Path,
    identity: Mapping[str, object],
    run_id: str,
    interrupt_after_steps: int | None = None,
    resume_from: Path | None = None,
) -> TrainingResult:
    model: ResNet18StudyModel | None = None
    if resume_from is None:
        seed_everything(request.training_config.seed)
        model = _make_pretrained_model(request)
    return _train_fixed_budget_gpu(
        examples,
        request.training_config,
        authorization=_authorization(request, approval, preflight_path, runtime),
        preflight=preflight,
        preflight_checksum=_sha256(preflight_path),
        preflight_path=preflight_path,
        request_path=request.request_path,
        approval_path=approval.path,
        allocation_environment={},
        resource=request.resources,
        model=model,
        artifact_dir=output_dir,
        fold=0,
        run_id=run_id,
        identity=identity,
        resume_from=resume_from,
        requested_resources=request.resources,
        interrupt_after_steps=interrupt_after_steps,
    )


def _predict_examples(model: ResNet18StudyModel, examples: tuple[Any, ...]) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for example in examples:
            images = torch.from_numpy(np.asarray(example.images, dtype=np.float32)).unsqueeze(0).to("cuda")
            slice_mask = torch.from_numpy(np.asarray(example.slice_mask, dtype=bool)).unsqueeze(0).to("cuda")
            series_mask = torch.from_numpy(np.asarray(example.series_mask, dtype=bool)).unsqueeze(0).to("cuda")
            descriptors = torch.from_numpy(
                np.asarray(example.series_descriptors, dtype=np.float32)
            ).unsqueeze(0).to("cuda")
            predictions.append(model.predict_proba(images, slice_mask, series_mask, descriptors)[0].cpu().numpy())
    return np.stack(predictions).astype(np.float32, copy=False)


def _resource_usage(
    *,
    before: Mapping[str, object],
    after_devices: list[dict[str, object]],
    elapsed_seconds: float,
    working_root: Path,
) -> dict[str, object]:
    import resource as process_resource

    usage = process_resource.getrusage(process_resource.RUSAGE_SELF)
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    after_disk = shutil.disk_usage(working_root)
    return {
        "actual_allocated_wall_seconds": elapsed_seconds,
        "campaign_charge_gpu_seconds": elapsed_seconds * KAGGLE_ALLOCATED_GPU_COUNT,
        "assigned_gpu_count": KAGGLE_ALLOCATED_GPU_COUNT,
        "visible_gpu_count": KAGGLE_VISIBLE_GPU_COUNT,
        "cpu_affinity_count": before.get("cpu_affinity_count"),
        "memory_limit_bytes": before.get("memory_limit_bytes"),
        "process_peak_rss_bytes": int(usage.ru_maxrss) * 1024,
        "process_user_cpu_seconds": usage.ru_utime,
        "process_system_cpu_seconds": usage.ru_stime,
        "working_storage_total_bytes": after_disk.total,
        "working_storage_free_before_bytes": before.get("working_storage_free_before_bytes"),
        "working_storage_free_after_bytes": after_disk.free,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "nvidia_smi_after": after_devices,
        "device_exclusivity": "not independently verifiable; no exclusive assignment claimed",
        "external_spend_usd": 0,
    }


def run_kaggle_gpu_validation(
    request_path: str | Path,
    approval_path: str | Path,
    *,
    output_dir: str | Path,
) -> Path:
    """Run the approved two-step real-input T4 smoke and resume equivalence check."""

    started = time.monotonic()
    kernel_started_raw = os.environ.get("RSNA_TICKET33_KERNEL_STARTED_EPOCH", "").strip()
    if kernel_started_raw:
        try:
            kernel_started_at = float(kernel_started_raw)
        except ValueError as exc:
            raise KaggleGpuError("notebook start timestamp is invalid") from exc
    else:
        kernel_started_at = time.time()
    if kernel_started_at > time.time() or kernel_started_at <= 0:
        raise KaggleGpuError("notebook start timestamp is outside this runtime")
    started_epoch = int(kernel_started_at)
    startup_elapsed = max(0.0, time.time() - kernel_started_at)
    request = load_kaggle_gpu_request(request_path)
    approval = load_kaggle_gpu_approval(approval_path, request)
    _load_resource_addendum(request)
    metadata = _validate_kernel_metadata(request)
    runtime, initial_devices, competition_mount = _validate_kaggle_runtime(request, metadata=metadata)
    if request.training_config.device != "cuda":
        raise KaggleGpuError("Kaggle run must retain the request's CUDA device identity")
    torch.set_num_threads(KAGGLE_MAX_CPU_COUNT)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    preflight_path = Path(output_dir).resolve() / "kaggle-cpu-preflight.json"
    output = Path(output_dir).resolve()
    if output == request.root or request.root in output.parents:
        raise KaggleGpuError("Kaggle run output must remain outside the immutable request bundle")
    output.mkdir(parents=True, exist_ok=True)
    preflight = _run_kaggle_cpu_preflight(preflight_path, runtime=runtime)
    fold_path = _bundle_file(request.root, cast(Mapping[str, object], request.payload["inputs"]).get("fold_manifest"), "fold manifest")
    integration_path = _bundle_file(request.root, cast(Mapping[str, object], request.payload["inputs"]).get("integration_manifest"), "integration manifest")
    fold_manifest = load_fold_manifest(fold_path)
    integration_manifest = load_integration_manifest(integration_path)
    inputs = cast(Mapping[str, object], request.payload["inputs"])
    if (
        fold_manifest.manifest_checksum != inputs.get("fold_manifest_checksum")
        or integration_manifest.manifest_checksum != inputs.get("integration_manifest_checksum")
        or integration_manifest.archive_fingerprint != inputs.get("source_fingerprint")
        or integration_manifest.source_version != inputs.get("source_version")
    ):
        raise KaggleGpuError("Kaggle input manifests differ from the approved request")
    examples, source_evidence = _prepare_training_examples(
        request,
        fold_manifest,
        integration_manifest,
        competition_mount,
    )
    torch.cuda.reset_peak_memory_stats(0)
    examples, feature_cache_evidence = _materialize_kaggle_frozen_features(
        request,
        examples,
        fold_manifest=fold_manifest,
        integration_manifest=integration_manifest,
        source_evidence=source_evidence,
        output_dir=output,
    )
    data_fingerprint = fingerprint_training_examples(examples)
    identity: dict[str, object] = {
        "ticket": 33,
        "purpose": "hardware-only private Kaggle T4 smoke; no candidate evidence",
        "source_version": integration_manifest.source_version,
        "source_fingerprint": integration_manifest.archive_fingerprint,
        "integration_manifest_checksum": integration_manifest.manifest_checksum,
        "fold_manifest_checksum": fold_manifest.manifest_checksum,
        "fold": 0,
        "kaggle_request_checksum": request.request_checksum,
        "data_fingerprint": data_fingerprint,
        "training_membership_sha256": source_evidence["selected_training_membership_sha256"],
        "held_out_inputs_excluded": True,
        "reports_used": False,
        "supervision": "reference-only-v1",
        "frozen_feature_cache": feature_cache_evidence,
        "approved_scientific_identity": dict(cast(Mapping[str, object], request.payload["scientific_identity"])),
    }
    run_id = _default_run_id(identity, request.training_config)
    uninterrupted_dir = output / "uninterrupted"
    interrupted_dir = output / "interrupted-resume"
    uninterrupted_started = time.monotonic()
    uninterrupted = _train_once(
        examples=examples,
        request=request,
        approval=approval,
        preflight=preflight,
        preflight_path=preflight_path,
        runtime=runtime,
        output_dir=uninterrupted_dir,
        identity=identity,
        run_id=run_id,
    )
    torch.cuda.synchronize()
    uninterrupted_elapsed = time.monotonic() - uninterrupted_started
    if uninterrupted.successful_optimizer_steps != 2 or uninterrupted.measured_optimizer_steps != 2:
        raise KaggleGpuError("uninterrupted Kaggle run did not measure exactly two optimizer steps")
    if uninterrupted.final_checkpoint_path is None:
        raise KaggleGpuError("uninterrupted Kaggle run did not save its final checkpoint")
    uninterrupted_model = cast(ResNet18StudyModel, uninterrupted.model)
    load_training_checkpoint(
        uninterrupted.final_checkpoint_path,
        model=uninterrupted_model,
        expected_config=request.training_config,
        expected_fold=0,
        expected_identity=identity,
        expected_checkpoint_kind="final_selected",
    )
    uninterrupted_output = _predict_examples(uninterrupted_model, examples)
    try:
        _train_once(
            examples=examples,
            request=request,
            approval=approval,
            preflight=preflight,
            preflight_path=preflight_path,
            runtime=runtime,
            output_dir=interrupted_dir,
            identity=identity,
            run_id=run_id,
            interrupt_after_steps=1,
        )
    except TrainingInterrupted as exc:
        if exc.checkpoint_path is None or not exc.checkpoint_path.is_file():
            raise KaggleGpuError("intentional interruption did not leave a durable checkpoint") from exc
        interruption_checkpoint = exc.checkpoint_path
    else:
        raise KaggleGpuError("intentional one-step interruption did not occur")
    resumed = _train_once(
        examples=examples,
        request=request,
        approval=approval,
        preflight=preflight,
        preflight_path=preflight_path,
        runtime=runtime,
        output_dir=interrupted_dir,
        identity=identity,
        run_id=run_id,
        resume_from=interruption_checkpoint,
    )
    torch.cuda.synchronize()
    if resumed.successful_optimizer_steps != 2 or resumed.resumed_from_checkpoint is None:
        raise KaggleGpuError("Kaggle run did not resume to the approved two-step budget")
    if resumed.final_checkpoint_path is None:
        raise KaggleGpuError("resumed Kaggle run did not save its final checkpoint")
    resumed_model = cast(ResNet18StudyModel, resumed.model)
    load_training_checkpoint(
        resumed.final_checkpoint_path,
        model=resumed_model,
        expected_config=request.training_config,
        expected_fold=0,
        expected_identity=identity,
        expected_checkpoint_kind="final_selected",
    )
    resumed_output = _predict_examples(resumed_model, examples)
    max_output_delta = float(np.max(np.abs(uninterrupted_output - resumed_output)))
    same_state = _model_state_sha256(uninterrupted_model) == _model_state_sha256(resumed_model)
    outputs_match = bool(np.allclose(uninterrupted_output, resumed_output, rtol=1e-6, atol=1e-7))
    if not same_state or not outputs_match:
        raise KaggleGpuError("interrupted/resumed model differs from uninterrupted output")
    elapsed = time.monotonic() - started + startup_elapsed
    if elapsed > request.wall_time_limit_seconds:
        raise KaggleGpuError("Kaggle validation exceeded its approved wall-time limit")
    torch.cuda.synchronize()
    after_devices = _parse_nvidia_smi()
    resource_usage = _resource_usage(
        before=runtime,
        after_devices=after_devices,
        elapsed_seconds=elapsed,
        working_root=Path("/kaggle/working"),
    )
    execution = cast(Mapping[str, object], request.payload["execution"])
    weights_path = _bundle_file(request.root, execution.get("pretrained_weights"), "pretrained weights")
    evidence = {
        "schema_version": KAGGLE_OUTPUT_SCHEMA,
        "status": "completed",
        "ticket": 33,
        "backend": "private Kaggle free notebook",
        "kernel_id": KAGGLE_KERNEL_ID,
        "request_checksum": request.request_checksum,
        "approval_sha256": approval.approval_checksum,
        "resource_addendum_sha256": _sha256(
            _bundle_file(request.root, execution.get("kaggle_resource_addendum"), "Kaggle resource addendum")
        ),
        "scientific_config_fingerprint": request.training_config.config_fingerprint,
        "scientific_identity": dict(cast(Mapping[str, object], request.payload["scientific_identity"])),
        "training_config": request.training_config.to_dict(),
        "source": source_evidence,
        "resources": request.resources,
        "runtime": runtime,
        "runtime_fingerprint": _runtime_fingerprint(runtime),
        "resource_usage": resource_usage,
        "training": {
            "successful_optimizer_steps": uninterrupted.successful_optimizer_steps,
            "optimizer_step_compute_seconds": uninterrupted.optimizer_step_compute_seconds,
            "measured_optimizer_steps": uninterrupted.measured_optimizer_steps,
            "steps_per_second": (
                uninterrupted.measured_optimizer_steps / uninterrupted.optimizer_step_compute_seconds
                if uninterrupted.optimizer_step_compute_seconds
                else None
            ),
            "training_study_examples_per_second": (
                uninterrupted.measured_optimizer_steps
                * request.training_config.batch_size
                / uninterrupted.optimizer_step_compute_seconds
                if uninterrupted.optimizer_step_compute_seconds
                else None
            ),
            "uninterrupted_wall_seconds": uninterrupted_elapsed,
            "loss_history": list(uninterrupted.loss_history),
            "sample_exposures": uninterrupted.sample_exposures,
            "supervised_entry_exposures": uninterrupted.supervised_entry_exposures,
            "initial_model_state_sha256": uninterrupted.initial_model_state_sha256,
            "gold_exposure_sequence_sha256": uninterrupted.gold_exposure_sequence_sha256,
        },
        "checkpoint": {
            "uninterrupted_final_sha256": checkpoint_sha256(uninterrupted.final_checkpoint_path),
            "interruption_latest_sha256": checkpoint_sha256(interruption_checkpoint),
            "resumed_final_sha256": checkpoint_sha256(resumed.final_checkpoint_path),
            "uninterrupted_final_reloaded": True,
            "interruption_checkpoint_reloaded": resumed.resumed_from_checkpoint is not None,
            "resumed_final_reloaded": True,
            "model_state_equal": same_state,
        },
        "output_consistency": {
            "max_absolute_score_delta": max_output_delta,
            "allclose_rtol": 1e-6,
            "allclose_atol": 1e-7,
            "predictions_equal_within_tolerance": outputs_match,
            "prediction_sha256_uninterrupted": hashlib.sha256(
                np.ascontiguousarray(uninterrupted_output).tobytes()
            ).hexdigest(),
            "prediction_sha256_resumed": hashlib.sha256(
                np.ascontiguousarray(resumed_output).tobytes()
            ).hexdigest(),
        },
        "retry": {"attempt_count": 1, "retry_allowed": False},
        "scope": {
            "hardware_and_recovery_evidence_only": True,
            "candidate_evidence": False,
            "public_release_authorized": False,
            "external_spend_usd": 0,
            "upstream_checkpoint_and_derivative_rights_resolved": False,
        },
        "started_epoch": started_epoch,
        "ended_epoch": int(time.time()),
        "elapsed_wall_seconds": elapsed,
    }
    destination = output / "kaggle-gpu-execution.json"
    atomic_json_write(destination, evidence)
    return destination


__all__ = [
    "KAGGLE_ACCELERATOR",
    "KAGGLE_APPROVAL_SCHEMA",
    "KAGGLE_COMPETITION_SLUG",
    "KAGGLE_KERNEL_ID",
    "KAGGLE_REQUEST_SCHEMA",
    "KaggleGpuApproval",
    "KaggleGpuError",
    "KaggleGpuRequest",
    "load_kaggle_gpu_approval",
    "load_kaggle_gpu_request",
    "run_kaggle_gpu_validation",
    "validate_kaggle_training_authorization",
]
