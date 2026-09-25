"""Immutable local candidate packaging for the MRI-only inference path.

The candidate seam owns the delivery contract: it validates the selected
checkpoint set, snapshots the shared inference code and pinned environments,
records the public OOF evidence, and publishes a read-only directory once.
Execution remains a separate operation and never writes inside the candidate.
This module deliberately does not import training or report-processing code.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import torch
from torch import nn

from .archive import atomic_json_write
from .dicom import DecoderPolicy
from .geometry import GeometryPolicy
from .inference import InferenceEngine
from .model import (
    HierarchicalStudyModel,
    ModelConfig,
    ModelConfigurationError,
    restore_study_model,
)
from .preprocessing import PackingConfig, PreprocessConfig
from .real_inference import (
    DEFAULT_REAL_DECODER_POLICY,
    RealInferenceError,
    load_integration_manifest,
    run_real_inference,
)
from .submission import validate_submission_csv


CANDIDATE_SCHEMA_VERSION = "rsna-knee-candidate-v1"
CANDIDATE_CONFIGURATION_SCHEMA_VERSION = "rsna-knee-candidate-inference-v1"
DEFAULT_CANDIDATE_CODE_ROOT = Path("src/rsna_knee")
DEFAULT_CLUSTER_REQUIREMENTS = Path("environments/cluster-cpu/requirements.lock")
DEFAULT_OFFLINE_REQUIREMENTS = Path("environments/offline/requirements.lock")
DEFAULT_NOTEBOOK_PATH = Path("notebooks/ticket11_offline_candidate.ipynb")
_FORBIDDEN_CODE_FILES = frozenset({"training.py", "training_cli.py"})
_REQUIRED_RUNTIME_PACKAGES = frozenset({"numpy", "pydicom", "torch"})
_NETWORK_IMPORTS = frozenset({"requests", "urllib", "urllib3", "httpx", "wget", "curl"})
_NETWORK_TEXT_TOKENS = (
    "http://",
    "https://",
    "requests.",
    "urllib.",
    "pip install",
    "conda install",
    "git clone",
    "wget ",
    "curl ",
)
_PRIVATE_METADATA_KEYS = frozenset(
    {
        "studyinstanceuid",
        "studyinstanceuids",
        "study_instance_uid",
        "study_instance_uids",
        "seriesinstanceuid",
        "seriesinstanceuids",
        "series_instance_uid",
        "series_instance_uids",
        "sopinstanceuid",
        "sopinstanceuids",
        "sop_instance_uid",
        "sop_instance_uids",
        "sourceentryname",
        "sourceentrynames",
        "source_entry_name",
        "source_entry_names",
        "heldoutstudyuid",
        "heldoutstudyuids",
        "held_out_study_uid",
        "held_out_study_uids",
        "trainingstudyuid",
        "trainingstudyuids",
        "training_study_uid",
        "training_study_uids",
    }
)


class CandidateError(ValueError):
    """Raised when a candidate cannot be safely frozen or loaded."""


class CandidateIntegrityError(CandidateError):
    """Raised when a frozen candidate or execution artifact is corrupted."""


@dataclass(frozen=True)
class CandidateBundle:
    """A loaded, checksum-verified immutable candidate directory."""

    root: Path
    candidate_id: str
    manifest_checksum: str
    manifest: Mapping[str, object]

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def model_paths(self) -> tuple[Path, ...]:
        model_payload = _mapping(self.manifest.get("model"), "model")
        members = model_payload.get("members")
        if not isinstance(members, list):
            raise CandidateIntegrityError("candidate model members are missing")
        return tuple(self.root / _nonempty(_mapping(item, "model member").get("path"), "model path") for item in members)

    @property
    def model_config(self) -> ModelConfig:
        model_payload = _mapping(self.manifest.get("model"), "model")
        return _model_config_from_dict(model_payload.get("configuration"))

    @property
    def inference_config(self) -> Mapping[str, object]:
        return _mapping(self.manifest.get("inference"), "inference")


@dataclass(frozen=True)
class CandidateExecution:
    """Result of one candidate execution and its validated submission."""

    evidence: Mapping[str, object]
    submission_validation: Mapping[str, object]
    output_path: Path


class _CandidateEnsemble(nn.Module):
    """Internal probability-averaging adapter for one or more frozen models."""

    def __init__(self, models: Sequence[HierarchicalStudyModel]) -> None:
        super().__init__()
        if not models:
            raise CandidateError("a candidate requires at least one model")
        first_config = models[0].config
        if any(model.config != first_config for model in models[1:]):
            raise CandidateError("candidate model configurations do not match")
        self.models = nn.ModuleList(models)
        self.config = first_config

    def forward(
        self,
        images: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = torch.stack(
            tuple(model(images, slice_mask, series_mask) for model in self.models),
            dim=0,
        )
        return logits.mean(dim=0)

    def predict_proba(
        self,
        images: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.stack(
            tuple(model.predict_proba(images, slice_mask, series_mask) for model in self.models),
            dim=0,
        )
        return probabilities.mean(dim=0).clamp(0.0, 1.0)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise CandidateError(f"could not read candidate input {path}") from exc
    return digest.hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CandidateIntegrityError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _nonempty(value: object, field: str) -> str:
    if value is None:
        raise CandidateIntegrityError(f"{field} must be non-empty")
    text = str(value).strip()
    if not text:
        raise CandidateIntegrityError(f"{field} must be non-empty")
    return text


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise CandidateError(f"{field} must be a positive integer")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise CandidateError(f"{field} must be a positive integer") from exc
    if parsed <= 0:
        raise CandidateError(f"{field} must be a positive integer")
    return parsed


def _safe_candidate_id(value: str) -> str:
    candidate_id = value.strip()
    if not candidate_id or candidate_id in {".", ".."} or "/" in candidate_id or "\\" in candidate_id:
        raise CandidateError("candidate_id must be a non-empty path-safe value")
    return candidate_id


def _model_config_to_dict(config: ModelConfig) -> dict[str, object]:
    return config.to_dict()


def _model_config_from_dict(value: object) -> ModelConfig:
    try:
        return ModelConfig.from_dict(value)
    except ModelConfigurationError as exc:
        raise CandidateIntegrityError("model configuration is invalid") from exc


def _preprocess_from_dict(value: object, *, packing: PackingConfig) -> PreprocessConfig:
    payload = _mapping(value, "preprocessing configuration")
    image_size = payload.get("image_size")
    percentiles = payload.get("normalization_percentiles", (0.0, 100.0))
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise CandidateIntegrityError("preprocessing image_size must contain two values")
    if not isinstance(percentiles, (list, tuple)) or len(percentiles) != 2:
        raise CandidateIntegrityError("preprocessing normalization_percentiles must contain two values")
    try:
        image_size_tuple: tuple[int, int] = (int(str(image_size[0])), int(str(image_size[1])))
        percentile_tuple: tuple[float, float] = (
            float(str(percentiles[0])),
            float(str(percentiles[1])),
        )
        geometry = GeometryPolicy.from_dict(payload.get("geometry_policy"))
        return PreprocessConfig(
            image_size=image_size_tuple,
            max_slices=packing.max_slices,
            max_series=packing.max_series,
            geometry_policy=geometry,
            normalization_domain=str(payload.get("normalization_domain", "valid_slice")),
            normalization_percentiles=percentile_tuple,
            use_voi_lut=bool(payload.get("use_voi_lut", False)),
            interpolation=str(payload.get("interpolation", "nearest")),
            padding_fill_value=float(str(payload.get("padding_fill_value", 0.0))),
            output_dtype=str(payload.get("output_dtype", "float32")),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise CandidateIntegrityError("preprocessing configuration is invalid") from exc


def _packing_from_checkpoint(payload: Mapping[str, object]) -> PackingConfig:
    config = _mapping(payload.get("config"), "checkpoint config")
    packing_payload = _mapping(config.get("packing"), "checkpoint packing")
    try:
        return PackingConfig(
            max_slices=int(str(packing_payload.get("max_slices"))),
            max_series=int(str(packing_payload.get("max_series"))),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise CandidateError("checkpoint packing configuration is invalid") from exc


def _checkpoint_contract(path: Path) -> tuple[Mapping[str, object], ModelConfig, PreprocessConfig, PackingConfig]:
    if not path.is_file() or path.is_symlink():
        raise CandidateError(f"model checkpoint is not a regular file: {path}")
    try:
        payload_value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CandidateError(f"could not load model checkpoint {path}") from exc
    payload = _mapping(payload_value, "checkpoint")
    if payload.get("checkpoint_kind") != "final_selected":
        raise CandidateError(f"candidate checkpoint must be final_selected: {path}")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise CandidateError(f"candidate checkpoint has no model_state_dict: {path}")
    config_payload = _mapping(payload.get("config"), "checkpoint config")
    model_config = _model_config_from_dict(config_payload.get("model"))
    packing = _packing_from_checkpoint(payload)
    preprocess = _preprocess_from_dict(config_payload.get("preprocessing"), packing=packing)
    if preprocess.image_size != model_config.image_size:
        raise CandidateError(f"checkpoint model and preprocessing image sizes differ: {path}")
    try:
        restore_study_model(model_config, state)
    except (ModelConfigurationError, RuntimeError, TypeError, ValueError) as exc:
        raise CandidateError(f"checkpoint model state is incompatible: {path}") from exc
    _nonempty(payload.get("config_fingerprint"), "checkpoint config_fingerprint")
    return payload, model_config, preprocess, packing


def _read_json(path: Path, description: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateError(f"could not read {description} {path}") from exc
    return _mapping(value, description)


def _private_metadata_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9_]", "", str(key).casefold())
            if normalized in _PRIVATE_METADATA_KEYS:
                found.add(str(key))
            found.update(_private_metadata_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_private_metadata_keys(child))
    return found


def _audit_public_oof(path: Path) -> tuple[Mapping[str, object], dict[str, object]]:
    payload = _read_json(path, "public OOF evidence")
    if not str(payload.get("schema_version", "")).startswith("rsna-knee-oof-evaluation-"):
        raise CandidateError("OOF evidence has an unexpected schema")
    if payload.get("fold_manifest_superseded") is not False:
        raise CandidateError("superseded OOF evidence cannot promote a candidate")
    prediction_count = _positive_int(payload.get("prediction_count"), "OOF prediction_count")
    fold_checksum = _nonempty(payload.get("fold_manifest_checksum"), "OOF fold_manifest_checksum")
    run_ids = payload.get("run_ids")
    if not isinstance(run_ids, list) or not run_ids or len(set(run_ids)) != len(run_ids):
        raise CandidateError("OOF evidence must identify one or more unique run IDs")
    private_keys = _private_metadata_keys(payload)
    if private_keys:
        raise CandidateError("public OOF evidence contains identifier-bearing fields")
    return payload, {
        "sha256": _sha256_file(path),
        "prediction_count": prediction_count,
        "fold_manifest_checksum": fold_checksum,
        "run_count": len(run_ids),
        "superseded": False,
    }


def _audit_notebook(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise CandidateError(f"notebook is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateError(f"notebook is not valid JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("nbformat") != 4:
        raise CandidateError("candidate notebook must be nbformat 4")
    text = path.read_text(encoding="utf-8").casefold()
    found = tuple(token for token in _NETWORK_TEXT_TOKENS if token in text)
    if found:
        raise CandidateError("candidate notebook contains runtime network instructions")
    return {"sha256": _sha256_file(path), "nbformat": 4, "network_instructions": False}


def _pinned_packages(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise CandidateError(f"requirements file is not a regular file: {path}")
    packages: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)$", line)
        if match is None:
            raise CandidateError(f"requirements must use exact pins: {path}: {raw_line}")
        packages[match.group(1).casefold().replace("_", "-")] = match.group(2)
    missing = _REQUIRED_RUNTIME_PACKAGES - set(packages)
    if missing:
        raise CandidateError(f"requirements are missing pinned runtime packages: {sorted(missing)}")
    return packages


def _audit_requirements(cluster_path: Path, offline_path: Path) -> dict[str, object]:
    cluster = _pinned_packages(cluster_path)
    offline = _pinned_packages(offline_path)
    for package in _REQUIRED_RUNTIME_PACKAGES:
        if cluster[package] != offline[package]:
            raise CandidateError(f"cluster and offline pin differs for {package}")
    if any(name in {"requests", "transformers", "reportlab"} for name in offline):
        raise CandidateError("offline environment contains an unapproved report/network dependency")
    return {
        "cluster": {"sha256": _sha256_file(cluster_path), "runtime_pins": {name: cluster[name] for name in sorted(_REQUIRED_RUNTIME_PACKAGES)}},
        "offline": {"sha256": _sha256_file(offline_path), "runtime_pins": {name: offline[name] for name in sorted(_REQUIRED_RUNTIME_PACKAGES)}},
        "shared_runtime_pins": True,
        "runtime_network_download": False,
    }


def _audit_code_root(code_root: Path) -> tuple[tuple[Path, ...], dict[str, object]]:
    if not code_root.is_dir() or code_root.is_symlink():
        raise CandidateError(f"candidate code root is not a directory: {code_root}")
    all_files = tuple(sorted(code_root.glob("*.py")))
    files = tuple(path for path in all_files if path.name not in _FORBIDDEN_CODE_FILES)
    if not files or not (code_root / "__init__.py").is_file():
        raise CandidateError("candidate code root must contain Python package files")
    if any(path.name in _FORBIDDEN_CODE_FILES for path in files):
        raise CandidateError("candidate code snapshot must not contain training orchestration")
    imports: set[str] = set()
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise CandidateError(f"candidate code entry is not a regular file: {path}")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise CandidateError(f"candidate code is not parseable: {path}") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
    forbidden_imports = sorted(imports & _NETWORK_IMPORTS)
    if forbidden_imports:
        raise CandidateError(f"candidate code contains network imports: {forbidden_imports}")
    return files, {
        "file_count": len(files),
        "excluded_files": sorted(path.name for path in all_files if path.name in _FORBIDDEN_CODE_FILES),
        "training_or_report_dependency": False,
        "network_imports": False,
    }


def _artifact_record(root: Path, path: Path, *, kind: str) -> dict[str, object]:
    relative = path.relative_to(root).as_posix()
    return {
        "path": relative,
        "kind": kind,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _write_candidate_json(root: Path, relative_path: str, payload: Mapping[str, object]) -> Path:
    destination = root / relative_path
    atomic_json_write(destination, payload)
    return destination


def _checkpoint_identity(payload: Mapping[str, object]) -> dict[str, str | None]:
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        identity = {}
    result: dict[str, str | None] = {}
    for field in (
        "source_version",
        "source_fingerprint",
        "fold_manifest_checksum",
        "integration_manifest_checksum",
        "data_fingerprint",
    ):
        value = identity.get(field)
        result[field] = str(value).strip() if value is not None and str(value).strip() else None
    return result


def _read_code_digest(artifacts: Sequence[Mapping[str, object]]) -> str:
    code_entries = [
        {"path": item["path"], "sha256": item["sha256"], "size_bytes": item["size_bytes"]}
        for item in artifacts
        if item.get("kind") == "shared_inference_code"
    ]
    return _json_sha256(code_entries)


def _set_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            if path.is_dir():
                path.chmod(0o555)
            else:
                path.chmod(0o444)
        except OSError as exc:
            raise CandidateError(f"could not make candidate immutable: {path}") from exc
    root.chmod(0o555)


def freeze_candidate(
    output_root: str | Path,
    *,
    model_checkpoints: Sequence[str | Path],
    oof_evidence_path: str | Path,
    code_root: str | Path = DEFAULT_CANDIDATE_CODE_ROOT,
    cluster_requirements_path: str | Path = DEFAULT_CLUSTER_REQUIREMENTS,
    offline_requirements_path: str | Path = DEFAULT_OFFLINE_REQUIREMENTS,
    notebook_path: str | Path = DEFAULT_NOTEBOOK_PATH,
    candidate_id: str = "ticket-11-local-candidate",
    code_revision: str,
    source_version: str | None = None,
    source_fingerprint: str | None = None,
    fold_manifest_checksum: str | None = None,
    oof_comparison_path: str | Path | None = None,
) -> CandidateBundle:
    """Freeze one candidate without overwriting an earlier candidate.

    All supplied artifacts are copied into a new sibling staging directory,
    checksummed, and published with one directory rename. The destination
    must not already exist. Model checkpoints are loaded and state-dict
    validated before publication.
    """

    candidate_id = _safe_candidate_id(candidate_id)
    if not code_revision.strip():
        raise CandidateError("code_revision must be non-empty")
    checkpoints = tuple(Path(path) for path in model_checkpoints)
    if not checkpoints:
        raise CandidateError("at least one model checkpoint is required")
    if len({path.resolve() for path in checkpoints}) != len(checkpoints):
        raise CandidateError("candidate checkpoints must be unique")
    code_path = Path(code_root)
    cluster_path = Path(cluster_requirements_path)
    offline_path = Path(offline_requirements_path)
    notebook = Path(notebook_path)
    oof_path = Path(oof_evidence_path)
    destination = Path(output_root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite protected candidate {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    contracts = tuple(_checkpoint_contract(path) for path in checkpoints)
    model_configs = tuple(contract[1] for contract in contracts)
    preprocess_configs = tuple(contract[2] for contract in contracts)
    packing_configs = tuple(contract[3] for contract in contracts)
    if len(set(model_configs)) != 1 or len(set(preprocess_configs)) != 1 or len(set(packing_configs)) != 1:
        raise CandidateError("candidate checkpoints do not share one inference configuration")
    _, oof_audit = _audit_public_oof(oof_path)
    requirements_audit = _audit_requirements(cluster_path, offline_path)
    code_files, code_audit = _audit_code_root(code_path)
    notebook_audit = _audit_notebook(notebook)

    checkpoint_payloads = tuple(contract[0] for contract in contracts)
    checkpoint_identities = tuple(_checkpoint_identity(payload) for payload in checkpoint_payloads)
    identity_fields = (
        "source_version",
        "source_fingerprint",
        "fold_manifest_checksum",
        "integration_manifest_checksum",
    )
    common_identity: dict[str, str | None] = {}
    for field in identity_fields:
        values = {identity[field] for identity in checkpoint_identities if identity[field] is not None}
        if len(values) > 1:
            raise CandidateError(f"candidate checkpoints disagree on {field}")
        common_identity[field] = next(iter(values), None)
    if source_version is not None:
        if common_identity["source_version"] is not None and common_identity["source_version"] != source_version:
            raise CandidateError("source_version disagrees with checkpoint provenance")
        common_identity["source_version"] = source_version
    if source_fingerprint is not None:
        if common_identity["source_fingerprint"] is not None and common_identity["source_fingerprint"] != source_fingerprint:
            raise CandidateError("source_fingerprint disagrees with checkpoint provenance")
        common_identity["source_fingerprint"] = source_fingerprint
    if fold_manifest_checksum is not None:
        if common_identity["fold_manifest_checksum"] is not None and common_identity["fold_manifest_checksum"] != fold_manifest_checksum:
            raise CandidateError("fold_manifest_checksum disagrees with checkpoint provenance")
        common_identity["fold_manifest_checksum"] = fold_manifest_checksum
    if oof_audit["fold_manifest_checksum"] != common_identity["fold_manifest_checksum"]:
        raise CandidateError("OOF evidence and checkpoint fold manifests differ")

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent))
    try:
        artifacts: list[dict[str, object]] = []
        model_members: list[dict[str, object]] = []
        seen_folds: set[int] = set()
        for index, (source, payload) in enumerate(zip(checkpoints, checkpoint_payloads)):
            raw_fold = payload.get("fold")
            if raw_fold is None:
                fold = index
            else:
                try:
                    fold = int(str(raw_fold))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise CandidateError("checkpoint fold is invalid") from exc
                if fold < 0:
                    raise CandidateError("checkpoint fold is invalid")
            if fold in seen_folds:
                raise CandidateError("candidate checkpoint folds must be unique")
            seen_folds.add(fold)
            relative = Path("models") / f"fold-{fold}.pt"
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            artifacts.append(_artifact_record(temporary, target, kind="model_checkpoint"))
            model_members.append(
                {
                    "fold": fold,
                    "path": relative.as_posix(),
                    "sha256": artifacts[-1]["sha256"],
                    "run_id": payload.get("run_id"),
                    "checkpoint_kind": payload.get("checkpoint_kind"),
                }
            )

        for source, relative, kind in (
            (oof_path, Path("evidence/oof-evidence.json"), "public_oof_evidence"),
            (cluster_path, Path("environment/cluster-requirements.lock"), "cluster_environment"),
            (offline_path, Path("environment/offline-requirements.lock"), "offline_environment"),
            (notebook, Path("notebook/ticket11_offline_candidate.ipynb"), "offline_notebook"),
        ):
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            artifacts.append(_artifact_record(temporary, target, kind=kind))
        if oof_comparison_path is not None:
            comparison = Path(oof_comparison_path)
            target = temporary / "evidence/oof-comparison.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(comparison, target)
            artifacts.append(_artifact_record(temporary, target, kind="public_oof_comparison"))

        for source in code_files:
            relative = Path("code/rsna_knee") / source.name
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            artifacts.append(_artifact_record(temporary, target, kind="shared_inference_code"))

        model_config = model_configs[0]
        preprocess_config = preprocess_configs[0]
        packing_config = packing_configs[0]
        inference_configuration: dict[str, object] = {
            "schema_version": CANDIDATE_CONFIGURATION_SCHEMA_VERSION,
            "reports_required": False,
            "runtime_network_download": False,
            "decoder": DEFAULT_REAL_DECODER_POLICY.to_dict(),
            "model": _model_config_to_dict(model_config),
            "preprocessing": preprocess_config.cache_config(),
            "packing": packing_config.to_dict(),
            "aggregation": {
                "kind": "mean_probabilities",
                "member_count": len(model_members),
            },
            "fallback": {"kind": "reference_prior", "requires_private_reference_labels": False},
        }
        configuration_path = _write_candidate_json(
            temporary,
            "configuration/inference.json",
            inference_configuration,
        )
        artifacts.append(_artifact_record(temporary, configuration_path, kind="inference_configuration"))
        code_digest = _read_code_digest(artifacts)
        provenance = {
            "code_revision": code_revision,
            "source_version": common_identity["source_version"],
            "source_fingerprint": common_identity["source_fingerprint"],
            "fold_manifest_checksum": common_identity["fold_manifest_checksum"],
            "integration_manifest_checksum": common_identity["integration_manifest_checksum"],
            "oof_evidence_sha256": oof_audit["sha256"],
            "oof_run_count": oof_audit["run_count"],
        }
        provenance_path = _write_candidate_json(temporary, "configuration/provenance.json", provenance)
        artifacts.append(_artifact_record(temporary, provenance_path, kind="provenance"))

        manifest_payload: dict[str, object] = {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "promotion": {
                "level": "locally_validated",
                "status": "locally_validated_only",
                "kaggle_verified": False,
                "owner_selection_required": True,
                "export_allowed": False,
            },
            "model": {
                "configuration": _model_config_to_dict(model_config),
                "members": model_members,
                "aggregation": "mean_probabilities",
            },
            "inference": inference_configuration,
            "environment": {
                **requirements_audit,
                "shared_code_digest": code_digest,
                "code_snapshot": "code/rsna_knee",
            },
            "provenance": provenance,
            "validation": {
                "public_oof": oof_audit,
                "private_metadata_audit": {
                    "candidate_manifest_contains_identifiers": False,
                    "notebook_contains_identifiers": False,
                    "oof_contains_identifiers": False,
                },
                "code": code_audit,
                "notebook": notebook_audit,
                "exact_submission_schema": True,
                "scores_finite_and_bounded": True,
            },
            "artifacts": sorted(artifacts, key=lambda item: str(item["path"])),
        }
        manifest_payload["manifest_checksum"] = _json_sha256(manifest_payload)
        manifest_path = _write_candidate_json(temporary, "manifest.json", manifest_payload)
        _set_read_only(temporary)
        try:
            os.replace(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite protected candidate {destination}") from exc
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_candidate(destination)


def _verify_manifest(root: Path, payload: Mapping[str, object], stored_checksum: str) -> None:
    without_checksum = dict(payload)
    without_checksum.pop("manifest_checksum", None)
    if _json_sha256(without_checksum) != stored_checksum:
        raise CandidateIntegrityError("candidate manifest checksum does not match its contents")
    if payload.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise CandidateIntegrityError("candidate schema is unsupported")
    if payload.get("promotion") is None or payload.get("inference") is None:
        raise CandidateIntegrityError("candidate promotion or inference metadata is missing")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise CandidateIntegrityError("candidate artifact manifest is empty")
    paths: set[str] = set()
    for item_value in artifacts:
        item = _mapping(item_value, "candidate artifact")
        relative = _nonempty(item.get("path"), "candidate artifact path")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in paths:
            raise CandidateIntegrityError(f"candidate artifact path is unsafe or duplicated: {relative}")
        paths.add(relative)
        artifact_path = root / path
        if not artifact_path.is_file() or artifact_path.is_symlink():
            raise CandidateIntegrityError(f"candidate artifact is missing: {relative}")
        try:
            expected_size = int(str(item.get("size_bytes")))
        except (TypeError, ValueError, OverflowError) as exc:
            raise CandidateIntegrityError(f"candidate artifact size is invalid: {relative}") from exc
        if artifact_path.stat().st_size != expected_size:
            raise CandidateIntegrityError(f"candidate artifact size changed: {relative}")
        if _sha256_file(artifact_path) != _nonempty(item.get("sha256"), "candidate artifact sha256"):
            raise CandidateIntegrityError(f"candidate artifact checksum changed: {relative}")
    model_payload = _mapping(payload.get("model"), "model")
    members = model_payload.get("members")
    if not isinstance(members, list) or not members:
        raise CandidateIntegrityError("candidate has no model members")
    for member_value in members:
        member = _mapping(member_value, "model member")
        member_path = _nonempty(member.get("path"), "model member path")
        if member_path not in paths or _nonempty(member.get("checkpoint_kind"), "checkpoint kind") != "final_selected":
            raise CandidateIntegrityError("candidate model member is not represented by a final checkpoint artifact")
    if "training.py" in paths or "training_cli.py" in paths:
        raise CandidateIntegrityError("candidate contains training orchestration")


def load_candidate(root: str | Path) -> CandidateBundle:
    """Load and verify every checksum in one frozen candidate."""

    candidate_root = Path(root)
    manifest_path = candidate_root / "manifest.json"
    if not candidate_root.is_dir() or candidate_root.is_symlink():
        raise CandidateIntegrityError(f"candidate root is not a directory: {candidate_root}")
    payload = _read_json(manifest_path, "candidate manifest")
    stored_checksum = _nonempty(payload.get("manifest_checksum"), "manifest_checksum")
    _verify_manifest(candidate_root, payload, stored_checksum)
    model_payload = _mapping(payload.get("model"), "model")
    model_config = _model_config_from_dict(model_payload.get("configuration"))
    inference_config = _mapping(payload.get("inference"), "inference")
    inference_model_config = _model_config_from_dict(inference_config.get("model"))
    if model_config != inference_model_config:
        raise CandidateIntegrityError("candidate model and inference configurations disagree")
    candidate_id = _nonempty(payload.get("candidate_id"), "candidate_id")
    return CandidateBundle(
        root=candidate_root.resolve(),
        candidate_id=candidate_id,
        manifest_checksum=stored_checksum,
        manifest=payload,
    )


def _load_candidate_models(bundle: CandidateBundle, device: str | torch.device) -> _CandidateEnsemble:
    models: list[HierarchicalStudyModel] = []
    for path in bundle.model_paths:
        payload, model_config, preprocess, packing = _checkpoint_contract(path)
        if model_config != bundle.model_config:
            raise CandidateIntegrityError("candidate checkpoint model configuration differs from manifest")
        inference = bundle.inference_config
        if _preprocess_from_dict(inference.get("preprocessing"), packing=packing) != preprocess:
            raise CandidateIntegrityError("candidate checkpoint preprocessing differs from manifest")
        try:
            state = cast(Mapping[str, torch.Tensor], payload["model_state_dict"])
            model = restore_study_model(model_config, state)
        except (ModelConfigurationError, RuntimeError, TypeError, ValueError) as exc:
            raise CandidateIntegrityError(f"could not restore candidate model {path}") from exc
        model.to(device)
        model.eval()
        models.append(model)
    ensemble = _CandidateEnsemble(models)
    ensemble.to(device)
    ensemble.eval()
    return ensemble


def _candidate_output_paths(
    bundle: CandidateBundle,
    output_path: Path,
    evidence_path: Path | None,
    private_evidence_path: Path | None,
) -> None:
    destinations = [output_path]
    if evidence_path is not None:
        destinations.append(evidence_path)
    if private_evidence_path is not None:
        destinations.append(private_evidence_path)
    if len(set(destinations)) != len(destinations):
        raise CandidateError("candidate execution destinations must be distinct")
    for destination in destinations:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite protected candidate execution artifact {destination}")
        try:
            destination.resolve().relative_to(bundle.root)
        except ValueError:
            continue
        raise CandidateError("candidate execution outputs must remain outside the frozen candidate")


def _validate_input_manifest(bundle: CandidateBundle, manifest_path: Path) -> tuple[str, ...]:
    """Require the execution manifest to use the frozen inference policies."""

    try:
        manifest = load_integration_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        raise CandidateIntegrityError("candidate input manifest could not be loaded") from exc
    inference = bundle.inference_config
    try:
        packing_payload = _mapping(inference.get("packing"), "candidate packing")
        packing = PackingConfig(
            max_slices=int(str(packing_payload.get("max_slices"))),
            max_series=int(str(packing_payload.get("max_series"))),
        )
        preprocess = _preprocess_from_dict(inference.get("preprocessing"), packing=packing)
        decoder = DecoderPolicy.from_dict(inference.get("decoder"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise CandidateIntegrityError("candidate inference policies are invalid") from exc
    if manifest.preprocess_config != preprocess:
        raise CandidateIntegrityError("input manifest preprocessing differs from frozen candidate")
    if manifest.packing_config != packing:
        raise CandidateIntegrityError("input manifest packing differs from frozen candidate")
    if manifest.decoder_policy != decoder:
        raise CandidateIntegrityError("input manifest decoder policy differs from frozen candidate")
    return manifest.study_uids


def run_candidate_inference(
    candidate_root: str | Path,
    manifest_path: str | Path,
    source_path: str | Path,
    output_path: str | Path,
    *,
    evidence_path: str | Path | None = None,
    private_evidence_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    device: str | torch.device = "cpu",
    recover_damaged_inputs: bool = False,
    require_slurm: bool = False,
    allow_local_smoke: bool = False,
) -> CandidateExecution:
    """Run one frozen candidate and publish only externally supplied outputs."""

    bundle = load_candidate(candidate_root)
    destination = Path(output_path)
    public_destination = Path(evidence_path) if evidence_path is not None else None
    private_destination = Path(private_evidence_path) if private_evidence_path is not None else None
    _candidate_output_paths(bundle, destination, public_destination, private_destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_uids = _validate_input_manifest(bundle, Path(manifest_path))
    started = time.perf_counter()
    model = _load_candidate_models(bundle, device)
    model_load_seconds = time.perf_counter() - started
    with tempfile.TemporaryDirectory(prefix="rsna-ticket11-execution-", dir=destination.parent) as temporary_name:
        temporary = Path(temporary_name)
        temporary_output = temporary / "submission.csv"
        temporary_private = temporary / "private-evidence.json"
        core_evidence = run_real_inference(
            manifest_path,
            source_path,
            temporary_output,
            private_evidence_path=temporary_private,
            model=cast(Any, model),
            device=device,
            cache_root=cache_root,
            evidence_ticket="11",
            recover_damaged_inputs=recover_damaged_inputs,
            require_slurm=require_slurm,
            allow_local_smoke=allow_local_smoke,
        )
        # The integration manifest stores studies as objects. Keep this read
        # local and identifier-bearing; the private execution report is the
        # only place where its row-level diagnostics may be retained.
        submission_validation = validate_submission_csv(temporary_output, expected_uids)
        enriched: dict[str, object] = dict(core_evidence)
        enriched["candidate"] = {
            "candidate_id": bundle.candidate_id,
            "candidate_manifest_checksum": bundle.manifest_checksum,
            "model_member_count": len(bundle.model_paths),
            "code_revision": _mapping(bundle.manifest.get("provenance"), "provenance").get("code_revision"),
        }
        enriched["submission_validation"] = submission_validation
        enriched["candidate_execution"] = {
            "model_load_seconds": model_load_seconds,
            "whole_pipeline_seconds": time.perf_counter() - started,
            "runtime_network_download": False,
            "reports_required": False,
        }
        if public_destination is not None:
            atomic_json_write(public_destination, enriched)
        if private_destination is not None:
            private_payload = _read_json(temporary_private, "private candidate execution evidence")
            private_payload = dict(private_payload)
            private_payload["candidate"] = {
                "candidate_id": bundle.candidate_id,
                "candidate_manifest_checksum": bundle.manifest_checksum,
            }
            atomic_json_write(private_destination, private_payload)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(temporary_output, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite protected candidate output {destination}") from exc
    return CandidateExecution(evidence=enriched, submission_validation=submission_validation, output_path=destination)


def benchmark_candidate_execution(
    candidate_root: str | Path,
    manifest_path: str | Path,
    source_path: str | Path,
    output_path: str | Path,
    *,
    evidence_path: str | Path,
    private_evidence_path: str | Path | None = None,
    expected_study_count: int = 1300,
    long_tail_studies: int | None = None,
    cache_root: str | Path | None = None,
    device: str | torch.device = "cpu",
    recover_damaged_inputs: bool = True,
    require_slurm: bool = False,
    allow_local_smoke: bool = False,
) -> CandidateExecution:
    """Execute and record measured-versus-projected whole-pipeline cost."""

    if expected_study_count <= 0:
        raise CandidateError("expected_study_count must be positive")
    if long_tail_studies is not None and (long_tail_studies < 0 or long_tail_studies > expected_study_count):
        raise CandidateError("long_tail_studies must be within the expected workload")
    execution = run_candidate_inference(
        candidate_root,
        manifest_path,
        source_path,
        output_path,
        evidence_path=evidence_path,
        private_evidence_path=private_evidence_path,
        cache_root=cache_root,
        device=device,
        recover_damaged_inputs=recover_damaged_inputs,
        require_slurm=require_slurm,
        allow_local_smoke=allow_local_smoke,
    )
    evidence = dict(execution.evidence)
    counts = _mapping(evidence.get("counts"), "candidate execution counts")
    measured_studies = _positive_int(counts.get("input_studies"), "measured input studies")
    execution_payload = _mapping(evidence.get("candidate_execution"), "candidate execution")
    raw_whole_pipeline_seconds = execution_payload.get("whole_pipeline_seconds")
    if not isinstance(raw_whole_pipeline_seconds, (int, float)):
        raise CandidateIntegrityError("candidate execution time is missing")
    whole_pipeline_seconds = float(raw_whole_pipeline_seconds)
    projected_seconds = whole_pipeline_seconds * expected_study_count / measured_studies
    benchmark = {
        "measured_scope": {
            "studies": measured_studies,
            "long_tail_studies": long_tail_studies,
            "scope_is_representative_only": True,
        },
        "target_scope": {
            "studies": expected_study_count,
            "long_tail_included": long_tail_studies is not None,
        },
        "whole_pipeline_seconds_observed": whole_pipeline_seconds,
        "projected_seconds_at_target": projected_seconds,
        "engineering_target_seconds": 25_920,
        "hard_limit_seconds": 32_400,
        "within_engineering_target": projected_seconds <= 25_920,
        "within_hard_limit_projection": projected_seconds <= 32_400,
        "hidden_execution_unknown": True,
        "full_scale_compliance_claim": False,
        "peak_vram_bytes": None,
        "model_and_bundle_bytes": sum(path.stat().st_size for path in load_candidate(candidate_root).root.rglob("*") if path.is_file()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    evidence["runtime_benchmark"] = benchmark
    public_path = Path(evidence_path)
    if public_path.exists():
        # This output was just written by run_candidate_inference. Re-publish
        # the enriched view only at this fresh, caller-owned destination.
        public_path.unlink()
    atomic_json_write(public_path, evidence)
    return CandidateExecution(evidence=evidence, submission_validation=execution.submission_validation, output_path=execution.output_path)


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(usage * (1024 if sys.platform != "darwin" else 1))
    except (ImportError, OSError, ValueError):
        return None


__all__ = [
    "CANDIDATE_CONFIGURATION_SCHEMA_VERSION",
    "CANDIDATE_SCHEMA_VERSION",
    "CandidateBundle",
    "CandidateError",
    "CandidateExecution",
    "CandidateIntegrityError",
    "benchmark_candidate_execution",
    "freeze_candidate",
    "load_candidate",
    "run_candidate_inference",
]
