"""Prepare and record the approval-gated Kaggle delivery seam.

This module deliberately does not upload notebooks, models, or submissions.
It validates one immutable local candidate, creates an exact execution request,
and records externally supplied Kaggle evidence in a new immutable artifact.
The candidate itself is never edited when verification or promotion evidence is
written.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import pydicom
import torch

from .archive import atomic_json_write
from .candidate import CandidateBundle, CandidateError, _load_candidate_models, load_candidate
from .dicom import (
    DecoderInfrastructureError,
    DecoderPolicy,
    DicomDecodeError,
    decode_stream,
)
from .domain import SeriesRecord, StudyRecord
from .geometry import GeometryPolicy
from .inference import InferenceEngine, StudyPrediction
from .preprocessing import PackingConfig, PreprocessConfig, SeriesFailure, pack_study, prepare_study
from .submission import SubmissionError, validate_submission_csv, write_submission
from .targets import NUM_TARGETS, SUBMISSION_COLUMNS


DEFAULT_KAGGLE_ADAPTER_PATH = Path("src/rsna_knee/kaggle.py")


KAGGLE_DELIVERY_SCHEMA_VERSION = "rsna-knee-kaggle-delivery-v1"
KAGGLE_APPROVAL_SCHEMA_VERSION = "rsna-knee-kaggle-owner-approval-v1"
KAGGLE_EXECUTION_SCHEMA_VERSION = "rsna-knee-kaggle-execution-v1"
KAGGLE_VERIFICATION_SCHEMA_VERSION = "rsna-knee-kaggle-verification-v1"
KAGGLE_PROMOTION_SCHEMA_VERSION = "rsna-knee-kaggle-promotion-v1"
KAGGLE_ENGINEERING_TARGET_SECONDS = 25_920
KAGGLE_HARD_LIMIT_SECONDS = 32_400
KAGGLE_TARGET_STUDIES = 1_300
KAGGLE_RUNTIME_STAGES = (
    "startup_setup",
    "imports",
    "model_metadata_loading",
    "dicom_read_decode",
    "preprocessing",
    "inference_ensembling",
    "csv_creation",
)
OWNER_APPROVAL_SCOPE = "ticket-12-exact-bundle-upload-and-execution"

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
_NETWORK_IMPORTS = frozenset({"requests", "urllib", "urllib3", "httpx", "wget", "curl", "socket"})
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
    }
)


class KaggleDeliveryError(ValueError):
    """Raised when a Kaggle request, execution, or promotion is unsafe."""


@dataclass(frozen=True)
class KaggleDeliveryRequest:
    """A checksum-verified owner-approval request for one exact bundle."""

    root: Path
    request_checksum: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class KaggleVerification:
    """A checksum-verified record of one accepted Kaggle execution."""

    root: Path
    verification_checksum: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class KagglePromotion:
    """A checksum-verified promotion pointer, separate from the candidate."""

    root: Path
    promotion_checksum: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class KaggleInferenceResult:
    """The offline notebook adapter's output and identifier-free evidence."""

    output_path: Path
    evidence: Mapping[str, object]
    private_evidence_path: Path | None = None


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise KaggleDeliveryError(f"artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise KaggleDeliveryError(f"could not read artifact {path}") from exc
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise KaggleDeliveryError(f"could not read {description} {path}") from exc
    if not isinstance(value, Mapping):
        raise KaggleDeliveryError(f"{description} must be a JSON object")
    return cast(Mapping[str, object], value)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise KaggleDeliveryError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _nonempty(value: object, field: str) -> str:
    if value is None:
        raise KaggleDeliveryError(f"{field} must be non-empty")
    text = str(value).strip()
    if not text:
        raise KaggleDeliveryError(f"{field} must be non-empty")
    return text


def _safe_identifier(value: str, field: str) -> str:
    text = _nonempty(value, field)
    if text in {".", ".."} or "/" in text or "\\" in text:
        raise KaggleDeliveryError(f"{field} must be path-safe")
    return text


def _required_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise KaggleDeliveryError(f"{field} must be boolean")
    return value


def _required_int(value: object, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise KaggleDeliveryError(f"{field} must be an integer")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise KaggleDeliveryError(f"{field} must be an integer") from exc
    if parsed < (1 if positive else 0):
        qualifier = "positive " if positive else "non-negative "
        raise KaggleDeliveryError(f"{field} must be a {qualifier}integer")
    return parsed


def _required_float(value: object, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise KaggleDeliveryError(f"{field} must be a number")
    try:
        parsed = float(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise KaggleDeliveryError(f"{field} must be a number") from exc
    if not math.isfinite(parsed) or (positive and parsed <= 0.0) or (not positive and parsed < 0.0):
        raise KaggleDeliveryError(f"{field} must be a finite {'positive ' if positive else 'non-negative '}number")
    return parsed


def _optional_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _required_int(value, field)


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


def _set_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            if path.is_dir():
                path.chmod(0o555)
            else:
                path.chmod(0o444)
        except OSError as exc:
            raise KaggleDeliveryError(f"could not make artifact immutable: {path}") from exc
    root.chmod(0o555)


def _publish_directory(output_root: Path, writer: Callable[[Path], None]) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite protected Kaggle artifact {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.partial-", dir=output_root.parent))
    try:
        writer(staging)
        _set_read_only(staging)
        try:
            os.rename(staging, output_root)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite protected Kaggle artifact {output_root}") from exc
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _ensure_outside(path: Path, protected: Sequence[Path], field: str) -> None:
    resolved = path.resolve()
    for item in protected:
        protected_resolved = item.resolve()
        try:
            resolved.relative_to(protected_resolved)
        except ValueError:
            try:
                protected_resolved.relative_to(resolved)
            except ValueError:
                continue
        raise KaggleDeliveryError(f"{field} must remain separate from protected artifact {item}")


def _notebook_audit(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise KaggleDeliveryError(f"delivery notebook is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise KaggleDeliveryError(f"delivery notebook is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping) or payload.get("nbformat") != 4:
        raise KaggleDeliveryError("delivery notebook must use nbformat 4")
    text = path.read_text(encoding="utf-8").casefold()
    if "candidate" not in text or "submission.csv" not in text:
        raise KaggleDeliveryError("delivery notebook must name the candidate and submission.csv")
    found = tuple(token for token in _NETWORK_TEXT_TOKENS if token in text)
    if found:
        raise KaggleDeliveryError("delivery notebook contains runtime network instructions")
    private_keys = _private_metadata_keys(payload)
    if private_keys:
        raise KaggleDeliveryError(
            "delivery notebook contains identifier-bearing fields: " + ", ".join(sorted(private_keys))
        )
    return {
        "filename": path.name,
        "sha256": _sha256_file(path),
        "nbformat": 4,
        "network_instructions": False,
        "private_metadata_keys": [],
    }


def _adapter_audit(path: Path) -> dict[str, object]:
    """Audit the path/orchestration adapter shipped beside the candidate."""

    if not path.is_file() or path.is_symlink():
        raise KaggleDeliveryError(f"delivery adapter is not a regular file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise KaggleDeliveryError(f"could not read delivery adapter {path}") from exc
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        raise KaggleDeliveryError("delivery adapter is not parseable Python") from exc
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    forbidden = sorted(imports & _NETWORK_IMPORTS)
    if forbidden:
        raise KaggleDeliveryError(f"delivery adapter contains network imports: {forbidden}")
    if "run_kaggle_baseline" not in text or "submission.csv" not in text:
        raise KaggleDeliveryError("delivery adapter does not expose the Kaggle inference path")
    return {
        "filename": path.name,
        "sha256": _sha256_file(path),
        "network_instructions": False,
    }


def _artifact_records(bundle: CandidateBundle) -> dict[str, Mapping[str, object]]:
    records_value = bundle.manifest.get("artifacts")
    if not isinstance(records_value, list):
        raise KaggleDeliveryError("candidate artifact records are missing")
    records: dict[str, Mapping[str, object]] = {}
    for item_value in records_value:
        item = _mapping(item_value, "candidate artifact")
        path = _nonempty(item.get("path"), "candidate artifact path")
        records[path] = item
    return records


def _candidate_reference(bundle: CandidateBundle) -> dict[str, object]:
    records = _artifact_records(bundle)
    members_value = _mapping(bundle.manifest.get("model"), "candidate model").get("members")
    if not isinstance(members_value, list) or not members_value:
        raise KaggleDeliveryError("candidate has no model checkpoints")
    model_checkpoints: list[dict[str, object]] = []
    model_size_bytes = 0
    for item_value in members_value:
        item = _mapping(item_value, "candidate model member")
        path = _nonempty(item.get("path"), "candidate model path")
        artifact = records.get(path)
        if artifact is None or artifact.get("kind") != "model_checkpoint":
            raise KaggleDeliveryError(f"candidate model is not checksum-covered: {path}")
        actual_sha = _sha256_file(bundle.root / path)
        expected_sha = _nonempty(artifact.get("sha256"), f"candidate model checksum {path}")
        if actual_sha != expected_sha or str(item.get("sha256", expected_sha)) != expected_sha:
            raise KaggleDeliveryError(f"candidate model checksum is inconsistent: {path}")
        model_size_bytes += _required_int(artifact.get("size_bytes"), f"candidate model size {path}", positive=True)
        model_checkpoints.append(
            {
                "fold": _required_int(item.get("fold"), f"candidate model fold {path}"),
                "path": path,
                "sha256": expected_sha,
                "checkpoint_kind": _nonempty(item.get("checkpoint_kind"), f"candidate checkpoint kind {path}"),
            }
        )
    promotion = _mapping(bundle.manifest.get("promotion"), "candidate promotion")
    if promotion.get("status") != "locally_validated_only" or promotion.get("kaggle_verified") is not False:
        raise KaggleDeliveryError("Kaggle preparation requires a locally validated, not Kaggle-verified candidate")
    provenance = _mapping(bundle.manifest.get("provenance"), "candidate provenance")
    environment = _mapping(bundle.manifest.get("environment"), "candidate environment")
    cluster_environment = _mapping(environment.get("cluster"), "candidate cluster environment")
    offline_environment = _mapping(environment.get("offline"), "candidate offline environment")
    if environment.get("runtime_network_download") is not False:
        raise KaggleDeliveryError("candidate environment permits runtime downloads")
    cluster_pins = _mapping(cluster_environment.get("runtime_pins"), "candidate cluster runtime pins")
    offline_pins = _mapping(offline_environment.get("runtime_pins"), "candidate offline runtime pins")
    if dict(cluster_pins) != dict(offline_pins):
        raise KaggleDeliveryError("candidate cluster and offline runtime pins differ")
    code_revision = _nonempty(provenance.get("code_revision"), "candidate code revision")
    code_digest = _nonempty(environment.get("shared_code_digest"), "candidate shared code digest")
    return {
        "candidate_id": bundle.candidate_id,
        "manifest_checksum": bundle.manifest_checksum,
        "code_revision": code_revision,
        "shared_code_digest": code_digest,
        "inference_configuration_sha256": _json_sha256(bundle.inference_config),
        "model_checkpoints": model_checkpoints,
        "model_size_bytes": model_size_bytes,
        "candidate_bundle_size_bytes": sum(
            _required_int(item.get("size_bytes"), f"candidate artifact size {path}", positive=True)
            for path, item in records.items()
        ),
        "environment": {
            "cluster_requirements_sha256": _nonempty(
                cluster_environment.get("sha256"), "candidate cluster requirements sha256"
            ),
            "offline_requirements_sha256": _nonempty(
                offline_environment.get("sha256"), "candidate offline requirements sha256"
            ),
            "runtime_pins": dict(offline_pins),
            "runtime_network_download": False,
        },
    }


def _local_submission_audit(path: Path, expected_study_uids: Sequence[str] | None) -> dict[str, object]:
    if expected_study_uids is not None:
        try:
            validation = validate_submission_csv(path, expected_study_uids)
        except (OSError, SubmissionError) as exc:
            raise KaggleDeliveryError(f"local submission failed exact validation: {path}") from exc
        return {**validation, "sha256": _sha256_file(path), "available": True}
    if not path.is_file() or path.is_symlink():
        raise KaggleDeliveryError(f"local submission is not a regular file: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise KaggleDeliveryError(f"could not read local submission {path}") from exc
    if not rows or rows[0] != list(SUBMISSION_COLUMNS):
        raise KaggleDeliveryError("local submission header does not match the exact target order")
    observed_uids: list[str] = []
    for row in rows[1:]:
        if len(row) != NUM_TARGETS + 1 or not row[0]:
            raise KaggleDeliveryError("local submission contains an invalid row")
        observed_uids.append(row[0])
        try:
            scores = [float(value) for value in row[1:]]
        except (TypeError, ValueError, OverflowError) as exc:
            raise KaggleDeliveryError("local submission contains a non-numeric score") from exc
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in scores):
            raise KaggleDeliveryError("local submission contains a non-finite or unbounded score")
    if len(set(observed_uids)) != len(observed_uids):
        raise KaggleDeliveryError("local submission contains duplicate study rows")
    return {
        "row_count": len(rows) - 1,
        "column_count": len(rows[0]),
        "exact_header": True,
        "exact_uid_order": False,
        "scores_finite": True,
        "scores_bounded": True,
        "sha256": _sha256_file(path),
        "available": True,
    }


def _local_evidence_audit(
    path: Path,
    candidate: Mapping[str, object],
    *,
    local_submission_path: Path | None,
    expected_study_uids: Sequence[str] | None,
) -> dict[str, object]:
    evidence = _read_json(path, "local candidate evidence")
    schema = str(evidence.get("schema_version", ""))
    if not schema.startswith("rsna-knee-real-integration-evidence-"):
        raise KaggleDeliveryError("local evidence has an unexpected schema")
    if evidence.get("status") != "completed_real_integration":
        raise KaggleDeliveryError("local evidence is not a completed real integration")
    if _private_metadata_keys(evidence):
        raise KaggleDeliveryError("local evidence contains identifier-bearing fields")
    evidence_candidate = _mapping(evidence.get("candidate"), "local evidence candidate")
    if evidence_candidate.get("candidate_manifest_checksum") != candidate.get("manifest_checksum"):
        raise KaggleDeliveryError("local evidence candidate manifest checksum does not match the candidate")

    counts = _mapping(evidence.get("counts"), "local evidence counts")
    input_studies = _required_int(counts.get("input_studies"), "local evidence input studies", positive=True)
    output_rows = _required_int(counts.get("output_rows"), "local evidence output rows", positive=True)
    if output_rows != input_studies:
        raise KaggleDeliveryError("local evidence output rows do not cover every measured study")
    if _required_int(counts.get("output_columns"), "local evidence output columns") != NUM_TARGETS + 1:
        raise KaggleDeliveryError("local evidence output does not have the thirteen-column contract")
    if counts.get("scores_finite") is not True or counts.get("scores_bounded") is not True:
        raise KaggleDeliveryError("local evidence does not prove finite bounded scores")
    submission_validation = _mapping(evidence.get("submission_validation"), "local submission validation")
    for field in ("exact_header", "exact_uid_order", "scores_finite", "scores_bounded"):
        if submission_validation.get(field) is not True:
            raise KaggleDeliveryError(f"local evidence does not prove submission {field}")
    if _required_int(submission_validation.get("row_count"), "local submission validation row count", positive=True) != input_studies:
        raise KaggleDeliveryError("local submission validation row count differs from local workload")
    if _required_int(submission_validation.get("column_count"), "local submission validation column count") != NUM_TARGETS + 1:
        raise KaggleDeliveryError("local submission validation column count is not thirteen")

    fallback_numerator = _required_int(
        counts.get("full_study_fallback_numerator"),
        "local fallback numerator",
    )
    fallback_denominator = _required_int(
        counts.get("full_study_fallback_denominator"),
        "local fallback denominator",
        positive=True,
    )
    if fallback_denominator != input_studies or fallback_numerator > fallback_denominator:
        raise KaggleDeliveryError("local fallback denominator does not cover the measured workload")
    fallback_rate = _required_float(counts.get("full_study_fallback_rate"), "local fallback rate")
    if not math.isclose(fallback_rate, fallback_numerator / fallback_denominator, rel_tol=1e-9, abs_tol=1e-12):
        raise KaggleDeliveryError("local fallback rate is inconsistent with its numerator and denominator")
    recovery = _mapping(evidence.get("recovery"), "local recovery evidence")
    recovery_fallback = _mapping(recovery.get("full_study_fallback"), "local recovery fallback")
    if (
        _required_int(recovery_fallback.get("numerator"), "recovery fallback numerator") != fallback_numerator
        or _required_int(recovery_fallback.get("denominator"), "recovery fallback denominator", positive=True)
        != fallback_denominator
    ):
        raise KaggleDeliveryError("local recovery fallback counts disagree with execution counts")
    if fallback_numerator:
        investigation = _mapping(recovery.get("fallback_investigation"), "local fallback investigation")
        if investigation.get("complete") is not True:
            raise KaggleDeliveryError("every local fallback case must be investigated before delivery")

    benchmark = _mapping(evidence.get("runtime_benchmark"), "local runtime benchmark")
    observed = _required_float(
        benchmark.get("whole_pipeline_seconds_observed"),
        "local observed whole-pipeline runtime",
        positive=True,
    )
    projected = _required_float(
        benchmark.get("projected_seconds_at_target"),
        "local projected runtime",
        positive=True,
    )
    if benchmark.get("engineering_target_seconds") != KAGGLE_ENGINEERING_TARGET_SECONDS:
        raise KaggleDeliveryError("local benchmark uses a different engineering target")
    if benchmark.get("hard_limit_seconds") != KAGGLE_HARD_LIMIT_SECONDS:
        raise KaggleDeliveryError("local benchmark uses a different hard runtime limit")
    if benchmark.get("within_engineering_target") is not (projected <= KAGGLE_ENGINEERING_TARGET_SECONDS):
        raise KaggleDeliveryError("local engineering-target result is inconsistent with its projection")
    if benchmark.get("within_hard_limit_projection") is not (projected <= KAGGLE_HARD_LIMIT_SECONDS):
        raise KaggleDeliveryError("local hard-limit result is inconsistent with its projection")
    if benchmark.get("hidden_execution_unknown") is not True:
        raise KaggleDeliveryError("local evidence must keep hidden execution explicitly unknown")
    if benchmark.get("full_scale_compliance_claim") is not False:
        raise KaggleDeliveryError("local evidence must not claim hidden-scale compliance")
    execution = _mapping(evidence.get("candidate_execution"), "local candidate execution")
    if execution.get("runtime_network_download") is not False or execution.get("reports_required") is not False:
        raise KaggleDeliveryError("local candidate execution is not offline and report-free")

    model_and_bundle_bytes = _optional_int(
        benchmark.get("model_and_bundle_bytes"),
        "local model and bundle size",
    )
    peak_vram_bytes = _optional_int(benchmark.get("peak_vram_bytes"), "local peak VRAM")

    local_submission: dict[str, object] = {"available": False}
    if local_submission_path is not None:
        local_submission = _local_submission_audit(local_submission_path, expected_study_uids)
        if local_submission["row_count"] != input_studies:
            raise KaggleDeliveryError("local submission row count differs from local evidence")
    return {
        "evidence_schema_version": schema,
        "evidence_sha256": _sha256_file(path),
        "measured_studies": input_studies,
        "measured_runtime_seconds": observed,
        "projected_runtime_seconds_at_1300": projected,
        "fallback": {
            "status": "measured",
            "numerator": fallback_numerator,
            "denominator": fallback_denominator,
            "rate": fallback_rate,
            "hidden_failure_rate_unknown": True,
        },
        "submission": {
            "row_count": input_studies,
            "column_count": NUM_TARGETS + 1,
            "exact_header": True,
            "exact_uid_order": True,
            "scores_finite": True,
            "scores_bounded": True,
            "local_artifact": local_submission,
        },
        "runtime": {
            "engineering_target_seconds": KAGGLE_ENGINEERING_TARGET_SECONDS,
            "hard_limit_seconds": KAGGLE_HARD_LIMIT_SECONDS,
            "within_engineering_target_projection": projected <= KAGGLE_ENGINEERING_TARGET_SECONDS,
            "within_hard_limit_projection": projected <= KAGGLE_HARD_LIMIT_SECONDS,
            "hidden_execution_unknown": True,
            "model_and_bundle_bytes": model_and_bundle_bytes,
            "peak_vram_bytes": peak_vram_bytes,
        },
        "recovery": {
            "save_reload_verified": recovery.get("save_reload_verified") is True,
            "interruption_recovery_verified": recovery.get("interruption_recovery_verified") is True,
            "rollback_seconds": _optional_int(recovery.get("rollback_seconds"), "local recovery rollback seconds"),
            "evidence_sha256": _sha256_file(path),
        },
    }


def _request_payload(root: Path) -> Mapping[str, object]:
    if not root.is_dir() or root.is_symlink():
        raise KaggleDeliveryError(f"Kaggle request root is not a directory: {root}")
    payload = _read_json(root / "request.json", "Kaggle delivery request")
    checksum = _nonempty(payload.get("request_checksum"), "request_checksum")
    without_checksum = dict(payload)
    without_checksum.pop("request_checksum", None)
    if _json_sha256(without_checksum) != checksum:
        raise KaggleDeliveryError("Kaggle delivery request checksum does not match its contents")
    if payload.get("schema_version") != KAGGLE_DELIVERY_SCHEMA_VERSION:
        raise KaggleDeliveryError("unsupported Kaggle delivery request schema")
    if payload.get("status") != "owner_approval_required":
        raise KaggleDeliveryError("Kaggle delivery request is not approval-gated")
    return payload


def load_kaggle_delivery(
    root: str | Path,
    *,
    candidate_root: str | Path | None = None,
    delivery_notebook_path: str | Path | None = None,
    delivery_adapter_path: str | Path | None = None,
) -> KaggleDeliveryRequest:
    """Load a request and optionally recheck the referenced local artifacts."""

    request_root = Path(root)
    payload = _request_payload(request_root)
    checksum = _nonempty(payload.get("request_checksum"), "request_checksum")
    if candidate_root is not None:
        bundle = load_candidate(candidate_root)
        candidate = _mapping(payload.get("candidate"), "request candidate")
        if candidate.get("candidate_id") != bundle.candidate_id or candidate.get("manifest_checksum") != bundle.manifest_checksum:
            raise KaggleDeliveryError("request does not reference the supplied candidate")
        if _candidate_reference(bundle) != dict(candidate):
            raise KaggleDeliveryError("request candidate reference differs from the supplied candidate")
    if delivery_notebook_path is not None:
        notebook = _notebook_audit(Path(delivery_notebook_path))
        expected = _mapping(payload.get("delivery_notebook"), "request delivery notebook")
        if notebook["sha256"] != expected.get("sha256"):
            raise KaggleDeliveryError("request notebook checksum differs from the supplied notebook")
    if delivery_adapter_path is not None:
        adapter = _adapter_audit(Path(delivery_adapter_path))
        expected = _mapping(payload.get("delivery_adapter"), "request delivery adapter")
        if adapter["sha256"] != expected.get("sha256"):
            raise KaggleDeliveryError("request adapter checksum differs from the supplied adapter")
    return KaggleDeliveryRequest(request_root.resolve(), checksum, payload)


def prepare_kaggle_delivery(
    candidate_root: str | Path,
    local_evidence_path: str | Path,
    output_root: str | Path,
    *,
    delivery_notebook_path: str | Path | None = None,
    delivery_adapter_path: str | Path = DEFAULT_KAGGLE_ADAPTER_PATH,
    local_submission_path: str | Path | None = None,
    expected_study_uids: Sequence[str] | None = None,
    request_id: str | None = None,
) -> KaggleDeliveryRequest:
    """Create one immutable, approval-required Kaggle execution request.

    Preparation is intentionally a local read/validate/write operation.  It
    never calls Kaggle and never changes the candidate or the local evidence.
    """

    bundle = load_candidate(candidate_root)
    candidate = _candidate_reference(bundle)
    candidate_root_path = Path(candidate_root).resolve()
    evidence_path = Path(local_evidence_path)
    notebook_path = (
        Path(delivery_notebook_path)
        if delivery_notebook_path is not None
        else candidate_root_path / "notebook" / "ticket11_offline_candidate.ipynb"
    )
    notebook = _notebook_audit(notebook_path)
    adapter_path = Path(delivery_adapter_path)
    adapter = _adapter_audit(adapter_path)
    evidence = _local_evidence_audit(
        evidence_path,
        candidate,
        local_submission_path=(Path(local_submission_path) if local_submission_path is not None else None),
        expected_study_uids=expected_study_uids,
    )
    request_name = request_id or f"ticket-12-{candidate['candidate_id']}"
    request_name = _safe_identifier(request_name, "request_id")
    output = Path(output_root)
    _ensure_outside(output, (candidate_root_path,), "Kaggle request")
    payload: dict[str, object] = {
        "schema_version": KAGGLE_DELIVERY_SCHEMA_VERSION,
        "request_id": request_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "owner_approval_required",
        "candidate": candidate,
        "delivery_notebook": {
            "filename": notebook["filename"],
            "sha256": notebook["sha256"],
            "source": "explicit_delivery_adapter" if delivery_notebook_path is not None else "frozen_candidate",
            "network_instructions": False,
        },
        "delivery_adapter": {
            "filename": adapter["filename"],
            "sha256": adapter["sha256"],
            "source": "explicit_delivery_adapter",
            "network_instructions": False,
        },
        "delivery_layout": {
            "request": {"relative_path": "request.json"},
            "candidate": {
                "relative_path": "candidate/",
                "manifest_checksum": candidate["manifest_checksum"],
            },
            "notebook": {
                "relative_path": str(notebook["filename"]),
                "sha256": notebook["sha256"],
            },
            "adapter": {
                "relative_path": f"code/rsna_knee/{adapter['filename']}",
                "sha256": adapter["sha256"],
            },
            "offline_requirements": {
                "relative_path": "candidate/environment/offline-requirements.lock",
                "sha256": _mapping(candidate["environment"], "candidate environment")[
                    "offline_requirements_sha256"
                ],
                "runtime_pins": _mapping(candidate["environment"], "candidate environment")["runtime_pins"],
            },
        },
        "local_validation": evidence,
        "approval": {
            "required": True,
            "approved": False,
            "status": "pending_owner_confirmation",
            "scope": OWNER_APPROVAL_SCOPE,
            "rules_and_eligibility_must_be_rechecked": True,
        },
        "upload": {
            "allowed": False,
            "submission_allowed": False,
            "actions_not_performed": ["notebook_upload", "model_upload", "competition_submission"],
        },
        "claims": {
            "kaggle_verified": False,
            "hidden_scale_runtime_verified": False,
            "hidden_fallback_rate_known": False,
            "owner_selected_final": False,
        },
    }
    payload["request_checksum"] = _json_sha256(payload)

    def writer(staging: Path) -> None:
        atomic_json_write(staging / "request.json", payload)

    _publish_directory(output, writer)
    return load_kaggle_delivery(output)


def _approval_payload(path: Path, request_checksum: str) -> Mapping[str, object]:
    if not path.is_file() or path.is_symlink():
        raise KaggleDeliveryError(f"owner approval is not a regular file: {path}")
    payload = _read_json(path, "owner approval")
    if payload.get("schema_version") != KAGGLE_APPROVAL_SCHEMA_VERSION:
        raise KaggleDeliveryError("owner approval has an unexpected schema")
    if payload.get("request_checksum") != request_checksum:
        raise KaggleDeliveryError("owner approval does not name this request")
    if payload.get("approval_scope") != OWNER_APPROVAL_SCOPE:
        raise KaggleDeliveryError("owner approval scope is not the exact ticket-12 bundle")
    if payload.get("approved") is not True:
        raise KaggleDeliveryError("owner approval is not granted")
    for field in (
        "rules_rechecked",
        "rules_accepted_by_owner",
        "account_eligible",
        "upload_allowed",
        "competition_submission_allowed",
    ):
        if payload.get(field) is not True:
            raise KaggleDeliveryError(f"owner approval does not confirm {field}")
    rules_evidence = _mapping(payload.get("rules_evidence"), "owner rules evidence")
    checked_at = _nonempty(rules_evidence.get("checked_at_utc"), "owner rules evidence timestamp")
    try:
        datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise KaggleDeliveryError("owner rules evidence timestamp is not ISO-8601") from exc
    sources = rules_evidence.get("sources")
    if not isinstance(sources, list) or not sources or not all(str(source).strip() for source in sources):
        raise KaggleDeliveryError("owner rules evidence must list its official sources")
    if not isinstance(rules_evidence.get("material_changes_reconciled"), bool):
        raise KaggleDeliveryError("owner rules evidence must state whether material changes were reconciled")
    _nonempty(payload.get("approver"), "owner approval approver")
    timestamp = _nonempty(payload.get("approved_at_utc"), "owner approval timestamp")
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise KaggleDeliveryError("owner approval timestamp is not ISO-8601") from exc
    return payload


def _private_execution_audit(
    path: Path,
    candidate_manifest_checksum: str,
    expected_study_count: int,
) -> tuple[str, int]:
    if not path.is_file() or path.is_symlink():
        raise KaggleDeliveryError(f"private Kaggle execution evidence is not a regular file: {path}")
    payload = _read_json(path, "private Kaggle execution evidence")
    if payload.get("schema_version") != KAGGLE_EXECUTION_SCHEMA_VERSION:
        raise KaggleDeliveryError("private Kaggle execution evidence has an unexpected schema")
    if payload.get("candidate_manifest_checksum") != candidate_manifest_checksum:
        raise KaggleDeliveryError("private Kaggle execution evidence uses a different candidate")
    studies = payload.get("studies")
    if not isinstance(studies, list) or len(studies) != expected_study_count:
        raise KaggleDeliveryError("private Kaggle execution evidence does not cover every study")
    seen_uids: set[str] = set()
    fallback_count = 0
    for item in studies:
        study = _mapping(item, "private Kaggle study evidence")
        study_uid = _nonempty(study.get("StudyInstanceUID"), "private study identifier")
        if study_uid in seen_uids:
            raise KaggleDeliveryError("private Kaggle execution evidence repeats a study")
        seen_uids.add(study_uid)
        prediction = _mapping(study.get("prediction"), "private study prediction evidence")
        if not isinstance(prediction.get("used_full_study_fallback"), bool):
            raise KaggleDeliveryError("private study prediction evidence must state fallback usage")
        if prediction.get("used_full_study_fallback") is True:
            fallback_count += 1
        if not isinstance(prediction.get("valid_series_count"), int):
            raise KaggleDeliveryError("private study prediction evidence must state valid series")
        if not isinstance(prediction.get("rejected_series"), list):
            raise KaggleDeliveryError("private study prediction evidence must list rejected series")
        if not isinstance(prediction.get("warnings"), list):
            raise KaggleDeliveryError("private study prediction evidence must list warnings")
    return _sha256_file(path), fallback_count


def _execution_payload(
    path: Path,
    request: KaggleDeliveryRequest,
    *,
    submission_path: Path | None,
    private_evidence_path: Path | None,
    expected_study_uids: Sequence[str] | None,
    engineering_margin_approved: bool,
) -> dict[str, object]:
    raw = _read_json(path, "Kaggle execution evidence")
    if raw.get("schema_version") != KAGGLE_EXECUTION_SCHEMA_VERSION:
        raise KaggleDeliveryError("Kaggle execution evidence has an unexpected schema")
    if _private_metadata_keys(raw):
        raise KaggleDeliveryError("Kaggle execution evidence contains identifier-bearing fields")
    if raw.get("status") != "accepted_by_kaggle":
        raise KaggleDeliveryError("Kaggle execution was not accepted by Kaggle")
    bundle = _mapping(raw.get("bundle"), "Kaggle execution bundle")
    request_candidate = _mapping(request.payload.get("candidate"), "request candidate")
    if bundle.get("request_checksum") != request.request_checksum:
        raise KaggleDeliveryError("Kaggle execution used a different delivery request")
    if bundle.get("candidate_manifest_checksum") != request_candidate.get("manifest_checksum"):
        raise KaggleDeliveryError("Kaggle execution used a different candidate manifest")
    if bundle.get("notebook_sha256") != _mapping(request.payload.get("delivery_notebook"), "request notebook").get("sha256"):
        raise KaggleDeliveryError("Kaggle execution used a different delivery notebook")
    if bundle.get("adapter_sha256") != _mapping(request.payload.get("delivery_adapter"), "request delivery adapter").get("sha256"):
        raise KaggleDeliveryError("Kaggle execution used a different delivery adapter")
    execution_models = bundle.get("model_checkpoints")
    if execution_models != request_candidate.get("model_checkpoints"):
        raise KaggleDeliveryError("Kaggle execution used a different checkpoint set")
    if bundle.get("model_size_bytes") != request_candidate.get("model_size_bytes"):
        raise KaggleDeliveryError("Kaggle execution used a different model size")
    if bundle.get("candidate_bundle_size_bytes") != request_candidate.get("candidate_bundle_size_bytes"):
        raise KaggleDeliveryError("Kaggle execution used a different candidate package size")
    if bundle.get("environment") != request_candidate.get("environment"):
        raise KaggleDeliveryError("Kaggle execution used a different pinned environment")

    environment = _mapping(raw.get("environment"), "Kaggle execution environment")
    if environment.get("runtime") != "kaggle":
        raise KaggleDeliveryError("execution environment is not Kaggle")
    if environment.get("internet_enabled") is not False or environment.get("runtime_network_download") is not False:
        raise KaggleDeliveryError("Kaggle execution was not demonstrably offline")
    if environment.get("reports_required") is not False:
        raise KaggleDeliveryError("Kaggle execution requires reports at inference")
    for field in ("python_version", "torch_version", "accelerator"):
        _nonempty(environment.get(field), f"Kaggle environment {field}")
    requested_environment = _mapping(request_candidate.get("environment"), "request candidate environment")
    requested_runtime_pins = _mapping(requested_environment.get("runtime_pins"), "request runtime pins")
    installed_runtime_pins_payload = _mapping(
        environment.get("installed_runtime_pins"),
        "Kaggle installed runtime pins",
    )
    installed_runtime_pins = {
        package: _nonempty(installed_runtime_pins_payload.get(package), f"Kaggle installed {package} pin")
        for package in ("numpy", "pydicom", "torch")
    }
    if installed_runtime_pins != {str(key): str(value) for key, value in requested_runtime_pins.items()}:
        raise KaggleDeliveryError("Kaggle installed runtime pins differ from the pinned offline environment")

    resources = _mapping(raw.get("resources"), "Kaggle execution resources")
    model_size_bytes = _required_int(resources.get("model_size_bytes"), "Kaggle model size", positive=True)
    candidate_bundle_size_bytes = _required_int(
        resources.get("candidate_bundle_size_bytes"),
        "Kaggle candidate package size",
        positive=True,
    )
    peak_vram_bytes = _required_int(resources.get("peak_vram_bytes"), "Kaggle peak VRAM")
    ensemble_model_count = _required_int(resources.get("ensemble_model_count"), "Kaggle ensemble model count", positive=True)
    if model_size_bytes != request_candidate.get("model_size_bytes"):
        raise KaggleDeliveryError("Kaggle resource evidence has a different model size")
    if candidate_bundle_size_bytes != request_candidate.get("candidate_bundle_size_bytes"):
        raise KaggleDeliveryError("Kaggle resource evidence has a different candidate package size")
    expected_checkpoints = request_candidate.get("model_checkpoints")
    if not isinstance(expected_checkpoints, list):
        raise KaggleDeliveryError("request candidate checkpoint set is invalid")
    if ensemble_model_count != len(expected_checkpoints):
        raise KaggleDeliveryError("Kaggle resource evidence has a different ensemble size")

    workload = _mapping(raw.get("workload"), "Kaggle execution workload")
    study_count = _required_int(workload.get("study_count"), "Kaggle workload study count", positive=True)
    _required_int(workload.get("series_count"), "Kaggle workload series count")
    _required_int(workload.get("slice_count"), "Kaggle workload slice count")
    if workload.get("actual_workload_recorded") is not True:
        raise KaggleDeliveryError("Kaggle execution does not record its actual workload")
    if workload.get("hidden_test") is not True or workload.get("example_only") is not False:
        raise KaggleDeliveryError("example-only execution cannot establish Kaggle verification")
    private_evidence_result = (
        _private_execution_audit(
            Path(private_evidence_path),
            str(request_candidate.get("manifest_checksum")),
            study_count,
        )
        if private_evidence_path is not None
        else None
    )
    private_evidence_sha256 = private_evidence_result[0] if private_evidence_result is not None else None
    private_fallback_count = private_evidence_result[1] if private_evidence_result is not None else None

    timing = _mapping(raw.get("timing"), "Kaggle execution timing")
    whole_seconds = _required_float(timing.get("whole_notebook_seconds"), "whole-notebook runtime", positive=True)
    if whole_seconds > KAGGLE_HARD_LIMIT_SECONDS:
        raise KaggleDeliveryError("Kaggle execution exceeded the nine-hour hard limit")
    stages = _mapping(timing.get("stages"), "Kaggle runtime stages")
    normalized_stages: dict[str, float] = {}
    for stage in KAGGLE_RUNTIME_STAGES:
        normalized_stages[stage] = _required_float(stages.get(stage), f"Kaggle runtime stage {stage}")
    if timing.get("stage_timing_source") != "notebook_wall_clock":
        raise KaggleDeliveryError("Kaggle runtime stages must be measured by the notebook wall clock")
    projection = _mapping(timing.get("projection"), "Kaggle runtime projection")
    if _required_int(projection.get("target_studies"), "Kaggle projection target studies", positive=True) != KAGGLE_TARGET_STUDIES:
        raise KaggleDeliveryError("Kaggle runtime projection must target approximately 1,300 studies")
    projected_seconds = _required_float(
        projection.get("projected_seconds_at_target"),
        "Kaggle projected runtime",
        positive=True,
    )
    _nonempty(projection.get("method"), "Kaggle runtime projection method")
    observed_within_target = whole_seconds <= KAGGLE_ENGINEERING_TARGET_SECONDS
    within_target = projected_seconds <= KAGGLE_ENGINEERING_TARGET_SECONDS
    if not within_target and not engineering_margin_approved:
        v16_status = "blocked_engineering_margin_not_approved"
    elif within_target:
        v16_status = "passed"
    else:
        v16_status = "passed_with_owner_margin_approval"

    output = _mapping(raw.get("output"), "Kaggle execution output")
    if output.get("filename") != "submission.csv":
        raise KaggleDeliveryError("Kaggle output is not named submission.csv")
    if output.get("accepted_by_kaggle") is not True:
        raise KaggleDeliveryError("Kaggle did not accept submission.csv")
    if _required_int(output.get("row_count"), "Kaggle output row count", positive=True) != study_count:
        raise KaggleDeliveryError("Kaggle output rows do not cover the recorded workload")
    if _required_int(output.get("column_count"), "Kaggle output column count") != NUM_TARGETS + 1:
        raise KaggleDeliveryError("Kaggle output does not have thirteen columns")
    for field in ("exact_header", "exact_uid_order", "scores_finite", "scores_bounded"):
        if output.get(field) is not True:
            raise KaggleDeliveryError(f"Kaggle output does not prove {field}")
    output_sha = str(output.get("sha256", "")).strip() or None
    if submission_path is not None:
        local_output = _local_submission_audit(Path(submission_path), expected_study_uids)
        if local_output["row_count"] != study_count:
            raise KaggleDeliveryError("accepted Kaggle output row count differs from the supplied CSV")
        if output_sha is not None and output_sha != local_output["sha256"]:
            raise KaggleDeliveryError("accepted Kaggle output checksum differs from the supplied CSV")
        output_sha = cast(str, local_output["sha256"])
    if output_sha is None:
        raise KaggleDeliveryError("accepted Kaggle output needs a checksum or a supplied submission.csv")

    fallback = _mapping(raw.get("fallback"), "Kaggle fallback evidence")
    fallback_status = str(fallback.get("status", "")).strip()
    denominator = _required_int(fallback.get("denominator"), "Kaggle fallback denominator", positive=True)
    if denominator != study_count:
        raise KaggleDeliveryError("Kaggle fallback denominator differs from the actual workload")
    measured_fallback_numerator: int | None = None
    if fallback_status == "unknown":
        if fallback.get("numerator") is not None or fallback.get("rate") is not None:
            raise KaggleDeliveryError("unknown fallback evidence must not contain invented zero values")
        if fallback.get("unknown_hidden_failure_rate") is not True:
            raise KaggleDeliveryError("unknown fallback evidence must explicitly remain unknown")
        fallback_summary: dict[str, object] = {
            "status": "unknown",
            "numerator": None,
            "denominator": denominator,
            "rate": None,
            "hidden_failure_rate_unknown": True,
        }
        v17_status = "blocked_hidden_fallback_rate_unknown"
        fallback_eligible = False
    elif fallback_status == "measured":
        numerator = _required_int(fallback.get("numerator"), "Kaggle fallback numerator")
        if numerator > denominator:
            raise KaggleDeliveryError("Kaggle fallback numerator exceeds its denominator")
        measured_fallback_numerator = numerator
        rate = _required_float(fallback.get("rate"), "Kaggle fallback rate")
        if not math.isclose(rate, numerator / denominator, rel_tol=1e-9, abs_tol=1e-12):
            raise KaggleDeliveryError("Kaggle fallback rate is inconsistent with its counts")
        if fallback.get("unknown_hidden_failure_rate") is not False:
            raise KaggleDeliveryError("measured fallback evidence must say that its rate is known")
        investigation = _mapping(fallback.get("investigation", {}), "Kaggle fallback investigation")
        if numerator > 0 and investigation.get("complete") is not True:
            raise KaggleDeliveryError("nonzero Kaggle fallback requires complete case investigation")
        fallback_summary = {
            "status": "measured",
            "numerator": numerator,
            "denominator": denominator,
            "rate": rate,
            "hidden_failure_rate_unknown": False,
            "investigation": {
                "complete": investigation.get("complete") is True,
                "investigated_cases": investigation.get("investigated_cases"),
                "unavoidable_cases": investigation.get("unavoidable_cases"),
            },
        }
        v17_status = "passed" if numerator == 0 else "blocked_nonzero_fallback"
        fallback_eligible = numerator == 0
    else:
        raise KaggleDeliveryError("Kaggle fallback status must be measured or unknown")

    if fallback_status == "measured" and private_fallback_count != measured_fallback_numerator:
        raise KaggleDeliveryError("private fallback report count differs from accepted fallback evidence")

    recovery = _mapping(raw.get("recovery"), "Kaggle recovery evidence")
    for field in ("save_reload_verified", "interruption_recovery_verified"):
        if not isinstance(recovery.get(field), bool):
            raise KaggleDeliveryError(f"Kaggle recovery evidence {field} must be boolean")
    rollback_seconds = _optional_int(recovery.get("rollback_seconds"), "Kaggle recovery rollback seconds")
    recovery_eligible = (
        recovery.get("save_reload_verified") is True
        and recovery.get("interruption_recovery_verified") is True
    )
    if fallback_eligible and recovery_eligible is False:
        v17_status = "blocked_recovery_not_verified"
    fallback_eligible = fallback_eligible and recovery_eligible
    accepted_output_preserved = submission_path is not None
    if fallback_eligible and not accepted_output_preserved:
        v17_status = "blocked_accepted_output_not_preserved"
    private_evidence_preserved = private_evidence_sha256 is not None
    if fallback_eligible and accepted_output_preserved and not private_evidence_preserved:
        v17_status = "blocked_private_failure_report_not_preserved"

    normalized = {
        "execution_evidence_sha256": _sha256_file(path),
        "environment": {
            "runtime": "kaggle",
            "internet_enabled": False,
            "runtime_network_download": False,
            "reports_required": False,
            "python_version": _nonempty(environment.get("python_version"), "Kaggle environment python_version"),
            "torch_version": _nonempty(environment.get("torch_version"), "Kaggle environment torch_version"),
            "accelerator": _nonempty(environment.get("accelerator"), "Kaggle environment accelerator"),
            "installed_runtime_pins": installed_runtime_pins,
        },
        "resources": {
            "model_size_bytes": model_size_bytes,
            "candidate_bundle_size_bytes": candidate_bundle_size_bytes,
            "peak_vram_bytes": peak_vram_bytes,
            "ensemble_model_count": ensemble_model_count,
        },
        "workload": {
            "study_count": study_count,
            "series_count": _required_int(workload.get("series_count"), "Kaggle workload series count"),
            "slice_count": _required_int(workload.get("slice_count"), "Kaggle workload slice count"),
            "actual_workload_recorded": True,
            "hidden_test": True,
            "example_only": False,
        },
        "timing": {
            "whole_notebook_seconds": whole_seconds,
            "stages": normalized_stages,
            "engineering_target_seconds": KAGGLE_ENGINEERING_TARGET_SECONDS,
            "hard_limit_seconds": KAGGLE_HARD_LIMIT_SECONDS,
            "within_engineering_target": within_target,
            "observed_within_engineering_target": observed_within_target,
            "within_hard_limit": True,
            "stage_timing_source": "notebook_wall_clock",
            "projection": {
                "target_studies": KAGGLE_TARGET_STUDIES,
                "projected_seconds_at_target": projected_seconds,
                "method": str(projection.get("method")),
            },
        },
        "output": {
            "filename": "submission.csv",
            "accepted_by_kaggle": True,
            "row_count": study_count,
            "column_count": NUM_TARGETS + 1,
            "exact_header": True,
            "exact_uid_order": True,
            "scores_finite": True,
            "scores_bounded": True,
            "sha256": output_sha,
            "accepted_output_preserved": accepted_output_preserved,
        },
        "fallback": fallback_summary,
        "private_evidence": {
            "sha256": private_evidence_sha256,
            "preserved": private_evidence_preserved,
        },
        "recovery": {
            "save_reload_verified": recovery.get("save_reload_verified"),
            "interruption_recovery_verified": recovery.get("interruption_recovery_verified"),
            "rollback_seconds": rollback_seconds,
        },
        "gates": {
            "v08_exact_offline_bundle_and_output": "passed",
            "v12_owner_approved_exact_request": "passed",
            "v16_whole_notebook_runtime": v16_status,
            "v17_complete_rows_and_fallback": v17_status,
        },
        "promotion_eligible": bool(
            fallback_eligible
            and accepted_output_preserved
            and private_evidence_preserved
            and (within_target or engineering_margin_approved)
        ),
    }
    return normalized


def record_kaggle_verification(
    delivery_root: str | Path,
    candidate_root: str | Path,
    approval_path: str | Path,
    execution_evidence_path: str | Path,
    output_root: str | Path,
    *,
    submission_path: str | Path | None = None,
    private_evidence_path: str | Path | None = None,
    expected_study_uids: Sequence[str] | None = None,
) -> KaggleVerification:
    """Record owner-approved external Kaggle evidence without mutating inputs."""

    request = load_kaggle_delivery(delivery_root, candidate_root=candidate_root)
    approval_file = Path(approval_path)
    approval = _approval_payload(approval_file, request.request_checksum)
    output = Path(output_root)
    candidate = Path(candidate_root)
    _ensure_outside(output, (request.root, candidate.resolve()), "Kaggle verification")
    execution_path = Path(execution_evidence_path)
    engineering_margin_approved = approval.get("engineering_margin_approved") is True
    execution = _execution_payload(
        execution_path,
        request,
        submission_path=(Path(submission_path) if submission_path is not None else None),
        private_evidence_path=(Path(private_evidence_path) if private_evidence_path is not None else None),
        expected_study_uids=expected_study_uids,
        engineering_margin_approved=engineering_margin_approved,
    )
    request_candidate = _mapping(request.payload.get("candidate"), "request candidate")
    payload: dict[str, object] = {
        "schema_version": KAGGLE_VERIFICATION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "kaggle_execution_accepted",
        "request_checksum": request.request_checksum,
        "candidate": dict(request_candidate),
        "approval": {
            "sha256": _sha256_file(approval_file),
            "approver": _nonempty(approval.get("approver"), "approval approver"),
            "approved_at_utc": _nonempty(approval.get("approved_at_utc"), "approval timestamp"),
            "engineering_margin_approved": engineering_margin_approved,
        },
        "execution": execution,
        "environment": execution["environment"],
        "workload": execution["workload"],
        "timing": execution["timing"],
        "output": execution["output"],
        "fallback": execution["fallback"],
        "gates": execution["gates"],
        "promotion": {
            "eligible": execution["promotion_eligible"],
            "level": "kaggle_verified" if execution["promotion_eligible"] else "kaggle_execution_recorded_pending_gates",
            "final_candidate": False,
            "owner_selection_required": True,
            "candidate_mutated": False,
            "export_allowed": False,
        },
        "preservation": {
            "request_unchanged": True,
            "candidate_unchanged": True,
            "accepted_output_preserved": submission_path is not None,
            "private_failure_report_preserved": private_evidence_path is not None,
        },
    }
    payload["verification_checksum"] = _json_sha256(payload)

    def writer(staging: Path) -> None:
        atomic_json_write(staging / "verification.json", payload)
        if submission_path is not None:
            private = staging / "private"
            private.mkdir(parents=True, exist_ok=True)
            source = Path(submission_path)
            if not source.is_file() or source.is_symlink():
                raise KaggleDeliveryError(f"accepted submission is not a regular file: {source}")
            shutil.copyfile(source, private / "submission.csv")
        if private_evidence_path is not None:
            private = staging / "private"
            private.mkdir(parents=True, exist_ok=True)
            source = Path(private_evidence_path)
            if not source.is_file() or source.is_symlink():
                raise KaggleDeliveryError(f"private execution evidence is not a regular file: {source}")
            shutil.copyfile(source, private / "execution.json")

    _publish_directory(output, writer)
    return load_kaggle_verification(output)


def _verification_payload(root: Path) -> Mapping[str, object]:
    if not root.is_dir() or root.is_symlink():
        raise KaggleDeliveryError(f"Kaggle verification root is not a directory: {root}")
    payload = _read_json(root / "verification.json", "Kaggle verification")
    checksum = _nonempty(payload.get("verification_checksum"), "verification_checksum")
    without_checksum = dict(payload)
    without_checksum.pop("verification_checksum", None)
    if _json_sha256(without_checksum) != checksum:
        raise KaggleDeliveryError("Kaggle verification checksum does not match its contents")
    if payload.get("schema_version") != KAGGLE_VERIFICATION_SCHEMA_VERSION:
        raise KaggleDeliveryError("unsupported Kaggle verification schema")
    if payload.get("status") != "kaggle_execution_accepted":
        raise KaggleDeliveryError("Kaggle verification is not an accepted execution")
    preservation = _mapping(payload.get("preservation"), "Kaggle verification preservation")
    if preservation.get("accepted_output_preserved") is True:
        execution = _mapping(payload.get("execution"), "Kaggle verification execution")
        output = _mapping(execution.get("output"), "Kaggle verification output")
        private_output = root / "private" / "submission.csv"
        if not private_output.is_file() or private_output.is_symlink():
            raise KaggleDeliveryError("Kaggle verification is missing its preserved submission.csv")
        if _sha256_file(private_output) != _nonempty(output.get("sha256"), "Kaggle output sha256"):
            raise KaggleDeliveryError("preserved Kaggle submission checksum does not match evidence")
    if preservation.get("private_failure_report_preserved") is True:
        execution = _mapping(payload.get("execution"), "Kaggle verification execution")
        private_evidence = _mapping(execution.get("private_evidence"), "Kaggle private evidence")
        private_output = root / "private" / "execution.json"
        if not private_output.is_file() or private_output.is_symlink():
            raise KaggleDeliveryError("Kaggle verification is missing its private execution evidence")
        if _sha256_file(private_output) != _nonempty(private_evidence.get("sha256"), "private evidence sha256"):
            raise KaggleDeliveryError("preserved private execution evidence checksum does not match evidence")
    return payload


def load_kaggle_verification(root: str | Path) -> KaggleVerification:
    """Load one immutable external-execution evidence record."""

    verification_root = Path(root)
    payload = _verification_payload(verification_root)
    return KaggleVerification(
        verification_root.resolve(),
        _nonempty(payload.get("verification_checksum"), "verification_checksum"),
        payload,
    )


def promote_kaggle_candidate(
    delivery_root: str | Path,
    candidate_root: str | Path,
    verification_root: str | Path,
    output_root: str | Path,
) -> KagglePromotion:
    """Append a Kaggle-verified pointer while keeping the candidate immutable."""

    request = load_kaggle_delivery(delivery_root, candidate_root=candidate_root)
    verification = load_kaggle_verification(verification_root)
    if verification.payload.get("request_checksum") != request.request_checksum:
        raise KaggleDeliveryError("verification does not belong to this delivery request")
    candidate = _mapping(request.payload.get("candidate"), "request candidate")
    if verification.payload.get("candidate") != candidate:
        raise KaggleDeliveryError("verification candidate reference differs from the request")
    promotion = _mapping(verification.payload.get("promotion"), "verification promotion")
    if promotion.get("eligible") is not True:
        raise KaggleDeliveryError("Kaggle verification is not eligible for promotion")
    output = Path(output_root)
    _ensure_outside(output, (request.root, Path(candidate_root).resolve(), Path(verification_root).resolve()), "Kaggle promotion")
    payload: dict[str, object] = {
        "schema_version": KAGGLE_PROMOTION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "kaggle_verified_only",
        "candidate": dict(candidate),
        "request_checksum": request.request_checksum,
        "verification_checksum": verification.verification_checksum,
        "promotion": {
            "level": "kaggle_verified",
            "final_candidate": False,
            "owner_selection_required": True,
            "candidate_mutated": False,
            "export_allowed": False,
        },
        "gates": _mapping(_mapping(verification.payload.get("execution"), "verification execution").get("gates"), "verification gates"),
        "claims": {
            "actual_kaggle_workload_recorded": True,
            "hidden_scale_runtime_verified": True,
            "hidden_fallback_rate_known": True,
            "owner_selected_final": False,
        },
    }
    payload["promotion_checksum"] = _json_sha256(payload)

    def writer(staging: Path) -> None:
        atomic_json_write(staging / "promotion.json", payload)

    _publish_directory(output, writer)
    return load_kaggle_promotion(output)


def _promotion_payload(root: Path) -> Mapping[str, object]:
    if not root.is_dir() or root.is_symlink():
        raise KaggleDeliveryError(f"Kaggle promotion root is not a directory: {root}")
    payload = _read_json(root / "promotion.json", "Kaggle promotion")
    checksum = _nonempty(payload.get("promotion_checksum"), "promotion_checksum")
    without_checksum = dict(payload)
    without_checksum.pop("promotion_checksum", None)
    if _json_sha256(without_checksum) != checksum:
        raise KaggleDeliveryError("Kaggle promotion checksum does not match its contents")
    if payload.get("schema_version") != KAGGLE_PROMOTION_SCHEMA_VERSION:
        raise KaggleDeliveryError("unsupported Kaggle promotion schema")
    if payload.get("status") != "kaggle_verified_only":
        raise KaggleDeliveryError("Kaggle promotion is not a verified-only pointer")
    promotion = _mapping(payload.get("promotion"), "Kaggle promotion state")
    if promotion.get("final_candidate") is not False or promotion.get("owner_selection_required") is not True:
        raise KaggleDeliveryError("Kaggle promotion cannot silently become final")
    return payload


def load_kaggle_promotion(root: str | Path) -> KagglePromotion:
    """Load one immutable Kaggle promotion pointer."""

    promotion_root = Path(root)
    payload = _promotion_payload(promotion_root)
    return KagglePromotion(
        promotion_root.resolve(),
        _nonempty(payload.get("promotion_checksum"), "promotion_checksum"),
        payload,
    )


def _candidate_runtime(bundle: CandidateBundle) -> tuple[DecoderPolicy, PreprocessConfig, PackingConfig]:
    inference = bundle.inference_config
    packing_payload = _mapping(inference.get("packing"), "candidate packing")
    packing = PackingConfig(
        max_slices=_required_int(packing_payload.get("max_slices"), "candidate max_slices", positive=True),
        max_series=_required_int(packing_payload.get("max_series"), "candidate max_series", positive=True),
    )
    preprocessing = _mapping(inference.get("preprocessing"), "candidate preprocessing")
    try:
        geometry = GeometryPolicy.from_dict(preprocessing.get("geometry_policy"))
    except (TypeError, ValueError) as exc:
        raise KaggleDeliveryError("candidate geometry policy is invalid") from exc
    image_size = preprocessing.get("image_size")
    percentiles = preprocessing.get("normalization_percentiles")
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise KaggleDeliveryError("candidate preprocessing image_size is invalid")
    if not isinstance(percentiles, (list, tuple)) or len(percentiles) != 2:
        raise KaggleDeliveryError("candidate preprocessing percentiles are invalid")
    preprocess = PreprocessConfig(
        image_size=(int(str(image_size[0])), int(str(image_size[1]))),
        max_slices=packing.max_slices,
        max_series=packing.max_series,
        geometry_policy=geometry,
        normalization_domain=str(preprocessing.get("normalization_domain", "valid_slice")),
        normalization_percentiles=(float(str(percentiles[0])), float(str(percentiles[1]))),
        use_voi_lut=bool(preprocessing.get("use_voi_lut", False)),
        interpolation=str(preprocessing.get("interpolation", "nearest")),
        padding_fill_value=float(str(preprocessing.get("padding_fill_value", 0.0))),
        output_dtype=str(preprocessing.get("output_dtype", "float32")),
    )
    if preprocess.image_size != bundle.model_config.image_size:
        raise KaggleDeliveryError("candidate model and preprocessing image sizes differ")
    try:
        decoder = DecoderPolicy.from_dict(inference.get("decoder"))
        decoder.validate_runtime()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise KaggleDeliveryError("candidate decoder policy cannot run in this environment") from exc
    return decoder, preprocess, packing


def _load_kaggle_ensemble(bundle: CandidateBundle, device: str | torch.device) -> Any:
    """Load the candidate's own tested model/aggregation implementation."""

    try:
        return _load_candidate_models(bundle, device)
    except CandidateError as exc:
        raise KaggleDeliveryError("candidate model ensemble could not be restored") from exc


def _read_metadata_rows(path: Path, required_fields: Sequence[str], description: str) -> list[dict[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise KaggleDeliveryError(f"{description} is not a regular file: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            fields = tuple(reader.fieldnames or ())
            if not fields or len(set(fields)) != len(fields) or any(not field.strip() for field in fields):
                raise KaggleDeliveryError(f"{description} has an invalid header")
            missing = set(required_fields) - set(fields)
            if missing:
                raise KaggleDeliveryError(f"{description} is missing fields: {sorted(missing)}")
            rows: list[dict[str, str]] = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise KaggleDeliveryError(f"{description} contains a malformed row")
                rows.append({key: value for key, value in row.items() if key is not None})
    except (OSError, UnicodeError, csv.Error) as exc:
        raise KaggleDeliveryError(f"could not read {description}: {path}") from exc
    return rows


def _row_text(row: Mapping[str, str], field: str, description: str) -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        raise KaggleDeliveryError(f"{description} has an empty {field}")
    return value


def _row_optional_bool(row: Mapping[str, str], field: str, description: str) -> bool | None:
    value = str(row.get(field, "")).strip()
    if not value:
        return None
    if value not in {"0", "1"}:
        raise KaggleDeliveryError(f"{description} has an invalid {field}={value!r}")
    return value == "1"


def _safe_input_component(value: str, field: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise KaggleDeliveryError(f"{field} is not a safe input path component")
    return value


def _series_input_files(series_dir: Path, input_root: Path) -> tuple[Path, ...]:
    if not series_dir.is_dir() or series_dir.is_symlink():
        return ()
    files: list[Path] = []
    for path in series_dir.rglob("*"):
        if path.is_symlink():
            raise KaggleDeliveryError(f"Kaggle series input contains a symlink: {path}")
        if path.is_file():
            files.append(path)
    return tuple(sorted(files, key=lambda path: path.relative_to(input_root).as_posix()))


def _load_kaggle_studies(
    input_root: Path,
    decoder: DecoderPolicy,
) -> tuple[tuple[StudyRecord, ...], dict[str, object], tuple[dict[str, object], ...], float]:
    if not input_root.is_dir() or input_root.is_symlink():
        raise KaggleDeliveryError(f"Kaggle input root is not a directory: {input_root}")
    test_csv = input_root / "test.csv"
    test_series_csv = input_root / "test_series.csv"
    series_root = input_root / "test_series"
    if not series_root.is_dir() or series_root.is_symlink():
        raise KaggleDeliveryError(f"Kaggle input is missing test_series/: {series_root}")
    study_rows = _read_metadata_rows(test_csv, ("StudyInstanceUID",), "test.csv")
    series_rows = _read_metadata_rows(
        test_series_csv,
        ("StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"),
        "test_series.csv",
    )
    study_uids = tuple(_row_text(row, "StudyInstanceUID", "test.csv") for row in study_rows)
    if not study_uids:
        raise KaggleDeliveryError("test.csv contains no studies")
    if len(set(study_uids)) != len(study_uids):
        raise KaggleDeliveryError("test.csv repeats a StudyInstanceUID")
    study_set = set(study_uids)
    by_study: dict[str, list[Mapping[str, str]]] = {uid: [] for uid in study_uids}
    seen_series: set[str] = set()
    for row in series_rows:
        study_uid = _row_text(row, "StudyInstanceUID", "test_series.csv")
        series_uid = _row_text(row, "SeriesInstanceUID", "test_series.csv")
        if study_uid not in study_set:
            raise KaggleDeliveryError(f"test_series.csv references a study absent from test.csv: {study_uid}")
        if series_uid in seen_series:
            raise KaggleDeliveryError(f"test_series.csv repeats a SeriesInstanceUID: {series_uid}")
        seen_series.add(series_uid)
        by_study[study_uid].append(row)

    decoder.validate_runtime()
    decode_started = time.perf_counter()
    private_records: list[dict[str, object]] = []
    expected_slices = 0
    decoded_slices = 0
    failure_reasons: dict[str, int] = {}
    studies: list[StudyRecord] = []
    for study_uid in study_uids:
        series_records: list[SeriesRecord] = []
        study_private: dict[str, object] = {"StudyInstanceUID": study_uid, "series": []}
        private_series = cast(list[dict[str, object]], study_private["series"])
        study_dir = series_root / _safe_input_component(study_uid, "StudyInstanceUID")
        for series_row in by_study[study_uid]:
            series_uid = _row_text(series_row, "SeriesInstanceUID", "test_series.csv")
            series_dir = study_dir / _safe_input_component(series_uid, "SeriesInstanceUID")
            if series_dir.exists() and series_dir.is_symlink():
                raise KaggleDeliveryError(f"Kaggle series directory is a symlink: {series_dir}")
            files = _series_input_files(series_dir, input_root)
            slices = []
            failures: list[dict[str, object]] = []
            expected_slices += len(files)
            for path in files:
                relative = path.relative_to(input_root).as_posix()
                try:
                    with path.open("rb") as handle:
                        decoded = decode_stream(handle, filename=relative, policy=decoder)
                    if decoded.study_instance_uid not in (None, study_uid):
                        raise DicomDecodeError("decoded StudyInstanceUID disagrees with test.csv")
                    if decoded.series_instance_uid not in (None, series_uid):
                        raise DicomDecodeError("decoded SeriesInstanceUID disagrees with test_series.csv")
                    slices.append(decoded)
                    decoded_slices += 1
                except DecoderInfrastructureError:
                    raise
                except (DicomDecodeError, OSError) as exc:
                    reason = type(exc).__name__
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                    failures.append({"source_entry_name": relative, "reason": reason, "detail": str(exc)})
            if not files:
                failure_reasons["missing_series_input"] = failure_reasons.get("missing_series_input", 0) + 1
                failures.append({"source_entry_name": "", "reason": "missing_series_input", "detail": str(series_dir)})
            series_records.append(
                SeriesRecord(
                    study_instance_uid=study_uid,
                    series_instance_uid=series_uid,
                    slices=tuple(slices),
                    anatomical_plane=str(series_row.get("Anatomical_Plane", "")).strip() or None,
                    fluid_sensitive=_row_optional_bool(series_row, "Fluid_Sensitive", "test_series.csv"),
                    fat_suppression=_row_optional_bool(series_row, "Fat_Suppression", "test_series.csv"),
                    source_entry_names=tuple(
                        path.relative_to(input_root).as_posix() for path in files
                    ),
                )
            )
            private_series.append(
                {
                    "SeriesInstanceUID": series_uid,
                    "source_entry_names": [path.relative_to(input_root).as_posix() for path in files],
                    "expected_slice_count": len(files),
                    "decoded_slice_count": len(slices),
                    "failures": failures,
                }
            )
        studies.append(StudyRecord(study_uid, tuple(series_records)))
        private_records.append(study_private)
    decode_seconds = time.perf_counter() - decode_started
    return (
        tuple(studies),
        {
            "study_count": len(studies),
            "series_count": len(series_rows),
            "slice_count": expected_slices,
            "decoded_slice_count": decoded_slices,
            "failure_reasons": dict(sorted(failure_reasons.items())),
        },
        tuple(private_records),
        decode_seconds,
    )


def _predict_kaggle_study(
    engine: InferenceEngine,
    study: StudyRecord,
    input_failures: Sequence[SeriesFailure],
    input_warnings: Sequence[str] = (),
) -> tuple[StudyPrediction, dict[str, float]]:
    preprocessing_started = time.perf_counter()
    prepared = prepare_study(study, engine.preprocess_config)
    packed = pack_study(
        study,
        prepared.series,
        engine.preprocess_config,
        packing_config=engine.packing_config,
    )
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    valid_series_count = int(packed.series_mask.sum())
    rejected_series = tuple(input_failures) + prepared.failures
    warnings = tuple(input_warnings) + prepared.warnings
    if valid_series_count == 0:
        return (
            StudyPrediction(
                study_instance_uid=study.study_instance_uid,
                scores=engine.fallback_scores.copy(),
                used_fallback=True,
                valid_series_count=0,
                rejected_series=rejected_series,
                sampled_source_indices=packed.sampled_source_indices,
                warnings=warnings,
            ),
            {"preprocessing_seconds": preprocessing_seconds, "inference_ensembling_seconds": 0.0},
        )
    inference_started = time.perf_counter()
    images = torch.from_numpy(packed.images).unsqueeze(0).to(engine.device)
    slice_mask = torch.from_numpy(packed.slice_mask).unsqueeze(0).to(engine.device)
    series_mask = torch.from_numpy(packed.series_mask).unsqueeze(0).to(engine.device)
    with torch.inference_mode():
        scores_tensor = cast(Any, engine.model).predict_proba(images, slice_mask, series_mask)
    scores = np.asarray(scores_tensor.squeeze(0).detach().cpu().numpy(), dtype=np.float64)
    inference_seconds = time.perf_counter() - inference_started
    if scores.shape != (NUM_TARGETS,) or not np.isfinite(scores).all() or (scores < 0.0).any() or (scores > 1.0).any():
        raise KaggleDeliveryError(f"model returned invalid scores for study {study.study_instance_uid}")
    return (
        StudyPrediction(
            study_instance_uid=study.study_instance_uid,
            scores=scores,
            used_fallback=False,
            valid_series_count=valid_series_count,
            rejected_series=rejected_series,
            sampled_source_indices=packed.sampled_source_indices,
            warnings=warnings,
        ),
        {"preprocessing_seconds": preprocessing_seconds, "inference_ensembling_seconds": inference_seconds},
    )


def run_kaggle_baseline(
    candidate_root: str | Path,
    input_root: str | Path,
    output_path: str | Path,
    *,
    evidence_path: str | Path | None = None,
    private_evidence_path: str | Path | None = None,
    delivery_request_checksum: str | None = None,
    delivery_notebook_sha256: str | None = None,
    delivery_adapter_sha256: str | None = None,
    delivery_request_root: str | Path | None = None,
    owner_approval_path: str | Path | None = None,
    delivery_notebook_path: str | Path | None = None,
    delivery_adapter_path: str | Path | None = None,
    notebook_stage_timings: Mapping[str, float] | None = None,
    device: str | torch.device = "cpu",
    allow_local_smoke: bool = False,
    notebook_started_at: float | None = None,
) -> KaggleInferenceResult:
    """Run the frozen model on Kaggle's test CSV/DICOM layout.

    The adapter reads ``test.csv``, ``test_series.csv``, and ``test_series/``
    from an already-mounted Kaggle input.  It reuses the candidate's decoder,
    preprocessing, hierarchical model and exact submission writer; it does
    not read reports or perform network operations.  Local execution requires
    ``allow_local_smoke=True`` and is not Kaggle verification.
    """

    if not allow_local_smoke and not (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE")
    ):
        raise KaggleDeliveryError("Kaggle baseline must run in a Kaggle notebook; use local smoke only explicitly")
    approval_sha256: str | None = None
    if not allow_local_smoke:
        required_paths = (
            (delivery_request_root, "delivery_request_root"),
            (owner_approval_path, "owner_approval_path"),
            (delivery_notebook_path, "delivery_notebook_path"),
            (delivery_adapter_path, "delivery_adapter_path"),
        )
        missing = [field for value, field in required_paths if value is None]
        if missing:
            raise KaggleDeliveryError("actual Kaggle execution requires: " + ", ".join(missing))
        if notebook_started_at is None:
            raise KaggleDeliveryError("actual Kaggle execution requires a notebook-start timer")
        if notebook_stage_timings is None:
            raise KaggleDeliveryError("actual Kaggle execution requires notebook startup/import timings")
        request = load_kaggle_delivery(
            cast(str | Path, delivery_request_root),
            candidate_root=candidate_root,
            delivery_notebook_path=cast(str | Path, delivery_notebook_path),
            delivery_adapter_path=cast(str | Path, delivery_adapter_path),
        )
        if delivery_request_checksum is not None and delivery_request_checksum != request.request_checksum:
            raise KaggleDeliveryError("supplied delivery request checksum differs from request.json")
        delivery_request_checksum = request.request_checksum
        request_notebook = _mapping(request.payload.get("delivery_notebook"), "request delivery notebook")
        request_adapter = _mapping(request.payload.get("delivery_adapter"), "request delivery adapter")
        request_notebook_sha256 = _nonempty(request_notebook.get("sha256"), "delivery notebook sha256")
        request_adapter_sha256 = _nonempty(request_adapter.get("sha256"), "delivery adapter sha256")
        if delivery_notebook_sha256 is not None and delivery_notebook_sha256 != request_notebook_sha256:
            raise KaggleDeliveryError("supplied delivery notebook checksum differs from request.json")
        if delivery_adapter_sha256 is not None and delivery_adapter_sha256 != request_adapter_sha256:
            raise KaggleDeliveryError("supplied delivery adapter checksum differs from request.json")
        delivery_notebook_sha256 = request_notebook_sha256
        delivery_adapter_sha256 = request_adapter_sha256
        approval_path = Path(cast(str | Path, owner_approval_path))
        _approval_payload(approval_path, request.request_checksum)
        approval_sha256 = _sha256_file(approval_path)
        _nonempty(delivery_request_checksum, "delivery_request_checksum")
        for stage in ("startup_setup", "imports"):
            _required_float(notebook_stage_timings.get(stage), f"notebook {stage} seconds")
    else:
        notebook_stage_timings = {"startup_setup": 0.0, "imports": 0.0}
    bundle = load_candidate(candidate_root)
    candidate = _candidate_reference(bundle)
    candidate_checkpoints = candidate.get("model_checkpoints")
    if not isinstance(candidate_checkpoints, list):
        raise KaggleDeliveryError("candidate checkpoint reference is invalid")
    output = Path(output_path)
    evidence_destination = Path(evidence_path) if evidence_path is not None else None
    private_destination = Path(private_evidence_path) if private_evidence_path is not None else None
    destinations = [output] + ([evidence_destination] if evidence_destination is not None else [])
    destinations += [private_destination] if private_destination is not None else []
    if len(set(destinations)) != len(destinations):
        raise KaggleDeliveryError("Kaggle output and evidence destinations must be distinct")
    for destination in destinations:
        if destination is None:
            continue
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite protected Kaggle output {destination}")
    _ensure_outside(output, (Path(candidate_root).resolve(),), "Kaggle output")
    if evidence_destination is not None:
        _ensure_outside(evidence_destination, (Path(candidate_root).resolve(),), "Kaggle evidence")
    if private_destination is not None:
        _ensure_outside(private_destination, (Path(candidate_root).resolve(),), "Kaggle private evidence")

    started = notebook_started_at if notebook_started_at is not None else time.perf_counter()
    input_path = Path(input_root)
    metadata_started = time.perf_counter()
    metadata_paths = (input_path / "test.csv", input_path / "test_series.csv")
    decoder, preprocess, packing = _candidate_runtime(bundle)
    studies, input_summary, private_records, decode_seconds = _load_kaggle_studies(
        input_path,
        decoder,
    )
    metadata_seconds = time.perf_counter() - metadata_started - decode_seconds
    device_object = torch.device(device)
    if device_object.type == "cuda":
        if not torch.cuda.is_available():
            raise KaggleDeliveryError("CUDA was requested but is unavailable")
        torch.cuda.reset_peak_memory_stats(device_object)
    model_started = time.perf_counter()
    model = _load_kaggle_ensemble(bundle, device)
    model_seconds = time.perf_counter() - model_started
    engine = InferenceEngine(
        cast(Any, model),
        preprocess,
        device=device,
        fallback_scores=tuple([0.5] * NUM_TARGETS),
        packing_config=packing,
    )
    predictions: list[StudyPrediction] = []
    preprocessing_seconds = 0.0
    inference_seconds = 0.0
    private_by_uid = {str(item["StudyInstanceUID"]): item for item in private_records}
    for study in studies:
        input_failures = tuple(
            SeriesFailure(
                str(series["SeriesInstanceUID"]),
                str(failure["reason"]),
            )
            for series in cast(list[Mapping[str, object]], private_by_uid[study.study_instance_uid]["series"])
            for failure in cast(list[Mapping[str, object]], series["failures"])
        )
        prediction, timing = _predict_kaggle_study(engine, study, input_failures)
        predictions.append(prediction)
        preprocessing_seconds += timing["preprocessing_seconds"]
        inference_seconds += timing["inference_ensembling_seconds"]
        private_prediction = private_by_uid[study.study_instance_uid]
        private_prediction["prediction"] = {
            "used_full_study_fallback": prediction.used_fallback,
            "valid_series_count": prediction.valid_series_count,
            "rejected_series": [
                {
                    "series_instance_uid": failure.series_instance_uid,
                    "reason": failure.reason,
                }
                for failure in prediction.rejected_series
            ],
            "warnings": list(prediction.warnings),
            "sampled_source_indices": [list(indices) for indices in prediction.sampled_source_indices],
        }
    output_started = time.perf_counter()
    expected_uids = tuple(study.study_instance_uid for study in studies)
    write_submission(
        output,
        expected_study_uids=expected_uids,
        predictions={item.study_instance_uid: item.scores for item in predictions},
    )
    submission_validation = validate_submission_csv(output, expected_uids)
    csv_seconds = time.perf_counter() - output_started
    whole_seconds = time.perf_counter() - started
    fallback_numerator = sum(item.used_fallback for item in predictions)
    fallback_denominator = len(predictions)
    fallback_rate = fallback_numerator / fallback_denominator
    if device_object.type == "cuda":
        peak_vram_bytes = int(torch.cuda.max_memory_allocated(device_object))
    else:
        peak_vram_bytes = 0
    projection_seconds = whole_seconds * KAGGLE_TARGET_STUDIES / len(studies)
    stage_timings = {
        "startup_setup": float(notebook_stage_timings["startup_setup"]),
        "imports": float(notebook_stage_timings["imports"]),
        "model_metadata_loading": max(0.0, metadata_seconds) + model_seconds,
        "dicom_read_decode": decode_seconds,
        "preprocessing": preprocessing_seconds,
        "inference_ensembling": inference_seconds,
        "csv_creation": csv_seconds,
    }
    evidence: dict[str, object] = {
        "schema_version": KAGGLE_EXECUTION_SCHEMA_VERSION,
        "ticket": "12",
        "status": "completed_kaggle_notebook",
        "bundle": {
            "request_checksum": delivery_request_checksum,
            "candidate_manifest_checksum": candidate["manifest_checksum"],
            "notebook_sha256": delivery_notebook_sha256,
            "adapter_sha256": delivery_adapter_sha256,
            "model_checkpoints": candidate_checkpoints,
            "model_size_bytes": candidate["model_size_bytes"],
            "candidate_bundle_size_bytes": candidate["candidate_bundle_size_bytes"],
            "environment": candidate["environment"],
        },
        "environment": {
            "runtime": "kaggle",
            "internet_enabled": False,
            "runtime_network_download": False,
            "reports_required": False,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "accelerator": str(device),
            "installed_runtime_pins": {
                "numpy": np.__version__,
                "pydicom": pydicom.__version__,
                "torch": torch.__version__,
            },
        },
        "resources": {
            "model_size_bytes": candidate["model_size_bytes"],
            "candidate_bundle_size_bytes": candidate["candidate_bundle_size_bytes"],
            "peak_vram_bytes": peak_vram_bytes,
            "ensemble_model_count": len(candidate_checkpoints),
        },
        "input": {
            "test_csv_sha256": _sha256_file(metadata_paths[0]),
            "test_series_csv_sha256": _sha256_file(metadata_paths[1]),
            "input_layout": "test.csv + test_series.csv + test_series/",
        },
        "workload": {
            "study_count": input_summary["study_count"],
            "series_count": input_summary["series_count"],
            "slice_count": input_summary["slice_count"],
            "decoded_slice_count": input_summary["decoded_slice_count"],
            "actual_workload_recorded": True,
            "hidden_test": not allow_local_smoke,
            "example_only": allow_local_smoke,
        },
        "timing": {
            "whole_notebook_seconds": whole_seconds,
            "stages": stage_timings,
            "stage_timing_source": "notebook_wall_clock" if not allow_local_smoke else "local_smoke_adapter",
            "projection": {
                "target_studies": KAGGLE_TARGET_STUDIES,
                "projected_seconds_at_target": projection_seconds,
                "method": "linear_study_count_scaling",
            },
        },
        "output": {
            "filename": "submission.csv",
            "accepted_by_kaggle": False,
            "accepted_by_local_contract": True,
            **submission_validation,
            "sha256": _sha256_file(output),
        },
        "fallback": {
            "status": "measured",
            "numerator": fallback_numerator,
            "denominator": fallback_denominator,
            "rate": fallback_rate,
            "unknown_hidden_failure_rate": allow_local_smoke,
            "investigation": {
                "complete": fallback_numerator == 0,
                "investigated_cases": 0,
                "unavoidable_cases": 0,
            },
        },
        "recovery": {
            "save_reload_verified": False,
            "interruption_recovery_verified": False,
            "rollback_seconds": None,
            "status": "not_demonstrated_by_inference_run",
        },
        "diagnostics": {
            "failure_reasons": input_summary["failure_reasons"],
            "fallback_cases_are_private": True,
        },
    }
    if approval_sha256 is not None:
        evidence["approval"] = {"sha256": approval_sha256}
    if evidence_destination is not None:
        atomic_json_write(evidence_destination, evidence)
    if private_destination is not None:
        atomic_json_write(
            private_destination,
            {
                "schema_version": KAGGLE_EXECUTION_SCHEMA_VERSION,
                "ticket": "12",
                "candidate_manifest_checksum": candidate["manifest_checksum"],
                "studies": list(private_records),
            },
        )
    return KaggleInferenceResult(output, evidence, private_destination)


__all__ = [
    "KAGGLE_APPROVAL_SCHEMA_VERSION",
    "DEFAULT_KAGGLE_ADAPTER_PATH",
    "KAGGLE_DELIVERY_SCHEMA_VERSION",
    "KAGGLE_ENGINEERING_TARGET_SECONDS",
    "KAGGLE_EXECUTION_SCHEMA_VERSION",
    "KAGGLE_HARD_LIMIT_SECONDS",
    "KAGGLE_PROMOTION_SCHEMA_VERSION",
    "KAGGLE_RUNTIME_STAGES",
    "KAGGLE_TARGET_STUDIES",
    "KAGGLE_VERIFICATION_SCHEMA_VERSION",
    "KaggleDeliveryError",
    "KaggleDeliveryRequest",
    "KaggleInferenceResult",
    "KagglePromotion",
    "KaggleVerification",
    "OWNER_APPROVAL_SCOPE",
    "load_kaggle_delivery",
    "load_kaggle_promotion",
    "load_kaggle_verification",
    "prepare_kaggle_delivery",
    "promote_kaggle_candidate",
    "record_kaggle_verification",
    "run_kaggle_baseline",
]
