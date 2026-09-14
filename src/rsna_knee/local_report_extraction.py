"""Approval-gated local language-model report extraction.

Ticket 16 compares one approved, locally executed pretrained language model
with the frozen deterministic report extractor from ticket 15.  This module
owns the resource-approval boundary, the frozen prompt, local artifact
verification, and (in later sections) the fold-clean comparison protocol.

The default proposal is deliberately not an approval.  No model directory is
loaded, downloaded, or used until a caller supplies an explicitly approved
manifest whose declared weight hash matches a local file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from math import isfinite
import os
from pathlib import Path, PurePath
import platform
import re
import subprocess
import time
from typing import Callable, Mapping, Protocol, Sequence, cast

from .archive import atomic_json_write
from .folds import ADAPTATION_EXCLUSIONS, FOLD_IDS, FOLD_SCHEMA_VERSION, FoldManifest, FoldMembership
from .report_extraction import (
    LatencySummary,
    ReportExtractionRun,
    ReportCorpus,
    ReportPartition,
    ReportRecord,
    ReportSourceError,
    TargetReportMetrics,
    validate_fold_report_boundaries,
)
from .report_labels import (
    LabelState,
    RuleOrigin,
    RuleOriginKind,
    TargetLabel,
    target_definitions,
    validate_rule_origin_for_folds,
)
from .targets import TARGET_COLUMNS


LOCAL_MODEL_APPROVAL_SCHEMA_VERSION = "rsna-knee-local-model-approval-v1"
LOCAL_MODEL_PROMPT_SCHEMA_VERSION = "rsna-knee-local-model-prompt-v1"
LOCAL_MODEL_OUTPUT_SCHEMA_VERSION = "rsna-knee-local-model-output-v1"
LOCAL_MODEL_EXTRACTION_SCHEMA_VERSION = "rsna-knee-local-model-extraction-v1"
LOCAL_MODEL_COMPARISON_SCHEMA_VERSION = "rsna-knee-local-model-comparison-v1"
LOCAL_MODEL_EXTRACTOR_VERSION = "approved-local-language-model-v1"
LOCAL_MODEL_PROMPT_VERSION = "flan-t5-report-json-v1"
LOCAL_MODEL_APPROVAL_DATE = "2026-09-14"

FLAN_T5_SMALL_MODEL_ID = "google/flan-t5-small"
FLAN_T5_SMALL_MODEL_REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
FLAN_T5_SMALL_MODEL_SOURCE = "https://huggingface.co/google/flan-t5-small"
FLAN_T5_SMALL_LICENSE = "Apache-2.0"
FLAN_T5_SMALL_WEIGHTS_PATH = "model.safetensors"
FLAN_T5_SMALL_WEIGHTS_SHA256 = "495fa51e204676f1a857a9fc13c4c89f3f5ba9f480b898cebca02add25e6d749"
FLAN_T5_SMALL_SUPPORTING_ARTIFACT_HASHES: tuple[tuple[str, str], ...] = (
    ("config.json", "439aa0fecf5a5546a1def68b1fc45e538e2c94528ce805378daf091e2bf6e4de"),
    (
        "generation_config.json",
        "f5a1c7e2be8092018d8835128987edf0111637dd98e90599cc80310fef75d95a",
    ),
    (
        "special_tokens_map.json",
        "5c87151ef0f72a99d1f766a4c418bd2a1f90aaa30a8e22fe5eca9641daebb64f",
    ),
    ("spiece.model", "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86"),
    ("tokenizer.json", "fe2ebbbbde2985be723e0ce18217853e4020c5e9d35bd07be2c27ab9d3ead57a"),
    (
        "tokenizer_config.json",
        "fcde0f79bffda3688119c94330866a8fbf8de20ae65a8c492c9bd47c704655a0",
    ),
)

LOCAL_MODEL_DATA_TRANSFER_POLICY = "local-only-no-report-transfer"
LOCAL_MODEL_EXECUTION_LOCATION = "cluster1-cpu"
LOCAL_MODEL_MAX_CPUS = 16
LOCAL_MODEL_MAX_MEMORY_GB = 64
LOCAL_MODEL_MAX_WALL_TIME_HOURS = 6
LOCAL_MODEL_MAX_EVIDENCE_CHARS = 512

LOCAL_MODEL_CODE_FINGERPRINT = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class LocalReportExtractionError(ValueError):
    """Raised when the ticket-16 local extraction boundary is unsafe."""


class LocalModelApprovalError(LocalReportExtractionError):
    """Raised when an exact owner-approved model resource is unavailable."""


class LocalModelArtifactError(LocalModelApprovalError):
    """Raised when a local model artifact does not match its approval."""


class LocalModelOutputError(LocalReportExtractionError):
    """Raised for malformed local-model output or incompatible output fields."""


class LocalModelDependencyError(LocalReportExtractionError):
    """Raised when the optional local-model runtime is not installed."""


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LocalModelApprovalError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_sha256(value: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LocalModelApprovalError(f"{field_name} must be a lowercase SHA-256 digest")


def _validate_date(value: str, field_name: str) -> None:
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise LocalModelApprovalError(f"{field_name} must be an ISO date") from exc


def _validate_relative_path(value: str, field_name: str) -> None:
    path = PurePath(value)
    if path.is_absolute() or ".." in path.parts or not value.strip():
        raise LocalModelApprovalError(f"{field_name} must be a relative path inside the model directory")


def _int_or_invalid(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else -1


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LocalModelApproval:
    """Exact model/resource proposal, optionally promoted by owner approval.

    ``artifact_sha256`` names one required weight file inside a local model
    directory.  The model directory is never resolved through a model hub or
    another network service by this package.
    """

    approval_id: str
    model_id: str
    model_version: str
    model_revision: str
    source_url: str
    license: str
    artifact_relative_path: str
    artifact_sha256: str
    public_availability_cutoff: str
    publicly_available_before_cutoff: bool
    eligibility_assessment: str
    intended_purpose: str
    leakage_assessment: str
    execution_location: str
    data_transfer_policy: str
    max_cpus: int
    max_memory_gb: int
    max_wall_time_hours: int
    expected_benefit: str = (
        "Quantify whether the approved local model improves report-state coverage, accuracy, "
        "auditability or latency over the deterministic baseline; no benefit is assumed."
    )
    reproducibility_method: str = (
        "Repeat the same immutable model artifact, frozen prompt, fold manifest and private "
        "report-record digest, then compare text-free semantic output digests."
    )
    deterministic_comparator: str = (
        "Ticket-15 frozen deterministic report extractor evaluated on the identical fold and "
        "report cohorts."
    )
    supporting_artifact_hashes: tuple[tuple[str, str], ...] = ()
    device: str = "cpu"
    approved: bool = False
    approval_reference: str | None = None
    approved_by: str | None = None
    approved_on: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "approval_id",
            "model_id",
            "model_version",
            "model_revision",
            "source_url",
            "license",
            "artifact_relative_path",
            "public_availability_cutoff",
            "eligibility_assessment",
            "intended_purpose",
            "leakage_assessment",
            "execution_location",
            "data_transfer_policy",
            "expected_benefit",
            "reproducibility_method",
            "deterministic_comparator",
            "device",
        ):
            _require_text(getattr(self, field_name), field_name)
        _validate_sha256(self.artifact_sha256, "artifact_sha256")
        _validate_date(self.public_availability_cutoff, "public_availability_cutoff")
        if not isinstance(self.publicly_available_before_cutoff, bool):
            raise LocalModelApprovalError("publicly_available_before_cutoff must be boolean")
        if self.model_revision != self.model_revision.strip():
            raise LocalModelApprovalError("model_revision must not contain surrounding whitespace")
        if re.fullmatch(r"[0-9a-f]{7,64}", self.model_revision) is None:
            raise LocalModelApprovalError(
                "model_revision must be an immutable hexadecimal model revision"
            )
        _validate_relative_path(self.artifact_relative_path, "artifact_relative_path")
        supporting_paths: list[str] = []
        for relative_path, digest in self.supporting_artifact_hashes:
            if not isinstance(relative_path, str) or not relative_path.strip():
                raise LocalModelApprovalError(
                    "supporting_artifact_hashes paths must be non-empty strings"
                )
            _validate_relative_path(relative_path, "supporting_artifact_hashes path")
            if relative_path == self.artifact_relative_path:
                raise LocalModelApprovalError(
                    "supporting_artifact_hashes must not repeat the primary artifact"
                )
            _validate_sha256(digest, f"supporting artifact {relative_path}")
            supporting_paths.append(relative_path)
        if len(set(supporting_paths)) != len(supporting_paths):
            raise LocalModelApprovalError("supporting_artifact_hashes paths must be unique")
        if tuple(sorted(supporting_paths)) != tuple(supporting_paths):
            raise LocalModelApprovalError("supporting_artifact_hashes paths must be sorted")
        for field_name in ("max_cpus", "max_memory_gb", "max_wall_time_hours"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise LocalModelApprovalError(f"{field_name} must be a positive integer")
        if self.max_cpus > LOCAL_MODEL_MAX_CPUS:
            raise LocalModelApprovalError(f"max_cpus cannot exceed {LOCAL_MODEL_MAX_CPUS}")
        if self.max_memory_gb > LOCAL_MODEL_MAX_MEMORY_GB:
            raise LocalModelApprovalError(f"max_memory_gb cannot exceed {LOCAL_MODEL_MAX_MEMORY_GB}")
        if self.max_wall_time_hours > LOCAL_MODEL_MAX_WALL_TIME_HOURS:
            raise LocalModelApprovalError(
                f"max_wall_time_hours cannot exceed {LOCAL_MODEL_MAX_WALL_TIME_HOURS}"
            )
        if self.device != "cpu":
            raise LocalModelApprovalError(
                "ticket-16 local model execution is CPU-only; GPU use needs a separate approval"
            )
        if self.execution_location != LOCAL_MODEL_EXECUTION_LOCATION:
            raise LocalModelApprovalError(
                f"local model execution must use {LOCAL_MODEL_EXECUTION_LOCATION}"
            )
        if self.data_transfer_policy != LOCAL_MODEL_DATA_TRANSFER_POLICY:
            raise LocalModelApprovalError(
                "ticket-16 requires local-only report processing with no report transfer"
            )
        if not isinstance(self.approved, bool):
            raise LocalModelApprovalError("approved must be boolean")
        for field_name in ("approval_reference", "approved_by", "approved_on"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise LocalModelApprovalError(f"{field_name} must be non-empty when supplied")
        if self.approved:
            if self.approval_reference is None or self.approved_by is None or self.approved_on is None:
                raise LocalModelApprovalError(
                    "an approved local model requires approval_reference, approved_by and approved_on"
                )
            _validate_date(self.approved_on, "approved_on")

    @property
    def status(self) -> str:
        return "approved" if self.approved else "proposed"

    @property
    def fingerprint(self) -> str:
        """Hash the exact model identity, weights and execution boundary."""

        return _json_sha256(self.identity_dict())

    def identity_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_revision": self.model_revision,
            "source_url": self.source_url,
            "license": self.license,
            "artifact_relative_path": self.artifact_relative_path,
            "artifact_sha256": self.artifact_sha256,
            "public_availability_cutoff": self.public_availability_cutoff,
            "publicly_available_before_cutoff": self.publicly_available_before_cutoff,
            "eligibility_assessment": self.eligibility_assessment,
            "intended_purpose": self.intended_purpose,
            "leakage_assessment": self.leakage_assessment,
            "expected_benefit": self.expected_benefit,
            "reproducibility_method": self.reproducibility_method,
            "deterministic_comparator": self.deterministic_comparator,
            "execution_location": self.execution_location,
            "data_transfer_policy": self.data_transfer_policy,
            "max_cpus": self.max_cpus,
            "max_memory_gb": self.max_memory_gb,
            "max_wall_time_hours": self.max_wall_time_hours,
            "device": self.device,
            "supporting_artifact_hashes": {
                relative_path: digest
                for relative_path, digest in self.supporting_artifact_hashes
            },
        }

    def to_public_dict(self) -> dict[str, object]:
        """Serialize approval/proposal metadata without a local path or report data."""

        return {
            "schema_version": LOCAL_MODEL_APPROVAL_SCHEMA_VERSION,
            "approval_id": self.approval_id,
            "status": self.status,
            "approved": self.approved,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_revision": self.model_revision,
            "source_url": self.source_url,
            "license": self.license,
            "artifact_relative_path": self.artifact_relative_path,
            "artifact_sha256": self.artifact_sha256,
            "supporting_artifact_hashes": {
                relative_path: digest
                for relative_path, digest in self.supporting_artifact_hashes
            },
            "public_availability_cutoff": self.public_availability_cutoff,
            "publicly_available_before_cutoff": self.publicly_available_before_cutoff,
            "eligibility_assessment": self.eligibility_assessment,
            "intended_purpose": self.intended_purpose,
            "leakage_assessment": self.leakage_assessment,
            "expected_benefit": self.expected_benefit,
            "reproducibility_method": self.reproducibility_method,
            "deterministic_comparator": self.deterministic_comparator,
            "execution_location": self.execution_location,
            "data_transfer_policy": self.data_transfer_policy,
            "compute": {
                "max_cpus": self.max_cpus,
                "max_memory_gb": self.max_memory_gb,
                "max_wall_time_hours": self.max_wall_time_hours,
                "device": self.device,
            },
            "fingerprint": self.fingerprint,
            "approval_reference_private": self.approval_reference is not None,
            "approved_by_private": self.approved_by is not None,
            "approved_on": self.approved_on,
        }

    def to_private_dict(self) -> dict[str, object]:
        result = self.to_public_dict()
        result["approval_reference"] = self.approval_reference
        result["approved_by"] = self.approved_by
        return result

    def require_approved(self) -> None:
        if not self.approved:
            raise LocalModelApprovalError(
                f"model {self.model_id!r} is not approved; owner approval is required before use"
            )
        if not self.publicly_available_before_cutoff:
            raise LocalModelApprovalError(
                "model public availability was not established before the declared cutoff"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LocalModelApproval":
        if payload.get("schema_version") != LOCAL_MODEL_APPROVAL_SCHEMA_VERSION:
            raise LocalModelApprovalError("local model approval has an unexpected schema version")
        compute = payload.get("compute")
        if not isinstance(compute, Mapping):
            raise LocalModelApprovalError("local model approval compute must be an object")
        status = payload.get("status")
        approved = payload.get("approved", status == "approved")
        if not isinstance(approved, bool):
            raise LocalModelApprovalError("local model approval approved must be boolean")
        expected_status = "approved" if approved else "proposed"
        if status is not None and status != expected_status:
            raise LocalModelApprovalError("local model approval status disagrees with approved")
        public_before_cutoff = payload.get("publicly_available_before_cutoff")
        if not isinstance(public_before_cutoff, bool):
            raise LocalModelApprovalError(
                "publicly_available_before_cutoff must be boolean"
            )
        raw_supporting_hashes = payload.get("supporting_artifact_hashes", {})
        if not isinstance(raw_supporting_hashes, Mapping):
            raise LocalModelApprovalError("supporting_artifact_hashes must be an object")
        supporting_hashes = tuple(
            sorted(
                (
                    _require_text(relative_path, "supporting_artifact_hashes path"),
                    _require_text(digest, f"supporting artifact {relative_path}"),
                )
                for relative_path, digest in raw_supporting_hashes.items()
            )
        )
        return cls(
            approval_id=_require_text(payload.get("approval_id"), "approval_id"),
            model_id=_require_text(payload.get("model_id"), "model_id"),
            model_version=_require_text(payload.get("model_version"), "model_version"),
            model_revision=_require_text(payload.get("model_revision"), "model_revision"),
            source_url=_require_text(payload.get("source_url"), "source_url"),
            license=_require_text(payload.get("license"), "license"),
            artifact_relative_path=_require_text(
                payload.get("artifact_relative_path"), "artifact_relative_path"
            ),
            artifact_sha256=_require_text(payload.get("artifact_sha256"), "artifact_sha256"),
            public_availability_cutoff=_require_text(
                payload.get("public_availability_cutoff"), "public_availability_cutoff"
            ),
            publicly_available_before_cutoff=public_before_cutoff,
            eligibility_assessment=_require_text(
                payload.get("eligibility_assessment"), "eligibility_assessment"
            ),
            intended_purpose=_require_text(payload.get("intended_purpose"), "intended_purpose"),
            leakage_assessment=_require_text(
                payload.get("leakage_assessment"), "leakage_assessment"
            ),
            expected_benefit=_require_text(payload.get("expected_benefit"), "expected_benefit"),
            reproducibility_method=_require_text(
                payload.get("reproducibility_method"), "reproducibility_method"
            ),
            deterministic_comparator=_require_text(
                payload.get("deterministic_comparator"), "deterministic_comparator"
            ),
            supporting_artifact_hashes=supporting_hashes,
            execution_location=_require_text(
                payload.get("execution_location"), "execution_location"
            ),
            data_transfer_policy=_require_text(
                payload.get("data_transfer_policy"), "data_transfer_policy"
            ),
            max_cpus=_int_or_invalid(compute.get("max_cpus")),
            max_memory_gb=_int_or_invalid(compute.get("max_memory_gb")),
            max_wall_time_hours=_int_or_invalid(compute.get("max_wall_time_hours")),
            device=_require_text(compute.get("device"), "compute.device"),
            approved=approved,
            approval_reference=_optional_text(payload.get("approval_reference")),
            approved_by=_optional_text(payload.get("approved_by")),
            approved_on=_optional_text(payload.get("approved_on")),
        )


@dataclass(frozen=True)
class LocalModelArtifactVerification:
    """Evidence that one local weight file matches the approved SHA-256."""

    artifact_path: Path
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if self.artifact_path.is_symlink() or not self.artifact_path.is_file():
            raise LocalModelArtifactError("verified model artifact path is not a file")
        _validate_sha256(self.sha256, "verified artifact sha256")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise LocalModelArtifactError("verified artifact size must be non-negative")


def verify_local_model_artifact(
    approval: LocalModelApproval,
    model_directory: str | Path,
) -> LocalModelArtifactVerification:
    """Verify an already-present model file without performing any network I/O."""

    if not isinstance(approval, LocalModelApproval):
        raise LocalModelApprovalError("model artifact verification requires LocalModelApproval")
    approval.require_approved()
    root = Path(model_directory)
    if not root.is_dir() or root.is_symlink():
        raise LocalModelArtifactError("approved local model directory is missing or is a symlink")
    root_resolved = root.resolve()

    def resolve_artifact(relative_name: str) -> Path:
        relative_path = PurePath(relative_name)
        current = root
        for part in relative_path.parts:
            current /= part
            if current.is_symlink():
                raise LocalModelArtifactError(
                    "approved model artifact paths cannot contain symlinks"
                )
        candidate_unresolved = root / relative_path
        candidate = candidate_unresolved.resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError as exc:
            raise LocalModelArtifactError(
                f"model artifact escapes the model directory: {relative_name}"
            ) from exc
        if not candidate_unresolved.is_file() or candidate_unresolved.is_symlink():
            raise LocalModelArtifactError(
                f"approved model artifact file is missing: {relative_name}"
            )
        return candidate

    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    primary = resolve_artifact(approval.artifact_relative_path)
    actual = file_sha256(primary)
    if actual != approval.artifact_sha256:
        raise LocalModelArtifactError(
            "local model artifact SHA-256 does not match the approved manifest"
        )
    for relative_name, expected_digest in approval.supporting_artifact_hashes:
        supporting_path = resolve_artifact(relative_name)
        if file_sha256(supporting_path) != expected_digest:
            raise LocalModelArtifactError(
                f"supporting local model artifact SHA-256 does not match: {relative_name}"
            )
    return LocalModelArtifactVerification(primary, actual, primary.stat().st_size)


def proposed_flan_t5_small_approval() -> LocalModelApproval:
    """Return the exact ticket-16 proposal; it is intentionally unapproved."""

    return LocalModelApproval(
        approval_id="ticket16-proposal-flan-t5-small-v1",
        model_id=FLAN_T5_SMALL_MODEL_ID,
        model_version="FLAN-T5-small",
        model_revision=FLAN_T5_SMALL_MODEL_REVISION,
        source_url=FLAN_T5_SMALL_MODEL_SOURCE,
        license=FLAN_T5_SMALL_LICENSE,
        artifact_relative_path=FLAN_T5_SMALL_WEIGHTS_PATH,
        artifact_sha256=FLAN_T5_SMALL_WEIGHTS_SHA256,
        supporting_artifact_hashes=FLAN_T5_SMALL_SUPPORTING_ARTIFACT_HASHES,
        public_availability_cutoff=LOCAL_MODEL_APPROVAL_DATE,
        publicly_available_before_cutoff=True,
        eligibility_assessment=(
            "Proposal only: Apache-2.0 model card and exact public revision recorded; "
            "competition-resource eligibility and any public-weight cutoff must be confirmed by the owner."
        ),
        intended_purpose=(
            "Local research comparison for conservative radiology-report state extraction; "
            "not clinical use and not competition inference."
        ),
        leakage_assessment=(
            "Generic instruction-tuned checkpoint; no competition-report fine-tuning is claimed. "
            "Training-data overlap is not independently established and must remain an explicit limitation."
        ),
        execution_location=LOCAL_MODEL_EXECUTION_LOCATION,
        data_transfer_policy=LOCAL_MODEL_DATA_TRANSFER_POLICY,
        expected_benefit=(
            "Quantify whether this small pretrained model changes coverage, accuracy, "
            "auditability or latency relative to ticket 15; no benefit is assumed."
        ),
        reproducibility_method=(
            "Repeat the exact local artifact, frozen prompt, fold manifest and private report "
            "record-set digest, comparing text-free semantic output digests."
        ),
        deterministic_comparator=(
            "Ticket-15 frozen deterministic extractor on identical fold and report cohorts."
        ),
        max_cpus=16,
        max_memory_gb=64,
        max_wall_time_hours=6,
        device="cpu",
    )


def load_local_model_approval(path: str | Path) -> LocalModelApproval:
    candidate = Path(path)
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalModelApprovalError(f"could not read local model approval: {candidate}") from exc
    if not isinstance(payload, Mapping):
        raise LocalModelApprovalError("local model approval must be a JSON object")
    return LocalModelApproval.from_dict(cast(Mapping[str, object], payload))


def write_local_model_proposal(
    path: str | Path,
    proposal: LocalModelApproval | None = None,
) -> Path:
    """Write a non-overwriting public proposal for owner review."""

    selected = proposal or proposed_flan_t5_small_approval()
    if selected.approved:
        raise LocalModelApprovalError("write_local_model_proposal accepts proposals, not approvals")
    return atomic_json_write(path, selected.to_public_dict())


DEFAULT_LOCAL_PROMPT_SYSTEM = (
    "You are a conservative radiology report extraction component. "
    "Use only explicit report evidence and the target criteria supplied below. "
    "Do not treat absence of a sentence as a negative finding. "
    "Return JSON only; do not include markdown, commentary, or patient identifiers."
)
DEFAULT_LOCAL_PROMPT_TEMPLATE = """Classify this knee MRI report for the twelve competition targets.

For every target return exactly one state: positive, negative, not-mentioned,
uncertain, unknown, or extraction-failed. Use positive or negative only when
the report explicitly supports the authoritative criterion. Use uncertain for
possible/suspected/equivocal or historical findings. Treat an explicit
negation as negative only when it satisfies the authoritative negative
criterion. Use unknown for unsupported wording, not-mentioned when the target is not discussed, and
extraction-failed only when the report cannot be interpreted.

Return this JSON shape exactly:
{{"schema_version":"{output_schema_version}","targets":[{{"target":"ACL","state":"not-mentioned","confidence":null,"evidence":""}}]}}

Authoritative target criteria:
{target_criteria}

Report text:
{report_text}
"""


def _prompt_target_criteria() -> str:
    return "\n".join(
        f"- {definition.target}: positive if {definition.positive_criterion}; "
        f"negative only if {definition.negative_criterion}"
        for definition in target_definitions()
    )


@dataclass(frozen=True)
class FrozenLocalPrompt:
    """Immutable prompt and generation settings used by every model call."""

    prompt_version: str = LOCAL_MODEL_PROMPT_VERSION
    system_instruction: str = DEFAULT_LOCAL_PROMPT_SYSTEM
    user_template: str = DEFAULT_LOCAL_PROMPT_TEMPLATE
    max_input_chars: int = 16_000
    max_new_tokens: int = 256
    temperature: float = 0.0
    do_sample: bool = False
    output_schema_version: str = LOCAL_MODEL_OUTPUT_SCHEMA_VERSION
    target_order: tuple[str, ...] = TARGET_COLUMNS

    def __post_init__(self) -> None:
        for field_name in (
            "prompt_version",
            "system_instruction",
            "user_template",
            "output_schema_version",
        ):
            _require_text(getattr(self, field_name), field_name)
        if self.target_order != TARGET_COLUMNS:
            raise LocalReportExtractionError("local prompt target order must be canonical")
        for field_name in ("max_input_chars", "max_new_tokens"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise LocalReportExtractionError(f"{field_name} must be a positive integer")
        if self.max_input_chars > 64_000:
            raise LocalReportExtractionError("local prompt input limit is unreasonably large")
        if self.max_new_tokens > 2_048:
            raise LocalReportExtractionError("local prompt output limit is unreasonably large")
        if not isinstance(self.do_sample, bool):
            raise LocalReportExtractionError("do_sample must be boolean")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not isfinite(float(self.temperature))
            or self.temperature < 0
        ):
            raise LocalReportExtractionError("temperature must be a finite non-negative number")
        if self.do_sample:
            if self.temperature <= 0:
                raise LocalReportExtractionError("sampled local generation needs positive temperature")
        elif self.temperature != 0.0:
            raise LocalReportExtractionError("frozen greedy local generation must use temperature 0")

    @property
    def is_frozen(self) -> bool:
        return True

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {
                "schema_version": LOCAL_MODEL_PROMPT_SCHEMA_VERSION,
                "prompt_version": self.prompt_version,
                "system_instruction": self.system_instruction,
                "user_template": self.user_template,
                "target_criteria": _prompt_target_criteria(),
                "max_input_chars": self.max_input_chars,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "do_sample": self.do_sample,
                "output_schema_version": self.output_schema_version,
                "target_order": list(self.target_order),
            }
        )

    def render(self, report_text: str) -> str:
        if not isinstance(report_text, str):
            raise LocalReportExtractionError("local prompt report text must be a string")
        if len(report_text) > self.max_input_chars:
            raise LocalReportExtractionError("report exceeds the frozen local prompt input limit")
        return self.system_instruction + "\n\n" + self.user_template.format(
            output_schema_version=self.output_schema_version,
            target_criteria=_prompt_target_criteria(),
            report_text=report_text,
        )

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": LOCAL_MODEL_PROMPT_SCHEMA_VERSION,
            "prompt_version": self.prompt_version,
            "system_instruction": self.system_instruction,
            "user_template": self.user_template,
            "target_criteria": _prompt_target_criteria(),
            "max_input_chars": self.max_input_chars,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
            "output_schema_version": self.output_schema_version,
            "target_order": list(self.target_order),
            "fingerprint": self.fingerprint,
            "frozen": True,
        }


def freeze_local_prompt() -> FrozenLocalPrompt:
    """Return the sole prompt admitted by the ticket-16 comparison protocol."""

    return FrozenLocalPrompt()


@dataclass(frozen=True)
class LocalModelGeneration:
    """One local generation response and optional tokenizer accounting."""

    text: str
    input_token_count: int | None = None
    output_token_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise LocalModelOutputError("local model generation text must be a string")
        for field_name in ("input_token_count", "output_token_count"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise LocalModelOutputError(f"{field_name} must be null or non-negative")


class LocalTextGenerator(Protocol):
    """Boundary for a text generator; implementations must be local-only."""

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> LocalModelGeneration | str:
        """Generate one response without sending the prompt outside the process."""


@dataclass(frozen=True)
class CallableLocalTextGenerator:
    """Small adapter useful for public synthetic tests and local wrappers."""

    function: Callable[[str], LocalModelGeneration | str]

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> LocalModelGeneration | str:
        del max_new_tokens, do_sample, temperature
        return self.function(prompt)


def _fold_is_valid(fold: int) -> bool:
    return isinstance(fold, int) and not isinstance(fold, bool) and fold in FOLD_IDS


def _local_execution_identity() -> tuple[str, str]:
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    partition = os.environ.get("SLURM_JOB_PARTITION", "").strip()
    if job_id:
        return "slurm", f"slurm:{partition or 'unknown'}:{job_id}"
    return "local", "local-process"


def _local_model_origin(approval: LocalModelApproval) -> RuleOrigin:
    return RuleOrigin(
        rule_id=f"local-model-{approval.model_id}-{approval.model_revision[:12]}",
        kind=RuleOriginKind.APPROVED_PRETRAINED_MODEL,
        source_id=approval.approval_id,
        source_url=approval.source_url,
        retrieved_on=approval.approved_on or approval.public_availability_cutoff,
        shared_across_folds=True,
    )


def _local_output_policy_origin() -> RuleOrigin:
    return RuleOrigin(
        rule_id="local-model-output-validation-v1",
        kind=RuleOriginKind.PROJECT_SAFETY_POLICY,
        source_id="ticket16-local-model-output-policy-v1",
        source_url=".scratch/rsna-2026-fresh/spec.md",
        retrieved_on=LOCAL_MODEL_APPROVAL_DATE,
        shared_across_folds=True,
    )


def _local_label(
    target: str,
    state: LabelState,
    *,
    confidence: float | None,
    evidence: tuple[str, ...],
    origin: RuleOrigin,
    rule_id_prefix: str = "local-model-output",
) -> TargetLabel:
    return TargetLabel(
        target=target,
        state=state,
        rule_id=f"{rule_id_prefix}-{state.value}",
        rule_origin=origin,
        confidence=confidence,
        evidence=evidence,
    )


@dataclass(frozen=True)
class LocalReportInterpretation:
    """A model-produced, text-free twelve-target report interpretation."""

    model_fingerprint: str
    prompt_fingerprint: str
    report_sha256: str
    labels: tuple[TargetLabel, ...]
    source_report_ref: str
    consuming_folds: tuple[int, ...]
    response_sha256: str | None
    prompt_char_count: int
    response_char_count: int
    input_token_count: int | None = None
    output_token_count: int | None = None
    failure_code: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("model_fingerprint", "prompt_fingerprint", "report_sha256"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise LocalReportExtractionError(f"{field_name} must be non-empty")
            if field_name != "report_sha256":
                _validate_sha256(value, field_name)
        _validate_sha256(self.report_sha256, "report_sha256")
        if self.response_sha256 is not None:
            _validate_sha256(self.response_sha256, "response_sha256")
        if tuple(item.target for item in self.labels) != TARGET_COLUMNS:
            raise LocalModelOutputError("local interpretation labels must use competition order")
        if len(self.consuming_folds) != 1 or not _fold_is_valid(self.consuming_folds[0]):
            raise LocalReportExtractionError("local interpretation must name exactly one valid fold")
        if not isinstance(self.source_report_ref, str) or not self.source_report_ref.strip():
            raise LocalReportExtractionError("source_report_ref must be non-empty")
        for field_name in ("prompt_char_count", "response_char_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LocalReportExtractionError(f"{field_name} must be non-negative")
        for field_name in ("input_token_count", "output_token_count"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise LocalReportExtractionError(f"{field_name} must be null or non-negative")
        if self.failure_code is not None and (
            not isinstance(self.failure_code, str) or not self.failure_code.strip()
        ):
            raise LocalReportExtractionError("failure_code must be non-empty when supplied")
        for label in self.labels:
            validate_rule_origin_for_folds(label.rule_origin, self.consuming_folds)
            if label.state in {LabelState.POSITIVE, LabelState.NEGATIVE}:
                if label.confidence is None:
                    raise LocalModelOutputError("confident local labels require confidence")
            elif label.confidence is not None:
                raise LocalModelOutputError("abstained local labels cannot carry confidence")

    @property
    def extractor_version(self) -> str:
        return LOCAL_MODEL_EXTRACTOR_VERSION

    @property
    def values(self) -> tuple[int | None, ...]:
        return tuple(item.value for item in self.labels)

    @property
    def supervision_mask(self) -> tuple[int, ...]:
        return tuple(item.supervision_mask for item in self.labels)

    @property
    def state_counts(self) -> dict[str, int]:
        counts = {state.value: 0 for state in LabelState}
        for label in self.labels:
            counts[label.state.value] += 1
        return counts

    @property
    def supervised_entry_count(self) -> int:
        return sum(self.supervision_mask)

    @property
    def auditable_entry_count(self) -> int:
        return sum(bool(label.evidence) for label in self.labels)

    def semantic_dict(self) -> dict[str, object]:
        return {
            "model_fingerprint": self.model_fingerprint,
            "prompt_fingerprint": self.prompt_fingerprint,
            "report_sha256": self.report_sha256,
            "source_report_ref": self.source_report_ref,
            "consuming_folds": list(self.consuming_folds),
            "response_sha256": self.response_sha256,
            "prompt_char_count": self.prompt_char_count,
            "response_char_count": self.response_char_count,
            "input_token_count": self.input_token_count,
            "output_token_count": self.output_token_count,
            "failure_code": self.failure_code,
            "labels": [label.to_private_dict() for label in self.labels],
        }

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": LOCAL_MODEL_EXTRACTION_SCHEMA_VERSION,
            "extractor_version": self.extractor_version,
            "model_fingerprint": self.model_fingerprint,
            "prompt_fingerprint": self.prompt_fingerprint,
            "fold": self.consuming_folds[0],
            "target_order": list(TARGET_COLUMNS),
            "states": [label.state.value for label in self.labels],
            "values": list(self.values),
            "supervision_mask": list(self.supervision_mask),
            "rule_ids": [label.rule_id for label in self.labels],
            "response_char_count": self.response_char_count,
            "prompt_char_count": self.prompt_char_count,
            "input_token_count": self.input_token_count,
            "output_token_count": self.output_token_count,
            "failure_code": self.failure_code,
            "auditable_entry_count": self.auditable_entry_count,
        }

    def to_private_dict(self) -> dict[str, object]:
        result = self.to_public_dict()
        result["source_report_ref"] = self.source_report_ref
        result["report_sha256"] = self.report_sha256
        result["response_sha256"] = self.response_sha256
        result["labels"] = [label.to_private_dict() for label in self.labels]
        return result


@dataclass(frozen=True)
class LocalReportLabelRecord:
    """One fold-scoped local-model interpretation and its artifact partition."""

    study_instance_uid: str
    fold: int
    partition: ReportPartition
    interpretation: LocalReportInterpretation

    def __post_init__(self) -> None:
        if not isinstance(self.study_instance_uid, str) or not self.study_instance_uid.strip():
            raise LocalReportExtractionError("local label study_instance_uid must be non-empty")
        if not _fold_is_valid(self.fold):
            raise LocalReportExtractionError("local label fold is invalid")
        if not isinstance(self.partition, ReportPartition):
            try:
                object.__setattr__(self, "partition", ReportPartition(str(self.partition)))
            except ValueError as exc:
                raise LocalReportExtractionError("unknown local label partition") from exc
        if not isinstance(self.interpretation, LocalReportInterpretation):
            raise LocalReportExtractionError("local label requires LocalReportInterpretation")
        if self.interpretation.consuming_folds != (self.fold,):
            raise LocalReportExtractionError("local label interpretation must name its consuming fold")

    @property
    def source_report_ref(self) -> str:
        return self.interpretation.source_report_ref

    @property
    def report_sha256(self) -> str:
        return self.interpretation.report_sha256

    @property
    def values(self) -> tuple[int | None, ...]:
        return self.interpretation.values

    @property
    def supervision_mask(self) -> tuple[int, ...]:
        return self.interpretation.supervision_mask

    @property
    def supervised_entry_count(self) -> int:
        return self.interpretation.supervised_entry_count

    @property
    def auditable_entry_count(self) -> int:
        return self.interpretation.auditable_entry_count

    def semantic_dict(self) -> dict[str, object]:
        return {
            "study_instance_uid": self.study_instance_uid,
            "fold": self.fold,
            "partition": self.partition.value,
            "interpretation": self.interpretation.semantic_dict(),
        }

    def to_public_dict(self) -> dict[str, object]:
        result = self.interpretation.to_public_dict()
        result["partition"] = self.partition.value
        return result

    def to_private_dict(self) -> dict[str, object]:
        result = self.interpretation.to_private_dict()
        result["artifact_kind"] = self.partition.value
        result["study_instance_uid"] = self.study_instance_uid
        return result


def _failed_local_interpretation(
    report: ReportRecord,
    *,
    fold: int,
    model_fingerprint: str,
    prompt_fingerprint: str,
    prompt_char_count: int,
    response_text: str = "",
    input_token_count: int | None = None,
    output_token_count: int | None = None,
    failure_code: str,
) -> LocalReportInterpretation:
    policy_origin = _local_output_policy_origin()
    return LocalReportInterpretation(
        model_fingerprint=model_fingerprint,
        prompt_fingerprint=prompt_fingerprint,
        report_sha256=report.report_sha256,
        labels=tuple(
            _local_label(
                target,
                LabelState.EXTRACTION_FAILED,
                confidence=None,
                evidence=(),
                origin=policy_origin,
            )
            for target in TARGET_COLUMNS
        ),
        source_report_ref=report.source_report_ref,
        consuming_folds=(fold,),
        response_sha256=hashlib.sha256(response_text.encode("utf-8")).hexdigest()
        if response_text
        else None,
        prompt_char_count=prompt_char_count,
        response_char_count=len(response_text),
        input_token_count=input_token_count,
        output_token_count=output_token_count,
        failure_code=failure_code,
    )


def _normalized_state(value: object) -> LabelState:
    if not isinstance(value, str):
        raise LocalModelOutputError("model state must be a string")
    normalized = value.strip().casefold().replace("_", "-")
    try:
        return LabelState(normalized)
    except ValueError as exc:
        raise LocalModelOutputError(f"unsupported model state {value!r}") from exc


def _model_evidence(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, str):
        raise LocalModelOutputError("model evidence must be a string")
    evidence = value.strip()
    if "\n" in evidence or "\r" in evidence:
        raise LocalModelOutputError("model evidence must be a single line")
    if len(evidence) > LOCAL_MODEL_MAX_EVIDENCE_CHARS:
        raise LocalModelOutputError("model evidence exceeds the audit limit")
    return (evidence,) if evidence else ()


def _model_confidence(value: object, state: LabelState) -> float | None:
    if state in {LabelState.POSITIVE, LabelState.NEGATIVE}:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise LocalModelOutputError("positive/negative model labels require confidence in [0, 1]")
        return float(value)
    if value is not None:
        raise LocalModelOutputError("abstained model labels must use null confidence")
    return None


def _target_payloads(raw_targets: object) -> dict[str, Mapping[str, object]]:
    if isinstance(raw_targets, Mapping):
        result: dict[str, Mapping[str, object]] = {}
        for raw_target, raw_value in raw_targets.items():
            if not isinstance(raw_target, str) or raw_target not in TARGET_COLUMNS:
                raise LocalModelOutputError("model output contains an unknown target")
            if not isinstance(raw_value, Mapping):
                raise LocalModelOutputError(f"model target {raw_target!r} must be an object")
            result[raw_target] = cast(Mapping[str, object], raw_value)
    elif isinstance(raw_targets, list):
        result = {}
        for raw_value in raw_targets:
            if not isinstance(raw_value, Mapping):
                raise LocalModelOutputError("model target entries must be objects")
            raw_target = raw_value.get("target")
            if not isinstance(raw_target, str) or raw_target not in TARGET_COLUMNS:
                raise LocalModelOutputError("model output contains an unknown target")
            if raw_target in result:
                raise LocalModelOutputError(f"model output repeats target {raw_target!r}")
            result[raw_target] = cast(Mapping[str, object], raw_value)
    else:
        raise LocalModelOutputError("model output targets must be an object or array")
    if set(result) != set(TARGET_COLUMNS):
        raise LocalModelOutputError("model output must contain every competition target exactly once")
    return result


def parse_local_model_output(
    output: LocalModelGeneration | str,
    *,
    report: ReportRecord,
    fold: int,
    approval: LocalModelApproval,
    model_fingerprint: str,
    prompt_fingerprint: str,
    prompt_char_count: int,
) -> LocalReportInterpretation:
    """Parse one strict JSON response, masking malformed target entries safely."""

    if not _fold_is_valid(fold):
        raise LocalReportExtractionError(f"fold must be one of {FOLD_IDS}")
    if not isinstance(report, ReportRecord):
        raise LocalReportExtractionError("local model output requires a ReportRecord")
    approval.require_approved()
    generated = output if isinstance(output, LocalModelGeneration) else LocalModelGeneration(output)
    response_text = generated.text
    try:
        payload = json.loads(response_text)
        if not isinstance(payload, Mapping):
            raise LocalModelOutputError("model output must be a JSON object")
        if set(payload) != {"schema_version", "targets"}:
            raise LocalModelOutputError("model output must contain exactly schema_version and targets")
        declared_schema = payload.get("schema_version")
        if declared_schema != LOCAL_MODEL_OUTPUT_SCHEMA_VERSION:
            raise LocalModelOutputError("model output has an unexpected schema version")
        targets = _target_payloads(payload.get("targets"))
    except json.JSONDecodeError:
        code = "invalid-json"
        return _failed_local_interpretation(
            report,
            fold=fold,
            model_fingerprint=model_fingerprint,
            prompt_fingerprint=prompt_fingerprint,
            prompt_char_count=prompt_char_count,
            response_text=response_text,
            input_token_count=generated.input_token_count,
            output_token_count=generated.output_token_count,
            failure_code=code,
        )
    except LocalModelOutputError as exc:
        message = str(exc)
        code = (
            "invalid-schema"
            if "schema" in message
            else "invalid-target-set"
            if "target" in message
            else "invalid-output"
        )
        return _failed_local_interpretation(
            report,
            fold=fold,
            model_fingerprint=model_fingerprint,
            prompt_fingerprint=prompt_fingerprint,
            prompt_char_count=prompt_char_count,
            response_text=response_text,
            input_token_count=generated.input_token_count,
            output_token_count=generated.output_token_count,
            failure_code=code,
        )

    origin = _local_model_origin(approval)
    policy_origin = _local_output_policy_origin()
    labels: list[TargetLabel] = []
    target_failure = False
    model_reported_failure = False
    for target in TARGET_COLUMNS:
        item = targets[target]
        try:
            state = _normalized_state(item.get("state"))
            confidence = _model_confidence(item.get("confidence"), state)
            evidence = _model_evidence(item.get("evidence"))
            if "value" in item:
                declared_value = item.get("value")
                expected_value = 1 if state is LabelState.POSITIVE else 0 if state is LabelState.NEGATIVE else None
                if declared_value != expected_value:
                    raise LocalModelOutputError("model value disagrees with state")
            if state is LabelState.EXTRACTION_FAILED:
                model_reported_failure = True
            labels.append(
                _local_label(
                    target,
                    state,
                    confidence=confidence,
                    evidence=evidence,
                    origin=origin,
                )
            )
        except LocalModelOutputError:
            target_failure = True
            labels.append(
                _local_label(
                    target,
                    LabelState.EXTRACTION_FAILED,
                    confidence=None,
                    evidence=(),
                    origin=policy_origin,
                    rule_id_prefix="local-model-output-invalid",
                )
            )
    return LocalReportInterpretation(
        model_fingerprint=model_fingerprint,
        prompt_fingerprint=prompt_fingerprint,
        report_sha256=report.report_sha256,
        labels=tuple(labels),
        source_report_ref=report.source_report_ref,
        consuming_folds=(fold,),
        response_sha256=hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
        prompt_char_count=prompt_char_count,
        response_char_count=len(response_text),
        input_token_count=generated.input_token_count,
        output_token_count=generated.output_token_count,
        failure_code=(
            "invalid-target-output"
            if target_failure
            else "model-reported-failure"
            if model_reported_failure
            else None
        ),
    )


@dataclass(frozen=True)
class LocalLanguageModelExtractor:
    """Frozen local model + approval + prompt identity used for ticket 16."""

    approval: LocalModelApproval
    prompt: FrozenLocalPrompt
    generator: LocalTextGenerator
    verified_artifact_sha256: str
    verified_artifact_size_bytes: int = 0
    model_load_elapsed_ns: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.approval, LocalModelApproval):
            raise LocalModelApprovalError("local model extractor requires LocalModelApproval")
        if not isinstance(self.prompt, FrozenLocalPrompt):
            raise LocalReportExtractionError("local model extractor requires FrozenLocalPrompt")
        self.approval.require_approved()
        if not self.prompt.is_frozen:
            raise LocalReportExtractionError("local model extractor requires a frozen prompt")
        if self.verified_artifact_sha256 != self.approval.artifact_sha256:
            raise LocalModelArtifactError("verified artifact does not match model approval")
        _validate_sha256(self.verified_artifact_sha256, "verified_artifact_sha256")
        if not callable(getattr(self.generator, "generate", None)):
            raise LocalReportExtractionError("local model generator must expose generate")
        for field_name in ("verified_artifact_size_bytes", "model_load_elapsed_ns"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LocalReportExtractionError(f"{field_name} must be non-negative")

    @property
    def is_frozen(self) -> bool:
        return True

    @property
    def extractor_version(self) -> str:
        return LOCAL_MODEL_EXTRACTOR_VERSION

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {
                "extractor_version": LOCAL_MODEL_EXTRACTOR_VERSION,
                "approval_fingerprint": self.approval.fingerprint,
                "prompt_fingerprint": self.prompt.fingerprint,
                "verified_artifact_sha256": self.verified_artifact_sha256,
                "code_fingerprint": LOCAL_MODEL_CODE_FINGERPRINT,
            }
        )

    @classmethod
    def from_verified_generator(
        cls,
        approval: LocalModelApproval,
        prompt: FrozenLocalPrompt,
        generator: LocalTextGenerator,
        *,
        artifact_sha256: str,
        artifact_size_bytes: int = 0,
        model_load_elapsed_ns: int = 0,
    ) -> "LocalLanguageModelExtractor":
        """Build an extractor after an external local loader verified the file hash."""

        return cls(
            approval=approval,
            prompt=prompt,
            generator=generator,
            verified_artifact_sha256=artifact_sha256,
            verified_artifact_size_bytes=artifact_size_bytes,
            model_load_elapsed_ns=model_load_elapsed_ns,
        )

    @classmethod
    def from_local_directory(
        cls,
        approval: LocalModelApproval,
        model_directory: str | Path,
        *,
        prompt: FrozenLocalPrompt | None = None,
        device: str = "cpu",
    ) -> "LocalLanguageModelExtractor":
        """Load one approved model directory with the optional runtime offline."""

        if device != "cpu":
            raise LocalModelApprovalError(
                "ticket-16 model loading is CPU-only; GPU execution needs separate approval"
            )
        started = time.perf_counter_ns()
        generator = TransformersLocalTextGenerator(
            approval,
            model_directory,
            device=device,
        )
        return cls(
            approval=approval,
            prompt=prompt or freeze_local_prompt(),
            generator=generator,
            verified_artifact_sha256=generator.verification.sha256,
            verified_artifact_size_bytes=generator.verification.size_bytes,
            model_load_elapsed_ns=time.perf_counter_ns() - started,
        )

    def extract(self, report: ReportRecord, *, fold: int) -> LocalReportInterpretation:
        if not isinstance(report, ReportRecord):
            raise LocalReportExtractionError("local model extractor requires a ReportRecord")
        if not _fold_is_valid(fold):
            raise LocalReportExtractionError(f"fold must be one of {FOLD_IDS}")
        try:
            prompt_text = self.prompt.render(report.report_text)
        except LocalReportExtractionError:
            return _failed_local_interpretation(
                report,
                fold=fold,
                model_fingerprint=self.fingerprint,
                prompt_fingerprint=self.prompt.fingerprint,
                prompt_char_count=0,
                failure_code="input-too-long",
            )
        try:
            generated = self.generator.generate(
                prompt_text,
                max_new_tokens=self.prompt.max_new_tokens,
                do_sample=self.prompt.do_sample,
                temperature=self.prompt.temperature,
            )
            if not isinstance(generated, (str, LocalModelGeneration)):
                raise LocalModelOutputError("local model generator returned an unsupported type")
        except LocalModelOutputError:
            return _failed_local_interpretation(
                report,
                fold=fold,
                model_fingerprint=self.fingerprint,
                prompt_fingerprint=self.prompt.fingerprint,
                prompt_char_count=len(prompt_text),
                failure_code="invalid-generator-output",
            )
        except Exception:
            return _failed_local_interpretation(
                report,
                fold=fold,
                model_fingerprint=self.fingerprint,
                prompt_fingerprint=self.prompt.fingerprint,
                prompt_char_count=len(prompt_text),
                failure_code="inference-error",
            )
        return parse_local_model_output(
            generated,
            report=report,
            fold=fold,
            approval=self.approval,
            model_fingerprint=self.fingerprint,
            prompt_fingerprint=self.prompt.fingerprint,
            prompt_char_count=len(prompt_text),
        )


class TransformersLocalTextGenerator:
    """Offline ``transformers`` adapter for the approved seq2seq checkpoint."""

    def __init__(
        self,
        approval: LocalModelApproval,
        model_directory: str | Path,
        *,
        device: str = "cpu",
    ) -> None:
        if not isinstance(approval, LocalModelApproval):
            raise LocalModelApprovalError(
                "local transformers loading requires LocalModelApproval"
            )
        approval.require_approved()
        if device != "cpu":
            raise LocalModelApprovalError(
                "TransformersLocalTextGenerator supports CPU only in ticket 16"
            )
        self.verification = verify_local_model_artifact(approval, model_directory)
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise LocalModelDependencyError(
                "ticket-16 requires the optional local transformers runtime"
            ) from exc
        root = Path(model_directory)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(root),
                local_files_only=True,
                trust_remote_code=False,
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                str(root),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
            )
            self._torch = torch
            self._model.to(device)
            self._model.eval()
        except Exception as exc:
            raise LocalModelDependencyError(
                "could not load the approved local model with local_files_only=True"
            ) from exc

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> LocalModelGeneration:
        encoded = self._tokenizer(prompt, return_tensors="pt", truncation=False)
        input_token_count = int(encoded["input_ids"].numel())
        with self._torch.inference_mode():
            generation_kwargs: dict[str, object] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                generation_kwargs["temperature"] = temperature
            generated = self._model.generate(**encoded, **generation_kwargs)
        text = self._tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        return LocalModelGeneration(
            text,
            input_token_count=input_token_count,
            output_token_count=int(generated.shape[-1]),
        )


def _count(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LocalReportExtractionError(f"{field_name} must be a non-negative integer")
    return value


def _ratio(value: float | None, field_name: str) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise LocalReportExtractionError(f"{field_name} must be null or within [0, 1]")


@dataclass(frozen=True)
class LocalTargetReportMetrics:
    """Accuracy, coverage, state and auditability counts for one target."""

    target: str
    report_count: int
    gold_positive_count: int
    gold_negative_count: int
    predicted_positive_count: int
    predicted_negative_count: int
    correct_count: int
    accuracy: float | None
    coverage: float | None
    not_mentioned_count: int
    uncertain_count: int
    unknown_count: int
    extraction_failed_count: int
    abstention_count: int
    failure_count: int
    mean_confidence: float | None
    auditable_count: int

    def __post_init__(self) -> None:
        if self.target not in TARGET_COLUMNS:
            raise LocalReportExtractionError(f"unknown local report target {self.target!r}")
        count_fields = (
            "report_count",
            "gold_positive_count",
            "gold_negative_count",
            "predicted_positive_count",
            "predicted_negative_count",
            "correct_count",
            "not_mentioned_count",
            "uncertain_count",
            "unknown_count",
            "extraction_failed_count",
            "abstention_count",
            "failure_count",
            "auditable_count",
        )
        for field_name in count_fields:
            _count(getattr(self, field_name), field_name)
        confident = self.predicted_positive_count + self.predicted_negative_count
        state_total = (
            confident
            + self.not_mentioned_count
            + self.uncertain_count
            + self.unknown_count
            + self.extraction_failed_count
        )
        if state_total != self.report_count:
            raise LocalReportExtractionError(f"{self.target} state counts do not add to report_count")
        if self.gold_positive_count + self.gold_negative_count != self.report_count:
            raise LocalReportExtractionError(f"{self.target} gold counts do not add to report_count")
        if self.correct_count > confident:
            raise LocalReportExtractionError(f"{self.target} correct count exceeds confident count")
        if self.auditable_count > self.report_count:
            raise LocalReportExtractionError(f"{self.target} audit count exceeds report count")
        if self.abstention_count != self.report_count - confident:
            raise LocalReportExtractionError(f"{self.target} abstention count is inconsistent")
        if self.failure_count != self.extraction_failed_count:
            raise LocalReportExtractionError(f"{self.target} failure count is inconsistent")
        _ratio(self.accuracy, f"{self.target} accuracy")
        _ratio(self.coverage, f"{self.target} coverage")
        _ratio(self.mean_confidence, f"{self.target} mean_confidence")
        if (confident == 0) != (self.accuracy is None):
            raise LocalReportExtractionError(f"{self.target} accuracy definedness is inconsistent")
        if (self.report_count == 0) != (self.coverage is None):
            raise LocalReportExtractionError(f"{self.target} coverage definedness is inconsistent")
        if (confident == 0) != (self.mean_confidence is None):
            raise LocalReportExtractionError(f"{self.target} confidence definedness is inconsistent")

    @property
    def confident_count(self) -> int:
        return self.predicted_positive_count + self.predicted_negative_count

    @property
    def abstention_rate(self) -> float | None:
        return self.abstention_count / self.report_count if self.report_count else None

    @property
    def failure_rate(self) -> float | None:
        return self.failure_count / self.report_count if self.report_count else None

    @property
    def auditability_rate(self) -> float | None:
        return self.auditable_count / self.report_count if self.report_count else None

    @property
    def state_counts(self) -> dict[str, int]:
        return {
            LabelState.POSITIVE.value: self.predicted_positive_count,
            LabelState.NEGATIVE.value: self.predicted_negative_count,
            LabelState.NOT_MENTIONED.value: self.not_mentioned_count,
            LabelState.UNCERTAIN.value: self.uncertain_count,
            LabelState.UNKNOWN.value: self.unknown_count,
            LabelState.EXTRACTION_FAILED.value: self.extraction_failed_count,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "report_count": self.report_count,
            "gold_positive_count": self.gold_positive_count,
            "gold_negative_count": self.gold_negative_count,
            "predicted_positive_count": self.predicted_positive_count,
            "predicted_negative_count": self.predicted_negative_count,
            "confident_count": self.confident_count,
            "correct_count": self.correct_count,
            "accuracy": self.accuracy,
            "coverage": self.coverage,
            "abstention_count": self.abstention_count,
            "abstention_rate": self.abstention_rate,
            "failure_count": self.failure_count,
            "failure_rate": self.failure_rate,
            "state_counts": self.state_counts,
            "uncertainty": {
                "not_mentioned": self.not_mentioned_count,
                "uncertain": self.uncertain_count,
                "unknown": self.unknown_count,
                "extraction_failed": self.extraction_failed_count,
            },
            "mean_confidence": self.mean_confidence,
            "auditable_count": self.auditable_count,
            "auditability_rate": self.auditability_rate,
        }


@dataclass(frozen=True)
class LocalFoldReportMetrics:
    """Held-out report metrics for one local-model fold."""

    fold: int
    report_count: int
    missing_report_count: int
    targets: tuple[LocalTargetReportMetrics, ...]

    def __post_init__(self) -> None:
        if not _fold_is_valid(self.fold):
            raise LocalReportExtractionError("local fold metrics use an invalid fold")
        _count(self.report_count, "report_count")
        _count(self.missing_report_count, "missing_report_count")
        if tuple(item.target for item in self.targets) != TARGET_COLUMNS:
            raise LocalReportExtractionError("local fold metrics must use competition target order")
        if any(item.report_count != self.report_count for item in self.targets):
            raise LocalReportExtractionError("local fold target counts disagree")

    @property
    def macro_accuracy(self) -> float | None:
        values = [item.accuracy for item in self.targets]
        return sum(cast(float, value) for value in values) / len(values) if all(
            value is not None for value in values
        ) else None

    @property
    def defined_target_macro_accuracy(self) -> float | None:
        values = [item.accuracy for item in self.targets if item.accuracy is not None]
        return sum(values) / len(values) if values else None

    @property
    def mean_coverage(self) -> float | None:
        values = [item.coverage for item in self.targets]
        return sum(cast(float, value) for value in values) / len(values) if all(
            value is not None for value in values
        ) else None

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "report_count": self.report_count,
            "missing_report_count": self.missing_report_count,
            "macro_accuracy": self.macro_accuracy,
            "defined_target_macro_accuracy": self.defined_target_macro_accuracy,
            "mean_coverage": self.mean_coverage,
            "targets": [item.to_dict() for item in self.targets],
        }


@dataclass(frozen=True)
class LocalModelCostSummary:
    """Measured local inference cost; monetary cost is intentionally unclaimed."""

    report_count: int
    prompt_char_count: int
    response_char_count: int
    input_token_count: int | None
    output_token_count: int | None
    model_load_elapsed_ns: int
    model_artifact_size_bytes: int
    monetary_cost_usd: float | None = None
    cost_basis: str = "local-cluster-allocation-cost-not-measured"

    def __post_init__(self) -> None:
        for field_name in (
            "report_count",
            "prompt_char_count",
            "response_char_count",
            "model_load_elapsed_ns",
            "model_artifact_size_bytes",
        ):
            _count(getattr(self, field_name), field_name)
        for field_name in ("input_token_count", "output_token_count"):
            value = getattr(self, field_name)
            if value is not None:
                _count(value, field_name)
        if self.monetary_cost_usd is not None and (
            isinstance(self.monetary_cost_usd, bool)
            or not isinstance(self.monetary_cost_usd, (int, float))
            or not isfinite(float(self.monetary_cost_usd))
            or self.monetary_cost_usd < 0
        ):
            raise LocalReportExtractionError("monetary_cost_usd must be null or non-negative")
        if not isinstance(self.cost_basis, str) or not self.cost_basis.strip():
            raise LocalReportExtractionError("cost_basis must be non-empty")

    @property
    def total_prompt_chars(self) -> int:
        return self.prompt_char_count

    def to_dict(self) -> dict[str, object]:
        return {
            "report_count": self.report_count,
            "prompt_char_count": self.prompt_char_count,
            "response_char_count": self.response_char_count,
            "input_token_count": self.input_token_count,
            "output_token_count": self.output_token_count,
            "model_load_elapsed_seconds": self.model_load_elapsed_ns / 1_000_000_000.0,
            "model_artifact_size_bytes": self.model_artifact_size_bytes,
            "monetary_cost_usd": self.monetary_cost_usd,
            "cost_basis": self.cost_basis,
        }


@dataclass(frozen=True)
class LocalFoldReportExtraction:
    """Private fold output with local-model training/evaluation partitions."""

    fold: int
    fold_manifest_checksum: str
    fold_training_study_uids: tuple[str, ...]
    fold_held_out_study_uids: tuple[str, ...]
    eligible_training_report_uids: tuple[str, ...]
    eligible_evaluation_gold_uids: tuple[str, ...]
    training_labels: tuple[LocalReportLabelRecord, ...]
    evaluation_labels: tuple[LocalReportLabelRecord, ...]
    missing_training_report_uids: tuple[str, ...]
    missing_evaluation_report_uids: tuple[str, ...]
    excluded_held_out_report_uids: tuple[str, ...]
    latency_samples_ns: tuple[int, ...]
    latency: LatencySummary
    cost: LocalModelCostSummary
    evaluation_metrics: LocalFoldReportMetrics

    def __post_init__(self) -> None:
        if not _fold_is_valid(self.fold):
            raise LocalReportExtractionError("local fold extraction uses an invalid fold")
        _validate_sha256(self.fold_manifest_checksum, "fold_manifest_checksum")
        if self.evaluation_metrics.fold != self.fold:
            raise LocalReportExtractionError("local fold metrics use a different fold")
        for field_name, values in (
            ("fold_training_study_uids", self.fold_training_study_uids),
            ("fold_held_out_study_uids", self.fold_held_out_study_uids),
            ("eligible_training_report_uids", self.eligible_training_report_uids),
            ("eligible_evaluation_gold_uids", self.eligible_evaluation_gold_uids),
            ("missing_training_report_uids", self.missing_training_report_uids),
            ("missing_evaluation_report_uids", self.missing_evaluation_report_uids),
            ("excluded_held_out_report_uids", self.excluded_held_out_report_uids),
        ):
            values_tuple = tuple(values)
            if any(not isinstance(value, str) or not value for value in values_tuple):
                raise LocalReportExtractionError(f"{field_name} contains an invalid study identifier")
            if tuple(sorted(values_tuple)) != values_tuple or len(set(values_tuple)) != len(values_tuple):
                raise LocalReportExtractionError(f"{field_name} must be unique and sorted")
        if set(self.fold_training_study_uids).intersection(self.fold_held_out_study_uids):
            raise LocalReportExtractionError("local fold training and held-out sets overlap")
        if not set(self.eligible_training_report_uids).issubset(self.fold_training_study_uids):
            raise LocalReportExtractionError("eligible local training reports are outside fold training")
        if not set(self.eligible_evaluation_gold_uids).issubset(self.fold_held_out_study_uids):
            raise LocalReportExtractionError("eligible local evaluation reports are outside held-out set")
        if not set(self.excluded_held_out_report_uids).issubset(self.fold_held_out_study_uids):
            raise LocalReportExtractionError("excluded local reports are outside held-out set")
        training_uids = {item.study_instance_uid for item in self.training_labels}
        evaluation_uids = {item.study_instance_uid for item in self.evaluation_labels}
        if len(training_uids) != len(self.training_labels) or len(evaluation_uids) != len(self.evaluation_labels):
            raise LocalReportExtractionError("local fold label partitions contain duplicate identifiers")
        if training_uids.intersection(evaluation_uids):
            raise LocalReportExtractionError("local fold label partitions overlap")
        if not training_uids.issubset(self.eligible_training_report_uids):
            raise LocalReportExtractionError("local training labels are outside eligible reports")
        if not evaluation_uids.issubset(self.eligible_evaluation_gold_uids):
            raise LocalReportExtractionError("local evaluation labels are outside eligible reports")
        if (
            training_uids.union(self.missing_training_report_uids)
            != set(self.eligible_training_report_uids)
            or training_uids.intersection(self.missing_training_report_uids)
        ):
            raise LocalReportExtractionError("local training report eligibility and missing IDs disagree")
        if (
            evaluation_uids.union(self.missing_evaluation_report_uids)
            != set(self.eligible_evaluation_gold_uids)
            or evaluation_uids.intersection(self.missing_evaluation_report_uids)
        ):
            raise LocalReportExtractionError("local evaluation report eligibility and missing IDs disagree")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.latency_samples_ns
        ):
            raise LocalReportExtractionError("local latency samples must be non-negative integers")
        if len(self.latency_samples_ns) != self.latency.report_count:
            raise LocalReportExtractionError("local latency samples do not match latency summary")
        if LatencySummary.from_durations(self.latency_samples_ns) != self.latency:
            raise LocalReportExtractionError("local latency summary does not match samples")
        if self.cost.report_count != len(self.training_labels) + len(self.evaluation_labels):
            raise LocalReportExtractionError("local cost count does not match extracted records")
        if self.latency.report_count != self.cost.report_count:
            raise LocalReportExtractionError("local latency and cost counts disagree")

    @property
    def training_supervised_entry_count(self) -> int:
        return sum(item.supervised_entry_count for item in self.training_labels)

    @property
    def evaluation_abstention_entry_count(self) -> int:
        return sum(
            sum(1 - value for value in item.supervision_mask)
            for item in self.evaluation_labels
        )

    def semantic_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "fold_manifest_checksum": self.fold_manifest_checksum,
            "fold_training_study_uids": list(self.fold_training_study_uids),
            "fold_held_out_study_uids": list(self.fold_held_out_study_uids),
            "eligible_training_report_uids": list(self.eligible_training_report_uids),
            "eligible_evaluation_gold_uids": list(self.eligible_evaluation_gold_uids),
            "training_labels": [item.semantic_dict() for item in self.training_labels],
            "evaluation_labels": [item.semantic_dict() for item in self.evaluation_labels],
            "missing_training_report_uids": list(self.missing_training_report_uids),
            "missing_evaluation_report_uids": list(self.missing_evaluation_report_uids),
            "excluded_held_out_report_uids": list(self.excluded_held_out_report_uids),
        }

    def to_public_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "training_label_count": len(self.training_labels),
            "eligible_training_report_count": len(self.eligible_training_report_uids),
            "eligible_evaluation_gold_count": len(self.eligible_evaluation_gold_uids),
            "fold_training_study_count": len(self.fold_training_study_uids),
            "fold_held_out_study_count": len(self.fold_held_out_study_uids),
            "training_supervised_entry_count": self.training_supervised_entry_count,
            "held_out_evaluation_report_count": len(self.evaluation_labels),
            "held_out_evaluation_abstention_entry_count": self.evaluation_abstention_entry_count,
            "missing_training_report_count": len(self.missing_training_report_uids),
            "missing_evaluation_report_count": len(self.missing_evaluation_report_uids),
            "excluded_held_out_report_count": len(self.excluded_held_out_report_uids),
            "evaluation_metrics": self.evaluation_metrics.to_dict(),
            "latency": self.latency.to_dict(),
            "cost": self.cost.to_dict(),
        }

    def to_private_dict(self) -> dict[str, object]:
        return {
            "schema_version": LOCAL_MODEL_EXTRACTION_SCHEMA_VERSION,
            "fold": self.fold,
            "fold_manifest_checksum": self.fold_manifest_checksum,
            "fold_training_study_uids": list(self.fold_training_study_uids),
            "fold_held_out_study_uids": list(self.fold_held_out_study_uids),
            "eligible_training_report_uids": list(self.eligible_training_report_uids),
            "eligible_evaluation_gold_uids": list(self.eligible_evaluation_gold_uids),
            "partitions": {
                "training_labels": [item.to_private_dict() for item in self.training_labels],
                "held_out_evaluation": [item.to_private_dict() for item in self.evaluation_labels],
            },
            "missing_training_report_uids": list(self.missing_training_report_uids),
            "missing_evaluation_report_uids": list(self.missing_evaluation_report_uids),
            "excluded_held_out_report_uids": list(self.excluded_held_out_report_uids),
            "evaluation_metrics": self.evaluation_metrics.to_dict(),
            "latency": self.latency.to_dict(),
            "cost": self.cost.to_dict(),
        }


def _local_target_metrics(
    records: Sequence[LocalReportLabelRecord],
    manifest: FoldManifest,
) -> tuple[LocalTargetReportMetrics, ...]:
    membership = {item.study_instance_uid: item for item in manifest.memberships}
    metrics: list[LocalTargetReportMetrics] = []
    for index, target in enumerate(TARGET_COLUMNS):
        gold_positive = 0
        gold_negative = 0
        predicted_positive = 0
        predicted_negative = 0
        correct = 0
        not_mentioned = 0
        uncertain = 0
        unknown = 0
        extraction_failed = 0
        confidence_values: list[float] = []
        auditable = 0
        for record in records:
            item = membership.get(record.study_instance_uid)
            if item is None or item.cohort != "gold":
                raise ReportSourceError(
                    f"local model evaluation references a non-gold or unknown study: "
                    f"{record.study_instance_uid}"
                )
            reference = item.reference_labels[index]
            if reference == 1:
                gold_positive += 1
            elif reference == 0:
                gold_negative += 1
            else:
                raise ReportSourceError("local model evaluation requires complete gold labels")
            label = record.interpretation.labels[index]
            if label.evidence:
                auditable += 1
            if label.state is LabelState.POSITIVE:
                predicted_positive += 1
                confidence_values.append(float(cast(float, label.confidence)))
                correct += label.value == reference
            elif label.state is LabelState.NEGATIVE:
                predicted_negative += 1
                confidence_values.append(float(cast(float, label.confidence)))
                correct += label.value == reference
            elif label.state is LabelState.NOT_MENTIONED:
                not_mentioned += 1
            elif label.state is LabelState.UNCERTAIN:
                uncertain += 1
            elif label.state is LabelState.UNKNOWN:
                unknown += 1
            elif label.state is LabelState.EXTRACTION_FAILED:
                extraction_failed += 1
            else:
                raise LocalModelOutputError(f"unsupported local state for {target}")
        report_count = len(records)
        confident = predicted_positive + predicted_negative
        metrics.append(
            LocalTargetReportMetrics(
                target=target,
                report_count=report_count,
                gold_positive_count=gold_positive,
                gold_negative_count=gold_negative,
                predicted_positive_count=predicted_positive,
                predicted_negative_count=predicted_negative,
                correct_count=correct,
                accuracy=correct / confident if confident else None,
                coverage=confident / report_count if report_count else None,
                not_mentioned_count=not_mentioned,
                uncertain_count=uncertain,
                unknown_count=unknown,
                extraction_failed_count=extraction_failed,
                abstention_count=report_count - confident,
                failure_count=extraction_failed,
                mean_confidence=sum(confidence_values) / len(confidence_values)
                if confidence_values
                else None,
                auditable_count=auditable,
            )
        )
    return tuple(metrics)


def _local_cost(
    records: Sequence[LocalReportLabelRecord],
    *,
    model_load_elapsed_ns: int = 0,
    model_artifact_size_bytes: int = 0,
) -> LocalModelCostSummary:
    interpretations = tuple(item.interpretation for item in records)
    input_tokens = [item.input_token_count for item in interpretations]
    output_tokens = [item.output_token_count for item in interpretations]
    return LocalModelCostSummary(
        report_count=len(records),
        prompt_char_count=sum(item.prompt_char_count for item in interpretations),
        response_char_count=sum(item.response_char_count for item in interpretations),
        input_token_count=sum(cast(int, value) for value in input_tokens)
        if interpretations and all(value is not None for value in input_tokens)
        else None,
        output_token_count=sum(cast(int, value) for value in output_tokens)
        if interpretations and all(value is not None for value in output_tokens)
        else None,
        model_load_elapsed_ns=model_load_elapsed_ns,
        model_artifact_size_bytes=model_artifact_size_bytes,
    )


@dataclass(frozen=True)
class LocalReportExtractionEvaluation:
    """Aggregate local-model evaluation and provenance for one fold repeat."""

    approval: LocalModelApproval
    prompt: FrozenLocalPrompt
    fold_manifest_checksum: str
    repeat_name: str
    report_source_version: str
    code_fingerprint: str
    code_revision: str
    environment_fingerprint: str
    execution_kind: str
    execution_identity: str
    upstream_input_manifest: Mapping[str, object]
    folds: tuple[LocalFoldReportMetrics, ...]
    pooled_targets: tuple[LocalTargetReportMetrics, ...]
    latency: LatencySummary
    cost: LocalModelCostSummary
    reproducibility_verified: bool
    semantic_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.approval, LocalModelApproval):
            raise LocalModelApprovalError("local evaluation requires LocalModelApproval")
        if not isinstance(self.prompt, FrozenLocalPrompt):
            raise LocalReportExtractionError("local evaluation requires FrozenLocalPrompt")
        self.approval.require_approved()
        if not self.prompt.is_frozen:
            raise LocalReportExtractionError("local evaluation requires a frozen prompt")
        _validate_sha256(self.fold_manifest_checksum, "fold_manifest_checksum")
        if not self.repeat_name.strip() or not self.report_source_version.strip():
            raise LocalReportExtractionError("local evaluation repeat/source metadata must be non-empty")
        _validate_sha256(self.code_fingerprint, "code_fingerprint")
        _validate_sha256(self.environment_fingerprint, "environment_fingerprint")
        if self.execution_kind not in {"local", "slurm"} or not self.execution_identity.strip():
            raise LocalReportExtractionError("local execution identity is invalid")
        if not isinstance(self.upstream_input_manifest, Mapping):
            raise LocalReportExtractionError("local upstream_input_manifest must be an object")
        if not isinstance(self.reproducibility_verified, bool):
            raise LocalReportExtractionError("local reproducibility_verified must be boolean")
        if tuple(item.fold for item in self.folds) != FOLD_IDS:
            raise LocalReportExtractionError("local evaluation must contain folds 0, 1 and 2")
        if tuple(item.target for item in self.pooled_targets) != TARGET_COLUMNS:
            raise LocalReportExtractionError("local pooled metrics must use competition order")
        _validate_sha256(self.semantic_digest, "semantic_digest")

    @property
    def extractor_version(self) -> str:
        return LOCAL_MODEL_EXTRACTOR_VERSION

    @property
    def pooled_report_count(self) -> int:
        return sum(item.report_count for item in self.folds)

    @property
    def pooled_missing_report_count(self) -> int:
        return sum(item.missing_report_count for item in self.folds)

    @property
    def macro_accuracy(self) -> float | None:
        values = [item.accuracy for item in self.pooled_targets]
        return sum(cast(float, value) for value in values) / len(values) if all(
            value is not None for value in values
        ) else None

    @property
    def defined_target_macro_accuracy(self) -> float | None:
        values = [item.accuracy for item in self.pooled_targets if item.accuracy is not None]
        return sum(values) / len(values) if values else None

    @property
    def mean_coverage(self) -> float | None:
        values = [item.coverage for item in self.pooled_targets]
        return sum(cast(float, value) for value in values) / len(values) if all(
            value is not None for value in values
        ) else None

    @property
    def lineage_metadata_verified(self) -> bool:
        """Check the recorded model, input, policy and adaptation lineage."""

        upstream = self.upstream_input_manifest
        fold_manifest = upstream.get("fold_manifest")
        report_corpus = upstream.get("report_corpus")
        extractor = upstream.get("extractor")
        execution_policy = upstream.get("execution_policy")
        if not all(
            isinstance(value, Mapping)
            for value in (fold_manifest, report_corpus, extractor, execution_policy)
        ):
            return False
        fold_manifest = cast(Mapping[str, object], fold_manifest)
        report_corpus = cast(Mapping[str, object], report_corpus)
        extractor = cast(Mapping[str, object], extractor)
        execution_policy = cast(Mapping[str, object], execution_policy)
        expected_loader_hashes = {
            self.approval.artifact_relative_path: self.approval.artifact_sha256,
            **dict(self.approval.supporting_artifact_hashes),
        }
        report_digest = report_corpus.get("record_set_sha256")
        return (
            fold_manifest.get("schema_version") == FOLD_SCHEMA_VERSION
            and fold_manifest.get("manifest_checksum") == self.fold_manifest_checksum
            and fold_manifest.get("repeat_name") == self.repeat_name
            and report_corpus.get("source_version") == self.report_source_version
            and isinstance(report_digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", report_digest) is not None
            and extractor.get("version") == self.extractor_version
            and extractor.get("code_fingerprint") == LOCAL_MODEL_CODE_FINGERPRINT
            and extractor.get("model_approval_fingerprint") == self.approval.fingerprint
            and extractor.get("model_artifact_sha256") == self.approval.artifact_sha256
            and extractor.get("loader_artifact_hashes") == expected_loader_hashes
            and extractor.get("prompt_fingerprint") == self.prompt.fingerprint
            and extractor.get("checkpoint_hashes") == [self.approval.artifact_sha256]
            and extractor.get("upstream_adaptation_inputs") == []
            and extractor.get("report_derived_rule_development") is False
            and execution_policy.get("location") == self.approval.execution_location
            and execution_policy.get("data_transfer") == self.approval.data_transfer_policy
            and execution_policy.get("device") == self.approval.device
            and upstream.get("held_out_adaptation_exclusions") == list(ADAPTATION_EXCLUSIONS)
        )

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": LOCAL_MODEL_EXTRACTION_SCHEMA_VERSION,
            "ticket": "16",
            "status": "completed_local_model_evaluation",
            "extractor": {
                "version": self.extractor_version,
                "model": self.approval.to_public_dict(),
                "prompt": self.prompt.to_public_dict(),
                "frozen_before_held_out_evaluation": True,
            },
            "fold_context": {
                "fold_manifest_checksum": self.fold_manifest_checksum,
                "repeat_name": self.repeat_name,
                "report_source_version": self.report_source_version,
                "fold_count": len(self.folds),
            },
            "provenance": {
                "code_revision": self.code_revision,
                "code_fingerprint": self.code_fingerprint,
                "environment_fingerprint": self.environment_fingerprint,
                "execution_kind": self.execution_kind,
                "execution_identity_private": True,
                "upstream_input_manifest": dict(self.upstream_input_manifest),
            },
            "boundaries": {
                "training_labels_are_report_only": True,
                "held_out_gold_reports_are_evaluation_only": True,
                "held_out_reports_excluded_from_training": True,
                "prompt_frozen_before_held_out_evaluation": True,
                "report_derived_prompt_or_rules_cross_fold": False,
                "reference_labels_take_precedence": True,
                "unknown_uncertain_failed_masked": True,
                "reports_remain_local": True,
                "lineage_metadata_verified": self.lineage_metadata_verified,
            },
            "privacy": {
                "raw_report_text_persisted": False,
                "source_report_references_private": True,
                "public_study_identifiers": False,
                "external_report_transfer": False,
            },
            "counts": {
                "held_out_evaluation_reports": self.pooled_report_count,
                "missing_held_out_gold_reports": self.pooled_missing_report_count,
                "defined_target_accuracy_count": sum(
                    item.accuracy is not None for item in self.pooled_targets
                ),
            },
            "pooled": {
                "macro_accuracy": self.macro_accuracy,
                "defined_target_macro_accuracy": self.defined_target_macro_accuracy,
                "mean_coverage": self.mean_coverage,
                "targets": [item.to_dict() for item in self.pooled_targets],
            },
            "folds": [item.to_dict() for item in self.folds],
            "latency": self.latency.to_dict(),
            "cost": self.cost.to_dict(),
            "reproducibility": {
                "verified": self.reproducibility_verified,
                "semantic_digest": self.semantic_digest,
                "repeatability_method": "same approved local model, frozen prompt, manifest and report digests",
            },
        }

    def to_private_dict(self) -> dict[str, object]:
        payload = self.to_public_dict()
        provenance = cast(dict[str, object], payload["provenance"])
        provenance["execution_identity"] = self.execution_identity
        provenance["approval"] = self.approval.to_private_dict()
        return payload


@dataclass(frozen=True)
class LocalReportExtractionRun:
    """Complete local-model result, including private fold label records."""

    evaluation: LocalReportExtractionEvaluation
    folds: tuple[LocalFoldReportExtraction, ...]
    semantic_digest: str
    report_corpus: ReportCorpus

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation, LocalReportExtractionEvaluation):
            raise LocalReportExtractionError("local run requires a local evaluation")
        if not isinstance(self.report_corpus, ReportCorpus):
            raise LocalReportExtractionError("local run requires a report corpus")
        if tuple(item.fold for item in self.folds) != FOLD_IDS:
            raise LocalReportExtractionError("local run must contain folds 0, 1 and 2")
        if any(
            item.fold_manifest_checksum != self.evaluation.fold_manifest_checksum
            for item in self.folds
        ):
            raise LocalReportExtractionError("local run and fold manifest checksums disagree")
        if self.evaluation.semantic_digest != self.semantic_digest:
            raise LocalReportExtractionError("local run and evaluation digests disagree")
        if self.evaluation.latency.report_count != sum(item.latency.report_count for item in self.folds):
            raise LocalReportExtractionError("local run and fold latency counts disagree")
        if self.evaluation.cost.report_count != sum(item.cost.report_count for item in self.folds):
            raise LocalReportExtractionError("local run and fold cost counts disagree")
        if tuple(item.evaluation_metrics for item in self.folds) != self.evaluation.folds:
            raise LocalReportExtractionError("local run and evaluation fold metrics disagree")

    @property
    def schema_version(self) -> str:
        return LOCAL_MODEL_EXTRACTION_SCHEMA_VERSION

    @property
    def reproducibility_verified(self) -> bool:
        return self.evaluation.reproducibility_verified

    @property
    def lineage_verified(self) -> bool:
        """Whether the serialized run proves the local V15 boundary."""

        report_corpus = self.evaluation.upstream_input_manifest.get("report_corpus")
        if not isinstance(report_corpus, Mapping):
            return False
        if report_corpus.get("record_set_sha256") != self.report_corpus.record_set_sha256:
            return False
        return self.evaluation.lineage_metadata_verified and all(
            fold.fold_manifest_checksum == self.evaluation.fold_manifest_checksum
            and not set(record.study_instance_uid for record in fold.training_labels).intersection(
                fold.fold_held_out_study_uids
            )
            for fold in self.folds
        )

    def to_public_dict(self) -> dict[str, object]:
        payload = self.evaluation.to_public_dict()
        payload["schema_version"] = LOCAL_MODEL_EXTRACTION_SCHEMA_VERSION
        payload["input"] = self.report_corpus.to_public_dict()
        payload["fold_partitions"] = [item.to_public_dict() for item in self.folds]
        boundaries = cast(dict[str, object], payload["boundaries"])
        boundaries["lineage_contract_verified"] = self.lineage_verified
        return payload

    def to_private_dict(self) -> dict[str, object]:
        payload = self.evaluation.to_private_dict()
        payload["schema_version"] = LOCAL_MODEL_EXTRACTION_SCHEMA_VERSION
        payload["input"] = self.report_corpus.to_private_dict()
        payload["fold_partitions"] = [item.to_private_dict() for item in self.folds]
        boundaries = cast(dict[str, object], payload["boundaries"])
        boundaries["lineage_contract_verified"] = self.lineage_verified
        return payload


def _validate_local_manifest(manifest: FoldManifest) -> None:
    if not isinstance(manifest, FoldManifest):
        raise ReportSourceError("local model extraction requires a FoldManifest")
    if manifest.superseded:
        raise ReportSourceError("superseded fold manifests cannot admit local-model evidence")
    if manifest.fold_ids != FOLD_IDS:
        raise ReportSourceError("local model extraction requires the canonical three-fold manifest")


def _validate_local_inputs(manifest: FoldManifest, corpus: ReportCorpus) -> None:
    _validate_local_manifest(manifest)
    if not isinstance(corpus, ReportCorpus):
        raise ReportSourceError("local model extraction requires a ReportCorpus")
    unknown = sorted(set(corpus.by_study) - set(manifest.study_to_fold))
    if unknown:
        raise ReportSourceError(f"report corpus contains unknown studies: {unknown[:3]}")


def validate_local_model_fold_boundaries(
    manifest: FoldManifest,
    fold: int,
    *,
    training_study_uids: Sequence[str],
    evaluation_study_uids: Sequence[str],
) -> None:
    """Apply ticket-15's report partition boundary to the local model path."""

    _validate_local_manifest(manifest)
    if not _fold_is_valid(fold):
        raise LocalReportExtractionError(f"fold must be one of {FOLD_IDS}")
    validate_fold_report_boundaries(
        manifest,
        fold,
        training_study_uids=training_study_uids,
        evaluation_study_uids=evaluation_study_uids,
    )


def _build_local_fold_report_extraction(
    manifest: FoldManifest,
    corpus: ReportCorpus,
    fold: int,
    extractor: LocalLanguageModelExtractor,
) -> LocalFoldReportExtraction:
    records_by_study = corpus.by_study
    held_out = set(manifest.held_out_study_uids(fold))
    expected_training = {
        item.study_instance_uid
        for item in manifest.report_only_memberships
        if item.fold != fold
    }
    expected_evaluation = {
        item.study_instance_uid
        for item in manifest.gold_memberships
        if item.fold == fold
    }
    training_uids = tuple(sorted(expected_training.intersection(records_by_study)))
    evaluation_uids = tuple(sorted(expected_evaluation.intersection(records_by_study)))
    validate_local_model_fold_boundaries(
        manifest,
        fold,
        training_study_uids=training_uids,
        evaluation_study_uids=evaluation_uids,
    )

    training_labels: list[LocalReportLabelRecord] = []
    evaluation_labels: list[LocalReportLabelRecord] = []
    durations: list[int] = []
    for uid in training_uids:
        started = time.perf_counter_ns()
        interpretation = extractor.extract(records_by_study[uid], fold=fold)
        durations.append(time.perf_counter_ns() - started)
        training_labels.append(
            LocalReportLabelRecord(uid, fold, ReportPartition.TRAINING_LABELS, interpretation)
        )
    for uid in evaluation_uids:
        started = time.perf_counter_ns()
        interpretation = extractor.extract(records_by_study[uid], fold=fold)
        durations.append(time.perf_counter_ns() - started)
        evaluation_labels.append(
            LocalReportLabelRecord(uid, fold, ReportPartition.HELD_OUT_EVALUATION, interpretation)
        )

    missing_training = tuple(sorted(expected_training - set(training_uids)))
    missing_evaluation = tuple(sorted(expected_evaluation - set(evaluation_uids)))
    excluded_held_out = tuple(sorted(held_out.intersection(records_by_study)))
    evaluation_metrics = _evaluate_local_fold_report_labels(
        manifest,
        fold,
        tuple(evaluation_labels),
        missing_report_count=len(missing_evaluation),
    )
    all_labels = tuple(training_labels) + tuple(evaluation_labels)
    return LocalFoldReportExtraction(
        fold=fold,
        fold_manifest_checksum=manifest.manifest_checksum,
        fold_training_study_uids=tuple(sorted(manifest.training_study_uids(fold))),
        fold_held_out_study_uids=tuple(sorted(manifest.held_out_study_uids(fold))),
        eligible_training_report_uids=tuple(sorted(expected_training)),
        eligible_evaluation_gold_uids=tuple(sorted(expected_evaluation)),
        training_labels=tuple(training_labels),
        evaluation_labels=tuple(evaluation_labels),
        missing_training_report_uids=missing_training,
        missing_evaluation_report_uids=missing_evaluation,
        excluded_held_out_report_uids=excluded_held_out,
        latency_samples_ns=tuple(durations),
        latency=LatencySummary.from_durations(durations),
        cost=_local_cost(all_labels),
        evaluation_metrics=evaluation_metrics,
    )


def _evaluate_local_fold_report_labels(
    manifest: FoldManifest,
    fold: int,
    records: Sequence[LocalReportLabelRecord],
    *,
    missing_report_count: int,
) -> LocalFoldReportMetrics:
    if not _fold_is_valid(fold):
        raise LocalReportExtractionError(f"fold must be one of {FOLD_IDS}")
    _count(missing_report_count, "missing_report_count")
    record_uids = [record.study_instance_uid for record in records]
    expected = {
        item.study_instance_uid
        for item in manifest.gold_memberships
        if item.fold == fold
    }
    if any(
        record.fold != fold or record.partition is not ReportPartition.HELD_OUT_EVALUATION
        for record in records
    ):
        raise LocalReportExtractionError("local fold evaluation contains a non-evaluation record")
    if not set(record_uids).issubset(expected) or len(set(record_uids)) != len(record_uids):
        raise LocalReportExtractionError("local fold evaluation does not match held-out gold membership")
    return LocalFoldReportMetrics(
        fold=fold,
        report_count=len(records),
        missing_report_count=missing_report_count,
        targets=_local_target_metrics(records, manifest),
    )


def _local_code_revision() -> str:
    configured = os.environ.get("RSNA_GIT_REVISION", "").strip()
    if configured:
        return configured
    repository = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    revision = completed.stdout.strip()
    return revision if revision else "unavailable"


def _local_environment_fingerprint() -> str:
    package_versions: dict[str, str] = {}
    for package_name in ("rsna-knee", "numpy", "torch", "transformers"):
        try:
            package_versions[package_name] = version(package_name)
        except PackageNotFoundError:
            package_versions[package_name] = "unavailable"
    return _json_sha256(
        {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": package_versions,
        }
    )


def _local_upstream_input_manifest(
    manifest: FoldManifest,
    corpus: ReportCorpus,
    extractor: LocalLanguageModelExtractor,
) -> dict[str, object]:
    membership_digest = _json_sha256(
        [
            {
                "study_instance_uid": item.study_instance_uid,
                "cohort": item.cohort,
                "fold": item.fold,
                "reference_labels": list(item.reference_labels),
            }
            for item in manifest.memberships
        ]
    )
    return {
        "fold_manifest": {
            "schema_version": FOLD_SCHEMA_VERSION,
            "manifest_checksum": manifest.manifest_checksum,
            "repeat_name": manifest.repeat_name,
            "eligible_study_set_sha256": membership_digest,
        },
        "report_corpus": {
            "schema_version": "rsna-knee-report-corpus-v1",
            "source_version": corpus.source_version,
            "record_set_sha256": corpus.record_set_sha256,
        },
        "extractor": {
            "version": extractor.extractor_version,
            "code_fingerprint": LOCAL_MODEL_CODE_FINGERPRINT,
            "model_approval_fingerprint": extractor.approval.fingerprint,
            "model_artifact_sha256": extractor.verified_artifact_sha256,
            "loader_artifact_hashes": {
                extractor.approval.artifact_relative_path: extractor.verified_artifact_sha256,
                **dict(extractor.approval.supporting_artifact_hashes),
            },
            "prompt_fingerprint": extractor.prompt.fingerprint,
            "checkpoint_hashes": [extractor.verified_artifact_sha256],
            "upstream_adaptation_inputs": [],
            "report_derived_rule_development": False,
        },
        "execution_policy": {
            "location": extractor.approval.execution_location,
            "data_transfer": extractor.approval.data_transfer_policy,
            "device": extractor.approval.device,
        },
        "held_out_adaptation_exclusions": list(ADAPTATION_EXCLUSIONS),
    }


def _build_local_run_once(
    manifest: FoldManifest,
    corpus: ReportCorpus,
    extractor: LocalLanguageModelExtractor,
) -> tuple[tuple[LocalFoldReportExtraction, ...], LatencySummary, LocalModelCostSummary, str]:
    folds = tuple(
        _build_local_fold_report_extraction(manifest, corpus, fold, extractor)
        for fold in FOLD_IDS
    )
    semantic_payload = {
        "fold_manifest_checksum": manifest.manifest_checksum,
        "repeat_name": manifest.repeat_name,
        "report_source_version": corpus.source_version,
        "model_approval_fingerprint": extractor.approval.fingerprint,
        "prompt_fingerprint": extractor.prompt.fingerprint,
        "extractor_fingerprint": extractor.fingerprint,
        "folds": [item.semantic_dict() for item in folds],
    }
    all_records = tuple(
        record
        for fold in folds
        for record in tuple(fold.training_labels) + tuple(fold.evaluation_labels)
    )
    return (
        folds,
        LatencySummary.from_durations(
            tuple(sample for item in folds for sample in item.latency_samples_ns)
        ),
        _local_cost(
            all_records,
            model_load_elapsed_ns=extractor.model_load_elapsed_ns,
            model_artifact_size_bytes=extractor.verified_artifact_size_bytes,
        ),
        _json_sha256(semantic_payload),
    )


def run_local_report_extraction(
    manifest: FoldManifest,
    corpus: ReportCorpus,
    extractor: LocalLanguageModelExtractor,
    *,
    verify_reproducibility: bool = True,
) -> LocalReportExtractionRun:
    """Run the approved local extractor over ticket-15's identical partitions."""

    _validate_local_inputs(manifest, corpus)
    if not isinstance(extractor, LocalLanguageModelExtractor) or not extractor.is_frozen:
        raise LocalModelApprovalError("local report extraction requires a frozen approved extractor")
    folds, latency, cost, semantic_digest = _build_local_run_once(manifest, corpus, extractor)
    reproducibility_verified = False
    if verify_reproducibility:
        _, _, _, repeated_digest = _build_local_run_once(manifest, corpus, extractor)
        reproducibility_verified = repeated_digest == semantic_digest
    pooled_records = tuple(
        record
        for fold in folds
        for record in fold.evaluation_labels
    )
    execution_kind, execution_identity = _local_execution_identity()
    evaluation = LocalReportExtractionEvaluation(
        approval=extractor.approval,
        prompt=extractor.prompt,
        fold_manifest_checksum=manifest.manifest_checksum,
        repeat_name=manifest.repeat_name,
        report_source_version=corpus.source_version,
        code_fingerprint=LOCAL_MODEL_CODE_FINGERPRINT,
        code_revision=_local_code_revision(),
        environment_fingerprint=_local_environment_fingerprint(),
        execution_kind=execution_kind,
        execution_identity=execution_identity,
        upstream_input_manifest=_local_upstream_input_manifest(manifest, corpus, extractor),
        folds=tuple(item.evaluation_metrics for item in folds),
        pooled_targets=_local_target_metrics(pooled_records, manifest),
        latency=latency,
        cost=cost,
        reproducibility_verified=reproducibility_verified,
        semantic_digest=semantic_digest,
    )
    return LocalReportExtractionRun(
        evaluation=evaluation,
        folds=folds,
        semantic_digest=semantic_digest,
        report_corpus=corpus,
    )


@dataclass(frozen=True)
class ExtractorTargetComparison:
    """One target's deterministic-versus-local metric comparison."""

    target: str
    baseline: TargetReportMetrics
    local_model: LocalTargetReportMetrics

    def __post_init__(self) -> None:
        if self.target not in TARGET_COLUMNS:
            raise LocalReportComparisonError(f"unknown comparison target {self.target!r}")
        if self.baseline.target != self.target or self.local_model.target != self.target:
            raise LocalReportComparisonError("comparison target records disagree")

    @staticmethod
    def _delta(left: float | None, right: float | None) -> float | None:
        return right - left if left is not None and right is not None else None

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "baseline": self.baseline.to_dict(),
            "local_model": self.local_model.to_dict(),
            "delta": {
                "accuracy": self._delta(self.baseline.accuracy, self.local_model.accuracy),
                "coverage": self._delta(self.baseline.coverage, self.local_model.coverage),
                "abstention_rate": self._delta(
                    self.baseline.abstention_rate,
                    self.local_model.abstention_rate,
                ),
                "failure_rate": self._delta(
                    self.baseline.failure_rate,
                    self.local_model.failure_rate,
                ),
            },
        }


class LocalReportComparisonError(LocalReportExtractionError):
    """Raised when two extractors cannot be compared on identical cohorts."""


@dataclass(frozen=True)
class ReportExtractorComparison:
    """Controlled ticket-16 comparison of the frozen baseline and local model."""

    baseline_run: ReportExtractionRun
    local_run: LocalReportExtractionRun
    targets: tuple[ExtractorTargetComparison, ...]
    identical_evaluation_cohorts: bool
    identical_training_eligibility: bool

    def __post_init__(self) -> None:
        if not isinstance(self.baseline_run, ReportExtractionRun):
            raise LocalReportComparisonError("comparison requires the ticket-15 baseline run")
        if not isinstance(self.local_run, LocalReportExtractionRun):
            raise LocalReportComparisonError("comparison requires a local-model run")
        if tuple(item.target for item in self.targets) != TARGET_COLUMNS:
            raise LocalReportComparisonError("comparison targets must use competition order")
        if not self.identical_evaluation_cohorts or not self.identical_training_eligibility:
            raise LocalReportComparisonError("comparison cohorts must be identical")

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {
                "baseline_semantic_digest": self.baseline_run.semantic_digest,
                "local_semantic_digest": self.local_run.semantic_digest,
                "targets": [item.to_dict() for item in self.targets],
            }
        )

    def to_public_dict(self) -> dict[str, object]:
        baseline = self.baseline_run.evaluation.to_public_dict()
        local = self.local_run.evaluation.to_public_dict()
        local_boundaries = cast(dict[str, object], local["boundaries"])
        local_boundaries["lineage_contract_verified"] = self.local_run.lineage_verified
        return {
            "schema_version": LOCAL_MODEL_COMPARISON_SCHEMA_VERSION,
            "ticket": "16",
            "status": "completed_controlled_comparison",
            "model_proposal_or_approval": self.local_run.evaluation.approval.to_public_dict(),
            "cohort": {
                "identical_evaluation_cohorts": self.identical_evaluation_cohorts,
                "identical_training_eligibility": self.identical_training_eligibility,
                "fold_count": len(self.local_run.folds),
                "held_out_evaluation_report_count": self.local_run.evaluation.pooled_report_count,
                "baseline_missing_evaluation_report_count": self.baseline_run.evaluation.pooled_missing_report_count,
                "local_missing_evaluation_report_count": self.local_run.evaluation.pooled_missing_report_count,
            },
            "protocol": {
                "deterministic_baseline_frozen": True,
                "local_prompt_frozen": True,
                "held_out_report_content_used_for_adaptation": False,
                "reference_labels_take_precedence": True,
                "unknown_uncertain_failed_masked": True,
                "report_processing_local_only": True,
            },
            "baseline": baseline,
            "local_model": local,
            "targets": [item.to_dict() for item in self.targets],
            "latency_cost": {
                "baseline": self.baseline_run.evaluation.latency.to_dict(),
                "local_model": {
                    "latency": self.local_run.evaluation.latency.to_dict(),
                    "cost": self.local_run.evaluation.cost.to_dict(),
                },
            },
            "auditability": {
                "baseline": "rule ids and private matched evidence are retained by ticket 15",
                "local_model": "private model evidence snippets, response hashes and failure codes are retained",
                "local_public_evidence_contains_raw_report_text": False,
            },
            "provenance": {
                "baseline_semantic_digest": self.baseline_run.semantic_digest,
                "local_model_semantic_digest": self.local_run.semantic_digest,
                "comparison_fingerprint": self.fingerprint,
            },
            "gates": {
                "V13_external_model_approval": self.local_run.evaluation.approval.approved,
                "V15_lineage_and_fold_isolation": self.local_run.lineage_verified,
                "repeatability": self.local_run.evaluation.reproducibility_verified,
            },
        }

    def to_private_dict(self) -> dict[str, object]:
        payload = self.to_public_dict()
        payload["private_cohort"] = {
            "evaluation_uids_by_fold": [
                {
                    "fold": fold.fold,
                    "baseline": sorted(
                        item.study_instance_uid for item in self.baseline_run.folds[fold.fold].evaluation_labels
                    ),
                    "local_model": sorted(
                        item.study_instance_uid for item in fold.evaluation_labels
                    ),
                }
                for fold in self.local_run.folds
            ],
            "training_eligibility_uids_by_fold": [
                {
                    "fold": fold.fold,
                    "baseline": list(self.baseline_run.folds[fold.fold].eligible_training_report_uids),
                    "local_model": list(fold.eligible_training_report_uids),
                }
                for fold in self.local_run.folds
            ],
        }
        payload["baseline_private"] = self.baseline_run.to_private_dict()
        payload["local_model_private"] = self.local_run.to_private_dict()
        return payload


def compare_report_extractors(
    baseline_run: ReportExtractionRun,
    local_run: LocalReportExtractionRun,
) -> ReportExtractorComparison:
    """Compare two completed runs only when their fold cohorts are identical."""

    if not isinstance(baseline_run, ReportExtractionRun):
        raise LocalReportComparisonError("baseline must be a ticket-15 ReportExtractionRun")
    if not isinstance(local_run, LocalReportExtractionRun):
        raise LocalReportComparisonError("local_model must be a LocalReportExtractionRun")
    baseline_evaluation = baseline_run.evaluation
    local_evaluation = local_run.evaluation
    if baseline_evaluation.fold_manifest_checksum != local_evaluation.fold_manifest_checksum:
        raise LocalReportComparisonError("fold manifest checksum differs")
    if baseline_evaluation.repeat_name != local_evaluation.repeat_name:
        raise LocalReportComparisonError("repeat name differs")
    if baseline_evaluation.report_source_version != local_evaluation.report_source_version:
        raise LocalReportComparisonError("report source version differs")
    if baseline_run.report_corpus.record_set_sha256 != local_run.report_corpus.record_set_sha256:
        raise LocalReportComparisonError("report corpus record set differs")

    evaluation_cohorts_match = True
    training_eligibility_match = True
    for baseline_fold, local_fold in zip(baseline_run.folds, local_run.folds):
        baseline_evaluation_uids = {
            item.study_instance_uid for item in baseline_fold.evaluation_labels
        }
        local_evaluation_uids = {
            item.study_instance_uid for item in local_fold.evaluation_labels
        }
        if baseline_evaluation_uids != local_evaluation_uids:
            evaluation_cohorts_match = False
        if (
            baseline_fold.eligible_evaluation_gold_uids
            != local_fold.eligible_evaluation_gold_uids
            or baseline_fold.eligible_training_report_uids
            != local_fold.eligible_training_report_uids
            or baseline_fold.missing_evaluation_report_uids
            != local_fold.missing_evaluation_report_uids
            or baseline_fold.missing_training_report_uids
            != local_fold.missing_training_report_uids
            or baseline_fold.excluded_held_out_report_uids
            != local_fold.excluded_held_out_report_uids
            or baseline_fold.fold_training_study_uids
            != local_fold.fold_training_study_uids
            or baseline_fold.fold_held_out_study_uids
            != local_fold.fold_held_out_study_uids
        ):
            training_eligibility_match = False
    if not evaluation_cohorts_match:
        raise LocalReportComparisonError("evaluation cohorts differ")
    if not training_eligibility_match:
        raise LocalReportComparisonError("training eligibility differs")

    targets = tuple(
        ExtractorTargetComparison(
            target=target,
            baseline=baseline_evaluation.pooled_targets[index],
            local_model=local_evaluation.pooled_targets[index],
        )
        for index, target in enumerate(TARGET_COLUMNS)
    )
    return ReportExtractorComparison(
        baseline_run=baseline_run,
        local_run=local_run,
        targets=targets,
        identical_evaluation_cohorts=evaluation_cohorts_match,
        identical_training_eligibility=training_eligibility_match,
    )


def write_local_model_comparison_reports(
    result: ReportExtractorComparison,
    *,
    public_path: str | Path,
    private_path: str | Path,
) -> tuple[Path, Path]:
    """Publish aggregate comparison and private audit artifacts atomically."""

    if not isinstance(result, ReportExtractorComparison):
        raise LocalReportComparisonError("comparison writer requires ReportExtractorComparison")
    public = Path(public_path)
    private = Path(private_path)
    if public.resolve() == private.resolve():
        raise LocalReportComparisonError("public and private comparison paths must differ")
    atomic_json_write(public, result.to_public_dict())
    try:
        atomic_json_write(private, result.to_private_dict())
    except BaseException:
        try:
            public.unlink()
        except FileNotFoundError:
            pass
        raise
    return public, private


__all__ = [
    "DEFAULT_LOCAL_PROMPT_SYSTEM",
    "DEFAULT_LOCAL_PROMPT_TEMPLATE",
    "FLAN_T5_SMALL_LICENSE",
    "FLAN_T5_SMALL_MODEL_ID",
    "FLAN_T5_SMALL_MODEL_REVISION",
    "FLAN_T5_SMALL_MODEL_SOURCE",
    "FLAN_T5_SMALL_SUPPORTING_ARTIFACT_HASHES",
    "FLAN_T5_SMALL_WEIGHTS_PATH",
    "FLAN_T5_SMALL_WEIGHTS_SHA256",
    "FrozenLocalPrompt",
    "LOCAL_MODEL_APPROVAL_DATE",
    "LOCAL_MODEL_APPROVAL_SCHEMA_VERSION",
    "LOCAL_MODEL_CODE_FINGERPRINT",
    "LOCAL_MODEL_COMPARISON_SCHEMA_VERSION",
    "LOCAL_MODEL_DATA_TRANSFER_POLICY",
    "LOCAL_MODEL_EXECUTION_LOCATION",
    "LOCAL_MODEL_EXTRACTOR_VERSION",
    "LOCAL_MODEL_EXTRACTION_SCHEMA_VERSION",
    "LOCAL_MODEL_MAX_CPUS",
    "LOCAL_MODEL_MAX_EVIDENCE_CHARS",
    "LOCAL_MODEL_MAX_MEMORY_GB",
    "LOCAL_MODEL_MAX_WALL_TIME_HOURS",
    "LOCAL_MODEL_OUTPUT_SCHEMA_VERSION",
    "LOCAL_MODEL_PROMPT_SCHEMA_VERSION",
    "LOCAL_MODEL_PROMPT_VERSION",
    "LocalModelApproval",
    "LocalModelApprovalError",
    "LocalModelArtifactError",
    "LocalModelArtifactVerification",
    "LocalModelDependencyError",
    "LocalModelGeneration",
    "LocalModelOutputError",
    "LocalModelCostSummary",
    "LocalFoldReportExtraction",
    "LocalFoldReportMetrics",
    "LocalLanguageModelExtractor",
    "LocalReportComparisonError",
    "LocalReportExtractionError",
    "LocalReportExtractionEvaluation",
    "LocalReportExtractionRun",
    "LocalReportInterpretation",
    "LocalReportLabelRecord",
    "LocalTargetReportMetrics",
    "ExtractorTargetComparison",
    "ReportExtractorComparison",
    "TransformersLocalTextGenerator",
    "CallableLocalTextGenerator",
    "compare_report_extractors",
    "freeze_local_prompt",
    "load_local_model_approval",
    "parse_local_model_output",
    "proposed_flan_t5_small_approval",
    "run_local_report_extraction",
    "validate_local_model_fold_boundaries",
    "verify_local_model_artifact",
    "write_local_model_comparison_reports",
    "write_local_model_proposal",
]
