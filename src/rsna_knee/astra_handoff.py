"""Fail-closed handoff mechanics for the Ticket 16 Astra Medium experiment.

This module deliberately stops at the Codex handoff seam.  It prepares private
report batches and validates responses that a human saves from a Codex session
after switching to Astra Medium.  It does not call a model, use a provider
SDK, manage an API key, estimate billing, or interpret a report itself.

The deep interface is intentionally small:

* :func:`build_astra_batch_plan` binds an owner-approved prompt, model
  identity, frozen folds and private report digests into immutable batches.
* :func:`validate_astra_output` validates one verbatim Astra response and
  converts malformed or incomplete responses into all-masked
  ``extraction-failed`` records.
* :func:`record_astra_output` preserves the raw response and private audit
  record without overwriting an existing attempt.

Ticket 15's deterministic extractor remains the comparator.  This module
does not create training labels, calculate accuracy, tune a prompt, or claim
report-only performance.  Luna can consume the validated private artifacts
later for those separate operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping, Sequence, cast

from .archive import atomic_json_write
from .folds import ADAPTATION_EXCLUSIONS, FOLD_IDS, FOLD_SCHEMA_VERSION, FoldManifest
from .report_extraction import (
    ReportCorpus,
    ReportExtractionRun,
    ReportPartition,
    ReportRecord,
    ReportSourceError,
    validate_fold_report_boundaries,
)
from .report_labels import LabelState
from .targets import TARGET_COLUMNS


ASTRA_HANDOFF_SCHEMA_VERSION = "rsna-knee-ticket16-astra-handoff-v1"
ASTRA_APPROVAL_SCHEMA_VERSION = "rsna-knee-ticket16-astra-approval-v1"
ASTRA_PROMPT_SCHEMA_VERSION = "rsna-knee-ticket16-astra-prompt-v1"
ASTRA_BATCH_SCHEMA_VERSION = "rsna-knee-ticket16-astra-batch-v1"
ASTRA_OUTPUT_SCHEMA_VERSION = "rsna-knee-ticket16-astra-output-v1"
ASTRA_AUDIT_SCHEMA_VERSION = "rsna-knee-ticket16-astra-audit-v1"
ASTRA_PUBLIC_EVIDENCE_SCHEMA_VERSION = "rsna-knee-ticket16-astra-public-evidence-v1"

ASTRA_EXPERIMENT_ID = "ticket16-astra-medium-codex-subscription-v1"
ASTRA_MODEL_NAME = "Astra Medium"
ASTRA_MODEL_SURFACE = "Codex subscription model switch"
ASTRA_MODEL_REVISION = "subscription-selected-astra-medium"
ASTRA_REASONING_EFFORT = "medium"
ASTRA_EFFORT = "medium"
ASTRA_EXECUTION_MODE = "manual-sequential-codex-handoff"
ASTRA_DATA_TRANSFER_POLICY = "private-reports-remain-in-user-controlled-codex-session"
ASTRA_APPROVAL_DATE = "2026-09-14"
ASTRA_BATCH_SIZE = 16
ASTRA_MAX_BATCH_SIZE = 16
ASTRA_BATCH_ID_PREFIX = "t16-astra-b"

ASTRA_SEMANTIC_STATES: tuple[str, ...] = (
    LabelState.POSITIVE.value,
    LabelState.NEGATIVE.value,
    LabelState.NOT_MENTIONED.value,
    LabelState.UNCERTAIN.value,
    LabelState.UNKNOWN.value,
)
ASTRA_PIPELINE_FAILURE_STATE = LabelState.EXTRACTION_FAILED.value


class AstraHandoffError(ValueError):
    """Base error for an invalid or unsafe Ticket 16 handoff artifact."""


class AstraApprovalError(AstraHandoffError):
    """Raised when the Astra identity or execution approval is incomplete."""


class AstraPromptError(AstraHandoffError):
    """Raised when a prompt artifact is absent, changed, or unapproved."""


class AstraBatchError(AstraHandoffError):
    """Raised when private reports cannot be partitioned safely."""


class AstraOutputValidationError(AstraHandoffError):
    """Raised for an output artifact that cannot be associated with this batch."""


class AstraContextMismatchError(AstraOutputValidationError):
    """Raised when output belongs to another prompt, model, effort, or batch."""


class AstraArtifactError(AstraHandoffError):
    """Raised when an immutable private/public artifact is inconsistent."""


class AstraResumeError(AstraArtifactError):
    """Raised when a caller tries to repeat a completed batch implicitly."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AstraHandoffError(f"{field_name} must be a non-empty string")
    return value


def _require_digest(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise AstraHandoffError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _require_date(value: object, field_name: str) -> str:
    text = _require_text(value, field_name)
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise AstraHandoffError(f"{field_name} must be an ISO date") from exc
    return text


def _require_nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AstraHandoffError(f"{field_name} must be a non-negative integer")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    result = _require_nonnegative_int(value, field_name)
    if result == 0:
        raise AstraHandoffError(f"{field_name} must be positive")
    return result


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise AstraHandoffError(f"{field_name} must be boolean")
    return value


def _validate_fold(fold: object, field_name: str = "fold") -> int:
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in FOLD_IDS:
        raise AstraBatchError(f"{field_name} must be one of {FOLD_IDS}")
    return fold


def _validate_fold_tuple(values: Iterable[int], field_name: str) -> tuple[int, ...]:
    result = tuple(_validate_fold(value, field_name) for value in values)
    if not result or len(set(result)) != len(result):
        raise AstraBatchError(f"{field_name} must contain unique folds")
    return result


def _validate_target_order(values: Iterable[str], field_name: str = "target_order") -> tuple[str, ...]:
    result = tuple(values)
    if result != TARGET_COLUMNS:
        raise AstraHandoffError(f"{field_name} must equal the canonical Ticket 15 target order")
    return result


def _validate_private_paths(public_path: Path, private_path: Path) -> None:
    if public_path.resolve() == private_path.resolve():
        raise AstraArtifactError("public and private artifact paths must differ")


def _write_immutable_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable artifact {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-",
        dir=destination.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite immutable artifact {destination}")
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def _write_immutable_json(path: str | Path, payload: object) -> Path:
    return atomic_json_write(path, payload)


def _coerce_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AstraHandoffError(f"{field_name} must be an object")
    return cast(Mapping[str, object], value)


def _exact_keys(payload: Mapping[str, object], expected: set[str], field_name: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise AstraOutputValidationError(f"{field_name} keys are not exact ({'; '.join(details)})")


ASTRA_OUTPUT_JSON_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "experiment_id",
        "model_identity_fingerprint",
        "reasoning_effort",
        "effort",
        "prompt_fingerprint",
        "batch_id",
        "input_digest",
        "studies",
    ],
    "properties": {
        "schema_version": {"const": ASTRA_OUTPUT_SCHEMA_VERSION},
        "experiment_id": {"const": ASTRA_EXPERIMENT_ID},
        "model_identity_fingerprint": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "reasoning_effort": {"const": ASTRA_REASONING_EFFORT},
        "effort": {"const": ASTRA_EFFORT},
        "prompt_fingerprint": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "batch_id": {"type": "string", "pattern": "^t16-astra-b[0-9]{4}-[0-9a-f]{16}$"},
        "input_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "studies": {
            "type": "array",
            "minItems": 1,
            "maxItems": ASTRA_MAX_BATCH_SIZE,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["study_instance_uid", "targets"],
                "properties": {
                    "study_instance_uid": {"type": "string", "minLength": 1},
                    "targets": {
                        "type": "array",
                        "minItems": len(TARGET_COLUMNS),
                        "maxItems": len(TARGET_COLUMNS),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["target", "state"],
                            "properties": {
                                "target": {"type": "string", "enum": list(TARGET_COLUMNS)},
                                "state": {"type": "string", "enum": list(ASTRA_SEMANTIC_STATES)},
                            },
                        },
                    },
                },
            },
        },
    },
}
ASTRA_OUTPUT_SCHEMA_FINGERPRINT = _json_sha256(ASTRA_OUTPUT_JSON_SCHEMA)


@dataclass(frozen=True)
class AstraMediumExperimentApproval:
    """The explicit Astra identity and subscription-only approval record.

    ``identity_approved`` records approval of the named experiment identity,
    not a hidden provider version or a remote API.  The current record is
    intentionally approved only for this handoff design and prompt review;
    :meth:`require_execution_ready` still requires an owner-approved prompt.
    """

    approval_id: str
    experiment_id: str
    model_name: str
    model_surface: str
    model_revision: str
    reasoning_effort: str
    effort: str
    execution_mode: str
    data_transfer_policy: str
    identity_approved: bool
    identity_approval_reference: str
    identity_approved_by: str
    identity_approved_on: str
    private_report_execution_approved: bool = True
    execution_approval_reference: str | None = None
    execution_approved_by: str | None = None
    execution_approved_on: str | None = None
    no_api_integration: bool = True
    no_api_key_handling: bool = True
    no_provider_sdk: bool = True
    no_external_model_wrapper: bool = True

    def __post_init__(self) -> None:
        for name in (
            "approval_id",
            "experiment_id",
            "model_name",
            "model_surface",
            "model_revision",
            "reasoning_effort",
            "effort",
            "execution_mode",
            "data_transfer_policy",
            "identity_approval_reference",
            "identity_approved_by",
        ):
            _require_text(getattr(self, name), name)
        _require_date(self.identity_approved_on, "identity_approved_on")
        if self.experiment_id != ASTRA_EXPERIMENT_ID:
            raise AstraApprovalError("approval is for a different Ticket 16 experiment")
        if self.model_name != ASTRA_MODEL_NAME or self.model_surface != ASTRA_MODEL_SURFACE:
            raise AstraApprovalError("approval does not identify Astra Medium through Codex")
        if self.model_revision != ASTRA_MODEL_REVISION:
            raise AstraApprovalError("Astra model revision identity is not the controlled subscription identity")
        if self.reasoning_effort != ASTRA_REASONING_EFFORT or self.effort != ASTRA_EFFORT:
            raise AstraApprovalError("Astra reasoning and effort must both be fixed to medium")
        if self.execution_mode != ASTRA_EXECUTION_MODE:
            raise AstraApprovalError("Astra execution must use the manual sequential Codex handoff")
        if self.data_transfer_policy != ASTRA_DATA_TRANSFER_POLICY:
            raise AstraApprovalError("Astra report transfer policy is not private Codex-session processing")
        for name in (
            "identity_approved",
            "private_report_execution_approved",
            "no_api_integration",
            "no_api_key_handling",
            "no_provider_sdk",
            "no_external_model_wrapper",
        ):
            if not isinstance(getattr(self, name), bool):
                raise AstraApprovalError(f"{name} must be boolean")
        if self.private_report_execution_approved:
            for name in (
                "execution_approval_reference",
                "execution_approved_by",
                "execution_approved_on",
            ):
                if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                    raise AstraApprovalError(
                        f"{name} is required when private report execution is approved"
                    )
            _require_date(self.execution_approved_on, "execution_approved_on")
        if not all(
            (
                self.no_api_integration,
                self.no_api_key_handling,
                self.no_provider_sdk,
                self.no_external_model_wrapper,
            )
        ):
            raise AstraApprovalError("the Astra handoff must not contain a programmatic model integration")

    @property
    def model_identity_fingerprint(self) -> str:
        return _json_sha256(self.identity_dict())

    @property
    def fingerprint(self) -> str:
        return _json_sha256(self.to_public_dict(include_fingerprint=False))

    @property
    def status(self) -> str:
        return "approved-for-prompt-review-and-manual-handoff" if self.identity_approved else "pending-approval"

    def identity_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "model_surface": self.model_surface,
            "model_revision": self.model_revision,
            "reasoning_effort": self.reasoning_effort,
            "effort": self.effort,
            "execution_mode": self.execution_mode,
            "data_transfer_policy": self.data_transfer_policy,
            "no_api_integration": self.no_api_integration,
            "no_api_key_handling": self.no_api_key_handling,
            "no_provider_sdk": self.no_provider_sdk,
            "no_external_model_wrapper": self.no_external_model_wrapper,
        }

    def to_public_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": ASTRA_APPROVAL_SCHEMA_VERSION,
            "approval_id": self.approval_id,
            "status": self.status,
            "experiment_id": self.experiment_id,
            "model": {
                "name": self.model_name,
                "surface": self.model_surface,
                "revision": self.model_revision,
                "identity_fingerprint": self.model_identity_fingerprint,
            },
            "reasoning_effort": self.reasoning_effort,
            "effort": self.effort,
            "execution_mode": self.execution_mode,
            "data_transfer_policy": self.data_transfer_policy,
            "approval_scope": {
                "identity_approved": self.identity_approved,
                "private_report_execution_approved": self.private_report_execution_approved,
                "prompt_approval_required": True,
            },
            "approval": {
                "identity_reference": self.identity_approval_reference,
                "identity_approved_by": self.identity_approved_by,
                "identity_approved_on": self.identity_approved_on,
                "execution_reference": self.execution_approval_reference,
                "execution_approved_by": self.execution_approved_by,
                "execution_approved_on": self.execution_approved_on,
            },
            "integration_policy": {
                "no_api_integration": self.no_api_integration,
                "no_api_key_handling": self.no_api_key_handling,
                "no_provider_sdk": self.no_provider_sdk,
                "no_external_model_wrapper": self.no_external_model_wrapper,
            },
        }
        if include_fingerprint:
            result["approval_fingerprint"] = self.fingerprint
        return result

    def require_identity_approved(self) -> None:
        if not self.identity_approved:
            raise AstraApprovalError("Astra Medium identity approval is not present")

    def require_execution_ready(self, prompt: "AstraPromptArtifact") -> None:
        self.require_identity_approved()
        if not self.private_report_execution_approved:
            raise AstraApprovalError("private report execution approval is not present")
        prompt.require_approved()
        if prompt.experiment_id != self.experiment_id:
            raise AstraApprovalError("prompt belongs to a different experiment")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AstraMediumExperimentApproval":
        if payload.get("schema_version") != ASTRA_APPROVAL_SCHEMA_VERSION:
            raise AstraApprovalError("unexpected Astra approval schema version")
        model = _coerce_mapping(payload.get("model"), "model")
        scope = _coerce_mapping(payload.get("approval_scope"), "approval_scope")
        approval = _coerce_mapping(payload.get("approval"), "approval")
        integration = _coerce_mapping(payload.get("integration_policy"), "integration_policy")
        result = cls(
            approval_id=_require_text(payload.get("approval_id"), "approval_id"),
            experiment_id=_require_text(payload.get("experiment_id"), "experiment_id"),
            model_name=_require_text(model.get("name"), "model.name"),
            model_surface=_require_text(model.get("surface"), "model.surface"),
            model_revision=_require_text(model.get("revision"), "model.revision"),
            reasoning_effort=_require_text(payload.get("reasoning_effort"), "reasoning_effort"),
            effort=_require_text(payload.get("effort"), "effort"),
            execution_mode=_require_text(payload.get("execution_mode"), "execution_mode"),
            data_transfer_policy=_require_text(
                payload.get("data_transfer_policy"), "data_transfer_policy"
            ),
            identity_approved=_require_bool(scope.get("identity_approved"), "approval_scope.identity_approved"),
            identity_approval_reference=_require_text(
                approval.get("identity_reference"), "approval.identity_reference"
            ),
            identity_approved_by=_require_text(
                approval.get("identity_approved_by"), "approval.identity_approved_by"
            ),
            identity_approved_on=_require_date(
                approval.get("identity_approved_on"), "approval.identity_approved_on"
            ),
            private_report_execution_approved=_require_bool(
                scope.get("private_report_execution_approved"),
                "approval_scope.private_report_execution_approved",
            ),
            execution_approval_reference=(
                str(approval["execution_reference"])
                if approval.get("execution_reference") is not None
                else None
            ),
            execution_approved_by=(
                str(approval["execution_approved_by"])
                if approval.get("execution_approved_by") is not None
                else None
            ),
            execution_approved_on=(
                str(approval["execution_approved_on"])
                if approval.get("execution_approved_on") is not None
                else None
            ),
            no_api_integration=_require_bool(
                integration.get("no_api_integration"), "integration_policy.no_api_integration"
            ),
            no_api_key_handling=_require_bool(
                integration.get("no_api_key_handling"), "integration_policy.no_api_key_handling"
            ),
            no_provider_sdk=_require_bool(
                integration.get("no_provider_sdk"), "integration_policy.no_provider_sdk"
            ),
            no_external_model_wrapper=_require_bool(
                integration.get("no_external_model_wrapper"),
                "integration_policy.no_external_model_wrapper",
            ),
        )
        if payload.get("status") != result.status:
            raise AstraApprovalError("Astra approval status does not match the approval record")
        declared_identity = model.get("identity_fingerprint")
        if declared_identity != result.model_identity_fingerprint:
            raise AstraApprovalError("Astra model identity fingerprint does not match the approval record")
        declared_fingerprint = payload.get("approval_fingerprint")
        if declared_fingerprint != result.fingerprint:
            raise AstraApprovalError("Astra approval fingerprint does not match the approval record")
        return result


def astra_medium_experiment_approval() -> AstraMediumExperimentApproval:
    """Return the explicit identity approval prepared by ``tsk3.txt``.

    This record approves the named manual handoff identity and its fixed
    effort settings.  It does not itself supply a medical prompt; a later
    owner-approved prompt artifact is still mandatory before any batch plan
    can be executed.
    """

    return AstraMediumExperimentApproval(
        approval_id="ticket16-astra-medium-codex-subscription-approval-v1",
        experiment_id=ASTRA_EXPERIMENT_ID,
        model_name=ASTRA_MODEL_NAME,
        model_surface=ASTRA_MODEL_SURFACE,
        model_revision=ASTRA_MODEL_REVISION,
        reasoning_effort=ASTRA_REASONING_EFFORT,
        effort=ASTRA_EFFORT,
        execution_mode=ASTRA_EXECUTION_MODE,
        data_transfer_policy=ASTRA_DATA_TRANSFER_POLICY,
        identity_approved=True,
        identity_approval_reference="tsk3.txt",
        identity_approved_by="task-owner instruction",
        identity_approved_on=ASTRA_APPROVAL_DATE,
        private_report_execution_approved=True,
        execution_approval_reference="tsk3.txt (preparation only; operator must still start manually)",
        execution_approved_by="task-owner instruction",
        execution_approved_on=ASTRA_APPROVAL_DATE,
    )


def load_astra_approval(path: str | Path) -> AstraMediumExperimentApproval:
    candidate = Path(path)
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AstraApprovalError(f"could not read Astra approval record: {candidate}") from exc
    return AstraMediumExperimentApproval.from_dict(_coerce_mapping(payload, "approval"))


def write_astra_approval(
    path: str | Path,
    approval: AstraMediumExperimentApproval | None = None,
) -> Path:
    selected = approval or astra_medium_experiment_approval()
    return _write_immutable_json(path, selected.to_public_dict())


@dataclass(frozen=True)
class AstraPromptArtifact:
    """An owner-supplied, immutable prompt artifact.

    There is intentionally no default medical prompt in this module.  The
    prompt text must arrive later from the owner, be approved explicitly, and
    then remain byte-for-byte represented by its fingerprint.
    """

    prompt_id: str
    prompt_version: str
    prompt_text: str
    experiment_id: str = ASTRA_EXPERIMENT_ID
    target_order: tuple[str, ...] = TARGET_COLUMNS
    output_schema_version: str = ASTRA_OUTPUT_SCHEMA_VERSION
    approved: bool = False
    approval_reference: str | None = None
    approved_by: str | None = None
    approved_on: str | None = None

    def __post_init__(self) -> None:
        for name in ("prompt_id", "prompt_version", "prompt_text", "experiment_id", "output_schema_version"):
            _require_text(getattr(self, name), name)
        _validate_target_order(self.target_order)
        if self.experiment_id != ASTRA_EXPERIMENT_ID:
            raise AstraPromptError("prompt artifact belongs to a different experiment")
        if self.output_schema_version != ASTRA_OUTPUT_SCHEMA_VERSION:
            raise AstraPromptError("prompt artifact uses an unsupported output schema")
        if not isinstance(self.approved, bool):
            raise AstraPromptError("approved must be boolean")
        if self.approved:
            for name in ("approval_reference", "approved_by", "approved_on"):
                if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                    raise AstraPromptError(f"{name} is required for an approved prompt")
            _require_date(self.approved_on, "approved_on")

    @property
    def status(self) -> str:
        return "approved" if self.approved else "pending-owner-approval"

    @property
    def prompt_sha256(self) -> str:
        return _text_sha256(self.prompt_text)

    @property
    def fingerprint(self) -> str:
        return _json_sha256(
            {
                "schema_version": ASTRA_PROMPT_SCHEMA_VERSION,
                "prompt_id": self.prompt_id,
                "prompt_version": self.prompt_version,
                "prompt_sha256": self.prompt_sha256,
                "prompt_text": self.prompt_text,
                "experiment_id": self.experiment_id,
                "target_order": list(self.target_order),
                "output_schema_version": self.output_schema_version,
            }
        )

    @property
    def is_frozen(self) -> bool:
        return self.approved

    def require_approved(self) -> None:
        if not self.approved:
            raise AstraPromptError("an explicitly owner-approved Astra prompt artifact is required")

    def approve(self, *, approval_reference: str, approved_by: str, approved_on: str) -> "AstraPromptArtifact":
        return replace(
            self,
            approved=True,
            approval_reference=approval_reference,
            approved_by=approved_by,
            approved_on=approved_on,
        )

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": ASTRA_PROMPT_SCHEMA_VERSION,
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "prompt_text": self.prompt_text,
            "experiment_id": self.experiment_id,
            "target_order": list(self.target_order),
            "output_schema_version": self.output_schema_version,
            "status": self.status,
            "approved": self.approved,
            "approval_reference": self.approval_reference,
            "approved_by": self.approved_by,
            "approved_on": self.approved_on,
            "prompt_sha256": self.prompt_sha256,
            "prompt_fingerprint": self.fingerprint,
            "frozen": self.is_frozen,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AstraPromptArtifact":
        if payload.get("schema_version") != ASTRA_PROMPT_SCHEMA_VERSION:
            raise AstraPromptError("unexpected Astra prompt schema version")
        result = cls(
            prompt_id=_require_text(payload.get("prompt_id"), "prompt_id"),
            prompt_version=_require_text(payload.get("prompt_version"), "prompt_version"),
            prompt_text=_require_text(payload.get("prompt_text"), "prompt_text"),
            experiment_id=_require_text(payload.get("experiment_id"), "experiment_id"),
            target_order=tuple(
                _require_text(value, "target_order entry")
                for value in cast(list[object], payload.get("target_order", []))
            ),
            output_schema_version=_require_text(
                payload.get("output_schema_version"), "output_schema_version"
            ),
            approved=payload.get("approved") is True,
            approval_reference=(
                str(payload["approval_reference"])
                if payload.get("approval_reference") is not None
                else None
            ),
            approved_by=(str(payload["approved_by"]) if payload.get("approved_by") is not None else None),
            approved_on=(str(payload["approved_on"]) if payload.get("approved_on") is not None else None),
        )
        if payload.get("status") != result.status or payload.get("frozen") != result.is_frozen:
            raise AstraPromptError("prompt status metadata disagrees with the prompt artifact")
        if payload.get("prompt_sha256") != result.prompt_sha256:
            raise AstraPromptError("prompt SHA-256 does not match prompt text")
        if payload.get("prompt_fingerprint") != result.fingerprint:
            raise AstraPromptError("prompt fingerprint does not match prompt contents")
        return result


def make_astra_prompt_artifact(
    prompt_text: str,
    *,
    prompt_id: str = "ticket16-owner-supplied-prompt",
    prompt_version: str = "owner-supplied-v1",
) -> AstraPromptArtifact:
    """Create an unapproved candidate; this does not authorize execution."""

    return AstraPromptArtifact(
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        prompt_text=prompt_text,
    )


def load_astra_prompt(path: str | Path) -> AstraPromptArtifact:
    candidate = Path(path)
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AstraPromptError(f"could not read Astra prompt artifact: {candidate}") from exc
    return AstraPromptArtifact.from_dict(_coerce_mapping(payload, "prompt"))


def write_astra_prompt(path: str | Path, prompt: AstraPromptArtifact) -> Path:
    if not isinstance(prompt, AstraPromptArtifact):
        raise AstraPromptError("prompt writer requires AstraPromptArtifact")
    return _write_immutable_json(path, prompt.to_public_dict())


@dataclass(frozen=True)
class AstraBatchStudy:
    """Private batch metadata for one unique report study.

    ``report_text`` is present only while preparing a private Codex input
    artifact.  It is never serialized by the public evidence or audit record.
    """

    study_instance_uid: str
    source_report_ref: str
    report_sha256: str
    cohort: str
    assigned_fold: int
    consuming_folds: tuple[int, ...]
    partitions_by_fold: tuple[tuple[int, str], ...]
    report_text: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_text(self.study_instance_uid, "study_instance_uid")
        _require_text(self.source_report_ref, "source_report_ref")
        _require_digest(self.report_sha256, "report_sha256")
        if self.cohort not in {"gold", "report_only"}:
            raise AstraBatchError("batch study cohort must be gold or report_only")
        _validate_fold(self.assigned_fold, "assigned_fold")
        folds = _validate_fold_tuple(self.consuming_folds, "consuming_folds")
        if folds != self.consuming_folds:
            object.__setattr__(self, "consuming_folds", folds)
        if self.cohort == "gold" and self.consuming_folds != (self.assigned_fold,):
            raise AstraBatchError("gold reports may be consumed only by their assigned evaluation fold")
        if self.cohort == "report_only" and self.assigned_fold in self.consuming_folds:
            raise AstraBatchError("report-only reports cannot be consumed by their assigned held-out fold")
        partitions = tuple(self.partitions_by_fold)
        if tuple(fold for fold, _ in partitions) != self.consuming_folds:
            raise AstraBatchError("partitions_by_fold must cover consuming_folds in order")
        expected_partition = (
            ReportPartition.HELD_OUT_EVALUATION.value
            if self.cohort == "gold"
            else ReportPartition.TRAINING_LABELS.value
        )
        if any(partition != expected_partition for _, partition in partitions):
            raise AstraBatchError("batch study partition disagrees with its cohort")
        if self.report_text is not None:
            if not isinstance(self.report_text, str):
                raise AstraBatchError("report_text must be a string when present")
            if _text_sha256(self.report_text) != self.report_sha256:
                raise AstraBatchError("report_text does not match report_sha256")

    @property
    def metadata_digest(self) -> str:
        return _json_sha256(self.to_metadata_dict())

    def to_metadata_dict(self) -> dict[str, object]:
        return {
            "study_instance_uid": self.study_instance_uid,
            "source_report_ref": self.source_report_ref,
            "report_sha256": self.report_sha256,
            "cohort": self.cohort,
            "assigned_fold": self.assigned_fold,
            "consuming_folds": list(self.consuming_folds),
            "partitions_by_fold": [
                {"fold": fold, "partition": partition}
                for fold, partition in self.partitions_by_fold
            ],
        }

    def to_private_dict(self, *, include_report_text: bool = False) -> dict[str, object]:
        result = self.to_metadata_dict()
        if include_report_text:
            if self.report_text is None:
                raise AstraArtifactError("this batch record has no report text to write")
            result["report_text"] = self.report_text
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AstraBatchStudy":
        partitions_raw = payload.get("partitions_by_fold")
        if not isinstance(partitions_raw, list):
            raise AstraArtifactError("partitions_by_fold must be an array")
        partitions: list[tuple[int, str]] = []
        for raw in partitions_raw:
            item = _coerce_mapping(raw, "partitions_by_fold entry")
            partitions.append(
                (
                    _validate_fold(item.get("fold"), "partition fold"),
                    _require_text(item.get("partition"), "partition"),
                )
            )
        report_text = payload.get("report_text")
        if report_text is not None and not isinstance(report_text, str):
            raise AstraArtifactError("report_text must be a string when present")
        raw_consuming = payload.get("consuming_folds")
        if not isinstance(raw_consuming, list):
            raise AstraArtifactError("consuming_folds must be an array")
        return cls(
            study_instance_uid=_require_text(payload.get("study_instance_uid"), "study_instance_uid"),
            source_report_ref=_require_text(payload.get("source_report_ref"), "source_report_ref"),
            report_sha256=_require_digest(payload.get("report_sha256"), "report_sha256"),
            cohort=_require_text(payload.get("cohort"), "cohort"),
            assigned_fold=_validate_fold(payload.get("assigned_fold"), "assigned_fold"),
            consuming_folds=tuple(
                _validate_fold(value, "consuming fold") for value in raw_consuming
            ),
            partitions_by_fold=tuple(partitions),
            report_text=report_text,
        )


@dataclass(frozen=True)
class AstraFoldCohort:
    """One fold's Ticket 15-compatible report eligibility sets."""

    fold: int
    fold_training_study_uids: tuple[str, ...]
    fold_held_out_study_uids: tuple[str, ...]
    eligible_training_report_uids: tuple[str, ...]
    eligible_evaluation_gold_uids: tuple[str, ...]
    missing_training_report_uids: tuple[str, ...]
    missing_evaluation_report_uids: tuple[str, ...]
    excluded_held_out_report_uids: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_fold(self.fold)
        for name in (
            "fold_training_study_uids",
            "fold_held_out_study_uids",
            "eligible_training_report_uids",
            "eligible_evaluation_gold_uids",
            "missing_training_report_uids",
            "missing_evaluation_report_uids",
            "excluded_held_out_report_uids",
        ):
            values = tuple(getattr(self, name))
            if len(set(values)) != len(values):
                raise AstraBatchError(f"{name} must not contain duplicate studies")
            if tuple(sorted(values)) != values:
                raise AstraBatchError(f"{name} must be sorted")
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise AstraBatchError(f"{name} must contain non-empty study identifiers")
            if values != getattr(self, name):
                object.__setattr__(self, name, values)

    def to_private_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "fold_training_study_uids": list(self.fold_training_study_uids),
            "fold_held_out_study_uids": list(self.fold_held_out_study_uids),
            "eligible_training_report_uids": list(self.eligible_training_report_uids),
            "eligible_evaluation_gold_uids": list(self.eligible_evaluation_gold_uids),
            "missing_training_report_uids": list(self.missing_training_report_uids),
            "missing_evaluation_report_uids": list(self.missing_evaluation_report_uids),
            "excluded_held_out_report_uids": list(self.excluded_held_out_report_uids),
        }

    def to_public_dict(self) -> dict[str, object]:
        private = self.to_private_dict()
        return {
            "fold": self.fold,
            "fold_training_study_count": len(self.fold_training_study_uids),
            "fold_held_out_study_count": len(self.fold_held_out_study_uids),
            "eligible_training_report_count": len(self.eligible_training_report_uids),
            "eligible_evaluation_gold_count": len(self.eligible_evaluation_gold_uids),
            "missing_training_report_count": len(self.missing_training_report_uids),
            "missing_evaluation_report_count": len(self.missing_evaluation_report_uids),
            "excluded_held_out_report_count": len(self.excluded_held_out_report_uids),
            "cohort_digest": _json_sha256(private),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AstraFoldCohort":
        def strings(name: str) -> tuple[str, ...]:
            raw = payload.get(name)
            if not isinstance(raw, list):
                raise AstraArtifactError(f"{name} must be an array")
            return tuple(_require_text(value, name) for value in raw)

        result = cls(
            fold=_validate_fold(payload.get("fold")),
            fold_training_study_uids=strings("fold_training_study_uids"),
            fold_held_out_study_uids=strings("fold_held_out_study_uids"),
            eligible_training_report_uids=strings("eligible_training_report_uids"),
            eligible_evaluation_gold_uids=strings("eligible_evaluation_gold_uids"),
            missing_training_report_uids=strings("missing_training_report_uids"),
            missing_evaluation_report_uids=strings("missing_evaluation_report_uids"),
            excluded_held_out_report_uids=strings("excluded_held_out_report_uids"),
        )
        declared_digest = payload.get("cohort_digest")
        if declared_digest is not None and declared_digest != _json_sha256(result.to_private_dict()):
            raise AstraArtifactError("cohort digest does not match fold cohort")
        return result


@dataclass(frozen=True)
class AstraComparatorCohorts:
    """Canonical, identifier-bearing private cohort contract for Luna."""

    fold_manifest_checksum: str
    repeat_name: str
    report_source_version: str
    report_record_set_sha256: str
    folds: tuple[AstraFoldCohort, ...]

    def __post_init__(self) -> None:
        _require_digest(self.fold_manifest_checksum, "fold_manifest_checksum")
        _require_text(self.repeat_name, "repeat_name")
        _require_text(self.report_source_version, "report_source_version")
        _require_digest(self.report_record_set_sha256, "report_record_set_sha256")
        if tuple(item.fold for item in self.folds) != FOLD_IDS:
            raise AstraBatchError("Astra comparator cohorts require folds 0, 1 and 2")

    @property
    def fingerprint(self) -> str:
        return _json_sha256(self.to_private_dict())

    @property
    def unique_available_report_count(self) -> int:
        return len(self.available_report_uids)

    @property
    def available_report_uids(self) -> frozenset[str]:
        available: set[str] = set()
        for fold in self.folds:
            available.update(set(fold.eligible_training_report_uids) - set(fold.missing_training_report_uids))
            available.update(set(fold.eligible_evaluation_gold_uids) - set(fold.missing_evaluation_report_uids))
        return frozenset(available)

    def to_private_dict(self) -> dict[str, object]:
        return {
            "schema_version": ASTRA_HANDOFF_SCHEMA_VERSION,
            "fold_manifest_checksum": self.fold_manifest_checksum,
            "repeat_name": self.repeat_name,
            "report_source_version": self.report_source_version,
            "report_record_set_sha256": self.report_record_set_sha256,
            "folds": [item.to_private_dict() for item in self.folds],
        }

    def to_public_dict(self) -> dict[str, object]:
        return {
            "fold_manifest_checksum": self.fold_manifest_checksum,
            "repeat_name": self.repeat_name,
            "report_source_version": self.report_source_version,
            "report_record_set_sha256": self.report_record_set_sha256,
            "fold_count": len(self.folds),
            "folds": [item.to_public_dict() for item in self.folds],
            "cohort_fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AstraComparatorCohorts":
        raw_folds = payload.get("folds")
        if not isinstance(raw_folds, list):
            raise AstraArtifactError("cohorts.folds must be an array")
        result = cls(
            fold_manifest_checksum=_require_digest(
                payload.get("fold_manifest_checksum"), "fold_manifest_checksum"
            ),
            repeat_name=_require_text(payload.get("repeat_name"), "repeat_name"),
            report_source_version=_require_text(
                payload.get("report_source_version"), "report_source_version"
            ),
            report_record_set_sha256=_require_digest(
                payload.get("report_record_set_sha256"), "report_record_set_sha256"
            ),
            folds=tuple(AstraFoldCohort.from_dict(_coerce_mapping(value, "cohort fold")) for value in raw_folds),
        )
        if payload.get("cohort_fingerprint") not in {None, result.fingerprint}:
            raise AstraArtifactError("cohort fingerprint does not match cohorts")
        return result


def _validate_manifest_for_astra(manifest: FoldManifest) -> None:
    if not isinstance(manifest, FoldManifest):
        raise AstraBatchError("Astra handoff requires a FoldManifest")
    if manifest.superseded:
        raise AstraBatchError("superseded fold manifests cannot feed Ticket 16")
    if manifest.fold_ids != FOLD_IDS:
        raise AstraBatchError("Astra handoff requires the canonical three-fold manifest")
    if manifest.adaptation_exclusions != ADAPTATION_EXCLUSIONS:
        raise AstraBatchError("Astra handoff requires the canonical adaptation exclusions")


def build_astra_comparator_cohorts(
    manifest: FoldManifest,
    corpus: ReportCorpus,
) -> AstraComparatorCohorts:
    """Build Ticket 15's fold contract without interpreting any report text."""

    _validate_manifest_for_astra(manifest)
    if not isinstance(corpus, ReportCorpus):
        raise ReportSourceError("Astra handoff requires a ReportCorpus")
    records_by_study = corpus.by_study
    unknown = sorted(set(records_by_study) - set(manifest.study_to_fold))
    if unknown:
        raise AstraBatchError(f"report corpus contains studies outside the frozen manifest: {unknown[:3]}")

    folds: list[AstraFoldCohort] = []
    for fold in FOLD_IDS:
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
        training = tuple(sorted(expected_training.intersection(records_by_study)))
        evaluation = tuple(sorted(expected_evaluation.intersection(records_by_study)))
        validate_fold_report_boundaries(
            manifest,
            fold,
            training_study_uids=training,
            evaluation_study_uids=evaluation,
        )
        folds.append(
            AstraFoldCohort(
                fold=fold,
                fold_training_study_uids=tuple(sorted(manifest.training_study_uids(fold))),
                fold_held_out_study_uids=tuple(sorted(manifest.held_out_study_uids(fold))),
                eligible_training_report_uids=tuple(sorted(expected_training)),
                eligible_evaluation_gold_uids=tuple(sorted(expected_evaluation)),
                missing_training_report_uids=tuple(sorted(expected_training - set(training))),
                missing_evaluation_report_uids=tuple(sorted(expected_evaluation - set(evaluation))),
                excluded_held_out_report_uids=tuple(sorted(held_out.intersection(records_by_study))),
            )
        )
    return AstraComparatorCohorts(
        fold_manifest_checksum=manifest.manifest_checksum,
        repeat_name=manifest.repeat_name,
        report_source_version=corpus.source_version,
        report_record_set_sha256=corpus.record_set_sha256,
        folds=tuple(folds),
    )


def assert_astra_cohorts_match_ticket15(
    cohorts: AstraComparatorCohorts,
    baseline_run: ReportExtractionRun,
) -> None:
    """Fail closed unless a later Luna comparison uses identical Ticket 15 sets."""

    if not isinstance(baseline_run, ReportExtractionRun):
        raise AstraBatchError("Ticket 15 comparator must be a ReportExtractionRun")
    evaluation = baseline_run.evaluation
    if cohorts.fold_manifest_checksum != evaluation.fold_manifest_checksum:
        raise AstraBatchError("Astra and Ticket 15 fold manifests differ")
    if cohorts.repeat_name != evaluation.repeat_name:
        raise AstraBatchError("Astra and Ticket 15 repeat names differ")
    if cohorts.report_source_version != evaluation.report_source_version:
        raise AstraBatchError("Astra and Ticket 15 report source versions differ")
    if cohorts.report_record_set_sha256 != baseline_run.report_corpus.record_set_sha256:
        raise AstraBatchError("Astra and Ticket 15 report record sets differ")
    if tuple(item.fold for item in baseline_run.folds) != FOLD_IDS:
        raise AstraBatchError("Ticket 15 comparator does not contain the canonical folds")
    for expected, actual in zip(cohorts.folds, baseline_run.folds):
        comparisons = (
            (set(expected.fold_training_study_uids), set(actual.fold_training_study_uids)),
            (set(expected.fold_held_out_study_uids), set(actual.fold_held_out_study_uids)),
            (set(expected.eligible_training_report_uids), set(actual.eligible_training_report_uids)),
            (set(expected.eligible_evaluation_gold_uids), set(actual.eligible_evaluation_gold_uids)),
            (set(expected.missing_training_report_uids), set(actual.missing_training_report_uids)),
            (set(expected.missing_evaluation_report_uids), set(actual.missing_evaluation_report_uids)),
            (set(expected.excluded_held_out_report_uids), set(actual.excluded_held_out_report_uids)),
        )
        if any(left != right for left, right in comparisons):
            raise AstraBatchError(f"Astra and Ticket 15 cohorts differ for fold {expected.fold}")


@dataclass(frozen=True)
class AstraBatch:
    """One stable private input batch, never a model invocation."""

    batch_id: str
    batch_index: int
    batch_count: int
    experiment_id: str
    model_identity_fingerprint: str
    prompt_fingerprint: str
    fold_manifest_checksum: str
    report_source_version: str
    report_record_set_sha256: str
    records: tuple[AstraBatchStudy, ...]
    input_digest: str

    def __post_init__(self) -> None:
        if self.experiment_id != ASTRA_EXPERIMENT_ID:
            raise AstraBatchError("batch belongs to a different experiment")
        _require_digest(self.model_identity_fingerprint, "model_identity_fingerprint")
        _require_digest(self.prompt_fingerprint, "prompt_fingerprint")
        _require_digest(self.fold_manifest_checksum, "fold_manifest_checksum")
        _require_text(self.report_source_version, "report_source_version")
        _require_digest(self.report_record_set_sha256, "report_record_set_sha256")
        _require_digest(self.input_digest, "input_digest")
        _require_positive_int(self.batch_count, "batch_count")
        if isinstance(self.batch_index, bool) or not isinstance(self.batch_index, int):
            raise AstraBatchError("batch_index must be an integer")
        if self.batch_index < 0 or self.batch_index >= self.batch_count:
            raise AstraBatchError("batch_index must be within batch_count")
        if not self.records or len(self.records) > ASTRA_MAX_BATCH_SIZE:
            raise AstraBatchError(f"batch record count must be between 1 and {ASTRA_MAX_BATCH_SIZE}")
        if tuple(sorted(item.study_instance_uid for item in self.records)) != tuple(
            item.study_instance_uid for item in self.records
        ):
            raise AstraBatchError("batch records must be sorted by StudyInstanceUID")
        if len({item.study_instance_uid for item in self.records}) != len(self.records):
            raise AstraBatchError("batch records contain duplicate studies")
        expected_digest = _json_sha256(
            {
                "experiment_id": self.experiment_id,
                "model_identity_fingerprint": self.model_identity_fingerprint,
                "prompt_fingerprint": self.prompt_fingerprint,
                "fold_manifest_checksum": self.fold_manifest_checksum,
                "report_source_version": self.report_source_version,
                "report_record_set_sha256": self.report_record_set_sha256,
                "batch_index": self.batch_index,
                "batch_count": self.batch_count,
                "records": [item.to_metadata_dict() for item in self.records],
            }
        )
        if self.input_digest != expected_digest:
            raise AstraBatchError("batch input digest does not match batch metadata")
        expected_id = f"{ASTRA_BATCH_ID_PREFIX}{self.batch_index + 1:04d}-{self.input_digest[:16]}"
        if self.batch_id != expected_id:
            raise AstraBatchError("batch_id is not the stable digest-derived batch identifier")

    @property
    def study_uids(self) -> tuple[str, ...]:
        return tuple(item.study_instance_uid for item in self.records)

    def to_codex_input_dict(self, prompt: AstraPromptArtifact) -> dict[str, object]:
        """Return the private JSON payload to attach to one Codex turn."""

        if prompt.fingerprint != self.prompt_fingerprint:
            raise AstraContextMismatchError("prompt does not match the batch prompt fingerprint")
        prompt.require_approved()
        if prompt.experiment_id != self.experiment_id:
            raise AstraContextMismatchError("prompt belongs to a different experiment")
        reports = []
        for record in self.records:
            if record.report_text is None:
                raise AstraArtifactError("loaded batch metadata has no report text for Codex input")
            reports.append(
                {
                    "study_instance_uid": record.study_instance_uid,
                    "report_text": record.report_text,
                }
            )
        return {
            "schema_version": ASTRA_BATCH_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "model": {
                "name": ASTRA_MODEL_NAME,
                "surface": ASTRA_MODEL_SURFACE,
                "identity_fingerprint": self.model_identity_fingerprint,
            },
            "reasoning_effort": ASTRA_REASONING_EFFORT,
            "effort": ASTRA_EFFORT,
            "prompt": {
                "prompt_id": prompt.prompt_id,
                "prompt_version": prompt.prompt_version,
                "prompt_text": prompt.prompt_text,
                "prompt_fingerprint": prompt.fingerprint,
                "output_schema_version": ASTRA_OUTPUT_SCHEMA_VERSION,
                "target_order": list(TARGET_COLUMNS),
            },
            "batch": {
                "batch_id": self.batch_id,
                "batch_index": self.batch_index,
                "batch_count": self.batch_count,
                "input_digest": self.input_digest,
            },
            "output_contract": {
                "schema_version": ASTRA_OUTPUT_SCHEMA_VERSION,
                "target_order": list(TARGET_COLUMNS),
                "accepted_states": list(ASTRA_SEMANTIC_STATES),
                "extraction_failed_policy": "pipeline-only; never emit as a semantic state",
            },
            "reports": reports,
        }

    def to_public_dict(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "batch_index": self.batch_index,
            "batch_count": self.batch_count,
            "study_count": len(self.records),
            "study_set_digest": _json_sha256(list(self.study_uids)),
            "input_digest": self.input_digest,
            "fold_manifest_checksum": self.fold_manifest_checksum,
            "report_source_version": self.report_source_version,
            "report_record_set_sha256": self.report_record_set_sha256,
        }

    def to_private_dict(self) -> dict[str, object]:
        return {
            "schema_version": ASTRA_BATCH_SCHEMA_VERSION,
            "batch_id": self.batch_id,
            "batch_index": self.batch_index,
            "batch_count": self.batch_count,
            "experiment_id": self.experiment_id,
            "model_identity_fingerprint": self.model_identity_fingerprint,
            "prompt_fingerprint": self.prompt_fingerprint,
            "reasoning_effort": ASTRA_REASONING_EFFORT,
            "effort": ASTRA_EFFORT,
            "fold_manifest_checksum": self.fold_manifest_checksum,
            "report_source_version": self.report_source_version,
            "report_record_set_sha256": self.report_record_set_sha256,
            "input_digest": self.input_digest,
            "records": [item.to_private_dict() for item in self.records],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AstraBatch":
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise AstraArtifactError("batch records must be an array")
        result = cls(
            batch_id=_require_text(payload.get("batch_id"), "batch_id"),
            batch_index=_require_nonnegative_int(payload.get("batch_index"), "batch_index"),
            batch_count=_require_positive_int(payload.get("batch_count"), "batch_count"),
            experiment_id=_require_text(payload.get("experiment_id"), "experiment_id"),
            model_identity_fingerprint=_require_digest(
                payload.get("model_identity_fingerprint"), "model_identity_fingerprint"
            ),
            prompt_fingerprint=_require_digest(payload.get("prompt_fingerprint"), "prompt_fingerprint"),
            fold_manifest_checksum=_require_digest(
                payload.get("fold_manifest_checksum"), "fold_manifest_checksum"
            ),
            report_source_version=_require_text(
                payload.get("report_source_version"), "report_source_version"
            ),
            report_record_set_sha256=_require_digest(
                payload.get("report_record_set_sha256"), "report_record_set_sha256"
            ),
            records=tuple(
                AstraBatchStudy.from_dict(_coerce_mapping(item, "batch record")) for item in raw_records
            ),
            input_digest=_require_digest(payload.get("input_digest"), "input_digest"),
        )
        if payload.get("schema_version") not in {None, ASTRA_BATCH_SCHEMA_VERSION}:
            raise AstraArtifactError("unexpected batch schema version")
        return result


@dataclass(frozen=True)
class AstraBatchPlan:
    """Complete immutable plan for a sequential Astra handoff."""

    approval: AstraMediumExperimentApproval
    prompt: AstraPromptArtifact
    cohorts: AstraComparatorCohorts
    batches: tuple[AstraBatch, ...]
    batch_size: int
    code_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.approval, AstraMediumExperimentApproval):
            raise AstraBatchError("plan requires AstraMediumExperimentApproval")
        self.approval.require_identity_approved()
        if not isinstance(self.prompt, AstraPromptArtifact):
            raise AstraBatchError("plan requires AstraPromptArtifact")
        self.prompt.require_approved()
        if self.prompt.experiment_id != self.approval.experiment_id:
            raise AstraBatchError("plan approval and prompt experiment IDs differ")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise AstraBatchError("batch_size must be an integer")
        if self.batch_size <= 0 or self.batch_size > ASTRA_MAX_BATCH_SIZE:
            raise AstraBatchError(f"batch_size must be between 1 and {ASTRA_MAX_BATCH_SIZE}")
        _require_digest(self.code_fingerprint, "code_fingerprint")
        if tuple(item.batch_index for item in self.batches) != tuple(range(len(self.batches))):
            raise AstraBatchError("plan batches must be contiguous and ordered")
        if self.batches and any(item.batch_count != len(self.batches) for item in self.batches):
            raise AstraBatchError("plan batches disagree about batch_count")
        all_uids = [uid for batch in self.batches for uid in batch.study_uids]
        if len(set(all_uids)) != len(all_uids):
            raise AstraBatchError("a study appears in more than one Astra batch")
        expected_records = set(self.cohorts.available_report_uids)
        if set(all_uids) != expected_records:
            raise AstraBatchError("Astra batches do not cover exactly the available comparator reports")
        for batch in self.batches:
            if batch.prompt_fingerprint != self.prompt.fingerprint:
                raise AstraBatchError("batch prompt fingerprint differs from plan prompt")
            if batch.model_identity_fingerprint != self.approval.model_identity_fingerprint:
                raise AstraBatchError("batch model identity differs from plan approval")
            if batch.fold_manifest_checksum != self.cohorts.fold_manifest_checksum:
                raise AstraBatchError("batch fold manifest differs from plan cohorts")
            if batch.report_source_version != self.cohorts.report_source_version:
                raise AstraBatchError("batch report source differs from plan cohorts")
            if batch.report_record_set_sha256 != self.cohorts.report_record_set_sha256:
                raise AstraBatchError("batch report digest differs from plan cohorts")

    @property
    def batch_count(self) -> int:
        return len(self.batches)

    @property
    def unique_study_count(self) -> int:
        return sum(len(batch.records) for batch in self.batches)

    @property
    def plan_fingerprint(self) -> str:
        return _json_sha256(
            {
                "schema_version": ASTRA_HANDOFF_SCHEMA_VERSION,
                "experiment_id": self.approval.experiment_id,
                "approval_fingerprint": self.approval.fingerprint,
                "model_identity_fingerprint": self.approval.model_identity_fingerprint,
                "prompt_fingerprint": self.prompt.fingerprint,
                "output_schema_fingerprint": ASTRA_OUTPUT_SCHEMA_FINGERPRINT,
                "batch_size": self.batch_size,
                "code_fingerprint": self.code_fingerprint,
                "cohort_fingerprint": self.cohorts.fingerprint,
                "batches": [batch.to_private_dict() for batch in self.batches],
            }
        )

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": ASTRA_HANDOFF_SCHEMA_VERSION,
            "status": "prepared_pending_manual_execution",
            "experiment": self.approval.to_public_dict(),
            "prompt": {
                "prompt_id": self.prompt.prompt_id,
                "prompt_version": self.prompt.prompt_version,
                "prompt_fingerprint": self.prompt.fingerprint,
                "prompt_sha256": self.prompt.prompt_sha256,
                "approved": self.prompt.approved,
            },
            "output_schema": {
                "schema_version": ASTRA_OUTPUT_SCHEMA_VERSION,
                "schema_fingerprint": ASTRA_OUTPUT_SCHEMA_FINGERPRINT,
                "target_order": list(TARGET_COLUMNS),
                "accepted_states": list(ASTRA_SEMANTIC_STATES),
                "failure_state": ASTRA_PIPELINE_FAILURE_STATE,
            },
            "batching": {
                "batch_size": self.batch_size,
                "batch_count": self.batch_count,
                "unique_study_count": self.unique_study_count,
                "batch_id_policy": "digest-derived; sorted StudyInstanceUID inputs",
            },
            "cohorts": self.cohorts.to_public_dict(),
            "protocol": {
                "canonical_three_fold_isolation": True,
                "gold_labels_from_frozen_manifest_take_precedence": True,
                "held_out_reports_never_admitted_to_training": True,
                "prompt_tuning_or_adaptation": False,
                "weak_label_training": False,
                "report_only_accuracy_claims": False,
                "luna_owns_validation_aggregation_comparison_metrics": True,
                "raw_report_text_public": False,
                "raw_model_output_public": False,
                "study_identifiers_public": False,
            },
            "fingerprints": {
                "plan": self.plan_fingerprint,
                "code": self.code_fingerprint,
                "approval": self.approval.fingerprint,
                "model_identity": self.approval.model_identity_fingerprint,
                "prompt": self.prompt.fingerprint,
                "output_schema": ASTRA_OUTPUT_SCHEMA_FINGERPRINT,
                "cohorts": self.cohorts.fingerprint,
            },
            "batches": [batch.to_public_dict() for batch in self.batches],
        }

    def to_private_dict(self) -> dict[str, object]:
        payload = self.to_public_dict()
        payload["private"] = {
            "approval": self.approval.to_public_dict(),
            "prompt": self.prompt.to_public_dict(),
            "cohorts": self.cohorts.to_private_dict(),
            "batches": [batch.to_private_dict() for batch in self.batches],
        }
        return payload

    @classmethod
    def from_private_dict(cls, payload: Mapping[str, object]) -> "AstraBatchPlan":
        if payload.get("schema_version") != ASTRA_HANDOFF_SCHEMA_VERSION:
            raise AstraArtifactError("unexpected Astra handoff plan schema version")
        private = _coerce_mapping(payload.get("private"), "private plan")
        approval = AstraMediumExperimentApproval.from_dict(
            _coerce_mapping(private.get("approval"), "private approval")
        )
        prompt = AstraPromptArtifact.from_dict(_coerce_mapping(private.get("prompt"), "private prompt"))
        cohorts = AstraComparatorCohorts.from_dict(
            _coerce_mapping(private.get("cohorts"), "private cohorts")
        )
        raw_batches = private.get("batches")
        if not isinstance(raw_batches, list):
            raise AstraArtifactError("private plan batches must be an array")
        batching = _coerce_mapping(payload.get("batching"), "batching")
        result = cls(
            approval=approval,
            prompt=prompt,
            cohorts=cohorts,
            batches=tuple(AstraBatch.from_dict(_coerce_mapping(item, "private batch")) for item in raw_batches),
            batch_size=_require_positive_int(batching.get("batch_size"), "batch_size"),
            code_fingerprint=_require_digest(
                _coerce_mapping(payload.get("fingerprints"), "fingerprints").get("code"),
                "code_fingerprint",
            ),
        )
        declared = _coerce_mapping(payload.get("fingerprints"), "fingerprints").get("plan")
        if declared != result.plan_fingerprint:
            raise AstraArtifactError("plan fingerprint does not match plan contents")
        return result


def _batch_record_for_membership(
    record: ReportRecord,
    membership_cohort: str,
    assigned_fold: int,
) -> AstraBatchStudy:
    consuming: tuple[int, ...]
    if membership_cohort == "gold":
        consuming = (assigned_fold,)
        partition = ReportPartition.HELD_OUT_EVALUATION.value
    else:
        consuming = tuple(fold for fold in FOLD_IDS if fold != assigned_fold)
        partition = ReportPartition.TRAINING_LABELS.value
    return AstraBatchStudy(
        study_instance_uid=record.study_instance_uid,
        source_report_ref=record.source_report_ref,
        report_sha256=record.report_sha256,
        cohort=membership_cohort,
        assigned_fold=assigned_fold,
        consuming_folds=consuming,
        partitions_by_fold=tuple((fold, partition) for fold in consuming),
        report_text=record.report_text,
    )


def _build_batch(
    records: Sequence[AstraBatchStudy],
    *,
    index: int,
    count: int,
    approval: AstraMediumExperimentApproval,
    prompt: AstraPromptArtifact,
    cohorts: AstraComparatorCohorts,
) -> AstraBatch:
    metadata = {
        "experiment_id": approval.experiment_id,
        "model_identity_fingerprint": approval.model_identity_fingerprint,
        "prompt_fingerprint": prompt.fingerprint,
        "fold_manifest_checksum": cohorts.fold_manifest_checksum,
        "report_source_version": cohorts.report_source_version,
        "report_record_set_sha256": cohorts.report_record_set_sha256,
        "batch_index": index,
        "batch_count": count,
        "records": [item.to_metadata_dict() for item in records],
    }
    input_digest = _json_sha256(metadata)
    return AstraBatch(
        batch_id=f"{ASTRA_BATCH_ID_PREFIX}{index + 1:04d}-{input_digest[:16]}",
        batch_index=index,
        batch_count=count,
        experiment_id=approval.experiment_id,
        model_identity_fingerprint=approval.model_identity_fingerprint,
        prompt_fingerprint=prompt.fingerprint,
        fold_manifest_checksum=cohorts.fold_manifest_checksum,
        report_source_version=cohorts.report_source_version,
        report_record_set_sha256=cohorts.report_record_set_sha256,
        records=tuple(records),
        input_digest=input_digest,
    )


def build_astra_batch_plan(
    manifest: FoldManifest,
    corpus: ReportCorpus,
    approval: AstraMediumExperimentApproval,
    prompt: AstraPromptArtifact,
    *,
    batch_size: int = ASTRA_BATCH_SIZE,
) -> AstraBatchPlan:
    """Prepare stable private batches after all approvals, without model use."""

    if not isinstance(approval, AstraMediumExperimentApproval):
        raise AstraApprovalError("batch planning requires AstraMediumExperimentApproval")
    approval.require_execution_ready(prompt)
    prompt.require_approved()
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise AstraBatchError("batch_size must be an integer")
    if batch_size <= 0 or batch_size > ASTRA_MAX_BATCH_SIZE:
        raise AstraBatchError(f"batch_size must be between 1 and {ASTRA_MAX_BATCH_SIZE}")
    cohorts = build_astra_comparator_cohorts(manifest, corpus)
    membership = {item.study_instance_uid: item for item in manifest.memberships}
    records = tuple(
        _batch_record_for_membership(
            record,
            membership[record.study_instance_uid].cohort,
            membership[record.study_instance_uid].fold,
        )
        for record in sorted(corpus.records, key=lambda item: item.study_instance_uid)
    )
    expected_uids = set(cohorts.available_report_uids)
    if tuple(item.study_instance_uid for item in records) != tuple(sorted(expected_uids)):
        raise AstraBatchError("batch input does not cover exactly the available comparator report set")
    batch_count = (len(records) + batch_size - 1) // batch_size
    batches = tuple(
        _build_batch(
            records[start : start + batch_size],
            index=index,
            count=batch_count,
            approval=approval,
            prompt=prompt,
            cohorts=cohorts,
        )
        for index, start in enumerate(range(0, len(records), batch_size))
    )
    return AstraBatchPlan(
        approval=approval,
        prompt=prompt,
        cohorts=cohorts,
        batches=batches,
        batch_size=batch_size,
        code_fingerprint=ASTRA_HANDOFF_CODE_FINGERPRINT,
    )


@dataclass(frozen=True)
class AstraTargetResult:
    target: str
    state: LabelState

    def __post_init__(self) -> None:
        if self.target not in TARGET_COLUMNS:
            raise AstraOutputValidationError(f"unknown Astra target {self.target!r}")
        if self.state.value not in ASTRA_SEMANTIC_STATES and self.state is not LabelState.EXTRACTION_FAILED:
            raise AstraOutputValidationError("unsupported Astra target state")

    @property
    def value(self) -> int | None:
        return 1 if self.state is LabelState.POSITIVE else 0 if self.state is LabelState.NEGATIVE else None

    @property
    def supervision_mask(self) -> int:
        return int(self.value is not None)

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "state": self.state.value,
            "value": self.value,
            "supervision_mask": self.supervision_mask,
        }


@dataclass(frozen=True)
class AstraValidatedStudy:
    study_instance_uid: str
    report_sha256: str
    cohort: str
    assigned_fold: int
    consuming_folds: tuple[int, ...]
    labels: tuple[AstraTargetResult, ...]

    def __post_init__(self) -> None:
        _require_text(self.study_instance_uid, "study_instance_uid")
        _require_digest(self.report_sha256, "report_sha256")
        if self.cohort not in {"gold", "report_only"}:
            raise AstraOutputValidationError("validated study has an invalid cohort")
        _validate_fold(self.assigned_fold, "assigned_fold")
        _validate_fold_tuple(self.consuming_folds, "consuming_folds")
        if tuple(item.target for item in self.labels) != TARGET_COLUMNS:
            raise AstraOutputValidationError("validated study targets are not canonical")

    @property
    def values(self) -> tuple[int | None, ...]:
        return tuple(item.value for item in self.labels)

    @property
    def supervision_mask(self) -> tuple[int, ...]:
        return tuple(item.supervision_mask for item in self.labels)

    def to_private_dict(self) -> dict[str, object]:
        return {
            "study_instance_uid": self.study_instance_uid,
            "report_sha256": self.report_sha256,
            "cohort": self.cohort,
            "assigned_fold": self.assigned_fold,
            "consuming_folds": list(self.consuming_folds),
            "labels": [item.to_dict() for item in self.labels],
        }

    def to_public_dict(self) -> dict[str, object]:
        return {
            "state_counts": {
                state: sum(item.state.value == state for item in self.labels)
                for state in (*ASTRA_SEMANTIC_STATES, ASTRA_PIPELINE_FAILURE_STATE)
            },
            "supervised_entry_count": sum(self.supervision_mask),
        }


def _failed_labels() -> tuple[AstraTargetResult, ...]:
    return tuple(
        AstraTargetResult(target=target, state=LabelState.EXTRACTION_FAILED)
        for target in TARGET_COLUMNS
    )


def _semantic_digest_for_validated(
    batch: AstraBatch,
    status: str,
    failure_code: str | None,
    studies: Sequence[AstraValidatedStudy],
) -> str:
    return _json_sha256(
        {
            "schema_version": ASTRA_OUTPUT_SCHEMA_VERSION,
            "experiment_id": batch.experiment_id,
            "model_identity_fingerprint": batch.model_identity_fingerprint,
            "reasoning_effort": ASTRA_REASONING_EFFORT,
            "effort": ASTRA_EFFORT,
            "prompt_fingerprint": batch.prompt_fingerprint,
            "batch_id": batch.batch_id,
            "input_digest": batch.input_digest,
            "status": status,
            "failure_code": failure_code,
            "studies": [
                {
                    "study_instance_uid": study.study_instance_uid,
                    "report_sha256": study.report_sha256,
                    "states": [item.state.value for item in study.labels],
                    "values": list(study.values),
                    "supervision_mask": list(study.supervision_mask),
                }
                for study in studies
            ],
        }
    )


@dataclass(frozen=True)
class AstraValidatedBatch:
    batch: AstraBatch
    status: str
    raw_output_sha256: str
    response_char_count: int
    studies: tuple[AstraValidatedStudy, ...]
    failure_code: str | None
    semantic_digest: str

    def __post_init__(self) -> None:
        if self.status not in {"validated", "extraction-failed"}:
            raise AstraOutputValidationError("validated batch status is invalid")
        _require_digest(self.raw_output_sha256, "raw_output_sha256")
        _require_nonnegative_int(self.response_char_count, "response_char_count")
        if tuple(item.study_instance_uid for item in self.studies) != self.batch.study_uids:
            raise AstraOutputValidationError("validated batch study set/order differs from input batch")
        if self.status == "validated" and self.failure_code is not None:
            raise AstraOutputValidationError("validated batch cannot carry a failure code")
        if self.status == "extraction-failed":
            if not self.failure_code:
                raise AstraOutputValidationError("failed batch requires a failure code")
            if any(item.state is not LabelState.EXTRACTION_FAILED for study in self.studies for item in study.labels):
                raise AstraOutputValidationError("failed batch must mask every target as extraction-failed")
        _require_digest(self.semantic_digest, "semantic_digest")
        expected = _semantic_digest_for_validated(
            self.batch,
            self.status,
            self.failure_code,
            self.studies,
        )
        if expected != self.semantic_digest:
            raise AstraOutputValidationError("semantic output digest does not match validated output")

    @property
    def is_complete(self) -> bool:
        return True

    def to_private_dict(self) -> dict[str, object]:
        return {
            "schema_version": ASTRA_AUDIT_SCHEMA_VERSION,
            "batch_id": self.batch.batch_id,
            "status": self.status,
            "raw_output_sha256": self.raw_output_sha256,
            "response_char_count": self.response_char_count,
            "failure_code": self.failure_code,
            "semantic_output_digest": self.semantic_digest,
            "experiment_id": self.batch.experiment_id,
            "model_identity_fingerprint": self.batch.model_identity_fingerprint,
            "reasoning_effort": ASTRA_REASONING_EFFORT,
            "effort": ASTRA_EFFORT,
            "prompt_fingerprint": self.batch.prompt_fingerprint,
            "input_digest": self.batch.input_digest,
            "studies": [item.to_private_dict() for item in self.studies],
        }

    def to_public_dict(self) -> dict[str, object]:
        return {
            "batch_id": self.batch.batch_id,
            "status": self.status,
            "study_count": len(self.studies),
            "raw_output_sha256": self.raw_output_sha256,
            "response_char_count": self.response_char_count,
            "failure_code": self.failure_code,
            "semantic_output_digest": self.semantic_digest,
        }


def _failed_batch(
    batch: AstraBatch,
    *,
    raw_output_sha256: str,
    response_char_count: int,
    failure_code: str,
) -> AstraValidatedBatch:
    studies = tuple(
        AstraValidatedStudy(
            study_instance_uid=record.study_instance_uid,
            report_sha256=record.report_sha256,
            cohort=record.cohort,
            assigned_fold=record.assigned_fold,
            consuming_folds=record.consuming_folds,
            labels=_failed_labels(),
        )
        for record in batch.records
    )
    return AstraValidatedBatch(
        batch=batch,
        status="extraction-failed",
        raw_output_sha256=raw_output_sha256,
        response_char_count=response_char_count,
        studies=studies,
        failure_code=failure_code,
        semantic_digest=_semantic_digest_for_validated(
            batch,
            "extraction-failed",
            failure_code,
            studies,
        ),
    )


def _context_check(payload: Mapping[str, object], batch: AstraBatch) -> None:
    expected = {
        "schema_version": ASTRA_OUTPUT_SCHEMA_VERSION,
        "experiment_id": batch.experiment_id,
        "model_identity_fingerprint": batch.model_identity_fingerprint,
        "reasoning_effort": ASTRA_REASONING_EFFORT,
        "effort": ASTRA_EFFORT,
        "prompt_fingerprint": batch.prompt_fingerprint,
        "batch_id": batch.batch_id,
        "input_digest": batch.input_digest,
    }
    for field_name, expected_value in expected.items():
        if field_name in payload and payload[field_name] != expected_value:
            raise AstraContextMismatchError(
                f"Astra output {field_name} does not match the planned batch"
            )


def _malformed_code(exc: AstraHandoffError) -> str:
    message = str(exc).lower()
    if "json" in message:
        return "invalid-json"
    if "schema" in message or "keys" in message:
        return "invalid-schema"
    if "duplicate" in message:
        return "duplicate-study-or-target"
    if "missing" in message:
        return "missing-study-or-target"
    if "state" in message:
        return "invalid-state"
    if "target" in message:
        return "invalid-target-set"
    return "invalid-output"


def validate_astra_output(raw_output: str, batch: AstraBatch) -> AstraValidatedBatch:
    """Validate one exact response; malformed output becomes all-masked failure.

    A response with a different prompt, model, effort, experiment, batch ID or
    input digest raises :class:`AstraContextMismatchError` instead of being
    accepted as a failure for this batch.  That distinction prevents a valid
    response from another experiment from entering this ledger.
    """

    if not isinstance(raw_output, str):
        raise AstraOutputValidationError("raw Astra output must be text")
    if not isinstance(batch, AstraBatch):
        raise AstraOutputValidationError("Astra validation requires AstraBatch")
    raw_digest = _text_sha256(raw_output)
    try:
        parsed = json.loads(raw_output)
        payload = _coerce_mapping(parsed, "Astra output")
        _exact_keys(
            payload,
            {
                "schema_version",
                "experiment_id",
                "model_identity_fingerprint",
                "reasoning_effort",
                "effort",
                "prompt_fingerprint",
                "batch_id",
                "input_digest",
                "studies",
            },
            "Astra output",
        )
        _context_check(payload, batch)
        if payload.get("schema_version") != ASTRA_OUTPUT_SCHEMA_VERSION:
            raise AstraOutputValidationError("schema version is invalid")
        raw_studies = payload.get("studies")
        if not isinstance(raw_studies, list):
            raise AstraOutputValidationError("studies must be an array")
        seen: set[str] = set()
        study_payloads: list[Mapping[str, object]] = []
        for raw_study in raw_studies:
            study = _coerce_mapping(raw_study, "study output")
            _exact_keys(study, {"study_instance_uid", "targets"}, "study output")
            uid = _require_text(study.get("study_instance_uid"), "study_instance_uid")
            if uid in seen:
                raise AstraOutputValidationError(f"duplicate study {uid}")
            seen.add(uid)
            study_payloads.append(study)
        if seen != set(batch.study_uids):
            missing = set(batch.study_uids) - seen
            extra = seen - set(batch.study_uids)
            if missing:
                raise AstraOutputValidationError(f"missing studies {sorted(missing)[:3]}")
            raise AstraOutputValidationError(f"unknown studies {sorted(extra)[:3]}")
        if tuple(study.get("study_instance_uid") for study in study_payloads) != batch.study_uids:
            raise AstraOutputValidationError("study output order differs from the batch")

        parsed_by_uid: dict[str, tuple[AstraTargetResult, ...]] = {}
        for study in study_payloads:
            uid = cast(str, study["study_instance_uid"])
            raw_targets = study.get("targets")
            if not isinstance(raw_targets, list):
                raise AstraOutputValidationError(f"targets for {uid} must be an array")
            labels: list[AstraTargetResult] = []
            seen_targets: set[str] = set()
            for raw_target in raw_targets:
                target = _coerce_mapping(raw_target, "target output")
                _exact_keys(target, {"target", "state"}, "target output")
                target_name = _require_text(target.get("target"), "target")
                if target_name in seen_targets:
                    raise AstraOutputValidationError(f"duplicate target {target_name}")
                seen_targets.add(target_name)
                if target_name not in TARGET_COLUMNS:
                    raise AstraOutputValidationError(f"unknown target {target_name}")
                state_value = target.get("state")
                if state_value not in ASTRA_SEMANTIC_STATES:
                    raise AstraOutputValidationError(
                        "model output may not emit extraction-failed as a semantic state"
                    )
                labels.append(
                    AstraTargetResult(
                        target=target_name,
                        state=LabelState(cast(str, state_value)),
                    )
                )
            if tuple(item.target for item in labels) != TARGET_COLUMNS:
                raise AstraOutputValidationError(f"target set/order for {uid} is incomplete")
            parsed_by_uid[uid] = tuple(labels)

        records = tuple(
            AstraValidatedStudy(
                study_instance_uid=record.study_instance_uid,
                report_sha256=record.report_sha256,
                cohort=record.cohort,
                assigned_fold=record.assigned_fold,
                consuming_folds=record.consuming_folds,
                labels=parsed_by_uid[record.study_instance_uid],
            )
            for record in batch.records
        )
        return AstraValidatedBatch(
            batch=batch,
            status="validated",
            raw_output_sha256=raw_digest,
            response_char_count=len(raw_output),
            studies=records,
            failure_code=None,
            semantic_digest=_semantic_digest_for_validated(batch, "validated", None, records),
        )
    except json.JSONDecodeError:
        return _failed_batch(
            batch,
            raw_output_sha256=raw_digest,
            response_char_count=len(raw_output),
            failure_code="invalid-json",
        )
    except AstraContextMismatchError:
        raise
    except AstraHandoffError as exc:
        return _failed_batch(
            batch,
            raw_output_sha256=raw_digest,
            response_char_count=len(raw_output),
            failure_code=_malformed_code(exc),
        )
    except (TypeError, ValueError):
        return _failed_batch(
            batch,
            raw_output_sha256=raw_digest,
            response_char_count=len(raw_output),
            failure_code="invalid-output",
        )


@dataclass(frozen=True)
class AstraBatchAudit:
    """Immutable private ledger entry for one raw response attempt."""

    batch_id: str
    attempt: int
    status: str
    raw_output_sha256: str
    response_char_count: int
    semantic_output_digest: str | None
    failure_code: str | None
    experiment_id: str
    model_identity_fingerprint: str
    reasoning_effort: str
    effort: str
    prompt_fingerprint: str
    input_digest: str
    studies: tuple[AstraValidatedStudy, ...] = ()
    rejected_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"validated", "extraction-failed", "rejected"}:
            raise AstraArtifactError("audit status is invalid")
        _require_positive_int(self.attempt, "attempt")
        _require_digest(self.raw_output_sha256, "raw_output_sha256")
        _require_nonnegative_int(self.response_char_count, "response_char_count")
        _require_digest(self.model_identity_fingerprint, "model_identity_fingerprint")
        _require_digest(self.prompt_fingerprint, "prompt_fingerprint")
        _require_digest(self.input_digest, "input_digest")
        if self.experiment_id != ASTRA_EXPERIMENT_ID:
            raise AstraArtifactError("audit experiment is not Ticket 16 Astra")
        if self.reasoning_effort != ASTRA_REASONING_EFFORT or self.effort != ASTRA_EFFORT:
            raise AstraArtifactError("audit effort is not fixed to medium")
        if self.status == "rejected" and not self.rejected_reason:
            raise AstraArtifactError("rejected audit requires a reason")
        if self.status != "rejected" and self.semantic_output_digest is None:
            raise AstraArtifactError("completed audit requires a semantic output digest")
        if self.semantic_output_digest is not None:
            _require_digest(self.semantic_output_digest, "semantic_output_digest")

    @property
    def completed(self) -> bool:
        return self.status in {"validated", "extraction-failed"}

    def to_private_dict(self) -> dict[str, object]:
        return {
            "schema_version": ASTRA_AUDIT_SCHEMA_VERSION,
            "batch_id": self.batch_id,
            "attempt": self.attempt,
            "status": self.status,
            "raw_output_sha256": self.raw_output_sha256,
            "response_char_count": self.response_char_count,
            "semantic_output_digest": self.semantic_output_digest,
            "failure_code": self.failure_code,
            "rejected_reason": self.rejected_reason,
            "experiment_id": self.experiment_id,
            "model_identity_fingerprint": self.model_identity_fingerprint,
            "reasoning_effort": self.reasoning_effort,
            "effort": self.effort,
            "prompt_fingerprint": self.prompt_fingerprint,
            "input_digest": self.input_digest,
            "studies": [item.to_private_dict() for item in self.studies],
            "raw_report_text_persisted_in_audit": False,
        }

    def to_public_dict(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "attempt": self.attempt,
            "status": self.status,
            "raw_output_sha256": self.raw_output_sha256,
            "response_char_count": self.response_char_count,
            "semantic_output_digest": self.semantic_output_digest,
            "failure_code": self.failure_code,
        }


def _audit_from_validated(result: AstraValidatedBatch, attempt: int) -> AstraBatchAudit:
    return AstraBatchAudit(
        batch_id=result.batch.batch_id,
        attempt=attempt,
        status=result.status,
        raw_output_sha256=result.raw_output_sha256,
        response_char_count=result.response_char_count,
        semantic_output_digest=result.semantic_digest,
        failure_code=result.failure_code,
        experiment_id=result.batch.experiment_id,
        model_identity_fingerprint=result.batch.model_identity_fingerprint,
        reasoning_effort=ASTRA_REASONING_EFFORT,
        effort=ASTRA_EFFORT,
        prompt_fingerprint=result.batch.prompt_fingerprint,
        input_digest=result.batch.input_digest,
        studies=result.studies,
    )


def _rejected_audit(batch: AstraBatch, raw_output: str, attempt: int, reason: str) -> AstraBatchAudit:
    return AstraBatchAudit(
        batch_id=batch.batch_id,
        attempt=attempt,
        status="rejected",
        raw_output_sha256=_text_sha256(raw_output),
        response_char_count=len(raw_output),
        semantic_output_digest=None,
        failure_code="context-mismatch",
        experiment_id=batch.experiment_id,
        model_identity_fingerprint=batch.model_identity_fingerprint,
        reasoning_effort=ASTRA_REASONING_EFFORT,
        effort=ASTRA_EFFORT,
        prompt_fingerprint=batch.prompt_fingerprint,
        input_digest=batch.input_digest,
        rejected_reason=reason,
    )


def _audit_paths(batch_dir: Path) -> tuple[Path, ...]:
    return tuple(sorted(batch_dir.glob("attempt-*.audit.json")))


def _load_audit(path: Path) -> AstraBatchAudit:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AstraArtifactError(f"could not read Astra audit artifact: {path}") from exc
    data = _coerce_mapping(payload, "audit")
    raw_studies = data.get("studies", [])
    if not isinstance(raw_studies, list):
        raise AstraArtifactError("audit studies must be an array")
    studies: list[AstraValidatedStudy] = []
    for raw in raw_studies:
        item = _coerce_mapping(raw, "audit study")
        raw_labels = item.get("labels")
        if not isinstance(raw_labels, list):
            raise AstraArtifactError("audit study labels must be an array")
        labels = []
        for raw_label in raw_labels:
            label = _coerce_mapping(raw_label, "audit label")
            labels.append(
                AstraTargetResult(
                    target=_require_text(label.get("target"), "audit label target"),
                    state=LabelState(_require_text(label.get("state"), "audit label state")),
                )
            )
        raw_consuming = item.get("consuming_folds")
        if not isinstance(raw_consuming, list):
            raise AstraArtifactError("audit consuming_folds must be an array")
        studies.append(
            AstraValidatedStudy(
                study_instance_uid=_require_text(item.get("study_instance_uid"), "audit study UID"),
                report_sha256=_require_digest(item.get("report_sha256"), "audit report_sha256"),
                cohort=_require_text(item.get("cohort"), "audit cohort"),
                assigned_fold=_validate_fold(item.get("assigned_fold"), "audit assigned_fold"),
                consuming_folds=tuple(
                    _validate_fold(value, "audit consuming fold")
                    for value in raw_consuming
                ),
                labels=tuple(labels),
            )
        )
    result = AstraBatchAudit(
        batch_id=_require_text(data.get("batch_id"), "audit batch_id"),
        attempt=_require_positive_int(data.get("attempt"), "audit attempt"),
        status=_require_text(data.get("status"), "audit status"),
        raw_output_sha256=_require_digest(data.get("raw_output_sha256"), "audit raw output digest"),
        response_char_count=_require_nonnegative_int(
            data.get("response_char_count"), "audit response_char_count"
        ),
        semantic_output_digest=(
            _require_digest(data.get("semantic_output_digest"), "audit semantic digest")
            if data.get("semantic_output_digest") is not None
            else None
        ),
        failure_code=(str(data["failure_code"]) if data.get("failure_code") is not None else None),
        rejected_reason=(
            str(data["rejected_reason"]) if data.get("rejected_reason") is not None else None
        ),
        experiment_id=_require_text(data.get("experiment_id"), "audit experiment_id"),
        model_identity_fingerprint=_require_digest(
            data.get("model_identity_fingerprint"), "audit model identity"
        ),
        reasoning_effort=_require_text(data.get("reasoning_effort"), "audit reasoning_effort"),
        effort=_require_text(data.get("effort"), "audit effort"),
        prompt_fingerprint=_require_digest(data.get("prompt_fingerprint"), "audit prompt fingerprint"),
        input_digest=_require_digest(data.get("input_digest"), "audit input digest"),
        studies=tuple(studies),
    )
    if data.get("schema_version") != ASTRA_AUDIT_SCHEMA_VERSION:
        raise AstraArtifactError("unexpected audit schema version")
    return result


def _existing_audits(plan: AstraBatchPlan, private_dir: Path, batch: AstraBatch) -> tuple[AstraBatchAudit, ...]:
    batch_dir = private_dir / "batches" / batch.batch_id
    audits_with_paths = tuple((path, _load_audit(path)) for path in _audit_paths(batch_dir))
    audits = tuple(audit for _, audit in audits_with_paths)
    for audit in audits:
        if audit.batch_id != batch.batch_id:
            raise AstraArtifactError("audit batch ID does not match its directory")
        if audit.experiment_id != plan.approval.experiment_id:
            raise AstraArtifactError("audit experiment does not match plan")
        if audit.model_identity_fingerprint != plan.approval.model_identity_fingerprint:
            raise AstraArtifactError("audit model identity does not match plan")
        if audit.prompt_fingerprint != plan.prompt.fingerprint:
            raise AstraArtifactError("audit prompt does not match plan")
        if audit.input_digest != batch.input_digest:
            raise AstraArtifactError("audit input digest does not match plan")
    for path, audit in audits_with_paths:
        raw_path = path.with_name(path.name.replace(".audit.json", ".raw.json"))
        if not raw_path.is_file() or raw_path.is_symlink():
            raise AstraArtifactError(f"audit is missing its immutable raw response: {raw_path}")
        raw_text = raw_path.read_text(encoding="utf-8")
        if _text_sha256(raw_text) != audit.raw_output_sha256:
            raise AstraArtifactError("raw response digest does not match its audit")
        if audit.status != "rejected":
            expected_uids = batch.study_uids
            if tuple(item.study_instance_uid for item in audit.studies) != expected_uids:
                raise AstraArtifactError("audit study set does not match its batch")
            expected_records = {item.study_instance_uid: item for item in batch.records}
            for item in audit.studies:
                source = expected_records[item.study_instance_uid]
                if item.report_sha256 != source.report_sha256:
                    raise AstraArtifactError("audit report digest does not match its batch")
                if item.cohort != source.cohort or item.assigned_fold != source.assigned_fold:
                    raise AstraArtifactError("audit fold/cohort metadata does not match its batch")
                if item.consuming_folds != source.consuming_folds:
                    raise AstraArtifactError("audit consuming folds do not match its batch")
            if audit.status == "extraction-failed":
                if any(
                    label.state is not LabelState.EXTRACTION_FAILED
                    for item in audit.studies
                    for label in item.labels
                ):
                    raise AstraArtifactError("failed audit contains a non-failure label")
            else:
                if any(
                    label.state is LabelState.EXTRACTION_FAILED
                    for item in audit.studies
                    for label in item.labels
                ):
                    raise AstraArtifactError("validated audit contains an extraction failure")
            expected_semantic = _semantic_digest_for_validated(
                batch,
                audit.status,
                audit.failure_code,
                audit.studies,
            )
            if audit.semantic_output_digest != expected_semantic:
                raise AstraArtifactError("audit semantic digest does not match its states")
    return audits


def completed_astra_batch_ids(plan: AstraBatchPlan, private_dir: str | Path) -> tuple[str, ...]:
    root = Path(private_dir)
    completed: list[str] = []
    for batch in plan.batches:
        audits = _existing_audits(plan, root, batch)
        complete = [audit for audit in audits if audit.completed]
        if len(complete) > 1:
            raise AstraResumeError(f"batch {batch.batch_id} has multiple completed attempts")
        if complete:
            completed.append(batch.batch_id)
    return tuple(completed)


def pending_astra_batches(plan: AstraBatchPlan, private_dir: str | Path) -> tuple[AstraBatch, ...]:
    completed = set(completed_astra_batch_ids(plan, private_dir))
    return tuple(batch for batch in plan.batches if batch.batch_id not in completed)


def record_astra_output(
    plan: AstraBatchPlan,
    batch_id: str,
    raw_output: str,
    private_dir: str | Path,
    *,
    attempt: int = 1,
) -> AstraBatchAudit:
    """Persist one verbatim response and its immutable validation audit.

    A validated or all-masked failure is a completed batch.  It is never
    repeated by :func:`pending_astra_batches`; a retry must be explicit and
    cannot overwrite the first attempt.  A context-mismatched response is
    preserved as ``rejected`` and the batch remains pending.
    """

    if not isinstance(raw_output, str):
        raise AstraOutputValidationError("raw Astra output must be text")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
        raise AstraResumeError("attempt must be a positive integer")
    batch = next((item for item in plan.batches if item.batch_id == batch_id), None)
    if batch is None:
        raise AstraResumeError(f"unknown Astra batch ID {batch_id!r}")
    root = Path(private_dir)
    existing = _existing_audits(plan, root, batch)
    if any(item.completed for item in existing):
        raise AstraResumeError(f"batch {batch_id} is already complete; resume will not repeat it")
    if any(item.attempt == attempt for item in existing):
        raise AstraResumeError(f"Astra attempt {attempt} already exists for batch {batch_id}")
    batch_dir = root / "batches" / batch.batch_id
    raw_path = batch_dir / f"attempt-{attempt:04d}.raw.json"
    audit_path = batch_dir / f"attempt-{attempt:04d}.audit.json"
    _write_immutable_text(raw_path, raw_output)
    try:
        result = validate_astra_output(raw_output, batch)
    except AstraContextMismatchError as exc:
        audit = _rejected_audit(batch, raw_output, attempt, str(exc))
        _write_immutable_json(audit_path, audit.to_private_dict())
        raise
    audit = _audit_from_validated(result, attempt)
    _write_immutable_json(audit_path, audit.to_private_dict())
    return audit


def _load_plan(path: str | Path) -> AstraBatchPlan:
    candidate = Path(path)
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AstraArtifactError(f"could not read Astra plan: {candidate}") from exc
    return AstraBatchPlan.from_private_dict(_coerce_mapping(payload, "plan"))


def write_astra_batch_plan(
    plan: AstraBatchPlan,
    *,
    public_path: str | Path,
    private_dir: str | Path,
) -> tuple[Path, Path]:
    """Write public metadata, private plan metadata, and private input files."""

    if not isinstance(plan, AstraBatchPlan):
        raise AstraArtifactError("plan writer requires AstraBatchPlan")
    public = Path(public_path)
    private = Path(private_dir)
    _validate_private_paths(public, private)
    private_plan = private / "plan.json"
    if public.exists() or public.is_symlink() or private_plan.exists() or private_plan.is_symlink():
        raise FileExistsError("Astra plan artifacts are immutable and already exist")
    for batch in plan.batches:
        input_path = private / "batches" / batch.batch_id / "input.json"
        if input_path.exists() or input_path.is_symlink():
            raise FileExistsError(f"Astra input artifact already exists: {input_path}")
    _write_immutable_json(public, plan.to_public_dict())
    try:
        for batch in plan.batches:
            input_payload = batch.to_codex_input_dict(plan.prompt)
            _write_immutable_json(private / "batches" / batch.batch_id / "input.json", input_payload)
        _write_immutable_json(private_plan, plan.to_private_dict())
    except BaseException:
        # The public plan is not useful without its private counterpart.  Do
        # not remove private inputs: they are immutable evidence of what was
        # prepared and may need manual cleanup by the owner.
        try:
            public.unlink()
        except FileNotFoundError:
            pass
        raise
    return public, private_plan


def load_astra_batch_plan(path: str | Path) -> AstraBatchPlan:
    return _load_plan(path)


def load_astra_audits(plan: AstraBatchPlan, private_dir: str | Path) -> tuple[AstraBatchAudit, ...]:
    audits: list[AstraBatchAudit] = []
    for batch in plan.batches:
        batch_audits = _existing_audits(plan, Path(private_dir), batch)
        if sum(item.completed for item in batch_audits) > 1:
            raise AstraResumeError(f"batch {batch.batch_id} has multiple completed attempts")
        audits.extend(batch_audits)
    return tuple(sorted(audits, key=lambda item: (item.batch_id, item.attempt)))


def write_astra_public_evidence(
    plan: AstraBatchPlan,
    private_dir: str | Path,
    public_path: str | Path,
) -> Path:
    """Publish text-free status only; no accuracy or report-only claim is emitted."""

    audits = load_astra_audits(plan, private_dir)
    completed = [audit for audit in audits if audit.completed]
    rejected = [audit for audit in audits if audit.status == "rejected"]
    completed_by_batch = {audit.batch_id: audit for audit in completed}
    payload = {
        "schema_version": ASTRA_PUBLIC_EVIDENCE_SCHEMA_VERSION,
        "status": (
            "ready_for_luna_validation"
            if len(completed_by_batch) == plan.batch_count and plan.batch_count > 0
            else "partially_completed"
            if completed_by_batch
            else "prepared_pending_manual_execution"
        ),
        "experiment": plan.approval.to_public_dict(),
        "plan": {
            "plan_fingerprint": plan.plan_fingerprint,
            "batch_count": plan.batch_count,
            "unique_study_count": plan.unique_study_count,
            "completed_batch_count": len(completed_by_batch),
            "pending_batch_count": plan.batch_count - len(completed_by_batch),
            "rejected_attempt_count": len(rejected),
        },
        "fingerprints": {
            "code": plan.code_fingerprint,
            "approval": plan.approval.fingerprint,
            "model_identity": plan.approval.model_identity_fingerprint,
            "prompt": plan.prompt.fingerprint,
            "output_schema": ASTRA_OUTPUT_SCHEMA_FINGERPRINT,
            "cohorts": plan.cohorts.fingerprint,
        },
        "batches": [
            completed_by_batch[batch.batch_id].to_public_dict()
            if batch.batch_id in completed_by_batch
            else {"batch_id": batch.batch_id, "status": "pending", "study_count": len(batch.records)}
            for batch in plan.batches
        ],
        "protocol": {
            "raw_report_text_public": False,
            "raw_model_output_public": False,
            "study_identifiers_public": False,
            "gold_labels_from_frozen_manifest_take_precedence": True,
            "held_out_reports_used_for_prompt_tuning": False,
            "weak_label_training_performed": False,
            "report_only_accuracy_claimed": False,
            "luna_owns_aggregation_comparison_and_metrics": True,
        },
    }
    return _write_immutable_json(public_path, payload)


def expected_astra_batch_count(unique_report_count: int, *, batch_size: int = ASTRA_BATCH_SIZE) -> int:
    """Return the stable planning formula without opening a report corpus."""

    _require_nonnegative_int(unique_report_count, "unique_report_count")
    if batch_size <= 0 or batch_size > ASTRA_MAX_BATCH_SIZE:
        raise AstraBatchError(f"batch_size must be between 1 and {ASTRA_MAX_BATCH_SIZE}")
    return (unique_report_count + batch_size - 1) // batch_size


def astra_output_example(batch: AstraBatch) -> dict[str, object]:
    """Return a structural example for documentation/tests, never a label."""

    return {
        "schema_version": ASTRA_OUTPUT_SCHEMA_VERSION,
        "experiment_id": batch.experiment_id,
        "model_identity_fingerprint": batch.model_identity_fingerprint,
        "reasoning_effort": ASTRA_REASONING_EFFORT,
        "effort": ASTRA_EFFORT,
        "prompt_fingerprint": batch.prompt_fingerprint,
        "batch_id": batch.batch_id,
        "input_digest": batch.input_digest,
        "studies": [
            {
                "study_instance_uid": record.study_instance_uid,
                "targets": [
                    {"target": target, "state": LabelState.UNKNOWN.value}
                    for target in TARGET_COLUMNS
                ],
            }
            for record in batch.records
        ],
    }


# This is deliberately calculated from this module only.  It is included in
# every plan so that code changes cannot silently reuse an old handoff.
ASTRA_HANDOFF_CODE_FINGERPRINT = _text_sha256(Path(__file__).read_text(encoding="utf-8"))


__all__ = [
    "ASTRA_APPROVAL_DATE",
    "ASTRA_APPROVAL_SCHEMA_VERSION",
    "ASTRA_AUDIT_SCHEMA_VERSION",
    "ASTRA_BATCH_ID_PREFIX",
    "ASTRA_BATCH_SCHEMA_VERSION",
    "ASTRA_BATCH_SIZE",
    "ASTRA_DATA_TRANSFER_POLICY",
    "ASTRA_EFFORT",
    "ASTRA_EXPERIMENT_ID",
    "ASTRA_HANDOFF_CODE_FINGERPRINT",
    "ASTRA_HANDOFF_SCHEMA_VERSION",
    "ASTRA_MAX_BATCH_SIZE",
    "ASTRA_MODEL_NAME",
    "ASTRA_MODEL_REVISION",
    "ASTRA_MODEL_SURFACE",
    "ASTRA_OUTPUT_JSON_SCHEMA",
    "ASTRA_OUTPUT_SCHEMA_FINGERPRINT",
    "ASTRA_OUTPUT_SCHEMA_VERSION",
    "ASTRA_PIPELINE_FAILURE_STATE",
    "ASTRA_PROMPT_SCHEMA_VERSION",
    "ASTRA_REASONING_EFFORT",
    "ASTRA_SEMANTIC_STATES",
    "AstraApprovalError",
    "AstraArtifactError",
    "AstraBatch",
    "AstraBatchAudit",
    "AstraBatchError",
    "AstraBatchPlan",
    "AstraBatchStudy",
    "AstraComparatorCohorts",
    "AstraContextMismatchError",
    "AstraFoldCohort",
    "AstraHandoffError",
    "AstraMediumExperimentApproval",
    "AstraOutputValidationError",
    "AstraPromptArtifact",
    "AstraPromptError",
    "AstraResumeError",
    "AstraTargetResult",
    "AstraValidatedBatch",
    "AstraValidatedStudy",
    "assert_astra_cohorts_match_ticket15",
    "astra_medium_experiment_approval",
    "astra_output_example",
    "build_astra_batch_plan",
    "build_astra_comparator_cohorts",
    "completed_astra_batch_ids",
    "expected_astra_batch_count",
    "load_astra_approval",
    "load_astra_audits",
    "load_astra_batch_plan",
    "load_astra_prompt",
    "make_astra_prompt_artifact",
    "pending_astra_batches",
    "record_astra_output",
    "validate_astra_output",
    "write_astra_approval",
    "write_astra_batch_plan",
    "write_astra_prompt",
    "write_astra_public_evidence",
]
