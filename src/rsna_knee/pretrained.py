"""Admission and checksum verification for the proposed public ResNet-18 weights."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Mapping

import torch


PRETRAINED_APPROVAL_SCHEMA = "rsna-ticket31-resource-approval-v1"
RESNET18_APPROVAL_SCHEMA = "rsna-resnet18-resource-approval-v1"
RESNET18_ARTIFACT_ID = "torchvision.ResNet18_Weights.IMAGENET1K_V1"
RESNET18_SOURCE_REVISION = "pytorch/vision@v0.22.1"
RESNET18_DOWNLOAD_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
RESNET18_ARTIFACT_BYTES = 46_830_571
RESNET18_RESOURCE_SCOPE = "ticket31-one-reference-fold-cpu"
TICKET32_RESOURCE_SCOPE = "ticket32-three-fold-reference-oof-cpu"
TICKET34_RESOURCE_SCOPE = "ticket34-fold0-report-supervision-cpu"
TICKET36_RESOURCE_SCOPE = "ticket36-expanded-three-fold-report-supervision-cpu"
TICKET33_RESOURCE_SCOPE = "ticket33-fold0-resnet18-gpu-experiment"
TICKET39_RESOURCE_SCOPE = "ticket39-layer4-resnet18-finetune-experiment"
_PRETRAINED_APPROVAL_SCHEMAS = frozenset(
    {
        PRETRAINED_APPROVAL_SCHEMA,
        RESNET18_APPROVAL_SCHEMA,
        "rsna-ticket36-resource-approval-v1",
    }
)
_RESNET18_RESOURCE_SCOPES = frozenset(
    {
        RESNET18_RESOURCE_SCOPE,
        TICKET32_RESOURCE_SCOPE,
        TICKET34_RESOURCE_SCOPE,
        TICKET36_RESOURCE_SCOPE,
        TICKET33_RESOURCE_SCOPE,
        TICKET39_RESOURCE_SCOPE,
    }
)


class PretrainedResourceError(ValueError):
    """Raised when a pretrained model is not admitted for this exact experiment."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_resource_approval(path: str | Path) -> Mapping[str, object]:
    """Require an explicit, artifact-specific owner decision before acquisition/use."""

    approval_path = Path(path)
    try:
        payload = json.loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PretrainedResourceError("resource approval could not be read") from exc
    if not isinstance(payload, Mapping):
        raise PretrainedResourceError("resource approval must be a JSON object")
    required = {
        "artifact_id": RESNET18_ARTIFACT_ID,
        "source_revision": RESNET18_SOURCE_REVISION,
        "download_url": RESNET18_DOWNLOAD_URL,
        "expected_bytes": RESNET18_ARTIFACT_BYTES,
        "sha256": RESNET18_SHA256,
        "decision": "approved",
        "rights_status": "eligible",
        "external_spend": 0,
    }
    if payload.get("schema_version") not in _PRETRAINED_APPROVAL_SCHEMAS:
        raise PretrainedResourceError("resource approval schema is unsupported")
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise PretrainedResourceError(
                f"resource approval does not establish {key}={expected!r}"
            )
    for key in ("expected_bytes", "external_spend"):
        numeric = payload.get(key)
        if not isinstance(numeric, int) or isinstance(numeric, bool):
            raise PretrainedResourceError(f"resource approval field {key} must be an integer")
    if payload.get("scope") not in _RESNET18_RESOURCE_SCOPES:
        raise PretrainedResourceError("resource approval scope is not admitted for ResNet-18")
    if payload.get("scope") == TICKET33_RESOURCE_SCOPE and payload.get(
        "rights_status_scope"
    ) != TICKET33_RESOURCE_SCOPE:
        raise PretrainedResourceError("Ticket 33 rights status is not scope-limited")
    if payload.get("scope") == TICKET39_RESOURCE_SCOPE:
        if payload.get("rights_status_scope") != TICKET39_RESOURCE_SCOPE:
            raise PretrainedResourceError("Ticket 39 rights status is not scope-limited")
        if payload.get("encoder_frozen") is not False:
            raise PretrainedResourceError("Ticket 39 approval must disclose encoder fine-tuning")
        if payload.get("trainable_encoder_portion") != "layer4":
            raise PretrainedResourceError("Ticket 39 approval must limit training to layer4")
    approved_by = payload.get("approved_by")
    approved_at = payload.get("approved_at")
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise PretrainedResourceError("resource approval must identify its approver")
    if not isinstance(approved_at, str) or not approved_at.strip():
        raise PretrainedResourceError("resource approval must include an approval timestamp")
    try:
        datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PretrainedResourceError("resource approval timestamp is invalid") from exc
    rights_basis = payload.get("rights_basis")
    if not isinstance(rights_basis, str) or not rights_basis.strip():
        raise PretrainedResourceError("resource approval must record its rights basis")
    return payload


def load_resnet18_weights(
    path: str | Path,
    approval_path: str | Path,
) -> Mapping[str, torch.Tensor]:
    """Revalidate approval and exact bytes before exposing source tensors to a model."""

    load_resource_approval(approval_path)
    artifact = Path(path)
    if not artifact.is_file() or artifact.is_symlink():
        raise PretrainedResourceError("pretrained artifact must be a regular file")
    if artifact.stat().st_size != RESNET18_ARTIFACT_BYTES or _sha256(artifact) != RESNET18_SHA256:
        raise PretrainedResourceError("pretrained artifact size or SHA-256 does not match admission")
    try:
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise PretrainedResourceError("pretrained artifact could not be safely decoded") from exc
    if not isinstance(payload, Mapping) or not payload:
        raise PretrainedResourceError("pretrained artifact does not contain a state dictionary")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in payload.items()):
        raise PretrainedResourceError("pretrained artifact state dictionary is malformed")
    return payload


__all__ = [
    "PRETRAINED_APPROVAL_SCHEMA",
    "RESNET18_ARTIFACT_BYTES",
    "RESNET18_ARTIFACT_ID",
    "RESNET18_DOWNLOAD_URL",
    "RESNET18_RESOURCE_SCOPE",
    "RESNET18_APPROVAL_SCHEMA",
    "TICKET32_RESOURCE_SCOPE",
    "TICKET34_RESOURCE_SCOPE",
    "TICKET36_RESOURCE_SCOPE",
    "RESNET18_SHA256",
    "RESNET18_SOURCE_REVISION",
    "TICKET33_RESOURCE_SCOPE",
    "TICKET39_RESOURCE_SCOPE",
    "PretrainedResourceError",
    "load_resnet18_weights",
    "load_resource_approval",
]
