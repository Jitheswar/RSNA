"""Frozen DINOv2 ViT-S/14 features and their fold-specific study head."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .feature_pooling import ANATOMICAL_POOLING_ID, SIMPLE_POOLING_ID, pool_frozen_features
from .preprocessing import SERIES_DESCRIPTOR_COLUMNS


DINOV2_MODEL_ID = "dinov2-vits14-frozen-slice-features-v1"
DINOV2_ENCODER_ID = "meta-facebook-dinov2-small-vit-s14"
DINOV2_SOURCE_REVISION = "facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056"
DINOV2_REPOSITORY = "facebook/dinov2-small"
DINOV2_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
DINOV2_WEIGHTS_FILENAME = "model.safetensors"
DINOV2_WEIGHTS_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
DINOV2_ARTIFACT_BYTES = 88_249_960
DINOV2_RESOURCE_SCOPE = "ticket38-private-research-three-fold-cpu"
DINOV2_APPROVAL_SCHEMA = "rsna-dinov2-resource-approval-v1"
DINOV2_FEATURE_DIM = 384
DINOV2_FEATURE_BATCH_SIZE = 8
DINOV2_IMAGE_SIZE = (224, 224)
DINOV2_POOLING_ID = ANATOMICAL_POOLING_ID
DINOV2_PREPROCESSING_ID = "ticket38-dinov2-224-whole-knee-v1"
DINOV2_FEATURE_REPRESENTATION = "dinov2-vits14-class-token-384-float32-v1"


class Dinov2ResourceError(ValueError):
    """Raised when DINOv2 weights are missing, unapproved, or changed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dinov2_resource_approval(path: str | Path) -> Mapping[str, object]:
    """Read the one approval record admitted for the frozen DINOv2 branch."""

    approval_path = Path(path)
    if not approval_path.is_file() or approval_path.is_symlink():
        raise Dinov2ResourceError("DINOv2 approval must be a regular file")
    try:
        payload = json.loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Dinov2ResourceError("DINOv2 approval record is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise Dinov2ResourceError("DINOv2 approval record must contain an object")
    required = {
        "schema_version": DINOV2_APPROVAL_SCHEMA,
        "scope": DINOV2_RESOURCE_SCOPE,
        "repository": DINOV2_REPOSITORY,
        "revision": DINOV2_REVISION,
        "weights_filename": DINOV2_WEIGHTS_FILENAME,
        "weights_sha256": DINOV2_WEIGHTS_SHA256,
        "expected_bytes": DINOV2_ARTIFACT_BYTES,
        "external_spend": 0,
        "decision": "approved_for_private_research",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise Dinov2ResourceError(f"DINOv2 approval does not establish {key}={expected!r}")
    for key in ("approved_by", "approved_at", "rights_basis", "pretraining_rights_status"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise Dinov2ResourceError(f"DINOv2 approval must record {key}")
    try:
        datetime.fromisoformat(str(payload["approved_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise Dinov2ResourceError("DINOv2 approval timestamp is invalid") from exc
    if re.fullmatch(r"[0-9a-f]{64}", str(payload.get("snapshot_manifest_sha256", ""))) is None:
        raise Dinov2ResourceError("DINOv2 approval requires its exact snapshot manifest digest")
    snapshot_path_value = payload.get("snapshot_manifest_path")
    if not isinstance(snapshot_path_value, str) or not snapshot_path_value.strip():
        raise Dinov2ResourceError("DINOv2 approval must identify its verified snapshot manifest")
    snapshot_path = Path(snapshot_path_value)
    if not snapshot_path.is_file() or snapshot_path.is_symlink():
        raise Dinov2ResourceError("DINOv2 snapshot manifest is missing or not a regular file")
    if _sha256_file(snapshot_path) != payload["snapshot_manifest_sha256"]:
        raise Dinov2ResourceError("DINOv2 snapshot manifest digest does not match its approval")
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Dinov2ResourceError("DINOv2 snapshot manifest is unreadable") from exc
    if (
        not isinstance(snapshot, Mapping)
        or snapshot.get("repository") != DINOV2_REPOSITORY
        or snapshot.get("revision") != DINOV2_REVISION
        or not isinstance(snapshot.get("files"), list)
    ):
        raise Dinov2ResourceError("DINOv2 snapshot manifest identity is invalid")
    snapshot_root = snapshot_path.parent
    seen_names: set[str] = set()
    for item in snapshot["files"]:
        if not isinstance(item, Mapping):
            raise Dinov2ResourceError("DINOv2 snapshot manifest entry is invalid")
        name = item.get("name")
        digest = item.get("sha256")
        size = item.get("size_bytes")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in seen_names
            or re.fullmatch(r"[0-9a-f]{64}", str(digest or "")) is None
            or not isinstance(size, int)
            or isinstance(size, bool)
        ):
            raise Dinov2ResourceError("DINOv2 snapshot manifest entry fields are invalid")
        seen_names.add(name)
        file_path = snapshot_root / name
        if (
            not file_path.is_file()
            or file_path.is_symlink()
            or file_path.stat().st_size != size
            or _sha256_file(file_path) != digest
        ):
            raise Dinov2ResourceError(f"DINOv2 snapshot file {name!r} failed verification")
    if DINOV2_WEIGHTS_FILENAME not in seen_names:
        raise Dinov2ResourceError("DINOv2 snapshot manifest omits the selected weight file")
    return payload


def load_dinov2_weights(
    path: str | Path,
    approval_path: str | Path,
) -> Mapping[str, torch.Tensor]:
    """Load only the approved safe-tensor artifact after verifying exact bytes."""

    approval = load_dinov2_resource_approval(approval_path)
    artifact = Path(path)
    if not artifact.is_file() or artifact.is_symlink():
        raise Dinov2ResourceError("DINOv2 weights must be a regular file")
    if artifact.stat().st_size != DINOV2_ARTIFACT_BYTES:
        raise Dinov2ResourceError("DINOv2 weight size does not match its approval")
    snapshot_weights = Path(str(approval["snapshot_manifest_path"])).parent / DINOV2_WEIGHTS_FILENAME
    if artifact.resolve() != snapshot_weights.resolve():
        raise Dinov2ResourceError("DINOv2 weight path does not match the admitted model snapshot")
    if _sha256_file(artifact) != DINOV2_WEIGHTS_SHA256:
        raise Dinov2ResourceError("DINOv2 weight SHA-256 does not match its approval")
    try:
        from safetensors.torch import load_file

        state = load_file(str(artifact), device="cpu")
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        raise Dinov2ResourceError("DINOv2 safetensors could not be safely decoded") from exc
    if not state or any(not isinstance(name, str) or not isinstance(value, torch.Tensor) for name, value in state.items()):
        raise Dinov2ResourceError("DINOv2 artifact does not contain a tensor state dictionary")
    return state


def _build_encoder() -> nn.Module:
    try:
        from transformers import Dinov2Config, Dinov2Model
    except ImportError as exc:
        raise RuntimeError(
            "DINOv2 inference requires the pinned transformers runtime dependencies"
        ) from exc
    config = Dinov2Config(
        hidden_size=DINOV2_FEATURE_DIM,
        num_hidden_layers=12,
        num_attention_heads=6,
        mlp_ratio=4,
        image_size=518,
        patch_size=14,
        num_channels=3,
        qkv_bias=True,
        layerscale_value=1.0,
        layer_norm_eps=1e-6,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        drop_path_rate=0.0,
        use_swiglu_ffn=False,
    )
    return Dinov2Model(config)


class Dinov2StudyModel(nn.Module):
    """Frozen DINOv2 class-token features with a trainable study prediction head."""

    imagenet_mean: torch.Tensor
    imagenet_std: torch.Tensor

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        if tuple(config.image_size) != DINOV2_IMAGE_SIZE or config.input_channels != 1:
            raise ValueError("DINOv2 requires one-channel 224 by 224 full-field inputs")
        if config.encoder_id != DINOV2_ENCODER_ID:
            raise ValueError("DINOv2 configuration must identify the admitted ViT-S/14 encoder")
        if config.encoder_weights_sha256 != DINOV2_WEIGHTS_SHA256:
            raise ValueError("DINOv2 configuration must name the admitted checkpoint digest")
        if config.encoder_source_revision != DINOV2_SOURCE_REVISION:
            raise ValueError("DINOv2 configuration must name the exact model revision")
        if config.pooling_id not in {ANATOMICAL_POOLING_ID, SIMPLE_POOLING_ID}:
            raise ValueError("DINOv2 pooling identity is unsupported")
        self.encoder = _build_encoder()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        if config.pooling_id == ANATOMICAL_POOLING_ID:
            pooled_width = 4 * (DINOV2_FEATURE_DIM + 6 + 1)
        else:
            pooled_width = DINOV2_FEATURE_DIM + len(SERIES_DESCRIPTOR_COLUMNS) + 1
        self.head = nn.Sequential(
            nn.Linear(pooled_width, config.head_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.head_hidden_size, config.num_targets),
        )
        self.register_buffer(
            "imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )

    def train(self, mode: bool = True) -> Dinov2StudyModel:
        super().train(mode)
        self.encoder.eval()
        return self

    def load_pretrained_state(self, state: Mapping[str, torch.Tensor]) -> None:
        """Load the verified Hugging Face DINOv2 state into the frozen encoder."""

        try:
            self.encoder.load_state_dict(state, strict=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("checkpoint is incompatible with the admitted DINOv2 ViT-S/14") from exc

    def encode_packed(
        self,
        images: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 6:
            raise ValueError("images must have shape (batch, series, slices, channels, height, width)")
        batch, series, slices, channels, height, width = images.shape
        if (height, width) != DINOV2_IMAGE_SIZE or channels != 1:
            raise ValueError("image shape does not match the DINOv2 configuration")
        if slice_mask.shape != (batch, series, slices) or series_mask.shape != (batch, series):
            raise ValueError("slice and series masks do not match the packed images")
        valid = slice_mask.to(dtype=torch.bool) & series_mask.to(dtype=torch.bool).unsqueeze(-1)
        flat = images.to(dtype=torch.float32).reshape(-1, 1, height, width)
        valid_indices = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
        features = flat.new_zeros((flat.shape[0], DINOV2_FEATURE_DIM))
        if valid_indices.numel():
            with torch.inference_mode():
                for start in range(0, valid_indices.numel(), DINOV2_FEATURE_BATCH_SIZE):
                    indices = valid_indices[start : start + DINOV2_FEATURE_BATCH_SIZE]
                    grayscale = flat.index_select(0, indices)
                    rgb = grayscale.expand(-1, 3, -1, -1)
                    normalized = (rgb - self.imagenet_mean) / self.imagenet_std
                    output = self.encoder(
                        pixel_values=normalized,
                        return_dict=True,
                    )
                    class_features = output.last_hidden_state[:, 0]
                    features.index_copy_(0, indices, class_features)
        return features.reshape(batch, series, slices, DINOV2_FEATURE_DIM)

    def forward_from_features(
        self,
        features: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
        series_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        pooled = pool_frozen_features(
            features,
            slice_mask,
            series_mask,
            series_descriptors,
            feature_width=DINOV2_FEATURE_DIM,
            pooling_id=self.config.pooling_id,
        )
        return self.head(pooled)

    def forward(
        self,
        images: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
        series_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        features = self.encode_packed(images, slice_mask, series_mask)
        return self.forward_from_features(features, slice_mask, series_mask, series_descriptors)

    def predict_proba(
        self,
        images: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
        series_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self(images, slice_mask, series_mask, series_descriptors)).clamp(0.0, 1.0)

    def predict_proba_from_features(
        self,
        features: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
        series_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.forward_from_features(features, slice_mask, series_mask, series_descriptors)
        ).clamp(0.0, 1.0)

    def encode_one_study(
        self,
        images: torch.Tensor,
        slice_mask: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 5 or slice_mask.ndim != 2 or series_mask.ndim != 1:
            raise ValueError("one-study feature extraction requires unbatched packed inputs")
        return self.encode_packed(
            images.unsqueeze(0), slice_mask.unsqueeze(0), series_mask.unsqueeze(0)
        ).squeeze(0)


__all__ = [
    "DINOV2_APPROVAL_SCHEMA",
    "DINOV2_ARTIFACT_BYTES",
    "DINOV2_ENCODER_ID",
    "DINOV2_FEATURE_DIM",
    "DINOV2_FEATURE_REPRESENTATION",
    "DINOV2_IMAGE_SIZE",
    "DINOV2_MODEL_ID",
    "DINOV2_PREPROCESSING_ID",
    "DINOV2_POOLING_ID",
    "DINOV2_REPOSITORY",
    "DINOV2_RESOURCE_SCOPE",
    "DINOV2_REVISION",
    "DINOV2_SOURCE_REVISION",
    "DINOV2_WEIGHTS_FILENAME",
    "DINOV2_WEIGHTS_SHA256",
    "Dinov2ResourceError",
    "Dinov2StudyModel",
    "load_dinov2_resource_approval",
    "load_dinov2_weights",
]
