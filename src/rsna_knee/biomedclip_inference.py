"""CPU inference for the pinned, independently admitted medical representation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .feature_pooling import ANATOMICAL_POOLING_ID, pool_frozen_features


CHECKPOINT_SHA256 = "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be"
CONFIG_SHA256 = "9a41f334a8c444678772c0ebb9ab854c97ab350bced3a17b803e258d39c23dc0"
FEATURE_IDENTITY_SHA256 = "7ba6b06160d908c6e03c2edb38bae73b0d90e50ab27c0eef2bbd1ce6cf3ac230"
HEAD_ID = "biomedclip-vitb16-projected-l2-plane-pooled-mlp32-v1"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
RUNTIME_VERSIONS = {
    "torch": "2.7.1+cpu", "torchvision": "0.22.1+cpu", "timm": "1.0.22",
    "open_clip_torch": "2.23.0", "transformers": "4.35.2",
    "numpy": "2.2.6", "pillow": "11.2.1",
}


class MedicalInferenceError(ValueError):
    """An input, weight or runtime differs from the fixed medical recipe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MedicalInferenceError(message)


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _float_cpu(value: torch.Tensor) -> bool:
    return value.device.type == "cpu" and value.dtype == torch.float32 and bool(torch.isfinite(value).all())


def _valid_mask(slice_mask: torch.Tensor, series_mask: torch.Tensor, batch: int) -> torch.Tensor:
    _require(slice_mask.shape == (batch, 16, 8) and series_mask.shape == (batch, 16)
             and slice_mask.dtype == series_mask.dtype == torch.bool
             and slice_mask.device.type == series_mask.device.type == "cpu", "Exact CPU boolean packing masks required")
    return slice_mask & series_mask.unsqueeze(-1)


class MedicalFoldHead(nn.Sequential):
    """Keep fold and supervision identity attached to the loaded inference head."""

    def __init__(self, fold: int, arm: str) -> None:
        _require(type(fold) is int and fold in (0, 1, 2)
                 and arm in ("reference-only", "report-supervision"), "Explicit medical head fold and arm required")
        super().__init__(nn.Linear(2076, 32, dtype=torch.float32, device="cpu"), nn.ReLU(inplace=True),
                         nn.Linear(32, 12, dtype=torch.float32, device="cpu"))
        self.fold = fold
        self.arm = arm


def reference_fallback(labels: Sequence[Sequence[int | None]]) -> torch.Tensor:
    """Laplace priors from explicitly supplied eligible trusted reference rows."""
    _require(bool(labels) and all(len(row) == 12 for row in labels), "Eligible twelve-target reference rows required")
    _require(all(value is None or type(value) is int and value in (0, 1) for row in labels for value in row),
             "Reference fallback requires binary or missing trusted labels")
    probabilities = []
    for column in range(12):
        values = [row[column] for row in labels if row[column] is not None]
        positive = sum(value == 1 for value in values)
        probabilities.append((positive + 1) / (len(values) + 2))
    return torch.tensor(probabilities, dtype=torch.float32)


def load_medical_head(path: Path, *, expected_sha256: str, expected_identity: dict) -> MedicalFoldHead:
    """Load only an exact audited final head; preserve constructor RNG state."""
    _require(_digest(path) == expected_sha256, "Final head checkpoint bytes differ")
    value = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(value, dict)
             and value.get("schema_version") == "rsna-frozen-head-checkpoint-v1"
             and value.get("head_id") == HEAD_ID and value.get("identity") == expected_identity
             and expected_identity.get("head_id") == HEAD_ID
             and expected_identity.get("feature_identity_sha256") == FEATURE_IDENTITY_SHA256
             and type(expected_identity.get("fold")) is int and expected_identity.get("fold") in (0, 1, 2)
             and expected_identity.get("arm") in ("reference-only", "report-supervision")
             and value.get("checkpoint_kind") == "final_selected"
             and isinstance(value.get("counts"), dict)
             and value["counts"].get("successful_optimizer_steps") == 7500,
             "Audited final medical head identity and budget required")
    state = value.get("head_state_dict")
    _require(isinstance(state, dict) and all(isinstance(v, torch.Tensor) and _float_cpu(v) for v in state.values()),
             "Finite FP32 head tensors required")
    with torch.random.fork_rng(devices=[]):
        head = MedicalFoldHead(expected_identity["fold"], expected_identity["arm"])
    try:
        head.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise MedicalInferenceError("Exact 2076-to-32-to-12 head architecture required") from exc
    head.eval().requires_grad_(False)
    return head


class BiomedClipVision(nn.Module):
    """Encode valid whole-field slices with one FP32 forward at a time."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, visual: nn.Module) -> None:
        super().__init__()
        self.visual = visual.eval().requires_grad_(False)
        _require(all(_float_cpu(v) for v in self.visual.state_dict().values()), "Finite CPU FP32 vision state required")
        self.register_buffer("mean", torch.tensor(CLIP_MEAN, dtype=torch.float32).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CLIP_STD, dtype=torch.float32).reshape(1, 3, 1, 1))

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, config_path: Path) -> BiomedClipVision:
        """Strict official vision factory; no text tower, downloads or alternate runtime."""
        _require(_digest(checkpoint_path) == CHECKPOINT_SHA256 and _digest(config_path) == CONFIG_SHA256,
                 "Pinned medical checkpoint and configuration bytes required")
        _require(torch.version.cuda is None and not torch.cuda.is_available(), "Admitted CPU-only runtime required")
        for name, expected in RUNTIME_VERSIONS.items():
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as exc:
                raise MedicalInferenceError(f"Missing admitted runtime package: {name}") from exc
            _require(actual == expected, f"Admitted runtime version differs: {name}")
        from open_clip.model import _build_vision_tower  # type: ignore[import-not-found]
        config = json.loads(config_path.read_text())
        value = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        _require(isinstance(value, dict), "Flat medical state dictionary required")
        visual_state = {key.removeprefix("visual."): tensor for key, tensor in value.items()
                        if isinstance(key, str) and key.startswith("visual.")}
        _require(len(visual_state) == 151 and all(isinstance(v, torch.Tensor) and _float_cpu(v) for v in visual_state.values()),
                 "All 151 finite FP32 vision tensors required")
        with torch.random.fork_rng(devices=[]):
            visual = _build_vision_tower(config["model_cfg"]["embed_dim"], config["model_cfg"]["vision_cfg"])
        try:
            visual.load_state_dict(visual_state, strict=True)
        except RuntimeError as exc:
            raise MedicalInferenceError("Strict official vision coverage required") from exc
        _require(set(visual.state_dict()) == set(visual_state), "Strict official vision coverage required")
        return cls(visual)

    @torch.inference_mode()
    def encode_packed(self, images: torch.Tensor, slice_mask: torch.Tensor,
                      series_mask: torch.Tensor) -> torch.Tensor:
        _require(images.ndim == 6 and images.shape[1:] == (16, 8, 1, 224, 224)
                 and _float_cpu(images), "Finite FP32 CPU whole-field packing required")
        batch = images.shape[0]
        _require(batch > 0, "At least one packed study required")
        valid = _valid_mask(slice_mask, series_mask, batch).reshape(-1)
        _require(all(v.device.type == "cpu" and v.dtype == torch.float32 for v in self.visual.state_dict().values())
                 and _float_cpu(self.mean) and _float_cpu(self.std), "Vision must retain CPU FP32 state")
        self.visual.eval()
        flat = images.reshape(-1, 1, 224, 224)
        features = torch.zeros((len(flat), 512), dtype=torch.float32)
        for index in torch.nonzero(valid, as_tuple=False).flatten().tolist():
            rgb = (flat[index:index + 1].repeat(1, 3, 1, 1) - self.mean) / self.std
            raw = self.visual(rgb)
            _require(isinstance(raw, torch.Tensor) and raw.shape == (1, 512) and _float_cpu(raw),
                     "Finite projected 512-D vision output required")
            vector = F.normalize(raw, dim=-1)
            _require(bool(torch.max(torch.abs(torch.linalg.vector_norm(vector, dim=-1) - 1)) <= 1e-6),
                     "Unit valid-slice medical output required")
            features[index] = vector[0]
        return features.reshape(batch, 16, 8, 512)


@dataclass(frozen=True)
class MedicalPrediction:
    probabilities: torch.Tensor
    fallback_used: torch.Tensor


class MedicalStudyPredictor(nn.Module):
    """Explicit held-out-fold prediction or the deployed three-head probability mean."""

    fallback_probabilities: torch.Tensor

    def __init__(self, vision: BiomedClipVision, heads: Mapping[int, MedicalFoldHead],
                 fallback_probabilities: torch.Tensor) -> None:
        super().__init__()
        _require(set(heads) == {0, 1, 2} and all(type(fold) is int for fold in heads), "Exactly the three audited fold heads required")
        _require(all(isinstance(head, MedicalFoldHead) and head.fold == fold for fold, head in heads.items())
                 and len({head.arm for head in heads.values()}) == 1, "Matching fold identities and a single supervision arm required")
        _require(fallback_probabilities.shape == (3, 12) and _float_cpu(fallback_probabilities)
                 and bool(((fallback_probabilities >= 0) & (fallback_probabilities <= 1)).all()),
                 "Explicit finite eligible-reference fallback priors required")
        for head in heads.values():
            _require(len(head) == 3 and isinstance(head[0], nn.Linear) and isinstance(head[1], nn.ReLU)
                     and isinstance(head[2], nn.Linear) and head[0].in_features == 2076
                     and head[0].out_features == head[2].in_features == 32 and head[2].out_features == 12
                     and all(_float_cpu(v) for v in head.state_dict().values()), "Exact finite FP32 medical head architecture required")
        self.vision = vision
        self.heads = nn.ModuleDict({str(fold): head.eval().requires_grad_(False) for fold, head in heads.items()})
        self.register_buffer("fallback_probabilities", fallback_probabilities.detach().clone())

    @torch.inference_mode()
    def predict_features(self, features: torch.Tensor, slice_mask: torch.Tensor,
                         series_mask: torch.Tensor, descriptors: torch.Tensor, *,
                         fold: int | None = None) -> MedicalPrediction:
        _require(fold is None or type(fold) is int and fold in (0, 1, 2), "Explicit valid held-out fold or ensemble mode required")
        _require(features.ndim == 4 and features.shape[1:] == (16, 8, 512) and _float_cpu(features),
                 "Finite FP32 medical feature bags required")
        batch = features.shape[0]
        _require(batch > 0 and descriptors.shape == (batch, 16, 10) and _float_cpu(descriptors),
                 "Finite original anatomical descriptors required")
        valid = _valid_mask(slice_mask, series_mask, batch)
        _require(not bool(features[~valid].any()), "Invalid feature slots must remain zero")
        if bool(valid.any()):
            _require(bool(torch.max(torch.abs(torch.linalg.vector_norm(features[valid], dim=-1) - 1)) <= 1e-6),
                     "Valid medical vectors must retain explicit unit normalization")
        _require(_float_cpu(self.fallback_probabilities) and all(_float_cpu(v) for h in self.heads.values() for v in h.state_dict().values()),
                 "Head inference must retain CPU FP32 state")
        pooled = pool_frozen_features(features, slice_mask, series_mask, descriptors,
                                      feature_width=512, pooling_id=ANATOMICAL_POOLING_ID)
        folds = (fold,) if fold is not None else (0, 1, 2)
        predictions = []
        for ident in folds:
            head = self.heads[str(ident)]
            # Retain the exact audited row grouping, including the final singleton.
            predictions.append(torch.cat([torch.sigmoid(head(pooled[start:start + 2]))
                                          for start in range(0, batch, 2)]))
        probabilities = torch.stack(predictions).mean(dim=0)
        fallback = ~valid.reshape(batch, -1).any(dim=1)
        priors = self.fallback_probabilities[list(folds)].mean(dim=0)
        probabilities[fallback] = priors
        _require(_float_cpu(probabilities) and bool(((probabilities >= 0) & (probabilities <= 1)).all()),
                 "Finite bounded twelve-target prediction required")
        return MedicalPrediction(probabilities, fallback)

    @torch.inference_mode()
    def predict_packed(self, images: torch.Tensor, slice_mask: torch.Tensor,
                       series_mask: torch.Tensor, descriptors: torch.Tensor, *,
                       fold: int | None = None) -> MedicalPrediction:
        features = self.vision.encode_packed(images, slice_mask, series_mask)
        return self.predict_features(features, slice_mask, series_mask, descriptors, fold=fold)


__all__ = ["BiomedClipVision", "MedicalFoldHead", "MedicalStudyPredictor", "MedicalPrediction", "MedicalInferenceError", "load_medical_head", "reference_fallback"]
