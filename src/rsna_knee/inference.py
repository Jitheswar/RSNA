"""Training-independent study inference and submission orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .domain import StudyRecord
from .model import HierarchicalStudyModel
from .preprocessing import (
    PackedStudy,
    PreprocessConfig,
    SeriesFailure,
    pack_study,
    prepare_study,
)
from .submission import write_submission
from .targets import NUM_TARGETS


class InferenceError(RuntimeError):
    """Raised for a global inference failure that cannot be a study fallback."""


@dataclass(frozen=True)
class StudyPrediction:
    """Scores and diagnostics for one study-level prediction."""

    study_instance_uid: str
    scores: np.ndarray
    used_fallback: bool
    valid_series_count: int
    rejected_series: tuple[SeriesFailure, ...]
    sampled_source_indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class InferenceBatch:
    """Results for an ordered set of studies and its optional submission."""

    results: tuple[StudyPrediction, ...]
    output_path: Path | None = None


class InferenceEngine:
    """Run the shared deterministic preprocessing/model path on CPU or device."""

    def __init__(
        self,
        model: HierarchicalStudyModel,
        preprocess_config: PreprocessConfig = PreprocessConfig(),
        *,
        device: str | torch.device = "cpu",
        fallback_scores: Sequence[float] | None = None,
    ) -> None:
        if tuple(model.config.image_size) != tuple(preprocess_config.image_size):
            raise ValueError("model and preprocessing image sizes must agree")
        self.model = model.to(device)
        self.model.eval()
        self.preprocess_config = preprocess_config
        self.device = torch.device(device)
        if fallback_scores is None:
            fallback: np.ndarray = np.full(NUM_TARGETS, 0.5, dtype=np.float64)
        else:
            fallback = np.asarray(fallback_scores, dtype=np.float64)
            if fallback.shape != (NUM_TARGETS,) or not np.isfinite(fallback).all():
                raise ValueError(f"fallback_scores must contain {NUM_TARGETS} finite values")
            if (fallback < 0.0).any() or (fallback > 1.0).any():
                raise ValueError("fallback_scores must be within [0, 1]")
        self.fallback_scores = fallback

    def _predict_packed(self, packed: PackedStudy) -> np.ndarray:
        images = torch.from_numpy(packed.images).unsqueeze(0).to(self.device)
        slice_mask = torch.from_numpy(packed.slice_mask).unsqueeze(0).to(self.device)
        series_mask = torch.from_numpy(packed.series_mask).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            scores = self.model.predict_proba(images, slice_mask, series_mask).squeeze(0).cpu().numpy()
        values = np.asarray(scores, dtype=np.float64)
        if values.shape != (NUM_TARGETS,) or not np.isfinite(values).all():
            raise InferenceError(f"model returned an invalid score vector for {packed.study_instance_uid}")
        if (values < 0.0).any() or (values > 1.0).any():
            raise InferenceError(f"model returned an out-of-range score vector for {packed.study_instance_uid}")
        return values

    def predict_study(self, study: StudyRecord) -> StudyPrediction:
        """Predict one study, retaining partial-input diagnostics and rows."""

        prepared = prepare_study(study, self.preprocess_config)
        packed = pack_study(study, prepared.series, self.preprocess_config)
        valid_series_count = int(packed.series_mask.sum())
        if valid_series_count == 0:
            return StudyPrediction(
                study_instance_uid=study.study_instance_uid,
                scores=self.fallback_scores.copy(),
                used_fallback=True,
                valid_series_count=0,
                rejected_series=prepared.failures,
                sampled_source_indices=packed.sampled_source_indices,
            )
        return StudyPrediction(
            study_instance_uid=study.study_instance_uid,
            scores=self._predict_packed(packed),
            used_fallback=False,
            valid_series_count=valid_series_count,
            rejected_series=prepared.failures,
            sampled_source_indices=packed.sampled_source_indices,
        )


def infer_studies(
    engine: InferenceEngine,
    studies: Sequence[StudyRecord],
    output_path: str | Path | None = None,
) -> InferenceBatch:
    """Run ordered study inference and optionally write a strict submission."""

    if not studies:
        raise InferenceError("at least one study is required")
    study_uids = tuple(study.study_instance_uid for study in studies)
    if len(set(study_uids)) != len(study_uids):
        raise InferenceError("study identifiers must be unique")
    results = tuple(engine.predict_study(study) for study in studies)
    destination = None
    if output_path is not None:
        destination = write_submission(
            output_path,
            expected_study_uids=study_uids,
            predictions={result.study_instance_uid: result.scores for result in results},
        )
    return InferenceBatch(results=results, output_path=destination)
