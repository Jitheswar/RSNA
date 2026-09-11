from __future__ import annotations

import numpy as np

from rsna_knee.domain import SeriesRecord, StudyRecord
from rsna_knee.inference import InferenceEngine, infer_studies
from rsna_knee.model import HierarchicalStudyModel, ModelConfig
from rsna_knee.preprocessing import PreprocessConfig
from rsna_knee.series_cache import SeriesCache
from rsna_knee.synthetic import make_synthetic_study


def test_synthetic_study_reaches_twelve_bounded_scores() -> None:
    preprocess = PreprocessConfig(image_size=(32, 32), max_slices=8, max_series=4)
    model = HierarchicalStudyModel(ModelConfig(image_size=(32, 32), channels=(2, 4, 8)))
    engine = InferenceEngine(model, preprocess)
    study = make_synthetic_study("synthetic-1", series_count=2, slices_per_series=5)

    result = engine.predict_study(study)

    assert result.study_instance_uid == study.study_instance_uid
    assert result.scores.shape == (12,)
    assert np.isfinite(result.scores).all()
    assert np.all((result.scores >= 0.0) & (result.scores <= 1.0))
    assert result.valid_series_count == 2
    assert result.used_fallback is False


def test_inference_keeps_a_row_for_a_study_with_no_valid_series(tmp_path) -> None:
    preprocess = PreprocessConfig(image_size=(32, 32), max_slices=8, max_series=4)
    model = HierarchicalStudyModel(ModelConfig(image_size=(32, 32), channels=(2, 4, 8)))
    engine = InferenceEngine(model, preprocess)
    invalid = StudyRecord("synthetic-invalid", (SeriesRecord("synthetic-invalid", "empty", ()),))
    valid = make_synthetic_study("synthetic-valid", series_count=1, slices_per_series=3)

    output_path = tmp_path / "submission.csv"
    batch = infer_studies(engine, (invalid, valid), output_path)

    assert [item.study_instance_uid for item in batch.results] == [
        "synthetic-invalid",
        valid.study_instance_uid,
    ]
    assert batch.results[0].used_fallback is True
    assert np.all(batch.results[0].scores == 0.5)
    assert output_path.exists()
    assert output_path.read_text().splitlines()[0].split(",") == [
        "StudyInstanceUID",
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    ]


def test_inference_can_use_the_series_cache_without_changing_predictions(tmp_path) -> None:
    preprocess = PreprocessConfig(image_size=(32, 32), max_slices=8, max_series=4)
    model = HierarchicalStudyModel(ModelConfig(image_size=(32, 32), channels=(2, 4, 8)))
    study = make_synthetic_study("synthetic-cached", series_count=2, slices_per_series=5)
    cache = SeriesCache(tmp_path / "cache")
    engine = InferenceEngine(model, preprocess, series_cache=cache)

    first = engine.predict_study(study)
    second = engine.predict_study(study)

    np.testing.assert_array_equal(first.scores, second.scores)
    assert first.used_fallback is False
    assert second.used_fallback is False
    assert len(cache.list_complete_keys()) == 2
