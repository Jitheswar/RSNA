from __future__ import annotations

import pytest

from rsna_knee.folds import FoldManifest, freeze_folds
from rsna_knee.image_model_comparison import (
    ImageModelComparisonError,
    compare_image_model_predictions,
)
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.targets import TARGET_COLUMNS


def _manifest() -> FoldManifest:
    source = SourceManifest(
        "train",
        tuple(
            SourceStudy(
                f"gold-{index}",
                True,
                tuple(index % 2 for _ in TARGET_COLUMNS),
            )
            for index in range(12)
        ),
        (),
    )
    return freeze_folds(source, source_version="ticket19-metric-fixture-v1", seed=31)


def test_paired_oof_comparison_reports_expected_delta_and_uncertainty() -> None:
    manifest = _manifest()
    labels = {
        item.study_instance_uid: item.reference_labels
        for item in manifest.gold_memberships
    }
    baseline = {
        uid: tuple(float(1 - label) for label in row)
        for uid, row in labels.items()
    }
    candidate = {
        uid: tuple(float(label) for label in row)
        for uid, row in labels.items()
    }

    comparison = compare_image_model_predictions(
        baseline,
        candidate,
        manifest,
        bootstrap_resamples=40,
        bootstrap_seed=19,
    )

    assert comparison["prediction_count"] == 12
    assert comparison["baseline"]["pooled_macro_auc"] == 0.0
    assert comparison["candidate"]["pooled_macro_auc"] == 1.0
    assert comparison["delta"]["pooled_macro_auc"] == 1.0
    assert all(item["delta_auc"] == 1.0 for item in comparison["per_target"])
    interval = comparison["paired_study_bootstrap"]["macro_auc"]
    assert interval["defined_resamples"] + interval["undefined_resamples"] == 40
    assert interval["lower"] == interval["upper"] == 1.0


def test_paired_oof_comparison_rejects_different_prediction_cohorts() -> None:
    manifest = _manifest()
    expected = {item.study_instance_uid: [0.5] * len(TARGET_COLUMNS) for item in manifest.gold_memberships}
    candidate = dict(expected)
    candidate.pop(next(iter(candidate)))

    with pytest.raises(ImageModelComparisonError, match="same complete gold cohort"):
        compare_image_model_predictions(expected, candidate, manifest, bootstrap_resamples=20)
