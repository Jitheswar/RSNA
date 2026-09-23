from __future__ import annotations

import pytest

from rsna_knee.folds import freeze_folds
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.pseudo_label_comparison import (
    PseudoLabelComparisonError,
    compare_fold_pseudo_label_predictions,
)
from rsna_knee.targets import NUM_TARGETS


def _manifest():
    studies = tuple(
        SourceStudy(
            f"gold-{index}",
            True,
            tuple((index + target) % 2 for target in range(NUM_TARGETS)),
        )
        for index in range(12)
    )
    return freeze_folds(
        SourceManifest("train", studies, ()),
        source_version="synthetic-ticket21-metric-v1",
        seed=213,
    )


def test_fold_oof_comparison_reports_paired_uncertainty_without_imputing_auc() -> None:
    manifest = _manifest()
    held_out = {
        item.study_instance_uid: item.reference_labels
        for item in manifest.gold_memberships
        if item.fold == 0
    }
    teacher = {
        uid: tuple(float(1 - label) for label in labels)
        for uid, labels in held_out.items()
    }
    student = {
        uid: tuple(float(label) for label in labels)
        for uid, labels in held_out.items()
    }

    result = compare_fold_pseudo_label_predictions(
        teacher,
        student,
        manifest,
        0,
        bootstrap_resamples=40,
        bootstrap_seed=29,
    )

    assert result["oof"]["prediction_count"] == 4
    assert result["teacher"]["macro_auc"] == 0.0
    assert result["student"]["macro_auc"] == 1.0
    assert result["delta"]["macro_auc"] == 1.0
    interval = result["paired_study_bootstrap"]["macro_auc"]
    assert interval["defined_resamples"] + interval["undefined_resamples"] == 40
    assert interval["lower"] == interval["median"] == interval["upper"] == 1.0
    assert result["fold_variation"]["estimable"] is False


def test_fold_oof_comparison_rejects_predictions_outside_held_out_gold() -> None:
    manifest = _manifest()
    expected = {
        item.study_instance_uid: (0.5,) * NUM_TARGETS
        for item in manifest.gold_memberships
        if item.fold == 0
    }
    missing = dict(expected)
    missing.pop(next(iter(missing)))

    with pytest.raises(PseudoLabelComparisonError, match="held-out gold identities"):
        compare_fold_pseudo_label_predictions(expected, missing, manifest, 0)
