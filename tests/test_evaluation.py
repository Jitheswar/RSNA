from __future__ import annotations

import pytest

from rsna_knee.folds import GroupingAssessment, freeze_folds
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.evaluation import evaluate_oof_predictions, roc_auc_score
from rsna_knee.targets import TARGET_COLUMNS


@pytest.mark.parametrize(
    ("labels", "scores", "expected"),
    [
        ((0, 1), (0.1, 0.9), 1.0),
        ((0, 1), (0.9, 0.1), 0.0),
        ((0, 1), (0.5, 0.5), 0.5),
    ],
)
def test_roc_auc_uses_rankings_and_average_ties(
    labels: tuple[int, ...], scores: tuple[float, ...], expected: float
) -> None:
    assert roc_auc_score(labels, scores) == expected


def test_roc_auc_is_undefined_when_one_class_is_missing() -> None:
    assert roc_auc_score((1, 1), (0.2, 0.8)) is None


def test_fold_evaluation_joins_only_the_declared_held_out_gold_rows() -> None:
    from rsna_knee.evaluation import EvaluationError, evaluate_fold_predictions

    source = SourceManifest(
        "train",
        tuple(
            SourceStudy(
                f"gold-{index}",
                True,
                tuple((index + target_index) % 2 for target_index in range(len(TARGET_COLUMNS))),
            )
            for index in range(9)
        ),
        (),
    )
    manifest = freeze_folds(source, source_version="evaluation-fixture-v1", seed=17)
    held_out = tuple(
        item.study_instance_uid for item in manifest.gold_memberships if item.fold == 0
    )
    predictions = {uid: [0.5] * len(TARGET_COLUMNS) for uid in held_out}

    result = evaluate_fold_predictions(predictions, manifest, 0)

    assert result.study_instance_uids == tuple(sorted(held_out))
    assert len(result.targets) == len(TARGET_COLUMNS)
    assert "study_instance_uids" not in result.to_public_dict()
    with pytest.raises(EvaluationError, match="identity mismatch"):
        evaluate_fold_predictions({**predictions, "not-held-out": [0.5] * 12}, manifest, 0)


def test_oof_evaluation_reports_pooled_and_fold_metrics_in_one_reference_join() -> None:
    source = SourceManifest(
        "train",
        tuple(
            SourceStudy(
                f"gold-{index}",
                True,
                tuple((index + target_index) % 2 for target_index in range(len(TARGET_COLUMNS))),
            )
            for index in range(12)
        ),
        (),
    )
    manifest = freeze_folds(source, source_version="oof-fixture-v1", seed=17)
    predictions = {
        item.study_instance_uid: [
            float(label) for label in item.reference_labels
        ]
        for item in manifest.gold_memberships
    }

    result = evaluate_oof_predictions(predictions, manifest)

    assert result.prediction_count == 12
    assert result.pooled_macro_auc == 1.0
    assert result.mean_fold_macro_auc == 1.0
    assert result.fold_macro_auc_std == 0.0
    assert tuple(item.target for item in result.pooled_targets) == TARGET_COLUMNS
    assert tuple(item.fold for item in result.folds) == (0, 1, 2)
    assert all(item.positive_count + item.negative_count == 12 for item in result.pooled_targets)
    assert "study_instance_uids" not in result.to_public_dict()


def test_oof_evaluation_rejects_missing_or_extra_gold_predictions() -> None:
    from rsna_knee.evaluation import EvaluationError

    source = SourceManifest(
        "train",
        tuple(
            SourceStudy(
                f"gold-{index}",
                True,
                tuple((index + target_index) % 2 for target_index in range(len(TARGET_COLUMNS))),
            )
            for index in range(9)
        ),
        (),
    )
    manifest = freeze_folds(source, source_version="oof-identity-v1", seed=19)
    predictions = {
        item.study_instance_uid: [0.5] * len(TARGET_COLUMNS)
        for item in manifest.gold_memberships
    }
    missing_uid = next(iter(predictions))
    predictions.pop(missing_uid)

    with pytest.raises(EvaluationError, match="OOF prediction identity mismatch"):
        evaluate_oof_predictions({**predictions, "extra": [0.5] * 12}, manifest)


def test_oof_bootstrap_is_reproducible_and_reports_undefined_resamples() -> None:
    source = SourceManifest(
        "train",
        tuple(
            SourceStudy(
                f"gold-{index}",
                True,
                tuple((index + target_index) % 2 for target_index in range(len(TARGET_COLUMNS))),
            )
            for index in range(12)
        ),
        (),
    )
    manifest = freeze_folds(source, source_version="oof-bootstrap-v1", seed=23)
    predictions = {
        item.study_instance_uid: [float(label) for label in item.reference_labels]
        for item in manifest.gold_memberships
    }

    first = evaluate_oof_predictions(
        predictions,
        manifest,
        bootstrap_resamples=2000,
        bootstrap_seed=41,
    )
    second = evaluate_oof_predictions(
        predictions,
        manifest,
        bootstrap_resamples=2000,
        bootstrap_seed=41,
    )

    assert first.bootstrap is not None
    assert first.bootstrap.to_dict() == second.bootstrap.to_dict()
    assert first.bootstrap.unit == "study"
    assert first.bootstrap.resamples == 2000
    assert first.bootstrap.macro_auc.undefined_resamples > 0
    assert first.bootstrap.macro_auc.defined_resamples < 2000
    assert all(item.undefined_resamples > 0 for item in first.bootstrap.targets)


def test_oof_bootstrap_switches_to_reliable_groups_without_splitting_them() -> None:
    source = SourceManifest(
        "train",
        tuple(
            SourceStudy(
                f"gold-{index}",
                True,
                tuple(
                    ((index // 2) + target_index) % 2
                    for target_index in range(len(TARGET_COLUMNS))
                ),
            )
            for index in range(12)
        ),
        (),
    )
    grouping = GroupingAssessment.reliable_key(
        {f"gold-{index}": f"group-{index // 2}" for index in range(12)},
        field_name="PatientID",
        scope="synthetic fixture",
        provenance="independent fixture audit",
        examined_fields=("PatientID",),
    )
    manifest = freeze_folds(
        source,
        source_version="oof-group-bootstrap-v1",
        seed=29,
        grouping=grouping,
    )
    predictions = {
        item.study_instance_uid: [float(label) for label in item.reference_labels]
        for item in manifest.gold_memberships
    }

    result = evaluate_oof_predictions(
        predictions,
        manifest,
        bootstrap_resamples=32,
        bootstrap_seed=5,
    )

    assert result.bootstrap is not None
    assert result.bootstrap.unit == "group"
    assert result.bootstrap.unit_count == 6
