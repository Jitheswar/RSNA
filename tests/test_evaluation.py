from __future__ import annotations

import pytest

from rsna_knee.folds import freeze_folds
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.evaluation import roc_auc_score
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
