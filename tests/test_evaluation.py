from __future__ import annotations

import numpy as np
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


def _diagnostic_manifest(seed: int = 7):
    rng = np.random.default_rng(seed)
    labels = (rng.random((30, len(TARGET_COLUMNS))) < 0.4).astype(int)
    labels[:3] = 1
    labels[3:6] = 0
    source = SourceManifest(
        "train",
        tuple(
            SourceStudy(f"gold-{index:02d}", True, tuple(int(value) for value in labels[index]))
            for index in range(30)
        ),
        (),
    )
    return freeze_folds(source, source_version="diagnostics-fixture-v1", seed=seed)


def _brute_force_auc(positive: list[float], negative: list[float]) -> tuple[float, int]:
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins, len(positive) * len(negative)


def test_pair_decomposition_partitions_pooled_pairs_exactly() -> None:
    manifest = _diagnostic_manifest()
    rng = np.random.default_rng(11)
    predictions = {
        item.study_instance_uid: [round(float(value), 2) for value in rng.random(len(TARGET_COLUMNS))]
        for item in manifest.gold_memberships
    }

    result = evaluate_oof_predictions(predictions, manifest)
    diagnostics = result.diagnostics

    assert diagnostics is not None
    within_terms = []
    for index, (pooled, pairs) in enumerate(zip(result.pooled_targets, diagnostics.pair_decomposition)):
        assert pairs.pooled_pairs == pooled.positive_count * pooled.negative_count
        assert pairs.pooled_auc == pytest.approx(pooled.auc, abs=1e-12)
        within_wins = within_pairs = 0.0
        for fold in (0, 1, 2):
            rows = [item for item in manifest.gold_memberships if item.fold == fold]
            wins, count = _brute_force_auc(
                [predictions[item.study_instance_uid][index] for item in rows if item.reference_labels[index] == 1],
                [predictions[item.study_instance_uid][index] for item in rows if item.reference_labels[index] == 0],
            )
            within_wins += wins
            within_pairs += count
        assert pairs.within_fold_auc == pytest.approx(within_wins / within_pairs, abs=1e-12)
        within_terms.append(within_wins / within_pairs)
    assert diagnostics.within_fold_macro_auc == pytest.approx(float(np.mean(within_terms)), abs=1e-12)


def test_training_prevalence_scores_show_a_pooled_artifact_but_no_within_fold_signal() -> None:
    manifest = _diagnostic_manifest()
    prevalence = {
        fold: [
            float(np.mean([item.reference_labels[index] for item in manifest.gold_memberships if item.fold != fold]))
            for index in range(len(TARGET_COLUMNS))
        ]
        for fold in (0, 1, 2)
    }
    predictions = {
        item.study_instance_uid: prevalence[item.fold] for item in manifest.gold_memberships
    }

    result = evaluate_oof_predictions(predictions, manifest, permutation_resamples=400, permutation_seed=3)
    diagnostics = result.diagnostics

    assert diagnostics is not None
    assert diagnostics.within_fold_macro_auc == 0.5
    assert diagnostics.prevalence_baseline.macro_auc == pytest.approx(result.pooled_macro_auc)
    assert diagnostics.prevalence_baseline.within_fold_macro_auc == 0.5
    assert all(value == 0.5 for value in diagnostics.prevalence_baseline.fold_macro_aucs)
    assert diagnostics.mean_score_vs_training_prevalence_pearson == pytest.approx(1.0)
    assert diagnostics.permutation_test is not None
    assert diagnostics.permutation_test.observed == 0.5
    assert diagnostics.permutation_test.rejects_no_signal is False


def test_within_fold_permutation_test_detects_real_ranking_signal_reproducibly() -> None:
    manifest = _diagnostic_manifest()
    predictions = {
        item.study_instance_uid: [0.2 + 0.6 * label for label in item.reference_labels]
        for item in manifest.gold_memberships
    }

    first = evaluate_oof_predictions(predictions, manifest, permutation_resamples=500, permutation_seed=9)
    second = evaluate_oof_predictions(predictions, manifest, permutation_resamples=500, permutation_seed=9)

    assert first.diagnostics is not None and first.diagnostics.permutation_test is not None
    test = first.diagnostics.permutation_test
    assert test.observed == first.diagnostics.within_fold_macro_auc == 1.0
    assert test.p_upper == pytest.approx(1 / 501)
    assert test.rejects_no_signal is True
    assert test.to_dict() == second.diagnostics.permutation_test.to_dict()
    assert test.to_dict()["measurable_image_signal"] is True


def test_bootstrap_reports_the_within_fold_macro_interval() -> None:
    manifest = _diagnostic_manifest()
    predictions = {
        item.study_instance_uid: [float(label) for label in item.reference_labels]
        for item in manifest.gold_memberships
    }

    result = evaluate_oof_predictions(predictions, manifest, bootstrap_resamples=64, bootstrap_seed=5)

    assert result.bootstrap is not None
    within = result.bootstrap.within_fold_macro_auc
    assert within is not None
    assert within.resamples == 64
    assert within.defined_resamples + within.undefined_resamples == 64
    assert within.median == 1.0
