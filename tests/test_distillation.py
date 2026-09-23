from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.distillation import (
    DistillationExperimentError,
    build_ticket22_fold_plan,
    compare_ticket22_oof_predictions,
    create_fold_soft_target_examples,
)
from rsna_knee.folds import FoldManifest, freeze_folds
from rsna_knee.image_model_comparison import ImageModelComparisonError
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.targets import NUM_TARGETS
from rsna_knee.training import SupervisionProvenance, TrainingExample
from rsna_knee.training import TrainingConfig, load_training_checkpoint, train_fixed_budget


def _manifest() -> FoldManifest:
    gold = tuple(
        SourceStudy(
            f"gold-{index}",
            True,
            tuple((index + target) % 2 for target in range(NUM_TARGETS)),
        )
        for index in range(12)
    )
    report_only = tuple(
        SourceStudy(f"report-{index}", True, (None,) * NUM_TARGETS)
        for index in range(6)
    )
    return freeze_folds(
        SourceManifest("train", gold + report_only, ()),
        source_version="synthetic-ticket22-v1",
        seed=221,
    )


def _candidate(uid: str, fold: int) -> TrainingExample:
    return TrainingExample(
        study_instance_uid=uid,
        images=np.zeros((1, 1, 1, 16, 16), dtype=np.float32),
        slice_mask=np.ones((1, 1), dtype=bool),
        series_mask=np.ones(1, dtype=bool),
        targets=np.zeros(NUM_TARGETS, dtype=np.float32),
        supervision_mask=np.zeros(NUM_TARGETS, dtype=bool),
        provenance=SupervisionProvenance(
            source_kind="reference",
            fold=fold,
            eligible_study_uids=(uid,),
            excluded_study_uids=(),
        ),
    )


def test_fold_plan_excludes_every_held_out_input_and_unavailable_candidate() -> None:
    manifest = _manifest()
    fold = 0
    report_only = manifest.report_only_memberships
    unavailable = next(item.study_instance_uid for item in report_only if item.fold != fold)
    available = tuple(
        item.study_instance_uid for item in manifest.memberships if item.study_instance_uid != unavailable
    )

    plan = build_ticket22_fold_plan(manifest, fold, available)

    assert plan.teacher_training_study_uids == tuple(
        sorted(item.study_instance_uid for item in manifest.gold_memberships if item.fold != fold)
    )
    assert set(plan.held_out_study_uids) == set(manifest.held_out_study_uids(fold))
    assert set(plan.eligible_soft_target_study_uids) == {
        item.study_instance_uid
        for item in report_only
        if item.fold != fold and item.study_instance_uid != unavailable
    }
    assert set(plan.excluded_report_only_study_uids) == {
        item.study_instance_uid
        for item in report_only
        if item.fold == fold or item.study_instance_uid == unavailable
    }
    assert not set(plan.held_out_study_uids).intersection(plan.eligible_soft_target_study_uids)


def test_soft_targets_keep_probabilities_and_teacher_checkpoint_lineage() -> None:
    manifest = _manifest()
    fold = 0
    plan = build_ticket22_fold_plan(
        manifest,
        fold,
        tuple(item.study_instance_uid for item in manifest.report_only_memberships),
    )
    uid = plan.eligible_soft_target_study_uids[0]
    probabilities = (0.15, 0.50, 0.85) + (0.25,) * (NUM_TARGETS - 3)

    examples = create_fold_soft_target_examples(
        (_candidate(uid, fold),),
        {uid: probabilities},
        fold_manifest=manifest,
        fold=fold,
        eligible_study_uids=(uid,),
        excluded_study_uids=plan.excluded_report_only_study_uids,
        teacher_checkpoint_sha256="a" * 64,
    )

    example = examples[0]
    assert example.target_kind == "teacher_probability"
    assert example.targets[:3].tolist() == pytest.approx(probabilities[:3])
    assert example.supervision_mask.all()
    assert example.provenance.source_kind == "pseudo"
    assert example.provenance.pseudo_label_version == "ticket22-teacher-soft-probabilities-v1"
    assert example.provenance.teacher_checkpoint_sha256 == "a" * 64


def test_soft_target_creation_rejects_held_out_candidates_and_bad_scores() -> None:
    manifest = _manifest()
    fold = 0
    held_out_uid = next(
        item.study_instance_uid
        for item in manifest.report_only_memberships
        if item.fold == fold
    )
    plan = build_ticket22_fold_plan(
        manifest,
        fold,
        tuple(item.study_instance_uid for item in manifest.report_only_memberships),
    )

    with pytest.raises(DistillationExperimentError, match="held-out"):
        create_fold_soft_target_examples(
            (_candidate(held_out_uid, fold),),
            {held_out_uid: (0.5,) * NUM_TARGETS},
            fold_manifest=manifest,
            fold=fold,
            eligible_study_uids=(held_out_uid,),
            excluded_study_uids=plan.excluded_report_only_study_uids,
            teacher_checkpoint_sha256="b" * 64,
        )

    eligible_uid = plan.eligible_soft_target_study_uids[0]
    with pytest.raises(DistillationExperimentError, match="probabilities"):
        create_fold_soft_target_examples(
            (_candidate(eligible_uid, fold),),
            {eligible_uid: (1.2,) + (0.5,) * (NUM_TARGETS - 1)},
            fold_manifest=manifest,
            fold=fold,
            eligible_study_uids=(eligible_uid,),
            excluded_study_uids=plan.excluded_report_only_study_uids,
            teacher_checkpoint_sha256="b" * 64,
        )


def test_ticket22_comparison_pairs_teacher_and_both_student_arms() -> None:
    manifest = _manifest()
    labels = {
        item.study_instance_uid: item.reference_labels
        for item in manifest.gold_memberships
    }
    perfect = {uid: tuple(float(value) for value in values) for uid, values in labels.items()}
    flat = {uid: (0.5,) * NUM_TARGETS for uid in labels}

    comparisons = compare_ticket22_oof_predictions(
        perfect,
        flat,
        perfect,
        manifest,
        bootstrap_resamples=30,
        bootstrap_seed=22,
    )

    assert set(comparisons) == {
        "undistilled_student_vs_teacher",
        "distilled_student_vs_undistilled_student",
        "distilled_student_vs_teacher",
    }
    assert comparisons["undistilled_student_vs_teacher"]["prediction_count"] == 12
    assert comparisons["undistilled_student_vs_teacher"]["delta"]["pooled_macro_auc"] == -0.5
    assert comparisons["distilled_student_vs_undistilled_student"]["delta"]["pooled_macro_auc"] == 0.5
    assert comparisons["distilled_student_vs_teacher"]["delta"]["pooled_macro_auc"] == 0.0


def test_ticket22_comparison_rejects_incomplete_student_oof() -> None:
    manifest = _manifest()
    expected = {
        item.study_instance_uid: (0.5,) * NUM_TARGETS
        for item in manifest.gold_memberships
    }
    incomplete = dict(expected)
    incomplete.pop(next(iter(incomplete)))

    with pytest.raises(ImageModelComparisonError, match="same complete gold cohort"):
        compare_ticket22_oof_predictions(expected, expected, incomplete, manifest, bootstrap_resamples=10)


def test_soft_target_training_records_parent_checkpoint_and_both_exposure_sources(tmp_path) -> None:
    manifest = _manifest()
    fold = 0
    plan = build_ticket22_fold_plan(
        manifest,
        fold,
        tuple(item.study_instance_uid for item in manifest.report_only_memberships),
    )
    soft_uid = plan.eligible_soft_target_study_uids[0]
    soft_example = create_fold_soft_target_examples(
        (_candidate(soft_uid, fold),),
        {soft_uid: (0.25,) * NUM_TARGETS},
        fold_manifest=manifest,
        fold=fold,
        eligible_study_uids=(soft_uid,),
        excluded_study_uids=plan.excluded_report_only_study_uids,
        teacher_checkpoint_sha256="c" * 64,
    )[0]
    gold_examples = tuple(
        TrainingExample(
            study_instance_uid=f"training-gold-{index}",
            images=np.full((1, 1, 1, 16, 16), float(index), dtype=np.float32),
            slice_mask=np.ones((1, 1), dtype=bool),
            series_mask=np.ones(1, dtype=bool),
            targets=np.full(NUM_TARGETS, float(index), dtype=np.float32),
            supervision_mask=np.ones(NUM_TARGETS, dtype=bool),
            provenance=SupervisionProvenance(
                source_kind="reference",
                fold=fold,
                eligible_study_uids=(f"training-gold-{index}",),
                excluded_study_uids=(),
            ),
        )
        for index in range(2)
    )
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=1,
        max_series=1,
        successful_optimizer_steps=2,
        checkpoint_interval_steps=1,
    )
    identity = {
        "ticket": 22,
        "fold_manifest_checksum": manifest.manifest_checksum,
        "teacher_checkpoint_sha256": "c" * 64,
        "soft_target_temperature": 1.0,
    }

    result = train_fixed_budget(
        gold_examples,
        config,
        additional_examples=(soft_example,),
        additional_batch_order_seed=config.batch_order_seed + 22,
        artifact_dir=tmp_path / "distilled-student",
        fold=fold,
        identity=identity,
        parent_provenance={"parent_checkpoint_sha256": "c" * 64},
    )

    assert result.successful_optimizer_steps == 2
    assert result.exposures_by_source is not None
    assert result.exposures_by_source["reference"]["sample_exposures"] > 0
    assert result.exposures_by_source["pseudo"]["sample_exposures"] > 0
    assert result.final_checkpoint_path is not None
    payload = load_training_checkpoint(
        result.final_checkpoint_path,
        expected_config=config,
        expected_fold=fold,
        expected_identity=identity,
        expected_checkpoint_kind="final_selected",
    )
    assert payload["parent_provenance"]["parent_checkpoint_sha256"] == "c" * 64
