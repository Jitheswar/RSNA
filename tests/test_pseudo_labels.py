from __future__ import annotations

from dataclasses import replace

import pytest

from rsna_knee.folds import FoldManifest, freeze_folds
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.pseudo_labels import (
    PseudoLabelAdmissionError,
    PseudoLabelTeacherProvenance,
    admit_fold_pseudo_labels,
    create_pseudo_label_generation,
)
from rsna_knee.targets import NUM_TARGETS
from rsna_knee.synthetic import make_synthetic_study
from rsna_knee.training import (
    TrainingConfig,
    TrainingDataError,
    build_fold_pseudo_candidate_examples,
    train_one_fold,
)


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
        source_version="synthetic-ticket21-v1",
        seed=211,
    )


def _teacher(manifest: FoldManifest, fold: int) -> PseudoLabelTeacherProvenance:
    return PseudoLabelTeacherProvenance(
        fold=fold,
        fold_manifest_checksum=manifest.manifest_checksum,
        repeat_name=manifest.repeat_name,
        training_study_uids=tuple(
            sorted(
                item.study_instance_uid
                for item in manifest.gold_memberships
                if item.fold != fold
            )
        ),
        held_out_study_uids=tuple(sorted(manifest.held_out_study_uids(fold))),
        source_fingerprint="a" * 64,
        code_revision="b" * 40,
        code_sha256="c" * 64,
        configuration_fingerprint="d" * 64,
        checkpoint_sha256="e" * 64,
    )


def test_fold_admission_masks_uncertain_scores_and_records_teacher_lineage() -> None:
    manifest = _manifest()
    fold = 0
    expected_candidates = {
        item.study_instance_uid
        for item in manifest.report_only_memberships
        if item.fold != fold
    }
    selected = tuple(sorted(expected_candidates)[:1])
    excluded = tuple(sorted(expected_candidates - set(selected)))
    scores = {
        selected[0]: (0.05, 0.50, 0.95) + (0.50,) * (NUM_TARGETS - 3),
    }

    generation = create_pseudo_label_generation(
        _teacher(manifest, fold),
        scores,
        eligible_study_uids=selected,
        excluded_study_uids=excluded,
    )
    admitted = admit_fold_pseudo_labels(
        manifest,
        fold,
        generation,
    )

    vector = admitted.by_study[selected[0]]
    assert vector.targets[:3] == (0, None, 1)
    assert vector.supervision_mask[:3] == (1, 0, 1)
    assert vector.source_kind == "pseudo"
    assert admitted.held_out_study_uids == tuple(sorted(manifest.held_out_study_uids(fold)))
    assert admitted.to_public_dict()["teacher"]["checkpoint_sha256"] == "e" * 64
    assert selected[0] not in str(admitted.to_public_dict())


def test_fold_admission_rejects_held_out_studies_in_generated_targets() -> None:
    manifest = _manifest()
    fold = 0
    held_out = next(
        item.study_instance_uid
        for item in manifest.report_only_memberships
        if item.fold == fold
    )
    training_report_only = {
        item.study_instance_uid
        for item in manifest.report_only_memberships
        if item.fold != fold
    }
    with pytest.raises(PseudoLabelAdmissionError, match="held-out"):
        generation = create_pseudo_label_generation(
            _teacher(manifest, fold),
            {
                uid: (0.5,) * NUM_TARGETS
                for uid in training_report_only | {held_out}
            },
            eligible_study_uids=tuple(sorted(training_report_only | {held_out})),
            excluded_study_uids=(),
        )
        admit_fold_pseudo_labels(manifest, fold, generation)


def test_fold_admission_rejects_a_teacher_with_a_report_only_training_input() -> None:
    manifest = _manifest()
    fold = 0
    report_candidates = tuple(
        sorted(
            item.study_instance_uid
            for item in manifest.report_only_memberships
            if item.fold != fold
        )
    )
    selected = report_candidates[:1]
    clean_teacher = _teacher(manifest, fold)
    contaminated_teacher = replace(
        clean_teacher,
        training_study_uids=tuple(sorted((*clean_teacher.training_study_uids, report_candidates[1]))),
    )
    generation = create_pseudo_label_generation(
        contaminated_teacher,
        {selected[0]: (0.5,) * NUM_TARGETS},
        eligible_study_uids=selected,
        excluded_study_uids=report_candidates[1:],
    )

    with pytest.raises(PseudoLabelAdmissionError, match="teacher training cohort"):
        admit_fold_pseudo_labels(manifest, fold, generation)


def test_teacher_provenance_rejects_unresolved_checkpoint_identity() -> None:
    manifest = _manifest()
    with pytest.raises(PseudoLabelAdmissionError, match="checkpoint hash"):
        replace(_teacher(manifest, 0), checkpoint_sha256="")


def test_admitted_pseudo_labels_reach_student_checkpoint_lineage(tmp_path) -> None:
    studies = tuple(
        make_synthetic_study(
            f"synthetic-study-{index}",
            series_count=1,
            slices_per_series=2,
            seed=400 + index,
        )
        for index in range(12)
    )
    gold = tuple(
        SourceStudy(
            study.study_instance_uid,
            True,
            tuple((index + target) % 2 for target in range(NUM_TARGETS)),
        )
        for index, study in enumerate(studies[:9])
    )
    report_only = tuple(
        SourceStudy(study.study_instance_uid, True, (None,) * NUM_TARGETS)
        for study in studies[9:]
    )
    manifest = freeze_folds(
        SourceManifest("train", gold + report_only, ()),
        source_version="synthetic-ticket21-student-v1",
        seed=217,
    )
    fold = 0
    report_candidates = tuple(
        sorted(
            item.study_instance_uid
            for item in manifest.report_only_memberships
            if item.fold != fold
        )
    )
    selected = report_candidates[:1]
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=2,
        max_series=1,
        successful_optimizer_steps=2,
        checkpoint_interval_steps=1,
    )
    candidate_examples = build_fold_pseudo_candidate_examples(
        {study.study_instance_uid: study for study in studies},
        manifest,
        fold,
        selected,
        preprocess_config=config.preprocess_config(),
        packing_config=config.packing_config(),
    )
    assert tuple(item.study_instance_uid for item in candidate_examples) == selected
    assert all(not item.supervision_mask.any() for item in candidate_examples)
    held_out_report_uid = next(
        item.study_instance_uid
        for item in manifest.report_only_memberships
        if item.fold == fold
    )
    with pytest.raises(TrainingDataError, match="report-only studies in the training partition"):
        build_fold_pseudo_candidate_examples(
            {study.study_instance_uid: study for study in studies},
            manifest,
            fold,
            (held_out_report_uid,),
        )
    generation = create_pseudo_label_generation(
        _teacher(manifest, fold),
        {selected[0]: (0.05,) * NUM_TARGETS},
        eligible_study_uids=selected,
        excluded_study_uids=report_candidates[1:],
    )
    admitted = admit_fold_pseudo_labels(manifest, fold, generation)

    result = train_one_fold(
        {study.study_instance_uid: study for study in studies},
        manifest,
        fold,
        tmp_path / "pseudo-student",
        config=config,
        source_version="synthetic-ticket21-student-v1",
        require_preflight=False,
        allow_local_smoke=True,
        supervision=admitted,
        ticket21_pseudolabel_design=True,
    )

    assert result.evaluation is not None
    assert set(result.training_study_uids) == set(admitted.training_study_uids)
    assert result.to_public_dict()["supervision"]["kind"] == "pseudo"
    assert result.exposures_by_source is not None
    assert result.exposures_by_source["pseudo"]["sample_exposures"] > 0
    private_run = result.to_private_dict()
    assert private_run["supervision"]["source_artifact_digest"] == generation.semantic_digest
    pseudo_examples = [
        item
        for item in private_run["supervision"]["resolved_studies"]
        if item["source_kind"] == "pseudo"
    ]
    assert len(pseudo_examples) == 1
    assert pseudo_examples[0]["source_teacher_checkpoint_sha256"] == "e" * 64
