from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch

from rsna_knee.training import (
    CheckpointError,
    LEGACY_CHECKPOINT_SCHEMA_VERSION,
    TrainingConfig,
    TrainingConfigurationError,
    TrainingExample,
    SupervisionProvenance,
    TrainingInterrupted,
    TrainingProgressError,
    TrainingResult,
    classify_retry_failure,
    fork_training_run,
    load_training_checkpoint,
    masked_bce_loss,
    retry_is_allowed,
    run_cpu_preflight,
    semantic_fingerprint,
    _json_sha256,
    _legacy_config_fingerprint,
    _legacy_scientific_config_payload,
    _train_fixed_budget_impl,
    train_fixed_budget,
    train_one_fold,
)
from rsna_knee.folds import freeze_folds
from rsna_knee.inventory import SourceManifest, SourceStudy
from rsna_knee.model import HierarchicalStudyModel
from rsna_knee.synthetic import make_synthetic_study
from rsna_knee.targets import TARGET_COLUMNS


def test_masked_bce_averages_only_supervised_entries() -> None:
    logits = torch.zeros((1, 12))
    logits[0, 2] = 2.0
    targets = torch.zeros((1, 12))
    targets[0, 0] = 1.0
    targets[0, 2] = 1.0
    supervision_mask = torch.zeros((1, 12), dtype=torch.bool)
    supervision_mask[0, (0, 2)] = True

    loss = masked_bce_loss(logits, targets, supervision_mask)

    assert loss is not None
    expected = (torch.nn.functional.softplus(torch.tensor(0.0)) + torch.nn.functional.softplus(torch.tensor(-2.0))) / 2
    assert torch.allclose(loss, expected)


def test_run_record_reports_heldout_model_inference_time() -> None:
    config = TrainingConfig(image_size=(16, 16), channels=(2, 4, 8))
    result = TrainingResult(
        model=HierarchicalStudyModel(config.model_config()),
        config=config,
        successful_optimizer_steps=0,
        scheduler_steps=0,
        attempted_batches=0,
        skipped_batches=0,
        loss_history=(),
        heldout_model_inference_seconds=1.25,
    )

    assert result.to_public_dict()["counts"]["heldout_model_inference_seconds"] == 1.25


def test_fully_masked_batches_make_no_optimizer_progress() -> None:
    example = TrainingExample(
        study_instance_uid="masked-study",
        images=np.zeros((1, 1, 1, 16, 16), dtype=np.float32),
        slice_mask=np.ones((1, 1), dtype=bool),
        series_mask=np.ones(1, dtype=bool),
        targets=np.zeros(12, dtype=np.float32),
        supervision_mask=np.zeros(12, dtype=bool),
    )
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=1,
        max_series=1,
        successful_optimizer_steps=1,
        persistent_no_progress_limit=2,
    )

    with pytest.raises(TrainingProgressError, match="no supervised progress"):
        train_fixed_budget((example,), config)


def _tiny_example(study_uid: str, target: int) -> TrainingExample:
    return TrainingExample(
        study_instance_uid=study_uid,
        images=np.full((1, 1, 1, 16, 16), 0.25 + target * 0.5, dtype=np.float32),
        slice_mask=np.ones((1, 1), dtype=bool),
        series_mask=np.ones(1, dtype=bool),
        targets=np.full(12, target, dtype=np.float32),
        supervision_mask=np.ones(12, dtype=bool),
    )


def test_fixed_budget_makes_learning_progress_and_records_each_step(tmp_path) -> None:
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=1,
        max_series=1,
        successful_optimizer_steps=12,
        checkpoint_interval_steps=4,
    )
    result = train_fixed_budget(
        (_tiny_example("tiny-negative", 0), _tiny_example("tiny-positive", 1)),
        config,
        artifact_dir=tmp_path / "run",
    )

    assert result.successful_optimizer_steps == 12
    assert result.scheduler_steps == 12
    assert len(result.loss_history) == 12
    assert result.loss_history[-1] < result.loss_history[0]
    assert result.final_checkpoint_path is not None
    assert result.final_checkpoint_path.exists()
    assert result.latest_checkpoint_path is not None
    assert result.latest_checkpoint_path.exists()

    payload = load_training_checkpoint(
        result.final_checkpoint_path,
        expected_config=config,
    )
    assert payload["selection_policy"] == "final_checkpoint"
    assert payload["progress"]["successful_optimizer_steps"] == 12
    with pytest.raises(TrainingConfigurationError, match="scientific configuration"):
        load_training_checkpoint(
            result.final_checkpoint_path,
            expected_config=TrainingConfig(
                image_size=(16, 16),
                channels=(2, 4, 8),
                max_slices=1,
                max_series=1,
                successful_optimizer_steps=13,
            ),
        )


def test_additive_weak_microbatch_preserves_gold_sequence_and_adds_exposure() -> None:
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=1,
        max_series=1,
        successful_optimizer_steps=2,
        checkpoint_interval_steps=1,
    )
    gold = (_tiny_example("gold-a", 0), _tiny_example("gold-b", 1))
    weak_provenance = SupervisionProvenance(
        source_kind="weak",
        source_id="ticket15-test-fixture",
        fold=0,
        weak_label_version="report-only-confident-binary-v1",
        mixture_policy="reference-precedence-plus-report-only-v1",
        source_report_ref="private://test-report",
        source_report_sha256="a" * 64,
    )
    weak = tuple(
        TrainingExample(
            study_instance_uid=f"weak-{name}",
            images=np.full((1, 1, 1, 16, 16), value, dtype=np.float32),
            slice_mask=np.ones((1, 1), dtype=bool),
            series_mask=np.ones(1, dtype=bool),
            targets=np.asarray([1.0, 0.0] + [0.0] * 10, dtype=np.float32),
            supervision_mask=np.asarray([True, True] + [False] * 10),
            provenance=weak_provenance,
        )
        for name, value in (("a", 0.2), ("b", 0.8))
    )
    weak_seed = config.batch_order_seed + 18_180_018

    gold_result = _train_fixed_budget_impl(gold, config)
    additive_result = _train_fixed_budget_impl(
        gold,
        config,
        additional_examples=weak,
        additional_batch_order_seed=weak_seed,
    )

    assert gold_result.gold_exposure_sequence_sha256 == additive_result.gold_exposure_sequence_sha256
    assert additive_result.gold_exposure_sequence_sha256 is not None
    assert additive_result.weak_exposure_sequence_sha256 is not None
    assert gold_result.initial_model_state_sha256 == additive_result.initial_model_state_sha256
    assert additive_result.successful_optimizer_steps == gold_result.successful_optimizer_steps == 2
    assert additive_result.scheduler_steps == gold_result.scheduler_steps == 2
    assert additive_result.exposures_by_source is not None
    assert additive_result.exposures_by_source["reference"] == {
        "sample_exposures": 4,
        "supervised_entry_exposures": 48,
    }
    assert additive_result.exposures_by_source["weak"] == {
        "sample_exposures": 4,
        "supervised_entry_exposures": 8,
    }
    assert additive_result.sample_exposures == 8
    assert additive_result.supervised_entry_exposures == 56


def test_one_fold_uses_reference_gold_only_and_evaluates_final_checkpoint(tmp_path) -> None:
    studies = tuple(
        make_synthetic_study(f"gold-{index}", series_count=1, slices_per_series=2, seed=100 + index)
        for index in range(9)
    )
    source = SourceManifest(
        "train",
        tuple(
            SourceStudy(
                study.study_instance_uid,
                True,
                tuple((index + target_index) % 2 for target_index in range(len(TARGET_COLUMNS))),
            )
            for index, study in enumerate(studies)
        )
        + (SourceStudy("report-only", True, (None,) * len(TARGET_COLUMNS)),),
        (),
    )
    fold_manifest = freeze_folds(source, source_version="synthetic-training-v1", seed=17)
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=2,
        max_series=1,
        successful_optimizer_steps=2,
        checkpoint_interval_steps=1,
    )

    result = train_one_fold(
        {study.study_instance_uid: study for study in studies},
        fold_manifest,
        0,
        tmp_path / "fold-0",
        config=config,
        source_version="synthetic-training-v1",
        require_preflight=False,
        allow_local_smoke=True,
    )

    assert result.successful_optimizer_steps == 2
    assert result.evaluation is not None
    assert set(result.training_study_uids).isdisjoint(result.held_out_study_uids)
    assert "report-only" not in result.training_study_uids
    assert (tmp_path / "fold-0" / "configuration.json").exists()
    assert (tmp_path / "fold-0" / "metrics.json").exists()
    assert (tmp_path / "fold-0" / "run.json").exists()
    assert result.evaluation.fold == 0


def test_one_fold_rejects_a_nonempty_trainer_directory(tmp_path) -> None:
    studies = tuple(
        make_synthetic_study(f"guard-{index}", series_count=1, slices_per_series=2, seed=200 + index)
        for index in range(9)
    )
    source = SourceManifest(
        "train",
        tuple(
            SourceStudy(
                study.study_instance_uid,
                True,
                tuple((index + target_index) % 2 for target_index in range(len(TARGET_COLUMNS))),
            )
            for index, study in enumerate(studies)
        ),
        (),
    )
    fold_manifest = freeze_folds(source, source_version="synthetic-training-v1", seed=19)
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=2,
        max_series=1,
        successful_optimizer_steps=1,
        checkpoint_interval_steps=1,
    )
    trainer_directory = tmp_path / "existing-trainer"
    trainer_directory.mkdir()
    marker = trainer_directory / "unrelated-artifact.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="existing experiment directory"):
        train_one_fold(
            {study.study_instance_uid: study for study in studies},
            fold_manifest,
            0,
            trainer_directory,
            config=config,
            source_version="synthetic-training-v1",
            require_preflight=False,
            allow_local_smoke=True,
        )

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_cpu_preflight_records_the_required_runtime_checks(tmp_path) -> None:
    result = run_cpu_preflight(
        output_path=tmp_path / "preflight.json",
        require_isolated_python=True,
    )

    assert result.passed is True
    assert all(result.checks.values())
    assert result.environment["python_version"].startswith("3.11")
    assert (tmp_path / "preflight.json").exists()


def test_interrupted_training_resumes_same_run_with_a_new_execution_attempt(tmp_path) -> None:
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=1,
        max_series=1,
        successful_optimizer_steps=4,
        checkpoint_interval_steps=1,
    )
    examples = (_tiny_example("resume-negative", 0), _tiny_example("resume-positive", 1))
    interrupted_dir = tmp_path / "interrupted"

    with pytest.raises(TrainingInterrupted):
        train_fixed_budget(
            examples,
            config,
            artifact_dir=interrupted_dir,
            interrupt_after_steps=2,
        )

    first_checkpoint = load_training_checkpoint(
        interrupted_dir / "checkpoints" / "latest.pt",
        expected_config=config,
    )
    resumed = train_fixed_budget(
        examples,
        config,
        artifact_dir=interrupted_dir,
        resume_from=interrupted_dir / "checkpoints" / "latest.pt",
    )

    uninterrupted = train_fixed_budget(
        examples,
        config,
        artifact_dir=tmp_path / "uninterrupted",
    )

    assert resumed.run_id == first_checkpoint["run_id"]
    assert resumed.successful_optimizer_steps == config.successful_optimizer_steps
    assert resumed.execution_attempt_id != first_checkpoint["execution_attempt_id"]
    assert resumed.model.state_dict().keys() == uninterrupted.model.state_dict().keys()
    for name, value in uninterrupted.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[name], value)
    attempts = sorted((interrupted_dir / "attempts").iterdir())
    assert len(attempts) == 2
    assert (attempts[0] / "failure.json").exists()
    assert (attempts[1] / "completion.json").exists()


def test_checkpoint_persists_amp_progress_metric_and_rng_state(tmp_path) -> None:
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=1,
        max_series=1,
        successful_optimizer_steps=2,
    )
    torch.manual_seed(7)
    from rsna_knee.model import HierarchicalStudyModel

    model = HierarchicalStudyModel(config.model_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    scaler = torch.amp.GradScaler("cpu")
    from rsna_knee.training import save_training_checkpoint, seed_everything

    seed_everything(44)
    random.random()
    np.random.random()
    torch.rand(1)
    checkpoint = tmp_path / "checkpoint.pt"
    save_training_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        config,
        successful_optimizer_steps=1,
        attempted_batches=2,
        skipped_batches=1,
        loss_history=(0.75,),
        epoch=1,
        global_step=1,
        best_metric=0.625,
        scaler=scaler,
        fold=1,
        identity={"data_fingerprint": "fixture-data", "fold": 1},
        overwrite=False,
    )
    expected_next_random = (random.random(), np.random.random(), torch.rand(1).item())
    seed_everything(99)

    restored_scaler = torch.amp.GradScaler("cpu")
    restored_model = HierarchicalStudyModel(config.model_config())
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=config.learning_rate)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lr_lambda=lambda _: 1.0
    )
    payload = load_training_checkpoint(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
        expected_config=config,
        expected_fold=1,
        restore_rng=True,
    )

    assert payload["checkpoint_version"] == payload["schema_version"]
    assert payload["checkpoint_kind"] == "latest_resumable"
    assert payload["amp_scaler_state_dict"] == scaler.state_dict()
    assert payload["best_metric"] == 0.625
    assert payload["progress"]["epoch"] == 1
    assert payload["progress"]["global_step"] == 1
    assert (random.random(), np.random.random(), torch.rand(1).item()) == pytest.approx(
        expected_next_random
    )

    final_checkpoint = tmp_path / "final.pt"
    save_training_checkpoint(
        final_checkpoint,
        model,
        optimizer,
        scheduler,
        config,
        successful_optimizer_steps=1,
        attempted_batches=2,
        skipped_batches=1,
        loss_history=(0.75,),
        checkpoint_kind="final_selected",
        overwrite=True,
    )
    with pytest.raises(FileExistsError):
        save_training_checkpoint(
            final_checkpoint,
            model,
            optimizer,
            scheduler,
            config,
            successful_optimizer_steps=1,
            attempted_batches=2,
            skipped_batches=1,
            loss_history=(0.75,),
            checkpoint_kind="final_selected",
            overwrite=True,
        )


def test_checkpoint_write_failure_preserves_the_previous_complete_checkpoint(tmp_path, monkeypatch) -> None:
    from rsna_knee.model import HierarchicalStudyModel
    from rsna_knee.training import atomic_torch_save, save_training_checkpoint

    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=1,
        max_series=1,
        successful_optimizer_steps=1,
    )
    model = HierarchicalStudyModel(config.model_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    checkpoint = tmp_path / "latest.pt"
    save_training_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        config,
        successful_optimizer_steps=0,
        attempted_batches=0,
        skipped_batches=0,
        loss_history=(),
        overwrite=False,
    )
    previous_bytes = checkpoint.read_bytes()

    def fail_save(*args, **kwargs):
        raise RuntimeError("injected interrupted write")

    monkeypatch.setattr("rsna_knee.training.torch.save", fail_save)
    with pytest.raises(CheckpointError):
        atomic_torch_save(checkpoint, {"partial": True}, overwrite=True)

    assert checkpoint.read_bytes() == previous_bytes
    assert not list(tmp_path.glob(".latest.pt.partial-*"))


def test_training_records_epoch_and_time_checkpoint_boundaries(tmp_path) -> None:
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=1,
        max_series=1,
        successful_optimizer_steps=4,
        checkpoint_interval_steps=100,
        checkpoint_interval_seconds=10.0,
    )
    examples = tuple(_tiny_example(f"timed-{index}", index % 2) for index in range(5))
    ticks = iter((0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0))

    result = train_fixed_budget(
        examples,
        config,
        artifact_dir=tmp_path / "timed",
        checkpoint_clock=lambda: next(ticks),
    )

    assert result.epoch == 1
    assert result.latest_checkpoint_path is not None
    latest = load_training_checkpoint(result.latest_checkpoint_path, expected_config=config)
    events = latest["checkpoint_events"]
    assert any("time_interval" in str(event["reason"]) for event in events)
    assert any(event["epoch"] == 1 for event in events)
    assert all(float(event["overdue_seconds"]) >= 0.0 for event in events)
    assert latest["checkpoint_kind"] == "latest_resumable"
    assert result.final_checkpoint_path is not None
    final = load_training_checkpoint(result.final_checkpoint_path, expected_config=config)
    assert final["checkpoint_kind"] == "final_selected"
    with pytest.raises(TrainingConfigurationError, match="checkpoint kind"):
        train_fixed_budget(
            examples,
            config,
            artifact_dir=tmp_path / "timed",
            resume_from=result.final_checkpoint_path,
        )


def test_resume_uses_last_complete_history_checkpoint_and_rejects_data_changes(tmp_path) -> None:
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=1,
        max_series=1,
        successful_optimizer_steps=3,
        checkpoint_interval_steps=1,
    )
    examples = (_tiny_example("recover-negative", 0), _tiny_example("recover-positive", 1))
    run_dir = tmp_path / "recover"
    identity = {"data_fingerprint": "source-a", "fold_manifest_checksum": "fold-a", "fold": 0}

    with pytest.raises(TrainingInterrupted):
        train_fixed_budget(
            examples,
            config,
            artifact_dir=run_dir,
            identity=identity,
            interrupt_after_steps=2,
        )

    latest = run_dir / "checkpoints" / "latest.pt"
    latest.write_bytes(b"interrupted write")

    resumed = train_fixed_budget(
        examples,
        config,
        artifact_dir=run_dir,
        identity=identity,
        resume_from=latest,
    )
    assert resumed.successful_optimizer_steps == 3
    assert resumed.resumed_from_checkpoint is not None
    assert resumed.resumed_from_checkpoint != str(latest)

    with pytest.raises(TrainingConfigurationError, match="semantic fingerprint"):
        train_fixed_budget(
            examples,
            config,
            artifact_dir=tmp_path / "changed",
            identity={**identity, "data_fingerprint": "source-b"},
            resume_from=run_dir / "checkpoints" / "latest.pt",
        )


def test_resume_records_and_replays_only_batches_after_last_checkpoint(tmp_path) -> None:
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=1,
        max_series=1,
        successful_optimizer_steps=4,
        checkpoint_interval_steps=100,
        checkpoint_interval_seconds=100_000.0,
    )
    examples = tuple(_tiny_example(f"replay-{index}", index % 2) for index in range(5))
    run_dir = tmp_path / "replay"

    with pytest.raises(TrainingInterrupted):
        train_fixed_budget(
            examples,
            config,
            artifact_dir=run_dir,
            interrupt_after_steps=2,
        )

    first_attempt = run_dir / "attempts" / "attempt-0001"
    failure = json.loads((first_attempt / "failure.json").read_text(encoding="utf-8"))
    assert failure["uncheckpointed_batches"] == 2
    assert failure["replayed_batches_on_next_resume"] == 2

    resumed = train_fixed_budget(
        examples,
        config,
        artifact_dir=run_dir,
        resume_from=run_dir / "checkpoints" / "latest.pt",
    )

    assert resumed.replayed_batches == 2
    assert resumed.successful_optimizer_steps == config.successful_optimizer_steps
    latest = load_training_checkpoint(resumed.latest_checkpoint_path, expected_config=config)
    assert latest["recovery"]["replayed_batches"] == 2
    assert latest["recovery"]["max_uncheckpointed_batches"] >= 2


def test_legacy_v1_checkpoint_migrates_for_resume_without_changing_run_identity(tmp_path) -> None:
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=1,
        max_series=1,
        successful_optimizer_steps=3,
        checkpoint_interval_steps=1,
    )
    examples = (_tiny_example("legacy-negative", 0), _tiny_example("legacy-positive", 1))
    source_dir = tmp_path / "source"
    identity = {"data_fingerprint": "legacy-data", "fold": None}
    interrupted = train_fixed_budget(
        examples,
        config,
        artifact_dir=source_dir,
        identity=identity,
    )
    assert interrupted.latest_checkpoint_path is not None
    current = torch.load(interrupted.latest_checkpoint_path, map_location="cpu", weights_only=False)
    legacy_config = dict(current["config"])
    legacy_config["progress"] = dict(legacy_config["progress"])
    legacy_config["progress"].pop("checkpoint_interval_seconds", None)
    legacy_config.pop("loss", None)
    legacy_config.pop("augmentation", None)
    legacy_fingerprint = _legacy_config_fingerprint(config)
    legacy_run_id = _json_sha256(
        {
            "config": _legacy_scientific_config_payload(legacy_config),
            "identity": identity,
        }
    )
    legacy = dict(current)
    legacy["schema_version"] = LEGACY_CHECKPOINT_SCHEMA_VERSION
    legacy.pop("checkpoint_version", None)
    legacy.pop("checkpoint_kind", None)
    legacy.pop("checkpoint_reason", None)
    legacy.pop("semantic_fingerprint", None)
    legacy.pop("execution_attempt_id", None)
    legacy.pop("epoch", None)
    legacy.pop("global_step", None)
    legacy.pop("amp_scaler_state_dict", None)
    legacy.pop("scaler_state_dict", None)
    legacy.pop("best_metric", None)
    legacy.pop("checkpoint_timing", None)
    legacy.pop("checkpoint_events", None)
    legacy.pop("sampler_state", None)
    legacy.pop("recovery", None)
    legacy.pop("parent_provenance", None)
    legacy["config"] = legacy_config
    legacy["config_fingerprint"] = legacy_fingerprint
    legacy["run_id"] = legacy_run_id
    legacy["identity"] = identity
    legacy["progress"] = {
        key: current["progress"][key]
        for key in (
            "epoch",
            "global_step",
            "successful_optimizer_steps",
            "scheduler_steps",
            "attempted_batches",
            "skipped_all_masked_batches",
            "loss_history",
        )
    }
    legacy_dir = tmp_path / "legacy"
    (legacy_dir / "checkpoints").mkdir(parents=True)
    (legacy_dir / "attempts").mkdir()
    from rsna_knee.archive import atomic_json_write

    atomic_json_write(
        legacy_dir / "configuration.json",
        {
            "schema_version": "rsna-knee-training-v1",
            "run_id": legacy_run_id,
            "fold": None,
            "config": legacy_config,
            "scientific_configuration_fingerprint": legacy_fingerprint,
            "identity": identity,
        },
    )
    from rsna_knee.training import atomic_torch_save

    legacy_checkpoint = legacy_dir / "checkpoints" / "latest.pt"
    atomic_torch_save(legacy_checkpoint, legacy, overwrite=False)

    migrated = load_training_checkpoint(
        legacy_checkpoint,
        expected_config=config,
        expected_identity=identity,
        expected_checkpoint_kind="latest_resumable",
    )
    assert migrated["schema_version"] != LEGACY_CHECKPOINT_SCHEMA_VERSION
    assert migrated["migrated_from_schema_version"] == LEGACY_CHECKPOINT_SCHEMA_VERSION

    resumed = train_fixed_budget(
        examples,
        config,
        artifact_dir=legacy_dir,
        identity=identity,
        resume_from=legacy_checkpoint,
    )
    assert resumed.run_id == legacy_run_id
    assert resumed.successful_optimizer_steps == config.successful_optimizer_steps


def test_semantic_identity_ignores_execution_details_but_covers_scientific_inputs() -> None:
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=1,
        max_series=1,
        successful_optimizer_steps=2,
    )
    identity = {"data_fingerprint": "source-a", "fold_manifest_checksum": "fold-a", "fold": 0}

    baseline = semantic_fingerprint(config, identity)
    operational_change = semantic_fingerprint(
        config,
        {**identity, "execution_attempt_id": "attempt-2", "slurm_job_id": "42"},
    )
    data_change = semantic_fingerprint(config, {**identity, "data_fingerprint": "source-b"})
    cadence_change = semantic_fingerprint(
        TrainingConfig(
            image_size=(16, 16),
            channels=(2, 4, 8),
            max_slices=1,
            max_series=1,
            successful_optimizer_steps=2,
            checkpoint_interval_steps=1,
            checkpoint_interval_seconds=1.0,
        ),
        identity,
    )
    budget_change = semantic_fingerprint(
        TrainingConfig(
            image_size=(16, 16),
            channels=(2, 4, 8),
            max_slices=1,
            max_series=1,
            successful_optimizer_steps=3,
        ),
        identity,
    )

    assert operational_change == baseline
    assert cadence_change == baseline
    assert data_change != baseline
    assert budget_change != baseline


def test_retry_classification_allows_only_one_transient_infrastructure_retry() -> None:
    transient = classify_retry_failure("externally interrupted by a transient filesystem failure")
    oom = classify_retry_failure(RuntimeError("CUDA out of memory"))

    assert transient.retryable is True
    assert retry_is_allowed("externally interrupted by a transient filesystem failure") is True
    assert retry_is_allowed("externally interrupted by a transient filesystem failure", previous_retries=1) is False
    assert oom.retryable is False
    assert retry_is_allowed(RuntimeError("CUDA out of memory")) is False


def test_semantic_change_requires_an_explicit_new_run_with_parent_provenance(tmp_path) -> None:
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=1,
        max_series=1,
        successful_optimizer_steps=2,
        checkpoint_interval_steps=1,
    )
    examples = (_tiny_example("fork-negative", 0), _tiny_example("fork-positive", 1))
    parent_dir = tmp_path / "parent"
    with pytest.raises(TrainingInterrupted):
        train_fixed_budget(
            examples,
            config,
            artifact_dir=parent_dir,
            identity={"data_fingerprint": "source-a"},
            interrupt_after_steps=1,
        )

    child = fork_training_run(
        examples,
        parent_dir / "checkpoints" / "latest.pt",
        config,
        tmp_path / "child",
        identity={"data_fingerprint": "source-b"},
    )

    assert child.run_id != load_training_checkpoint(parent_dir / "checkpoints" / "latest.pt")["run_id"]
    child_configuration = (tmp_path / "child" / "configuration.json").read_text(encoding="utf-8")
    assert '"parent_run_id"' in child_configuration
    assert child.successful_optimizer_steps == config.successful_optimizer_steps


def test_one_fold_resume_keeps_immutable_configuration_and_writes_attempt_provenance(tmp_path) -> None:
    studies = tuple(
        make_synthetic_study(
            f"resume-fold-{index}", series_count=1, slices_per_series=2, seed=200 + index
        )
        for index in range(9)
    )
    source = SourceManifest(
        "train",
        tuple(
            SourceStudy(
                study.study_instance_uid,
                True,
                tuple((index + target_index) % 2 for target_index in range(len(TARGET_COLUMNS))),
            )
            for index, study in enumerate(studies)
        ),
        (),
    )
    fold_manifest = freeze_folds(source, source_version="synthetic-resume-v1", seed=21)
    config = TrainingConfig(
        image_size=(16, 16),
        channels=(2, 4, 8),
        max_slices=2,
        max_series=1,
        successful_optimizer_steps=2,
        checkpoint_interval_steps=1,
    )
    run_dir = tmp_path / "fold-resume"
    study_map = {study.study_instance_uid: study for study in studies}

    with pytest.raises(TrainingInterrupted):
        train_one_fold(
            study_map,
            fold_manifest,
            0,
            run_dir,
            config=config,
            source_version="synthetic-resume-v1",
            require_preflight=False,
            allow_local_smoke=True,
            interrupt_after_steps=1,
        )

    resumed = train_one_fold(
        study_map,
        fold_manifest,
        0,
        run_dir,
        config=config,
        source_version="synthetic-resume-v1",
        require_preflight=False,
        allow_local_smoke=True,
        resume_from=run_dir / "checkpoints" / "latest.pt",
    )

    assert resumed.evaluation is not None
    assert resumed.run_id == json.loads(
        (run_dir / "configuration.json").read_text(encoding="utf-8")
    )["run_id"]
    assert len(tuple((run_dir / "attempts").iterdir())) == 2
