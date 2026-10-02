import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rsna_knee.training import TrainingConfig, TrainingExample, TrainingInterrupted, CheckpointError, train_fixed_budget, load_training_checkpoint

SPEC = importlib.util.spec_from_file_location("configuration_handoff", Path(__file__).parents[1] / "scripts/carry_training_configuration.py")
handoff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handoff)


def example(index):
    return TrainingExample(study_instance_uid=f"synthetic-{index}",
        images=np.full((1, 1, 1, 16, 16), .25 + (index % 2) * .5, dtype=np.float32),
        slice_mask=np.ones((1, 1), dtype=bool), series_mask=np.ones(1, dtype=bool),
        targets=np.full(12, index % 2, dtype=np.float32), supervision_mask=np.ones(12, dtype=bool))


def test_real_trainer_resume_in_new_directory_preserves_exact_updates_and_state(tmp_path):
    torch.set_num_threads(1)
    config = TrainingConfig(image_size=(16, 16), channels=(2, 4, 8), max_slices=1, max_series=1,
                            successful_optimizer_steps=4, checkpoint_interval_steps=100)
    examples = tuple(example(i) for i in range(7))
    prior, recovery = tmp_path / "prior", tmp_path / "recovery"
    with pytest.raises(TrainingInterrupted):
        train_fixed_budget(examples, config, artifact_dir=prior, interrupt_after_steps=2, checkpoint_clock=lambda: 0.)
    checkpoint = prior / "checkpoints/latest.pt"
    # This is the actual failed call pattern: checkpoint supplied, new artifact
    # directory, no immutable experiment configuration carried with it.
    with pytest.raises(CheckpointError, match="immutable experiment configuration"):
        train_fixed_budget(examples, config, artifact_dir=recovery, resume_from=checkpoint, checkpoint_clock=lambda: 0.)
    assert not (recovery / "checkpoints/latest.pt").exists()
    source = prior / "configuration.json"
    data = source.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    handoff.carry_configuration(source, recovery / "configuration.json", expected_sha256=sha)
    resumed = train_fixed_budget(examples, config, artifact_dir=recovery, resume_from=checkpoint, checkpoint_clock=lambda: 0.)
    continuous = train_fixed_budget(examples, config, artifact_dir=tmp_path / "continuous", checkpoint_clock=lambda: 0.)
    assert source.read_bytes() == (recovery / "configuration.json").read_bytes() == data
    assert resumed.successful_optimizer_steps == resumed.scheduler_steps == 4
    assert resumed.loss_history == continuous.loss_history
    assert resumed.exposures_by_source == continuous.exposures_by_source
    assert all(torch.equal(v, continuous.model.state_dict()[k]) for k, v in resumed.model.state_dict().items())
    for result in (resumed, continuous):
        state = load_training_checkpoint(result.final_checkpoint_path, expected_config=config)
        assert all(float(item["step"]) == 4 for item in state["optimizer_state_dict"]["state"].values())
    assert load_training_checkpoint(checkpoint, expected_config=config)["progress"]["successful_optimizer_steps"] == 2


def config_file(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(dict(run_id="fixed", config={}, scientific_configuration_fingerprint="fixed",
                                    semantic_fingerprint="fixed", identity={}, fold=0)))
    return source, hashlib.sha256(source.read_bytes()).hexdigest()


def test_handoff_never_overwrites_another_configuration(tmp_path):
    source, sha = config_file(tmp_path)
    destination = tmp_path / "destination.json"
    destination.write_text("preserve")
    with pytest.raises(ValueError, match="Refusing"):
        handoff.carry_configuration(source, destination, expected_sha256=sha)
    assert destination.read_text() == "preserve"


def test_changed_source_never_creates_recovery_configuration(tmp_path):
    source, _ = config_file(tmp_path)
    destination = tmp_path / "destination.json"
    with pytest.raises(ValueError, match="bytes changed"):
        handoff.carry_configuration(source, destination, expected_sha256="0" * 64)
    assert not destination.exists()
