from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from rsna_knee.evaluation import predict_training_examples
from rsna_knee.gpu import (
    GPU_APPROVAL_SCHEMA_VERSION,
    GPU_APPROVAL_SCOPE,
    GpuResourceRequest,
    load_gpu_approval,
    load_gpu_experiment_request,
    prepare_gpu_experiment,
)
from rsna_knee.model import ModelConfig, build_study_model, restore_study_model
from rsna_knee.preprocessing import SERIES_DESCRIPTOR_COLUMNS
from rsna_knee.resnet18 import (
    ANATOMICAL_POOLING_ID,
    RESNET18_ENCODER_ID,
    RESNET18_FINETUNE_MODEL_ID,
    RESNET18_FIXTURE_ENCODER_ID,
)
from rsna_knee.training import (
    PREFLIGHT_SCHEMA_VERSION,
    TrainingConfig,
    TrainingConfigurationError,
    TrainingExample,
    load_training_checkpoint,
    train_fixed_budget,
)
from rsna_knee.pretrained import (
    RESNET18_APPROVAL_SCHEMA,
    RESNET18_ARTIFACT_BYTES,
    RESNET18_ARTIFACT_ID,
    RESNET18_DOWNLOAD_URL,
    RESNET18_SHA256,
    RESNET18_SOURCE_REVISION,
    TICKET39_RESOURCE_SCOPE,
)


def _model_config() -> ModelConfig:
    return ModelConfig(
        image_size=(32, 32),
        model_id=RESNET18_FINETUNE_MODEL_ID,
        pooling_id=ANATOMICAL_POOLING_ID,
        encoder_id=RESNET18_FIXTURE_ENCODER_ID,
    )


def test_limited_resnet_finetuning_updates_only_layer4_and_head_and_reloads() -> None:
    torch.manual_seed(7)
    config = _model_config()
    model = build_study_model(config)
    model.train()

    images = torch.rand((1, 1, 2, 1, 32, 32))
    slice_mask = torch.tensor([[[True, True]]])
    series_mask = torch.tensor([[True]])
    descriptors = torch.zeros((1, 1, len(SERIES_DESCRIPTOR_COLUMNS)))
    descriptors[..., 0] = 1.0
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    logits = model(images, slice_mask, series_mask, descriptors)
    assert logits.shape == (1, 12)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()

    assert all(
        parameter.grad is None
        for name, parameter in model.encoder.named_parameters()
        if not name.startswith("layer4.")
    )
    assert any(
        parameter.grad is not None
        for parameter in model.encoder.layer4.parameters()
    )
    optimizer.step()

    assert all(
        torch.equal(before[f"encoder.{name}"], value)
        for name, value in model.encoder.named_parameters()
        if not name.startswith("layer4.")
    )
    assert any(
        not torch.equal(before[f"encoder.layer4.{name}"], value)
        for name, value in model.encoder.layer4.named_parameters()
    )

    restored = restore_study_model(config.to_dict(), model.state_dict())
    restored.eval()
    model.eval()
    with torch.inference_mode():
        expected = model(images, slice_mask, series_mask, descriptors)
        actual = restored(images, slice_mask, series_mask, descriptors)
    torch.testing.assert_close(actual, expected)

    example = TrainingExample(
        study_instance_uid="synthetic-ticket39-study",
        images=images[0].numpy(),
        slice_mask=slice_mask[0].numpy(),
        series_mask=series_mask[0].numpy(),
        targets=np.zeros(12, dtype=np.float32),
        supervision_mask=np.ones(12, dtype=bool),
        series_descriptors=descriptors[0].numpy(),
    )
    offline_scores = predict_training_examples(restored, (example,))[example.study_instance_uid]
    np.testing.assert_allclose(offline_scores, torch.sigmoid(expected[0]).numpy(), atol=1e-6)


def test_ticket39_saved_checkpoint_replays_offline_and_rejects_mismatched_config(
    tmp_path,
) -> None:
    config = TrainingConfig(
        image_size=(224, 224),
        max_slices=1,
        max_series=1,
        model_id=RESNET18_FINETUNE_MODEL_ID,
        pooling_id=ANATOMICAL_POOLING_ID,
        encoder_id=RESNET18_FIXTURE_ENCODER_ID,
        encoder_learning_rate=1e-5,
        successful_optimizer_steps=1,
        checkpoint_interval_steps=1,
        seed=3901,
    )
    descriptors = np.zeros((1, len(SERIES_DESCRIPTOR_COLUMNS)), dtype=np.float32)
    descriptors[0, 0] = 1.0

    def example(uid: str, positive_target: int | None) -> TrainingExample:
        targets = np.zeros(12, dtype=np.float32)
        if positive_target is not None:
            targets[positive_target] = 1.0
        return TrainingExample(
            study_instance_uid=uid,
            images=np.full((1, 1, 1, 224, 224), 0.4, dtype=np.float32),
            slice_mask=np.ones((1, 1), dtype=bool),
            series_mask=np.ones(1, dtype=bool),
            targets=targets,
            supervision_mask=np.ones(12, dtype=bool),
            series_descriptors=descriptors.copy(),
        )

    examples = (
        example("ticket39-train-negative", None),
        example("ticket39-train-positive", 0),
    )
    previous_thread_count = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = train_fixed_budget(
            examples,
            config,
            model=build_study_model(config.model_config()),
            artifact_dir=tmp_path / "ticket39-checkpoint",
            fold=0,
        )
        assert result.final_checkpoint_path is not None
        restored = build_study_model(config.model_config())
        checkpoint = load_training_checkpoint(
            result.final_checkpoint_path,
            model=restored,
            expected_config=config,
            expected_fold=0,
            expected_checkpoint_kind="final_selected",
        )
        assert checkpoint["checkpoint_kind"] == "final_selected"
        optimizer_state = checkpoint.get("optimizer_state_dict")
        assert isinstance(optimizer_state, dict)
        assert [group["lr"] for group in optimizer_state["param_groups"]] == [1e-5, 1e-3]

        expected = predict_training_examples(result.model, examples)
        actual = predict_training_examples(restored, examples)
        for uid in expected:
            np.testing.assert_allclose(actual[uid], expected[uid], rtol=0.0, atol=1e-7)

        mismatched_config = replace(config, encoder_learning_rate=2e-5)
        with pytest.raises(TrainingConfigurationError, match="does not match"):
            load_training_checkpoint(
                result.final_checkpoint_path,
                expected_config=mismatched_config,
                expected_fold=0,
                expected_checkpoint_kind="final_selected",
            )
    finally:
        torch.set_num_threads(previous_thread_count)


def test_ticket39_gpu_request_binds_the_limited_trainable_stage_and_exact_resource(
    tmp_path,
) -> None:
    preflight_path = tmp_path / "cpu-preflight.json"
    checks = {
        "python_311": True,
        "user_environment_python": True,
        "cluster1_cpu_slurm": True,
        "core_dependencies_import": True,
        "known_roc_auc": True,
        "finite_model_loss_and_gradients": True,
        "checkpoint_save_reload": True,
        "inference_import_isolated": True,
    }
    preflight_path.write_text(
        json.dumps(
            {
                "schema_version": PREFLIGHT_SCHEMA_VERSION,
                "passed": True,
                "checks": checks,
                "environment": {"python_version": "3.11.16"},
                "details": {"required_slurm": True, "required_isolated_python": True},
            }
        ),
        encoding="utf-8",
    )
    resource_approval_path = tmp_path / "ticket39-resource-approval.json"
    resource_approval_path.write_text(
        json.dumps(
            {
                "schema_version": RESNET18_APPROVAL_SCHEMA,
                "artifact_id": RESNET18_ARTIFACT_ID,
                "source_revision": RESNET18_SOURCE_REVISION,
                "download_url": RESNET18_DOWNLOAD_URL,
                "expected_bytes": RESNET18_ARTIFACT_BYTES,
                "sha256": RESNET18_SHA256,
                "scope": TICKET39_RESOURCE_SCOPE,
                "decision": "approved",
                "rights_status": "eligible",
                "rights_status_scope": TICKET39_RESOURCE_SCOPE,
                "encoder_frozen": False,
                "trainable_encoder_portion": "layer4",
                "external_spend": 0,
                "approved_by": "project owner",
                "approved_at": "2026-09-27T00:00:00Z",
                "rights_basis": "Synthetic test fixture.",
            }
        ),
        encoding="utf-8",
    )
    config = TrainingConfig(
        image_size=(224, 224),
        model_id=RESNET18_FINETUNE_MODEL_ID,
        pooling_id=ANATOMICAL_POOLING_ID,
        encoder_id=RESNET18_ENCODER_ID,
        encoder_weights_sha256=RESNET18_SHA256,
        encoder_learning_rate=1e-5,
        device="cuda",
        successful_optimizer_steps=2,
    )
    execution = {
        "pretrained_resource": {
            "artifact_id": RESNET18_ARTIFACT_ID,
            "source_revision": RESNET18_SOURCE_REVISION,
            "download_url": RESNET18_DOWNLOAD_URL,
            "sha256": RESNET18_SHA256,
            "expected_bytes": RESNET18_ARTIFACT_BYTES,
            "weights_path": str((tmp_path / "weights.pth").resolve()),
            "resource_approval_path": str(resource_approval_path.resolve()),
            "scope": TICKET39_RESOURCE_SCOPE,
            "decision": "approved",
            "rights_status": "eligible",
            "rights_status_scope": TICKET39_RESOURCE_SCOPE,
            "encoder_frozen": False,
            "trainable_encoder_portion": "layer4",
            "runtime_network": False,
            "external_spend": 0,
        }
    }
    request = prepare_gpu_experiment(
        tmp_path / "request",
        preflight_path=preflight_path,
        resource=GpuResourceRequest(
            partition="cluster1-gpu-add",
            wall_time_minutes=30,
            smoke_test=True,
        ),
        training_config=config,
        experiment_id="ticket39-layer4-smoke",
        fold=0,
        source_version="fixture-source-v1",
        source_fingerprint="fixture-source-sha256",
        fold_manifest_checksum="fixture-fold-sha256",
        integration_manifest_checksum="fixture-integration-sha256",
        retry_allowed=False,
        execution=execution,
    )
    loaded = load_gpu_experiment_request(request.root)
    assert loaded.training_config.to_dict() == config.to_dict()

    owner_approval_path = tmp_path / "gpu-approval.json"
    owner_approval_path.write_text(
        json.dumps(
            {
                "schema_version": GPU_APPROVAL_SCHEMA_VERSION,
                "request_checksum": request.request_checksum,
                "approval_scope": GPU_APPROVAL_SCOPE,
                "approved": True,
                "retry_allowed": False,
                "approver": "project owner",
                "approved_at_utc": "2026-09-27T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    approved = load_gpu_approval(owner_approval_path, request)
    assert approved.payload["request_checksum"] == request.request_checksum
