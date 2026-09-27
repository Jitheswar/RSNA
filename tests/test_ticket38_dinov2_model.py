from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rsna_knee.candidate import CandidateError, _audit_requirements
from rsna_knee.dinov2 import (
    DINOV2_ENCODER_ID,
    DINOV2_FEATURE_DIM,
    DINOV2_MODEL_ID,
    DINOV2_POOLING_ID,
    DINOV2_REPOSITORY,
    DINOV2_RESOURCE_SCOPE,
    DINOV2_SOURCE_REVISION,
    DINOV2_WEIGHTS_SHA256,
    DINOV2_APPROVAL_SCHEMA,
    Dinov2ResourceError,
    load_dinov2_resource_approval,
)
from rsna_knee.feature_cache import FeatureCacheError, FrozenFeatureCache
from rsna_knee.image_variant import TICKET38_DINOV2_VARIANT, image_model_training_config
from rsna_knee.model import ModelConfig, ModelConfigurationError, build_study_model, restore_study_model
from rsna_knee.oof import OOFRunRecord
from rsna_knee.ticket38 import (
    Ticket38AnalysisError,
    _cache_identity,
    _require_dinov2_records,
    _verify_ticket32_baseline,
)
from rsna_knee.training import _model_input_study_uids


def _config() -> ModelConfig:
    return ModelConfig(
        image_size=(224, 224),
        channels=(64, 128, 256),
        model_id=DINOV2_MODEL_ID,
        pooling_id=DINOV2_POOLING_ID,
        encoder_id=DINOV2_ENCODER_ID,
        encoder_weights_sha256=DINOV2_WEIGHTS_SHA256,
        encoder_source_revision=DINOV2_SOURCE_REVISION,
    )


def test_dinov2_config_reconstructs_a_frozen_study_predictor() -> None:
    config = _config()

    model = build_study_model(ModelConfig.from_dict(config.to_dict()))
    images = torch.ones((1, 1, 1, 1, 224, 224), dtype=torch.float32)
    slice_mask = torch.ones((1, 1, 1), dtype=torch.bool)
    series_mask = torch.ones((1, 1), dtype=torch.bool)
    descriptors = torch.zeros((1, 1, 10), dtype=torch.float32)
    descriptors[0, 0, 0] = 1.0

    model.train()
    scores = model.predict_proba(images, slice_mask, series_mask, descriptors)
    reloaded = restore_study_model(config, model.state_dict())

    assert scores.shape == (1, 12)
    assert torch.isfinite(scores).all()
    assert not model.encoder.training
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert np.allclose(
        reloaded.predict_proba(images, slice_mask, series_mask, descriptors).detach().numpy(),
        scores.detach().numpy(),
        atol=1e-6,
    )


def test_dinov2_config_rejects_unpinned_encoder_revision() -> None:
    payload = _config().to_dict()
    payload["encoder_source_revision"] = "main"

    with pytest.raises(ModelConfigurationError):
        ModelConfig.from_dict(payload)


def test_ticket38_training_variant_freezes_the_224_pixel_anatomical_policy() -> None:
    config = image_model_training_config(TICKET38_DINOV2_VARIANT)

    assert config.model_config() == _config()
    assert config.preprocess_config().image_size == (224, 224)
    assert config.preprocess_config().representation_id == "ticket38-dinov2-224-whole-knee-v1"
    assert config.packing_config().max_series == 16
    assert config.packing_config().max_slices == 8
    assert config.device == "cpu"


def test_ticket38_limits_archive_study_selection_to_gold_reference_uids() -> None:
    gold = ("gold-b", "gold-a")

    assert _model_input_study_uids(DINOV2_MODEL_ID, gold) == ("gold-a", "gold-b")
    assert _model_input_study_uids("mechanics-baseline", gold) is None
    with pytest.raises(ValueError, match="unique"):
        _model_input_study_uids(DINOV2_MODEL_ID, ("gold-a", "gold-a"))


def test_dinov2_approval_rejects_a_changed_model_revision(tmp_path: Path) -> None:
    approval = {
        "schema_version": DINOV2_APPROVAL_SCHEMA,
        "scope": DINOV2_RESOURCE_SCOPE,
        "repository": DINOV2_REPOSITORY,
        "revision": "main",
    }
    changed_path = tmp_path / "approval.json"
    changed_path.write_text(json.dumps(approval), encoding="utf-8")

    with pytest.raises(Dinov2ResourceError, match="revision"):
        load_dinov2_resource_approval(changed_path)


def test_dinov2_cache_pins_the_384_wide_representation_and_rejects_other_widths(
    tmp_path: Path,
) -> None:
    identity = {
        "encoder_id": DINOV2_ENCODER_ID,
        "revision": DINOV2_SOURCE_REVISION,
        "weights_sha256": DINOV2_WEIGHTS_SHA256,
        "representation": "dinov2-vits14-class-token-384-float32-v1",
    }
    images = np.zeros((1, 2, 1, 224, 224), dtype=np.float32)
    slice_mask = np.ones((1, 2), dtype=bool)
    series_mask = np.ones((1,), dtype=bool)
    descriptors = np.zeros((1, 10), dtype=np.float32)
    cache = FrozenFeatureCache(tmp_path / "features", identity, feature_width=DINOV2_FEATURE_DIM)
    features, was_cached = cache.get_or_compute(
        "study-1",
        images,
        slice_mask,
        series_mask,
        descriptors,
        lambda: np.ones((1, 2, DINOV2_FEATURE_DIM), dtype=np.float32),
    )

    assert features.shape == (1, 2, DINOV2_FEATURE_DIM)
    assert not was_cached
    reader = FrozenFeatureCache(
        tmp_path / "features", identity, feature_width=DINOV2_FEATURE_DIM, read_only=True
    )
    cached, was_cached = reader.get_or_compute(
        "study-1", images, slice_mask, series_mask, descriptors, lambda: pytest.fail("recomputed")
    )
    assert was_cached
    assert np.array_equal(cached, features)

    with pytest.raises(FeatureCacheError, match="invalid study feature dimensions"):
        cache.get_or_compute(
            "study-2",
            images,
            slice_mask,
            series_mask,
            descriptors,
            lambda: np.ones((1, 2, 512), dtype=np.float32),
        )


def test_dinov2_candidate_dependencies_require_the_exact_offline_pins(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    cluster = root / "environments/cluster-cpu-ticket38/requirements.lock"
    offline = root / "environments/offline-dinov2/requirements.lock"
    audit = _audit_requirements(cluster, offline, require_dinov2=True)

    assert audit["dinov2_dependencies_required"] is True
    assert audit["runtime_network_download"] is False
    changed = tmp_path / "offline.lock"
    changed.write_text(
        offline.read_text(encoding="utf-8").replace("transformers==4.53.3", "transformers==4.53.4"),
        encoding="utf-8",
    )
    with pytest.raises(CandidateError, match="transformers==4.53.3"):
        _audit_requirements(cluster, changed, require_dinov2=True)


def test_ticket38_analysis_reads_private_cache_identity_from_validated_configuration(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "fold-0"
    run_dir.mkdir()
    run_path = run_dir / "run.json"
    run_path.write_text("{}", encoding="utf-8")
    expected_cache_identity = {
        "encoder_source_revision": DINOV2_SOURCE_REVISION,
        "encoder_weights_sha256": DINOV2_WEIGHTS_SHA256,
    }
    private_identity = {
        "source_version": "rsna-2026-08-04",
        "fold_manifest_checksum": "f" * 64,
        "initialization": {
            "source_revision": DINOV2_SOURCE_REVISION,
            "sha256": DINOV2_WEIGHTS_SHA256,
        },
        "frozen_feature_cache": {"identity": expected_cache_identity},
    }
    (run_dir / "configuration.json").write_text(
        json.dumps({"run_id": "run-0", "identity": private_identity}),
        encoding="utf-8",
    )
    record = OOFRunRecord(
        path=run_path,
        fold=0,
        run_id="run-0",
        config_fingerprint="config-0",
        predictions={},
        identity={"source_version": "rsna-2026-08-04"},
    )

    _require_dinov2_records((record,))
    assert _cache_identity(record) == expected_cache_identity


def test_ticket38_analysis_rejects_a_changed_ticket32_comparator(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    plan_path = project_root / "artifacts/private/ticket38-dinov2/ticket38-run-plan.json"
    baseline_root = project_root / "artifacts/private/ticket32-resnet18"
    plan_path.parent.mkdir(parents=True)
    entries = []
    for fold in range(3):
        run_path = baseline_root / f"fold-{fold}/run.json"
        run_path.parent.mkdir(parents=True)
        payload = {"run_id": f"ticket32-fold-{fold}"}
        run_path.write_text(json.dumps(payload), encoding="utf-8")
        entries.append(
            {
                "fold": fold,
                "run_id": payload["run_id"],
                "run_path": run_path.relative_to(project_root).as_posix(),
                "run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
            }
        )
    plan = {"baseline": {"ticket32_resnet18_runs": entries}}
    canonical = json.dumps(
        plan, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    plan["plan_sha256"] = hashlib.sha256(canonical).hexdigest()
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    assert len(_verify_ticket32_baseline(plan_path, baseline_root)) == 3
    changed_path = baseline_root / "fold-1/run.json"
    changed_path.write_text(json.dumps({"run_id": "ticket32-fold-1", "changed": True}), encoding="utf-8")
    with pytest.raises(Ticket38AnalysisError, match="digest"):
        _verify_ticket32_baseline(plan_path, baseline_root)
