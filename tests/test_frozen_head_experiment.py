"""Check pooling equivalence, additive sampling and checkpoint integrity."""

import numpy as np
import pytest
import torch

from rsna_knee.feature_pooling import ANATOMICAL_POOLING_ID, pool_frozen_features
from rsna_knee.frozen_head_experiment import (
    FEATURE_WIDTH, build_head, pool_study, reload_head, save_head_checkpoint,
    sequence_digest, train_head, zero_image_features,
)
from rsna_knee.training import _batch_indices, masked_bce_loss


def inputs():
    rng = np.random.default_rng(81)
    features = rng.normal(size=(4, 3, FEATURE_WIDTH)).astype(np.float32)
    slices = np.ones((4, 3), dtype=bool)
    slices[2, 2] = False
    series = np.ones(4, dtype=bool)
    series[3] = False
    descriptors = np.zeros((4, 10), dtype=np.float32)
    descriptors[np.arange(4), np.arange(4)] = 1
    descriptors[:, 5] = 1
    descriptors[:, 9] = 1
    return features, slices, series, descriptors


def test_prepooled_head_updates_match_pooling_inside_each_step():
    torch.set_num_threads(1)
    args = inputs()
    pooled = pool_study(*args).unsqueeze(0).repeat(4, 1)
    targets = torch.tensor([[index % 2] * 12 for index in range(4)], dtype=torch.float32)
    mask = torch.ones_like(targets, dtype=torch.bool)
    mask[2, 3:] = False
    prepooled, counts = train_head(
        pooled=pooled, targets=targets, mask=mask, study_uids=["a", "b", "c", "d"],
        reference_indices=[0, 1], weak_indices=[2, 3], successful_steps=3, batch_size=2,
        initialization_seed=12, batch_order_seed=20, weak_seed_offset=34,
        learning_rate=0.001,
    )
    direct = build_head(12)
    optimizer = torch.optim.Adam(direct.parameters(), lr=0.001)
    for step in range(3):
        indices = list(_batch_indices(2, 2, step, 20))
        indices += [2 + index for index in _batch_indices(2, 2, step, 54)]
        direct_pool = pool_frozen_features(
            torch.from_numpy(args[0]).unsqueeze(0).repeat(4, 1, 1, 1),
            torch.from_numpy(args[1]).unsqueeze(0).repeat(4, 1, 1),
            torch.from_numpy(args[2]).unsqueeze(0).repeat(4, 1),
            torch.from_numpy(args[3]).unsqueeze(0).repeat(4, 1, 1),
            feature_width=FEATURE_WIDTH, pooling_id=ANATOMICAL_POOLING_ID,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = masked_bce_loss(direct(direct_pool), targets[indices], mask[indices])
        loss.backward()
        optimizer.step()
    for actual, expected in zip(prepooled.parameters(), direct.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert counts["distinct_weak_studies_exposed"] == 2


def test_image_ablation_retains_descriptors_and_plane_presence():
    args = inputs()
    full = pool_study(*args).unsqueeze(0)
    zero = pool_study(np.zeros_like(args[0]), *args[1:]).unsqueeze(0)
    torch.testing.assert_close(zero_image_features(full), zero, rtol=0, atol=0)
    assert bool((zero != 0).any())


def test_matched_pair_preserves_gold_initialization_and_order():
    pooled = pool_study(*inputs()).unsqueeze(0).repeat(5, 1)
    targets = torch.zeros((5, 12))
    mask = torch.ones_like(targets, dtype=torch.bool)
    options = dict(pooled=pooled, targets=targets, mask=mask,
                   study_uids=["a", "b", "c", "d", "e"], reference_indices=[0, 1],
                   successful_steps=5, batch_size=2, initialization_seed=18,
                   batch_order_seed=19, weak_seed_offset=20, learning_rate=0.001)
    _, gold = train_head(**options, weak_indices=[])
    _, reports = train_head(**options, weak_indices=[2, 3, 4])
    assert gold["initial_head_state_sha256"] == reports["initial_head_state_sha256"]
    assert gold["gold_exposure_sequence_sha256"] == reports["gold_exposure_sequence_sha256"]
    sequence = [options["study_uids"][index] for step in range(5)
                for index in _batch_indices(2, 2, step, 19)]
    assert gold["gold_exposure_sequence_sha256"] == sequence_digest(sequence)
    assert reports["distinct_weak_studies_exposed"] == 3
    assert reports["exposures_by_source"]["weak"]["sample_exposures"] == 10


def test_checkpoint_reload_reproduces_predictions_and_rejects_wrong_identity(tmp_path):
    model = build_head(4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    path = tmp_path / "final.pt"
    identity = {"plan_sha256": "f" * 64, "fold": 0, "arm": "reference-only"}
    save_head_checkpoint(path, model, optimizer, {"successful_optimizer_steps": 1500}, identity, final=True)
    reloaded, counts = reload_head(path, identity)
    pooled = pool_study(*inputs()).unsqueeze(0)
    torch.testing.assert_close(model(pooled), reloaded(pooled), rtol=0, atol=0)
    assert counts["successful_optimizer_steps"] == 1500
    with pytest.raises(ValueError, match="contract"):
        reload_head(path, {**identity, "fold": 1})
    with pytest.raises(FileExistsError):
        save_head_checkpoint(path, model, optimizer, counts, identity, final=True)


def test_overlapping_training_partitions_are_rejected():
    pooled = pool_study(*inputs()).unsqueeze(0)
    with pytest.raises(ValueError, match="overlap"):
        train_head(pooled=pooled, targets=torch.zeros((1, 12)), mask=torch.ones((1, 12), dtype=torch.bool),
                   study_uids=["a"], reference_indices=[0], weak_indices=[0], successful_steps=1,
                   batch_size=2, initialization_seed=18, batch_order_seed=19,
                   weak_seed_offset=20, learning_rate=0.001)
