"""Recovery must preserve the inherited job and prevent a duplicate stage 0."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def coordinator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('coordinate_ticket64_recovery_v2')


def test_first_full_stage_uses_original_job_directory_and_identity(coordinator):
    path, sha = coordinator.stage_location(0, 0)
    assert path == coordinator.OLD / 'full-backbone/fold-0/stage-0'
    assert sha == coordinator.SCIENCE_SHA
    path, sha = coordinator.stage_location(0, 1)
    assert path == coordinator.ROOT / 'full-backbone/fold-0/stage-1'
    assert sha == coordinator.CHECKSUM
    assert sha != coordinator.SCIENCE_SHA


def test_recovery_fills_free_slot_without_relaunching_live_first_stage(coordinator):
    states = [['active'] + ['unlaunched']*7] + [['unlaunched']*8 for _ in range(2)]
    assert coordinator.next_stage('complete', states, 1) == (1, 0)
    assert coordinator.next_stage('complete', states, 2) is None
    states[0][0] = 'complete'
    assert coordinator.next_stage('complete', states, 1) == (0, 1)


def test_recovery_waits_for_exact_control_success(coordinator):
    states = [['complete'] + ['unlaunched']*7] + [['unlaunched']*8 for _ in range(2)]
    assert coordinator.next_stage('active', states, 0) is None
