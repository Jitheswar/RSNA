"""Operational recovery must not copy live state or repeat an inherited budget."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import ticket73_configured_full_continuation as fit
import coordinate_ticket73_configured_full_continuation as coordinator
import validate_ticket73_configured_full_comparison as replay


def live_fixture(tmp_path, monkeypatch):
    old, new = tmp_path / "v3", tmp_path / "v4"
    stage = old / "full-backbone/fold-1/stage-3"
    stage.mkdir(parents=True)
    new.mkdir()
    fit.write(stage / "accepted-launch.json", dict(job_id=8113, plan_sha256=fit.V3_SHA, launch_attempt=1))
    monkeypatch.setattr(fit, "V3", old)
    monkeypatch.setattr(fit, "ROOT", new)
    monkeypatch.setattr(fit, "admission", lambda checksum: {})
    return old, new, stage


def test_preserve_all_updates_and_live_stage_identity():
    rows = fit.remaining_stages()
    assert len(rows) == 15
    assert sum(r['target_steps'] - r['prior_steps'] for r in rows) == 13500
    assert [sum(r['target_steps'] - r['prior_steps'] for r in rows if r['fold'] == f) for f in range(3)] == [3500, 3500, 6500]
    assert fit.stage_location(1, 3, 'new') == (fit.V3 / 'full-backbone/fold-1/stage-3', fit.V3_SHA)
    assert fit.stage_location(1, 4, 'new') == (fit.ROOT / 'full-backbone/fold-1/stage-4', 'new')
    assert fit.stage_location(0, 3, 'new')[0] == fit.PRIOR / 'full-backbone/fold-0/stage-3'
    assert replay.final_checkpoint_path('full-backbone', 1) == replay.FIT / 'full-backbone/fold-1/training/checkpoints/final.pt'


def test_live_running_checkpoint_never_copied(tmp_path, monkeypatch):
    _, new, _ = live_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(fit.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='JobId=8113 JobState=RUNNING', stderr=''))
    assert fit.close_live_input('new') is False
    assert not list(new.iterdir())


@pytest.mark.parametrize('state', ['FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY'])
def test_failed_live_job_never_inherited_or_restarted(tmp_path, monkeypatch, state):
    _, new, _ = live_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(fit.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout=f'JobId=8113 JobState={state}', stderr=''))
    with pytest.raises(ValueError, match='failed'):
        fit.close_live_input('new')
    assert not list(new.iterdir())


def test_purged_live_without_terminal_not_inherited(tmp_path, monkeypatch):
    live_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(fit.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1, stdout='', stderr='Invalid job id specified'))
    with pytest.raises(ValueError, match='purged'):
        fit.close_live_input('new')


def completed_fixture(tmp_path, monkeypatch):
    old, new, stage = live_fixture(tmp_path, monkeypatch)
    controller = stage / 'controller-8113'
    controller.mkdir()
    (controller / 'terminal-controller.txt').write_text('retained exact backend terminal')
    fit.write(controller / 'terminal-observation.json', dict(state='COMPLETED'))
    checkpoint = old / 'complete.pt'
    checkpoint.write_bytes(b'4000-updates-model-Adam-RNG-state')
    record = dict(job_id='8113', successful_steps=4000, checkpoint_path=str(checkpoint), checkpoint_sha256=fit.digest(checkpoint), prior_checkpoint_sha256='3000')
    fit.write(stage / 'stage-result.json', record)
    monkeypatch.setattr(fit, 'successful_terminal', lambda *a: (7200, controller / 'terminal-controller.txt'))
    monkeypatch.setattr(fit, 'stage_record', lambda f, s, sha: (record if s == 3 else dict(checkpoint_sha256='3000'), 'actual_successful_terminal'))
    return new, checkpoint, record


def test_completed_live_seed_closed_once_and_tampering_rejected(tmp_path, monkeypatch):
    new, checkpoint, _ = completed_fixture(tmp_path, monkeypatch)
    assert fit.close_live_input('new') is True
    seed = new / 'fold-1-recovery-seed.pt'
    assert seed.read_bytes() == checkpoint.read_bytes()
    closure = (new / 'live-fold-1-input.json').read_bytes()
    assert fit.close_live_input('new') is True
    assert (new / 'live-fold-1-input.json').read_bytes() == closure
    seed.write_bytes(b'changed')
    with pytest.raises(ValueError, match='closure changed'):
        fit.close_live_input('new')


def test_completed_worker_with_failed_terminal_never_copied(tmp_path, monkeypatch):
    new, _, _ = completed_fixture(tmp_path, monkeypatch)
    def fail(*a):
        raise ValueError('Exact successful actual CPU terminal required')
    monkeypatch.setattr(fit, 'successful_terminal', fail)
    with pytest.raises(ValueError, match='successful actual'):
        fit.close_live_input('new')
    assert not list(new.iterdir())


def test_wrong_predecessor_never_inherited(tmp_path, monkeypatch):
    new, _, record = completed_fixture(tmp_path, monkeypatch)
    record['prior_checkpoint_sha256'] = 'another-ancestor'
    with pytest.raises(ValueError, match='predecessor'):
        fit.close_live_input('new')
    assert not list(new.iterdir())


def test_pending_live_fold_does_not_start_its_successor():
    states = [['unlaunched'] * 4, ['active'] + ['unlaunched'] * 3, ['unlaunched'] * 7]
    assert coordinator.next_unused_stage(states, 1) == (0, 4)
    assert coordinator.next_unused_stage(states, 2) is None
    states[0] = ['complete'] * 4
    assert coordinator.next_unused_stage(states, 1) == (2, 1)


def test_first_fold1_successor_requires_closed_complete_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(fit, 'ROOT', tmp_path)
    monkeypatch.setattr(fit, 'stage_record', lambda *a: (dict(successful_steps=4000, checkpoint_sha256='fixed'), 'actual_successful_terminal'))
    with pytest.raises(ValueError, match='closure required'):
        fit.previous_stage(1, 4, 'new')
