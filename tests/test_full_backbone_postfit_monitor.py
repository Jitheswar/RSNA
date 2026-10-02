"""Postfit automation waits for complete evidence and never repeats submission."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def monitor(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('monitor_ticket64_final_verification')


def test_postfit_cannot_prepare_from_incomplete_fits(monitor):
    assert monitor.action_for_state(fits_complete=False, occupied_slots=0, prepared=False,
        intent=False, accepted=False, terminal=False) == 'wait_for_fits'


def test_postfit_preserves_two_cpu_slots(monitor):
    assert monitor.action_for_state(fits_complete=True, occupied_slots=2, prepared=False,
        intent=False, accepted=False, terminal=False) == 'wait_for_cpu_slot'


def test_postfit_advances_only_unconsumed_requests(monitor):
    assert monitor.action_for_state(fits_complete=True, occupied_slots=0, prepared=False,
        intent=False, accepted=False, terminal=False) == 'prepare'
    assert monitor.action_for_state(fits_complete=True, occupied_slots=0, prepared=True,
        intent=False, accepted=False, terminal=False) == 'launch'
    with pytest.raises(ValueError, match='never retry'):
        monitor.action_for_state(fits_complete=True, occupied_slots=0, prepared=True,
            intent=True, accepted=False, terminal=False)


def test_accepted_handle_is_watched_even_if_queue_observation_is_empty(monitor):
    assert monitor.action_for_state(fits_complete=True, occupied_slots=0, prepared=True,
        intent=True, accepted=True, terminal=False) == 'watch'
    assert monitor.action_for_state(fits_complete=True, occupied_slots=0, prepared=True,
        intent=True, accepted=True, terminal=True) == 'reconcile'


def test_last_incomplete_stage_prevents_readiness(monitor, monkeypatch, tmp_path):
    monkeypatch.setattr(monitor.replay, 'FIT', tmp_path)
    monkeypatch.setattr(monitor.replay, 'OLD', tmp_path / 'old')
    monkeypatch.setattr(monitor, 'stage_state', lambda path, *args: 'active' if str(path).endswith('fold-2/stage-7') else 'complete')
    assert not monitor.all_fits_complete()


def test_active_submission_intent_is_not_a_terminal_failure(monitor, monkeypatch, tmp_path):
    monkeypatch.setattr(monitor.replay, 'FIT', tmp_path)
    monkeypatch.setattr(monitor.replay, 'OLD', tmp_path / 'old')
    directory = tmp_path / 'full-backbone/fold-2/stage-7'; directory.mkdir(parents=True)
    (directory / 'launch-intent.json').write_text('{}')
    monkeypatch.setattr(monitor, 'stage_state', lambda *args: 'complete')
    assert not monitor.all_fits_complete()


def test_failed_fit_terminal_stops_postfit_automation(monitor, monkeypatch):
    def failed(*args):
        raise ValueError('Stage lacks exact successful CPU terminal')
    monkeypatch.setattr(monitor, 'stage_state', failed)
    with pytest.raises(ValueError, match='successful CPU terminal'):
        monitor.all_fits_complete()
