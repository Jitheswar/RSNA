import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPT_ROOT = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPT_ROOT))
spec = importlib.util.spec_from_file_location('ticket77_life', SCRIPT_ROOT / 'ticket77_cpu_reference_lifecycle.py')
life = importlib.util.module_from_spec(spec)
spec.loader.exec_module(life)
sys.path.pop(0)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / 'run';root.mkdir()
    (tmp_path / 'artifacts/private').mkdir(parents=True)
    monkeypatch.setattr(life, 'ROOT', root)
    monkeypatch.setattr(life, 'PROJECT', tmp_path)
    monkeypatch.setattr(life, 'admission', lambda _: {})
    monkeypatch.setattr(life, 'live_waiter', lambda: None)
    return root


def test_full_cpu_slots_never_create_pending_third_job(fixture, monkeypatch):
    commands = []
    def queue(command, **kwargs):
        commands.append(command)
        assert command[0] == 'squeue'
        return 'one|cluster1-cpu\ntwo|cluster1-cpu\n'
    monkeypatch.setattr(life.subprocess, 'check_output', queue)
    assert life.launch('frozen') is False
    assert len(commands) == 1
    assert not (fixture / 'launch-intent.json').exists()


def test_consumed_intent_never_repeats_backend_call(fixture, monkeypatch):
    life.write(fixture / 'launch-intent.json', dict(plan_sha256='frozen', launch_attempt=1))
    monkeypatch.setattr(life.subprocess, 'check_output', lambda *a, **kw: pytest.fail('Consumed request must not call backend'))
    with pytest.raises(ValueError, match='Consumed'):
        life.launch('frozen')


def test_missing_live_joint_owner_cannot_consume_intent(fixture, monkeypatch):
    def stopped(): raise ValueError('Actual live amended resource owner required')
    monkeypatch.setattr(life, 'live_waiter', stopped)
    monkeypatch.setattr(life.subprocess, 'check_output', lambda *a, **kw: pytest.fail('Stopped owner must not call backend'))
    with pytest.raises(ValueError, match='owner'):
        life.launch('frozen')
    assert not (fixture / 'launch-intent.json').exists()
