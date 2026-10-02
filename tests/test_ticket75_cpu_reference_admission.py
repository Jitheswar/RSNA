"""Reject preparation without complete protected-route and immutable-source proof."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/ticket75_cpu_reference.py'
SPEC = importlib.util.spec_from_file_location('ticket75_cpu_reference', SCRIPT)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


@pytest.fixture
def pilot_request(tmp_path):
    selection = tmp_path / 'selection.json'
    selection.write_text('{}')
    proof = tmp_path / 'joint.json'
    proof.write_text(json.dumps(dict(status='ADMITTED', fit_plan_sha256=worker.FIT_SHA,
        T74_future_analysis_cpu_node_seconds=7200, pilot_complete_calendar_seconds=28800,
        pilot_retained_assets_bytes=8_000_000_000, T74_future_admission_accounts_outstanding_pilot=True)))
    return dict(schema_version=worker.SCHEMA, fit_plan_sha256=worker.FIT_SHA, selection_sha256=worker.SELECTION_SHA,
        selection_path=str(selection), resources=dict(partition='cluster1-cpu', nodes=1, tasks=1, cpus_per_task=16,
            memory_gib=64, walltime_seconds=1800, code_watchdog_seconds=1740, maximum_launch_attempts=1,
            exclusive=False, requeue=False, gpu_count=0, tpu_count=0, external_spend_usd=0),
        joint_route_ownership_closed=True, joint_route_proof_path=str(proof), joint_route_proof_sha256=worker.digest(proof),
        candidate_promotion=False, competition_submission=False, original_fit_updates_credited=False,
        accelerator_launch_admitted=False,
        frozen_files=[dict(path=str(p), sha256=worker.digest(p)) for p in (SCRIPT, selection, proof)])


def test_complete_contract_passes_metadata_gate_only(pilot_request):
    assert worker.validate_request(pilot_request)['walltime_seconds'] == 1800
    assert not pilot_request['accelerator_launch_admitted']


@pytest.mark.parametrize('defect', ['missing_T74', 'missing_pilot_calendar', 'missing_pilot_storage', 'future_omission'])
def test_rehashed_joint_proof_cannot_drop_protected_reserves(pilot_request, defect):
    path = Path(pilot_request['joint_route_proof_path'])
    proof = json.loads(path.read_text())
    key = dict(missing_T74='T74_future_analysis_cpu_node_seconds', missing_pilot_calendar='pilot_complete_calendar_seconds',
        missing_pilot_storage='pilot_retained_assets_bytes', future_omission='T74_future_admission_accounts_outstanding_pilot')[defect]
    proof[key] = False if defect == 'future_omission' else 0
    path.write_text(json.dumps(proof))
    pilot_request['joint_route_proof_sha256'] = worker.digest(path)
    next(r for r in pilot_request['frozen_files'] if r['path'] == str(path))['sha256'] = worker.digest(path)
    with pytest.raises(ValueError, match='complete pilot route'):
        worker.validate_request(pilot_request)


@pytest.mark.parametrize('key', ['candidate_promotion', 'competition_submission', 'original_fit_updates_credited', 'accelerator_launch_admitted'])
def test_disposable_worker_cannot_expand_scope(pilot_request, key):
    pilot_request[key] = True
    with pytest.raises(ValueError, match='Disposable'):
        worker.validate_request(pilot_request)


@pytest.mark.parametrize('change', [dict(gpu_count=1), dict(tpu_count=8), dict(maximum_launch_attempts=2),
    dict(walltime_seconds=3600), dict(exclusive=True)])
def test_resource_expansion_or_retry_rejected(pilot_request, change):
    pilot_request['resources'].update(change)
    with pytest.raises(ValueError, match='bounded'):
        worker.validate_request(pilot_request)


def test_mutated_metadata_rejected_before_loading_MRI(pilot_request):
    Path(pilot_request['selection_path']).write_text('changed')
    with pytest.raises(ValueError, match='immutable'):
        worker.validate_request(pilot_request)


def test_unbound_executable_rejected(pilot_request):
    pilot_request['frozen_files'] = [r for r in pilot_request['frozen_files'] if r['path'] != str(SCRIPT)]
    with pytest.raises(ValueError, match='Executable'):
        worker.validate_request(pilot_request)


def test_missing_joint_admission_rejected(pilot_request):
    pilot_request['joint_route_ownership_closed'] = False
    with pytest.raises(ValueError, match='ownership'):
        worker.validate_request(pilot_request)
