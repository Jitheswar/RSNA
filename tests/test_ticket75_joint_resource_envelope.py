import importlib.util
from pathlib import Path
import sys
from datetime import datetime, timezone
from copy import deepcopy

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('ticket75_joint', ROOT / 'scripts/prepare_ticket75_joint_resource_envelope.py')
route = importlib.util.module_from_spec(spec)
spec.loader.exec_module(route)
sys.path.pop(0)


def parent():
    return route.read(ROOT / '.scratch/rsna-leaderboard/resource-approvals/ticket73-configured-continuation-20261002.json')


def test_whole_remaining_route_and_actual_files_retained():
    previous = parent(); original = deepcopy(previous)
    result = route.projection(previous, completed_control_fits_verified=True, existing=1_376_000_381_317,
        free=22_609_842_913_280, now=datetime(2026, 10, 2, 22, tzinfo=timezone.utc))
    assert previous == original
    assert result['completed_files_deleted'] is False
    assert result['completed_control_future_allowance_credit_bytes'] == 230_566_485_483
    assert result['storage']['protected_allowance_bytes'] >= 4 * 76_855_495_161
    assert result['storage']['projected_apparent_bytes'] < 2_000_000_000_000
    for key, value in previous['remaining_calendar'].items():
        assert result['remaining_calendar'][key] == value
    assert sum(result['remaining_calendar'].values()) == 561600
    assert result['remaining_calendar']['additional_T74_fixed_analysis_seconds'] == 7200


@pytest.mark.parametrize('defect', ['unverified', 'changed_envelope', 'late', 'full_storage', 'insufficient_free'])
def test_missing_completion_or_failed_whole_route_cannot_admit(defect):
    previous = parent()
    args = dict(completed_control_fits_verified=True, existing=1_376_000_381_317,
        free=22_609_842_913_280, now=datetime(2026, 10, 2, 22, tzinfo=timezone.utc))
    if defect == 'unverified': args['completed_control_fits_verified'] = False
    if defect == 'changed_envelope': previous['projection']['checkpoint_storage_allowance_bytes'] += 1
    if defect == 'late': args['now'] = datetime(2026, 10, 10, tzinfo=timezone.utc)
    if defect == 'full_storage': args['existing'] = 1_999_000_000_000
    if defect == 'insufficient_free': args['free'] = 8_000_000_000
    with pytest.raises(ValueError):
        route.projection(previous, **args)
