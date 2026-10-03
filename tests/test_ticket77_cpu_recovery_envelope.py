"""Whole-route admission rejects lost reserves, late finish and storage overflow."""
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import importlib.util
from pathlib import Path
import sys

import pytest

PROJECT = Path(__file__).parents[1]
sys.path.insert(0,str(PROJECT/'scripts'))
spec = importlib.util.spec_from_file_location('ticket77_recovery',PROJECT/'scripts/prepare_ticket77_cpu_reference_recovery.py')
recovery = importlib.util.module_from_spec(spec);spec.loader.exec_module(recovery)


def envelope():
    return deepcopy(recovery.read(recovery.PREVIOUS)['envelope'])


def test_new_reserve_keeps_every_existing_calendar_and_storage_bound():
    old = envelope();result = deepcopy(old)
    now = datetime(2026,10,3,2,tzinfo=timezone.utc)
    recovery.add_cache_reserve(result,now,22_000_000_000_000)
    for key,value in old['remaining_calendar'].items(): assert result['remaining_calendar'][key] == value
    assert sum(result['remaining_calendar'].values()) == 565200
    assert result['complete_route_finish_utc'] == (now+timedelta(seconds=565200)).isoformat()
    assert result['storage']['projected_apparent_bytes'] == old['storage']['projected_apparent_bytes']+100_000_000
    assert result['storage']['protected_allowance_bytes'] == old['storage']['protected_allowance_bytes']
    assert result['completed_control_future_allowance_credit_bytes'] == old['completed_control_future_allowance_credit_bytes']
    assert result['completed_files_deleted'] is False


@pytest.mark.parametrize('defect',['late','storage','free','released_root','released_pilot','double_reserve'])
def test_whole_route_failure_prevents_admission(defect):
    result = envelope();now = datetime(2026,10,3,2,tzinfo=timezone.utc);free = 22_000_000_000_000
    if defect == 'late': now = datetime(2026,10,10,tzinfo=timezone.utc)
    if defect == 'storage': result['storage']['projected_apparent_bytes'] = 1_999_999_999_999
    if defect == 'free': free = result['storage']['protected_allowance_bytes']+8_300_000_000
    if defect == 'released_root': result['remaining_calendar']['additional_root_recovery_cpu_seconds'] = 0
    if defect == 'released_pilot': result['remaining_calendar']['additional_complete_pilot_route_seconds'] = 0
    if defect == 'double_reserve': result['remaining_calendar']['additional_cache_recovery_cpu_seconds'] = 1800
    with pytest.raises(ValueError): recovery.add_cache_reserve(result,now,free)
