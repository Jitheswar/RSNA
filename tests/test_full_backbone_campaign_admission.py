"""Guard expensive staged jobs using observed costs rather than desired speed."""

from datetime import datetime, timezone
import importlib.util
from pathlib import Path

import pytest


def load_campaign():
    import sys
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location("full_campaign",scripts / "ticket64_full_backbone_comparison.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))
    return module


def measurements(step=9.0,startup=373.0,peak=18_000_000_000):
    admission={"folds":[{"reference_studies":n} for n in (38,39,39)]}
    record=dict(step_and_prior_checkpoint_intervals_seconds=[step,step,step],
        pre_training_seconds=startup,checkpoint_bytes=135_100_000,peak_rss_bytes=peak,successful_steps=3)
    return admission,[dict(record) for _ in range(3)]


def test_actual_cost_selects_bounded_stages_and_reserves_epoch_history_and_final_delivery():
    campaign=load_campaign();admission,records=measurements()
    result=campaign.resource_projection(admission,records,datetime(2026,10,2,tzinfo=timezone.utc))
    assert result["steps_per_stage"]==1000
    assert result["full_stages_per_fold"]==8
    assert result["full_stage_wall_seconds"]<=21600
    assert result["full_stage_watchdog_seconds"]<result["full_stage_wall_seconds"]
    # Short gold epochs save hundreds of histories; the 500-step interval alone
    # materially underestimates storage. Final-fit/delivery are reserved too.
    assert result["history_count_upper_bound_per_fit"]>390
    assert result["checkpoint_storage_allowance_bytes"]>300_000_000_000
    assert result["conditional_final_fit_stages"]==8
    assert result["conditional_final_fit_authorized"] is False
    assert result["gpu_delivery_authorized"] is False


@pytest.mark.parametrize("step,startup,peak,reason",[
    (float("nan"),373,18_000_000_000,"Finite"),
    (30,373,18_000_000_000,"bounded stages"),
    (9,373,40_000_000_000,"memory"),
])
def test_admission_rejects_unmeasured_unbounded_or_excess_memory_cost(step,startup,peak,reason):
    campaign=load_campaign();admission,records=measurements(step,startup,peak)
    with pytest.raises(ValueError,match=reason):
        campaign.resource_projection(admission,records,datetime(2026,10,2,tzinfo=timezone.utc))


def test_full_route_is_rejected_near_science_freeze_without_reducing_scientific_steps():
    campaign=load_campaign();admission,records=measurements()
    with pytest.raises(ValueError,match="calendar headroom"):
        campaign.resource_projection(admission,records,datetime(2026,10,10,tzinfo=timezone.utc))
    assert campaign.STEPS==7500
