"""Recovery must reject incomplete, changed or shape-incompatible state."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location("checkpoint_audit", Path(__file__).parents[1] / "scripts/ticket72_checkpoint_recovery_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture():
    exposures = {k: dict(sample_exposures=0, supervised_entry_exposures=0) for k in ("reference", "weak", "mixed", "pseudo")}
    exposures["reference"] = dict(sample_exposures=8000, supervised_entry_exposures=96000)
    exposures["weak"] = dict(sample_exposures=8000, supervised_entry_exposures=39000)
    progress = {k: 4000 for k in ("successful_optimizer_steps", "attempted_batches", "scheduler_steps", "global_step", "next_batch_index")}
    progress.update(skipped_all_masked_batches=0, consecutive_no_progress=0, sample_exposures=16000,
                    exposures_by_source=exposures, supervised_entry_exposures=135000, epoch=210, batch_in_epoch=10)
    payload = dict(progress=progress, sampler_state=dict(kind="deterministic-cyclic", seed=20260912, next_batch_index=4000),
                   checkpoint_kind="latest_resumable", checkpoint_reason="optimizer_step_interval+planned_interruption")
    return payload, dict(progress=copy.deepcopy(progress)), exposures


def test_exact_original_boundary_accepted():
    payload, record, exposures = fixture()
    audit.validate_progress(payload, record, 4000, exposures, 38)


@pytest.mark.parametrize("key,value", [("successful_optimizer_steps", 3999), ("attempted_batches", 4001),
    ("scheduler_steps", 3999), ("next_batch_index", 0), ("sample_exposures", 15999),
    ("supervised_entry_exposures", 134999), ("epoch", 209), ("batch_in_epoch", 0),
    ("skipped_all_masked_batches", 1), ("consecutive_no_progress", 1)])
def test_consistently_tampered_worker_and_checkpoint_still_rejected(key, value):
    payload, record, exposures = fixture()
    payload["progress"][key] = value
    record["progress"][key] = value
    with pytest.raises(ValueError):
        audit.validate_progress(payload, record, 4000, exposures, 38)


@pytest.mark.parametrize("key,value", [("checkpoint_kind", "final_selected"), ("checkpoint_reason", "time_interval"),
    ("sampler_state", dict(kind="deterministic-cyclic", seed=20261002, next_batch_index=4000))])
def test_wrong_resume_kind_reason_or_sampler_rejected(key, value):
    payload, record, exposures = fixture()
    payload[key] = value
    with pytest.raises(ValueError):
        audit.validate_progress(payload, record, 4000, exposures, 38)


def adam_fixture():
    params = [torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(3))]
    optimizer = torch.optim.Adam([dict(params=[params[0]], lr=1e-5), dict(params=[params[1]], lr=1e-4)])
    for param in params:
        optimizer.state[param] = dict(step=torch.tensor(4000.), exp_avg=torch.zeros_like(param), exp_avg_sq=torch.ones_like(param))
    return optimizer, params


def test_complete_shape_compatible_adam_accepted():
    optimizer, _ = adam_fixture()
    audit.validate_adam(optimizer, 4000)


@pytest.mark.parametrize("defect", ["missing", "step", "shape", "nonfinite", "negative", "rate", "betas"])
def test_resume_rejects_incomplete_or_changed_adam(defect):
    optimizer, params = adam_fixture()
    state = optimizer.state[params[0]]
    if defect == "missing":
        del optimizer.state[params[0]]
    elif defect == "step":
        state["step"] = torch.tensor(3000.)
    elif defect == "shape":
        state["exp_avg"] = torch.zeros(3)
    elif defect == "nonfinite":
        state["exp_avg_sq"][0] = float("nan")
    elif defect == "negative":
        state["exp_avg_sq"][0] = -1
    elif defect == "rate":
        optimizer.param_groups[0]["lr"] = 1e-4
    else:
        optimizer.param_groups[0]["betas"] = (0.8, 0.99)
    with pytest.raises(ValueError):
        audit.validate_adam(optimizer, 4000)
