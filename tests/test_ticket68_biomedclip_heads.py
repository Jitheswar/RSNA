"""Protect matched initialization, independent evidence and scheduling boundaries."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))


def module(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/(name+".py"))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


worker=module("ticket68_biomedclip_heads")
validator=module("validate_ticket68_biomedclip_heads")
preparer=module("prepare_ticket68_biomedclip_heads")
coordinator=module("coordinate_ticket68_biomedclip_heads")


def test_initial_head_matches_six_historical_random_heads(monkeypatch):
    from rsna_knee import frozen_head_experiment as heads
    from rsna_knee.model import ModelConfig
    from rsna_knee.pretrained import RESNET18_SHA256
    from rsna_knee.resnet18 import RESNET18_ENCODER_ID,RESNET18_MODEL_ID,ANATOMICAL_POOLING_ID
    from rsna_knee.training import _model_state_sha256
    for name in ("FEATURE_WIDTH","POOLED_WIDTH","HEAD_ID","build_head"):monkeypatch.setattr(heads,name,getattr(heads,name))
    config=ModelConfig(image_size=(224,224),channels=(64,128,256),model_id=RESNET18_MODEL_ID,
        pooling_id=ANATOMICAL_POOLING_ID,encoder_id=RESNET18_ENCODER_ID,encoder_weights_sha256=RESNET18_SHA256)
    configured=worker.configure_head(dict(original_model_config=config.to_dict()))
    assert _model_state_sha256(configured.build_head(20260911))=="ea67e0aa11250dfccbfd2e593a78c445684bd8970ac7e9e8151bbc62e3c52d52"


@pytest.mark.parametrize("change",[None,"step","learning_rate","negative_second_moment"])
def test_independent_adam_audit_rejects_wrong_training_state(change):
    head=torch.nn.Sequential(torch.nn.Linear(2076,32),torch.nn.ReLU(),torch.nn.Linear(32,12))
    optimizer=torch.optim.Adam(head.parameters(),lr=.001)
    for _ in range(3):
        optimizer.zero_grad();head(torch.ones(2,2076)).sum().backward();optimizer.step()
    checkpoint=dict(head_state_dict=head.state_dict(),optimizer_state_dict=optimizer.state_dict())
    if change=="step":checkpoint['optimizer_state_dict']['state'][0]['step']=torch.tensor(2.)
    if change=="learning_rate":checkpoint['optimizer_state_dict']['param_groups'][0]['lr']=.01
    if change=="negative_second_moment":checkpoint['optimizer_state_dict']['state'][0]['exp_avg_sq'].flatten()[0]=-1
    if change:
        with pytest.raises(ValueError):validator.audit_optimizer(checkpoint,steps=3)
    else:validator.audit_optimizer(checkpoint,steps=3)


@pytest.mark.parametrize("fit,verification,occupied,expected",[
    ("unlaunched","unlaunched",2,None),("unlaunched","unlaunched",1,"fit"),
    ("active","unlaunched",1,None),("complete","unlaunched",1,"verification"),
    ("complete","active",1,None),("complete","complete",1,None),
])
def test_fixed_experiment_stages_respect_slots_and_order(fit,verification,occupied,expected):
    assert coordinator.next_component(fit,verification,occupied)==expected


@pytest.mark.parametrize("failure",["failed_terminal","incomplete_coverage"])
def test_complete_bank_required_before_creating_fit_request(tmp_path,monkeypatch,failure):
    bank=tmp_path/"bank";bank.mkdir();(bank/"verification").mkdir()
    (bank/"plan.json").write_text(json.dumps(dict(feature_identity_sha256="feature")))
    (bank/"feature-manifest.json").write_text(json.dumps(dict(study_count=4228,feature_identity_sha256="feature")))
    (bank/"verification/independent-validation.json").write_text(json.dumps(dict(verified_studies=4227 if failure=="incomplete_coverage" else 4228,
        verified_valid_slices=187272,replayed_fixed_native_slices=344,maximum_replay_abs_error=0)))
    (bank/"terminal-verification.json").write_text(json.dumps(dict(status="COMPLETE",passed=failure!="failed_terminal",plan_sha256=worker.digest(bank/"plan.json"),
        independent_validation_sha256=worker.digest(bank/"verification/independent-validation.json"),manifest_sha256=worker.digest(bank/"feature-manifest.json"))))
    monkeypatch.setattr(preparer,"BANK",bank);monkeypatch.setattr(preparer,"BANK_SHA",worker.digest(bank/"plan.json"))
    monkeypatch.setattr(preparer,"ROOT",tmp_path/"fit");monkeypatch.setattr(preparer,"DECISION",tmp_path/"decision.json")
    with pytest.raises(ValueError,match="whole-bank"):preparer.main()
    assert not preparer.ROOT.exists() and not preparer.DECISION.exists()


@pytest.mark.parametrize("change",[None,"wrong_job","failed_exit","excess_runtime"])
def test_actual_terminal_requires_matching_success_and_admitted_bound(tmp_path,change):
    controller=tmp_path/"controller-123";controller.mkdir()
    observation=dict(job_id="456" if change=="wrong_job" else "123",state="COMPLETED",returncode=0)
    fields=dict(JobId="123",JobState="COMPLETED",ExitCode="1:0" if change=="failed_exit" else "0:0",Partition="cluster1-cpu",NumNodes="1",NumCPUs="16",NumTasks="1",
        MinMemoryNode="64G",TimeLimit="00:30:00",OverSubscribe="OK",Requeue="0",Restarts="0",RunTime="00:31:00" if change=="excess_runtime" else "00:04:00")
    (controller/"terminal-observation.json").write_text(json.dumps(observation))
    (controller/"terminal-controller.txt").write_text(" ".join(f"{k}={v}" for k,v in fields.items()))
    if change:
        with pytest.raises(ValueError):validator.successful_terminal(tmp_path,123,"00:30:00")
    else:assert validator.successful_terminal(tmp_path,123,"00:30:00")[0]==240


def test_comparison_requires_positive_interval_and_all_folds():
    good=dict(paired_study_bootstrap=dict(macro_auc=dict(lower=.01)),delta=dict(folds=[dict(delta_macro_auc=.01)]*3))
    assert worker.positive_comparison(good)
    good['delta']['folds'][1]=dict(delta_macro_auc=-.01)
    assert not worker.positive_comparison(good)


def test_prediction_replay_rejects_membership_and_nonfinite_values():
    actual={'synthetic':[.5]*12};expected={'synthetic':[.5]*12}
    assert validator.maximum_error(actual,expected)==0
    with pytest.raises(ValueError):validator.maximum_error(actual,{'different':[.5]*12})
    actual['synthetic'][0]=float('nan')
    with pytest.raises(ValueError):validator.maximum_error(actual,expected)


def test_fit_scheduler_restored_after_failed_transfer(tmp_path,monkeypatch):
    controller=tmp_path/"controller";controller.mkdir();bank=tmp_path/"bank";bank.mkdir();root=tmp_path/"fit";root.mkdir()
    decision=tmp_path/"decision.json";decision.write_text(json.dumps(dict(delegated_registered_ticket64_fit_scheduling=True)))
    (root/"plan.json").write_text(json.dumps(dict(decision_path=str(decision))))
    (bank/"terminal-verification.json").write_text(json.dumps(dict(passed=True)))
    (controller/"admission.json").write_text(json.dumps(dict(frozen_files=[],bank_plan_sha256="fixed",source_snapshot=str(tmp_path),science_freeze_utc="2099-10-11T00:00:00+00:00")))
    monkeypatch.setattr(coordinator,"CONTROLLER",controller);monkeypatch.setattr(coordinator,"ROOT",root);monkeypatch.setattr(coordinator,"BANK",bank)
    monkeypatch.setattr(coordinator,"digest",lambda _:"fixed");monkeypatch.setattr(coordinator,"active",lambda unit:unit==coordinator.FIT_UNIT)
    monkeypatch.setattr(coordinator.signal,"signal",lambda *_:None)
    monkeypatch.setattr(coordinator.subprocess,"run",lambda *args,**kwargs:SimpleNamespace(returncode=0))
    import prepare_ticket68_biomedclip_heads
    monkeypatch.setattr(prepare_ticket68_biomedclip_heads,"main",lambda _:"fixed")
    resumed=[];monkeypatch.setattr(coordinator,"resume",lambda:resumed.append(True))
    with pytest.raises(ValueError,match="release scheduling"):coordinator.watch("fixed")
    assert resumed==[True] and (controller/"original-fit-scheduler-restored.json").exists()


def test_consumed_controller_cannot_restart(tmp_path,monkeypatch):
    (tmp_path/"admission.json").write_text(json.dumps(dict(frozen_files=[])))
    (tmp_path/"coordinator-start.json").write_text("{}")
    monkeypatch.setattr(coordinator,"CONTROLLER",tmp_path);monkeypatch.setattr(coordinator,"digest",lambda _:"fixed")
    monkeypatch.setattr(coordinator.signal,"signal",lambda *_:None)
    with pytest.raises(FileExistsError):coordinator.watch("fixed")
