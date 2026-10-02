"""Protect complete partition membership, valid-slice packing and slot ownership."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))


def module(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/(name+".py"))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


extract=module("ticket67_biomedclip_extract")
coordinator=module("coordinate_ticket67_biomedclip_cohort")
validator=module("validate_ticket67_biomedclip_cohort")


def test_balancing_preserves_membership_and_is_order_independent():
    rows=[dict(study_instance_uid=str(i),valid_slices=8*(i%16+1)) for i in range(53)]
    result=extract.balanced_partitions(rows)
    assert result==extract.balanced_partitions(rows[::-1])
    assert sorted(uid for part in result for uid in part["study_uids"])==sorted(row["study_instance_uid"] for row in rows)
    assert sum(part["valid_slices"] for part in result)==sum(row["valid_slices"] for row in rows)
    assert max(part["valid_slices"] for part in result)-min(part["valid_slices"] for part in result)<=128


def test_embedding_packing_retains_indices_and_zero_padding():
    indices=np.array([0,9,127],dtype=np.int64);values=np.zeros((3,512),np.float32)
    values[0,1]=1;values[1,2]=1;values[2,3]=1
    packed=extract.packed_features(values,indices).reshape(128,512)
    assert np.array_equal(packed[indices],values)
    invalid=np.ones(128,bool);invalid[indices]=False;assert not packed[invalid].any()
    with pytest.raises(ValueError):extract.packed_features(values,indices[::-1])
    with pytest.raises(ValueError):extract.packed_features(values*2,indices)


@pytest.mark.parametrize("states,verification,slots,expected",[
    (["unlaunched"]*4,"unlaunched",2,None),
    (["active","unlaunched","unlaunched","unlaunched"],"unlaunched",1,None),
    (["complete","unlaunched","unlaunched","unlaunched"],"unlaunched",1,1),
    (["complete"]*4,"unlaunched",1,"verification"),
    (["complete"]*4,"active",1,None),
])
def test_one_cohort_job_and_two_total_slots(states,verification,slots,expected):
    assert coordinator.next_component(states,verification,slots)==expected


def test_fitting_scheduler_restored_after_failed_ownership_transfer(tmp_path,monkeypatch):
    monkeypatch.setattr(coordinator,"ROOT",tmp_path);monkeypatch.setattr(coordinator,"PROJECT",tmp_path)
    monkeypatch.setattr(coordinator,"digest",lambda path:"fixed")
    (tmp_path/"plan.json").write_text(json.dumps(dict(decision_path=str(tmp_path/"decision.json"),frozen_files=[])))
    (tmp_path/"decision.json").write_text(json.dumps(dict(decision="approved",plan_sha256="fixed",delegated_registered_ticket64_fit_scheduling=True)))
    calls=iter([True,True,True]);monkeypatch.setattr(coordinator,"active",lambda:next(calls))
    monkeypatch.setattr(coordinator.subprocess,"run",lambda *args,**kwargs:SimpleNamespace(returncode=0))
    resumed=[];monkeypatch.setattr(coordinator,"resume",lambda:resumed.append(True))
    with pytest.raises(ValueError,match="release ownership"):coordinator.watch("fixed")
    assert resumed==[True] and (tmp_path/"original-fit-scheduler-restored.json").exists()


def test_consumed_coordinator_cannot_launch_again(tmp_path,monkeypatch):
    monkeypatch.setattr(coordinator,"ROOT",tmp_path);monkeypatch.setattr(coordinator,"digest",lambda path:"fixed")
    (tmp_path/"plan.json").write_text(json.dumps(dict(decision_path=str(tmp_path/"decision.json"),frozen_files=[])))
    (tmp_path/"decision.json").write_text(json.dumps(dict(decision="approved",plan_sha256="fixed",delegated_registered_ticket64_fit_scheduling=True)))
    (tmp_path/"coordinator-start.json").write_text("{}")
    with pytest.raises(FileExistsError):coordinator.watch("fixed")


@pytest.mark.parametrize("changed",[None,"wrong_job","failed_exit","gpu","wrong_memory"])
def test_terminal_evidence_must_match_actual_successful_cpu_allocation(tmp_path,changed):
    controller=tmp_path/"controller-123";controller.mkdir()
    observation=dict(job_id="123",state="COMPLETED",returncode=0)
    fields=dict(JobId="123",JobState="COMPLETED",ExitCode="0:0",Partition="cluster1-cpu",
        NumNodes="1",NumCPUs="16",NumTasks="1",MinMemoryNode="64G",TimeLimit="03:00:00",
        OverSubscribe="OK",Requeue="0",Restarts="0",RunTime="01:02:03",TRES="cpu=16,mem=64G")
    if changed=="wrong_job":observation["job_id"]="456"
    if changed=="failed_exit":fields["ExitCode"]="1:0"
    if changed=="gpu":fields["TRES"]+=",gres/gpu=1"
    if changed=="wrong_memory":fields["MinMemoryNode"]="128G"
    (controller/"terminal-observation.json").write_text(json.dumps(observation))
    (controller/"terminal-controller.txt").write_text(" ".join(f"{k}={v}" for k,v in fields.items()))
    if changed:
        with pytest.raises(ValueError,match="CPU terminal"):validator.terminal(tmp_path,123,"03:00:00")
    else:
        seconds,checksum=validator.terminal(tmp_path,123,"03:00:00")
        assert seconds==3723 and len(checksum)==64
