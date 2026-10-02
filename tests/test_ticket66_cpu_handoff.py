"""Preserve the fitting scheduler when a bounded pilot handoff cannot proceed."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SOURCE=Path(__file__).resolve().parents[1]/"scripts/coordinate_ticket66_cpu_handoff.py"
spec=importlib.util.spec_from_file_location("ticket66_handoff",SOURCE)
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)


@pytest.mark.parametrize("failure",["stop_timeout","failed_fit"])
def test_scheduler_restored_if_handoff_fails(tmp_path,monkeypatch,failure):
    project=tmp_path;root=project/"pilot";root.mkdir()
    monkeypatch.setattr(controller,"PROJECT",project);monkeypatch.setattr(controller,"ROOT",root)
    monkeypatch.setattr(controller,"digest",lambda path:"fixed")
    admission=dict(source_sha256="fixed",frozen_files=[],maximum_wait_for_slot_seconds=21600,
        science_freeze_utc="2026-10-11T00:00:00+00:00",
        watched_fits=[dict(accepted_path="receipt",accepted_sha256="fixed",directory=str(root),expected_steps=1000,job_id="123")])
    (root/"handoff-admission.json").write_text(json.dumps(admission))
    decision=project/".scratch/rsna-leaderboard/resource-approvals/ticket66-cpu-handoff-20261002.json"
    decision.parent.mkdir(parents=True);decision.write_text(json.dumps(dict(decision="approved",handoff_admission_sha256="fixed")))
    active_calls=iter([True,False,True]);monkeypatch.setattr(controller,"active",lambda:next(active_calls))
    monkeypatch.setattr(controller,"cpu_jobs",lambda:[["123","cluster1-cpu","R","rsna-ticket64-full"]])
    resumed=[];monkeypatch.setattr(controller,"resume",lambda:resumed.append(True))
    def run(*args,**kwargs):
        if failure=="stop_timeout":raise TimeoutError("stop observation expired")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(controller.subprocess,"run",run)
    def failed_fit(*args):raise ValueError("original fit failed")
    monkeypatch.setitem(sys.modules,"coordinate_ticket64_full_backbone_comparison",SimpleNamespace(stage_state=failed_fit))
    with pytest.raises((TimeoutError,ValueError)):controller.watch("fixed")
    assert resumed==[True]
    assert (root/"fitting-scheduler-resumed.json").exists()
    assert not (root/"launch-intent.json").exists()


def test_consumed_handoff_is_never_repeated(tmp_path,monkeypatch):
    monkeypatch.setattr(controller,"ROOT",tmp_path)
    monkeypatch.setattr(controller,"PROJECT",tmp_path)
    monkeypatch.setattr(controller,"digest",lambda path:"fixed")
    monkeypatch.setattr(controller,"read",lambda path:dict(source_sha256="fixed",frozen_files=[],decision="approved",handoff_admission_sha256="fixed"))
    (tmp_path/"handoff-start.json").write_text("{}")
    def forbidden():raise AssertionError("Cannot stop or launch again")
    monkeypatch.setattr(controller,"active",forbidden)
    with pytest.raises(FileExistsError):controller.watch("fixed")


@pytest.mark.parametrize("change",["none","native_nan","native_error","failed_terminal","different_result"])
def test_terminal_requires_bounded_independent_evidence(tmp_path,monkeypatch,change):
    monkeypatch.setattr(controller,"ROOT",tmp_path)
    root=tmp_path/"controller-123";root.mkdir()
    (root/"terminal-observation.json").write_text(json.dumps(dict(state="COMPLETED",returncode=0)))
    fields=dict(JobId="123",JobState="COMPLETED",ExitCode="0:0",Partition="cluster1-cpu",NumNodes="1",NumCPUs="16",
        NumTasks="1",MinMemoryNode="64G",TimeLimit="00:16:00",OverSubscribe="OK",Requeue="0",Restarts="0",RunTime="00:02:00")
    if change=="failed_terminal":fields["ExitCode"]="1:0"
    (root/"terminal-controller.txt").write_text(" ".join(f"{k}={v}" for k,v in fields.items()))
    (tmp_path/"plan.json").write_text(json.dumps(dict(expected_valid_slices=344)))
    result=dict(status="COMPLETE",plan_sha256="fixed",job_id="123",peak_rss_bytes=1_000_000_000,candidate_promotion=False,
        study_count=8,valid_slices=344,strict_state_coverage=True,fitting=False,labels_used=False,text_tower_constructed=False,
        network=False,elapsed_seconds=70)
    (tmp_path/"pilot-result.json").write_text(json.dumps(result))
    verification=dict(status="COMPLETE",plan_sha256="fixed",result_sha256=controller.digest(tmp_path/"pilot-result.json"),
        job_id="123",verified_studies=8,verified_slices=344,maximum_native_output_abs_error=0.,original_dicom_replay=True,
        elapsed_seconds=40)
    if change=="native_nan":verification["maximum_native_output_abs_error"]=float("nan")
    if change=="native_error":verification["maximum_native_output_abs_error"]=0.01
    if change=="different_result":verification["result_sha256"]="different"
    (tmp_path/"independent-validation.json").write_text(json.dumps(verification))
    if change=="none":
        controller.reconcile(dict(pilot_plan_sha256="fixed"),123)
        assert json.loads((tmp_path/"terminal-verification.json").read_text())["allocated_cpu_node_seconds"]==120
    else:
        with pytest.raises(ValueError):controller.reconcile(dict(pilot_plan_sha256="fixed"),123)
        assert not (tmp_path/"terminal-verification.json").exists()
