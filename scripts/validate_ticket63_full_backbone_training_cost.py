"""Bind diagnostic staged-state replay and complete costs to the actual terminal."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch

from rsna_knee.training import load_training_checkpoint, _model_state_sha256, _batch_indices
from rsna_knee.folds import load_fold_manifest
from rsna_knee.ticket34 import admit_ticket30_fold_supervision
from ticket63_full_backbone_training_cost import config, initial_model, t36


def read(path):return json.loads(path.read_text())
def digest(path):
    with path.open("rb") as handle:return hashlib.file_digest(handle,"sha256").hexdigest()


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b,strict=True))
    return a==b


def validate(root):
    plan=read(root / "plan.json");checksum=digest(root / "plan.json")
    result=read(root / "cost-result.json");accepted=read(root / "accepted-launch.json")
    job=str(accepted["job_id"]);path=root / f"controller-{job}/terminal-controller.txt"
    fields=dict(t.split("=",1) for t in path.read_text().split() if "=" in t)
    observed=read(path.with_name("terminal-observation.json"))
    expected=dict(JobId=job,JobState="COMPLETED",ExitCode="0:0",Partition="cluster1-cpu",NumNodes="1",
        NumCPUs="16",NumTasks="1",MinMemoryNode="64G",TimeLimit="01:30:00",OverSubscribe="OK",Requeue="0",Restarts="0")
    assert accepted["plan_sha256"]==result["plan_sha256"]==checksum and result["job_id"]==job
    assert all(fields.get(k)==v for k,v in expected.items()) and "gpu" not in fields["TRES"].lower()
    assert observed["job_id"]==job and observed["state"]=="COMPLETED" and observed["returncode"]==0
    decision=read(Path(plan["decision_path"]))
    assert decision["decision"]=="approved" and decision["plan_sha256"]==checksum and decision["resources"]==plan["resources"]
    assert all(digest(Path(r["path"]))==r["sha256"] for r in plan["frozen_files"])
    assert result["status"]=="COMPLETE" and not result["held_out_predictions"] and not result["full_training_authorized"]
    assert result["exact_fresh_process_staged_model_optimizer_progress_replay"] and result["normalization_buffers_unchanged"]
    assert all(result["all_encoder_stages_changed"].values()) and not result["gpu_count"] and not result["external_spend_usd"]
    assert result["admission_result_sha256"]==digest(root / "admission-result.json")
    admission=read(root / "admission-result.json")
    assert admission["status"]=="COMPLETE" and admission["loaded_studies"]==4228 and not admission["full_pixel_arrays_retained"]
    assert [(r["fold"],r["reference_studies"],r["weak_studies"]) for r in admission["folds"]]==[(0,38,2731),(1,39,2846),(2,39,2749)]
    stages={s:read(root / f"{s}-result.json") for s in ("interrupt","resume","continuous")}
    assert all(result["stage_result_sha256"][s]==digest(root / f"{s}-result.json") for s in stages)
    assert all(r["status"]=="COMPLETE" and r["plan_sha256"]==checksum and r["job_id"]==job and not r["held_out_predictions"] for r in stages.values())
    assert [stages[s]["successful_steps"] for s in stages]==[1,3,3]
    assert stages["interrupt"]["planned_stage_boundary"] and not stages["resume"]["planned_stage_boundary"]
    for name,folder in (("resume","staged"),("continuous","continuous")):
        assert stages[name]["checkpoint_sha256"]==digest(root / f"{folder}/checkpoints/final.pt")
    torch.set_num_threads(1)
    saved=[load_training_checkpoint(root / f"{folder}/checkpoints/final.pt",expected_config=config(),expected_fold=0)
        for folder in ("staged","continuous")]
    for key in ("model_state_dict","optimizer_state_dict","scheduler_state_dict","progress","identity","sampler_state"):
        assert equal(saved[0][key],saved[1][key]),key
    warm=initial_model()
    assert all(r["initial_model_state_sha256"]==_model_state_sha256(warm) for r in stages.values())
    initial=warm.state_dict();state=saved[0]["model_state_dict"]
    for prefix in result["all_encoder_stages_changed"]:
        assert any(not torch.equal(v,initial[k]) for k,v in state.items() if k.startswith(prefix))
    assert all(torch.equal(v,initial[k]) for k,v in state.items() if k.endswith(("running_mean","running_var","num_batches_tracked")))
    assert [r["lr"] for r in saved[0]["optimizer_state_dict"]["param_groups"]]==[1e-5,1e-4]
    manifest=load_fold_manifest(t36.FOLD_MANIFEST_PATH)
    parent=next(r for r in read(Path(plan["cohort_parent_plan"]))["folds"] if r["fold"]==0)
    uids=[r["study_instance_uid"] for r in read(Path(plan["pixel_root"]) / "pixel-manifest.json")["pixels"]]
    labels=admit_ticket30_fold_supervision(manifest,0,t36.TICKET30_ARTIFACT,
        expected_artifact_sha256=t36.TICKET30_AUDITED_ARTIFACT_SHA256).for_image_studies(uids)
    assert labels.to_public_dict()==parent["supervision_admission"]
    exposures={kind:dict(sample_exposures=0,supervised_entry_exposures=0) for kind in ("reference","weak","mixed","pseudo")}
    for attempt in range(3):
        for kind,pool,seed in (("reference",parent["gold_training_study_uids"],20260912),
            ("weak",parent["weak_training_study_uids"],20260912+34034034)):
            for i in _batch_indices(len(pool),2,attempt,seed):
                uid=pool[i];assert uid not in manifest.held_out_study_uids(0)
                exposures[kind]["sample_exposures"]+=1
                exposures[kind]["supervised_entry_exposures"]+=sum(labels.by_study[uid].supervision_mask)
    progress=saved[0]["progress"]
    assert progress["successful_optimizer_steps"]==progress["attempted_batches"]==3 and progress["sample_exposures"]==12
    assert progress["exposures_by_source"]==exposures
    seconds=int((datetime.fromisoformat(fields["EndTime"])-datetime.fromisoformat(fields["StartTime"])).total_seconds())
    return dict(passed=True,job_id=int(job),plan_sha256=checksum,cost_result_sha256=digest(root / "cost-result.json"),
        terminal_state="COMPLETED",exit_code="0:0",allocated_cpu_node_seconds=seconds,actual_resources_match=True,
        exact_staged_model_optimizer_scheduler_progress_and_sampler_replay=True,
        original_fold_targets_masks_exposures_verified=True,initial_admitted_encoder_head_verified=True,
        normalization_buffers_unchanged=True,all_encoder_stages_changed=True,
        actual_complete_startup_and_checkpoint_cost_admitted=True,predictive_gain_established=False,
        full_training_selected=False,observed_at_utc=datetime.now(timezone.utc).isoformat())


if __name__=="__main__":
    root=Path(sys.argv[1]).resolve();output=root / "terminal-verification.json"
    if output.exists():raise FileExistsError("Immutable independent terminal reconciliation already exists")
    result=validate(root)
    with output.open("x") as handle:json.dump(result,handle,indent=2,sort_keys=True);handle.write("\n")
    print(json.dumps(result))
