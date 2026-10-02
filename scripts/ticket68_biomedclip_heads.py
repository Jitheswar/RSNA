"""Fit fixed medical heads against the already verified matched ResNet experiment."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ARMS=("reference-only","report-supervision")
HEAD_ID="biomedclip-vitb16-projected-l2-plane-pooled-mlp32-v1"


def digest(path):
    with Path(path).open("rb") as handle:return hashlib.file_digest(handle,"sha256").hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with Path(path).open("x") as handle:json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False);handle.write("\n")


def require(condition,message):
    if not condition:raise ValueError(message)


def check_allocation(plan,verification=False):
    import torch
    job=os.environ["SLURM_JOB_ID"]
    value=subprocess.check_output(["scontrol","show","job",job,"-o"],text=True)
    fields=dict(t.split("=",1) for t in value.split() if "=" in t)
    expected=dict(JobId=job,JobState="RUNNING",Partition="cluster1-cpu",NumNodes="1",NumCPUs="16",NumTasks="1",
        MinMemoryNode="64G",TimeLimit="03:00:00" if verification else "00:30:00",OverSubscribe="OK",Requeue="0",Restarts="0")
    require(all(fields.get(k)==v for k,v in expected.items()) and "gpu" not in fields.get("TRES","").lower()
        and torch.version.cuda is None and not torch.cuda.is_available(),"Exact CPU-only allocation required")
    return value


def configure_head(plan):
    from rsna_knee import frozen_head_experiment as heads
    from rsna_knee.model import ModelConfig,build_study_model
    from rsna_knee.training import seed_everything
    heads.FEATURE_WIDTH=512;heads.POOLED_WIDTH=2076;heads.HEAD_ID=HEAD_ID
    # Reproduce the original constructor's RNG consumption. These are random
    # head weights, never learned weights transferred across representations.
    def original_random_head(seed):
        seed_everything(seed)
        return build_study_model(ModelConfig.from_dict(plan["original_model_config"])).head
    heads.build_head=original_random_head
    return heads


def load_bank(path,checksum,array_checksum):
    import numpy as np
    import torch
    from rsna_knee.training import _array_sha256
    require(digest(path)==checksum,"Exact pooled feature bytes required")
    with np.load(path,allow_pickle=False) as data:
        values=data["pooled"].copy();uids=data["study_uids"].tolist()
    require(values.dtype==np.float32 and values.shape==(4228,2076) and np.isfinite(values).all()
        and uids==sorted(set(uids)) and _array_sha256(values)==array_checksum,"Exact finite unfitted cohort required")
    return torch.from_numpy(values),uids


def fold_tensors(plan,fold_row,uids):
    import torch
    from rsna_knee.folds import load_fold_manifest
    from rsna_knee.ticket34 import admit_ticket30_fold_supervision
    manifest=load_fold_manifest(plan["fold_manifest_path"])
    require(not manifest.superseded and manifest.manifest_checksum==plan["fold_manifest_checksum"],"Original corrected folds required")
    fold=fold_row["fold"];gold={m.study_instance_uid:m for m in manifest.gold_memberships};index={u:i for i,u in enumerate(uids)}
    train,held,weak=[fold_row[k] for k in ("gold_training_study_uids","gold_heldout_study_uids","weak_training_study_uids")]
    require(len(gold)==58 and len(set(train+held+weak))==len(train+held+weak)
        and set(held)=={u for u,m in gold.items() if m.fold==fold} and set(train)==set(gold)-set(held)
        and not set(train+weak)&set(manifest.held_out_study_uids(fold)) and set(train+held+weak)<=set(uids),"Exact disjoint original fitting membership required")
    admission=admit_ticket30_fold_supervision(manifest,fold,plan["report_artifact_path"],
        expected_artifact_sha256=plan["report_artifact_sha256"]).for_image_studies(uids)
    require(admission.to_public_dict()==fold_row["supervision_admission"],"Original report admission required")
    targets=torch.zeros((len(uids),12),dtype=torch.float32);mask=torch.zeros_like(targets,dtype=torch.bool)
    for uid in train+weak:
        labels=gold[uid].reference_labels if uid in gold else admission.by_study[uid].targets
        for col,value in enumerate(labels):
            if value is not None:targets[index[uid],col]=float(value);mask[index[uid],col]=True
    require(not mask[[index[u] for u in held]].any(),"Held-out targets must remain masked")
    return manifest,targets,mask,index,train,held,weak


def predict(head,pooled,members,index):
    import torch
    result={};head.eval()
    with torch.inference_mode():
        for start in range(0,len(members),2):
            batch=members[start:start+2];values=torch.sigmoid(head(pooled[[index[u] for u in batch]])).numpy()
            require(torch.isfinite(torch.from_numpy(values)).all(),"Finite head predictions required")
            result.update({u:v.tolist() for u,v in zip(batch,values,strict=True)})
    return result


def positive_comparison(comparison):
    lower=comparison["paired_study_bootstrap"]["macro_auc"]["lower"]
    return lower is not None and lower>0 and all(r["delta_macro_auc"] is not None and r["delta_macro_auc"]>0 for r in comparison["delta"]["folds"])


def analyze(plan,manifest,medical,resnet,zero):
    import numpy as np
    from rsna_knee.image_model_comparison import compare_image_model_predictions
    blend={u:((np.asarray(medical[ARMS[1]][u])+np.asarray(resnet[ARMS[1]][u]))/2).tolist() for u in medical[ARMS[1]]}
    pairs=dict(primary=(resnet[ARMS[1]],medical[ARMS[1]]),reference_representation=(resnet[ARMS[0]],medical[ARMS[0]]),
        report_supervision=(medical[ARMS[0]],medical[ARMS[1]]),reference_image_signal=(zero[ARMS[0]],medical[ARMS[0]]),
        report_image_signal=(zero[ARMS[1]],medical[ARMS[1]]),fixed_half_blend=(resnet[ARMS[1]],blend))
    comparisons={k:compare_image_model_predictions(a,b,manifest,bootstrap_resamples=2000,bootstrap_seed=20261002) for k,(a,b) in pairs.items()}
    image_pass=positive_comparison(comparisons["report_image_signal"])
    disposition="provisional_medical_review" if image_pass and positive_comparison(comparisons["primary"]) else (
        "provisional_fixed_blend_review" if image_pass and positive_comparison(comparisons["fixed_half_blend"]) else "inconclusive_no_promotion")
    return dict(comparisons=comparisons,fixed_half_blend_predictions=blend,disposition=disposition,candidate_promotion=False,
        development_cohort_reused=True,patient_group_ids_unavailable=True,scored_anchor_comparison_verified=False,offline_delivery_verified=False)


def run(plan_path,checksum):
    started=time.monotonic();plan=read(plan_path);require(digest(plan_path)==checksum,"Exact medical fitting plan required")
    decision=read(plan["decision_path"]);require(decision["decision"]=="approved" and decision["plan_sha256"]==checksum,"Exact fitting resource decision required")
    root=Path(plan["run_root"]);job=os.environ["SLURM_JOB_ID"];allocation=check_allocation(plan)
    write(root/"fit/running-allocation.json",dict(controller=allocation,job_id=job))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError("Medical head watchdog")));signal.alarm(1740)
    for row in plan["frozen_files"]:require(digest(row["path"])==row["sha256"],"Frozen fitting source/input changed")
    import torch
    import numpy as np
    from rsna_knee.training import _array_sha256
    torch.set_num_threads(16);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    pooled,uids=load_bank(plan["medical_pooled_path"],plan["medical_pooled_sha256"],plan["medical_pooled_array_sha256"])
    heads=configure_head(plan);medical,resnet,zero=[{a:{} for a in ARMS} for _ in range(3)];records=[]
    parameters={k:plan["optimization"][k] for k in ("successful_steps","batch_size","initialization_seed","batch_order_seed","weak_seed_offset","learning_rate","checkpoint_interval_steps")}
    for fold_row in plan["folds"]:
        manifest,targets,mask,index,train,held,weak=fold_tensors(plan,fold_row,uids);fold=fold_row["fold"]
        for arm in ARMS:
            prior=read(Path(plan["resnet_root"])/arm/f"fold-{fold}/run.json")
            directory=root/arm/f"fold-{fold}"
            identity=dict(plan_sha256=checksum,fold=fold,arm=arm,head_id=HEAD_ID,feature_identity_sha256=plan["feature_identity_sha256"],
                fold_manifest_checksum=plan["fold_manifest_checksum"],pooled_sha256=plan["medical_pooled_sha256"])
            require(prior["held_out_study_uids"]==held,"Matched ResNet held-out order differs")
            def checkpoint(head,optimizer,counts):
                require(counts["initial_head_state_sha256"]==prior["counts"]["initial_head_state_sha256"],"Matched random initial head differs")
                final=counts["successful_optimizer_steps"]==7500
                heads.save_head_checkpoint(directory/"checkpoints"/("final.pt" if final else "latest.pt"),head,optimizer,counts,identity,final=final)
            head,counts=heads.train_head(pooled=pooled,targets=targets,mask=mask,study_uids=uids,
                reference_indices=[index[u] for u in train],weak_indices=[index[u] for u in weak] if arm==ARMS[1] else [],checkpoint=checkpoint,**parameters)
            require(counts["skipped_batches"]==0 and all(counts[k]==prior["counts"][k] for k in
                ("initial_head_state_sha256","gold_exposure_sequence_sha256","weak_exposure_sequence_sha256","exposures_by_source","distinct_weak_studies_exposed","weak_training_pool_count")),"Exact matched initialization/exposures required")
            restored,replay_counts=heads.reload_head(directory/"checkpoints/final.pt",identity)
            actual=predict(head,pooled,held,index);replayed=predict(restored,pooled,held,index)
            require(actual==replayed and counts==replay_counts,"Exact final checkpoint replay required")
            medical[arm].update(replayed);zero[arm].update(predict(restored,heads.zero_image_features(pooled),held,index));resnet[arm].update(prior["predictions"])
            record=dict(status="COMPLETE",identity=identity,counts=counts,predictions=replayed,zero_image_predictions={u:zero[arm][u] for u in held},
                targets_array_sha256=_array_sha256(targets.numpy()),mask_array_sha256=_array_sha256(mask.numpy()),
                training_study_uids=train+(weak if arm==ARMS[1] else []),held_out_study_uids=held,held_out_targets_masked=True,
                checkpoint_sha256=digest(directory/"checkpoints/final.pt"),job_id=job,candidate_promotion=False)
            write(directory/"run.json",record);records.append(dict(fold=fold,arm=arm,path=str(directory/"run.json"),sha256=digest(directory/"run.json")))
            print(json.dumps(dict(fold=fold,arm=arm,successful_steps=7500,elapsed_seconds=time.monotonic()-started)),flush=True)
    result=dict(status="COMPLETE",plan_sha256=checksum,job_id=job,run_records=records,**analyze(plan,manifest,medical,resnet,zero),
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gpu_seconds=0,tpu_seconds=0,external_spend_usd=0)
    write(root/"analysis.json",result);signal.alarm(0)
    print(json.dumps(dict(status="COMPLETE",disposition=result["disposition"],elapsed_seconds=result["elapsed_seconds"])),flush=True)


if __name__=="__main__":run(sys.argv[1],sys.argv[2])
