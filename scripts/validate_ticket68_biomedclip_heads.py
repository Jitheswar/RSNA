"""Audit all medical fits and independently reproduce both representations from MRI."""

import os
from pathlib import Path
import resource
import signal
import sys
import time

from ticket68_biomedclip_heads import ARMS,HEAD_ID,digest,read,write,require,check_allocation,configure_head,load_bank,fold_tensors,predict,analyze


def maximum_error(actual,expected):
    import numpy as np
    require(bool(actual) and set(actual)==set(expected),"Exact prediction membership required")
    require(all(np.asarray(actual[u]).shape==np.asarray(expected[u]).shape==(12,)
        and np.isfinite(actual[u]).all() and np.isfinite(expected[u]).all() for u in actual),"Finite twelve-target predictions required")
    return max(float(np.max(np.abs(np.asarray(actual[u])-np.asarray(expected[u])))) for u in actual)


def successful_terminal(directory,job,limit):
    observation=read(directory/f"controller-{job}/terminal-observation.json")
    path=directory/f"controller-{job}/terminal-controller.txt"
    fields=dict(t.split("=",1) for t in path.read_text().split() if "=" in t)
    expected=dict(JobId=str(job),JobState="COMPLETED",ExitCode="0:0",Partition="cluster1-cpu",NumNodes="1",NumCPUs="16",NumTasks="1",
        MinMemoryNode="64G",TimeLimit=limit,OverSubscribe="OK",Requeue="0",Restarts="0")
    require(str(observation["job_id"])==str(job) and observation["state"]=="COMPLETED" and observation["returncode"]==0
        and all(fields.get(k)==v for k,v in expected.items()) and "gpu" not in fields.get("TRES","").lower(),"Exact successful CPU terminal required")
    h,m,s=map(int,fields["RunTime"].split(":"));elapsed=h*3600+m*60+s
    lh,lm,ls=map(int,limit.split(":"));require(0<=elapsed<=lh*3600+lm*60+ls,"Actual allocation exceeded admitted bound")
    return elapsed,digest(path)


def audit_optimizer(checkpoint,steps=7500):
    import torch
    optimizer=checkpoint["optimizer_state_dict"];groups=optimizer["param_groups"]
    require(len(groups)==1 and groups[0]["lr"]==.001 and groups[0]["betas"]==(.9,.999) and groups[0]["eps"]==1e-8
        and groups[0]["weight_decay"]==0 and len(groups[0]["params"])==4 and len(optimizer["state"])==4,"Exact original Adam configuration required")
    shapes=[value.shape for value in checkpoint["head_state_dict"].values()]
    for ident,shape in zip(groups[0]["params"],shapes,strict=True):
        state=optimizer["state"][ident]
        require(float(state["step"])==steps and all(state[k].shape==shape and torch.isfinite(state[k]).all() for k in ("exp_avg","exp_avg_sq"))
            and (state["exp_avg_sq"]>=0).all(),"Exact finite final Adam step/moments required")


def main(plan_path,checksum):
    started=time.monotonic();plan=read(plan_path);require(digest(plan_path)==checksum,"Exact independent replay plan required")
    decision=read(plan["decision_path"]);require(decision["decision"]=="approved" and decision["plan_sha256"]==checksum,"Exact verification resource decision required")
    root=Path(plan["run_root"]);job=os.environ["SLURM_JOB_ID"]
    write(root/"verification/running-allocation.json",dict(job_id=job,controller=check_allocation(plan,True)))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError("Original MRI verification watchdog")));signal.alarm(10740)
    for row in plan["frozen_files"]:require(digest(row["path"])==row["sha256"],"Frozen verifier/fitting/input bytes changed")
    fit_launch=read(root/"fit/accepted-launch.json")
    require(fit_launch["plan_sha256"]==checksum and fit_launch["launch_attempt"]==1,"Exact sole medical fit required")
    fit_seconds,fit_terminal_sha=successful_terminal(root/"fit",fit_launch["job_id"],"00:30:00")
    result=read(root/"analysis.json")
    require(result["status"]=="COMPLETE" and result["plan_sha256"]==checksum and str(result["job_id"])==str(fit_launch["job_id"])
        and not result["candidate_promotion"] and len(result["run_records"])==6,"All six fixed medical runs required")
    import numpy as np
    import torch
    from rsna_knee.training import _array_sha256,_model_state_sha256,_batch_indices
    from rsna_knee.model import ModelConfig,build_study_model
    from rsna_knee.feature_pooling import pool_frozen_features,ANATOMICAL_POOLING_ID
    from rsna_knee.pixel_cache import load_pixel_entry
    from open_clip.model import _build_vision_tower,CustomTextCLIP
    from ticket62_mri_pixel_materialize import iter_packed_sources
    torch.set_num_threads(16);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    pooled,uids=load_bank(plan["medical_pooled_path"],plan["medical_pooled_sha256"],plan["medical_pooled_array_sha256"])
    parent_pool,parent_uids=load_bank(Path(plan["resnet_root"])/"pooled-cohort.npz",plan["resnet_pooled_sha256"],plan["resnet_pooled_array_sha256"])
    require(uids==parent_uids,"Identical unsupervised cohort membership required")
    heads=configure_head(plan);medical,resnet,zero=[{a:{} for a in ARMS} for _ in range(3)];audit=[];checkpoint_states={};gold=set();index={u:i for i,u in enumerate(uids)}
    for fold_row in plan["folds"]:
        manifest,targets,mask,_,train,held,weak=fold_tensors(plan,fold_row,uids);fold=fold_row["fold"];gold.update(held)
        for arm in ARMS:
            directory=root/arm/f"fold-{fold}";record=read(directory/"run.json")
            registered=next(r for r in result["run_records"] if r["fold"]==fold and r["arm"]==arm)
            require(registered["path"]==str(directory/"run.json") and digest(directory/"run.json")==registered["sha256"],"Complete registered run bytes required")
            require(record["identity"]==dict(plan_sha256=checksum,fold=fold,arm=arm,head_id=HEAD_ID,feature_identity_sha256=plan["feature_identity_sha256"],
                fold_manifest_checksum=plan["fold_manifest_checksum"],pooled_sha256=plan["medical_pooled_sha256"]),"Exact medical checkpoint identity required")
            require(digest(directory/"checkpoints/final.pt")==record["checkpoint_sha256"],"Exact final medical checkpoint bytes required")
            checkpoint=torch.load(directory/"checkpoints/final.pt",map_location="cpu",weights_only=True)
            require(checkpoint["identity"]==record["identity"] and checkpoint["counts"]==record["counts"] and checkpoint["head_id"]==HEAD_ID
                and checkpoint["checkpoint_kind"]=="final_selected" and record["status"]=="COMPLETE" and str(record["job_id"])==str(fit_launch["job_id"])
                and record["training_study_uids"]==train+(weak if arm==ARMS[1] else []) and record["held_out_study_uids"]==held
                and record["held_out_targets_masked"] and not record["candidate_promotion"],"Complete sole final fit identity/membership required")
            counts=record["counts"];initial=heads.build_head(plan["optimization"]["initialization_seed"])
            require(_model_state_sha256(initial)==counts["initial_head_state_sha256"]==plan["initial_head_state_sha256"]
                and torch.equal(torch.get_rng_state(),checkpoint["rng_state"]),"Original random initial state/RNG required")
            require(counts["successful_optimizer_steps"]==counts["attempted_batches"]==7500 and counts["skipped_batches"]==0
                and len(counts["loss_history"])==7500 and np.isfinite(counts["loss_history"]).all(),"Exact finite fixed budget required")
            require(record["targets_array_sha256"]==_array_sha256(targets.numpy()) and record["mask_array_sha256"]==_array_sha256(mask.numpy()),"Exact original fitted targets/masks required")
            prior=read(Path(plan["resnet_root"])/arm/f"fold-{fold}/run.json")
            for kind,members,offset in (("reference",train,0),("weak",weak if arm==ARMS[1] else [],34034034)):
                sequence=[members[p] for step in range(7500) for p in _batch_indices(len(members),2,step,20260912+offset)] if members else []
                key="gold_exposure_sequence_sha256" if kind=="reference" else "weak_exposure_sequence_sha256"
                supervised=sum(int(mask[index[u]].sum()) for u in sequence)
                require(counts[key]==heads.sequence_digest(sequence)==prior["counts"][key] and counts["exposures_by_source"][kind]==dict(sample_exposures=len(sequence),supervised_entry_exposures=supervised)
                    and counts["exposures_by_source"][kind]==prior["counts"]["exposures_by_source"][kind],"Original ordered source exposure/targets required")
                if kind=="weak":require(counts["distinct_weak_studies_exposed"]==len(set(sequence)) and counts["weak_training_pool_count"]==len(members),"Original weak coverage required")
            audit_optimizer(checkpoint)
            restored,_=heads.reload_head(directory/"checkpoints/final.pt",record["identity"])
            actual=predict(restored,pooled,held,index);ablated=predict(restored,heads.zero_image_features(pooled),held,index)
            require(maximum_error(actual,record["predictions"])<=1e-6 and maximum_error(ablated,record["zero_image_predictions"])<=1e-6,"Independent stored medical head replay required")
            parent=torch.load(Path(plan["resnet_root"])/arm/f"fold-{fold}/checkpoints/final.pt",map_location="cpu",weights_only=True)
            require(parent["counts"]["successful_optimizer_steps"]==7500 and parent["counts"]["initial_head_state_sha256"]==plan["initial_head_state_sha256"],"Matched parent checkpoint budget/init required")
            resnet_head=heads.build_head(0);resnet_head.load_state_dict(parent["head_state_dict"],strict=True)
            require(maximum_error(predict(resnet_head,parent_pool,held,index),prior["predictions"])<=1e-6,"Independent matched parent pooled replay required")
            medical[arm].update(record["predictions"]);zero[arm].update(record["zero_image_predictions"]);resnet[arm].update(prior["predictions"])
            checkpoint_states[fold,arm]=(checkpoint["head_state_dict"],parent["head_state_dict"])
            audit.append(dict(fold=fold,arm=arm,checkpoint_sha256=record["checkpoint_sha256"],all_targets_masks_init_exposures_adam_checked=True))
    require(len(gold)==58,"All corrected gold studies required")
    expected_analysis=analyze(plan,manifest,medical,resnet,zero)
    require(all(result[k]==v for k,v in expected_analysis.items()),"Every fixed stored comparison must independently reproduce")
    bank=read(plan["bank_plan_path"]);pixel_plan=read(plan["pixel_plan_path"]);archive=Path(pixel_plan["source_archive_path"]);stat=archive.stat()
    require(stat.st_size==pixel_plan["source_archive_bytes"] and stat.st_mtime_ns==pixel_plan["source_archive_mtime_ns"],"Original source archive stat identity required")
    pixels={r["study_instance_uid"]:r for r in read(bank["pixel_manifest_path"])["pixels"]}
    features={r["study_instance_uid"]:r for r in read(Path(bank["run_root"])/"feature-manifest.json")["features"]}
    config=read(bank["config_path"]);state=torch.load(bank["checkpoint_path"],map_location="cpu",weights_only=True)
    visual={k[7:]:v for k,v in state.items() if k.startswith("visual.")}
    tower=_build_vision_tower(config["model_cfg"]["embed_dim"],config["model_cfg"]["vision_cfg"]).eval();tower.load_state_dict(visual,strict=True)
    require(len(visual)==151 and set(tower.state_dict())==set(visual),"Strict official medical vision coverage required")
    tower.requires_grad_(False);model=torch.nn.Module();model.visual=tower;del state,visual
    mean=torch.tensor(config["preprocess_cfg"]["mean"],dtype=torch.float32).reshape(1,3,1,1);std=torch.tensor(config["preprocess_cfg"]["std"],dtype=torch.float32).reshape(1,3,1,1)
    resnet_model=build_study_model(ModelConfig.from_dict(plan["original_model_config"]));resnet_model.load_pretrained_state(torch.load(plan["resnet_weights_path"],map_location="cpu",weights_only=True));resnet_model.eval()
    native_medical=pooled.clone();native_resnet=parent_pool.clone();source_audit=[];maximum_feature_error=0.;native_slices=0
    with torch.inference_mode():
        for uid,packed,source,_,_ in iter_packed_sources(pixel_plan,sorted(gold)):
            row=pixels[uid];entry=load_pixel_entry(Path(bank["pixel_root"])/row["pixel_file"],expected_sha256=row["pixel_file_sha256"],expected_study_uid=uid,
                expected_identity_sha256=bank["pixel_identity_sha256"],expected_source_identity_sha256=source,expected_plan_sha256=bank["pixel_plan_sha256"])
            require(np.array_equal(entry.packed_images(),packed.images) and all(np.array_equal(getattr(entry,k),getattr(packed,k)) for k in ("slice_mask","series_mask","series_descriptors"))
                and entry.metadata["sampled_source_indices"]==[list(v) for v in packed.sampled_source_indices],"Every original gold pixel/mask/descriptor/order must replay")
            valid=packed.slice_mask&packed.series_mask[:,None];vectors=[]
            for image in packed.images[valid]:
                rgb=(torch.from_numpy(image[None]).repeat(1,3,1,1)-mean)/std
                vectors.append(CustomTextCLIP.encode_image(model,rgb,normalize=True).numpy()[0].copy())
            values=np.zeros((16,8,512),np.float32);values[valid]=np.stack(vectors);native_slices+=len(vectors)
            with np.load(Path(bank["feature_root"])/features[uid]["feature_file"],allow_pickle=False) as data:
                error=float(np.max(np.abs(values-data["features"])))
                require(np.isfinite(error) and error<=1e-6,"All original gold medical vision outputs must reproduce")
                maximum_feature_error=max(maximum_feature_error,error)
            masks=[torch.from_numpy(getattr(packed,k)).unsqueeze(0) for k in ("slice_mask","series_mask","series_descriptors")]
            medical_value=pool_frozen_features(torch.from_numpy(values).unsqueeze(0),*masks,feature_width=512,pooling_id=ANATOMICAL_POOLING_ID).squeeze(0)
            native_medical[index[uid]]=medical_value
            resnet_features=resnet_model.encode_packed(torch.from_numpy(packed.images).unsqueeze(0),*masks[:2])
            native_resnet[index[uid]]=resnet_model._pool_features(resnet_features,*masks).squeeze(0)
            source_audit.append(dict(study_instance_uid=uid,source_identity_sha256=source,valid_slices=len(vectors),medical_feature_max_abs_error=error))
            if len(source_audit)%8==0:print(dict(native_studies=len(source_audit),elapsed_seconds=time.monotonic()-started),flush=True)
    require({r["study_instance_uid"] for r in source_audit}==gold and native_slices==sum(pixels[u]["valid_slices"] for u in gold),"Actual complete gold native model replay required")
    native_med,native_parent,native_zero=[{a:{} for a in ARMS} for _ in range(3)];maximum_prediction_error=0.
    for fold_row in plan["folds"]:
        fold=fold_row["fold"];held=fold_row["gold_heldout_study_uids"]
        for arm in ARMS:
            med_state,parent_state=checkpoint_states[fold,arm];head=heads.build_head(0);head.load_state_dict(med_state,strict=True)
            actual=predict(head,native_medical,held,index);ablated=predict(head,heads.zero_image_features(native_medical),held,index)
            head.load_state_dict(parent_state,strict=True);parent=predict(head,native_resnet,held,index)
            errors=[maximum_error(actual,{u:medical[arm][u] for u in held}),maximum_error(ablated,{u:zero[arm][u] for u in held}),maximum_error(parent,{u:resnet[arm][u] for u in held})]
            require(max(errors)<=1e-6,"Every final medical/parent/zero original-MRI prediction must reproduce")
            maximum_prediction_error=max(maximum_prediction_error,*errors);native_med[arm].update(actual);native_zero[arm].update(ablated);native_parent[arm].update(parent)
    native_analysis=analyze(plan,manifest,native_med,native_parent,native_zero)
    require(native_analysis["comparisons"]==expected_analysis["comparisons"]
        and native_analysis["disposition"]==expected_analysis["disposition"]
        and maximum_error(native_analysis["fixed_half_blend_predictions"],expected_analysis["fixed_half_blend_predictions"])<=1e-6,
        "Native prediction ranks must reproduce every fixed comparison")
    receipt=dict(status="COMPLETE",plan_sha256=checksum,job_id=job,fit_job_id=fit_launch["job_id"],fit_allocated_cpu_node_seconds=fit_seconds,fit_terminal_sha256=fit_terminal_sha,
        analysis_sha256=digest(root/"analysis.json"),verified_final_heads=6,verified_medical_predictions=116,verified_resnet_predictions=116,
        original_mri_studies=58,original_mri_valid_slices=native_slices,maximum_medical_feature_abs_error=maximum_feature_error,maximum_prediction_abs_error=maximum_prediction_error,
        all_fitted_targets_masks_init_exposures_adam_checked=True,all_fixed_native_comparisons_reproduced=True,audits=audit,source_audit=source_audit,
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        disposition=expected_analysis["disposition"],candidate_promotion=False,gpu_seconds=0,tpu_seconds=0,external_spend_usd=0)
    write(root/"verification/independent-validation.json",receipt);signal.alarm(0)
    print(dict(status="COMPLETE",maximum_prediction_abs_error=maximum_prediction_error,elapsed_seconds=receipt["elapsed_seconds"]),flush=True)


if __name__=="__main__":main(sys.argv[1],sys.argv[2])
