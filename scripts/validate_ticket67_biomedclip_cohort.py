"""Validate every frozen feature entry and independently replay the fixed MRI pilot."""

import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import subprocess
import time


def digest(path):
    with Path(path).open("rb") as handle:return hashlib.file_digest(handle,"sha256").hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open("x") as handle:json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False);handle.write("\n")


def terminal(directory,job,time_limit):
    path=directory/f"controller-{job}";observation=read(path/"terminal-observation.json")
    fields=dict(p.split("=",1) for p in (path/"terminal-controller.txt").read_text().split() if "=" in p)
    expected=dict(JobId=str(job),JobState="COMPLETED",ExitCode="0:0",Partition="cluster1-cpu",NumNodes="1",NumCPUs="16",NumTasks="1",
        MinMemoryNode="64G",TimeLimit=time_limit,OverSubscribe="OK",Requeue="0",Restarts="0")
    if (str(observation["job_id"])!=str(job) or observation["state"]!="COMPLETED" or observation["returncode"]!=0
        or any(fields.get(k)!=v for k,v in expected.items()) or "gpu" in fields.get("TRES","").lower()):
        raise ValueError("Exact successful CPU terminal required")
    hours,minutes,seconds=map(int,fields["RunTime"].split(":"));elapsed=hours*3600+minutes*60+seconds
    return elapsed,digest(path/"terminal-controller.txt")


def main(plan_path,checksum):
    started=time.monotonic();plan_path=Path(plan_path)
    if digest(plan_path)!=checksum:raise ValueError("Exact cohort verification plan required")
    plan=read(plan_path);root=Path(plan["run_root"]);output=root/"verification";job=os.environ["SLURM_JOB_ID"]
    decision=read(plan["decision_path"])
    if decision["decision"]!="approved" or decision["plan_sha256"]!=checksum:raise ValueError("Exact resource admission required")
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError("Cohort verification watchdog")))
    signal.alarm(plan["resources"]["verification_watchdog_seconds"])
    allocation=subprocess.check_output(["scontrol","show","job",job,"-o"],text=True);(output/"running-allocation.txt").write_text(allocation)
    fields=dict(p.split("=",1) for p in allocation.split() if "=" in p)
    if (any(fields.get(k)!=v for k,v in dict(JobId=job,JobState="RUNNING",Partition="cluster1-cpu",NumNodes="1",NumCPUs="16",NumTasks="1",MinMemoryNode="64G",TimeLimit="01:00:00",OverSubscribe="OK",Requeue="0",Restarts="0").items())
        or "gpu" in fields.get("TRES","").lower()):
        raise ValueError("Actual verification allocation differs")
    for row in plan["frozen_files"]:
        if digest(row["path"])!=row["sha256"]:raise ValueError("Frozen verification dependency changed")
    import numpy as np
    import torch
    from open_clip.model import _build_vision_tower,CustomTextCLIP
    from rsna_knee.pixel_cache import load_pixel_entry
    from rsna_knee.prefix_cache import array_digest
    from rsna_knee.feature_pooling import pool_frozen_features,ANATOMICAL_POOLING_ID
    torch.set_num_threads(16);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available() or torch.version.cuda is not None:raise ValueError("CPU-only independent replay required")
    rows=[];terminals=[]
    for part in plan["partitions"]:
        directory=root/f"partition-{part['partition']}";launch=read(directory/"accepted-launch.json");part_job=launch["job_id"]
        if launch["plan_sha256"]!=checksum or launch["launch_attempt"]!=1:raise ValueError("Exact sole extraction attempt required")
        seconds,sha=terminal(directory,part_job,"03:00:00")
        if seconds>10800:raise ValueError("Extraction exceeded admitted allocation")
        result=read(directory/f"job-{part_job}/partition-result.json")
        if (result["status"]!="COMPLETE" or result["plan_sha256"]!=checksum or str(result["job_id"])!=str(part_job)
            or result["partition"]!=part["partition"] or result["valid_slices"]!=part["valid_slices"]
            or {r["study_instance_uid"] for r in result["features"]}!=set(part["study_uids"])
            or result["study_count"]!=len(part["study_uids"])):raise ValueError("Complete partition identity differs")
        terminals.append(dict(job_id=part_job,allocated_cpu_node_seconds=seconds,terminal_sha256=sha,result_sha256=digest(directory/f"job-{part_job}/partition-result.json")))
        rows.extend(result["features"])
    source={row["study_instance_uid"]:row for row in read(plan["pixel_manifest_path"])["pixels"]}
    if len(rows)!=4228 or len({row["study_instance_uid"] for row in rows})!=4228 or set(source)!={r["study_instance_uid"] for r in rows}:
        raise ValueError("Complete exact cohort membership required")
    if {p.name for p in Path(plan["feature_root"]).glob("*.npz")}!={r["feature_file"] for r in rows}:
        raise ValueError("Feature directory contains missing or extra entries")
    pilot_root=Path(plan["pilot_root"]);pilot=read(pilot_root/"pilot-result.json")
    selected={r["study_instance_uid"]:r for r in pilot["records"]}
    config=read(plan["config_path"]);state=torch.load(plan["checkpoint_path"],map_location="cpu",weights_only=True)
    visual={k[7:]:v for k,v in state.items() if k.startswith("visual.")}
    tower=_build_vision_tower(config["model_cfg"]["embed_dim"],config["model_cfg"]["vision_cfg"]).eval();tower.load_state_dict(visual,strict=True)
    if len(visual)!=151 or set(tower.state_dict())!=set(visual):raise ValueError("Strict vision coverage differs")
    tower.requires_grad_(False);model=torch.nn.Module();model.visual=tower;del state,visual
    mean=torch.tensor(config["preprocess_cfg"]["mean"],dtype=torch.float32).reshape(1,3,1,1)
    std=torch.tensor(config["preprocess_cfg"]["std"],dtype=torch.float32).reshape(1,3,1,1)
    pooled=[];ordered=[];total=0;replayed=0;maximum_error=0.;read_seconds=0.
    with torch.inference_mode():
        for row in sorted(rows,key=lambda r:r["study_instance_uid"]):
            uid=row["study_instance_uid"];prior=source[uid];path=Path(plan["feature_root"])/row["feature_file"]
            if digest(path)!=row["feature_file_sha256"]:raise ValueError("Published feature bytes changed")
            tick=time.monotonic()
            entry=load_pixel_entry(Path(plan["pixel_root"])/prior["pixel_file"],expected_sha256=prior["pixel_file_sha256"],
                expected_study_uid=uid,expected_identity_sha256=plan["pixel_identity_sha256"],
                expected_source_identity_sha256=prior["source_identity_sha256"],expected_plan_sha256=plan["pixel_plan_sha256"])
            read_seconds+=time.monotonic()-tick
            with np.load(path,allow_pickle=False) as data:
                metadata=json.loads(str(data["metadata"]));features=data["features"];valid=entry.slice_mask&entry.series_mask[:,None]
                if (features.shape!=(16,8,512) or features.dtype!=np.float32 or not np.isfinite(features).all()
                    or np.any(features[~valid]) or int(valid.sum())!=row["valid_slices"]
                    or np.max(np.abs(np.linalg.norm(features[valid],axis=1)-1))>1e-6):raise ValueError("Feature numerics, normalization or padding differ")
                for name,expected in dict(slice_mask=entry.slice_mask,series_mask=entry.series_mask,series_descriptors=entry.series_descriptors,valid_flat_indices=entry.valid_flat_indices).items():
                    if not np.array_equal(data[name],expected):raise ValueError("Unfitted source masks/descriptors/order differ")
                for name in ["features","slice_mask","series_mask","series_descriptors","valid_flat_indices"]:
                    if array_digest(data[name])!=row["array_sha256"][name] or metadata["array_sha256"][name]!=row["array_sha256"][name]:raise ValueError("Feature semantic array digest differs")
                if (metadata["study_instance_uid"]!=uid or metadata["plan_sha256"]!=checksum
                    or metadata["feature_identity_sha256"]!=plan["feature_identity_sha256"]
                    or metadata["source_identity_sha256"]!=prior["source_identity_sha256"]
                    or metadata["source_kind"]!=prior["source_kind"] or metadata["pixel_file_sha256"]!=prior["pixel_file_sha256"]
                    or metadata["pixel_array_sha256"]!=array_digest(entry.pixels)
                    or str(metadata["job_id"])!=str(row["job_id"]) or metadata["partition"]!=row["partition"]
                    or metadata["sampled_source_indices"]!=entry.metadata["sampled_source_indices"]):raise ValueError("Feature/source/execution identity differs")
                value=pool_frozen_features(torch.from_numpy(features).unsqueeze(0),torch.from_numpy(entry.slice_mask).unsqueeze(0),
                    torch.from_numpy(entry.series_mask).unsqueeze(0),torch.from_numpy(entry.series_descriptors).unsqueeze(0),feature_width=512,pooling_id=ANATOMICAL_POOLING_ID).numpy()[0]
                if value.shape!=(2076,) or not np.isfinite(value).all():raise ValueError("Unfitted anatomical pooled representation differs")
                pooled.append(value.copy());ordered.append(uid);total+=len(entry.pixels)
                if uid in selected:
                    if digest(selected[uid]["output_path"])!=selected[uid]["output_sha256"]:
                        raise ValueError("Independently replayed original-DICOM pilot bytes changed")
                    with np.load(selected[uid]["output_path"],allow_pickle=False) as native:
                        error=float(np.max(np.abs(features-native["features"])))
                        if not np.isfinite(error) or error>1e-6:raise ValueError("Fixed original-DICOM pilot output differs")
                        maximum_error=max(maximum_error,error)
                    for index,pixels in zip(entry.valid_flat_indices,entry.pixels):
                        image=(torch.from_numpy(pixels[None]).repeat(1,3,1,1)-mean)/std
                        expected=CustomTextCLIP.encode_image(model,image,normalize=True).numpy()[0]
                        error=float(np.max(np.abs(features.reshape(128,512)[index]-expected)))
                        if not np.isfinite(error) or error>1e-6:raise ValueError("Fresh-process official encoder replay differs")
                        maximum_error=max(maximum_error,error);replayed+=1
            if len(ordered)%256==0:print(json.dumps(dict(verified_studies=len(ordered),elapsed_seconds=time.monotonic()-started)),flush=True)
    if total!=187272 or replayed!=344:raise ValueError("Complete image and fixed native replay coverage required")
    matrix=np.stack(pooled).astype(np.float32,copy=False)
    with (root/"pooled-features.npz").open("xb") as handle:np.savez(handle,pooled=matrix,study_uids=np.array(ordered))
    manifest=dict(status="COMPLETE",schema_version="rsna-ticket67-biomedclip-feature-cohort-v1",plan_sha256=checksum,
        feature_identity_sha256=plan["feature_identity_sha256"],study_count=4228,valid_slices=total,feature_dimension=512,
        features=sorted(rows,key=lambda r:r["study_instance_uid"]),pooled_file=str(root/"pooled-features.npz"),pooled_file_sha256=digest(root/"pooled-features.npz"),pooled_array_sha256=array_digest(matrix),
        pooling_id=ANATOMICAL_POOLING_ID,pooled_width=2076,labels_used=False,fitting=False)
    write(root/"feature-manifest.json",manifest)
    receipt=dict(status="COMPLETE",job_id=job,plan_sha256=checksum,manifest_sha256=digest(root/"feature-manifest.json"),
        verified_studies=4228,verified_valid_slices=total,replayed_fixed_native_slices=replayed,maximum_replay_abs_error=maximum_error,
        original_source_pixel_proof="Ticket 62 all 4228 exact source replays; Ticket 66 fixed eight original DICOM/model replays",
        actual_full_native_vision_replay_all_studies=False,all_feature_bytes_masks_norms_sources_checked=True,
        cache_read_seconds=read_seconds,elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        extraction_terminals=terminals,labels_used=False,fitting=False,gpu_seconds=0,tpu_seconds=0,external_spend_usd=0,candidate_promotion=False)
    write(output/"independent-validation.json",receipt);signal.alarm(0)
    print(json.dumps({k:v for k,v in receipt.items() if k!="extraction_terminals"}),flush=True)


if __name__=="__main__":main(sys.argv[1],sys.argv[2])
