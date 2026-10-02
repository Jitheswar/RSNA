"""Extract one fixed partition of unfitted projected MRI vision features."""

import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import subprocess


def digest(path):
    with Path(path).open("rb") as handle:return hashlib.file_digest(handle,"sha256").hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open("x") as handle:json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False);handle.write("\n")


def balanced_partitions(rows):
    if len({row["study_instance_uid"] for row in rows})!=len(rows) or any(not 0<row["valid_slices"]<=128 for row in rows):
        raise ValueError("Distinct valid MRI membership required")
    partitions=[dict(partition=i,study_uids=[],valid_slices=0) for i in range(4)]
    for row in sorted(rows,key=lambda r:(-r["valid_slices"],hashlib.sha256(r["study_instance_uid"].encode()).hexdigest())):
        part=min(partitions,key=lambda p:(p["valid_slices"],p["partition"]))
        part["study_uids"].append(row["study_instance_uid"]);part["valid_slices"]+=row["valid_slices"]
    return partitions


def packed_features(vectors,indices):
    import numpy as np
    if (vectors.shape!=(len(indices),512) or vectors.dtype!=np.float32 or not np.isfinite(vectors).all()
        or indices.dtype!=np.int64 or not len(indices) or np.any(indices<0) or np.any(indices>=128)
        or np.any(np.diff(indices)<=0) or np.max(np.abs(np.linalg.norm(vectors,axis=1)-1))>1e-6):
        raise ValueError("Finite ordered unit 512-D valid-slice vectors required")
    values=np.zeros((128,512),np.float32);values[indices]=vectors
    return values.reshape(16,8,512)


def run(plan_path,checksum,partition):
    started=time.monotonic();plan_path=Path(plan_path)
    if digest(plan_path)!=checksum:raise ValueError("Exact feature extraction plan required")
    plan=read(plan_path);root=Path(plan["run_root"]);job=os.environ["SLURM_JOB_ID"]
    decision=read(plan["decision_path"])
    if decision["decision"]!="approved" or decision["plan_sha256"]!=checksum:raise ValueError("Exact resource decision required")
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError("Feature extraction watchdog")))
    signal.alarm(plan["resources"]["code_watchdog_seconds"])
    directory=root/f"partition-{partition}";attempt=directory/f"job-{job}";attempt.mkdir(mode=0o700)
    allocation=subprocess.check_output(["scontrol","show","job",job,"-o"],text=True)
    (attempt/"running-allocation.txt").write_text(allocation)
    fields=dict(p.split("=",1) for p in allocation.split() if "=" in p)
    expected=dict(JobId=job,JobState="RUNNING",Partition="cluster1-cpu",NumNodes="1",NumCPUs="16",NumTasks="1",
        MinMemoryNode="64G",TimeLimit="03:00:00",OverSubscribe="OK",Requeue="0",Restarts="0")
    if any(fields.get(k)!=v for k,v in expected.items()) or "gpu" in fields.get("TRES","").lower():raise ValueError("CPU allocation differs")
    for row in plan["frozen_files"]:
        if digest(row["path"])!=row["sha256"]:raise ValueError("Frozen source/runtime/admission differs")
    import numpy as np
    import torch
    from open_clip.model import _build_vision_tower,CustomTextCLIP
    from rsna_knee.pixel_cache import load_pixel_entry
    from rsna_knee.prefix_cache import array_digest
    torch.set_num_threads(16);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available() or torch.version.cuda is not None:raise ValueError("CPU-only runtime required")
    config=read(plan["config_path"]);state=torch.load(plan["checkpoint_path"],map_location="cpu",weights_only=True)
    visual={key[7:]:value for key,value in state.items() if key.startswith("visual.")}
    tower=_build_vision_tower(config["model_cfg"]["embed_dim"],config["model_cfg"]["vision_cfg"]).eval()
    tower.load_state_dict(visual,strict=True);tower.requires_grad_(False)
    if len(visual)!=151 or set(tower.state_dict())!=set(visual):raise ValueError("Strict exact vision coverage required")
    model=torch.nn.Module();model.visual=tower;del state,visual
    mean=torch.tensor(config["preprocess_cfg"]["mean"],dtype=torch.float32).reshape(1,3,1,1)
    std=torch.tensor(config["preprocess_cfg"]["std"],dtype=torch.float32).reshape(1,3,1,1)
    source={row["study_instance_uid"]:row for row in read(plan["pixel_manifest_path"])["pixels"]}
    part=plan["partitions"][partition];records=[];slice_seconds=read_seconds=write_seconds=0.
    with torch.inference_mode():
        for uid in part["study_uids"]:
            row=source[uid];tick=time.monotonic()
            entry=load_pixel_entry(Path(plan["pixel_root"])/row["pixel_file"],expected_sha256=row["pixel_file_sha256"],
                expected_study_uid=uid,expected_identity_sha256=plan["pixel_identity_sha256"],
                expected_source_identity_sha256=row["source_identity_sha256"],expected_plan_sha256=plan["pixel_plan_sha256"])
            read_seconds+=time.monotonic()-tick
            if len(entry.pixels)!=row["valid_slices"]:raise ValueError("Valid slice membership differs")
            vectors=[];tick=time.monotonic()
            for pixels in entry.pixels:
                image=(torch.from_numpy(pixels[None]).repeat(1,3,1,1)-mean)/std
                vector=CustomTextCLIP.encode_image(model,image,normalize=True)
                if vector.shape!=(1,512) or not torch.isfinite(vector).all():raise ValueError("Vision output differs")
                vectors.append(vector.numpy()[0].copy())
            slice_seconds+=time.monotonic()-tick
            features=packed_features(np.stack(vectors),entry.valid_flat_indices)
            arrays=dict(features=features,slice_mask=entry.slice_mask,series_mask=entry.series_mask,
                series_descriptors=entry.series_descriptors,valid_flat_indices=entry.valid_flat_indices)
            hashes={name:array_digest(value) for name,value in arrays.items()}
            metadata=dict(study_instance_uid=uid,source_identity_sha256=row["source_identity_sha256"],
                source_kind=row["source_kind"],pixel_file_sha256=row["pixel_file_sha256"],pixel_array_sha256=array_digest(entry.pixels),
                feature_identity_sha256=plan["feature_identity_sha256"],plan_sha256=checksum,job_id=job,partition=partition,
                array_sha256=hashes,sampled_source_indices=entry.metadata["sampled_source_indices"])
            filename=hashlib.sha256(uid.encode()).hexdigest()+".npz";path=Path(plan["feature_root"])/filename
            tick=time.monotonic()
            with path.open("xb") as handle:np.savez(handle,**arrays,metadata=np.array(json.dumps(metadata,sort_keys=True)))
            write_seconds+=time.monotonic()-tick
            records.append(dict(study_instance_uid=uid,feature_file=filename,feature_file_sha256=digest(path),
                source_identity_sha256=row["source_identity_sha256"],source_kind=row["source_kind"],
                valid_slices=len(vectors),job_id=job,partition=partition,array_sha256=hashes))
            if len(records)%64==0:print(json.dumps(dict(partition=partition,studies=len(records),elapsed_seconds=time.monotonic()-started)),flush=True)
    if len(records)!=len(part["study_uids"]) or sum(row["valid_slices"] for row in records)!=part["valid_slices"]:
        raise ValueError("Complete fixed partition required")
    result=dict(status="COMPLETE",plan_sha256=checksum,job_id=job,partition=partition,study_count=len(records),
        valid_slices=part["valid_slices"],features=records,strict_visual_tensors=151,elapsed_seconds=time.monotonic()-started,
        read_seconds=read_seconds,inference_seconds=slice_seconds,write_seconds=write_seconds,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,labels_used=False,fitting=False,
        gpu_seconds=0,tpu_seconds=0,external_spend_usd=0,candidate_promotion=False)
    write(attempt/"partition-result.json",result);signal.alarm(0)
    print(json.dumps({k:v for k,v in result.items() if k!="features"}),flush=True)


if __name__=="__main__":run(sys.argv[1],sys.argv[2],int(sys.argv[3]))
