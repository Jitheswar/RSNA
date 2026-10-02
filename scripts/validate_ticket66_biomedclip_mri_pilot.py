"""Fresh-process official vision replay from original DICOMs, inside the pilot allocation."""

import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time


def digest(path):
    with Path(path).open("rb") as handle:return hashlib.file_digest(handle,"sha256").hexdigest()


def main(root,checksum):
    started=time.monotonic();root=Path(root)
    plan=json.loads((root/"plan.json").read_text());result=json.loads((root/"pilot-result.json").read_text())
    if digest(root/"plan.json")!=checksum or result["plan_sha256"]!=checksum or result["status"]!="COMPLETE":
        raise ValueError("Exact complete pilot result required")
    remaining=int(float(os.environ["RSNA_TICKET66_DEADLINE_UNIX"])-time.time())
    if remaining<=0:raise TimeoutError("Whole allocation code watchdog reached")
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError("Independent replay watchdog")))
    signal.alarm(remaining)
    import numpy as np
    import torch
    from open_clip.model import _build_vision_tower,CustomTextCLIP
    from rsna_knee.real_inference import load_integration_manifest,iter_real_studies
    from rsna_knee.preprocessing import prepare_study,pack_study
    from rsna_knee.pixel_cache import load_pixel_entry
    from rsna_knee.prefix_cache import array_digest
    torch.set_num_threads(16);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available() or torch.version.cuda is not None:raise ValueError("CPU-only native replay required")
    for row in plan["frozen_files"]:
        if digest(row["path"])!=row["sha256"]:raise ValueError("Frozen source/runtime/input changed")
    config=json.loads(Path(plan["config_path"]).read_text())
    state=torch.load(plan["checkpoint_path"],map_location="cpu",weights_only=True)
    visual={key[7:]:value for key,value in state.items() if key.startswith("visual.")}
    model=_build_vision_tower(config["model_cfg"]["embed_dim"],config["model_cfg"]["vision_cfg"]).eval()
    model.load_state_dict(visual,strict=True);model.requires_grad_(False)
    if len(visual)!=151 or set(model.state_dict())!=set(visual):raise ValueError("Official vision state coverage differs")
    shell=torch.nn.Module();shell.visual=model;del state,visual
    integration=load_integration_manifest(plan["integration_path"])
    if integration.manifest_checksum!=plan["integration_manifest_checksum"]:raise ValueError("Original MRI policy differs")
    archive=Path(plan["source_archive_path"])
    if archive.stat().st_size!=plan["source_archive_bytes"] or archive.stat().st_mtime_ns!=plan["source_archive_mtime_ns"]:
        raise ValueError("Original archive differs")
    rows={row["study_instance_uid"]:row for row in plan["selected_pixels"]}
    records={row["study_instance_uid"]:row for row in result["records"]}
    if set(rows)!=set(records) or len(rows)!=8:raise ValueError("Complete fixed output cohort required")
    mean=torch.tensor(plan["representation"]["mean"],dtype=torch.float32).reshape(1,3,1,1)
    std=torch.tensor(plan["representation"]["std"],dtype=torch.float32).reshape(1,3,1,1)
    maximum_error=0.;verified_slices=0;verified=[]
    with torch.inference_mode():
        for loaded in iter_real_studies(integration,archive,require_complete_series=True,study_uids=plan["selected_study_uids"]):
            uid=loaded.study.study_instance_uid;row=rows[uid];record=records[uid]
            prepared=prepare_study(loaded.study,integration.preprocess_config)
            if prepared.failures:raise ValueError("Independent original DICOM preparation failed")
            packed=pack_study(loaded.study,prepared.series,integration.preprocess_config,packing_config=integration.packing_config)
            entry=load_pixel_entry(Path(plan["pixel_root"])/row["pixel_file"],expected_sha256=row["pixel_file_sha256"],
                expected_study_uid=uid,expected_identity_sha256=plan["pixel_identity_sha256"],
                expected_source_identity_sha256=row["source_identity_sha256"],expected_plan_sha256=plan["pixel_plan_sha256"])
            if not np.array_equal(packed.images,entry.packed_images()) or array_digest(entry.pixels)!=record["pixel_array_sha256"]:
                raise ValueError("Independent native pixel identity differs")
            output=Path(record["output_path"])
            if digest(output)!=record["output_sha256"]:raise ValueError("Published pilot output changed")
            with np.load(output,allow_pickle=False) as data:
                for name,expected in [("slice_mask",packed.slice_mask),("series_mask",packed.series_mask),("series_descriptors",packed.series_descriptors),("valid_flat_indices",entry.valid_flat_indices)]:
                    if not np.array_equal(data[name],expected):raise ValueError("Output masks/descriptor/order differs")
                features=data["features"]
                if features.shape!=(16,8,512) or features.dtype!=np.float32 or not np.isfinite(features).all():
                    raise ValueError("Complete finite 512-D packed output required")
                flat=features.reshape(128,512);valid=entry.valid_flat_indices
                invalid=np.ones(128,bool);invalid[valid]=False
                if np.any(flat[invalid]) or float(np.var(flat[valid],axis=0).sum())<=0:
                    raise ValueError("Invalid padding or constant image features")
                native=packed.images.reshape(128,1,224,224)
                for index in valid:
                    image=(torch.from_numpy(native[index:index+1].copy()).repeat(1,3,1,1)-mean)/std
                    expected=CustomTextCLIP.encode_image(shell,image,normalize=True).numpy()[0]
                    error=float(np.max(np.abs(flat[index]-expected)))
                    if error>plan["parity_tolerance"] or abs(float(np.linalg.norm(flat[index]))-1)>1e-6:
                        raise ValueError("Independent original-MRI official-output replay differs")
                    maximum_error=max(maximum_error,error);verified_slices+=1
            verified.append(dict(study_instance_uid=uid,output_sha256=digest(output),native_replay=True))
    if len(verified)!=8 or verified_slices!=plan["expected_valid_slices"]:raise ValueError("Incomplete native replay")
    receipt=dict(status="COMPLETE",plan_sha256=checksum,result_sha256=digest(root/"pilot-result.json"),
        job_id=os.environ["SLURM_JOB_ID"],verified_studies=8,verified_slices=verified_slices,strict_visual_tensors=151,
        maximum_native_output_abs_error=maximum_error,original_dicom_replay=True,outputs=verified,
        elapsed_seconds=time.monotonic()-started,model_text_tower=False,labels_used=False,fitting=False,candidate_promotion=False)
    with (root/"independent-validation.json").open("x") as handle:json.dump(receipt,handle,indent=2,sort_keys=True);handle.write("\n")
    signal.alarm(0);print(json.dumps({k:v for k,v in receipt.items() if k!="outputs"}))


if __name__=="__main__":main(sys.argv[1],sys.argv[2])
