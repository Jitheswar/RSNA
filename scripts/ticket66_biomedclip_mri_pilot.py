"""One strict offline vision-load and original-MRI parity/cost admission."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import zipfile

PROJECT = Path(os.environ.get("RSNA_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
ROOT = PROJECT / "artifacts/private/ticket66-biomedclip-mri-pilot-v1"
PIXELS = PROJECT / "artifacts/private/ticket62-mri-pixel-cohort-v1"
ASSET = PROJECT / "artifacts/private/ticket65-biomedclip-asset-admission-v1"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket66-biomedclip-mri-pilot-20261002.json"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def select_studies(rows):
    eligible = [row for row in rows if row["source_kind"] == "ticket35-prepared-series-cache-read-only"]
    selected = sorted(eligible, key=lambda row: hashlib.sha256(row["study_instance_uid"].encode()).hexdigest())[:8]
    if len(selected) != 8 or len({row["study_instance_uid"] for row in selected}) != 8:
        raise ValueError("Exactly eight distinct report-only MRI studies required")
    return selected


def prepare():
    import open_clip
    import timm
    import torch
    import torchvision
    import pydicom
    import numpy
    from rsna_knee.real_inference import load_integration_manifest

    if (ROOT / "plan.json").exists() or DECISION.exists():
        raise FileExistsError("Pilot already frozen")
    if read(ROOT / "runtime-acquisition.json")["status"] != "COMPLETE" or read(ASSET / "independent-asset-verification.json")["status"] != "COMPLETE":
        raise ValueError("Actual runtime and independent asset acquisition required")
    source = Path(open_clip.__file__).parent
    research = PROJECT / "artifacts/private/medical-image-encoder-options-20261002"
    for name in ("model.py", "timm_model.py"):
        if digest(source / name) != digest(research / f"openclip223--{name}"):
            raise ValueError("Official OpenCLIP vision source differs")
    prior = read(PIXELS / "plan.json"); manifest=read(PIXELS / "pixel-manifest.json")
    if manifest["status"] != "COMPLETE" or manifest["study_count"] != 4228:
        raise ValueError("Complete independently verified pixel cohort required")
    selected = select_studies(manifest["pixels"])
    integration = load_integration_manifest(prior["integration_path"])
    if integration.manifest_checksum != prior["integration_manifest_checksum"]:
        raise ValueError("Original MRI manifest identity differs")
    selected_uids = [row["study_instance_uid"] for row in selected]
    archive=Path(prior["source_archive_path"])
    if archive.stat().st_size!=prior["source_archive_bytes"] or archive.stat().st_mtime_ns!=prior["source_archive_mtime_ns"]:
        raise ValueError("Original MRI archive identity differs")
    with zipfile.ZipFile(archive) as source_archive:
        for study in integration.studies:
            if study.study_instance_uid in selected_uids:
                for series in study.series:
                    for name in series.entries:
                        if source_archive.getinfo(name).file_size<=0:
                            raise ValueError("Selected original MRI source entry absent or empty")
    snapshot = ROOT / "source-snapshot"
    shutil.copytree(PROJECT / "src/rsna_knee", snapshot / "src/rsna_knee",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(__file__, snapshot / Path(__file__).name)
    fixed = [AUTH, PIXELS / "plan.json", PIXELS / "pixel-manifest.json", Path(prior["integration_path"]),
        ASSET / "plan.json", ASSET / "result.json", ASSET / "independent-asset-verification.json",
        ASSET / "evidence/open_clip_config.json", ASSET / "evidence/LICENSE.md", ROOT / "runtime-import-validation.json",
        Path(read(ASSET / "plan.json")["checkpoint_path"]), ROOT / "runtime-plan.json", ROOT / "runtime-acquisition.json",
        PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket66-runtime-dependency-addendum-20261002.json"]
    fixed.extend(Path(prior["pixel_root"]) / row["pixel_file"] for row in selected)
    fixed.extend(path for path in snapshot.rglob("*") if path.is_file())
    sites = [Path(open_clip.__file__).parent.parent, Path(timm.__file__).parent.parent,
             Path(torch.__file__).parent.parent]
    for site, packages in zip(sites, [("open_clip", "transformers", "tokenizers", "ftfy", "wcwidth", "sentencepiece"),
                                    ("timm", "torchvision", "PIL"), ("torch", "numpy", "pydicom")]):
        for package in packages:
            fixed.extend(path for path in (site/package).rglob("*") if path.is_file()
                         and "__pycache__" not in path.parts and path.suffix != ".pyc")
    fixed += sorted((ROOT / "wheels").glob("*.whl"))
    fixed.append(ROOT / "runtime/lib/python3.11/site-packages/rsna-admitted-cpu-dependencies.pth")
    versions = {name:importlib.metadata.version(name) for name in
        ("open_clip_torch", "transformers", "tokenizers", "timm", "torch", "torchvision", "pillow", "numpy", "pydicom", "ftfy", "wcwidth")}
    if versions["open_clip_torch"] != "2.23.0" or versions["transformers"] != "4.35.2":
        raise ValueError("Published dependency identities required")
    config = read(ASSET / "evidence/open_clip_config.json")
    plan = dict(schema_version="rsna-ticket66-biomedclip-mri-pilot-v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_snapshot=str(snapshot), selected_study_uids=selected_uids, selected_pixels=selected,
        selection="first eight report-only studies by ascending SHA256(UID), no labels used",
        expected_valid_slices=sum(row["valid_slices"] for row in selected), integration_path=prior["integration_path"],
        integration_manifest_checksum=integration.manifest_checksum, source_archive_path=prior["source_archive_path"],
        source_archive_bytes=prior["source_archive_bytes"], source_archive_mtime_ns=prior["source_archive_mtime_ns"],
        pixel_root=prior["pixel_root"], pixel_plan_sha256=digest(PIXELS / "plan.json"), pixel_identity_sha256=prior["pixel_identity_sha256"],
        checkpoint_path=read(ASSET / "plan.json")["checkpoint_path"], config_path=str(ASSET / "evidence/open_clip_config.json"),
        representation=dict(image_size=224, grayscale_to_rgb="repeat", crop=False, microbatch=1, precision="float32",
            mean=config["preprocess_cfg"]["mean"], std=config["preprocess_cfg"]["std"], output_dimension=512,
            output="learned projected vector, explicit L2 normalize", preserve_masks_and_descriptors=True),
        parity_tolerance=1e-6, runtime_versions=versions,
        resources=dict(partition="cluster1-cpu", nodes=1, tasks=1, cpus_per_task=16, memory_gib=64,
            walltime_seconds=960, code_watchdog_seconds=900, maximum_launch_attempts=1, maximum_owner_cpu_jobs=2,
            exclusive=False, requeue=False, gpu_count=0, external_spend_usd=0),
        frozen_files=[dict(path=str(p),sha256=digest(p)) for p in sorted(set(fixed))],
        network_during_execution=False, labels_used=False, fitting=False, model_text_tower=False,
        full_extraction_authorized=False, candidate_promotion=False, competition_submission=False)
    write(ROOT / "plan.json", plan)
    checksum=digest(ROOT / "plan.json")
    write(DECISION, dict(schema_version="rsna-campaign-autonomous-decision-v1",decision="approved",
        decision_made_by="campaign agent under standing owner authorization",authorization_sha256=digest(AUTH),plan_sha256=checksum,
        resources=plan["resources"],new_exact_owner_numeric_approval_claimed=False,gpu_ledger_changed=False,tpu_ledger_changed=False,
        basis="Independent pinned medical checkpoint and isolated official runtime; measure strict vision/native MRI parity and CPU cost before any full extraction or predictive experiment.",
        temporary_fit_coordinator_handoff_allowed=True,additional_calendar_reserve_seconds=960,
        full_extraction_authorized=False,fitting_authorized=False,candidate_promotion=False))
    (ROOT / "run.sbatch").write_text(f"""#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket66-biomedclip-pilot
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:16:00
#SBATCH --no-requeue
#SBATCH --output={ROOT}/slurm-%j.out
#SBATCH --error={ROOT}/slurm-%j.err
#SBATCH --chdir={PROJECT}
set -euo pipefail
umask 077
export RSNA_PROJECT_ROOT={PROJECT}
export PYTHONPATH={snapshot}/src
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
{ROOT}/runtime/bin/python {snapshot}/{Path(__file__).name} run {checksum}
""")
    print(json.dumps(dict(plan_sha256=checksum,selected_studies=8,valid_slices=plan["expected_valid_slices"],frozen_files=len(plan["frozen_files"]))))


def run(checksum):
    started=time.monotonic()
    if digest(ROOT / "plan.json") != checksum or read(DECISION)["plan_sha256"] != checksum:
        raise ValueError("Exact pilot admission required")
    plan=read(ROOT / "plan.json");job=os.environ["SLURM_JOB_ID"]
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError("BiomedCLIP pilot watchdog")))
    signal.alarm(plan["resources"]["code_watchdog_seconds"])
    allocation=subprocess.check_output(["scontrol","show","job",job,"-o"],text=True)
    (ROOT / "running-allocation.txt").write_text(allocation)
    fields=dict(item.split("=",1) for item in allocation.split() if "=" in item)
    expected=dict(JobId=job,JobState="RUNNING",Partition="cluster1-cpu",NumNodes="1",NumCPUs="16",NumTasks="1",MinMemoryNode="64G",TimeLimit="00:16:00",OverSubscribe="OK",Requeue="0",Restarts="0")
    if any(fields.get(k)!=v for k,v in expected.items()) or "gpu" in fields.get("TRES","").lower():
        raise ValueError("Actual allocation differs from CPU admission")
    for row in plan["frozen_files"]:
        if digest(row["path"]) != row["sha256"]:raise ValueError("Frozen pilot input/runtime/source changed")
    archive=Path(plan["source_archive_path"])
    if archive.stat().st_size!=plan["source_archive_bytes"] or archive.stat().st_mtime_ns!=plan["source_archive_mtime_ns"]:
        raise ValueError("Original MRI archive identity changed")
    import numpy as np
    import torch
    from open_clip.model import _build_vision_tower,CustomTextCLIP
    from open_clip.timm_model import TimmModel
    from rsna_knee.real_inference import load_integration_manifest,iter_real_studies
    from rsna_knee.preprocessing import prepare_study,pack_study
    from rsna_knee.pixel_cache import load_pixel_entry
    from rsna_knee.prefix_cache import array_digest

    torch.set_num_threads(16);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available() or torch.version.cuda is not None:raise ValueError("CPU-only runtime required")
    load_started=time.monotonic();config=read(plan["config_path"])
    state=torch.load(plan["checkpoint_path"],map_location="cpu",weights_only=True)
    visual_state={key.removeprefix("visual."):value for key,value in state.items() if key.startswith("visual.")}
    if len(visual_state)!=151:raise ValueError("Exact vision namespace required")
    cfg=config["model_cfg"]["vision_cfg"]
    model=TimmModel(cfg["timm_model_name"],config["model_cfg"]["embed_dim"],image_size=cfg["image_size"],
        pool=cfg["timm_pool"],proj=cfg["timm_proj"],pretrained=False).eval()
    reference=_build_vision_tower(config["model_cfg"]["embed_dim"],cfg).eval()
    for tower in (model,reference):
        tower.load_state_dict(visual_state,strict=True)
        if set(tower.state_dict())!=set(visual_state):raise ValueError("Strict model coverage differs")
        tower.requires_grad_(False)
    shell=torch.nn.Module();shell.visual=reference
    load_seconds=time.monotonic()-load_started;del state,visual_state
    mean=torch.tensor(plan["representation"]["mean"],dtype=torch.float32).reshape(1,3,1,1)
    std=torch.tensor(plan["representation"]["std"],dtype=torch.float32).reshape(1,3,1,1)
    integration=load_integration_manifest(plan["integration_path"])
    if integration.manifest_checksum!=plan["integration_manifest_checksum"]:raise ValueError("Native MRI policy differs")
    rows={row["study_instance_uid"]:row for row in plan["selected_pixels"]}
    outputs=ROOT / "outputs";outputs.mkdir(mode=0o700)
    records=[];slice_times=[];native_times=[];max_error=0.;max_repeat_error=0.
    with torch.inference_mode():
        iterator=iter_real_studies(integration,archive,require_complete_series=True,study_uids=plan["selected_study_uids"])
        for _ in range(8):
            tick=time.monotonic();loaded=next(iterator);uid=loaded.study.study_instance_uid;row=rows[uid]
            prepared=prepare_study(loaded.study,integration.preprocess_config)
            if prepared.failures:raise ValueError("Native MRI preparation failed")
            packed=pack_study(loaded.study,prepared.series,integration.preprocess_config,packing_config=integration.packing_config)
            entry=load_pixel_entry(Path(plan["pixel_root"])/row["pixel_file"],expected_sha256=row["pixel_file_sha256"],
                expected_study_uid=uid,expected_identity_sha256=plan["pixel_identity_sha256"],
                expected_source_identity_sha256=row["source_identity_sha256"],expected_plan_sha256=plan["pixel_plan_sha256"])
            if (not np.array_equal(packed.images,entry.packed_images()) or not np.array_equal(packed.slice_mask,entry.slice_mask)
                or not np.array_equal(packed.series_mask,entry.series_mask) or not np.array_equal(packed.series_descriptors,entry.series_descriptors)
                or entry.metadata["sampled_source_indices"]!=[list(value) for value in packed.sampled_source_indices]):
                raise ValueError("Original DICOM/cache pixels or packing differ")
            native_times.append(time.monotonic()-tick)
            values=[];study_error=0.;study_repeat=0.
            native=packed.images.reshape(128,1,224,224)[entry.valid_flat_indices]
            for index,pixels in enumerate(entry.pixels):
                tick=time.monotonic();x=(torch.from_numpy(pixels[None]).repeat(1,3,1,1)-mean)/std
                value=torch.nn.functional.normalize(model(x),dim=-1)
                slice_times.append(time.monotonic()-tick)
                expected_x=(torch.from_numpy(native[index:index+1].copy()).repeat(1,3,1,1)-mean)/std
                target=CustomTextCLIP.encode_image(shell,expected_x,normalize=True)
                repeated=torch.nn.functional.normalize(model(x),dim=-1)
                error=float((value-target).abs().max());repeat=float((value-repeated).abs().max())
                max_error=max(max_error,error);max_repeat_error=max(max_repeat_error,repeat)
                study_error=max(study_error,error);study_repeat=max(study_repeat,repeat)
                if (value.shape!=(1,512) or not torch.isfinite(value).all() or error>plan["parity_tolerance"]
                    or repeat>plan["parity_tolerance"] or abs(float(value.norm())-1)>1e-6):
                    raise ValueError("Vision output, repeatability or native/reference parity failed")
                values.append(value.numpy()[0].copy())
            features=np.zeros((128,512),np.float32);features[entry.valid_flat_indices]=np.stack(values)
            output=outputs/(hashlib.sha256(uid.encode()).hexdigest()+".npz")
            np.savez(output,features=features.reshape(16,8,512),slice_mask=entry.slice_mask,series_mask=entry.series_mask,
                series_descriptors=entry.series_descriptors,valid_flat_indices=entry.valid_flat_indices)
            records.append(dict(study_instance_uid=uid,valid_slices=len(values),output_path=str(output),output_sha256=digest(output),
                pixel_array_sha256=array_digest(entry.pixels),native_pixel_replay_exact=True,
                feature_variance=float(np.var(values,axis=0).sum()),max_reference_abs_error=study_error,max_repeat_abs_error=study_repeat))
            print(json.dumps(dict(completed_studies=len(records),elapsed_seconds=time.monotonic()-started)),flush=True)
        if next(iterator,None) is not None:raise ValueError("Unexpected additional native source study")
    if len(records)!=8 or sum(row["valid_slices"] for row in records)!=plan["expected_valid_slices"] or not all(row["feature_variance"]>0 for row in records):
        raise ValueError("Complete nonconstant fixed cohort output required")
    result=dict(status="COMPLETE",plan_sha256=checksum,job_id=job,records=records,study_count=8,
        valid_slices=len(slice_times),strict_visual_tensor_count=151,strict_state_coverage=True,
        output_dimension=512,native_pixel_replay_exact=True,max_reference_abs_error=max_error,max_repeat_abs_error=max_repeat_error,
        vision_load_seconds=load_seconds,median_valid_slice_seconds=float(np.median(slice_times)),
        maximum_valid_slice_seconds=max(slice_times),timed_inference_seconds=sum(slice_times),native_read_pack_seconds=sum(native_times),
        maximum_native_read_pack_seconds=max(native_times),full_cohort_valid_slices=187272,
        full_model_inference_forecast_seconds=2*(load_seconds+187272*max(slice_times)),
        full_native_extraction_forecast_seconds=2*(load_seconds+187272*max(slice_times)+4228*max(native_times)),
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        fitting=False,labels_used=False,text_tower_constructed=False,network=False,gpu_seconds=0,tpu_seconds=0,
        external_spend_usd=0,candidate_promotion=False,full_extraction_authorized=False)
    write(ROOT / "pilot-result.json",result);signal.alarm(0)
    print(json.dumps({k:v for k,v in result.items() if k!="records"}),flush=True)


if __name__=="__main__":
    if sys.argv[1]=="prepare":prepare()
    elif sys.argv[1]=="run":run(sys.argv[2])
    else:raise ValueError("prepare or run required")
