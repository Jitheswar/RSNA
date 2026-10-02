"""Admit the exact unfitted medical encoder cohort from measured complete inputs."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from ticket67_biomedclip_extract import balanced_partitions

PROJECT=Path(os.environ.get("RSNA_PROJECT_ROOT",Path(__file__).resolve().parents[1]))
ROOT=PROJECT/"artifacts/private/ticket67-biomedclip-cohort-v1"
PILOT=PROJECT/"artifacts/private/ticket66-biomedclip-mri-pilot-v1"
PIXELS=PROJECT/"artifacts/private/ticket62-mri-pixel-cohort-v1"
AUTH=PROJECT/".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
DECISION=PROJECT/".scratch/rsna-leaderboard/resource-approvals/ticket67-biomedclip-cohort-20261002.json"


def digest(path):
    with Path(path).open("rb") as handle:return hashlib.file_digest(handle,"sha256").hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open("x") as handle:json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False);handle.write("\n")


def main():
    if ROOT.exists() or DECISION.exists():raise FileExistsError("Cohort admission already exists")
    pilot_plan=read(PILOT/"plan.json");pilot=read(PILOT/"pilot-result.json");replay=read(PILOT/"independent-validation.json");terminal=read(PILOT/"terminal-verification.json")
    if (not terminal["passed"] or terminal["status"]!="COMPLETE" or replay["maximum_native_output_abs_error"]!=0
        or terminal["result_sha256"]!=digest(PILOT/"pilot-result.json") or terminal["independent_validation_sha256"]!=digest(PILOT/"independent-validation.json")
        or replay["verified_studies"]!=8 or replay["verified_slices"]!=344 or not pilot["strict_state_coverage"]):
        raise ValueError("Actual successful independently replayed pilot required")
    for row in pilot_plan["frozen_files"]:
        if digest(row["path"])!=row["sha256"]:raise ValueError("Admitted pilot source/runtime/input changed")
    pixel_plan=read(PIXELS/"plan.json");manifest=read(PIXELS/"pixel-manifest.json");pixel_validation=read(PIXELS/"independent-validation.json");pixel_terminal=read(PIXELS/"terminal-verification.json")
    if (not pixel_terminal["passed"] or not pixel_terminal["original_source_pixel_replay_all_studies_exact"]
        or pixel_terminal["independent_validation_sha256"]!=digest(PIXELS/"independent-validation.json")
        or pixel_validation["manifest_sha256"]!=digest(PIXELS/"pixel-manifest.json")
        or manifest["study_count"]!=4228 or manifest["valid_slices"]!=187272 or len(manifest["pixels"])!=4228
        or sum(row["valid_slices"] for row in manifest["pixels"])!=187272):raise ValueError("Complete original-source pixel integrity required")
    partitions=balanced_partitions(manifest["pixels"])
    for part in partitions:
        part["forecast_seconds"]=2*(pilot["vision_load_seconds"]+part["valid_slices"]*pilot["maximum_valid_slice_seconds"]
            +len(part["study_uids"])*pixel_validation["maximum_cache_read_reconstruction_seconds"])+600
        if part["forecast_seconds"]>10740:raise ValueError("Fixed partition cannot fit bounded CPU job")
    recovery_path=PROJECT/".scratch/rsna-leaderboard/resource-approvals/ticket64-full-backbone-recovery-20261002.json"
    recovery=read(recovery_path)
    now=datetime.now(timezone.utc);freeze=datetime.fromisoformat(recovery["projection"]["science_freeze_utc"])
    extra_calendar=4*10800+3600+5*3600
    projected_finish=now+timedelta(seconds=recovery["projection"]["maximum_projected_calendar_seconds"]+86400+extra_calendar)
    if projected_finish>=freeze:raise ValueError("Whole existing route, extra cohort and review cannot fit science freeze")
    usage=subprocess.check_output(["du","--apparent-size","-s","-B1",str(PROJECT),
        str(PROJECT.parent/"models/qwen3-8b-b968826d9c46dd6066d109eabc6255188de91218")],text=True)
    existing=sum(int(row.split()[0]) for row in usage.splitlines())
    retained=recovery["projection"]["checkpoint_storage_allowance_bytes"]+5_000_000_000
    allowance=2_000_000_000
    if existing+retained+allowance>=2_000_000_000_000 or shutil.disk_usage(PROJECT).free<allowance:
        raise ValueError("Full feature bank cannot fit protected storage")
    ROOT.mkdir(mode=0o700);(ROOT/"features").mkdir(mode=0o700)
    snapshot=ROOT/"source-snapshot"
    shutil.copytree(Path(pilot_plan["source_snapshot"])/"src",snapshot/"src")
    scripts=[Path(__file__),PROJECT/"scripts/ticket67_biomedclip_extract.py",PROJECT/"scripts/validate_ticket67_biomedclip_cohort.py",PROJECT/"scripts/coordinate_ticket67_biomedclip_cohort.py"]
    for script in scripts:shutil.copyfile(script,snapshot/script.name)
    identity=dict(model="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",revision="9f341de24bfb00180f1b847274256e9b65a3a32e",
        checkpoint_sha256=digest(pilot_plan["checkpoint_path"]),representation="biomedclip-vitb16-projected-l2-fp32-v1",
        feature_dimension=512,image_size=224,grayscale_to_rgb="repeat",crop=False,microbatch=1,precision="float32",
        explicit_l2_normalization=True,pixel_identity_sha256=pixel_plan["pixel_identity_sha256"],
        config_sha256=digest(pilot_plan["config_path"]),runtime_versions=pilot_plan["runtime_versions"])
    fixed=[AUTH,recovery_path,PILOT/"plan.json",PILOT/"pilot-result.json",PILOT/"independent-validation.json",PILOT/"terminal-verification.json",
        PIXELS/"plan.json",PIXELS/"pixel-manifest.json",PIXELS/"independent-validation.json",PIXELS/"terminal-verification.json"]
    fixed+= [Path(row["path"]) for row in pilot_plan["frozen_files"]]
    fixed+= [p for p in snapshot.rglob("*") if p.is_file()]
    fixed+= [PROJECT/"scripts/coordinate_ticket64_recovery_v2.py",PROJECT/"scripts/coordinate_ticket64_full_backbone_comparison.py",PROJECT/"scripts/monitor_ticket48_cpu_pilot.py"]
    plan=dict(schema_version="rsna-ticket67-biomedclip-cohort-v1",created_at_utc=now.isoformat(),run_root=str(ROOT),source_snapshot=str(snapshot),
        decision_path=str(DECISION),runtime_python=str(PILOT/"runtime/bin/python"),checkpoint_path=pilot_plan["checkpoint_path"],config_path=pilot_plan["config_path"],
        pixel_manifest_path=str(PIXELS/"pixel-manifest.json"),pixel_root=pixel_plan["pixel_root"],pixel_plan_sha256=digest(PIXELS/"plan.json"),
        pixel_identity_sha256=pixel_plan["pixel_identity_sha256"],feature_root=str(ROOT/"features"),feature_identity=identity,
        feature_identity_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        partitions=partitions,study_count=4228,valid_slices=187272,pilot_root=str(PILOT),
        resources=dict(partition="cluster1-cpu",nodes=1,tasks=1,cpus_per_task=16,memory_gib=64,walltime_seconds=10800,
            code_watchdog_seconds=10740,verification_walltime_seconds=3600,verification_watchdog_seconds=3540,
            maximum_owner_cpu_jobs=2,maximum_simultaneous_extraction_jobs=1,maximum_launches_per_partition=1,
            maximum_verification_launches=1,exclusive=False,requeue=False,gpu_count=0,external_spend_usd=0),
        projection=dict(cost_multiplier=2,maximum_observed_slice_seconds=pilot["maximum_valid_slice_seconds"],
            cache_read_total_seconds=pixel_validation["all_cache_read_reconstruction_seconds"],
            maximum_cache_read_seconds=pixel_validation["maximum_cache_read_reconstruction_seconds"],
            maximum_allocated_cpu_node_seconds=46800,maximum_queue_wait_seconds_per_job=3600,
            additional_calendar_allowance_seconds=extra_calendar,retained_prior_calendar_seconds=recovery["projection"]["maximum_projected_calendar_seconds"],
            retained_review_seconds=86400,conservative_complete_route_finish_utc=projected_finish.isoformat(),science_freeze_utc=freeze.isoformat()),
        storage=dict(du_output=usage,existing_apparent_bytes=existing,retained_ticket64_allowance_bytes=retained,
            additional_allowance_bytes=allowance,maximum_padded_feature_payload_bytes=4228*128*512*4,
            valid_only_vector_bytes=187272*512*4,pooled_matrix_bytes=4228*2076*4,projected_apparent_bytes=existing+retained+allowance),
        frozen_files=[dict(path=str(p),sha256=digest(p)) for p in sorted(set(fixed))],
        network=False,labels_used=False,fitting=False,candidate_promotion=False,paired_accuracy_comparison_authorized=False)
    write(ROOT/"plan.json",plan);checksum=digest(ROOT/"plan.json")
    launchers=[]
    for part in partitions:
        directory=ROOT/f"partition-{part['partition']}";directory.mkdir(mode=0o700)
        launcher=directory/"run.sbatch"
        launcher.write_text(launcher_text(plan,checksum,directory,False,part["partition"]))
        launchers.append(dict(path=str(launcher),sha256=digest(launcher),partition=part["partition"]))
    verification=ROOT/"verification";verification.mkdir(mode=0o700)
    launcher=verification/"run.sbatch";launcher.write_text(launcher_text(plan,checksum,verification,True,None))
    launchers.append(dict(path=str(launcher),sha256=digest(launcher),partition=None))
    write(ROOT/"prepared-launchers.json",dict(plan_sha256=checksum,launchers=launchers))
    write(DECISION,dict(schema_version="rsna-campaign-autonomous-decision-v1",decision="approved",
        decision_made_by="campaign agent under standing owner authorization",authorization_sha256=digest(AUTH),plan_sha256=checksum,
        resources=plan["resources"],projection=plan["projection"],storage=plan["storage"],new_exact_owner_numeric_approval_claimed=False,
        basis="Exact original-MRI medical encoder outputs independently replayed; complete measured input bank plus conservative model maximum support four fixed partitions and independent cohort verification.",
        delegated_registered_ticket64_fit_scheduling=True,ticket64_science_or_sources_changed=False,
        conditional_final_fit_authorized=False,paired_accuracy_comparison_authorized=False,gpu_ledger_changed=False,tpu_ledger_changed=False))
    print(json.dumps(dict(plan_sha256=checksum,feature_identity_sha256=plan["feature_identity_sha256"],projected_finish_utc=projected_finish.isoformat(),projection=[{k:v for k,v in p.items() if k!="study_uids"} for p in partitions])))


def launcher_text(plan,checksum,directory,verification,partition):
    time_limit="01:00:00" if verification else "03:00:00"
    command=f"{plan['runtime_python']} {plan['source_snapshot']}/validate_ticket67_biomedclip_cohort.py {ROOT}/plan.json {checksum}" if verification else f"{plan['runtime_python']} {plan['source_snapshot']}/ticket67_biomedclip_extract.py {ROOT}/plan.json {checksum} {partition}"
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket67-{'verify' if verification else f'part-{partition}'}
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time={time_limit}
#SBATCH --no-requeue
#SBATCH --output={directory}/slurm-%j.out
#SBATCH --error={directory}/slurm-%j.err
#SBATCH --chdir={PROJECT}
set -euo pipefail
umask 077
export RSNA_PROJECT_ROOT={PROJECT}
export PYTHONPATH={plan['source_snapshot']}/src
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
{command}
"""


if __name__=="__main__":main()
