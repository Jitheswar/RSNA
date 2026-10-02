"""Admit matched medical fitting only after actual complete feature verification."""

from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

from ticket68_biomedclip_heads import ARMS,digest,read,write,require,configure_head

PROJECT=Path(os.environ.get("RSNA_PROJECT_ROOT",Path(__file__).resolve().parents[1])).resolve()
BANK=PROJECT/"artifacts/private/ticket67-biomedclip-cohort-v1"
BANK_SHA="e9ae761facc5ca3345925f38c36682ac18ff94e6e92d5c018e8a3eb7a9f5256a"
ROOT=PROJECT/"artifacts/private/ticket68-biomedclip-matched-heads-v1"
RESNET=PROJECT/"artifacts/private/ticket55-resnet-long-report-head-v2"
AUTH=PROJECT/".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
DECISION=PROJECT/".scratch/rsna-leaderboard/resource-approvals/ticket68-biomedclip-heads-20261002.json"


def main(source_directory=None):
    require(not ROOT.exists() and not DECISION.exists(),"Consumed matched-head preparation cannot repeat")
    require(digest(BANK/"plan.json")==BANK_SHA,"Exact fixed medical feature bank required")
    bank=read(BANK/"plan.json");terminal=read(BANK/"terminal-verification.json");manifest=read(BANK/"feature-manifest.json")
    validation=read(BANK/"verification/independent-validation.json")
    require(terminal["status"]=="COMPLETE" and terminal["passed"] and terminal["plan_sha256"]==BANK_SHA
        and terminal["independent_validation_sha256"]==digest(BANK/"verification/independent-validation.json")
        and terminal["manifest_sha256"]==digest(BANK/"feature-manifest.json") and validation["verified_studies"]==4228
        and validation["verified_valid_slices"]==187272 and validation["replayed_fixed_native_slices"]==344
        and 0<=validation["maximum_replay_abs_error"]<=1e-6 and manifest["study_count"]==4228
        and manifest["feature_identity_sha256"]==bank["feature_identity_sha256"],"Actual successful whole-bank verification required")
    original=read(RESNET/"plan.json");parent=read(original["parent_plan"]);prior_validation=read(RESNET/"validation.json")
    require(digest(RESNET/"plan.json")=="8e6899d7af4688e1512686559ff9c38bc94673cc2364d8264fb7e95588928e49","Exact historical ResNet comparison required")
    require(prior_validation["status"]=="COMPLETE" and prior_validation["final_predictions_verified"]==116
        and prior_validation["plan_sha256"]==digest(RESNET/"plan.json") and prior_validation["maximum_prediction_error"]<=1e-6,
        "Independently verified matched ResNet experiment required")
    usage=subprocess.check_output(["du","--apparent-size","-s","-B1",str(PROJECT),
        str(PROJECT.parent/"models/qwen3-8b-b968826d9c46dd6066d109eabc6255188de91218")],text=True)
    existing=sum(int(r.split()[0]) for r in usage.splitlines());protected=bank["storage"]["retained_ticket64_allowance_bytes"]+bank["storage"]["additional_allowance_bytes"]
    require(existing+protected+1_000_000_000<2_000_000_000_000 and shutil.disk_usage(PROJECT).free>1_000_000_000,"Protected full-route storage required")
    now=datetime.now(timezone.utc);freeze=datetime.fromisoformat(bank["projection"]["science_freeze_utc"])
    calendar=bank["projection"]["retained_prior_calendar_seconds"]+bank["projection"]["retained_review_seconds"]+19800
    require(now+timedelta(seconds=calendar)<freeze,"Complete prior route plus review and new experiment must fit freeze")
    from rsna_knee import ticket36 as t36
    from rsna_knee.training import _model_state_sha256
    model_config=original["original_resnet_configuration"]["model"]
    head=configure_head(dict(original_model_config=model_config)).build_head(original["optimization"]["initialization_seed"])
    initial_sha=_model_state_sha256(head)
    files=[AUTH,RESNET/"plan.json",RESNET/"validation.json",RESNET/"pooled-cohort.npz",RESNET/"pooled-cohort-evidence.json",
        Path(original["parent_plan"]),t36.FOLD_MANIFEST_PATH,t36.TICKET30_ARTIFACT,t36.WEIGHTS_PATH,
        BANK/"plan.json",BANK/"terminal-verification.json",BANK/"feature-manifest.json",BANK/"pooled-features.npz",BANK/"verification/independent-validation.json"]
    files.extend(Path(r["path"]) for r in bank["frozen_files"])
    feature_files=[dict(path=str(Path(bank["feature_root"])/r["feature_file"]),sha256=r["feature_file_sha256"]) for r in manifest["features"]]
    files.extend(Path(r["path"]) for r in original["frozen_files"])
    feature_files.extend(original["feature_files"])
    maximum_fit_seconds=0.
    for arm in ARMS:
        for fold in range(3):
            directory=RESNET/arm/f"fold-{fold}";record=read(directory/"run.json")
            require(record["status"]=="COMPLETE" and record["counts"]["successful_optimizer_steps"]==7500
                and record["counts"]["initial_head_state_sha256"]==initial_sha
                and digest(directory/"checkpoints/final.pt")==record["checkpoint_sha256"],"Exact final comparator and random initialization required")
            maximum_fit_seconds=max(maximum_fit_seconds,record["counts"]["elapsed_seconds"])
            files.extend((directory/"run.json",directory/"checkpoints/final.pt"))
    fit_forecast=2*6*maximum_fit_seconds+600
    require(fit_forecast<1740,"Six matched medical heads cannot fit fixed allocation")
    pilot=read(Path(bank["pilot_root"])/"pilot-result.json")
    medical_native_forecast=2*(58*pilot["maximum_native_read_pack_seconds"]+58*128*pilot["maximum_valid_slice_seconds"]+pilot["vision_load_seconds"])+600
    verification_forecast=medical_native_forecast+5400
    require(verification_forecast<10740,"Full original-MRI plus existing matched replay reserve cannot fit")
    # Reconcile bytes before creating any immutable execution request.
    for row in bank["frozen_files"]+feature_files:require(digest(row["path"])==row["sha256"],"Verified source/runtime/feature bytes changed")
    ROOT.mkdir(mode=0o700);snapshot=ROOT/"source-snapshot"
    shutil.copytree(Path(bank["source_snapshot"])/"src",snapshot/"src")
    source_directory=Path(source_directory or Path(__file__).resolve().parent)
    for name in ("ticket68_biomedclip_heads.py","prepare_ticket68_biomedclip_heads.py","validate_ticket68_biomedclip_heads.py","coordinate_ticket68_biomedclip_heads.py"):
        shutil.copyfile(source_directory/name,snapshot/name)
    pixel_plan=read(Path(bank["pixel_manifest_path"]).parent/"plan.json")
    reader=next(r for r in pixel_plan["frozen_files"] if Path(r["path"]).name=="ticket62_mri_pixel_materialize.py")
    require(digest(reader["path"])==reader["sha256"],"Exact original MRI reader required")
    shutil.copyfile(reader["path"],snapshot/"ticket62_mri_pixel_materialize.py")
    files.extend(p for p in snapshot.rglob("*") if p.is_file())
    pool=read(RESNET/"pooled-cohort-evidence.json")
    plan=dict(schema_version="rsna-ticket68-matched-biomedclip-heads-v1",created_at_utc=now.isoformat(),run_root=str(ROOT),source_snapshot=str(snapshot),
        decision_path=str(DECISION),authorization_sha256=digest(AUTH),runtime_python=bank["runtime_python"],bank_plan_path=str(BANK/"plan.json"),bank_plan_sha256=BANK_SHA,
        feature_identity_sha256=bank["feature_identity_sha256"],medical_pooled_path=manifest["pooled_file"],medical_pooled_sha256=manifest["pooled_file_sha256"],
        medical_pooled_array_sha256=manifest["pooled_array_sha256"],resnet_root=str(RESNET),resnet_pooled_sha256=pool["pooled_file_sha256"],resnet_pooled_array_sha256=pool["pooled_array_sha256"],
        original_model_config=model_config,initial_head_state_sha256=initial_sha,optimization=original["optimization"],folds=parent["folds"],
        fold_manifest_path=str(t36.FOLD_MANIFEST_PATH),fold_manifest_checksum=parent["fold_manifest_checksum"],report_artifact_path=str(t36.TICKET30_ARTIFACT),
        report_artifact_sha256=t36.TICKET30_AUDITED_ARTIFACT_SHA256,resnet_weights_path=str(t36.WEIGHTS_PATH),pixel_plan_path=str(Path(bank["pixel_manifest_path"]).parent/"plan.json"),
        resources=dict(partition="cluster1-cpu",nodes=1,tasks=1,cpus_per_task=16,memory_gib=64,fit_walltime_seconds=1800,fit_watchdog_seconds=1740,
            verification_walltime_seconds=10800,verification_watchdog_seconds=10740,maximum_owner_cpu_jobs=2,maximum_simultaneous_experiment_jobs=1,
            maximum_fit_launches=1,maximum_verification_launches=1,exclusive=False,requeue=False,gpu_count=0,external_spend_usd=0),
        projection=dict(fit_forecast_seconds=fit_forecast,medical_native_forecast_seconds=medical_native_forecast,retained_other_native_replay_seconds=5400,
            verification_forecast_seconds=verification_forecast,additional_cpu_node_seconds=12600,additional_calendar_seconds=19800,
            retained_prior_calendar_seconds=bank["projection"]["retained_prior_calendar_seconds"],retained_review_seconds=86400,
            science_freeze_utc=freeze.isoformat(),conservative_complete_route_finish_utc=(now+timedelta(seconds=calendar)).isoformat()),
        storage=dict(du_output=usage,existing_apparent_bytes=existing,retained_prior_allowance_bytes=protected,additional_allowance_bytes=1_000_000_000,projected_apparent_bytes=existing+protected+1_000_000_000),
        analysis=dict(primary="medical_report_minus_matched_resnet_report",bootstrap_resamples=2000,seed=20261002,confidence_level=.95,fixed_medical_weight=.5,
            adaptive_gold_cohort=True,patient_group_ids_unavailable=True,provisional_only=True),
        frozen_files=[dict(path=str(p),sha256=digest(p)) for p in sorted(set(files))]+feature_files,
        labels_from_original_admission_only=True,new_weight_download=False,network=False,candidate_promotion=False,scored_anchor_comparison_authorized=False,offline_delivery_authorized=False)
    write(ROOT/"plan.json",plan);checksum=digest(ROOT/"plan.json")
    write(DECISION,dict(decision="approved",decision_made_by="campaign agent under standing owner authorization",authorization_sha256=digest(AUTH),plan_sha256=checksum,
        new_exact_owner_numeric_approval_claimed=False,resources=plan["resources"],projection=plan["projection"],storage=plan["storage"],delegated_registered_ticket64_fit_scheduling=True,
        ticket64_science_or_sources_changed=False,gpu_ledger_changed=False,tpu_ledger_changed=False,candidate_promotion=False))
    launchers=[]
    for kind,limit,script in (("fit","00:30:00","ticket68_biomedclip_heads.py"),("verification","03:00:00","validate_ticket68_biomedclip_heads.py")):
        directory=ROOT/kind;directory.mkdir(mode=0o700);launcher=directory/"run.sbatch"
        launcher.write_text(f"""#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket68-{kind}
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time={limit}
#SBATCH --no-requeue
#SBATCH --output={directory}/slurm-%j.out
#SBATCH --error={directory}/slurm-%j.err
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
{bank['runtime_python']} {snapshot}/{script} {ROOT}/plan.json {checksum}
""")
        subprocess.run(["bash","-n",str(launcher)],check=True)
        launchers.append(dict(kind=kind,path=str(launcher),sha256=digest(launcher)))
    write(ROOT/"prepared-launchers.json",dict(plan_sha256=checksum,launchers=launchers))
    print(json.dumps(dict(plan_sha256=checksum,projection=plan["projection"])),flush=True)
    return checksum


if __name__=="__main__":main()
