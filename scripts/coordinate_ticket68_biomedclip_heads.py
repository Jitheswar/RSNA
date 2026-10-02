"""Wait for verified features, then share slots with unchanged registered fits."""

from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from ticket68_biomedclip_heads import digest,read,write,require
from validate_ticket68_biomedclip_heads import successful_terminal

PROJECT=Path(os.environ.get("RSNA_PROJECT_ROOT",Path(__file__).resolve().parents[1])).resolve()
CONTROLLER=PROJECT/"artifacts/private/ticket68-biomedclip-head-controller-v1"
ROOT=PROJECT/"artifacts/private/ticket68-biomedclip-matched-heads-v1"
BANK=PROJECT/"artifacts/private/ticket67-biomedclip-cohort-v1"
BANK_UNIT="rsna-ticket67-biomedclip-cohort-v1.service"
FIT_UNIT="rsna-ticket64-comparison-coordinator-v2.service"
FIT_ROOT=PROJECT/"artifacts/private/ticket64-full-backbone-comparison-v2"
FIT_SHA="41028974908d8ea8c4fe336f86de0263782100c76b9375dc7693bf60ac213eaf"


def active(unit):return subprocess.run(["systemctl","--user","is-active","--quiet",unit]).returncode==0


def resume():
    if active(FIT_UNIT):return
    subprocess.run(["systemd-run","--user","--unit="+FIT_UNIT.removesuffix(".service"),"--collect",
        "--setenv=RSNA_PROJECT_ROOT="+str(PROJECT),"--setenv=PYTHONDONTWRITEBYTECODE=1",
        str(PROJECT/".venv/bin/python"),str(PROJECT/"scripts/coordinate_ticket64_recovery_v2.py"),FIT_SHA],check=True,timeout=30)


def cpu_jobs():
    rows=subprocess.check_output(["squeue","-u",os.environ["USER"],"-h","-o","%i %P %t"],text=True,timeout=15).splitlines()
    return [r.split() for r in rows if r.split()[1]=="cluster1-cpu"]


def component(kind,checksum):
    directory=ROOT/kind
    if not (directory/"accepted-launch.json").exists():
        require(not (directory/"launch-intent.json").exists(),"Consumed intent lacks accepted receipt; no retry")
        return "unlaunched",None
    accepted=read(directory/"accepted-launch.json");job=accepted["job_id"]
    require(accepted["plan_sha256"]==checksum and accepted["launch_attempt"]==1,"Exact sole experiment attempt required")
    if not (directory/f"controller-{job}/terminal-observation.json").exists():return "active",job
    successful_terminal(directory,job,"00:30:00" if kind=="fit" else "03:00:00")
    result=read(ROOT/"analysis.json" if kind=="fit" else directory/"independent-validation.json")
    require(result["status"]=="COMPLETE" and result["plan_sha256"]==checksum and str(result["job_id"])==str(job),"Exact complete experiment result required")
    return "complete",job


def next_component(fit,verification,occupied):
    if occupied>=2:return None
    if fit=="unlaunched":return "fit"
    if fit=="complete" and verification=="unlaunched":return "verification"
    return None


def launch(kind,checksum):
    require(len(cpu_jobs())<2,"Both owner CPU slots occupied")
    prepared=read(ROOT/"prepared-launchers.json");row=next(r for r in prepared["launchers"] if r["kind"]==kind)
    require(prepared["plan_sha256"]==checksum and digest(row["path"])==row["sha256"],"Exact prepared experiment launcher required")
    directory=ROOT/kind
    write(directory/"launch-intent.json",dict(plan_sha256=checksum,launcher_sha256=row["sha256"],launch_attempt=1,at_utc=datetime.now(timezone.utc).isoformat()))
    job=int(subprocess.check_output(["sbatch","--parsable",row["path"]],text=True,timeout=30).strip().split(";")[0])
    write(directory/"accepted-launch.json",dict(plan_sha256=checksum,launch_attempt=1,job_id=job))
    subprocess.run(["systemd-run","--user",f"--unit=rsna-ticket68-{kind}-{job}-observer","--collect",str(PROJECT/".venv/bin/python"),
        str(PROJECT/"scripts/monitor_ticket48_cpu_pilot.py"),str(directory),str(job)],check=True,timeout=30)


def prepare_controller():
    require(not CONTROLLER.exists(),"Immutable pending-head controller already exists")
    bank=read(BANK/"plan.json")
    CONTROLLER.mkdir(mode=0o700);snapshot=CONTROLLER/"source-snapshot"
    shutil.copytree(Path(bank["source_snapshot"])/"src",snapshot/"src")
    names=("ticket68_biomedclip_heads.py","prepare_ticket68_biomedclip_heads.py","validate_ticket68_biomedclip_heads.py","coordinate_ticket68_biomedclip_heads.py")
    paths=[]
    for name in names:
        source=PROJECT/"scripts"/name;shutil.copyfile(source,snapshot/name);paths.append(source)
    paths.extend(p for p in snapshot.rglob("*") if p.is_file())
    paths.extend(PROJECT/"scripts"/name for name in ("coordinate_ticket64_recovery_v2.py","coordinate_ticket64_full_backbone_comparison.py","monitor_ticket48_cpu_pilot.py"))
    admission=dict(schema_version="rsna-ticket68-pending-head-controller-v1",source_snapshot=str(snapshot),bank_plan_sha256=digest(BANK/"plan.json"),
        science_freeze_utc=bank["projection"]["science_freeze_utc"],fitting_plan_admitted=False,fitting_launched=False,automatic_retry=False,candidate_promotion=False,
        frozen_files=[dict(path=str(p),sha256=digest(p)) for p in sorted(set(paths))])
    write(CONTROLLER/"admission.json",admission)
    print(json.dumps(dict(controller_admission_sha256=digest(CONTROLLER/"admission.json"),source_snapshot=str(snapshot))),flush=True)


def watch(admission_sha):
    require(digest(CONTROLLER/"admission.json")==admission_sha,"Exact pending experiment controller required")
    admission=read(CONTROLLER/"admission.json")
    for row in admission["frozen_files"]:require(digest(row["path"])==row["sha256"],"Frozen pending controller changed")
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("Experiment scheduling interrupted")))
    with (CONTROLLER/"coordinator.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(CONTROLLER/"coordinator-start.json",dict(admission_sha256=admission_sha,at_utc=datetime.now(timezone.utc).isoformat()))
        held=False;missing_since={};queued_since={}
        try:
            while True:
                require(datetime.now(timezone.utc)<datetime.fromisoformat(admission["science_freeze_utc"]),"Science freeze reached while waiting for complete bank")
                require(digest(BANK/"plan.json")==admission["bank_plan_sha256"],"Exact pending bank plan required")
                if (BANK/"terminal-verification.json").exists() and not active(BANK_UNIT):
                    require(read(BANK/"terminal-verification.json")["passed"] and active(FIT_UNIT),"Successful bank and restored original fit scheduling required")
                    break
                require(active(BANK_UNIT),"Feature scheduler lacks live unit or successful terminal; no retry")
                state=dict(at_utc=datetime.now(timezone.utc).isoformat(),phase="wait_for_complete_bank",fitting_plan_admitted=False,fitting_launched=False)
                temporary=CONTROLLER/"state.partial";temporary.write_text(json.dumps(state,indent=2)+"\n");temporary.replace(CONTROLLER/"state.json")
                time.sleep(15)
            from prepare_ticket68_biomedclip_heads import main as prepare
            checksum=prepare(Path(admission["source_snapshot"]))
            plan=read(ROOT/"plan.json");decision=read(plan["decision_path"])
            require(decision["delegated_registered_ticket64_fit_scheduling"],"Exact fitting scheduling delegation required")
            # Preparation leaves the original scheduler moving; release it only
            # once this controller is ready to advance unchanged registered fits.
            require(active(FIT_UNIT),"Original fit coordinator must be live before ownership transfer")
            held=True;subprocess.run(["systemctl","--user","stop",FIT_UNIT],check=True,timeout=30)
            require(not active(FIT_UNIT),"Original coordinator did not release scheduling ownership")
            sys.path.insert(0,str(PROJECT/"scripts"))
            from coordinate_ticket64_recovery_v2 import stage_location
            from coordinate_ticket64_full_backbone_comparison import stage_state,next_stage,launch_environment
            fit_plan=read(FIT_ROOT/"plan.json");require(digest(FIT_ROOT/"plan.json")==FIT_SHA,"Exact registered fitting plan required")
            while True:
                jobs=cpu_jobs();fit,fit_job=component("fit",checksum);verification,verify_job=component("verification",checksum)
                if verification=="complete":
                    validation=read(ROOT/"verification/independent-validation.json")
                    require(validation["verified_final_heads"]==6 and validation["original_mri_studies"]==58
                        and validation["verified_medical_predictions"]==validation["verified_resnet_predictions"]==116
                        and validation["maximum_prediction_abs_error"]<=1e-6 and validation["all_fixed_native_comparisons_reproduced"],"Complete native/source/final-head verification required")
                    seconds,sha=successful_terminal(ROOT/"verification",verify_job,"03:00:00")
                    write(ROOT/"terminal-verification.json",dict(status="COMPLETE",passed=True,plan_sha256=checksum,verification_job_id=verify_job,
                        allocated_verification_cpu_node_seconds=seconds,verification_terminal_sha256=sha,independent_validation_sha256=digest(ROOT/"verification/independent-validation.json"),
                        analysis_sha256=digest(ROOT/"analysis.json"),disposition=validation["disposition"],candidate_promotion=False,gpu_seconds=0,tpu_seconds=0,external_spend_usd=0))
                    return
                for status,job in ((fit,fit_job),(verification,verify_job)):
                    if status!="active":continue
                    matches=[r for r in jobs if r[0]==str(job)]
                    if not matches:
                        require(time.monotonic()-missing_since.setdefault(job,time.monotonic())<=300,"Accepted experiment lacks queue or retained terminal; no retry")
                    else:missing_since.pop(job,None)
                    if matches and matches[0][2]=="PD":
                        if time.monotonic()-queued_since.setdefault(job,time.monotonic())>3600:
                            subprocess.run(["scancel",str(job)],check=True,timeout=30);raise TimeoutError("Experiment queue allowance exceeded; no retry")
                    else:queued_since.pop(job,None)
                control=stage_state(FIT_ROOT/"control",FIT_SHA,None,"02:00:00")
                folds=[[stage_state(stage_location(f,s)[0],stage_location(f,s)[1],min(7500,(s+1)*1000),"05:30:00") for s in range(8)] for f in range(3)]
                for fold,states in enumerate(folds):
                    for stage,status in enumerate(states):
                        if status!="active":continue
                        fit_active_job=str(read(stage_location(fold,stage)[0]/"accepted-launch.json")["job_id"])
                        if fit_active_job not in [r[0] for r in jobs]:
                            require(time.monotonic()-missing_since.setdefault(fit_active_job,time.monotonic())<=300,"Registered fitting lacks queue or retained terminal; no retry")
                        else:missing_since.pop(fit_active_job,None)
                item=next_component(fit,verification,len(jobs));registered=next_stage(control,folds,len(jobs)) if item is None else None
                state=dict(at_utc=datetime.now(timezone.utc).isoformat(),phase="execute_fixed_experiment",plan_sha256=checksum,fit=fit,verification=verification,
                    occupied_cpu_jobs=[r[0] for r in jobs],next_experiment=item,next_registered_fit=registered,original_fit_scheduler_held=True,fit_science_changed=False)
                temporary=CONTROLLER/"state.partial";temporary.write_text(json.dumps(state,indent=2)+"\n");temporary.replace(CONTROLLER/"state.json")
                if item is not None or registered is not None:
                    require(datetime.now(timezone.utc)<datetime.fromisoformat(plan["projection"]["science_freeze_utc"]),"Science freeze reached")
                    if item is not None:launch(item,checksum)
                    else:
                        fold,stage=registered;require(not fold==stage==0,"Inherited first stage cannot relaunch")
                        subprocess.run([str(PROJECT/".venv/bin/python"),str(Path(fit_plan["source_snapshot"])/"ticket64_full_backbone_recovery_v2.py"),
                            "launch",FIT_SHA,"full",str(fold),str(stage)],env=launch_environment(fit_plan),check=True)
                    continue
                time.sleep(15)
        finally:
            if held:
                resume();write(CONTROLLER/"original-fit-scheduler-restored.json",dict(at_utc=datetime.now(timezone.utc).isoformat(),active=active(FIT_UNIT),fit_jobs_restarted=False))


if __name__=="__main__":
    if sys.argv[1]=="prepare-controller":prepare_controller()
    else:watch(sys.argv[1])
