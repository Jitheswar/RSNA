"""Share two CPU slots between the fixed feature bank and unchanged registered fits."""

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from validate_ticket67_biomedclip_cohort import terminal

PROJECT=Path(os.environ.get("RSNA_PROJECT_ROOT",Path(__file__).resolve().parents[1]))
ROOT=PROJECT/"artifacts/private/ticket67-biomedclip-cohort-v1"
FIT_ROOT=PROJECT/"artifacts/private/ticket64-full-backbone-comparison-v2"
FIT_SHA="41028974908d8ea8c4fe336f86de0263782100c76b9375dc7693bf60ac213eaf"
FIT_UNIT="rsna-ticket64-comparison-coordinator-v2.service"


def digest(path):
    with Path(path).open("rb") as handle:return hashlib.file_digest(handle,"sha256").hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open("x") as handle:json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False);handle.write("\n")


def active():return subprocess.run(["systemctl","--user","is-active","--quiet",FIT_UNIT]).returncode==0


def resume():
    if active():return
    subprocess.run(["systemd-run","--user","--unit="+FIT_UNIT.removesuffix(".service"),"--collect",
        "--setenv=RSNA_PROJECT_ROOT="+str(PROJECT),"--setenv=PYTHONDONTWRITEBYTECODE=1",
        str(PROJECT/".venv/bin/python"),str(PROJECT/"scripts/coordinate_ticket64_recovery_v2.py"),FIT_SHA],check=True,timeout=30)


def cpu_jobs():
    rows=subprocess.check_output(["squeue","-u",os.environ["USER"],"-h","-o","%i %P %t"],text=True,timeout=15).splitlines()
    return [r.split() for r in rows if r.split()[1]=="cluster1-cpu"]


def component(directory,checksum,verification=False):
    accepted=directory/"accepted-launch.json"
    if not accepted.exists():
        if (directory/"launch-intent.json").exists():raise ValueError("Consumed cohort intent lacks accepted receipt; no retry")
        return "unlaunched",None
    launch=read(accepted);job=launch["job_id"]
    if launch["plan_sha256"]!=checksum or launch["launch_attempt"]!=1:raise ValueError("Exact sole cohort launch required")
    if not (directory/f"controller-{job}/terminal-observation.json").exists():return "active",job
    seconds,sha=terminal(directory,job,"01:00:00" if verification else "03:00:00")
    if verification:
        validation=read(directory/"independent-validation.json")
        if validation["status"]!="COMPLETE" or validation["plan_sha256"]!=checksum or str(validation["job_id"])!=str(job):
            raise ValueError("Complete exact cohort verification required")
    else:
        result=read(directory/f"job-{job}/partition-result.json")
        if result["status"]!="COMPLETE" or result["plan_sha256"]!=checksum or str(result["job_id"])!=str(job):
            raise ValueError("Complete exact extraction result required")
    return "complete",job


def next_component(states,verification,occupied):
    if occupied>=2:return None
    if "active" not in states:
        if "unlaunched" in states:return states.index("unlaunched")
        if all(s=="complete" for s in states) and verification=="unlaunched":return "verification"
    return None


def launch(plan,checksum,item):
    if len(cpu_jobs())>=2:raise ValueError("Both CPU slots occupied")
    directory=ROOT/"verification" if item=="verification" else ROOT/f"partition-{item}"
    prepared=read(ROOT/"prepared-launchers.json")
    row=next(r for r in prepared["launchers"] if r["path"]==str(directory/"run.sbatch"))
    if prepared["plan_sha256"]!=checksum or digest(row["path"])!=row["sha256"]:raise ValueError("Frozen cohort launcher differs")
    write(directory/"launch-intent.json",dict(plan_sha256=checksum,launcher_sha256=row["sha256"],launch_attempt=1,at_utc=datetime.now(timezone.utc).isoformat()))
    job=int(subprocess.check_output(["sbatch","--parsable",row["path"]],text=True,timeout=30).strip().split(";")[0])
    write(directory/"accepted-launch.json",dict(plan_sha256=checksum,job_id=job,launch_attempt=1))
    subprocess.run(["systemd-run","--user",f"--unit=rsna-ticket67-{item}-{job}-observer","--collect",str(PROJECT/".venv/bin/python"),
        str(PROJECT/"scripts/monitor_ticket48_cpu_pilot.py"),str(directory),str(job)],check=True,timeout=30)
    return job


def watch(checksum):
    if digest(ROOT/"plan.json")!=checksum:raise ValueError("Exact immutable feature plan required")
    plan=read(ROOT/"plan.json");decision=read(plan["decision_path"])
    if (decision["decision"]!="approved" or decision["plan_sha256"]!=checksum
        or not decision["delegated_registered_ticket64_fit_scheduling"]):raise ValueError("Exact combined scheduling decision required")
    for row in plan["frozen_files"]:
        if digest(row["path"])!=row["sha256"]:raise ValueError("Frozen cohort/controller/input changed")
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("Combined scheduler interrupted")))
    with (ROOT/"coordinator.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/"coordinator-start.json").exists():raise FileExistsError("Consumed coordinator cannot restart automatically")
        write(ROOT/"coordinator-start.json",dict(plan_sha256=checksum,at_utc=datetime.now(timezone.utc).isoformat()))
        held=False;missing_since={};queued_since={}
        try:
            if not active():raise ValueError("Original fitting coordinator not live before ownership transfer")
            held=True;subprocess.run(["systemctl","--user","stop",FIT_UNIT],check=True,timeout=30)
            if active():raise ValueError("Original fitting scheduler did not release ownership")
            sys.path.insert(0,str(PROJECT/"scripts"))
            from coordinate_ticket64_recovery_v2 import stage_location
            from coordinate_ticket64_full_backbone_comparison import stage_state,next_stage,launch_environment
            fit_plan=read(FIT_ROOT/"plan.json")
            if digest(FIT_ROOT/"plan.json")!=FIT_SHA:raise ValueError("Registered fitting plan differs")
            while True:
                jobs=cpu_jobs();components=[component(ROOT/f"partition-{i}",checksum) for i in range(4)]
                states=[s for s,j in components];verification,verify_job=component(ROOT/"verification",checksum,True)
                if verification=="complete":
                    receipt=read(ROOT/"verification/independent-validation.json")
                    seconds,sha=terminal(ROOT/"verification",verify_job,"01:00:00")
                    if (receipt["verified_studies"]!=4228 or receipt["verified_valid_slices"]!=187272 or receipt["replayed_fixed_native_slices"]!=344
                        or not 0<=receipt["maximum_replay_abs_error"]<=1e-6 or receipt["manifest_sha256"]!=digest(ROOT/"feature-manifest.json")):
                        raise ValueError("Complete independent feature evidence required")
                    write(ROOT/"terminal-verification.json",dict(status="COMPLETE",passed=True,plan_sha256=checksum,verification_job_id=verify_job,
                        verification_allocated_cpu_node_seconds=seconds,verification_terminal_sha256=sha,independent_validation_sha256=digest(ROOT/"verification/independent-validation.json"),
                        manifest_sha256=digest(ROOT/"feature-manifest.json"),feature_identity_sha256=plan["feature_identity_sha256"],candidate_promotion=False,gpu_seconds=0,tpu_seconds=0,external_spend_usd=0))
                    return
                for state,job in components+[(verification,verify_job)]:
                    if state=="active":
                        if str(job) not in [r[0] for r in jobs]:
                            if time.monotonic()-missing_since.setdefault(job,time.monotonic())>300:raise ValueError("Accepted cohort job lacks live handle or retained terminal; no retry")
                        else:missing_since.pop(job,None)
                        matches=[r for r in jobs if r[0]==str(job)]
                        if matches and matches[0][2]=="PD":
                            if time.monotonic()-queued_since.setdefault(job,time.monotonic())>3600:
                                subprocess.run(["scancel",str(job)],check=True,timeout=30);raise TimeoutError("Cohort queue allowance exceeded; no retry")
                        else:queued_since.pop(job,None)
                control=stage_state(FIT_ROOT/"control",FIT_SHA,None,"02:00:00")
                folds=[[stage_state(stage_location(f,s)[0],stage_location(f,s)[1],min(7500,(s+1)*1000),"05:30:00") for s in range(8)] for f in range(3)]
                for fold,stages in enumerate(folds):
                    for stage,state in enumerate(stages):
                        if state!="active":continue
                        location=stage_location(fold,stage)[0];fit_job=str(read(location/"accepted-launch.json")["job_id"])
                        if fit_job not in [r[0] for r in jobs]:
                            if time.monotonic()-missing_since.setdefault(fit_job,time.monotonic())>300:
                                raise ValueError("Registered fit lacks live queue or retained terminal; no retry")
                        else:missing_since.pop(fit_job,None)
                item=next_component(states,verification,len(jobs));fit=next_stage(control,folds,len(jobs)) if item is None else None
                observation=dict(at_utc=datetime.now(timezone.utc).isoformat(),partitions=states,verification=verification,
                    occupied_cpu_jobs=[r[0] for r in jobs],next_cohort_component=item,next_registered_fit=fit,
                    original_fit_scheduler_held=True,registered_fit_scheduling_delegated=True,fit_science_changed=False,automatic_retry=False)
                temp=ROOT/"coordinator-state.partial";temp.write_text(json.dumps(observation,indent=2)+"\n");temp.replace(ROOT/"coordinator-state.json")
                if item is not None or fit is not None:
                    if datetime.now(timezone.utc)>=datetime.fromisoformat(plan["projection"]["science_freeze_utc"]):raise TimeoutError("Science freeze reached")
                    if item is not None:launch(plan,checksum,item)
                    else:
                        fold,stage=fit
                        if fold==stage==0:raise ValueError("Inherited original first stage cannot relaunch")
                        subprocess.run([str(PROJECT/".venv/bin/python"),str(Path(fit_plan["source_snapshot"])/"ticket64_full_backbone_recovery_v2.py"),
                            "launch",FIT_SHA,"full",str(fold),str(stage)],env=launch_environment(fit_plan),check=True)
                    continue
                time.sleep(15)
        finally:
            if held:
                resume();write(ROOT/"original-fit-scheduler-restored.json",dict(at_utc=datetime.now(timezone.utc).isoformat(),active=active(),fit_jobs_restarted=False))


if __name__=="__main__":watch(sys.argv[1])
