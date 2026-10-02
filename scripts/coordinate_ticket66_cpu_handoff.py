"""Admit one CPU pilot between fixed fitting stages, then restore scheduling."""

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

PROJECT=Path(os.environ.get("RSNA_PROJECT_ROOT",Path(__file__).resolve().parents[1]))
ROOT=PROJECT/"artifacts/private/ticket66-biomedclip-mri-pilot-v1"
T64=PROJECT/"artifacts/private/ticket64-full-backbone-comparison-v2"
UNIT="rsna-ticket64-comparison-coordinator-v2.service"
T64_SHA="41028974908d8ea8c4fe336f86de0263782100c76b9375dc7693bf60ac213eaf"


def digest(path):
    with Path(path).open("rb") as handle:return hashlib.file_digest(handle,"sha256").hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open("x") as handle:json.dump(value,handle,indent=2,sort_keys=True);handle.write("\n")


def cpu_jobs():
    rows=subprocess.check_output(["squeue","-u",os.environ["USER"],"-h","-o","%i %P %t %j"],text=True,timeout=15).splitlines()
    return [row.split() for row in rows if row.split()[1]=="cluster1-cpu"]


def active():return subprocess.run(["systemctl","--user","is-active","--quiet",UNIT]).returncode==0


def state(value):
    temp=ROOT/"handoff-state.partial";temp.write_text(json.dumps(value,indent=2)+"\n");temp.replace(ROOT/"handoff-state.json")


def resume():
    if active():return
    subprocess.run(["systemd-run","--user","--unit="+UNIT.removesuffix(".service"),"--collect",
        "--setenv=RSNA_PROJECT_ROOT="+str(PROJECT),"--setenv=PYTHONDONTWRITEBYTECODE=1",
        str(PROJECT/".venv/bin/python"),str(PROJECT/"scripts/coordinate_ticket64_recovery_v2.py"),T64_SHA],check=True,timeout=30)


def prepare():
    if (ROOT/"handoff-admission.json").exists():raise FileExistsError("Handoff already frozen")
    plan=read(ROOT/"plan.json");checksum=digest(ROOT/"plan.json")
    decision=read(PROJECT/".scratch/rsna-leaderboard/resource-approvals/ticket66-biomedclip-mri-pilot-20261002.json")
    if decision["decision"]!="approved" or decision["plan_sha256"]!=checksum or not decision["temporary_fit_coordinator_handoff_allowed"]:
        raise ValueError("Exact pilot and scheduling handoff admission required")
    snapshot=ROOT/"controller-source-v1";snapshot.mkdir(mode=0o700)
    shutil.copyfile(__file__,snapshot/Path(__file__).name)
    validator=PROJECT/"scripts/validate_ticket66_biomedclip_mri_pilot.py";shutil.copyfile(validator,snapshot/validator.name)
    pipeline=ROOT/"run-pipeline.sh"
    pipeline.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
export RSNA_TICKET66_DEADLINE_UNIX="$({ROOT}/runtime/bin/python -c 'import time; print(time.time()+900)')"
{ROOT}/runtime/bin/python {plan['source_snapshot']}/ticket66_biomedclip_mri_pilot.py run {checksum}
{ROOT}/runtime/bin/python {snapshot}/validate_ticket66_biomedclip_mri_pilot.py {ROOT} {checksum}
""")
    launcher=ROOT/"run.sbatch";body=launcher.read_text()
    command=f"{ROOT}/runtime/bin/python {plan['source_snapshot']}/ticket66_biomedclip_mri_pilot.py run {checksum}\n"
    if body.count(command)!=1:raise ValueError("Exact unlaunched pilot launcher required")
    launcher.write_text(body.replace(command,f"timeout --signal=TERM --kill-after=10 900 bash {pipeline}\n"))
    jobs=cpu_jobs();watched=[]
    if len(jobs)>2 or not active():raise ValueError("Live fitting coordinator and authorized CPU slots required")
    accepted=list(T64.glob("full-backbone/fold-*/stage-*/accepted-launch.json"))
    for row in jobs:
        if not row[3].startswith("rsna-ticket64-"):raise ValueError("Handoff cannot displace unrelated CPU work")
        path=next(p for p in accepted if str(read(p)["job_id"])==row[0])
        fold=int(path.parent.parent.name.split("-")[-1]);stage=int(path.parent.name.split("-")[-1])
        watched.append(dict(job_id=row[0],directory=str(path.parent),expected_steps=min(7500,(stage+1)*1000),
            accepted_path=str(path),accepted_sha256=digest(path)))
    fixed=[ROOT/"plan.json",launcher,pipeline,*snapshot.iterdir(),
        PROJECT/"scripts/coordinate_ticket64_recovery_v2.py",PROJECT/"scripts/coordinate_ticket64_full_backbone_comparison.py",
        PROJECT/"scripts/monitor_ticket48_cpu_pilot.py",T64/"plan.json"]
    admission=dict(schema_version="rsna-ticket66-cpu-handoff-v1",pilot_plan_sha256=checksum,watched_fits=watched,
        source_sha256=digest(snapshot/Path(__file__).name),frozen_files=[dict(path=str(p),sha256=digest(p)) for p in fixed],
        maximum_owner_cpu_jobs=2,maximum_launch_attempts=1,maximum_wait_for_slot_seconds=21600,
        maximum_queue_wait_seconds=3600,additional_calendar_reserve_seconds=4560,
        science_freeze_utc="2026-10-11T00:00:00+00:00",fit_updates_or_checkpoints_changed=False,
        resume_fitting_immediately_after_pilot_admission=True,automatic_retry=False,candidate_promotion=False)
    write(ROOT/"handoff-admission.json",admission)
    write(PROJECT/".scratch/rsna-leaderboard/resource-approvals/ticket66-cpu-handoff-20261002.json",
        dict(schema_version="rsna-campaign-autonomous-decision-v1",decision="approved",decision_made_by="campaign agent under standing owner authorization",
            authorization_sha256=digest(PROJECT/".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"),
            handoff_admission_sha256=digest(ROOT/"handoff-admission.json"),pilot_plan_sha256=checksum,
            maximum_owner_cpu_jobs=2,maximum_pilot_launches=1,additional_calendar_reserve_seconds=4560,
            basis="Temporary scheduling handoff leaves both live fits running and preserves every fixed stage; single pilot and independent replay consume at most 960 CPU-node seconds, with at most one-hour queue allowance, still before science freeze.",
            new_exact_owner_numeric_approval_claimed=False,automatic_retry=False,gpu_ledger_changed=False,tpu_ledger_changed=False,external_spend_usd=0))
    print(json.dumps(dict(handoff_sha256=digest(ROOT/"handoff-admission.json"),watched_jobs=[r["job_id"] for r in watched])))


def reconcile(admission,job):
    controller=ROOT/f"controller-{job}";terminal=read(controller/"terminal-observation.json")
    text=(controller/"terminal-controller.txt").read_text();fields=dict(p.split("=",1) for p in text.split() if "=" in p)
    expected=dict(JobId=str(job),JobState="COMPLETED",ExitCode="0:0",Partition="cluster1-cpu",NumNodes="1",NumCPUs="16",NumTasks="1",MinMemoryNode="64G",TimeLimit="00:16:00",OverSubscribe="OK",Requeue="0",Restarts="0")
    if terminal["state"]!="COMPLETED" or terminal["returncode"]!=0 or any(fields.get(k)!=v for k,v in expected.items()):
        raise ValueError("Pilot lacks exact successful CPU terminal")
    result=read(ROOT/"pilot-result.json");verification=read(ROOT/"independent-validation.json");plan=read(ROOT/"plan.json")
    error=verification["maximum_native_output_abs_error"]
    if (result["status"]!="COMPLETE" or verification["status"]!="COMPLETE" or result["plan_sha256"]!=admission["pilot_plan_sha256"]
        or verification["plan_sha256"]!=admission["pilot_plan_sha256"]
        or verification["result_sha256"]!=digest(ROOT/"pilot-result.json") or str(result["job_id"])!=str(job)
        or str(verification["job_id"])!=str(job) or verification["verified_studies"]!=8
        or verification["verified_slices"]!=plan["expected_valid_slices"] or not math.isfinite(error) or not 0<=error<=1e-6
        or not 0<result["peak_rss_bytes"]<64*1024**3 or result["candidate_promotion"]
        or result["study_count"]!=8 or result["valid_slices"]!=plan["expected_valid_slices"]
        or not result["strict_state_coverage"] or not verification["original_dicom_replay"]
        or result["fitting"] or result["labels_used"] or result["text_tower_constructed"] or result["network"]
        or result["elapsed_seconds"]+verification["elapsed_seconds"]>900):raise ValueError("Complete bounded independent native evidence required")
    hours,minutes,seconds=map(int,fields["RunTime"].split(":"));elapsed=hours*3600+minutes*60+seconds
    if elapsed>960:raise ValueError("Pilot allocation exceeds admission")
    write(ROOT/"terminal-verification.json",dict(status="COMPLETE",passed=True,job_id=job,pilot_plan_sha256=admission["pilot_plan_sha256"],
        terminal_sha256=digest(controller/"terminal-controller.txt"),result_sha256=digest(ROOT/"pilot-result.json"),
        independent_validation_sha256=digest(ROOT/"independent-validation.json"),allocated_cpu_node_seconds=elapsed,
        gpu_seconds=0,tpu_seconds=0,external_spend_usd=0,candidate_promotion=False))


def watch(checksum):
    if digest(ROOT/"handoff-admission.json")!=checksum:raise ValueError("Exact frozen handoff required")
    admission=read(ROOT/"handoff-admission.json")
    decision=read(PROJECT/".scratch/rsna-leaderboard/resource-approvals/ticket66-cpu-handoff-20261002.json")
    if decision["decision"]!="approved" or decision["handoff_admission_sha256"]!=checksum or digest(__file__)!=admission["source_sha256"]:
        raise ValueError("Approved frozen controller required")
    for row in admission["frozen_files"]:
        if digest(row["path"])!=row["sha256"]:raise ValueError("Frozen handoff source or launcher changed")
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("Handoff interrupted")))
    with (ROOT/"handoff.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/"handoff-start.json").exists():raise FileExistsError("Consumed handoff cannot restart automatically")
        write(ROOT/"handoff-start.json",dict(handoff_sha256=checksum,at_utc=datetime.now(timezone.utc).isoformat()))
        paused=False;job=None;started=time.monotonic();missing_since={}
        try:
            if not active():raise ValueError("Original fit scheduler no longer live; inspect before handoff")
            paused=True;subprocess.run(["systemctl","--user","stop",UNIT],check=True,timeout=30)
            if active():raise ValueError("Fit scheduler did not release scheduling ownership")
            sys.path.insert(0,str(PROJECT/"scripts"))
            from coordinate_ticket64_full_backbone_comparison import stage_state
            while True:
                jobs=cpu_jobs()
                if time.monotonic()-started>admission["maximum_wait_for_slot_seconds"]:raise TimeoutError("CPU-slot wait expired; no pilot launch")
                if datetime.now(timezone.utc)>=datetime.fromisoformat(admission["science_freeze_utc"]):raise TimeoutError("Science freeze reached")
                complete=[]
                for row in admission["watched_fits"]:
                    if digest(row["accepted_path"])!=row["accepted_sha256"]:raise ValueError("Original fitting launch receipt changed")
                    stage=stage_state(Path(row["directory"]),T64_SHA,row["expected_steps"],"05:30:00")
                    complete.append(stage=="complete")
                    if stage=="active" and row["job_id"] not in [r[0] for r in jobs]:
                        if time.monotonic()-missing_since.setdefault(row["job_id"],time.monotonic())>300:
                            raise ValueError("Original fit lacks live queue or retained terminal; no retry")
                    else:missing_since.pop(row["job_id"],None)
                observation=dict(at_utc=datetime.now(timezone.utc).isoformat(),phase="wait_for_cpu_slot",occupied_cpu_jobs=[r[0] for r in jobs],fit_scheduler_temporarily_held=True,fit_jobs_cancelled=False)
                state(observation)
                if len(jobs)<2 and any(complete):break
                time.sleep(15)
            write(ROOT/"launch-intent.json",dict(plan_sha256=admission["pilot_plan_sha256"],handoff_sha256=checksum,launch_attempt=1,launcher_sha256=digest(ROOT/"run.sbatch")))
            job=int(subprocess.check_output(["sbatch","--parsable",str(ROOT/"run.sbatch")],text=True,timeout=30).strip().split(";")[0])
            write(ROOT/"accepted-launch.json",dict(job_id=job,plan_sha256=admission["pilot_plan_sha256"],launch_attempt=1))
            subprocess.run(["systemd-run","--user",f"--unit=rsna-ticket66-{job}-observer","--collect",
                str(PROJECT/".venv/bin/python"),str(PROJECT/"scripts/monitor_ticket48_cpu_pilot.py"),str(ROOT),str(job)],check=True,timeout=30)
        finally:
            if paused:
                resume()
                write(ROOT/"fitting-scheduler-resumed.json",dict(at_utc=datetime.now(timezone.utc).isoformat(),pilot_job_id=job,active=active(),fit_job_restarted=False))
        accepted_at=time.monotonic();controller=ROOT/f"controller-{job}"
        while not (controller/"terminal-observation.json").exists():
            state(dict(at_utc=datetime.now(timezone.utc).isoformat(),phase="wait_for_pilot",pilot_job_id=job,
                fitting_scheduler_active=active(),occupied_cpu_jobs=[r[0] for r in cpu_jobs()],automatic_retry=False))
            if (controller/"latest.json").exists():
                try:job_state=read(controller/"latest.json")["state"]
                except (json.JSONDecodeError,OSError):job_state="transient"
                if job_state=="PENDING" and time.monotonic()-accepted_at>admission["maximum_queue_wait_seconds"]:
                    subprocess.run(["scancel",str(job)],check=True,timeout=30)
                    write(ROOT/"queue-limit-cancellation.json",dict(job_id=job,maximum_queue_wait_seconds=3600,automatic_retry=False))
                    break
            time.sleep(5)
        if (controller/"terminal-observation.json").exists():reconcile(admission,job)
        state(dict(at_utc=datetime.now(timezone.utc).isoformat(),phase="complete" if (ROOT/"terminal-verification.json").exists() else "queue_limit_cancelled",
            pilot_job_id=job,fitting_scheduler_active=active(),terminal_verified=(ROOT/"terminal-verification.json").exists(),automatic_retry=False))
        print(json.dumps(dict(pilot_job_id=job,fitting_scheduler_active=active(),terminal_verified=(ROOT/"terminal-verification.json").exists())))


if __name__=="__main__":
    if sys.argv[1]=="prepare":prepare()
    elif sys.argv[1]=="watch":watch(sys.argv[2])
    else:raise ValueError("prepare or watch required")
