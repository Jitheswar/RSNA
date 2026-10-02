"""Prespecified longer whole-backbone comparison using the shared CPU trainer.

Preparation requires a terminal, independently verified Ticket 63 result.
No fitting request is made by preparation; every stage has a consumed launch.
"""

from dataclasses import replace
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from rsna_knee import ticket36 as t36
from rsna_knee.evaluation import predict_training_examples
from rsna_knee.folds import load_fold_manifest
from rsna_knee.image_model_comparison import compare_image_model_predictions
from rsna_knee.model import build_study_model
from rsna_knee.resnet18 import RESNET18_MODEL_ID
from rsna_knee.training import (
    TrainingExample, TrainingInterrupted, SupervisionProvenance, _batch_indices,
    _model_state_sha256, fingerprint_training_examples, load_training_checkpoint,
    train_fixed_budget,
)
from ticket63_full_backbone_training_cost import (
    PROJECT, PIXELS, PARENT, PILOT, AUTH, config, digest, read, write,
    load_bank, examples_for_fold, initial_model,
)


ROOT = PROJECT / "artifacts/private/ticket64-full-backbone-comparison-v1"
COST = PROJECT / "artifacts/private/ticket63-full-backbone-training-cost-v1"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket64-full-backbone-comparison-20261002.json"
SPEC = PROJECT / ".scratch/rsna-leaderboard/issues/64-test-long-full-backbone-report-adaptation.md"
ARMS = ("head-continuation", "full-backbone")
STEPS = 7500


def terminal(root, checksum, limit):
    accepted = read(root / "accepted-launch.json")
    job = str(accepted["job_id"])
    controller = root / f"controller-{job}/terminal-controller.txt"
    observation = read(controller.with_name("terminal-observation.json"))
    fields = dict(t.split("=", 1) for t in controller.read_text().split() if "=" in t)
    expected = dict(JobId=job, JobState="COMPLETED", ExitCode="0:0", Partition="cluster1-cpu",
        NumNodes="1", NumCPUs="16", NumTasks="1", MinMemoryNode="64G", TimeLimit=limit,
        OverSubscribe="OK", Requeue="0", Restarts="0")
    if (accepted["plan_sha256"] != checksum or observation["job_id"] != job
            or observation["state"] != "COMPLETED" or observation["returncode"] != 0
            or any(fields.get(k) != v for k, v in expected.items())
            or "gpu" in fields.get("TRES", "").lower()):
        raise ValueError("Exact successful source/resource/terminal admission required")
    elapsed = (datetime.fromisoformat(fields["EndTime"]) - datetime.fromisoformat(fields["StartTime"])).total_seconds()
    return accepted, controller, elapsed


def time_string(seconds):
    return f"{seconds//3600:02}:{seconds%3600//60:02}:{seconds%60:02}"


def resource_projection(admission, records, now):
    intervals = [v for r in records for v in r["step_and_prior_checkpoint_intervals_seconds"][:r["successful_steps"]]]
    step = max(intervals)
    startup = max(r["pre_training_seconds"] for r in records)
    checkpoint_bytes = max(r["checkpoint_bytes"] for r in records)
    peak = max(r["peak_rss_bytes"] for r in records)
    if not all(math.isfinite(v) and v > 0 for v in [step, startup, checkpoint_bytes, peak]):
        raise ValueError("Finite actual shared-trainer cost observations required")
    # Includes serialization at every diagnostic step; the future trainer also
    # checkpoints after short gold epochs, not merely every 500 updates.
    stage = next((n for n in (1500, 1250, 1000, 750, 500)
        if 2*step*n+2*startup+600+120 <= 21600), None)
    if stage is None:
        raise ValueError("Actual cost cannot admit the prespecified duration with bounded stages")
    stages = math.ceil(STEPS/stage)
    wall = math.ceil((2*step*stage+2*startup+600+120)/900)*900
    feature_memory = 4228*16*8*512*4
    if 2*peak+feature_memory >= 60*1024**3:
        raise ValueError("Actual full-cohort memory does not admit simultaneous features/training")
    epochs = max(math.ceil(STEPS/math.ceil(f["reference_studies"]/2)) for f in admission["folds"])
    history_count = 1+epochs+math.ceil(STEPS/500)+math.ceil(2*step*STEPS/900)+stages
    # Six comparison fits and one conditional final-fit reserve, retaining all
    # checkpoint histories, latest/final files and concurrent write temporaries.
    checkpoint_storage = (history_count+4)*checkpoint_bytes*7
    control_wall = 7200
    independent_replay_wall = 5400
    offline_delivery_reserve = 7200
    max_node_seconds = stages*wall*4+control_wall+independent_replay_wall+offline_delivery_reserve
    # Three treatment folds in two waves on the two authorized CPU slots,
    # followed by one conditional final fit. Reserve six hours for existing
    # allocations/queue transition, then one full evidence-review day.
    calendar_seconds = stages*wall*3+control_wall+independent_replay_wall+offline_delivery_reserve+21600
    freeze = datetime(2026,10,11,tzinfo=timezone.utc)
    if now.timestamp()+calendar_seconds+86400 >= freeze.timestamp():
        raise ValueError("Complete conservative route lacks calendar headroom before science freeze")
    return dict(observed_maximum_step_and_prior_checkpoint_seconds=step,
        observed_maximum_stage_startup_seconds=startup, cost_multiplier=2,
        steps_per_stage=stage, full_stages_per_fold=stages, full_stage_wall_seconds=wall,
        full_stage_watchdog_seconds=wall-60, history_count_upper_bound_per_fit=history_count,
        maximum_checkpoint_bytes=checkpoint_bytes, checkpoint_storage_allowance_bytes=checkpoint_storage,
        feature_bank_memory_bytes=feature_memory, observed_peak_rss_bytes=peak,
        projected_peak_rss_bytes=2*peak+feature_memory,
        control_wall_seconds=control_wall, independent_replay_wall_seconds=independent_replay_wall,
        offline_delivery_reserve_seconds=offline_delivery_reserve,
        conditional_final_fit_stages=stages, conditional_final_fit_authorized=False,
        maximum_reserved_cpu_node_seconds=max_node_seconds,
        maximum_projected_calendar_seconds=calendar_seconds,
        calendar_admission="two authorized CPU slots, two treatment waves, conditional final fit, six-hour queue transition and one review day before October 11",
        science_freeze_utc=freeze.isoformat(), gpu_delivery_authorized=False)


def prepare():
    if ROOT.exists() or DECISION.exists():
        raise FileExistsError("Immutable longer comparison request already exists")
    cost_plan = read(COST / "plan.json")
    cost = read(COST / "cost-result.json")
    verification = read(COST / "terminal-verification.json")
    accepted, controller, _ = terminal(COST,digest(COST / "plan.json"),"01:30:00")
    if (cost["status"] != "COMPLETE" or not verification["passed"]
            or cost["plan_sha256"] != digest(COST / "plan.json")
            or cost["job_id"] != str(accepted["job_id"])
            or not cost["exact_fresh_process_staged_model_optimizer_progress_replay"]
            or not cost["normalization_buffers_unchanged"]
            or cost["held_out_predictions"]):
        raise ValueError("Complete matching real-MRI shared trainer evidence required")
    for item in cost_plan["frozen_files"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Admitted shared-trainer source/input changed")
    admission = read(COST / "admission-result.json")
    records = [read(COST / f"{s}-result.json") for s in ("interrupt","resume","continuous")]
    if (admission["loaded_studies"] != 4228 or admission["plan_sha256"] != cost["plan_sha256"]
            or cost["admission_result_sha256"] != digest(COST / "admission-result.json")
            or any(cost["stage_result_sha256"][s] != digest(COST / f"{s}-result.json")
                for s in ("interrupt","resume","continuous"))):
        raise ValueError("Actual all-fold construction and stage measurements required")
    projection = resource_projection(admission, records, datetime.now(timezone.utc))
    usage = subprocess.check_output(["du","--apparent-size","-s","-B1",str(PROJECT),
        str(PROJECT.parent / "models/qwen3-8b-b968826d9c46dd6066d109eabc6255188de91218")],text=True)
    existing = sum(int(line.split()[0]) for line in usage.splitlines())
    additional = projection["checkpoint_storage_allowance_bytes"]+5_000_000_000
    if existing+additional >= 2_000_000_000_000 or additional >= shutil.disk_usage(PROJECT).free:
        raise ValueError("Fresh complete checkpoint/delivery storage does not fit")
    ROOT.mkdir(mode=0o700,parents=True)
    snapshot = ROOT / "source-snapshot"
    shutil.copytree(PROJECT / "src/rsna_knee", snapshot / "src/rsna_knee",
        ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    for script in (Path(__file__), PROJECT / "scripts/ticket63_full_backbone_training_cost.py"):
        shutil.copyfile(script,snapshot / script.name)
    shutil.copyfile(SPEC,ROOT / "prospective-spec.md")
    parent_plan = Path(read(PARENT / "plan.json")["parent_plan"])
    files = [AUTH,ROOT / "prospective-spec.md",COST / "plan.json",COST / "cost-result.json",
        COST / "terminal-verification.json",COST / "admission-result.json",controller,
        controller.with_name("terminal-observation.json"),PIXELS / "plan.json",
        PIXELS / "pixel-manifest.json",PIXELS / "independent-validation.json",
        PIXELS / "terminal-verification.json",PARENT / "plan.json",PARENT / "validation.json",
        t36.FOLD_MANIFEST_PATH,t36.TICKET30_ARTIFACT,t36.WEIGHTS_PATH,
        t36.FEATURE_ROOT / "feature-manifest.json",parent_plan,PILOT / "plan.json"]
    files.extend(COST / f"{s}-result.json" for s in ("interrupt","resume","continuous"))
    for fold in range(3):
        files.extend([PARENT / f"report-supervision/fold-{fold}/run.json",
            PARENT / f"report-supervision/fold-{fold}/checkpoints/final.pt"])
    files.extend(p for p in snapshot.rglob("*") if p.is_file())
    plan = dict(schema_version="rsna-ticket64-long-full-backbone-comparison-plan-v1",
        created_at_utc=datetime.now(timezone.utc).isoformat(),run_root=str(ROOT),source_snapshot=str(snapshot),
        decision_path=str(DECISION),authorization_sha256=digest(AUTH),pixel_root=str(PIXELS),
        cohort_parent_plan=str(parent_plan),projection=projection,arms=list(ARMS),successful_steps=STEPS,
        shared_training_config=replace(config(STEPS),checkpoint_interval_steps=500).to_dict(),
        stage_boundary_policy="exact planned pause publication; fixed scientific budget",
        input_fingerprints={str(r["fold"]):r["training_example_fingerprints"] for r in admission["folds"]},
        analysis=dict(bootstrap_resamples=2000,seed=20261002,maximum_prediction_error=1e-6,
            intermediate_evaluation=False,automatic_promotion=False,reused_development_cohort=True),
        resources=dict(partition="cluster1-cpu",nodes=1,tasks=1,cpus_per_task=16,memory_gib=64,
            maximum_concurrent_cpu_jobs=2,maximum_launches_per_stage=1,exclusive=False,requeue=False,
            gpu_count=0,external_spend_usd=0),
        storage=dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),du_output=usage,
            existing_apparent_bytes=existing,additional_allowance_bytes=additional,
            projected_campaign_peak_bytes=existing+additional,soft_cap_bytes=2_000_000_000_000),
        runtime_versions={n:importlib.metadata.version(n) for n in ("numpy","torch")},
        frozen_files=[dict(path=str(p),sha256=digest(p)) for p in files],
        new_downloads=False,network_during_execution=False,competition_submission=False,
        candidate_promotion=False,conditional_final_fit_authorized=False)
    write(ROOT / "plan.json",plan)
    checksum = digest(ROOT / "plan.json")
    write(DECISION,dict(schema_version="rsna-campaign-autonomous-decision-v1",decision="approved",
        decision_made_by="campaign agent under standing owner authorization",authorization_sha256=digest(AUTH),
        plan_sha256=checksum,resources=plan["resources"],projection=projection,
        basis="Complete long matched whole-backbone comparison from independently verified actual mixed-source startup/step/checkpoint and full-pixel admission, with explicit complete history/storage/calendar/final-delivery reserves.",
        new_exact_owner_numeric_approval_claimed=False,gpu_ledger_changed=False,tpu_ledger_changed=False,
        conditional_final_fit_authorized=False,candidate_promotion=False))
    launchers=[]
    for fold in range(3):
        for stage in range(projection["full_stages_per_fold"]):
            directory=ROOT / f"full-backbone/fold-{fold}/stage-{stage}"
            directory.mkdir(parents=True,mode=0o700)
            launchers.append(make_launcher(directory,"full",checksum,projection["full_stage_wall_seconds"],fold,stage))
    directory=ROOT / "control";directory.mkdir(mode=0o700)
    launchers.append(make_launcher(directory,"control",checksum,projection["control_wall_seconds"]))
    write(ROOT / "prepared-validation.json",dict(plan_sha256=checksum,launchers=launchers,launched=False))
    print(json.dumps(dict(plan_sha256=checksum,projection=projection,launched=False)))


def make_launcher(directory,action,checksum,wall,fold=None,stage=None):
    snapshot=ROOT / "source-snapshot"
    args=f" {fold} {stage}" if action=="full" else ""
    script=directory / "run.sbatch"
    with script.open("x") as handle:
        handle.write(f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket64-{action}-{fold}-{stage}
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time={time_string(wall)}
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
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
{PROJECT}/.venv/bin/python {snapshot / Path(__file__).name} {action} {checksum}{args}
''')
    subprocess.run(["bash","-n",str(script)],check=True)
    return dict(path=str(script),sha256=digest(script),action=action,fold=fold,stage=stage)


def launch(checksum,action,fold=None,stage=None):
    plan=read(ROOT / "plan.json");projection=plan["projection"]
    decision=read(plan["decision_path"])
    if (digest(ROOT / "plan.json") != checksum or decision["decision"] != "approved"
            or decision["plan_sha256"] != checksum or decision["resources"] != plan["resources"]
            or decision["projection"] != projection or digest(AUTH) != plan["authorization_sha256"]):
        raise ValueError("Exact frozen fitting admission required")
    if action=="full":
        if fold not in (0,1,2) or not isinstance(stage,int) or not 0 <= stage < projection["full_stages_per_fold"]:
            raise ValueError("Unknown registered full-backbone stage")
        directory=ROOT / f"full-backbone/fold-{fold}/stage-{stage}"
        if stage:
            previous=directory.parent / f"stage-{stage-1}"
            terminal(previous,checksum,time_string(projection["full_stage_wall_seconds"]))
            record=read(previous / "stage-result.json")
            if (record["status"] != "COMPLETE" or record["plan_sha256"] != checksum
                    or record["successful_steps"] != stage*projection["steps_per_stage"]
                    or record["checkpoint_sha256"] != digest(record["checkpoint_path"])):
                raise ValueError("Exact completed prior stage required")
    elif action=="control":
        directory=ROOT / "control"
    else:
        raise ValueError("Only the registered comparison arms can launch")
    prepared=read(ROOT / "prepared-validation.json")
    row=next(r for r in prepared["launchers"] if r["path"]==str(directory / "run.sbatch"))
    if prepared["plan_sha256"] != checksum or digest(row["path"]) != row["sha256"]:
        raise ValueError("Exact frozen generated launcher required")
    for item in plan["frozen_files"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Frozen source/admission changed")
    jobs=subprocess.check_output(["squeue","-u",os.environ["USER"],"-h","-o","%i %P %t"],text=True).splitlines()
    active=[r for r in jobs if r.split()[1]=="cluster1-cpu"]
    if len(active)>=2:
        raise RuntimeError("Both authorized CPU slots remain occupied; preserve existing jobs")
    # No retries: creating this intent consumes the exact stage's sole launch.
    write(directory / "launch-intent.json",dict(plan_sha256=checksum,launcher_sha256=row["sha256"],
        launch_attempt=1,at_utc=datetime.now(timezone.utc).isoformat(),observed_active_cpu_jobs=active))
    receipt=subprocess.check_output(["sbatch","--parsable",row["path"]],text=True).strip()
    job=int(receipt.split(";")[0])
    write(directory / "accepted-launch.json",dict(job_id=job,plan_sha256=checksum,launch_attempt=1,
        launcher_sha256=row["sha256"],accepted_at_utc=datetime.now(timezone.utc).isoformat()))
    unit=f"rsna-ticket64-{action}-{fold}-{stage}-{job}-observer"
    subprocess.run(["systemd-run","--user",f"--unit={unit}","--collect",str(PROJECT / ".venv/bin/python"),
        str(PROJECT / "scripts/monitor_ticket48_cpu_pilot.py"),str(directory),str(job)],check=True)
    print(json.dumps(dict(job_id=job,plan_sha256=checksum,action=action,fold=fold,stage=stage)))


def runtime(checksum, wall):
    if digest(ROOT / "plan.json") != checksum:
        raise ValueError("Frozen comparison plan changed")
    plan = read(ROOT / "plan.json")
    decision = read(plan["decision_path"])
    if (decision["decision"] != "approved" or decision["plan_sha256"] != checksum
            or decision["resources"] != plan["resources"] or decision["projection"] != plan["projection"]
            or digest(AUTH) != plan["authorization_sha256"]):
        raise ValueError("Exact comparison/resource decision changed")
    for item in plan["frozen_files"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input changed: {item['path']}")
    if (Path(__file__).resolve().parent != Path(plan["source_snapshot"])
            or os.environ.get("PYTHONPATH") != str(Path(plan["source_snapshot"]) / "src")
            or any(importlib.metadata.version(n) != v for n,v in plan["runtime_versions"].items())):
        raise ValueError("Frozen source/runtime required")
    job = os.environ["SLURM_JOB_ID"]
    controller = subprocess.check_output(["scontrol","show","job",job,"-o"],text=True)
    fields = dict(t.split("=",1) for t in controller.split() if "=" in t)
    expected = dict(JobId=job,JobState="RUNNING",Partition="cluster1-cpu",NumNodes="1",NumCPUs="16",
        NumTasks="1",MinMemoryNode="64G",TimeLimit=time_string(wall),OverSubscribe="OK",Requeue="0",Restarts="0")
    if any(fields.get(k) != v for k,v in expected.items()) or "gpu" in fields.get("TRES","").lower():
        raise ValueError("Actual bounded CPU allocation differs")
    torch.set_num_threads(16);torch.set_num_interop_threads(1)
    if torch.cuda.is_available() or torch.version.cuda is not None:
        raise ValueError("CPU-only fitting required")
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError("Fit watchdog")))
    signal.alarm(wall-60)
    return plan,controller


def feature_bank(bank):
    manifest = read(t36.FEATURE_ROOT / "feature-manifest.json")
    result = {}
    for row in manifest["features"]:
        uid = row["study_instance_uid"]
        if uid not in bank:
            continue
        entry = t36._load_feature_file(t36.FEATURE_ROOT / row["feature_file"],study_uid=uid,
            batch_index=row["batch_index"],feature_identity_fingerprint=manifest["identity_fingerprint"])
        pixels = bank[uid]
        if (entry["file_sha256"] != row["feature_file_sha256"]
                or entry["metadata"]["source_identity_sha256"] != pixels["images"].expected_source_identity_sha256
                or any(not np.array_equal(entry[k],pixels[k]) for k in ("slice_mask","series_mask","series_descriptors"))):
            raise ValueError("Matched control feature/source/packing identity differs")
        result[uid] = entry["features"]
    if set(result) != set(bank):
        raise ValueError("Matched feature cohort incomplete")
    return result


def fold_inputs(plan,fold):
    row = next(r for r in read(plan["cohort_parent_plan"])["folds"] if r["fold"] == fold)
    selected = set(row["gold_training_study_uids"]+row["weak_training_study_uids"])
    bank = load_bank(plan,selected)
    gold,weak = examples_for_fold(plan,bank,fold)
    fingerprints = [fingerprint_training_examples(pool) for pool in (gold,weak)]
    if fingerprints != plan["input_fingerprints"][str(fold)]:
        raise ValueError("Complete admitted scientific inputs changed")
    return bank,gold,weak


def config_for_arm(arm):
    full = replace(config(STEPS),checkpoint_interval_steps=500)
    return full if arm == "full-backbone" else replace(full,model_id=RESNET18_MODEL_ID,encoder_learning_rate=None)


def identity(plan,checksum,fold,arm):
    return dict(plan_sha256=checksum,fold=fold,arm=arm,input_fingerprints=plan["input_fingerprints"][str(fold)],
        pixel_manifest_sha256=digest(PIXELS / "pixel-manifest.json"),
        source_feature_manifest_sha256=digest(t36.FEATURE_ROOT / "feature-manifest.json"),
        warm_checkpoint_sha256=digest(PARENT / f"report-supervision/fold-{fold}/checkpoints/final.pt"))


def initial_for_arm(fold,arm):
    full = initial_model(fold)
    if arm == "full-backbone":
        return full
    frozen = build_study_model(config_for_arm(arm).model_config())
    frozen.load_state_dict(full.state_dict(),strict=True)
    return frozen


def verify_initial_batch(model,gold,weak,features):
    batch = [pool[i] for pool,seed in ((gold,20260912),(weak,20260912+34034034))
        for i in _batch_indices(len(pool),2,0,seed)]
    tensors = [torch.from_numpy(np.stack([getattr(e,k) for e in batch]))
        for k in ("images","slice_mask","series_mask","series_descriptors")]
    encoded = torch.from_numpy(np.stack([features[e.study_instance_uid] for e in batch]))
    model.eval()
    with torch.inference_mode():
        error = float((model(*tensors)-model.forward_from_features(encoded,*tensors[1:])).abs().max())
    if error > 1e-6:
        raise ValueError("Actual warm native/control initial logits differ")
    return error


def final_predictions(model,plan,fold,arm):
    manifest = load_fold_manifest(t36.FOLD_MANIFEST_PATH)
    uids = sorted(manifest.held_out_study_uids(fold))
    bank = load_bank(plan,set(uids))
    features = feature_bank(bank) if arm == "head-continuation" else None
    gold = {r.study_instance_uid:r for r in manifest.gold_memberships}
    examples = [TrainingExample(uid,**{**bank[uid],"images":None if features is not None else bank[uid]["images"]},
        targets=np.asarray(gold[uid].reference_labels,dtype=np.float32),supervision_mask=np.ones(12,dtype=np.bool_),
        provenance=SupervisionProvenance(),features=None if features is None else features[uid]) for uid in uids]
    tick = time.monotonic()
    predictions = {uid:v.tolist() for uid,v in predict_training_examples(model,examples,batch_size=2).items()}
    native_seconds = time.monotonic()-tick
    ablated = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(examples),2):
            batch = examples[start:start+2]
            tensors = [torch.from_numpy(np.stack([getattr(e,k) for e in batch]))
                for k in ("slice_mask","series_mask","series_descriptors")]
            scores = model.predict_proba_from_features(torch.zeros((len(batch),16,8,512)),*tensors).numpy()
            ablated.update({e.study_instance_uid:v.tolist() for e,v in zip(batch,scores,strict=True)})
    return dict(predictions=predictions,encoded_image_ablation=ablated,held_out_inference_seconds=native_seconds)


def full_stage(checksum,fold,stage):
    projection = read(ROOT / "plan.json")["projection"]
    plan,controller = runtime(checksum,projection["full_stage_wall_seconds"])
    if fold not in (0,1,2) or not 0 <= stage < projection["full_stages_per_fold"]:
        raise ValueError("Unknown exact registered fold/stage")
    directory = ROOT / "full-backbone" / f"fold-{fold}"
    stage_root = directory / f"stage-{stage}"
    if (stage_root / "stage-result.json").exists():
        raise FileExistsError("Consumed full-backbone stage")
    start = time.monotonic();bank,gold,weak = fold_inputs(plan,fold)
    model = initial_for_arm(fold,"full-backbone")
    initial_state = _model_state_sha256(model)
    initial_buffers = {k:v.clone() for k,v in model.state_dict().items() if k.endswith(("running_mean","running_var","num_batches_tracked"))}
    parity = None
    if stage == 0:
        selected = {pool[i].study_instance_uid for pool,seed in ((gold,20260912),(weak,20260912+34034034))
            for i in _batch_indices(len(pool),2,0,seed)}
        parity = verify_initial_batch(model,gold,weak,feature_bank({k:v for k,v in bank.items() if k in selected}))
    cfg = config_for_arm("full-backbone");stop = min(STEPS,(stage+1)*projection["steps_per_stage"])
    checkpoint_path = directory / "training/checkpoints/latest.pt"
    previous_sha = None
    if stage:
        prior = read(directory / f"stage-{stage-1}/stage-result.json")
        terminal(directory / f"stage-{stage-1}",checksum,time_string(projection["full_stage_wall_seconds"]))
        previous_sha = digest(checkpoint_path)
        if prior["successful_steps"] != stage*projection["steps_per_stage"] or prior["checkpoint_sha256"] != previous_sha:
            raise ValueError("Prior exact staged state differs")
    tick = time.monotonic();stopped = False
    try:
        result = train_fixed_budget(gold,cfg,additional_examples=weak,additional_batch_order_seed=20260912+34034034,
            model=model,artifact_dir=directory / "training",fold=fold,identity=identity(plan,checksum,fold,"full-backbone"),
            resume_from=checkpoint_path if stage else None,requested_resources=plan["resources"],
            interrupt_after_steps=stop if stop < STEPS else None)
    except TrainingInterrupted:
        if stop == STEPS:
            raise
        stopped = True
    trainer_seconds = time.monotonic()-tick
    path = checkpoint_path if stopped else directory / "training/checkpoints/final.pt"
    checkpoint = load_training_checkpoint(path,model=model,expected_config=cfg,expected_fold=fold)
    if checkpoint["progress"]["successful_optimizer_steps"] != stop or checkpoint["progress"]["attempted_batches"] != stop:
        raise ValueError("Exact planned stage counter was not durably published")
    if not all(torch.equal(model.state_dict()[k],v) for k,v in initial_buffers.items()):
        raise ValueError("Admitted encoder normalization changed")
    changed = None
    if stop == STEPS:
        initial = initial_model(fold).state_dict()
        changed = {prefix:any(not torch.equal(v,initial[k]) for k,v in model.state_dict().items() if k.startswith(prefix))
            for prefix in ("encoder.conv1","encoder.layer1","encoder.layer2","encoder.layer3","encoder.layer4","head")}
        if not all(changed.values()):
            raise ValueError("Registered whole-backbone stages did not change")
    prediction = final_predictions(model,plan,fold,"full-backbone") if stop == STEPS else {}
    write(stage_root / "stage-result.json",dict(status="COMPLETE",plan_sha256=checksum,job_id=os.environ["SLURM_JOB_ID"],
        fold=fold,stage=stage,controller=controller,successful_steps=stop,planned_boundary=stopped,
        scientific_config_fingerprint=cfg.config_fingerprint,checkpoint_sha256=digest(path),checkpoint_path=str(path),
        prior_checkpoint_sha256=previous_sha,initial_model_state_sha256=initial_state,initial_native_feature_logit_error=parity,
        trainer_seconds=trainer_seconds,elapsed_seconds=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        progress=checkpoint["progress"],normalization_buffers_unchanged=True,all_encoder_stages_changed=changed,
        intermediate_prediction=False,**prediction,candidate_promotion=False))


def control_run(checksum):
    projection = read(ROOT / "plan.json")["projection"]
    plan,controller = runtime(checksum,projection["control_wall_seconds"])
    start = time.monotonic();records=[]
    for fold in range(3):
        bank,gold,weak = fold_inputs(plan,fold);features = feature_bank(bank)
        model = initial_for_arm(fold,"head-continuation");initial_state = _model_state_sha256(model)
        initial_encoder = {k:v.clone() for k,v in model.encoder.state_dict().items()}
        gold,weak = [[replace(e,images=None,features=features[e.study_instance_uid]) for e in pool] for pool in (gold,weak)]
        directory = ROOT / "head-continuation" / f"fold-{fold}"
        result = train_fixed_budget(gold,config_for_arm("head-continuation"),additional_examples=weak,
            additional_batch_order_seed=20260912+34034034,model=model,artifact_dir=directory / "training",fold=fold,
            identity=identity(plan,checksum,fold,"head-continuation"),requested_resources=plan["resources"])
        assert result.successful_optimizer_steps == STEPS and result.skipped_batches == 0
        assert all(torch.equal(v,model.encoder.state_dict()[k]) for k,v in initial_encoder.items())
        predictions = final_predictions(model,plan,fold,"head-continuation")
        path = directory / "training/checkpoints/final.pt"
        checkpoint = load_training_checkpoint(path,expected_config=config_for_arm("head-continuation"),expected_fold=fold)
        record = dict(status="COMPLETE",plan_sha256=checksum,job_id=os.environ["SLURM_JOB_ID"],fold=fold,
            controller=controller,initial_model_state_sha256=initial_state,checkpoint_path=str(path),checkpoint_sha256=digest(path),
            progress=checkpoint["progress"],encoder_unchanged=True,**predictions)
        write(directory / "run.json",record);records.append(dict(fold=fold,run_sha256=digest(directory / "run.json")))
        del bank,gold,weak,features,model,result,checkpoint;gc.collect()
    write(ROOT / "control/stage-result.json",dict(status="COMPLETE",plan_sha256=checksum,
        job_id=os.environ["SLURM_JOB_ID"],controller=controller,folds=records,elapsed_seconds=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,candidate_promotion=False))


def analyze(checksum):
    plan = read(ROOT / "plan.json")
    if digest(ROOT / "plan.json") != checksum:
        raise ValueError("Comparison plan changed")
    projection=plan["projection"]
    terminal(ROOT / "control",checksum,time_string(projection["control_wall_seconds"]))
    all_predictions={arm:{} for arm in ARMS};ablated={arm:{} for arm in ARMS};parent_predictions={};records=[]
    for fold in range(3):
        treatment=read(ROOT / f"full-backbone/fold-{fold}/stage-{projection['full_stages_per_fold']-1}/stage-result.json")
        control=read(ROOT / f"head-continuation/fold-{fold}/run.json")
        for stage in range(projection["full_stages_per_fold"]):
            terminal(ROOT / f"full-backbone/fold-{fold}/stage-{stage}",checksum,time_string(projection["full_stage_wall_seconds"]))
        for arm,record in zip(ARMS,(control,treatment),strict=True):
            if (record["status"] != "COMPLETE" or record["plan_sha256"] != checksum
                    or record["progress"]["successful_optimizer_steps"] != STEPS
                    or record["checkpoint_sha256"] != digest(record["checkpoint_path"])):
                raise ValueError("Complete exact final state required")
            all_predictions[arm].update(record["predictions"]);ablated[arm].update(record["encoded_image_ablation"])
            records.append(dict(arm=arm,fold=fold,checkpoint_sha256=record["checkpoint_sha256"]))
        if control["initial_model_state_sha256"] != treatment["initial_model_state_sha256"]:
            raise ValueError("Matched initial tensors differ")
        for key in ("attempted_batches","sample_exposures","supervised_entry_exposures","exposures_by_source"):
            if control["progress"][key] != treatment["progress"][key]:
                raise ValueError("Matched source/target exposures differ")
        parent_predictions.update(read(PARENT / f"report-supervision/fold-{fold}/run.json")["predictions"])
    manifest=load_fold_manifest(t36.FOLD_MANIFEST_PATH)
    def compare(base,new):
        return compare_image_model_predictions(base,new,manifest,bootstrap_resamples=2000,bootstrap_seed=20261002)
    comparisons=dict(primary=compare(all_predictions["head-continuation"],all_predictions["full-backbone"]))
    for arm in ARMS:
        comparisons[f"{arm}-vs-parent"]=compare(parent_predictions,all_predictions[arm])
        comparisons[f"{arm}-image-signal"]=compare(ablated[arm],all_predictions[arm])
    primary=comparisons["primary"]
    lower=primary["paired_study_bootstrap"]["macro_auc"]["lower"]
    passed=(lower is not None and lower>0 and all(r["delta_macro_auc"] is not None and r["delta_macro_auc"]>0 for r in primary["delta"]["folds"])
        and all(comparisons[f"{arm}-image-signal"]["paired_study_bootstrap"]["macro_auc"]["lower"] is not None
            and comparisons[f"{arm}-image-signal"]["paired_study_bootstrap"]["macro_auc"]["lower"]>0 for arm in ARMS))
    write(ROOT / "analysis.json",dict(status="COMPLETE",plan_sha256=checksum,comparisons=comparisons,run_records=records,
        provisional_review_gate_passed=passed,independent_original_mri_replay_still_required=True,
        automatic_promotion=False,candidate_promotion=False,first_place_established=False))


if __name__ == "__main__":
    action=sys.argv[1]
    if action=="prepare":prepare()
    elif action=="full":full_stage(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]))
    elif action=="control":control_run(sys.argv[2])
    elif action=="analyze":analyze(sys.argv[2])
    elif action=="launch":launch(sys.argv[2],sys.argv[3],int(sys.argv[4]) if len(sys.argv)>4 else None,
        int(sys.argv[5]) if len(sys.argv)>5 else None)
    else:raise ValueError("Unknown exact comparison action")
