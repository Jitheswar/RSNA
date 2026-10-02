"""Measure the real shared trainer's complete pixel admission and staged delivery.

The three diagnostic updates are never an accuracy experiment or a candidate.
Each stage runs in a fresh process with the complete, original fold-0 pools.
"""

from datetime import datetime, timezone
import gc
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

import numpy as np
import torch

from rsna_knee import ticket36 as t36
from rsna_knee.folds import load_fold_manifest
from rsna_knee.model import build_study_model
from rsna_knee.pixel_cache import PackedPixelArray, load_pixel_entry
from rsna_knee.resnet18 import RESNET18_FULL_FINETUNE_MODEL_ID
from rsna_knee.ticket34 import admit_ticket30_fold_supervision
from rsna_knee.training import (
    TrainingConfig, TrainingExample, TrainingInterrupted, SupervisionProvenance,
    _array_sha256, _model_state_sha256, fingerprint_training_examples,
    load_training_checkpoint, seed_everything, train_fixed_budget,
)


PROJECT = t36.PROJECT_ROOT
ROOT = PROJECT / "artifacts/private/ticket63-full-backbone-training-cost-v1"
PIXELS = PROJECT / "artifacts/private/ticket62-mri-pixel-cohort-v1"
PARENT = PROJECT / "artifacts/private/ticket55-resnet-long-report-head-v2"
PILOT = PROJECT / "artifacts/private/ticket61-cpu-full-backbone-pilot-v1"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket63-full-backbone-training-cost-20261002.json"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def config(steps=3):
    model = read(PILOT / "plan.json")["model"]
    return TrainingConfig(
        image_size=(224, 224), channels=tuple(model["channels"]),
        successful_optimizer_steps=steps, learning_rate=1e-4, encoder_learning_rate=1e-5,
        model_id=RESNET18_FULL_FINETUNE_MODEL_ID, pooling_id=model["pooling_id"],
        encoder_id=model["encoder_id"], encoder_weights_sha256=model["encoder_weights_sha256"],
        head_hidden_size=model["head_hidden_size"], seed=20261002,
        batch_order_seed=20260912, checkpoint_interval_steps=1,
        checkpoint_interval_seconds=900, device="cpu",
    )


def prepare():
    if ROOT.exists() or DECISION.exists():
        raise FileExistsError("Immutable shared-trainer cost request already exists")
    validation = read(PIXELS / "independent-validation.json")
    terminal = read(PIXELS / "terminal-verification.json")
    if (validation["status"] != "COMPLETE" or not terminal["passed"]
            or terminal["independent_validation_sha256"] != digest(PIXELS / "independent-validation.json")
            or validation["manifest_sha256"] != digest(PIXELS / "pixel-manifest.json")
            or not validation["all_original_source_pixel_replays_exact"]):
        raise ValueError("Complete independently verified exact MRI pixel admission required")
    resources = dict(partition="cluster1-cpu", nodes=1, tasks=1, cpus_per_task=16,
        memory_gib=64, walltime_seconds=5400, code_watchdog_seconds=5340,
        maximum_launch_attempts=1, maximum_concurrent_jobs=1, exclusive=False,
        requeue=False, gpu_count=0, external_spend_usd=0)
    # Multiple complete reads/fingerprints and four fresh processes, plus exact
    # real-MRI staged versus uninterrupted replay. Use observed full-cache cost.
    forecast = 16 * validation["all_cache_read_reconstruction_seconds"] + 600
    if forecast > resources["code_watchdog_seconds"]:
        raise ValueError("Measured complete-cache cost does not admit the bounded pilot")
    ROOT.mkdir(mode=0o700, parents=True)
    snapshot = ROOT / "source-snapshot"
    shutil.copytree(PROJECT / "src/rsna_knee", snapshot / "src/rsna_knee",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(__file__, snapshot / Path(__file__).name)
    cohort_parent = Path(read(PARENT / "plan.json")["parent_plan"])
    fixed = [AUTH, PIXELS / "plan.json", PIXELS / "pixel-manifest.json",
        PIXELS / "independent-validation.json", PIXELS / "terminal-verification.json",
        PILOT / "plan.json", PILOT / "pilot-result.json", PILOT / "terminal-verification.json",
        PARENT / "plan.json", PARENT / "validation.json", cohort_parent,
        t36.FOLD_MANIFEST_PATH, t36.TICKET30_ARTIFACT, t36.WEIGHTS_PATH,
        t36.FEATURE_ROOT / "feature-manifest.json"]
    for fold in range(3):
        fixed.extend([PARENT / f"report-supervision/fold-{fold}/run.json",
            PARENT / f"report-supervision/fold-{fold}/checkpoints/final.pt"])
    fixed.extend(p for p in snapshot.rglob("*") if p.is_file())
    used = sum(int(line.split()[0]) for line in subprocess.check_output([
        "du", "--apparent-size", "-s", "-B1", str(PROJECT),
        str(PROJECT.parent / "models/qwen3-8b-b968826d9c46dd6066d109eabc6255188de91218")], text=True).splitlines())
    if used + 5_000_000_000 > 2_000_000_000_000 or shutil.disk_usage(PROJECT).free < 5_000_000_000:
        raise ValueError("Fresh storage does not admit diagnostic checkpoint retention")
    plan = dict(schema_version="rsna-ticket63-shared-full-backbone-cost-plan-v1",
        created_at_utc=datetime.now(timezone.utc).isoformat(), run_root=str(ROOT),
        source_snapshot=str(snapshot), decision_path=str(DECISION), resources=resources,
        authorization_sha256=digest(AUTH), pixel_root=str(PIXELS), parent_root=str(PARENT),
        cohort_parent_plan=str(cohort_parent), forecast_node_seconds=forecast,
        tracked_storage_bytes=used, additional_storage_allowance_bytes=5_000_000_000,
        training_config=config().to_dict(), training_fold=0,
        successful_diagnostic_steps=3, planned_interrupt_step=1,
        stages=["admission", "interrupt", "resume", "continuous", "verify"],
        future_route=dict(successful_steps_per_fold=7500, folds=3, prospective_final_fits=1,
            steps_per_stage=1500, checkpoint_interval_steps=500,
            checkpoint_interval_seconds=900, fit_authorized=False),
        runtime_versions={n: importlib.metadata.version(n) for n in ("numpy", "torch")},
        frozen_files=[dict(path=str(p), sha256=digest(p)) for p in fixed],
        new_downloads=False, network_during_execution=False, held_out_predictions=False,
        competition_submission=False, candidate_promotion=False, external_spend_usd=0)
    write(ROOT / "plan.json", plan)
    checksum = digest(ROOT / "plan.json")
    write(DECISION, dict(schema_version="rsna-campaign-autonomous-decision-v1", decision="approved",
        decision_made_by="campaign agent under standing owner authorization",
        authorization_sha256=digest(AUTH), plan_sha256=checksum, resources=resources,
        basis="Measure complete admitted pixel I/O, actual shared trainer startup, diagnostic mixed-source updates, fresh-process exact staged resume and checkpoint delivery before freezing a complete longer three-fold route.",
        new_exact_owner_numeric_approval_claimed=False, full_training_authorized=False,
        gpu_ledger_changed=False, tpu_ledger_changed=False, candidate_promotion=False))
    launcher = ROOT / "run.sbatch"
    with launcher.open("x") as handle:
        handle.write(f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket63-training-cost
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
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
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
for stage in admission interrupt resume continuous verify; do
  {PROJECT}/.venv/bin/python {snapshot / Path(__file__).name} "$stage" {checksum}
done
''')
    write(ROOT / "prepared-validation.json", dict(plan_sha256=checksum,
        launcher_sha256=digest(launcher), launched=False, full_training_authorized=False))
    print(json.dumps(dict(plan_sha256=checksum, forecast_node_seconds=forecast, launcher=str(launcher))))


def runtime(checksum):
    if digest(ROOT / "plan.json") != checksum:
        raise ValueError("Cost plan changed")
    plan = read(ROOT / "plan.json")
    decision = read(plan["decision_path"])
    if (decision["decision"] != "approved" or decision["plan_sha256"] != checksum
            or decision["resources"] != plan["resources"] or digest(AUTH) != plan["authorization_sha256"]):
        raise ValueError("Cost admission differs")
    for item in plan["frozen_files"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input changed: {item['path']}")
    for name, version in plan["runtime_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Pinned runtime changed")
    if os.environ.get("PYTHONPATH") != str(Path(plan["source_snapshot"]) / "src"):
        raise ValueError("Frozen source runtime required")
    job = os.environ["SLURM_JOB_ID"]
    controller = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True)
    fields = dict(t.split("=", 1) for t in controller.split() if "=" in t)
    expected = dict(JobId=job, JobState="RUNNING", Partition="cluster1-cpu", NumNodes="1",
        NumCPUs="16", NumTasks="1", MinMemoryNode="64G", TimeLimit="01:30:00",
        OverSubscribe="OK", Requeue="0", Restarts="0")
    if any(fields.get(k) != v for k, v in expected.items()) or "gpu" in fields.get("TRES", "").lower():
        raise ValueError("Actual bounded allocation differs")
    torch.set_num_threads(16); torch.set_num_interop_threads(1)
    if torch.cuda.is_available() or torch.version.cuda is not None:
        raise ValueError("CPU-only runtime required")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Cost watchdog")))
    signal.alarm(plan["resources"]["code_watchdog_seconds"])
    return plan, controller


def load_bank(plan, selected=None):
    manifest = read(Path(plan["pixel_root"]) / "pixel-manifest.json")
    rows = manifest["pixels"]
    if selected is not None:
        rows = [r for r in rows if r["study_instance_uid"] in selected]
    result = {}
    for row in rows:
        uid = row["study_instance_uid"]
        path = Path(plan["pixel_root"]) / "pixels" / row["pixel_file"]
        entry = load_pixel_entry(path, expected_sha256=row["pixel_file_sha256"],
            expected_study_uid=uid, expected_identity_sha256=manifest["pixel_identity_sha256"],
            expected_source_identity_sha256=row["source_identity_sha256"],
            expected_plan_sha256=manifest["plan_sha256"])
        pixels = PackedPixelArray(path, row["pixel_file_sha256"], uid,
            manifest["pixel_identity_sha256"], row["source_identity_sha256"], manifest["plan_sha256"])
        result[uid] = dict(images=pixels, slice_mask=entry.slice_mask,
            series_mask=entry.series_mask, series_descriptors=entry.series_descriptors)
    if selected is not None and set(result) != set(selected):
        raise ValueError("Selected source cohort incomplete")
    return result


def examples_for_fold(plan, bank, fold):
    manifest = load_fold_manifest(t36.FOLD_MANIFEST_PATH)
    parent = next(r for r in read(plan["cohort_parent_plan"])["folds"] if r["fold"] == fold)
    admission = admit_ticket30_fold_supervision(manifest, fold, t36.TICKET30_ARTIFACT,
        expected_artifact_sha256=t36.TICKET30_AUDITED_ARTIFACT_SHA256).for_image_studies(
            [r["study_instance_uid"] for r in read(PIXELS / "pixel-manifest.json")["pixels"]])
    if admission.to_public_dict() != parent["supervision_admission"]:
        raise ValueError("Original fold supervision changed")
    pools = [parent["gold_training_study_uids"], parent["weak_training_study_uids"]]
    held_out = set(manifest.held_out_study_uids(fold))
    if set(pools[0]) & set(pools[1]) or held_out & set(pools[0] + pools[1]):
        raise ValueError("Overlapping sources or held-out contamination")
    if set(pools[0]) != set(admission.reference_study_uids) or set(pools[1]) != set(admission.admitted_report_only_study_uids):
        raise ValueError("Original training membership changed")
    result = []
    for pool, kind in zip(pools, ("reference", "weak"), strict=True):
        items = []
        for uid in pool:
            label = admission.by_study[uid]
            provenance = admission.provenance_for(uid)
            provenance["eligible_study_uids"] = tuple(provenance["eligible_study_uids"])
            provenance["excluded_study_uids"] = tuple(provenance["excluded_study_uids"])
            if label.source_kind != kind or not any(label.supervision_mask):
                raise ValueError("Empty or incorrectly sourced supervision")
            items.append(TrainingExample(study_instance_uid=uid, **bank[uid],
                targets=np.asarray([0 if v is None else v for v in label.targets], dtype=np.float32),
                supervision_mask=np.asarray(label.supervision_mask, dtype=np.bool_),
                provenance=SupervisionProvenance(**provenance)))
        result.append(items)
    return result


def initial_model(fold=0):
    seed_everything(config().seed)
    model = build_study_model(config().model_config())
    model.load_pretrained_state(torch.load(t36.WEIGHTS_PATH, map_location="cpu", weights_only=True))
    warm = torch.load(PARENT / f"report-supervision/fold-{fold}/checkpoints/final.pt",
        map_location="cpu", weights_only=True)
    model.head.load_state_dict(warm["head_state_dict"], strict=True)
    return model


def admission_run(plan, checksum, controller):
    tick = time.monotonic(); bank = load_bank(plan); load_seconds = time.monotonic()-tick
    assert len(bank) == 4228
    folds = []
    for fold in range(3):
        tick = time.monotonic(); gold, weak = examples_for_fold(plan, bank, fold)
        construction = time.monotonic()-tick; tick = time.monotonic()
        identity = [fingerprint_training_examples(pool) for pool in (gold, weak)]
        folds.append(dict(fold=fold, reference_studies=len(gold), weak_studies=len(weak),
            construction_seconds=construction, fingerprint_seconds=time.monotonic()-tick,
            training_example_fingerprints=identity, held_out_excluded=True))
        del gold, weak; gc.collect()
        print(json.dumps(folds[-1]), flush=True)
    write(ROOT / "admission-result.json", dict(status="COMPLETE", plan_sha256=checksum,
        controller=controller, loaded_studies=len(bank), load_seconds=load_seconds, folds=folds,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        full_pixel_arrays_retained=False, held_out_predictions=False))


def stage_run(stage, plan, checksum, controller):
    start = time.monotonic()
    parent = read(plan["cohort_parent_plan"])["folds"][0]
    selected = set(parent["gold_training_study_uids"] + parent["weak_training_study_uids"])
    bank = load_bank(plan, selected); gold, weak = examples_for_fold(plan, bank, 0)
    initialization = initial_model(); initial_sha = _model_state_sha256(initialization)
    fingerprints = [fingerprint_training_examples(pool) for pool in (gold, weak)]
    admitted = next(r for r in read(ROOT / "admission-result.json")["folds"] if r["fold"] == 0)
    assert fingerprints == admitted["training_example_fingerprints"]
    pre_training_seconds = time.monotonic()-start
    directory = ROOT / ("continuous" if stage == "continuous" else "staged")
    identity = dict(cost_plan_sha256=checksum, pixel_manifest_sha256=digest(PIXELS / "pixel-manifest.json"),
        cohort_parent_plan_sha256=digest(plan["cohort_parent_plan"]),
        training_example_fingerprints=fingerprints, diagnostic_only=True)
    clock_reads = []
    def measured_clock():
        value = time.monotonic(); clock_reads.append(value); return value
    tick = time.monotonic(); stopped = False
    try:
        result = train_fixed_budget(gold, config(), additional_examples=weak,
            additional_batch_order_seed=config().batch_order_seed+34034034,
            model=initialization, artifact_dir=directory, fold=0, identity=identity,
            checkpoint_clock=measured_clock,
            requested_resources=plan["resources"],
            resume_from=directory / "checkpoints/latest.pt" if stage == "resume" else None,
            interrupt_after_steps=1 if stage == "interrupt" else None)
    except TrainingInterrupted:
        if stage != "interrupt":
            raise
        stopped = True
    trainer_seconds = time.monotonic()-tick
    checkpoint_path = directory / ("checkpoints/latest.pt" if stopped else "checkpoints/final.pt")
    tick = time.monotonic()
    checkpoint = load_training_checkpoint(checkpoint_path, expected_config=config(), expected_fold=0)
    reload_seconds = time.monotonic()-tick
    expected_steps = 1 if stopped else 3
    assert checkpoint["progress"]["successful_optimizer_steps"] == expected_steps
    assert checkpoint["progress"]["attempted_batches"] == expected_steps
    if stopped:
        assert not (directory / "checkpoints/final.pt").exists()
    else:
        assert result.successful_optimizer_steps == 3 and result.skipped_batches == 0
    write(ROOT / f"{stage}-result.json", dict(status="COMPLETE", plan_sha256=checksum,
        job_id=os.environ["SLURM_JOB_ID"], controller=controller, initial_model_state_sha256=initial_sha,
        pre_training_seconds=pre_training_seconds, trainer_seconds=trainer_seconds,
        checkpoint_reload_seconds=reload_seconds, checkpoint_sha256=digest(checkpoint_path),
        checkpoint_clock_monotonic_reads=clock_reads,
        step_and_prior_checkpoint_intervals_seconds=[b-a for a,b in zip(clock_reads,clock_reads[1:])],
        checkpoint_bytes=checkpoint_path.stat().st_size, successful_steps=expected_steps,
        planned_stage_boundary=stopped, elapsed_seconds=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        held_out_predictions=False, candidate_promotion=False))


def verify_run(plan, checksum, controller):
    staged = load_training_checkpoint(ROOT / "staged/checkpoints/final.pt", expected_config=config())
    continuous = load_training_checkpoint(ROOT / "continuous/checkpoints/final.pt", expected_config=config())
    for key in ("progress", "identity", "sampler_state", "scheduler_state_dict"):
        assert staged[key] == continuous[key], key
    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and torch.equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
        if isinstance(a, (tuple, list)):
            return isinstance(b, type(a)) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
        return a == b
    assert equal(staged["model_state_dict"], continuous["model_state_dict"])
    assert equal(staged["optimizer_state_dict"], continuous["optimizer_state_dict"])
    initial = initial_model().state_dict()
    state = staged["model_state_dict"]
    changes = {prefix: any(not torch.equal(v, initial[k]) for k, v in state.items() if k.startswith(prefix))
        for prefix in ("encoder.conv1", "encoder.layer1", "encoder.layer2", "encoder.layer3", "encoder.layer4", "head")}
    assert all(changes.values())
    assert all(torch.equal(v, initial[k]) for k, v in state.items()
        if k.endswith(("running_mean", "running_var", "num_batches_tracked")))
    records = [read(ROOT / f"{s}-result.json") for s in ("interrupt", "resume", "continuous")]
    assert len({r["initial_model_state_sha256"] for r in records}) == 1
    assert all(r["plan_sha256"] == checksum and r["job_id"] == os.environ["SLURM_JOB_ID"] for r in records)
    write(ROOT / "cost-result.json", dict(status="COMPLETE", plan_sha256=checksum,
        job_id=os.environ["SLURM_JOB_ID"], controller=controller,
        admission_result_sha256=digest(ROOT / "admission-result.json"),
        stage_result_sha256={s:digest(ROOT / f"{s}-result.json") for s in ("interrupt", "resume", "continuous")},
        exact_fresh_process_staged_model_optimizer_progress_replay=True,
        all_encoder_stages_changed=changes, normalization_buffers_unchanged=True,
        complete_comparison_plan_still_required=True, full_training_authorized=False,
        held_out_predictions=False, gpu_count=0, external_spend_usd=0, candidate_promotion=False,
        actual_terminal_still_required=True))


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "prepare":
        prepare()
    else:
        plan, controller = runtime(sys.argv[2])
        if stage == "admission":
            admission_run(plan, sys.argv[2], controller)
        elif stage in {"interrupt", "resume", "continuous"}:
            stage_run(stage, plan, sys.argv[2], controller)
        elif stage == "verify":
            verify_run(plan, sys.argv[2], controller)
        else:
            raise ValueError("Unknown exact cost stage")
