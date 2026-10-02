"""One bounded whole-encoder CPU throughput and checkpoint-delivery pilot."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from rsna_knee import ticket36 as t36
from rsna_knee.domain import SeriesRecord, StudyRecord
from rsna_knee.folds import load_fold_manifest
from rsna_knee.model import ModelConfig, build_study_model
from rsna_knee.preprocessing import pack_study
from rsna_knee.real_inference import load_integration_manifest
from rsna_knee.resnet18 import RESNET18_FULL_FINETUNE_MODEL_ID, RESNET18_MODEL_ID
from rsna_knee.full_backbone import fixed_shape_masked_bce
from rsna_knee.series_cache import SeriesCache
from rsna_knee.ticket34 import admit_ticket30_fold_supervision
from rsna_knee.training import _array_sha256, masked_bce_loss, seed_everything


PROJECT = t36.PROJECT_ROOT
ROOT = PROJECT / "artifacts/private/ticket61-cpu-full-backbone-pilot-v1"
PRIOR = PROJECT / "artifacts/private/ticket56-cpu-layer4-pilot"
PARENT = PROJECT / "artifacts/private/ticket55-resnet-long-report-head-v2"
AUTH = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket61-cpu-full-backbone-pilot-20261002.json"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def prepare():
    if ROOT.exists() or DECISION.exists():
        raise FileExistsError("Immutable whole-backbone pilot already exists")
    prior = read(PRIOR / "plan.json")
    result = read(PRIOR / "pilot-result.json")
    if result["status"] != "COMPLETE" or result["valid_slices"] != 232:
        raise ValueError("Prior real-MRI input admission is required")
    for row in prior["frozen_files"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("Prior admitted resource changed")
    ROOT.mkdir(mode=0o700, parents=True)
    snapshot = ROOT / "source-snapshot"
    shutil.copytree(PROJECT / "src/rsna_knee", snapshot / "src/rsna_knee",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(__file__, snapshot / Path(__file__).name)
    shutil.copyfile(PROJECT / "scripts/validate_ticket61_cpu_full_backbone.py", snapshot / "validate_ticket61_cpu_full_backbone.py")
    fixed = [Path(item["path"]) for item in prior["frozen_files"]]
    fixed += [PRIOR / "plan.json", PRIOR / "pilot-result.json"]
    fixed += [p for p in snapshot.rglob("*") if p.is_file()]
    resources = {**prior["resources"], "walltime_seconds": 900, "code_watchdog_seconds": 780}
    plan = {"schema_version": "rsna-ticket61-full-backbone-cpu-pilot-plan-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_root": str(ROOT), "source_snapshot": str(snapshot), "decision_path": str(DECISION),
            "resources": resources, "authorization_sha256": digest(AUTH),
            "fold_manifest_checksum": prior["fold_manifest_checksum"], "training_fold": 0,
            "selected_training_report_uids": prior["selected_training_report_uids"],
            "selected_valid_slice_counts": prior["selected_valid_slice_counts"],
            "selection": prior["selection"], "expected_input_array_sha256": result["input_array_sha256"],
            "model": {**prior["model"], "model_id": RESNET18_FULL_FINETUNE_MODEL_ID},
            "optimizer": prior["optimizer"],
            "profiling": {"batch_size": 4, "warmup_updates": 1, "timed_updates": 3,
                          "seed": 20261002, "parity_tolerance": 1e-6},
            "projection": {"cost_allowance": 2, "updates_per_fold_scenarios": [1500,7500],
                           "folds": 3, "final_fit_runs": 1, "per_job_limit_seconds": 21600,
                           "complete_route_selected": False},
            "frozen_files": [{"path": str(p), "sha256": digest(p)} for p in fixed],
            "runtime_versions": {name: importlib.metadata.version(name) for name in ("numpy","torch")},
            "network_during_execution": False, "new_downloads": False, "gpu_count": 0,
            "external_spend_usd": 0, "competition_submission": False, "candidate_promotion": False}
    write(ROOT / "plan.json", plan)
    checksum = digest(ROOT / "plan.json")
    write(DECISION, {"schema_version": "rsna-campaign-autonomous-decision-v1", "decision": "approved",
          "decision_made_by": "campaign agent under standing owner authorization",
          "authorization_sha256": digest(AUTH), "plan_sha256": checksum, "resources": resources,
          "basis": "Measure independent CPU whole-backbone viability on four already admitted training-side MRI reports while one environment-only TPU request is queued.",
          "new_exact_owner_numeric_approval_claimed": False, "full_training_authorized": False,
          "gpu_ledger_changed": False, "tpu_ledger_changed": False})
    launcher = ROOT / "run.sbatch"
    launcher.write_text(f"""#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket61-full-backbone-pilot
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:15:00
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
{PROJECT}/.venv/bin/python {snapshot / Path(__file__).name} run {ROOT / 'plan.json'} {checksum}
{PROJECT}/.venv/bin/python {snapshot / 'validate_ticket61_cpu_full_backbone.py'} {ROOT}
""")
    print(json.dumps({"plan_sha256":checksum,"launcher":str(launcher)}))


def run(plan_path, checksum):
    started = time.monotonic()
    if digest(plan_path) != checksum:
        raise ValueError("Frozen full-backbone plan changed")
    plan = read(plan_path)
    decision = read(plan["decision_path"])
    if (decision["decision"] != "approved" or decision["plan_sha256"] != checksum
            or decision["resources"] != plan["resources"] or digest(AUTH) != plan["authorization_sha256"]):
        raise ValueError("CPU resource decision differs")
    root = Path(plan["run_root"])
    job = os.environ["SLURM_JOB_ID"]
    controller = subprocess.check_output(["scontrol","show","job",job,"-o"],text=True)
    fields = dict(t.split("=",1) for t in controller.split() if "=" in t)
    expected = {"JobState":"RUNNING","Partition":"cluster1-cpu","NumNodes":"1","NumCPUs":"16",
                "NumTasks":"1","MinMemoryNode":"64G","TimeLimit":"00:15:00","OverSubscribe":"OK","Requeue":"0","Restarts":"0"}
    if any(fields.get(k)!=v for k,v in expected.items()) or "gpu" in fields.get("TRES","").lower():
        raise ValueError("Actual CPU allocation differs")
    write(root / "running-allocation.json", {"controller":controller})
    watchdog = threading.Timer(plan["resources"]["code_watchdog_seconds"],lambda:os._exit(124))
    watchdog.daemon=True; watchdog.start()
    for row in plan["frozen_files"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("Frozen source or input bytes changed")
    for name,version in plan["runtime_versions"].items():
        if importlib.metadata.version(name)!=version:
            raise ValueError("Pinned CPU runtime differs")
    torch.set_num_threads(16);torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise ValueError("CPU-only pilot required")
    manifest = load_fold_manifest(t36.FOLD_MANIFEST_PATH)
    parent = read(read(PARENT / "plan.json")["parent_plan"])
    fold0 = next(row for row in parent["folds"] if row["fold"] == 0)
    selected = plan["selected_training_report_uids"]
    if (manifest.manifest_checksum != plan["fold_manifest_checksum"]
            or not set(selected) <= set(fold0["weak_training_study_uids"])
            or set(selected) & set(manifest.held_out_study_uids(0))
            or set(selected) & {m.study_instance_uid for m in manifest.gold_memberships}):
        raise ValueError("Held-out or gold input entered report profiling")
    admission = admit_ticket30_fold_supervision(manifest, 0, t36.TICKET30_ARTIFACT,
        expected_artifact_sha256=t36.TICKET30_AUDITED_ARTIFACT_SHA256)
    targets = torch.tensor([[float(value) if value is not None else 0.0
                             for value in admission.by_study[uid].targets] for uid in selected])
    mask = torch.tensor([[value is not None for value in admission.by_study[uid].targets]
                         for uid in selected], dtype=torch.bool)
    if not mask.any(dim=1).all():
        raise ValueError("Every profiled report requires admitted supervision")
    integration = load_integration_manifest(t36.INTEGRATION_PATH)
    index = read(t36.SERIES_INDEX_PATH)["entries"]
    reference = {r["study_instance_uid"]: r for r in read(t36.FEATURE_ROOT / "feature-manifest.json")["features"]}
    studies = {s.study_instance_uid: s for s in integration.studies}
    cache = SeriesCache(t36.TICKET35_SERIES_CACHE, read_only=True)
    packed_rows = []
    for uid in selected:
        prior = reference[uid]
        if prior["source_kind"] != "ticket35-prepared-series-cache-read-only":
            raise ValueError("Probe must use existing report image caches")
        source = studies[uid]
        prepared, records = [], []
        for item in source.series:
            row = index[item.series_instance_uid]
            if row["study_instance_uid"] != uid or digest(cache.entry_path(row["cache_key"]) / "metadata.json") != row["metadata_sha256"]:
                raise ValueError("Prepared series identity differs")
            prepared.append(cache.load_validated_entry(row["cache_key"], study_instance_uid=uid,
                series_instance_uid=item.series_instance_uid, expected_source_entry_names=item.entries,
                config=integration.preprocess_config))
            records.append(SeriesRecord(study_instance_uid=uid, series_instance_uid=item.series_instance_uid,
                slices=(), anatomical_plane=item.anatomical_plane, fluid_sensitive=item.fluid_sensitive,
                fat_suppression=item.fat_suppression, diagnostic_status=item.diagnostic_status,
                source_entry_names=item.entries))
        if t36._study_source_digest(source, index, archive_fingerprint=integration.archive_fingerprint) != prior["source_identity_sha256"]:
            raise ValueError("MRI source differs from the admitted representation")
        packed = pack_study(StudyRecord(uid, tuple(records)), prepared, integration.preprocess_config,
                            packing_config=integration.packing_config)
        prior_path = t36.FEATURE_ROOT / prior["feature_file"]
        if digest(prior_path) != prior["feature_file_sha256"]:
            raise ValueError("Admitted source feature identity differs")
        with np.load(prior_path, allow_pickle=False) as previous:
            for key, value in {"slice_mask": packed.slice_mask, "series_mask": packed.series_mask,
                               "series_descriptors": packed.series_descriptors}.items():
                if not np.array_equal(previous[key], value):
                    raise ValueError("Packing or anatomical descriptors differ")
        if int((packed.slice_mask & packed.series_mask[:, None]).sum()) != plan["selected_valid_slice_counts"][uid]:
            raise ValueError("Selected actual slice count differs")
        packed_rows.append(packed)
    images = torch.from_numpy(np.stack([row.images for row in packed_rows])).float()
    slices = torch.from_numpy(np.stack([row.slice_mask for row in packed_rows])).bool()
    series = torch.from_numpy(np.stack([row.series_mask for row in packed_rows])).bool()
    descriptors = torch.from_numpy(np.stack([row.series_descriptors for row in packed_rows])).float()
    input_digest = _array_sha256(images.numpy())
    if input_digest != plan["expected_input_array_sha256"]:
        raise ValueError("Independently admitted prior MRI arrays differ")
    seed_everything(plan["profiling"]["seed"])
    model = build_study_model(ModelConfig.from_dict(plan["model"]))
    model.load_pretrained_state(torch.load(t36.WEIGHTS_PATH,map_location="cpu",weights_only=True))
    warm_head = torch.load(PARENT / "report-supervision/fold-0/checkpoints/final.pt",map_location="cpu",weights_only=True)
    model.head.load_state_dict(warm_head["head_state_dict"],strict=True)
    model.train()
    initial = {k:v.detach().clone() for k,v in model.state_dict().items()}
    frozen_config = {**plan["model"],"model_id":RESNET18_MODEL_ID}
    frozen = build_study_model(ModelConfig.from_dict(frozen_config)).eval()
    frozen.load_state_dict(initial,strict=True)
    with torch.no_grad():
        initial_error = float((model(images,slices,series,descriptors)-frozen(images,slices,series,descriptors)).abs().max())
    if initial_error > plan["profiling"]["parity_tolerance"]:
        raise ValueError("Static full-backbone and native admitted initial logits differ")
    del frozen
    np.savez(root / "diagnostic-batch.npz", images=images.numpy(),slice_mask=slices.numpy(),
             series_mask=series.numpy(),series_descriptors=descriptors.numpy(),targets=targets.numpy(),supervision_mask=mask.numpy())
    optimizer = torch.optim.Adam([
        {"params":model.encoder.parameters(),"lr":plan["optimizer"]["encoder_learning_rate"]},
        {"params":model.head.parameters(),"lr":plan["optimizer"]["head_learning_rate"]}])
    timings,losses=[],[]
    for step in range(plan["profiling"]["warmup_updates"]+plan["profiling"]["timed_updates"]):
        tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
        loss=fixed_shape_masked_bce(model(images,slices,series,descriptors),targets,mask)
        if not torch.isfinite(loss):raise ValueError("Nonfinite admitted-report loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Missing or nonfinite whole-encoder gradient")
        optimizer.step();losses.append(float(loss.detach()))
        if step>=plan["profiling"]["warmup_updates"]:timings.append(time.monotonic()-tick)
    changes={}
    for group in ("conv1","layer1","layer2","layer3","layer4"):
        changes[group]=max(float((v-initial[k]).abs().max()) for k,v in model.state_dict().items() if k.startswith("encoder."+group+"."))
    changes["head"]=max(float((v-initial[k]).abs().max()) for k,v in model.state_dict().items() if k.startswith("head."))
    buffers=max(float((v-initial[k]).abs().max()) for k,v in model.state_dict().items() if k.endswith(("running_mean","running_var","num_batches_tracked")))
    if any(v<=0 for v in changes.values()) or buffers!=0:raise ValueError("Whole-backbone update scope or fixed normalization differs")
    model.eval()
    with torch.no_grad():logits=model(images,slices,series,descriptors).cpu().numpy()
    np.save(root / "diagnostic-logits.npy",logits,allow_pickle=False)
    torch.save({"model_config":model.config.to_dict(),"state_dict":model.state_dict()},root / "diagnostic-checkpoint.pt")
    median=float(np.median(timings));allowance=plan["projection"]["cost_allowance"]
    scenarios=[{"updates_per_fold":n,"training_seconds_per_fold":allowance*median*n,
                "three_folds_plus_final_fit_training_seconds":allowance*median*n*4,
                "single_job_training_fits":allowance*median*n<=21600}
               for n in plan["projection"]["updates_per_fold_scenarios"]]
    write(root / "pilot-result.json",{"schema_version":"rsna-ticket61-full-backbone-cpu-pilot-result-v1",
          "status":"COMPLETE","observed_at_utc":datetime.now(timezone.utc).isoformat(),"job_id":job,"plan_sha256":checksum,
          "selected_report_study_count":4,"valid_slices":int((slices&series[:,:,None]).sum()),
          "padded_slices":int(slices.numel()),"input_array_sha256":input_digest,"held_out_inputs_excluded":True,
          "admitted_supervised_entries_per_update":int(mask.sum()),"native_initial_logit_max_abs_error":initial_error,
          "timed_update_seconds":timings,"median_update_seconds":median,"training_losses":losses,
          "parameter_max_abs_changes":changes,"normalization_buffer_max_abs_change":buffers,
          "projected_training_scenarios":scenarios,"elapsed_seconds":time.monotonic()-started,
          "peak_process_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          "outputs":[{"file":p,"sha256":digest(root/p)} for p in ("diagnostic-batch.npz","diagnostic-logits.npy","diagnostic-checkpoint.pt")],
          "projection_limits":"Four warm-cache report studies. Static bags process 512 padded slots regardless of valid slices. Excludes unmeasured full-cohort random I/O, preprocessing, checkpoint boundaries, evaluation, final fit overhead and delivery. No complete-route admission.",
          "full_training_selected":False,"candidate_promotion":False,"competition_submission":False,"gpu_count":0,"external_spend_usd":0})
    watchdog.cancel()
    print(json.dumps({"status":"COMPLETE","median_update_seconds":median,"scenarios":scenarios}),flush=True)


if __name__ == "__main__":
    if sys.argv[1]=="prepare":prepare()
    else:run(Path(sys.argv[2]),sys.argv[3])
