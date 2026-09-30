"""Fixed-budget matched ConvNeXt head experiment on sealed features."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

import numpy as np
import torch

from . import ticket36 as t36
from .evaluation import evaluate_oof_predictions
from .folds import load_fold_manifest
from .frozen_head_experiment import (
    HEAD_ID, reload_head, save_head_checkpoint, train_head, zero_image_features,
)
from .image_model_comparison import compare_image_model_predictions
from .ticket34 import TICKET34_WEAK_BATCH_SEED_OFFSET, admit_ticket30_fold_supervision


PROJECT = t36.PROJECT_ROOT
ROOT = PROJECT / "artifacts/private/ticket50-convnext-supervision"
PLAN = ROOT / "plan-v2.json"
APPROVAL = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket50-convnext-supervision-v2.json"
COHORT = PROJECT / "artifacts/private/ticket49-convnext-feature-cohort"
COMPARATOR = PROJECT / "artifacts/private/ticket44-full-weak-exposure"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def freeze() -> dict:
    if PLAN.exists():
        raise FileExistsError("Ticket 50 plan is already frozen")
    verification = read(COHORT / "verification-v1/verification-result.json")
    if verification["status"] != "complete" or verification["study_count"] != 4228:
        raise ValueError("complete independently verified features required")
    terminal = (COHORT / "verification-v1/controller-7937/terminal-controller.txt").read_text()
    if "JobState=COMPLETED" not in terminal or "ExitCode=0:0" not in terminal:
        raise ValueError("independent verification did not complete")
    feature_path = Path(verification["feature_manifest_path"])
    bank_path = Path(verification["pooled_cohort_path"])
    if digest(feature_path) != verification["feature_manifest_sha256"] or digest(bank_path) != verification["pooled_cohort_sha256"]:
        raise ValueError("sealed features/pooled bank changed")
    manifest = load_fold_manifest(t36.FOLD_MANIFEST_PATH)
    if manifest.superseded or len(manifest.gold_memberships) != 58:
        raise ValueError("corrected gold fold manifest required")
    comparator_plan = read(COMPARATOR / "plan.json")
    if comparator_plan["fold_manifest_checksum"] != manifest.manifest_checksum:
        raise ValueError("comparator fold identity differs")
    source = ROOT / "source-snapshot-v2"
    shutil.copytree(PROJECT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__"))
    inputs = [feature_path, bank_path, COHORT / "verification-v1/verification-result.json",
              COHORT / "verification-v1/controller-7937/terminal-controller.txt", COHORT / "plan.json",
              t36.FOLD_MANIFEST_PATH, t36.INTEGRATION_PATH, t36.TICKET30_ARTIFACT,
              COMPARATOR / "plan.json", COMPARATOR / "analysis-recovery-v1/comparison.json",
              COMPARATOR / "learning-controls-v1/evidence.json"]
    comparator_paths = {arm: [] for arm in ["reference-only", "report-supervision"]}
    for arm in comparator_paths:
        for fold in range(3):
            path = COMPARATOR / arm / f"fold-{fold}/run.json"
            record = read(path)
            expected = {member.study_instance_uid for member in manifest.gold_memberships if member.fold == fold}
            if (record["status"] != "completed_fixed_budget_training" or record["fold"] != fold
                    or record["counts"]["successful_optimizer_steps"] != 1500
                    or set(record["predictions"]) != expected
                    or record["config"] != comparator_plan["training_configuration"]
                    or record["held_out_inputs_excluded_from_fitting"] is not True):
                raise ValueError("fixed ResNet comparator contract differs")
            inputs.append(path)
            comparator_paths[arm].append(str(path))
    with np.load(bank_path, allow_pickle=False) as bank:
        uids = [str(uid) for uid in bank["study_uids"]]
        if bank["pooled"].shape != (4228, 3100) or len(set(uids)) != 4228:
            raise ValueError("sealed bank dimensions/membership differ")
    fold_rows = []
    for fold in range(3):
        admitted = admit_ticket30_fold_supervision(manifest, fold, t36.TICKET30_ARTIFACT,
            expected_artifact_sha256=t36.TICKET30_AUDITED_ARTIFACT_SHA256).for_image_studies(uids)
        fold_rows.append({"fold": fold,
            "gold_training_study_uids": sorted(member.study_instance_uid for member in manifest.gold_memberships if member.fold != fold),
            "gold_heldout_study_uids": sorted(member.study_instance_uid for member in manifest.gold_memberships if member.fold == fold),
            "weak_training_study_uids": sorted(admitted.admitted_report_only_study_uids),
            "weak_training_pool_count": len(admitted.admitted_report_only_study_uids),
            "supervision_admission": admitted.to_public_dict()})
    inputs.extend(path for path in source.rglob("*") if path.is_file())
    plan = {
        "schema_version": "rsna-ticket50-frozen-head-plan-v1", "ticket": 50,
        "run_root": str(ROOT), "source_snapshot": str(source), "approval_path": str(APPROVAL),
        "frozen_files": [{"path": str(path), "sha256": digest(path)} for path in inputs],
        "feature_manifest_path": str(feature_path), "feature_manifest_sha256": digest(feature_path),
        "pooled_cohort_path": str(bank_path), "pooled_cohort_sha256": digest(bank_path),
        "fold_manifest_checksum": manifest.manifest_checksum, "folds": fold_rows,
        "comparator_paths": comparator_paths,
        "head": {"head_id": HEAD_ID, "input_width": 3100, "hidden_width": 32, "targets": 12,
                 "activation": "ReLU", "trainable": "head only", "feature_width": 768,
                 "pooling_id": "plane-stratified-series-mean-v1"},
        "optimization": {"successful_steps": 1500, "batch_size": 2, "initialization_seed": 20260911,
            "batch_order_seed": 20260912, "weak_seed_offset": TICKET34_WEAK_BATCH_SEED_OFFSET,
            "learning_rate": 0.001, "optimizer": "Adam", "schedule": "constant",
            "checkpoint_interval_steps": 15, "loss": "masked BCE mean over combined supervised entries",
            "augmentation": "none", "selection": "final successful step only"},
        "analysis": {"bootstrap_resamples": 2000, "seed": 20260930, "confidence": 0.95,
                     "permutation_resamples": 10000, "fixed_blend_convnext_weight": 0.5,
                     "candidate_promotion": False, "development_cv_reused": True},
        "resources": {"partition": "cluster1-cpu", "nodes": 1, "tasks": 1,
                      "cpus_per_task": 16, "memory_gib": 64, "walltime_seconds": 7200,
                      "maximum_concurrent_jobs": 1, "maximum_launch_attempts": 1,
                      "gpu_count": 0, "external_spend_usd": 0, "exclusive": False},
        "new_checkpoint_download": False, "network_during_execution": False,
        "final_candidate_selection": False, "publication": False,
    }
    write(PLAN, plan)
    sha = digest(PLAN)
    write(APPROVAL, {
        "schema_version": "rsna-ticket50-resource-approval-v1", "decision": "approved",
        "approved_by": "Project owner via existing conversation-wide authorization: approved for anything everything; no matter what 1st place",
        "authorization_interpretation": "Applies existing zero-spend campaign authorization to this exact separate CPU matched-head plan after complete feature verification. No Ticket 50-specific approval words claimed; no Ticket 39 amendment or GPU capacity expansion.",
        "scope": "ticket50-convnext-supervision", "plan_sha256": sha,
        "resources": plan["resources"], "checkpoint_rights_scope": "already admitted Ticket 48/49 private experiment",
        "new_weight_download": False, "network_during_execution": False,
        "publication": False, "candidate_promotion": False,
    })
    launcher = ROOT / "run-v2.sbatch"
    with launcher.open("x") as handle:
        handle.write(f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket50-heads
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output={ROOT}/slurm-%j.out
#SBATCH --error={ROOT}/slurm-%j.err
#SBATCH --chdir={PROJECT}
set -euo pipefail
umask 077
export RSNA_PROJECT_ROOT={PROJECT}
export PYTHONPATH={source}/src
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
{PROJECT}/.venv/bin/python -m rsna_knee.ticket50 run {sha}
''')
    write(ROOT / "prepared-validation-v2.json", {"plan_sha256": sha, "launch_sha256": digest(launcher),
                                             "fold_count": 3, "training_runs": 6, "launched": False})
    return {"plan_sha256": sha, "folds": [row["weak_training_pool_count"] for row in fold_rows]}


def run(expected_sha: str) -> dict:
    started = time.monotonic()
    if digest(PLAN) != expected_sha:
        raise ValueError("exact frozen training plan differs")
    plan = read(PLAN)
    approval = read(APPROVAL)
    if approval.get("decision") != "approved" or approval.get("plan_sha256") != expected_sha:
        raise ValueError("exact CPU head training authorization missing")
    allocation = subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], text=True)
    fields = dict(token.split("=", 1) for token in allocation.split() if "=" in token)
    if any(fields.get(key) != value for key, value in {
        "JobState": "RUNNING", "Partition": "cluster1-cpu", "NumNodes": "1", "NumCPUs": "16",
        "NumTasks": "1", "TimeLimit": "02:00:00", "MinMemoryNode": "64G", "OverSubscribe": "OK",
    }.items()) or "gpu" in fields.get("TRES", "").lower():
        raise ValueError("live training allocation differs")
    write(ROOT / f"running-allocation-{os.environ['SLURM_JOB_ID']}.json", {"controller": allocation})
    for row in plan["frozen_files"]:
        if digest(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"frozen input/source changed: {row['path']}")
    if (torch.__version__ != "2.7.1+cpu" or np.__version__ != "2.2.6"
            or os.environ.get("PYTHONPATH") != str(Path(plan["source_snapshot"]) / "src")):
        raise ValueError("CPU runtime/import identity differs")
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    manifest = load_fold_manifest(t36.FOLD_MANIFEST_PATH)
    if manifest.manifest_checksum != plan["fold_manifest_checksum"]:
        raise ValueError("fold identity differs")
    with np.load(plan["pooled_cohort_path"], allow_pickle=False) as bank:
        pooled = torch.from_numpy(bank["pooled"].copy())
        uids = [str(uid) for uid in bank["study_uids"]]
        metadata = json.loads(str(bank["metadata"]))
    if metadata["feature_manifest_sha256"] != plan["feature_manifest_sha256"]:
        raise ValueError("pooled bank feature lineage differs")
    uid_index = {uid: index for index, uid in enumerate(uids)}
    gold = {member.study_instance_uid: member for member in manifest.gold_memberships}
    predictions = {arm: {} for arm in ["reference-only", "report-supervision"]}
    zero_predictions = {arm: {} for arm in predictions}
    run_records = []
    optimization = plan["optimization"]
    parameters = {key: optimization[key] for key in ["successful_steps", "batch_size", "initialization_seed",
                   "batch_order_seed", "weak_seed_offset", "learning_rate", "checkpoint_interval_steps"]}
    for row in plan["folds"]:
        fold = row["fold"]
        admitted = admit_ticket30_fold_supervision(manifest, fold, t36.TICKET30_ARTIFACT,
            expected_artifact_sha256=t36.TICKET30_AUDITED_ARTIFACT_SHA256).for_image_studies(uids)
        if admitted.to_public_dict() != row["supervision_admission"]:
            raise ValueError("frozen fold report admission differs")
        training_uids = row["gold_training_study_uids"]
        heldout_uids = row["gold_heldout_study_uids"]
        weak_uids = row["weak_training_study_uids"]
        if set(training_uids + weak_uids) & set(manifest.held_out_study_uids(fold)):
            raise ValueError("held-out studies entered optimization")
        targets = torch.zeros((len(uids), 12), dtype=torch.float32)
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for uid in training_uids + heldout_uids + weak_uids:
            values = gold[uid].reference_labels if uid in gold else admitted.by_study[uid].targets
            for target, value in enumerate(values):
                if value is not None:
                    targets[uid_index[uid], target] = float(value)
                    mask[uid_index[uid], target] = True
        matched = []
        heldout_indices = [uid_index[uid] for uid in heldout_uids]
        for arm in predictions:
            run_root = ROOT / arm / f"fold-{fold}"
            run_root.mkdir(mode=0o700, parents=True)
            identity = {"plan_sha256": expected_sha, "fold": fold, "arm": arm,
                        "head": plan["head"], "feature_manifest_sha256": plan["feature_manifest_sha256"],
                        "pooled_cohort_sha256": plan["pooled_cohort_sha256"],
                        "fold_manifest_checksum": manifest.manifest_checksum}
            def checkpoint(head, optimizer, counts):
                final = counts["successful_optimizer_steps"] == optimization["successful_steps"]
                save_head_checkpoint(run_root / "checkpoints" / ("final.pt" if final else "latest.pt"),
                                     head, optimizer, counts, identity, final=final)
            head, counts = train_head(pooled=pooled, targets=targets, mask=mask, study_uids=uids,
                reference_indices=[uid_index[uid] for uid in training_uids],
                weak_indices=[uid_index[uid] for uid in weak_uids] if arm == "report-supervision" else [],
                checkpoint=checkpoint, **parameters)
            reloaded, replay_counts = reload_head(run_root / "checkpoints/final.pt", identity)
            if replay_counts != counts:
                raise ValueError("reloaded successful-step/exposure state differs")
            with torch.inference_mode():
                values = torch.sigmoid(head(pooled[heldout_indices]))
                replayed = torch.sigmoid(reloaded(pooled[heldout_indices]))
                ablated = torch.sigmoid(reloaded(zero_image_features(pooled[heldout_indices])))
            if not torch.equal(values, replayed) or not torch.isfinite(values).all() or not torch.isfinite(ablated).all():
                raise ValueError("checkpoint replay failed or predictions are nonfinite")
            arm_predictions = {uid: replayed[index].tolist() for index, uid in enumerate(heldout_uids)}
            predictions[arm].update(arm_predictions)
            zero_predictions[arm].update({uid: ablated[index].tolist() for index, uid in enumerate(heldout_uids)})
            record = {"schema_version": "rsna-ticket50-frozen-head-run-v1", "status": "complete",
                "identity": identity, "counts": counts, "predictions": arm_predictions,
                "checkpoint_replay_max_abs_error": 0.0,
                "held_out_inputs_excluded_from_fitting": True,
                "training_study_uids": training_uids + (weak_uids if arm == "report-supervision" else []),
                "held_out_study_uids": heldout_uids,
                "supervision_admission": row["supervision_admission"] if arm == "report-supervision" else None}
            if arm == "report-supervision" and counts["distinct_weak_studies_exposed"] != len(weak_uids):
                raise ValueError("full weak training pool was not exposed")
            write(run_root / "run.json", record)
            run_records.append({"fold": fold, "arm": arm, "path": str(run_root / "run.json"),
                                "sha256": digest(run_root / "run.json")})
            matched.append(counts)
            print(json.dumps({"fold": fold, "arm": arm, "completed_steps": counts["successful_optimizer_steps"],
                              "elapsed_seconds": time.monotonic() - started}), flush=True)
        for key in ["initial_head_state_sha256", "gold_exposure_sequence_sha256"]:
            if matched[0][key] != matched[1][key]:
                raise ValueError(f"matched gold/report pair differs: {key}")
    resnet = {arm: {} for arm in predictions}
    for arm in resnet:
        for path in plan["comparator_paths"][arm]:
            resnet[arm].update(read(Path(path))["predictions"])
    blend = {uid: ((np.asarray(resnet["report-supervision"][uid]) + predictions["report-supervision"][uid]) / 2).tolist()
             for uid in predictions["report-supervision"]}
    comparisons = {}
    pairs = {
        "report_minus_gold": (predictions["reference-only"], predictions["report-supervision"]),
        "convnext_report_minus_resnet_report": (resnet["report-supervision"], predictions["report-supervision"]),
        "convnext_gold_minus_resnet_gold": (resnet["reference-only"], predictions["reference-only"]),
        "fixed_half_blend_minus_resnet_report": (resnet["report-supervision"], blend),
        "gold_minus_zero_features": (zero_predictions["reference-only"], predictions["reference-only"]),
        "report_minus_zero_features": (zero_predictions["report-supervision"], predictions["report-supervision"]),
    }
    for name, (baseline, candidate) in pairs.items():
        comparisons[name] = compare_image_model_predictions(baseline, candidate, manifest,
            bootstrap_resamples=plan["analysis"]["bootstrap_resamples"], bootstrap_seed=plan["analysis"]["seed"])
        print(json.dumps({"analysis_complete": name}), flush=True)
    evaluations = {arm: evaluate_oof_predictions(values, manifest,
        bootstrap_resamples=2000, bootstrap_seed=20260930, permutation_resamples=10000,
        permutation_seed=20260930).to_public_dict() for arm, values in predictions.items()}
    def passes(name):
        value = comparisons[name]
        lower = value["paired_study_bootstrap"]["macro_auc"]["lower"]
        return lower is not None and lower > 0 and all(row["delta_macro_auc"] is not None and row["delta_macro_auc"] > 0 for row in value["delta"]["folds"])
    disposition = ("provisional_standalone_review" if passes("convnext_report_minus_resnet_report")
                   else "provisional_fixed_blend_review" if passes("fixed_half_blend_minus_resnet_report")
                   else "inconclusive_no_promotion")
    output = {"schema_version": "rsna-ticket50-frozen-head-analysis-v1", "plan_sha256": expected_sha,
        "job_id": os.environ["SLURM_JOB_ID"], "comparisons": comparisons, "evaluations": evaluations,
        "run_records": run_records, "fixed_blend_predictions": blend, "disposition": disposition,
        "candidate_promotion": False, "development_cohort_reused": True,
        "elapsed_seconds": time.monotonic() - started,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "external_spend_usd": 0, "gpu_count": 0}
    write(ROOT / "analysis.json", output)
    lines = ["# Ticket 50 — frozen ConvNeXt head experiment", "",
        "Fixed 1,500-step matched runs use the same repeatedly reused 58-study development cohort. These are OOF development results; no competition score or final candidate is established.", "",
        "| Comparison | Baseline AUC | Candidate AUC | Delta | Paired 95% interval |",
        "| --- | ---: | ---: | ---: | --- |"]
    for name, value in comparisons.items():
        interval = value["paired_study_bootstrap"]["macro_auc"]
        lines.append(f"| {name} | {value['baseline']['pooled_macro_auc']} | {value['candidate']['pooled_macro_auc']} | {value['delta']['pooled_macro_auc']} | {interval['lower']} to {interval['upper']} |")
    lines.extend(["", f"Disposition: **{disposition}**. Candidate promotion remains false.", "",
                  "The full evidence includes per-target support, undefined bootstrap outcomes, all fold deltas, descriptor-preserving feature ablations, 10,000 within-fold permutations, complete report-pool exposure and exact held-out checkpoint replay.", ""])
    with (ROOT / "analysis.md").open("x") as handle:
        handle.write("\n".join(lines))
    return {"disposition": disposition, "analysis_path": str(ROOT / "analysis.json")}


if __name__ == "__main__":
    import sys
    print(json.dumps(freeze() if sys.argv[1] == "freeze" else run(sys.argv[2])))
