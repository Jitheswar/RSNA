"""One fixed duration experiment on admitted, immutable ConvNeXt features."""

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
from rsna_knee.folds import load_fold_manifest
from rsna_knee.frozen_head_experiment import (
    reload_head, save_head_checkpoint, train_head, zero_image_features,
)
from rsna_knee.image_model_comparison import compare_image_model_predictions
from rsna_knee.ticket34 import admit_ticket30_fold_supervision


PROJECT = t36.PROJECT_ROOT
PARENT = PROJECT / "artifacts/private/ticket50-convnext-supervision"
ROOT = PROJECT / "artifacts/private/ticket54-convnext-long-report-head"
AUTHORIZATION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/campaign-owner-autonomy-20261001.json"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket54-convnext-long-report-head-20261001.json"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    path.chmod(0o600)


def prepare():
    if ROOT.exists() or DECISION.exists():
        raise FileExistsError("Immutable Ticket 54 request already exists")
    parent = read(PARENT / "plan-v2.json")
    analysis = read(PARENT / "analysis.json")
    if analysis["gpu_count"] != 0 or len(analysis["run_records"]) != 6:
        raise ValueError("Complete original paired experiment is required")
    inputs = [PARENT / "plan-v2.json", PARENT / "analysis.json", AUTHORIZATION,
              Path(parent["pooled_cohort_path"]), Path(parent["feature_manifest_path"]),
              t36.FOLD_MANIFEST_PATH, t36.TICKET30_ARTIFACT]
    for key in ("pooled_cohort", "feature_manifest"):
        if digest(parent[key + "_path"]) != parent[key + "_sha256"]:
            raise ValueError("Admitted feature inputs changed")
    for arm in ("reference-only", "report-supervision"):
        for fold in range(3):
            path = PARENT / arm / f"fold-{fold}/run.json"
            old = read(path)
            if (old["status"] != "complete" or old["counts"]["successful_optimizer_steps"] != 1500
                    or old["held_out_inputs_excluded_from_fitting"] is not True):
                raise ValueError("Original duration comparator differs")
            inputs.extend((path, path.parent / "checkpoints/final.pt"))
    ROOT.mkdir(mode=0o700, parents=True)
    snapshot = ROOT / "source-snapshot"
    shutil.copytree(PROJECT / "src/rsna_knee", snapshot / "src/rsna_knee",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(Path(__file__), snapshot / Path(__file__).name)
    inputs.extend(path for path in snapshot.rglob("*") if path.is_file())
    resources = {"partition": "cluster1-cpu", "nodes": 1, "tasks": 1,
                 "cpus_per_task": 16, "memory_gib": 64, "walltime_seconds": 1800,
                 "code_watchdog_seconds": 1200, "gpu_count": 0, "exclusive": False,
                 "maximum_concurrent_jobs": 1, "maximum_launch_attempts": 1,
                 "external_spend_usd": 0, "requeue": False}
    plan = {"schema_version": "rsna-ticket54-convnext-duration-plan-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_root": str(ROOT),
            "source_snapshot": str(snapshot), "parent_plan": str(PARENT / "plan-v2.json"),
            "parent_plan_sha256": digest(PARENT / "plan-v2.json"),
            "authorization_sha256": digest(AUTHORIZATION), "decision_path": str(DECISION),
            "fold_manifest_checksum": parent["fold_manifest_checksum"],
            "optimization": {**parent["optimization"], "successful_steps": 7500,
                             "checkpoint_interval_steps": 1500},
            "unchanged_science": "Features, folds, head, report admission, optimizer, loss, learning rate, initialization and samplers match Ticket 50; only successful-step budget changes.",
            "prefix_replay": {"successful_steps": 1500, "maximum_absolute_error": 1e-6,
                              "check": ["ordered exposure digests", "initial state", "final state", "held-out predictions"]},
            "analysis": {"bootstrap_resamples": 2000, "seed": 20261001,
                         "pairs": ["gold_long_minus_short", "report_long_minus_short", "report_minus_gold",
                                   "gold_minus_zero_image", "report_minus_zero_image"],
                         "provisional_review_gate": "Long report versus its short comparator and image ablation must both have positive paired lower bounds and positive directions in all three folds. No automatic promotion.",
                         "cohort_reused": True, "not_untouched_generalization": True},
            "runtime_versions": {name: importlib.metadata.version(name) for name in ("numpy", "torch")},
            "resources": resources, "network_during_execution": False,
            "new_downloads": False, "competition_submission": False,
            "final_candidate_selection": False, "publication": False,
            "frozen_files": [{"path": str(path), "sha256": digest(path)} for path in inputs]}
    write(ROOT / "plan.json", plan)
    checksum = digest(ROOT / "plan.json")
    write(DECISION, {"schema_version": "rsna-campaign-autonomous-decision-v1",
          "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "decision": "approved",
          "decision_made_by": "campaign agent under standing owner authorization",
          "authorization_path": str(AUTHORIZATION), "authorization_sha256": digest(AUTHORIZATION),
          "new_exact_owner_numeric_approval_claimed": False, "plan_sha256": checksum,
          "resources": resources, "basis": "Test underexposure of already admitted report-supervised frozen features while the sole GPU scoring request is pending.",
          "gpu_ledger_cap_changed": False, "new_gpu_request": False,
          "new_weight_download": False, "candidate_promotion": False})
    launcher = ROOT / "run.sbatch"
    with launcher.open("x") as handle:
        handle.write(f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket54-duration
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
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
''')
    write(ROOT / "preparation-validation.json", {"plan_sha256": checksum,
          "launcher_sha256": digest(launcher), "compiled": True,
          "fold_count": 3, "training_runs": 6, "launched": False})
    print(json.dumps({"plan_sha256": checksum, "launcher": str(launcher)}))


def run(plan_path, expected_sha):
    started = time.monotonic()
    if digest(plan_path) != expected_sha:
        raise ValueError("Frozen duration plan changed")
    plan = read(plan_path)
    decision = read(plan["decision_path"])
    if (decision["decision"] != "approved" or decision["plan_sha256"] != expected_sha
            or decision["resources"] != plan["resources"]
            or digest(AUTHORIZATION) != plan["authorization_sha256"]):
        raise ValueError("Exact autonomous resource decision differs")
    allocation = subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], text=True)
    fields = dict(token.split("=", 1) for token in allocation.split() if "=" in token)
    if any(fields.get(key) != value for key, value in {
        "JobState": "RUNNING", "Partition": "cluster1-cpu", "NumNodes": "1", "NumCPUs": "16",
        "NumTasks": "1", "TimeLimit": "00:30:00", "MinMemoryNode": "64G", "OverSubscribe": "OK",
        "Requeue": "0", "Restarts": "0",
    }.items()) or "gpu" in (fields.get("TRES", "") + fields.get("TresPerNode", "")).lower():
        raise ValueError("Live CPU allocation differs")
    root = Path(plan["run_root"])
    write(root / f"running-allocation-{os.environ['SLURM_JOB_ID']}.json", {"controller": allocation})
    for row in plan["frozen_files"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("Frozen source, prior checkpoint or input changed")
    for name, version in plan["runtime_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Frozen runtime version changed")
    if os.environ.get("PYTHONPATH") != str(Path(plan["source_snapshot"]) / "src") or torch.cuda.is_available():
        raise ValueError("Immutable CPU runtime is required")
    watchdog = threading.Timer(plan["resources"]["code_watchdog_seconds"], lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    parent = read(plan["parent_plan"])
    manifest = load_fold_manifest(t36.FOLD_MANIFEST_PATH)
    if manifest.manifest_checksum != plan["fold_manifest_checksum"]:
        raise ValueError("Corrected development folds differ")
    with np.load(parent["pooled_cohort_path"], allow_pickle=False) as bank:
        pooled = torch.from_numpy(bank["pooled"].copy())
        uids = [str(uid) for uid in bank["study_uids"]]
        metadata = json.loads(str(bank["metadata"]))
    if pooled.shape != (4228, 3100) or len(set(uids)) != 4228 or metadata["feature_manifest_sha256"] != parent["feature_manifest_sha256"]:
        raise ValueError("Sealed feature bank lineage or dimensions differ")
    index = {uid: i for i, uid in enumerate(uids)}
    gold = {member.study_instance_uid: member for member in manifest.gold_memberships}
    arms = ("reference-only", "report-supervision")
    predictions = {arm: {} for arm in arms}
    short = {arm: {} for arm in arms}
    ablated = {arm: {} for arm in arms}
    records = []
    optimization = plan["optimization"]
    parameters = {key: optimization[key] for key in ("successful_steps", "batch_size", "initialization_seed",
                  "batch_order_seed", "weak_seed_offset", "learning_rate", "checkpoint_interval_steps")}
    for row in parent["folds"]:
        fold = row["fold"]
        admission = admit_ticket30_fold_supervision(manifest, fold, t36.TICKET30_ARTIFACT,
            expected_artifact_sha256=t36.TICKET30_AUDITED_ARTIFACT_SHA256).for_image_studies(uids)
        if admission.to_public_dict() != row["supervision_admission"]:
            raise ValueError("Original report admission differs")
        training, heldout, weak = [row[key] for key in ("gold_training_study_uids", "gold_heldout_study_uids", "weak_training_study_uids")]
        if (set(training + weak) & set(manifest.held_out_study_uids(fold))
                or set(training) != {uid for uid, member in gold.items() if member.fold != fold}
                or set(heldout) != {uid for uid, member in gold.items() if member.fold == fold}
                or set(weak) != set(admission.admitted_report_only_study_uids)):
            raise ValueError("Original training partitions or fold isolation differ")
        targets = torch.zeros((len(uids), 12), dtype=torch.float32)
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for uid in training + weak:
            values = gold[uid].reference_labels if uid in gold else admission.by_study[uid].targets
            for column, value in enumerate(values):
                if value is not None:
                    targets[index[uid], column] = float(value)
                    mask[index[uid], column] = True
        heldout_indices = [index[uid] for uid in heldout]
        if mask[heldout_indices].any():
            raise ValueError("Held-out labels entered the training target tensor")
        matched = []
        for arm in arms:
            old_path = PARENT / arm / f"fold-{fold}/run.json"
            original = read(old_path)
            original_checkpoint = torch.load(old_path.parent / "checkpoints/final.pt", map_location="cpu", weights_only=True)
            if original_checkpoint["identity"] != original["identity"]:
                raise ValueError("Original checkpoint identity differs")
            short[arm].update(original["predictions"])
            run_root = root / arm / f"fold-{fold}"
            identity = {"plan_sha256": expected_sha, "fold": fold, "arm": arm,
                        "head": parent["head"], "feature_manifest_sha256": parent["feature_manifest_sha256"],
                        "pooled_cohort_sha256": parent["pooled_cohort_sha256"],
                        "fold_manifest_checksum": manifest.manifest_checksum}
            prefix = []
            def checkpoint(head, optimizer, counts):
                step = counts["successful_optimizer_steps"]
                if step == 1500:
                    with torch.inference_mode():
                        scores = torch.sigmoid(head(pooled[heldout_indices]))
                    error = max(abs(x - y) for i, uid in enumerate(heldout)
                                for x, y in zip(scores[i].tolist(), original["predictions"][uid], strict=True))
                    state_error = max(float((value - original_checkpoint["head_state_dict"][key]).abs().max())
                                      for key, value in head.state_dict().items())
                    keys = ("initial_head_state_sha256", "gold_exposure_sequence_sha256", "weak_exposure_sequence_sha256",
                            "attempted_batches", "skipped_batches", "exposures_by_source")
                    if (any(counts[key] != original["counts"][key] for key in keys)
                            or error > plan["prefix_replay"]["maximum_absolute_error"]
                            or state_error > plan["prefix_replay"]["maximum_absolute_error"]):
                        raise ValueError(f"Original 1500-step prefix differs: score={error}, state={state_error}")
                    prefix.append({"prediction_max_abs_error": error, "state_max_abs_error": state_error,
                                   "ordered_exposures_match": True, "initial_state_matches": True})
                final = step == optimization["successful_steps"]
                save_head_checkpoint(run_root / "checkpoints" / ("final.pt" if final else "latest.pt"),
                                     head, optimizer, counts, identity, final=final)
            head, counts = train_head(pooled=pooled, targets=targets, mask=mask, study_uids=uids,
                reference_indices=[index[uid] for uid in training],
                weak_indices=[index[uid] for uid in weak] if arm == "report-supervision" else [],
                checkpoint=checkpoint, **parameters)
            restored, replay_counts = reload_head(run_root / "checkpoints/final.pt", identity)
            if replay_counts != counts or len(prefix) != 1:
                raise ValueError("Final state or original prefix replay is missing")
            with torch.inference_mode():
                scores = torch.sigmoid(head(pooled[heldout_indices]))
                replayed = torch.sigmoid(restored(pooled[heldout_indices]))
                zero = torch.sigmoid(restored(zero_image_features(pooled[heldout_indices])))
            if not torch.equal(scores, replayed) or not torch.isfinite(replayed).all() or not torch.isfinite(zero).all():
                raise ValueError("Final checkpoint replay or predictions differ")
            if counts["skipped_batches"] or (arm == "report-supervision" and counts["distinct_weak_studies_exposed"] != len(weak)):
                raise ValueError("Successful budget or complete report exposure differs")
            predictions[arm].update({uid: replayed[i].tolist() for i, uid in enumerate(heldout)})
            ablated[arm].update({uid: zero[i].tolist() for i, uid in enumerate(heldout)})
            record = {"schema_version": "rsna-ticket54-convnext-duration-run-v1", "status": "COMPLETE",
                      "identity": identity, "counts": counts, "prefix_replay": prefix[0],
                      "predictions": {uid: predictions[arm][uid] for uid in heldout},
                      "training_study_uids": training + (weak if arm == "report-supervision" else []),
                      "held_out_study_uids": heldout, "held_out_inputs_excluded_from_fitting": True,
                      "held_out_target_entries_masked": True, "checkpoint_replay_max_abs_error": 0.0,
                      "final_checkpoint_sha256": digest(run_root / "checkpoints/final.pt")}
            write(run_root / "run.json", record)
            records.append({"fold": fold, "arm": arm, "path": str(run_root / "run.json"), "sha256": digest(run_root / "run.json")})
            matched.append(counts)
            print(json.dumps({"fold": fold, "arm": arm, "steps": counts["successful_optimizer_steps"],
                              "elapsed_seconds": time.monotonic() - started}), flush=True)
        if any(matched[0][key] != matched[1][key] for key in ("initial_head_state_sha256", "gold_exposure_sequence_sha256")):
            raise ValueError("Matched gold/report exposures or initialization differ")
    pairs = {
        "gold_long_minus_short": (short["reference-only"], predictions["reference-only"]),
        "report_long_minus_short": (short["report-supervision"], predictions["report-supervision"]),
        "report_minus_gold": (predictions["reference-only"], predictions["report-supervision"]),
        "gold_minus_zero_image": (ablated["reference-only"], predictions["reference-only"]),
        "report_minus_zero_image": (ablated["report-supervision"], predictions["report-supervision"]),
    }
    comparisons = {name: compare_image_model_predictions(base, candidate, manifest,
        bootstrap_resamples=plan["analysis"]["bootstrap_resamples"], bootstrap_seed=plan["analysis"]["seed"])
        for name, (base, candidate) in pairs.items()}
    def passes(name):
        value = comparisons[name]
        lower = value["paired_study_bootstrap"]["macro_auc"]["lower"]
        return lower is not None and lower > 0 and all(row["delta_macro_auc"] is not None and row["delta_macro_auc"] > 0 for row in value["delta"]["folds"])
    disposition = "provisional_development_review" if passes("report_long_minus_short") and passes("report_minus_zero_image") else "inconclusive_no_promotion"
    output = {"schema_version": "rsna-ticket54-convnext-duration-analysis-v1", "status": "COMPLETE",
              "observed_at_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": expected_sha,
              "job_id": os.environ["SLURM_JOB_ID"], "comparisons": comparisons, "run_records": records,
              "disposition": disposition, "candidate_promotion": False, "gpu_count": 0,
              "external_spend_usd": 0, "development_cohort_reused": True,
              "elapsed_seconds": time.monotonic() - started,
              "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "interpretation_limit": "Repeatedly reused reference OOF development cohort; not an untouched generalization estimate, public score or final private rank."}
    write(root / "analysis.json", output)
    lines = ["# Ticket 54 — fixed longer ConvNeXt report-head budget", "",
             "Final 7,500-step checkpoints only; three matched gold/report pairs. Reused OOF development data.", "",
             "| Comparison | Baseline AUC | Candidate AUC | Delta | Paired 95% interval |",
             "| --- | ---: | ---: | ---: | --- |"]
    for name, value in comparisons.items():
        interval = value["paired_study_bootstrap"]["macro_auc"]
        lines.append(f"| {name} | {value['baseline']['pooled_macro_auc']:.6f} | {value['candidate']['pooled_macro_auc']:.6f} | {value['delta']['pooled_macro_auc']:+.6f} | {interval['lower']:+.6f} to {interval['upper']:+.6f} |")
    lines.extend(["", f"Disposition: **{disposition}**. No automatic promotion, submission or first-place claim.",
                  "Full evidence retains target support, all fold directions, undefined bootstrap outcomes, actual exposures and original/final checkpoint replay.", ""])
    with (root / "analysis.md").open("x") as handle:
        handle.write("\n".join(lines))
    watchdog.cancel()
    print(json.dumps({"status": "COMPLETE", "disposition": disposition, "analysis_path": str(root / "analysis.json")}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        run(Path(sys.argv[2]), sys.argv[3])
