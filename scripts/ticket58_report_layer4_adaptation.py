"""Fit the prospectively registered six matched warm-head/adaptation runs."""

from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import threading
import time

from ticket58_prefix_cost import PROJECT, PREFIX, PARENT, AUTH, digest, read, write


COST = PROJECT / "artifacts/private/ticket58-prefix-cost-v1"
ROOT = PROJECT / "artifacts/private/ticket58-report-layer4-adaptation-v1"
DECISION = PROJECT / ".scratch/rsna-leaderboard/resource-approvals/ticket58-report-layer4-fit-20261001.json"
ARMS = ("head-continuation", "layer4-adaptation")


def prepare():
    if ROOT.exists() or DECISION.exists():
        raise FileExistsError("Immutable six-run adaptation request already prepared")
    cost = read(COST / "cost-result.json")
    accepted = read(COST / "accepted-launch.json")
    terminal = COST / f"controller-{accepted['job_id']}/terminal-controller.txt"
    fields = dict(token.split("=", 1) for token in terminal.read_text().split() if "=" in token)
    cost_plan = read(COST / "plan.json")
    if (cost["status"] != "COMPLETE" or cost["cost_disposition"] != "eligible_for_separate_fixed_fit_decision"
            or cost["job_id"] != str(accepted["job_id"]) or cost["plan_sha256"] != digest(COST / "plan.json")
            or fields.get("JobId") != cost["job_id"] or fields.get("JobState") != "COMPLETED"
            or fields.get("ExitCode") != "0:0" or cost["full_loaded_studies"] != 4228
            or not cost["fitted_probe_states_discarded"] or cost["held_out_predictions_computed"]
            or any(row["maximum_frozen_logit_error"] > 1e-6 or row["normalization_buffer_max_abs_change"] != 0
                   or row["frozen_parameter_max_abs_change"] != 0 for row in cost["folds"])):
        raise ValueError("Successful matching full-cache cost/scope evidence is required")
    for row in cost_plan["frozen_files"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("Actual complete-cohort cost dependency changed")
    from rsna_knee import ticket36 as t36
    ROOT.mkdir(mode=0o700)
    snapshot = ROOT / "source-snapshot"
    shutil.copytree(PROJECT / "src/rsna_knee", snapshot / "src/rsna_knee",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for script in (Path(__file__), PROJECT / "scripts/ticket58_prefix_cost.py"):
        shutil.copyfile(script, snapshot / script.name)
    spec = ROOT / "prospective-spec.md"
    shutil.copyfile(PROJECT / ".scratch/rsna-leaderboard/issues/58-test-report-supervised-layer4-adaptation.md", spec)
    inputs = [AUTH, COST / "plan.json", COST / "cost-result.json", terminal,
              terminal.with_name("terminal-observation.json"), PREFIX / "plan.json",
              PREFIX / "features/feature-manifest.json", PREFIX / "verification/verification-result.json",
              PARENT / "plan.json", PARENT / "validation.json", PARENT / "pooled-cohort.npz",
              PARENT / "pooled-cohort-evidence.json", Path(cost_plan["cohort_parent_plan"]),
              t36.FOLD_MANIFEST_PATH, t36.TICKET30_ARTIFACT, t36.WEIGHTS_PATH, spec]
    for fold in range(3):
        inputs.extend((PARENT / f"report-supervision/fold-{fold}/run.json",
                       PARENT / f"report-supervision/fold-{fold}/checkpoints/final.pt"))
    inputs.extend(path for path in snapshot.rglob("*") if path.is_file())
    resources = {"partition": "cluster1-cpu", "nodes": 1, "tasks": 1, "cpus_per_task": 16,
                 "memory_gib": 64, "walltime_seconds": 10800, "code_watchdog_seconds": 10560,
                 "maximum_launch_attempts": 1, "maximum_concurrent_jobs": 1, "gpu_count": 0,
                 "exclusive": False, "requeue": False, "external_spend_usd": 0}
    plan = {"schema_version": "rsna-ticket58-report-layer4-fit-plan-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_root": str(ROOT),
            "source_snapshot": str(snapshot), "decision_path": str(DECISION), "resources": resources,
            "prefix_root": str(PREFIX), "parent_root": str(PARENT),
            "cohort_parent_plan": cost_plan["cohort_parent_plan"],
            "fold_manifest_path": str(t36.FOLD_MANIFEST_PATH),
            "fold_manifest_checksum": read(PARENT / "plan.json")["fold_manifest_checksum"],
            "feature_manifest_sha256": digest(PREFIX / "features/feature-manifest.json"),
            "pooled_control_sha256": digest(PARENT / "pooled-cohort.npz"),
            "model": {**read(PREFIX / "plan.json")["model"], "model_id": "resnet18-layer4-finetune-v1"},
            "optimization": {"successful_steps": 1500, "checkpoint_interval_steps": 500,
                "draws_per_source": 2, "batch_order_seed": 20260912, "weak_seed_offset": 34034034,
                "initialization_seed": 20261001, "head_learning_rate": 1e-4,
                "layer4_learning_rate": 1e-5, "optimizer": "Adam", "augmentation": False,
                "normalization_buffers_fixed": True, "optimizer_restarted_from_warm_weights": True},
            "inference": {"threads": 16, "batch_size": 2, "maximum_absolute_error": 1e-6},
            "analysis": {"bootstrap_resamples": 2000, "seed": 20261001,
                "primary": "adaptation_minus_matched_head_continuation",
                "provisional_gate": "Positive primary paired lower bound and positive all-fold directions with image controls; independent actual native replay still required.",
                "development_cohort_reused": True},
            "cost_projection_seconds_with_allowance": cost["projected_complete_fit_seconds_with_allowance"],
            "runtime_versions": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "pydicom")},
            "frozen_files": [{"path": str(path), "sha256": digest(path)} for path in inputs],
            "network_during_execution": False, "new_downloads": False, "candidate_promotion": False,
            "competition_submission": False, "final_selection": False}
    write(ROOT / "plan.json", plan)
    checksum = digest(ROOT / "plan.json")
    write(DECISION, {"schema_version": "rsna-campaign-autonomous-decision-v1", "decision": "approved",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_made_by": "campaign agent under standing owner authorization", "authorization_sha256": digest(AUTH),
        "plan_sha256": checksum, "resources": resources,
        "basis": "Complete the registered six-run matched warm-head/layer4 comparison after actual full-bank loading, random-read, gradient/scope and exact frozen-logit checks. Measured conservative projection5676.86s fits one3h CPU allocation; probe weights are discarded.",
        "new_exact_owner_numeric_approval_claimed": False, "new_gpu_request": False,
        "gpu_ledger_cap_changed": False, "candidate_promotion": False})
    with (ROOT / "run.sbatch").open("x") as handle:
        handle.write(f'''#!/usr/bin/env bash
#SBATCH --job-name=rsna-ticket58-report-layer4
#SBATCH --partition=cluster1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
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
{PROJECT}/.venv/bin/python {snapshot}/{Path(__file__).name} run {ROOT}/plan.json {checksum}
''')
    subprocess.run(["bash", "-n", str(ROOT / "run.sbatch")], check=True)
    print({"plan_sha256": checksum, "launcher": str(ROOT / "run.sbatch"), "launched": False})


def save_checkpoint(path, model, optimizer, counts, identity, final):
    import torch
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if final and path.exists():
        raise FileExistsError("Final fixed-budget checkpoint already exists")
    temporary = path.with_suffix(".partial")
    torch.save({"schema_version": "rsna-ticket58-adaptation-checkpoint-v1",
        "checkpoint_kind": "final_selected" if final else "latest_resumable",
        "model_config": model.config.to_dict(), "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "identity": identity, "counts": counts,
        "rng_state": torch.get_rng_state()}, temporary)
    temporary.chmod(0o600)
    if final:
        os.link(temporary, path)
        temporary.unlink()
    else:
        temporary.replace(path)


def run(plan_path, checksum):
    started = time.monotonic()
    if digest(plan_path) != checksum:
        raise ValueError("Exact six-run scientific plan changed")
    plan = read(plan_path)
    decision = read(plan["decision_path"])
    if decision["decision"] != "approved" or decision["plan_sha256"] != checksum or decision["resources"] != plan["resources"]:
        raise ValueError("Exact six-run resource decision differs")
    root = Path(plan["run_root"])
    allocation = subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], text=True)
    fields = dict(token.split("=", 1) for token in allocation.split() if "=" in token)
    expected = {"JobState": "RUNNING", "Partition": "cluster1-cpu", "NumNodes": "1", "NumCPUs": "16",
                "NumTasks": "1", "MinMemoryNode": "64G", "TimeLimit": "03:00:00", "OverSubscribe": "OK",
                "Requeue": "0", "Restarts": "0"}
    if any(fields.get(key) != value for key, value in expected.items()) or "gpu" in fields.get("TRES", "").lower():
        raise ValueError("Actual bounded six-run CPU allocation differs")
    (root / "running-allocation.txt").write_text(allocation)
    watchdog = threading.Timer(plan["resources"]["code_watchdog_seconds"], lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    for row in plan["frozen_files"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("Frozen six-run source or scientific input changed")
    for name, version in plan["runtime_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Frozen CPU fit runtime changed")

    import json
    import numpy as np
    import torch
    from rsna_knee import ticket36 as t36
    from rsna_knee.folds import load_fold_manifest
    from rsna_knee.frozen_head_experiment import sequence_digest
    from rsna_knee.image_model_comparison import compare_image_model_predictions
    from rsna_knee.model import ModelConfig, build_study_model
    from rsna_knee.prefix_adaptation import prefix_logits
    from rsna_knee.prefix_cache import load_prefix_entry
    from rsna_knee.ticket34 import admit_ticket30_fold_supervision
    from rsna_knee.training import _array_sha256, _batch_indices, _model_state_sha256, masked_bce_loss, seed_everything

    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available() or torch.version.cuda is not None:
        raise ValueError("Six-run adaptation requires the admitted CPU runtime")
    prefix_root, parent = Path(plan["prefix_root"]), Path(plan["parent_root"])
    manifest = read(prefix_root / "features/feature-manifest.json")
    prefix_plan = read(prefix_root / "plan.json")
    rows = sorted(manifest["features"], key=lambda row: row["study_instance_uid"])
    uids = [row["study_instance_uid"] for row in rows]
    indices = {uid: position for position, uid in enumerate(uids)}
    entries = []
    for number, row in enumerate(rows):
        entries.append(load_prefix_entry(prefix_root / "features" / row["feature_file"],
            expected_sha256=row["feature_file_sha256"], expected_study_uid=row["study_instance_uid"],
            expected_identity_sha256=manifest["feature_identity_sha256"],
            expected_source_identity_sha256=row["source_identity_sha256"],
            expected_plan_sha256=manifest["extraction_plan_sha256"]))
        if (number + 1) % 512 == 0:
            print(json.dumps({"loaded_studies": number + 1}), flush=True)
    if len(entries) != 4228 or sum(len(entry.prefix) for entry in entries) != 187272:
        raise ValueError("Actual complete verified activation bank differs")
    with np.load(parent / "pooled-cohort.npz", allow_pickle=False) as bank:
        if bank["study_uids"].astype(str).tolist() != uids:
            raise ValueError("Pooled control study mapping differs")
        pooled = torch.from_numpy(bank["pooled"].copy())
    if pooled.shape != (4228, 2076) or _array_sha256(pooled.numpy()) != read(parent / "pooled-cohort-evidence.json")["pooled_array_sha256"]:
        raise ValueError("Actual immutable control arrays differ")
    zero = pooled.clone().reshape(-1, 4, 519)
    zero[:, :, :512] = 0
    zero = zero.reshape(-1, 2076)
    folds = read(plan["cohort_parent_plan"])["folds"]
    fold_manifest = load_fold_manifest(plan["fold_manifest_path"])
    if fold_manifest.manifest_checksum != plan["fold_manifest_checksum"] or fold_manifest.superseded:
        raise ValueError("Corrected three-fold identity differs")
    gold = {member.study_instance_uid: member for member in fold_manifest.gold_memberships}
    weights = torch.load(prefix_plan["checkpoint_path"], map_location="cpu", weights_only=True)
    predictions, ablated = ({arm: {} for arm in ARMS} for _ in range(2))
    parent_predictions, records = {}, []
    opt = plan["optimization"]
    def make_model(head_state, arm):
        seed_everything(opt["initialization_seed"])
        model = build_study_model(ModelConfig.from_dict(plan["model"]))
        model.load_pretrained_state(weights)
        model.head.load_state_dict(head_state, strict=True)
        model.encoder.layer4.requires_grad_(arm == "layer4-adaptation")
        model.train()
        return model
    def predict(model, positions, arm, zero_images=False):
        values = {}
        with torch.inference_mode():
            for start in range(0, len(positions), 2):
                batch = positions[start:start + 2]
                logits = (model.head((zero if zero_images else pooled)[batch]) if arm == "head-continuation"
                          else prefix_logits(model, [entries[i] for i in batch], zero_images=zero_images))
                scores = torch.sigmoid(logits).numpy()
                if not np.isfinite(scores).all():
                    raise ValueError("Nonfinite final fixed-budget prediction")
                values.update({uids[i]: value.tolist() for i, value in zip(batch, scores, strict=True)})
        return values
    for fold in folds:
        training, reports, heldout = [fold[key] for key in ("gold_training_study_uids", "weak_training_study_uids", "gold_heldout_study_uids")]
        if set(training + reports) & set(fold_manifest.held_out_study_uids(fold["fold"])):
            raise ValueError("Held-out study entered an optimizer partition")
        admission = admit_ticket30_fold_supervision(fold_manifest, fold["fold"], t36.TICKET30_ARTIFACT,
            expected_artifact_sha256=t36.TICKET30_AUDITED_ARTIFACT_SHA256).for_image_studies(uids)
        if admission.to_public_dict() != fold["supervision_admission"]:
            raise ValueError("Original admitted report supervision differs")
        targets = torch.zeros((len(uids), 12))
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for uid in training + reports:
            values = gold[uid].reference_labels if uid in gold else admission.by_study[uid].targets
            for column, value in enumerate(values):
                if value is not None:
                    targets[indices[uid], column] = float(value)
                    mask[indices[uid], column] = True
        heldout_positions = [indices[uid] for uid in heldout]
        if mask[heldout_positions].any():
            raise ValueError("Held-out target entries entered optimizer tensors")
        parent_run = read(parent / f"report-supervision/fold-{fold['fold']}/run.json")
        parent_checkpoint_path = parent / f"report-supervision/fold-{fold['fold']}/checkpoints/final.pt"
        if digest(parent_checkpoint_path) != parent_run["checkpoint_sha256"]:
            raise ValueError("Independently admitted warm-head bytes changed")
        parent_checkpoint = torch.load(parent_checkpoint_path, map_location="cpu", weights_only=True)
        parent_predictions.update(parent_run["predictions"])
        matched = []
        for arm in ARMS:
            model = make_model(parent_checkpoint["head_state_dict"], arm)
            initial = {key: value.clone() for key, value in model.state_dict().items()}
            identity = {"plan_sha256": checksum, "fold": fold["fold"], "arm": arm,
                "initial_model_state_sha256": _model_state_sha256(model),
                "initial_head_state_sha256": _model_state_sha256(model.head),
                "initial_encoder_state_sha256": _model_state_sha256(model.encoder),
                "parent_checkpoint_sha256": digest(parent_checkpoint_path),
                "feature_manifest_sha256": plan["feature_manifest_sha256"],
                "fold_manifest_checksum": fold_manifest.manifest_checksum,
                "target_array_sha256": _array_sha256(targets.numpy()), "target_mask_sha256": _array_sha256(mask.numpy())}
            groups = [{"params": model.head.parameters(), "lr": opt["head_learning_rate"]}]
            if arm == "layer4-adaptation":
                groups.insert(0, {"params": model.encoder.layer4.parameters(), "lr": opt["layer4_learning_rate"]})
            optimizer = torch.optim.Adam(groups)
            gold_sequence, report_sequence, losses = [], [], []
            source_counts = {kind: {"sample_exposures": 0, "supervised_entry_exposures": 0}
                             for kind in ("reference", "weak")}
            run_root = root / arm / f"fold-{fold['fold']}"
            run_started = time.monotonic()
            for attempt in range(opt["successful_steps"]):
                selected = []
                for kind, members, seed in (("reference", training, opt["batch_order_seed"]),
                                           ("weak", reports, opt["batch_order_seed"] + opt["weak_seed_offset"])):
                    batch_uids = [members[i] for i in _batch_indices(len(members), opt["draws_per_source"], attempt, seed)]
                    batch = [indices[uid] for uid in batch_uids]
                    (gold_sequence if kind == "reference" else report_sequence).extend(batch_uids)
                    source_counts[kind]["sample_exposures"] += len(batch)
                    source_counts[kind]["supervised_entry_exposures"] += int(mask[batch].sum())
                    selected.extend(batch)
                optimizer.zero_grad(set_to_none=True)
                logits = (model.head(pooled[selected]) if arm == "head-continuation"
                          else prefix_logits(model, [entries[i] for i in selected]))
                loss = masked_bce_loss(logits, targets[selected], mask[selected])
                if loss is None or not torch.isfinite(loss):
                    raise ValueError("Entirely masked or nonfinite fixed training loss")
                loss.backward()
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters() if parameter.requires_grad):
                    raise ValueError("Missing or nonfinite fixed training gradient")
                optimizer.step()
                losses.append(float(loss.detach()))
                if len(losses) % opt["checkpoint_interval_steps"] == 0:
                    counts = {"successful_optimizer_steps": len(losses), "attempted_batches": attempt + 1,
                        "skipped_batches": 0, "loss_history": losses,
                        "gold_exposure_sequence_sha256": sequence_digest(gold_sequence),
                        "weak_exposure_sequence_sha256": sequence_digest(report_sequence),
                        "distinct_weak_studies_exposed": len(set(report_sequence)),
                        "weak_training_pool_count": len(reports), "exposures_by_source": source_counts,
                        "elapsed_seconds": time.monotonic() - run_started}
                    final = len(losses) == opt["successful_steps"]
                    save_checkpoint(run_root / "checkpoints" / ("final.pt" if final else "latest.pt"),
                                    model, optimizer, counts, identity, final)
                    print(json.dumps({"fold": fold["fold"], "arm": arm, "completed_updates": len(losses),
                                      "elapsed_seconds": time.monotonic() - run_started}), flush=True)
            current = model.state_dict()
            frozen_error = max(float((value - initial[key]).abs().max()) for key, value in current.items()
                if not key.startswith(("encoder.layer4.", "head.")))
            buffer_error = max(float((value - initial[key]).abs().max()) for key, value in current.items()
                if key.endswith(("running_mean", "running_var", "num_batches_tracked")))
            changes = {group: max(float((value - initial[key]).abs().max()) for key, value in current.items()
                                 if key.startswith(group)) for group in ("encoder.layer4.", "head.")}
            if (frozen_error != 0 or buffer_error != 0 or changes["head."] <= 0
                    or (arm == "layer4-adaptation" and changes["encoder.layer4."] <= 0)
                    or (arm == "head-continuation" and changes["encoder.layer4."] != 0)):
                raise ValueError("Final trainable scope or normalization differs")
            checkpoint_path = run_root / "checkpoints/final.pt"
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if saved["identity"] != identity or saved["counts"] != counts or saved["checkpoint_kind"] != "final_selected":
                raise ValueError("Final checkpoint contract differs")
            restored = make_model(parent_checkpoint["head_state_dict"], arm)
            restored.load_state_dict(saved["model_state_dict"], strict=True)
            model.eval()
            restored.eval()
            full, replayed = predict(model, heldout_positions, arm), predict(restored, heldout_positions, arm)
            if full != replayed:
                raise ValueError("Final checkpoint cached-prefix replay differs")
            predictions[arm].update(replayed)
            ablated[arm].update(predict(restored, heldout_positions, arm, zero_images=True))
            record = {"schema_version": "rsna-ticket58-adaptation-run-v1", "status": "COMPLETE",
                "identity": identity, "counts": counts, "predictions": replayed,
                "zero_image_predictions": {uid: ablated[arm][uid] for uid in heldout},
                "training_study_uids": training + reports, "held_out_study_uids": heldout,
                "held_out_inputs_excluded_from_fitting": True, "held_out_target_entries_masked": True,
                "frozen_parameter_max_abs_change": frozen_error, "normalization_buffer_max_abs_change": buffer_error,
                "parameter_max_abs_changes": changes, "checkpoint_sha256": digest(checkpoint_path),
                "cached_checkpoint_replay_max_abs_error": 0.0, "native_final_replay_verified": False}
            write(run_root / "run.json", record)
            records.append({"fold": fold["fold"], "arm": arm, "path": str(run_root / "run.json"), "sha256": digest(run_root / "run.json")})
            matched.append((identity, counts))
            del optimizer, model, restored, initial, current, saved
        for key in ("initial_model_state_sha256", "initial_head_state_sha256", "initial_encoder_state_sha256",
                    "target_array_sha256", "target_mask_sha256"):
            if matched[0][0][key] != matched[1][0][key]:
                raise ValueError("Matched initial state or target identity differs")
        for key in ("gold_exposure_sequence_sha256", "weak_exposure_sequence_sha256", "exposures_by_source"):
            if matched[0][1][key] != matched[1][1][key]:
                raise ValueError("Matched ordered source exposures differ")
        del targets, mask, parent_checkpoint
    pairs = {"adaptation_minus_matched_head_continuation": (predictions[ARMS[0]], predictions[ARMS[1]]),
        "head_continuation_minus_warm_parent": (parent_predictions, predictions[ARMS[0]]),
        "adaptation_minus_warm_parent": (parent_predictions, predictions[ARMS[1]]),
        "head_continuation_minus_zero_image": (ablated[ARMS[0]], predictions[ARMS[0]]),
        "adaptation_minus_zero_image": (ablated[ARMS[1]], predictions[ARMS[1]])}
    comparisons = {name: compare_image_model_predictions(base, candidate, fold_manifest,
        bootstrap_resamples=plan["analysis"]["bootstrap_resamples"], bootstrap_seed=plan["analysis"]["seed"])
        for name, (base, candidate) in pairs.items()}
    def passes(name):
        comparison = comparisons[name]
        lower = comparison["paired_study_bootstrap"]["macro_auc"]["lower"]
        return lower is not None and lower > 0 and all(row["delta_macro_auc"] is not None
            and row["delta_macro_auc"] > 0 for row in comparison["delta"]["folds"])
    disposition = ("provisional_adaptation_review_pending_independent_native_replay"
        if passes(plan["analysis"]["primary"]) and passes("adaptation_minus_zero_image") else "inconclusive_no_promotion")
    write(root / "analysis.json", {"schema_version": "rsna-ticket58-report-layer4-analysis-v1", "status": "COMPLETE",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(), "job_id": os.environ["SLURM_JOB_ID"],
        "plan_sha256": checksum, "run_records": records, "comparisons": comparisons,
        "disposition": disposition, "native_final_replay_verified": False,
        "development_cohort_reused": True, "candidate_promotion": False, "competition_submission": False,
        "gpu_count": 0, "external_spend_usd": 0, "elapsed_seconds": time.monotonic() - started,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
    watchdog.cancel()
    print(json.dumps({"status": "COMPLETE", "disposition": disposition, "runs": len(records)}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        run(Path(sys.argv[2]), sys.argv[3])
