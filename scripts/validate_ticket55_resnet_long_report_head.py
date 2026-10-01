"""Independently replay and recompute the fixed Ticket 55 experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rsna_knee.folds import load_fold_manifest
from rsna_knee.image_model_comparison import compare_image_model_predictions
from rsna_knee.training import _array_sha256, _batch_indices


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sequence_digest(sequence: list[str]) -> str:
    return hashlib.sha256(json.dumps(sequence, separators=(",", ":"),
                                    ensure_ascii=True).encode()).hexdigest()


def predict(state: dict, pooled: torch.Tensor, indices: list[int], uids: list[str]) -> dict:
    head = nn.Sequential(nn.Linear(2076, 32), nn.ReLU(inplace=True), nn.Linear(32, 12))
    head.load_state_dict(state, strict=True)
    head.eval()
    result = {}
    with torch.inference_mode():
        for start in range(0, len(indices), 2):
            batch = indices[start:start + 2]
            scores = torch.sigmoid(head(pooled[batch])).numpy()
            require(bool(np.isfinite(scores).all()), "Nonfinite independently replayed prediction")
            result.update({uids[index]: value.tolist()
                           for index, value in zip(batch, scores, strict=True)})
    return result


def maximum_error(actual: dict, expected: dict) -> float:
    require(set(actual) == set(expected), "Independent prediction membership differs")
    return max(float(np.max(np.abs(np.asarray(actual[uid]) - np.asarray(expected[uid]))))
               for uid in actual)


def validate(root: Path, expected_plan_sha256: str) -> dict:
    plan = read(root / "plan.json")
    require(digest(root / "plan.json") == expected_plan_sha256, "Requested plan bytes changed")
    require(Path(plan["run_root"]) == root, "Requested run root differs")
    require(plan["optimization"]["successful_steps"] == 7500
            and plan["analysis"]["fixed_convnext_weight"] == 0.5
            and plan["prefix_maximum_absolute_error"] == 1e-6,
            "Fixed budget, blend or replay tolerance differs")
    for row in plan["frozen_files"] + plan["feature_files"]:
        require(digest(Path(row["path"])) == row["sha256"], "Frozen input bytes changed")
    require(all(importlib.metadata.version(name) == version
                for name, version in plan["runtime_versions"].items()), "Frozen replay runtime differs")
    accepted = read(root / "accepted-launch.json")
    terminal = read(root / "terminal-observation.json")
    controller = (root / "terminal-controller.txt").read_text()
    fields = dict(token.split("=", 1) for token in controller.split() if "=" in token)
    require(accepted["plan_sha256"] == expected_plan_sha256
            and terminal["plan_sha256"] == expected_plan_sha256
            and accepted["job_id"] == terminal["job_id"] == int(fields["JobId"])
            and terminal["controller"] == controller
            and terminal["state"] == fields["JobState"] == "COMPLETED"
            and terminal["exit_code"] == fields["ExitCode"] == "0:0",
            "Matching successful terminal controller evidence is required")
    expected_allocation = {"Partition": "cluster1-cpu", "NumNodes": "1", "NumCPUs": "16",
                           "NumTasks": "1", "MinMemoryNode": "64G", "TimeLimit": "00:30:00",
                           "OverSubscribe": "OK", "Requeue": "0", "Restarts": "0"}
    require(all(fields.get(key) == value for key, value in expected_allocation.items())
            and "gpu" not in fields.get("TRES", "").lower(), "Actual allocation differs")
    seconds = (datetime.fromisoformat(fields["EndTime"])
               - datetime.fromisoformat(fields["StartTime"])).total_seconds()
    parent = read(Path(plan["parent_plan"]))
    project = root.parents[2]
    fold_paths = [Path(row["path"]) for row in plan["frozen_files"]
                  if Path(row["path"]).name == "fold-manifest.json"]
    require(len(fold_paths) == 1, "Exactly one frozen fold manifest is required")
    manifest = load_fold_manifest(fold_paths[0])
    require(manifest.manifest_checksum == plan["fold_manifest_checksum"] and not manifest.superseded,
            "Corrected development fold identity differs")
    gold = {member.study_instance_uid: member for member in manifest.gold_memberships}
    pool_evidence = read(root / "pooled-cohort-evidence.json")
    require(pool_evidence["pooled_file_sha256"] == digest(root / "pooled-cohort.npz")
            and pool_evidence["source_entries_verified"] == 4228, "Pooled bank integrity differs")
    with np.load(root / "pooled-cohort.npz", allow_pickle=False) as bank:
        values = bank["pooled"].copy()
        uids = bank["study_uids"].tolist()
    require(values.shape == (4228, 2076) and bool(np.isfinite(values).all())
            and uids == sorted(set(uids)) and set(gold) <= set(uids)
            and _array_sha256(values) == pool_evidence["pooled_array_sha256"],
            "Pooled array identity, dimensions or study mapping differs")
    old_root = project / "artifacts/private/ticket44-full-weak-exposure"
    convnext_root = project / "artifacts/private/ticket54-convnext-long-report-head"
    original_pool = read(root.with_name("ticket55-resnet-long-report-head") / "pooled-cohort-evidence.json")
    require(pool_evidence == original_pool, "Recovery pooled bank differs from the original admitted bank")
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    pooled = torch.from_numpy(values)
    index = {uid: position for position, uid in enumerate(uids)}
    zero = pooled.clone().reshape(-1, 4, 519)
    zero[:, :, :512] = 0
    zero = zero.reshape(-1, 2076)
    arms = ("reference-only", "report-supervision")
    final, short, ablated, convnext = [{arm: {} for arm in arms} for _ in range(4)]
    audit, records, errors = [], [], []
    optimization = plan["optimization"]
    for fold_plan in parent["folds"]:
        fold = fold_plan["fold"]
        training = fold_plan["gold_training_study_uids"]
        heldout = fold_plan["gold_heldout_study_uids"]
        weak = fold_plan["weak_training_study_uids"]
        require(set(heldout) == {uid for uid, member in gold.items() if member.fold == fold}
                and set(training) == set(gold) - set(heldout)
                and not set(training + weak) & set(manifest.held_out_study_uids(fold)),
                "Training or held-out fold membership differs")
        heldout_indices = [index[uid] for uid in heldout]
        paired_counts = []
        for arm in arms:
            run_path = root / arm / f"fold-{fold}/run.json"
            run = read(run_path)
            checkpoint_path = run_path.parent / "checkpoints/final.pt"
            require(digest(checkpoint_path) == run["checkpoint_sha256"], "Final checkpoint hash differs")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            expected_identity = {"plan_sha256": expected_plan_sha256, "fold": fold, "arm": arm,
                                 "head": plan["head"], "fold_manifest_checksum": manifest.manifest_checksum,
                                 "pooled_cohort_sha256": digest(root / "pooled-cohort.npz")}
            require(run["status"] == "COMPLETE" and run["identity"] == checkpoint["identity"] == expected_identity
                    and checkpoint["checkpoint_kind"] == "final_selected"
                    and checkpoint["head_id"] == plan["head"]["head_id"]
                    and checkpoint["counts"] == run["counts"]
                    and run["training_study_uids"] == training + (weak if arm == "report-supervision" else [])
                    and run["held_out_study_uids"] == heldout
                    and run["held_out_inputs_excluded_from_fitting"]
                    and run["held_out_target_entries_masked"], "Run/checkpoint contract differs")
            counts = run["counts"]
            require(counts["successful_optimizer_steps"] == counts["attempted_batches"] == 7500
                    and counts["skipped_batches"] == 0 and len(counts["loss_history"]) == 7500
                    and bool(np.isfinite(counts["loss_history"]).all()), "Fixed training budget differs")
            for kind, members, offset in (("reference", training, 0),
                                          ("weak", weak if arm == "report-supervision" else [],
                                           optimization["weak_seed_offset"])):
                sequence = [members[position] for step in range(7500)
                            for position in _batch_indices(len(members), optimization["batch_size"], step,
                                                           optimization["batch_order_seed"] + offset)] if members else []
                key = "gold_exposure_sequence_sha256" if kind == "reference" else "weak_exposure_sequence_sha256"
                require(counts[key] == sequence_digest(sequence)
                        and counts["exposures_by_source"][kind]["sample_exposures"] == len(sequence),
                        "Ordered training exposures differ")
                if kind == "weak":
                    require(counts["distinct_weak_studies_exposed"] == len(set(sequence))
                            and counts["weak_training_pool_count"] == len(members), "Report coverage differs")
            prefix = run["prefix_replay"]
            require(0 <= prefix["prediction_max_abs_error"] <= plan["prefix_maximum_absolute_error"]
                    and 0 <= prefix["state_max_abs_error"] <= plan["prefix_maximum_absolute_error"]
                    and prefix["initial_full_model_state_matches"] and prefix["ordered_exposures_match"],
                    "Original prefix gate was not satisfied")
            actual = predict(checkpoint["head_state_dict"], pooled, heldout_indices, uids)
            error = maximum_error(actual, run["predictions"])
            require(error <= plan["prefix_maximum_absolute_error"], "Independent final replay differs")
            require(not set(final[arm]) & set(actual), "Held-out predictions overlap folds")
            final[arm].update(actual)
            ablated[arm].update(predict(checkpoint["head_state_dict"], zero, heldout_indices, uids))
            short[arm].update(read(old_root / arm / f"fold-{fold}/run.json")["predictions"])
            convnext[arm].update(read(convnext_root / arm / f"fold-{fold}/run.json")["predictions"])
            records.append({"fold": fold, "arm": arm, "path": str(run_path), "sha256": digest(run_path)})
            audit.append({"fold": fold, "arm": arm, "checkpoint_sha256": digest(checkpoint_path),
                          "independent_replay_max_abs_error": error, "independent_replay_threads": 16})
            errors.append(error)
            paired_counts.append(counts)
        require(all(paired_counts[0][key] == paired_counts[1][key]
                    for key in ("initial_head_state_sha256", "gold_exposure_sequence_sha256")),
                "Matched arm initialization or reference exposure differs")
    blend = {uid: ((np.asarray(final["report-supervision"][uid])
                    + convnext["report-supervision"][uid]) / 2).tolist() for uid in gold}
    pairs = {"gold_long_minus_short": (short["reference-only"], final["reference-only"]),
             "report_long_minus_short": (short["report-supervision"], final["report-supervision"]),
             "report_minus_gold": (final["reference-only"], final["report-supervision"]),
             "gold_minus_zero_image": (ablated["reference-only"], final["reference-only"]),
             "report_minus_zero_image": (ablated["report-supervision"], final["report-supervision"]),
             "convnext_long_minus_resnet_long": (final["report-supervision"], convnext["report-supervision"]),
             "fixed_half_blend_minus_resnet_long": (final["report-supervision"], blend)}
    comparisons = {name: compare_image_model_predictions(base, candidate, manifest,
        bootstrap_resamples=plan["analysis"]["bootstrap_resamples"], bootstrap_seed=plan["analysis"]["seed"])
        for name, (base, candidate) in pairs.items()}
    def passes(name: str) -> bool:
        value = comparisons[name]
        lower = value["paired_study_bootstrap"]["macro_auc"]["lower"]
        return lower is not None and lower > 0 and all(row["delta_macro_auc"] is not None
            and row["delta_macro_auc"] > 0 for row in value["delta"]["folds"])
    disposition = ("provisional_fixed_blend_review" if passes("fixed_half_blend_minus_resnet_long")
                   and passes("report_minus_zero_image") else "inconclusive_no_promotion")
    analysis = read(root / "analysis.json")
    require(analysis["status"] == "COMPLETE" and analysis["plan_sha256"] == expected_plan_sha256
            and analysis["job_id"] == str(accepted["job_id"])
            and analysis["run_records"] == records and analysis["comparisons"] == comparisons
            and analysis["fixed_half_blend_predictions"] == blend
            and analysis["disposition"] == disposition and analysis["candidate_promotion"] is False
            and analysis["development_cohort_reused"] is True
            and analysis["gpu_count"] == 0 and analysis["external_spend_usd"] == 0,
            "Analysis, comparisons or declared disposition differ")
    return {"schema_version": "rsna-ticket55-independent-result-validation-v1", "status": "COMPLETE",
            "observed_at_utc": datetime.now(timezone.utc).isoformat(), "job_id": accepted["job_id"],
            "plan_sha256": expected_plan_sha256, "analysis_sha256": digest(root / "analysis.json"),
            "terminal_controller_sha256": digest(root / "terminal-controller.txt"),
            "allocated_cpu_node_seconds": seconds, "frozen_files_verified": len(plan["frozen_files"]),
            "feature_files_verified": len(plan["feature_files"]), "runs_verified": len(audit),
            "final_predictions_verified": sum(len(arm) for arm in final.values()),
            "independent_checkpoint_replays": audit, "maximum_prediction_error": max(errors),
            "comparisons_recomputed": list(comparisons), "original_prefix_gates_verified": True,
            "original_pooled_bank_identical": True, "disposition": disposition,
            "candidate_promotion": False, "development_cohort_reused": True,
            "gpu_count": 0, "external_spend_usd": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Immutable validation output already exists")
    result = validate(args.root.resolve(), args.expected_plan_sha256)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "runs_verified": result["runs_verified"],
                      "final_predictions_verified": result["final_predictions_verified"],
                      "disposition": result["disposition"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
