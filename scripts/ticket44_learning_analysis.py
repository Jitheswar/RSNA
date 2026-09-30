"""Predeclared long-versus-short learning comparison and image controls."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rsna_knee import ticket36 as t36, ticket44 as t44
from rsna_knee.evaluation import evaluate_oof_predictions, predict_training_examples
from rsna_knee.folds import load_fold_manifest
from rsna_knee.image_model_comparison import compare_image_model_predictions
from rsna_knee.model import build_study_model
from rsna_knee.real_inference import load_integration_manifest
from rsna_knee.training import load_training_checkpoint


def main() -> None:
    if os.environ.get("SLURM_JOB_PARTITION") != "cluster1-cpu":
        raise RuntimeError("CPU Slurm allocation required")
    torch.set_num_threads(16)
    plan = t44._load_plan()
    t44._approval(plan)
    completed = t44.RUN_ROOT / "analysis-recovery-v1/comparison.json"
    if not completed.is_file():
        raise RuntimeError("strict matched comparison and checkpoint replay must pass first")
    parent = t36._load_plan()
    feature_manifest = t36._load_feature_manifest(parent)
    manifest = load_fold_manifest(t36.FOLD_MANIFEST_PATH)
    integration = load_integration_manifest(t36.INTEGRATION_PATH)
    config = replace(t36._make_training_config(integration), successful_optimizer_steps=1500)
    rows = {r["study_instance_uid"]: r for r in feature_manifest["features"]}
    features = {}
    for member in manifest.gold_memberships:
        uid = member.study_instance_uid
        row = rows[uid]
        feature = t36._load_feature_file(t36._feature_path(uid), study_uid=uid, batch_index=row["batch_index"], feature_identity_fingerprint=parent["feature_identity_fingerprint"])
        if feature["file_sha256"] != row["feature_file_sha256"]:
            raise RuntimeError("sealed feature file changed")
        features[uid] = feature
    evidence = {"schema_version": "rsna-ticket44-learning-controls-v1", "analysis_job_id": os.environ["SLURM_JOB_ID"], "plan_sha256": plan["plan_sha256"], "matched_comparison_sha256": t36._sha256_file(completed), "bootstrap_seed": 20260929, "bootstrap_resamples": 2000, "permutation_resamples": 10000, "arms": {}, "scientific_training": False, "external_spend_usd": 0, "development_cv_not_untouched_test": True}
    for arm in plan["ordered_arms"]:
        short, long, ablated = {}, {}, {}
        original_hashes = []
        for fold in range(3):
            short_path = t36.RUN_ROOT / arm / f"fold-{fold}/run.json"
            long_path = t44.RUN_ROOT / arm / f"fold-{fold}/run.json"
            short_run, long_run = t44._read(short_path), t44._read(long_path)
            expected = {r.study_instance_uid for r in manifest.gold_memberships if r.fold == fold}
            for run, expected_steps in ((short_run, 100), (long_run, 1500)):
                if run["status"] != "completed_fixed_budget_training" or run["fold"] != fold or set(run["predictions"]) != expected or run["counts"]["successful_optimizer_steps"] != expected_steps:
                    raise RuntimeError("short/long run cohort or training budget changed")
            for name in ("model", "packing", "preprocessing", "seeds", "loss", "augmentation", "selection_policy"):
                if short_run["config"].get(name) != long_run["config"].get(name):
                    raise RuntimeError(f"short/long scientific identity differs: {name}")
            short.update(short_run["predictions"])
            long.update(long_run["predictions"])
            original_hashes.append({"fold": fold, "short_run_sha256": t36._sha256_file(short_path), "long_run_sha256": t36._sha256_file(long_path)})
            training = tuple(r.study_instance_uid for r in manifest.gold_memberships if r.fold != fold)
            heldout = tuple(r for r in manifest.gold_memberships if r.fold == fold)
            examples = tuple(t36._reference_example(r.study_instance_uid, features[r.study_instance_uid], r.reference_labels, fold=fold, fold_manifest_checksum=manifest.manifest_checksum, training_uids=training, heldout_uids=manifest.held_out_study_uids(fold)) for r in heldout)
            model = build_study_model(config.model_config())
            load_training_checkpoint(long_path.parent / "checkpoints/final.pt", model=model, expected_config=config, expected_fold=fold)
            zero_examples = tuple(replace(item, features=np.zeros_like(item.features)) for item in examples)
            ablated.update({uid: tuple(float(x) for x in values) for uid, values in predict_training_examples(model, zero_examples, batch_size=2).items()})
        full = evaluate_oof_predictions(long, manifest, bootstrap_resamples=2000, bootstrap_seed=20260929, permutation_resamples=10000, permutation_seed=20260929)
        evidence["arms"][arm] = {"long_oof": full.to_public_dict(), "long_minus_short": compare_image_model_predictions(short, long, manifest, bootstrap_resamples=2000, bootstrap_seed=20260929), "long_minus_zero_features": compare_image_model_predictions(ablated, long, manifest, bootstrap_resamples=2000, bootstrap_seed=20260929), "run_hashes": original_hashes}
    constant = evaluate_oof_predictions({r.study_instance_uid: (0.5,) * 12 for r in manifest.gold_memberships}, manifest, include_diagnostics=False)
    if constant.pooled_macro_auc != 0.5:
        raise RuntimeError("constant-score control failed")
    evidence["constant_score_control"] = constant.to_public_dict()
    output = t44.RUN_ROOT / "learning-controls-v1"
    t36._write_private_json(output / "evidence.json", evidence)
    lines = ["# Ticket 44 — learning-budget comparison and controls", "", "These analyses were predeclared in Ticket 44 and use the same repeatedly reused 58-study development cohort. No fitting or tuning occurred.", "", "| Arm | Short AUC | Long AUC | Long minus short | Paired 95% interval | Full minus zero-feature AUC | Paired 95% interval |", "| --- | ---: | ---: | ---: | --- | ---: | --- |"]
    for arm, row in evidence["arms"].items():
        paired = row["long_minus_short"]
        ablation = row["long_minus_zero_features"]
        interval = paired["paired_study_bootstrap"]["macro_auc"]
        ai = ablation["paired_study_bootstrap"]["macro_auc"]
        lines.append(f"| {arm} | {paired['baseline']['pooled_macro_auc']:.6f} | {paired['candidate']['pooled_macro_auc']:.6f} | {paired['delta']['pooled_macro_auc']:+.6f} | {interval['lower']:+.6f} to {interval['upper']:+.6f} | {ablation['delta']['pooled_macro_auc']:+.6f} | {ai['lower']:+.6f} to {ai['upper']:+.6f} |")
    lines.extend(["", "Full evidence includes within-fold versus cross-fold AUC, training-prevalence control, 10,000 within-fold permutations per arm, per-target support and paired intervals, all undefined outcomes, and source hashes. The image ablation zeros features while retaining the fitted head, masks and descriptors. Constant-score AUC is 0.5.", ""])
    t36._write_private_text(output / "report.md", "\n".join(lines))
    print(json.dumps({"status": "completed", "report": str(output / "report.md")}))


if __name__ == "__main__":
    main()
