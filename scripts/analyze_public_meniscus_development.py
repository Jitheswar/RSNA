"""Analyze the single verified meniscus vote as reused development evidence."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from public_meniscus_overlay import ID, TARGET, TARGETS, apply_meniscus_overlay
from rsna_knee.folds import load_fold_manifest
from rsna_knee.image_model_comparison import compare_image_model_predictions


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    args = parser.parse_args()
    root = args.development_root.resolve()
    request = json.loads((root / "request.json").read_text())
    verification = json.loads((root / "validation.json").read_text())
    if (verification["status"] != "COMPLETE" or verification["request_sha256"] != sha(root / "request.json")
            or request["medial_meniscus_specialist_weight"] != 0.10 or request["weights_swept"]
            or request["model_fitting"] or request["competition_submission"]):
        raise ValueError("Only the independently verified fixed development pair is admitted")
    output = root / "output"
    receipt_path = output / "paired_development.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt["status"] != "COMPLETE" or receipt["study_count"] != 58
            or receipt["anchor_code_cells_unchanged"] != 23 or receipt["specialist_fallback"] != 0
            or not receipt["actual_cpu_mri_parity"] or not receipt["anchor_random_states_preserved"]
            or receipt["input_cache_sha256"] != request["native_cache_sha256"]
            or receipt["input_mask_sha256"] != request["native_mask_sha256"]):
        raise ValueError("Required actual inference evidence differs")
    filenames = {"development-anchor.csv", "development-overlay.csv", "development-transformer.csv",
                 "development-hybrid.csv", "public0033_bag_raw.csv", "public0033_cached_inference_receipt.json"}
    if set(receipt["output_sha256"]) != filenames:
        raise ValueError("Prediction output contract differs")
    for filename in filenames:
        if sha(output / filename) != receipt["output_sha256"][filename]:
            raise ValueError("Verified prediction output bytes changed")
    def frame(filename):
        return pd.read_csv(output / filename, dtype={ID: str}, float_precision="round_trip")
    anchor, overlay, transformer, hybrid, specialist = [frame(filename) for filename in
        ("development-anchor.csv", "development-overlay.csv", "development-transformer.csv",
         "development-hybrid.csv", "public0033_bag_raw.csv")]
    expected = apply_meniscus_overlay(anchor, transformer, hybrid, specialist)
    if not overlay.equals(expected):
        raise ValueError("Saved predictions differ from the one frozen vote")
    unchanged = [ID] + [target for target in TARGETS if target != TARGET]
    if not anchor[unchanged].equals(overlay[unchanged]):
        raise ValueError("Another target or study identity changed")
    ordered_ids = anchor[ID].tolist()
    if hashlib.sha256("".join(uid + "\n" for uid in ordered_ids).encode()).hexdigest() != request["development_study_uid_sha256"]:
        raise ValueError("Frozen development cohort or order changed")
    manifest = load_fold_manifest(args.fold_manifest)
    if manifest.superseded or manifest.manifest_checksum != "8460234d0fcbfd2eb9ca3a3e4165929f50a86acac37d2cd03be1164c2f287fb8":
        raise ValueError("Corrected development subgroup identity differs")
    def predictions(value):
        return dict(zip(ordered_ids, value[TARGETS].to_numpy().tolist(), strict=True))
    comparison = compare_image_model_predictions(predictions(anchor), predictions(overlay), manifest,
        bootstrap_resamples=request["analysis"]["paired_study_bootstrap_resamples"],
        bootstrap_seed=request["analysis"]["seed"])
    # This helper also serves OOF experiments. Here no fold-specific model is
    # retrained: its fold summaries are fixed development subgroups only.
    comparison["schema_version"] = "rsna-fixed-meniscus-development-comparison-v1"
    limitation = "Reused Gold58 development cohort; public parent selection used these labels. No fold-specific retraining, clean OOF, untouched generalization estimate or competition-score claim."
    comparison["interpretation_limit"] = limitation
    result = {"schema_version": "rsna-public-meniscus-development-analysis-v1",
              "observed_at_utc": datetime.now(timezone.utc).isoformat(), "status": "COMPLETE",
              "request_sha256": sha(root / "request.json"), "validation_sha256": sha(root / "validation.json"),
              "prediction_receipt_sha256": sha(receipt_path), "fold_manifest_sha256": sha(args.fold_manifest),
              "comparison": comparison, "eleven_other_targets_identical": True,
              "fixed_vote_recomputed_exactly": True, "study_count": 58,
              "visible_elapsed_seconds": receipt["elapsed_seconds"],
              "visible_seconds_per_study": receipt["elapsed_seconds"] / 58,
              "hidden_scale_runtime_verified": False, "fold_summaries_are_subgroups_not_retraining": True,
              "clean_oof": False, "candidate_promotion": False,
              "competition_submission": False, "final_selection": False, "limitation": limitation}
    with (root / "development-analysis.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    medial = next(row for row in comparison["per_target"] if row["target"] == TARGET)
    print(json.dumps({"macro_auc_delta": comparison["delta"]["pooled_macro_auc"],
                      "medial_meniscus": medial, "interpretation_limit": limitation}, indent=2))


if __name__ == "__main__":
    main()
