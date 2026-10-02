"""Replay a retained real-MRI diagnostic checkpoint in a fresh CPU process."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from rsna_knee import ticket36 as t36
from rsna_knee.folds import load_fold_manifest
from rsna_knee.model import restore_study_model
from rsna_knee.resnet18 import RESNET18_FULL_FINETUNE_MODEL_ID
from rsna_knee.ticket34 import admit_ticket30_fold_supervision
from rsna_knee.training import _array_sha256


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    root = Path(sys.argv[1]).resolve()
    plan = json.loads((root / "plan.json").read_text())
    result = json.loads((root / "pilot-result.json").read_text())
    assert result["status"] == "COMPLETE" and result["plan_sha256"] == digest(root / "plan.json")
    assert result["gpu_count"] == 0 and result["external_spend_usd"] == 0
    assert not result["candidate_promotion"] and not result["competition_submission"]
    for item in plan["frozen_files"]:
        assert digest(item["path"]) == item["sha256"], item["path"]
    for item in result["outputs"]:
        assert digest(root / item["file"]) == item["sha256"], item["file"]
    manifest = load_fold_manifest(t36.FOLD_MANIFEST_PATH)
    assert manifest.manifest_checksum == plan["fold_manifest_checksum"]
    selected = plan["selected_training_report_uids"]
    assert len(selected) == len(set(selected)) == 4
    assert not set(selected) & set(manifest.held_out_study_uids(0))
    assert not set(selected) & {row.study_instance_uid for row in manifest.gold_memberships}
    admission = admit_ticket30_fold_supervision(manifest, 0, t36.TICKET30_ARTIFACT,
        expected_artifact_sha256=t36.TICKET30_AUDITED_ARTIFACT_SHA256)
    targets = np.array([[float(v) if v is not None else 0.0 for v in admission.by_study[uid].targets]
                        for uid in selected], dtype=np.float32)
    mask = np.array([[v is not None for v in admission.by_study[uid].targets]
                     for uid in selected], dtype=np.bool_)
    with np.load(root / "diagnostic-batch.npz", allow_pickle=False) as batch:
        assert _array_sha256(batch["images"]) == plan["expected_input_array_sha256"] == result["input_array_sha256"]
        assert np.array_equal(batch["targets"], targets)
        assert np.array_equal(batch["supervision_mask"], mask) and mask.any(axis=1).all()
        tensors = [torch.from_numpy(batch[key].copy()) for key in
                   ("images", "slice_mask", "series_mask", "series_descriptors")]
    assert int((tensors[1] & tensors[2][:, :, None]).sum()) == result["valid_slices"] == 232
    assert tensors[1].numel() == result["padded_slices"] == 512
    assert result["normalization_buffer_max_abs_change"] == 0
    assert set(result["parameter_max_abs_changes"]) == {"conv1", "layer1", "layer2", "layer3", "layer4", "head"}
    assert all(value > 0 for value in result["parameter_max_abs_changes"].values())
    assert len(result["timed_update_seconds"]) == 3 and all(t > 0 for t in result["timed_update_seconds"])
    assert np.isfinite(result["training_losses"]).all()
    assert result["native_initial_logit_max_abs_error"] <= plan["profiling"]["parity_tolerance"]
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available()
    checkpoint = torch.load(root / "diagnostic-checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["model_config"] == plan["model"]
    model = restore_study_model(checkpoint["model_config"], checkpoint["state_dict"]).eval()
    assert model.config.model_id == RESNET18_FULL_FINETUNE_MODEL_ID
    with torch.no_grad():
        logits = model(*tensors).numpy()
    expected = np.load(root / "diagnostic-logits.npy", allow_pickle=False)
    assert np.isfinite(logits).all() and np.array_equal(logits, expected)
    for scenario in result["projected_training_scenarios"]:
        seconds = 2 * float(np.median(result["timed_update_seconds"])) * scenario["updates_per_fold"]
        assert scenario["training_seconds_per_fold"] == seconds
        assert scenario["three_folds_plus_final_fit_training_seconds"] == seconds * 4
    verification = {"status": "COMPLETE", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": digest(root / "plan.json"), "result_sha256": digest(root / "pilot-result.json"),
        "actual_job_id": result["job_id"], "independent_report_targets_and_masks": True,
        "prior_admitted_mri_array_identity": True, "held_out_and_gold_excluded": True,
        "strict_full_backbone_checkpoint_reload": True, "prediction_max_abs_error": 0.0,
        "checkpoint_sha256": digest(root / "diagnostic-checkpoint.pt"), "full_training_selected": False,
        "predictive_gain_established": False, "actual_terminal_still_required": True}
    with (root / "independent-validation.json").open("x") as handle:
        json.dump(verification, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
