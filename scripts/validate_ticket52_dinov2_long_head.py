"""Read-only independent integrity/replay audit of the completed CPU heads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from rsna_knee.feature_pooling import ANATOMICAL_POOLING_ID, pool_frozen_features
from rsna_knee.folds import load_fold_manifest
from rsna_knee.training import _batch_indices


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    output = root / ("job-" + sys.argv[2])
    plan = json.loads((root / "plan.json").read_text())
    plan_sha = digest(root / "plan.json")
    if any(digest(Path(r["path"])) != r["sha256"] for r in plan["frozen_files"]):
        raise RuntimeError("frozen inputs changed")
    evidence = json.loads((output / "evidence.json").read_text())
    manifest = load_fold_manifest(plan["fold_manifest_path"])
    members = {m.study_instance_uid: m for m in manifest.gold_memberships}
    uids = sorted(members)
    rows = {r["study_instance_uid"]: r for r in json.loads(Path(plan["reference_feature_manifest"]).read_text())["features"]}
    torch.set_num_threads(1)
    vectors = []
    for uid in uids:
        features = np.load(Path(plan["feature_cache"]) / "entries" / (hashlib.sha256(uid.encode()).hexdigest() + ".npy"), allow_pickle=False)
        mask_path = Path(rows[uid]["feature_file"])
        if not mask_path.is_absolute():
            mask_path = Path(plan["reference_feature_manifest"]).parent / mask_path
        with np.load(mask_path, allow_pickle=False) as masks, torch.no_grad():
            vectors.append(pool_frozen_features(torch.from_numpy(features)[None],
                torch.from_numpy(masks["slice_mask"].astype(bool))[None],
                torch.from_numpy(masks["series_mask"].astype(bool))[None],
                torch.from_numpy(masks["series_descriptors"])[None], feature_width=384,
                pooling_id=ANATOMICAL_POOLING_ID).squeeze(0))
    pooled = torch.stack(vectors)
    all_predictions, maximum_error, checkpoints = {}, 0.0, []
    for fold in range(3):
        path = output / f"fold-{fold}/final-head.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        run = json.loads((path.parent / "run.json").read_text())
        training = [i for i, uid in enumerate(uids) if members[uid].fold != fold]
        heldout = [i for i, uid in enumerate(uids) if members[uid].fold == fold]
        identity = checkpoint["identity"]
        if (identity["plan_sha256"] != plan_sha or identity["fold"] != fold
                or identity["training_study_uids"] != [uids[i] for i in training]
                or identity["held_out_study_uids"] != [uids[i] for i in heldout]
                or checkpoint["successful_steps"] != 1500 or checkpoint["checkpoint_selection"] != "final_step"
                or run["checkpoint_sha256"] != digest(path)
                or set(run["predictions"]) != {uids[i] for i in heldout}):
            raise RuntimeError("final head identity or fold coverage differs")
        sequence = [uids[training[i]] for step in range(1500)
                    for i in _batch_indices(len(training), 2, step, 20260912)]
        sequence_sha = hashlib.sha256(json.dumps(sequence, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
        if (sequence_sha != run["counts"]["gold_exposure_sequence_sha256"]
                or run["counts"]["gold_sample_exposures"] != 3000
                or run["counts"]["supervised_entry_exposures"] != 36000
                or len(run["loss_history"]) != 1500 or not np.isfinite(run["loss_history"]).all()
                or run["short_source_replay"]["maximum_absolute_score_error"] > 1e-6
                or not run["short_source_replay"]["ordered_gold_exposure_digest_matches"]):
            raise RuntimeError("successful steps, reference exposures or source replay differ")
        head = nn.Sequential(nn.Linear(1564, 32), nn.ReLU(inplace=True), nn.Linear(32, 12))
        head.load_state_dict(checkpoint["head_state_dict"], strict=True)
        head.eval()
        with torch.no_grad():
            for start in range(0, len(heldout), 2):
                indices = heldout[start:start + 2]
                values = torch.sigmoid(head(pooled[indices])).numpy()
                for i, value in zip(indices, values, strict=True):
                    uid = uids[i]
                    maximum_error = max(maximum_error, float(np.max(np.abs(value - np.asarray(run["predictions"][uid])))))
                    if uid in all_predictions or not np.isfinite(value).all():
                        raise RuntimeError("OOF coverage or finite scores invalid")
                    all_predictions[uid] = value.tolist()
        checkpoints.append({"fold": fold, "checkpoint_sha256": digest(path), "run_sha256": digest(path.parent / "run.json")})
    primary = evidence["primary_dino1500_minus_resnet1500"]
    lower = primary["paired_study_bootstrap"]["macro_auc"]["lower"]
    qualifies = lower is not None and lower > 0 and all(r["delta_macro_auc"] is not None and r["delta_macro_auc"] > 0 for r in primary["delta"]["folds"])
    expected_disposition = "eligible_for_followup_no_promotion" if qualifies else "inconclusive_no_promotion"
    if (maximum_error > 1e-6 or set(all_predictions) != set(members)
            or evidence["plan_sha256"] != plan_sha or evidence["disposition"] != expected_disposition
            or evidence["candidate_promotion"] is not False or evidence["constant_control"]["macro_auc"] != 0.5):
        raise RuntimeError("final replay, complete cohort or predeclared decision differs")
    terminal = json.loads((root / ("controller-" + sys.argv[2]) / "terminal-observation.json").read_text())
    allocation = (root / ("controller-" + sys.argv[2]) / "terminal-controller.txt").read_text()
    if terminal["state"] != "COMPLETED" or "ExitCode=0:0" not in allocation or "TRES=cpu=16,node=1,billing=16" not in allocation:
        raise RuntimeError("successful zero-GPU allocation proof differs")
    result = {"schema_version": "rsna-ticket52-independent-validation-v1", "plan_sha256": plan_sha,
              "job_id": sys.argv[2], "frozen_files_verified": len(plan["frozen_files"]),
              "oof_studies": len(all_predictions), "checkpoint_replay_maximum_absolute_error": maximum_error,
              "checkpoints": checkpoints, "evidence_sha256": digest(output / "evidence.json"),
              "terminal_allocation_sha256": digest(root / ("controller-" + sys.argv[2]) / "terminal-controller.txt"),
              "disposition": expected_disposition, "validation_status": "passed"}
    with (root / "independent-validation.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "passed", "studies": len(all_predictions), "maximum_replay_error": maximum_error}))


if __name__ == "__main__":
    main()
