"""Recover Ticket 44 analysis from immutable training outputs; never retrain."""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

import torch

from rsna_knee import ticket36 as t36, ticket44 as t44
from rsna_knee.evaluation import predict_training_examples
from rsna_knee.folds import FOLD_IDS, load_fold_manifest
from rsna_knee.model import build_study_model
from rsna_knee.oof import load_oof_run_record
from rsna_knee.real_inference import load_integration_manifest
from rsna_knee.supervision_comparison import compare_supervision_oof, render_supervision_comparison_markdown
from rsna_knee.ticket34 import TICKET34_ADDITIVE_DESIGN, TICKET34_WEAK_BATCH_SEED_OFFSET, TICKET34_WEAK_SAMPLING_POLICY, admit_ticket30_fold_supervision
from rsna_knee.training import _batch_indices, _exposure_sequence_sha256, load_training_checkpoint


def analysis_view(source: Path, destination: Path, *, snapshot_sha256: str, job_id: str) -> dict:
    """Make an explicitly derived record with independently verified source identity.

    The caller must verify the frozen snapshot and its launcher before calling.
    Original configuration, environment fingerprints and checkpoints are preserved.
    A snapshot identity is not represented as a recovered Git commit.
    """
    if destination.exists():
        raise ValueError("analysis view destination already exists")
    configuration = json.loads((source / "configuration.json").read_text())
    environment = configuration["environment"]
    if environment.get("code_revision") is not None or environment.get("slurm_job_id") != job_id:
        raise ValueError("source record is not the diagnosed missing-revision allocation")
    if len(snapshot_sha256) != 64 or any(c not in "0123456789abcdef" for c in snapshot_sha256):
        raise ValueError("invalid source snapshot digest")
    destination.mkdir(parents=True, mode=0o700)
    hashes = {}
    for relative in ("run.json", "configuration.json", "metrics.json", "checkpoints/final.pt", "checkpoints/latest.pt"):
        original = source / relative
        if original.is_symlink() or not original.is_file():
            raise ValueError(f"missing or unsafe source artifact: {original}")
        hashes[relative] = t36._sha256_file(original)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative != "configuration.json":
            shutil.copyfile(original, target)
    original_environment = dict(environment)
    environment["code_revision"] = f"verified-source-snapshot-sha256:{snapshot_sha256}"
    environment["environment_fingerprint"] = t36._json_sha256({k: v for k, v in environment.items() if k != "environment_fingerprint"})
    configuration["analysis_provenance_supplement"] = {
        "schema_version": "rsna-ticket44-derived-analysis-view-v1",
        "source_run_directory": str(source.resolve()),
        "original_artifact_sha256": hashes,
        "original_environment": original_environment,
        "verified_training_source_snapshot_sha256": snapshot_sha256,
        "git_revision_recovered": False,
        "training_outputs_modified": False,
    }
    t36._write_private_json(destination / "configuration.json", configuration)
    if any(t36._sha256_file(source / relative) != digest for relative, digest in hashes.items()):
        raise ValueError("original training artifacts changed during analysis-view construction")
    return configuration["analysis_provenance_supplement"]


def main() -> int:
    if os.environ.get("SLURM_JOB_PARTITION") != "cluster1-cpu" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("analysis requires a Slurm CPU allocation")
    torch.set_num_threads(16)
    started = time.perf_counter()
    plan = t44._load_plan()
    approval = t44._approval(plan)
    snapshot = Path(plan["source_snapshot"])
    if t44._source_manifest(snapshot) != plan["source_manifest"]:
        raise RuntimeError("training source snapshot differs from frozen plan")
    launcher = (snapshot / "slurm/run_ticket44_full_weak_exposure.sbatch").read_text()
    if 'PYTHONPATH="$snapshot/src' not in launcher:
        raise RuntimeError("frozen launcher does not bind training to the snapshot")
    allocation = t44._read(t44.RUN_ROOT / "allocations/7830-restart-0.json")["record"]
    if "JobId=7830 " not in allocation or "Command=" + str(t44.SOURCE_SCRIPT) not in allocation:
        raise RuntimeError("original allocation is not bound to the expected training launcher")
    output = t44.RUN_ROOT / "analysis-recovery-v1"
    if output.exists():
        raise RuntimeError("analysis recovery output already exists; preserve prior attempts")
    output.mkdir(mode=0o700)
    parent = t36._load_plan()
    feature_manifest = t36._load_feature_manifest(parent)
    if feature_manifest["feature_manifest_sha256"] != plan["feature_manifest_sha256"]:
        raise RuntimeError("sealed feature manifest changed")
    manifest = load_fold_manifest(t36.FOLD_MANIFEST_PATH)
    integration = load_integration_manifest(t36.INTEGRATION_PATH)
    config = replace(t36._make_training_config(integration), successful_optimizer_steps=1500)
    if config.to_dict() != plan["training_configuration"]:
        raise RuntimeError("frozen scientific configuration changed")
    gold = {r.study_instance_uid: r for r in manifest.gold_memberships}
    rows = {r["study_instance_uid"]: r for r in feature_manifest["features"]}
    features = {}
    for uid in gold:
        row = rows[uid]
        feature = t36._load_feature_file(t36._feature_path(uid), study_uid=uid, batch_index=row["batch_index"], feature_identity_fingerprint=parent["feature_identity_fingerprint"])
        if feature["file_sha256"] != row["feature_file_sha256"]:
            raise RuntimeError("held-out feature file differs from sealed manifest")
        features[uid] = feature
    views = {arm: [] for arm in plan["ordered_arms"]}
    provenance, replay, coverage = [], [], []
    for fold in FOLD_IDS:
        heldout = tuple(r.study_instance_uid for r in manifest.gold_memberships if r.fold == fold)
        training = tuple(r.study_instance_uid for r in manifest.gold_memberships if r.fold != fold)
        examples = tuple(t36._reference_example(uid, features[uid], gold[uid].reference_labels, fold=fold, fold_manifest_checksum=manifest.manifest_checksum, training_uids=training, heldout_uids=manifest.held_out_study_uids(fold)) for uid in heldout)
        pair = {}
        for arm in plan["ordered_arms"]:
            source = t44.RUN_ROOT / arm / f"fold-{fold}"
            original = load_oof_run_record(source, expected_supervision="reference" if arm == "reference-only" else "report", expected_report_ticket=36)
            attempt = t44._read(source / "attempts/7830-restart-0/attempt.json")
            if attempt["run_id"] != original.run_id or attempt["environment"]["slurm_job_id"] != "7830":
                raise RuntimeError("execution-attempt provenance differs from original run")
            cache = t44._read(source / "attempts/7830-restart-0/frozen-feature-cache.json")
            if cache["feature_manifest_sha256"] != plan["feature_manifest_sha256"] or cache["read_only"] is not True:
                raise RuntimeError("run feature provenance differs from frozen plan")
            model = build_study_model(config.model_config())
            load_training_checkpoint(source / "checkpoints/final.pt", model=model, expected_config=config, expected_fold=fold)
            scores = predict_training_examples(model, examples, batch_size=2)
            error = max(abs(a - b) for uid in heldout for a, b in zip(scores[uid], original.predictions[uid], strict=True))
            if set(scores) != set(heldout) or error > 1e-6:
                raise RuntimeError(f"{arm} fold {fold} checkpoint replay failed: {error}")
            replay.append({"arm": arm, "fold": fold, "held_out_studies": len(scores), "max_abs_error": error, "checkpoint_sha256": t36._sha256_file(source / "checkpoints/final.pt")})
            destination = output / arm / f"fold-{fold}"
            provenance.append(analysis_view(source, destination, snapshot_sha256=plan["source_manifest"]["sha256"], job_id="7830"))
            compute_path = source / "ticket44-compute.json"
            compute = t44._read(compute_path)
            if compute.get("schema_version") != "rsna-ticket44-run-compute-v1" or compute.get("job_id") != "7830" or compute.get("arm") != arm or compute.get("fold") != fold:
                raise RuntimeError("original observed compute evidence differs from run")
            derived_compute = dict(compute)
            derived_compute.update({"schema_version": "rsna-knee-observed-compute-v1", "original_compute_path": str(compute_path), "original_compute_sha256": t36._sha256_file(compute_path)})
            t36._write_private_json(destination.parent / f"fold-{fold}.compute.json", derived_compute)
            if arm == "report-supervision":
                shutil.copyfile(source.parent / f"fold-{fold}.public.json", destination.parent / f"fold-{fold}.public.json")
            views[arm].append(destination)
            pair[arm] = t44._read(source / "run.json")
        a, b = pair["reference-only"]["counts"], pair["report-supervision"]["counts"]
        if any(c["successful_optimizer_steps"] != 1500 for c in (a, b)) or a["initial_model_state_sha256"] != b["initial_model_state_sha256"] or a["gold_exposure_sequence_sha256"] != b["gold_exposure_sequence_sha256"]:
            raise RuntimeError("matched pair violates fixed step, initialization or gold exposure contract")
        supervision = admit_ticket30_fold_supervision(manifest, fold, t36.TICKET30_ARTIFACT, expected_artifact_sha256=t36.TICKET30_AUDITED_ARTIFACT_SHA256).for_image_studies(integration.study_uids)
        weak_uids = tuple(supervision.admitted_report_only_study_uids)
        weak_seed = config.batch_order_seed + TICKET34_WEAK_BATCH_SEED_OFFSET
        from types import SimpleNamespace
        weak_examples = tuple(SimpleNamespace(study_instance_uid=uid) for uid in weak_uids)
        attempted = b["attempted_batches"]
        visited = {weak_uids[index] for step in range(attempted) for index in _batch_indices(len(weak_uids), config.batch_size, step, weak_seed)}
        digest = _exposure_sequence_sha256(weak_examples, config, attempted, weak_seed)
        if visited != set(weak_uids) or digest != b["weak_exposure_sequence_sha256"]:
            raise RuntimeError("report run did not cover the full admitted weak cohort")
        coverage.append({"fold": fold, "admitted_weak_studies": len(weak_uids), "distinct_weak_studies_exposed": len(visited), "weak_sample_exposures": b["exposures_by_source"]["weak"]["sample_exposures"], "weak_supervised_entry_exposures": b["exposures_by_source"]["weak"]["supervised_entry_exposures"], "weak_exposure_sequence_sha256": digest})
    comparison = compare_supervision_oof(views["reference-only"], views["report-supervision"], manifest, bootstrap_resamples=2000, bootstrap_seed=20260929, bootstrap_confidence_level=0.95, require_additive_gold_exposure=True, additive_design_id=TICKET34_ADDITIVE_DESIGN, weak_sampling_policy=TICKET34_WEAK_SAMPLING_POLICY, report_ticket_number=36)
    result = comparison.to_public_dict()
    result.update({"ticket": 44, "plan_sha256": plan["plan_sha256"], "approval_sha256": t36._sha256_file(t44.APPROVAL_PATH), "weak_coverage": coverage, "checkpoint_replay": replay, "source_provenance_supplements": provenance, "reference_100_step_comparison_sha256": t36._sha256_file(t36.RUN_ROOT / "comparison.json"), "analysis_elapsed_seconds": time.perf_counter() - started, "analysis_job_id": os.environ["SLURM_JOB_ID"]})
    interval = comparison.macro_auc_delta_interval
    decision = "provisional_candidate_review" if interval.lower is not None and interval.lower > 0 and comparison.positive_folds == 3 else "reject_long_report_supervision" if interval.upper is not None and interval.upper < 0 else "inconclusive_no_promotion"
    disposition = {"ticket": 44, "decision": decision, "candidate_promotion_authorized": False, "final_candidate_freeze": "blocked", "pooled_macro_auc_delta": comparison.pooled_macro_auc_delta, "paired_95_interval": [interval.lower, interval.upper], "undefined_resamples": interval.undefined_resamples, "approval_scope": approval["scope"]}
    t36._write_private_json(output / "comparison.json", result)
    t36._write_private_json(output / "decision.json", disposition)
    t36._write_private_text(output / "comparison.md", "# Ticket 44 — full weak-cohort exposure\n\n" + render_supervision_comparison_markdown(comparison) + "\n\nSource identity comes from verified frozen snapshot bytes in explicitly derived analysis records. The original missing Git revision and all training artifacts remain preserved. Checkpoint replay covers all 58 held-out studies in both arms.\n")
    print(json.dumps(disposition, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
