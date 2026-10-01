"""Matched 1,500-step CPU follow-up using Ticket 36's sealed features."""

from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping, cast

import torch

from . import ticket36 as t36
from .folds import FOLD_IDS, load_fold_manifest
from .supervision_comparison import compare_supervision_oof, render_supervision_comparison_markdown
from .ticket34 import TICKET34_ADDITIVE_DESIGN, TICKET34_WEAK_BATCH_SEED_OFFSET, TICKET34_WEAK_SAMPLING_POLICY, admit_ticket30_fold_supervision
from .training import _batch_indices, _exposure_sequence_sha256, load_training_checkpoint, require_cpu_preflight, run_cpu_preflight, train_one_fold
from .real_inference import load_integration_manifest
from .pretrained import TICKET44_RESOURCE_SCOPE, RESNET18_ARTIFACT_ID, RESNET18_SOURCE_REVISION, RESNET18_DOWNLOAD_URL, RESNET18_ARTIFACT_BYTES, RESNET18_SHA256, load_resource_approval


ROOT = t36.PROJECT_ROOT
RUN_ROOT = ROOT / "artifacts/private/ticket44-full-weak-exposure"
PLAN_PATH = RUN_ROOT / "plan.json"
APPROVAL_PATH = ROOT / ".scratch/rsna-leaderboard/resource-approvals/ticket44-full-weak-exposure.json"
SOURCE_SCRIPT = ROOT / "slurm/run_ticket44_full_weak_exposure.sbatch"
RESOURCES = {
    "partition": "cluster1-cpu", "nodes": 1, "tasks": 1,
    "cpus_per_task": 16, "memory_gib": 64,
    "walltime_seconds": 7200, "gpu_count": 0,
    "maximum_concurrent_jobs": 1, "external_spend_usd": 0,
}
REQUESTED = {
    "requested_partition": "cluster1-cpu", "requested_nodes": 1,
    "requested_cpus": 16, "requested_memory_gb": 64,
    "requested_wall_time_hours": 2, "exclusive": False,
    "maximum_concurrent_jobs": 1, "allocated_gpu_count": 0,
    "external_spend": 0,
}
PREFLIGHT_RESOURCES = {
    "partition": "cluster1-cpu", "nodes": 1, "ntasks": 1,
    "cpus_per_task": 16, "memory_gb": 64,
    "wall_time_minutes": 120, "exclusive": False,
}


class Ticket44Error(ValueError):
    pass


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise Ticket44Error(f"missing or unsafe frozen input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Ticket44Error(f"expected a JSON object: {path}")
    return value


def _digest(value: object) -> str:
    return t36._json_sha256(value)


def _source_manifest(root: Path) -> dict:
    paths = sorted((root / "src/rsna_knee").glob("*.py")) + [root / "slurm/run_ticket44_full_weak_exposure.sbatch"]
    rows = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise Ticket44Error(f"missing or unsafe source: {path}")
        rows.append({"path": path.relative_to(root).as_posix(), "sha256": t36._sha256_file(path), "bytes": path.stat().st_size})
    result = {"schema_version": "rsna-ticket44-source-v1", "files": rows}
    result["sha256"] = _digest(result)
    return result


def _snapshot_source(manifest: Mapping[str, object]) -> Path:
    t36._private_directory(RUN_ROOT)
    t36._private_directory(RUN_ROOT / "private")
    parent = RUN_ROOT / "private/source-snapshots"
    t36._private_directory(parent)
    destination = parent / str(manifest["sha256"])
    if destination.exists():
        if destination.is_symlink() or _source_manifest(destination) != manifest:
            raise Ticket44Error("existing Ticket 44 source snapshot differs")
        return destination
    staging = Path(tempfile.mkdtemp(prefix=".ticket44-source-", dir=parent))
    try:
        for row in cast(list[dict], manifest["files"]):
            relative = Path(row["path"])
            source = ROOT / relative
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(source, target)
            target.chmod(0o600)
        if _source_manifest(staging) != manifest or _source_manifest(ROOT) != manifest:
            raise Ticket44Error("source changed while freezing Ticket 44")
        os.replace(staging, destination)
        destination.chmod(0o700)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def freeze() -> dict:
    """Freeze exact science, input files, source and CPU request before allocation."""
    if PLAN_PATH.exists():
        return _load_plan()
    parent_plan = t36._load_plan()
    feature_manifest = t36._load_feature_manifest(parent_plan)
    integration = load_integration_manifest(t36.INTEGRATION_PATH)
    fold_manifest = load_fold_manifest(t36.FOLD_MANIFEST_PATH)
    config = replace(t36._make_training_config(integration), successful_optimizer_steps=1500)
    parent_config = json.loads(json.dumps(config.to_dict()))
    parent_config["optimization"]["successful_optimizer_steps"] = 100
    if parent_config != parent_plan["training_configuration"]:
        raise Ticket44Error("Ticket 44 changes more than the successful-step budget")
    manifest = _source_manifest(ROOT)
    snapshot = _snapshot_source(manifest)
    input_paths = {
        "ticket36_plan": t36.PLAN_PATH,
        "feature_manifest": t36.FEATURE_ROOT / "feature-manifest.json",
        "fold_manifest": t36.FOLD_MANIFEST_PATH,
        "integration_manifest": t36.INTEGRATION_PATH,
        "ticket30_labels": t36.TICKET30_ARTIFACT,
        "pretrained_checkpoint": t36.WEIGHTS_PATH,
    }
    inputs = {name: {"path": str(path.resolve()), "sha256": t36._sha256_file(path)} for name, path in input_paths.items()}
    plan = {
        "schema_version": "rsna-ticket44-plan-v1", "ticket": 44,
        "parent_plan_sha256": parent_plan["plan_sha256"],
        "feature_identity_fingerprint": parent_plan["feature_identity_fingerprint"],
        "feature_manifest_sha256": feature_manifest["feature_manifest_sha256"],
        "fold_manifest_checksum": fold_manifest.manifest_checksum,
        "integration_manifest_checksum": integration.manifest_checksum,
        "inputs": inputs,
        "source_manifest": manifest,
        "source_snapshot": str(snapshot.resolve()),
        "training_configuration": config.to_dict(),
        "ordered_arms": ["reference-only", "report-supervision"],
        "folds": list(FOLD_IDS),
        "bootstrap_resamples": 2000, "bootstrap_seed": 20260929,
        "resources": RESOURCES, "requested_resources": REQUESTED,
        "output_root": str(RUN_ROOT.resolve()),
        "candidate_promotion": False, "final_candidate_freeze": False,
        "publication": False,
    }
    plan["plan_sha256"] = _digest(plan)
    t36._write_private_json(PLAN_PATH, plan)
    return plan


def _load_plan() -> dict:
    plan = _read(PLAN_PATH)
    t36._verify_checksum_payload(plan, "plan_sha256", "Ticket 44 plan")
    if plan.get("ticket") != 44 or plan.get("schema_version") != "rsna-ticket44-plan-v1":
        raise Ticket44Error("unsupported Ticket 44 plan")
    snapshot = Path(str(plan["source_snapshot"]))
    if snapshot.is_symlink() or _source_manifest(snapshot) != plan["source_manifest"]:
        raise Ticket44Error("frozen Ticket 44 source has changed")
    if Path(__file__).resolve() != (snapshot / "src/rsna_knee/ticket44.py").resolve():
        if os.environ.get("SLURM_JOB_ID"):
            raise Ticket44Error("Slurm did not load the frozen Ticket 44 source")
    if plan["resources"] != RESOURCES or plan["requested_resources"] != REQUESTED:
        raise Ticket44Error("Ticket 44 CPU request differs from the frozen plan")
    for row in cast(dict[str, dict], plan["inputs"]).values():
        path = Path(row["path"])
        if path.is_symlink() or t36._sha256_file(path) != row["sha256"]:
            raise Ticket44Error(f"frozen Ticket 44 input changed: {path}")
    return plan


def _approval(plan: Mapping[str, object]) -> dict:
    approval = _read(APPROVAL_PATH)
    expected = {
        "schema_version": "rsna-ticket44-resource-approval-v1",
        "scope": TICKET44_RESOURCE_SCOPE,
        "decision": "approved", "approval_status": "approved_by_project_owner",
        "plan_sha256": plan["plan_sha256"],
        "resources": RESOURCES, "external_spend_usd": 0,
        "candidate_promotion": False, "final_candidate_freeze": False,
        "publication": False,
    }
    if any(approval.get(key) != value for key, value in expected.items()):
        raise Ticket44Error("Ticket 44 approval does not match the frozen plan and resource request")
    if not approval.get("approved_by") or not approval.get("approved_at"):
        raise Ticket44Error("Ticket 44 approval lacks provenance")
    if (
        approval.get("rights_status_scope") != TICKET44_RESOURCE_SCOPE
        or approval.get("artifact_id") != RESNET18_ARTIFACT_ID
        or approval.get("source_revision") != RESNET18_SOURCE_REVISION
        or approval.get("download_url") != RESNET18_DOWNLOAD_URL
        or approval.get("expected_bytes") != RESNET18_ARTIFACT_BYTES
        or approval.get("sha256") != RESNET18_SHA256
        or approval.get("checkpoint_sha256") != RESNET18_SHA256
        or approval.get("new_weight_download") is not False
        or approval.get("network_during_execution") is not False
    ):
        raise Ticket44Error("Ticket 44 approval does not cover the exact pretrained checkpoint")
    load_resource_approval(APPROVAL_PATH)
    return approval


def _verify_allocation() -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.isdecimal():
        raise Ticket44Error("Ticket 44 training requires Slurm")
    record = subprocess.run(["scontrol", "show", "job", "-o", job_id], check=True, capture_output=True, text=True).stdout
    fields = dict(token.split("=", 1) for token in record.split() if "=" in token)
    expected = {
        "JobId": job_id, "Command": str(SOURCE_SCRIPT),
        "Partition": "cluster1-cpu", "NumNodes": "1", "NumCPUs": "16",
        "NumTasks": "1", "CPUs/Task": "16", "MinMemoryNode": "64G",
        "TimeLimit": "02:00:00", "OverSubscribe": "OK",
    }
    if any(fields.get(key) != value for key, value in expected.items()):
        raise Ticket44Error("live Slurm allocation differs from the exact Ticket 44 CPU request")
    if "gpu" in fields.get("TRES", "").lower() or "gpu" in fields.get("Gres", "").lower():
        raise Ticket44Error("Ticket 44 allocation has a GPU")
    torch.set_num_threads(16)
    return record


def _run_fold_arm(arm: str, fold: int, studies: Mapping, fold_manifest, integration, config, examples, supervision, preflight, evidence, plan) -> None:
    run_dir = RUN_ROOT / arm / f"fold-{fold}"
    if (run_dir / "run.json").is_file():
        record = _read(run_dir / "run.json")
        if record.get("status") != "completed_fixed_budget_training" or not (run_dir / "checkpoints/final.pt").is_file():
            raise Ticket44Error("existing completed run has incomplete evidence")
        return
    t36._private_directory(run_dir.parent)
    latest = run_dir / "checkpoints/latest.pt"
    if run_dir.exists() and any(run_dir.iterdir()) and not latest.is_file():
        raise Ticket44Error("partial run has no resumable checkpoint")
    started = time.perf_counter()
    prior = resource.getrusage(resource.RUSAGE_SELF)
    attempt = f"{os.environ['SLURM_JOB_ID']}-restart-{os.environ.get('SLURM_RESTART_COUNT', '0')}"
    result = train_one_fold(
        studies, fold_manifest, fold, run_dir, config=config,
        source_version=integration.source_version,
        source_fingerprint=integration.archive_fingerprint,
        cache_fingerprint=_digest({"feature_identity_fingerprint": plan["feature_identity_fingerprint"], "feature_manifest_sha256": evidence["feature_manifest_sha256"]}),
        integration_manifest_checksum=integration.manifest_checksum,
        supervision=supervision, ticket34_additive_design=True,
        preflight=preflight, require_preflight=True, allow_local_smoke=False,
        requested_resources=REQUESTED,
        pretrained_weights_path=t36.WEIGHTS_PATH,
        resource_approval_path=APPROVAL_PATH,
        precomputed_feature_examples=examples,
        precomputed_feature_evidence=evidence,
        resume_from=latest if latest.is_file() else None,
        execution_attempt_id=attempt,
    )
    if result.final_checkpoint_path is None:
        raise Ticket44Error("completed training did not produce a final checkpoint")
    load_training_checkpoint(result.final_checkpoint_path, model=result.model, expected_config=config, expected_fold=fold)
    after = resource.getrusage(resource.RUSAGE_SELF)
    compute = {
        "schema_version": "rsna-ticket44-run-compute-v1", "arm": arm, "fold": fold,
        "job_id": os.environ["SLURM_JOB_ID"], "elapsed_seconds": time.perf_counter() - started,
        "user_cpu_seconds": after.ru_utime - prior.ru_utime,
        "system_cpu_seconds": after.ru_stime - prior.ru_stime,
        "peak_rss_kb": after.ru_maxrss,
    }
    t36._write_private_json(run_dir / "ticket44-compute.json", compute)
    if arm == "report-supervision":
        assert supervision is not None
        t36._write_private_json(run_dir.parent / f"fold-{fold}.public.json", {
            "schema_version": "rsna-ticket36-report-training-evidence-v1", "ticket": 36,
            "fold": fold, "fold_manifest_checksum": fold_manifest.manifest_checksum,
            "preflight_passed": preflight.passed,
            "report_extraction_semantic_digest": t36.TICKET30_AUDITED_ARTIFACT_SHA256,
            "frozen_report_artifact_sha256": t36.TICKET30_AUDITED_ARTIFACT_SHA256,
            "supervision_admission": supervision.to_public_dict(),
            "full_fold_supervision_admission": admit_ticket30_fold_supervision(fold_manifest, fold, t36.TICKET30_ARTIFACT, expected_artifact_sha256=t36.TICKET30_AUDITED_ARTIFACT_SHA256).to_public_dict(),
            "feature_manifest_sha256": evidence["feature_manifest_sha256"],
            "held_out_inputs_excluded_from_adaptation": True, "raw_report_text_read": False,
        })


def run() -> dict:
    plan = _load_plan()
    approval = _approval(plan)
    allocation = _verify_allocation()
    parent_plan = t36._load_plan()
    feature_manifest = t36._load_feature_manifest(parent_plan)
    fold_manifest = load_fold_manifest(t36.FOLD_MANIFEST_PATH)
    integration = load_integration_manifest(t36.INTEGRATION_PATH)
    config = replace(t36._make_training_config(integration), successful_optimizer_steps=1500)
    if config.to_dict() != plan["training_configuration"] or feature_manifest["feature_manifest_sha256"] != plan["feature_manifest_sha256"]:
        raise Ticket44Error("Ticket 44 input or training configuration changed")
    attempt = f"{os.environ['SLURM_JOB_ID']}-restart-{os.environ.get('SLURM_RESTART_COUNT', '0')}"
    t36._write_private_json(RUN_ROOT / "allocations" / f"{attempt}.json", {"schema_version": "rsna-ticket44-allocation-v1", "record": allocation})
    preflight = run_cpu_preflight(
        output_path=RUN_ROOT / f"cpu-preflight-{attempt}.json",
        require_isolated_python=True, require_slurm=True,
        required_slurm_resources=PREFLIGHT_RESOURCES,
    )
    require_cpu_preflight(preflight, require_slurm=True, require_isolated_python=True)
    features = t36._load_training_features(parent_plan, feature_manifest)
    if set(features) != set(integration.study_uids):
        raise Ticket44Error("frozen feature cohort is incomplete")
    evidence = {
        "identity": parent_plan["feature_identity"],
        "identity_fingerprint": parent_plan["feature_identity_fingerprint"],
        "feature_manifest_sha256": feature_manifest["feature_manifest_sha256"],
        "feature_count": feature_manifest["feature_study_count"],
        "feature_cache_schema_version": t36.FEATURE_CACHE_SCHEMA_VERSION,
        "source_ticket35_series_cache": str(t36.TICKET35_SERIES_CACHE.resolve()),
        "source_ticket35_cache_access": "read-only", "read_only": True,
    }
    outputs: dict[str, list[Path]] = {"reference-only": [], "report-supervision": []}
    coverage: list[dict] = []
    gold = {item.study_instance_uid: item for item in fold_manifest.gold_memberships}
    for fold in FOLD_IDS:
        supervision = admit_ticket30_fold_supervision(fold_manifest, fold, t36.TICKET30_ARTIFACT, expected_artifact_sha256=t36.TICKET30_AUDITED_ARTIFACT_SHA256).for_image_studies(integration.study_uids)
        heldout = tuple(item.study_instance_uid for item in fold_manifest.gold_memberships if item.fold == fold)
        training = tuple(item.study_instance_uid for item in fold_manifest.gold_memberships if item.fold != fold)
        partition = tuple(fold_manifest.held_out_study_uids(fold))
        reference_examples = tuple(t36._reference_example(uid, features[uid], gold[uid].reference_labels, fold=fold, fold_manifest_checksum=fold_manifest.manifest_checksum, training_uids=training, heldout_uids=partition) for uid in training)
        heldout_examples = tuple(t36._reference_example(uid, features[uid], gold[uid].reference_labels, fold=fold, fold_manifest_checksum=fold_manifest.manifest_checksum, training_uids=training, heldout_uids=partition) for uid in heldout)
        weak_examples = tuple(t36._weak_example(uid, features[uid], supervision) for uid in supervision.admitted_report_only_study_uids)
        studies = t36._load_single_heldout_study(integration, fold_manifest, fold)
        for arm, weak, admitted in (("reference-only", (), None), ("report-supervision", weak_examples, supervision)):
            _run_fold_arm(arm, fold, studies, fold_manifest, integration, config, (reference_examples, heldout_examples, weak), admitted, preflight, evidence, plan)
            outputs[arm].append(RUN_ROOT / arm / f"fold-{fold}")
        gold_run = _read(outputs["reference-only"][-1] / "run.json")
        report_run = _read(outputs["report-supervision"][-1] / "run.json")
        a, b = gold_run["counts"], report_run["counts"]
        if a["successful_optimizer_steps"] != 1500 or b["successful_optimizer_steps"] != 1500:
            raise Ticket44Error("a fold did not complete the exact successful-step budget")
        if a["initial_model_state_sha256"] != b["initial_model_state_sha256"] or a["gold_exposure_sequence_sha256"] != b["gold_exposure_sequence_sha256"]:
            raise Ticket44Error("matched pair differs in initialization or ordered gold exposure")
        attempted = int(b["attempted_batches"])
        weak_seed = config.batch_order_seed + TICKET34_WEAK_BATCH_SEED_OFFSET
        visited = {weak_examples[index].study_instance_uid for step in range(attempted) for index in _batch_indices(len(weak_examples), config.batch_size, step, weak_seed)}
        digest = _exposure_sequence_sha256(weak_examples, config, attempted, weak_seed)
        if visited != set(supervision.admitted_report_only_study_uids) or digest != b["weak_exposure_sequence_sha256"]:
            raise Ticket44Error("the report arm did not expose every admitted weak study as planned")
        coverage.append({
            "fold": fold, "admitted_weak_studies": len(weak_examples),
            "distinct_weak_studies_exposed": len(visited),
            "weak_sample_exposures": b["exposures_by_source"]["weak"]["sample_exposures"],
            "weak_supervised_entry_exposures": b["exposures_by_source"]["weak"]["supervised_entry_exposures"],
            "reference_supervised_entry_exposures": b["exposures_by_source"]["reference"]["supervised_entry_exposures"],
            "attempted_batches": attempted,
            "weak_exposure_sequence_sha256": digest,
        })
    comparison = compare_supervision_oof(
        outputs["reference-only"], outputs["report-supervision"], fold_manifest,
        bootstrap_resamples=2000, bootstrap_seed=20260929,
        bootstrap_confidence_level=0.95,
        require_additive_gold_exposure=True,
        additive_design_id=TICKET34_ADDITIVE_DESIGN,
        weak_sampling_policy=TICKET34_WEAK_SAMPLING_POLICY,
        report_ticket_number=36,
    )
    result = comparison.to_public_dict()
    result.update({"ticket": 44, "parent_ticket": 36, "plan_sha256": plan["plan_sha256"], "approval_sha256": t36._sha256_file(APPROVAL_PATH), "weak_coverage": coverage, "reference_100_step_comparison_sha256": t36._sha256_file(t36.RUN_ROOT / "comparison.json")})
    t36._write_private_json(RUN_ROOT / "comparison.json", result)
    t36._write_private_text(RUN_ROOT / "comparison.md", "# Ticket 44 — full weak-cohort exposure\n\n" + render_supervision_comparison_markdown(comparison))
    interval = comparison.macro_auc_delta_interval
    if comparison.pooled_macro_auc_delta is not None and interval.lower is not None and interval.lower > 0 and comparison.positive_folds == 3:
        decision = "provisional_candidate_review"
    elif interval.upper is not None and interval.upper < 0:
        decision = "reject_long_report_supervision"
    else:
        decision = "inconclusive_no_promotion"
    decision_record = {"schema_version": "rsna-ticket44-development-decision-v1", "ticket": 44, "decision": decision, "candidate_promotion_authorized": False, "final_candidate_freeze": "blocked", "publication": "blocked", "pooled_macro_auc_delta": comparison.pooled_macro_auc_delta, "paired_95_interval": [interval.lower, interval.upper], "undefined_resamples": interval.undefined_resamples, "comparison_path": str(RUN_ROOT / "comparison.json"), "approval_scope": approval["scope"]}
    t36._write_private_json(RUN_ROOT / "decision.json", decision_record)
    return decision_record


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    try:
        result = freeze() if args.command == "freeze" else run()
    except (Ticket44Error, t36.Ticket36Error, OSError, ValueError) as exc:
        parser.exit(2, f"Ticket 44 failed: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
