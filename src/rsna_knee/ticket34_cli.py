"""Execute the approval-gated fold-0 Ticket 34 supervision pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence, cast

from .archive import atomic_json_write
from .folds import FoldManifest, load_fold_manifest
from .image_variant import TICKET31_RESNET18_VARIANT, image_model_training_config
from .model import ANATOMICAL_POOLING_ID
from .pretrained import (
    RESNET18_ARTIFACT_BYTES,
    RESNET18_SHA256,
    RESNET18_SOURCE_REVISION,
    RESNET18_APPROVAL_SCHEMA,
    TICKET34_RESOURCE_SCOPE,
    PretrainedResourceError,
    load_resource_approval,
)
from .real_inference import IntegrationManifest, iter_real_studies, load_integration_manifest
from .series_cache import SeriesCache
from .supervision import AdmittedFoldSupervision
from .ticket34 import (
    TICKET30_AUDITED_ARTIFACT_SHA256,
    TICKET34_ADDITIVE_DESIGN,
    TICKET34_WEAK_BATCH_SEED_OFFSET,
    TICKET34_WEAK_SAMPLING_POLICY,
    Ticket34Error,
    admit_ticket30_fold_supervision,
    compare_supervision_pilot,
)
from .training import (
    CpuPreflightResult,
    PreflightError,
    TrainingConfig,
    TrainingResult,
    require_cpu_preflight,
    run_cpu_preflight,
    train_one_fold,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RUN_ROOT = _PROJECT_ROOT / "artifacts/private/ticket34-resnet18-supervision"
_FOLD_MANIFEST_CHECKSUM = "8460234d0fcbfd2eb9ca3a3e4165929f50a86acac37d2cd03be1164c2f287fb8"
_INTEGRATION_MANIFEST_CHECKSUM = "c0b11e22cd2575af7d964fbe3b8b12603ad0c6291a6a1705e0fb53fcfd957b94"
_SOURCE_FINGERPRINT = "25a70674e5af8808c4cf6d7e0da6c8b8f894e1856a11e064470ea0bcdab25648"
_TICKET32_FEATURE_CACHE_ROOT = _PROJECT_ROOT / "artifacts/private/ticket32-resnet18/frozen-feature-cache"
_TICKET32_FEATURE_CACHE_IDENTITY = "c2cdb5cc8fc27ced67688caa9f695a955adebabb951546c6bc2d1d512495b6b3"
_RESOURCE_REQUIREMENTS: Mapping[str, object] = {
    "partition": "cluster1-cpu",
    "nodes": 1,
    "ntasks": 1,
    "cpus_per_task": 16,
    "memory_gb": 64,
    "wall_time_minutes": 360,
    "exclusive": False,
}
_REQUESTED_RESOURCES: Mapping[str, object] = {
    "requested_partition": "cluster1-cpu",
    "requested_nodes": 1,
    "requested_cpus": 16,
    "requested_memory_gb": 64,
    "requested_wall_time_hours": 6,
    "exclusive": False,
    "allocated_gpu_count": 0,
    "external_spend": 0,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--integration-manifest",
        type=Path,
        default=_PROJECT_ROOT / "artifacts/private/ticket-05-real-training-input/private/integration-manifest.json",
    )
    parser.add_argument("--source", type=Path, default=_PROJECT_ROOT / "rsna-knee-abnormality-detection.zip")
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=_PROJECT_ROOT / "artifacts/private/ticket-04-real-v2/private/fold-manifest.json",
    )
    parser.add_argument(
        "--ticket30-artifact",
        type=Path,
        default=Path("/home/ch.jitheswar/ticket-30-report-audit-20260925/private/admitted-report-labels-v4.json"),
    )
    parser.add_argument(
        "--resource-approval",
        type=Path,
        default=_PROJECT_ROOT / ".scratch/rsna-leaderboard/resource-approvals/ticket34-resnet18-imagenet1k-v1.json",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=_PROJECT_ROOT / "artifacts/private/ticket31-resnet18/weights/resnet18-f37072fd.pth",
    )
    parser.add_argument(
        "--preflight-evidence",
        type=Path,
        default=_RUN_ROOT / "cpu-preflight.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_RUN_ROOT,
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uid_set_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def _validate_resource_approval(path: Path) -> Mapping[str, object]:
    approval = load_resource_approval(path)
    if (
        approval.get("schema_version") != RESNET18_APPROVAL_SCHEMA
        or approval.get("scope") != TICKET34_RESOURCE_SCOPE
        or approval.get("artifact_id") != "torchvision.ResNet18_Weights.IMAGENET1K_V1"
        or approval.get("source_revision") != RESNET18_SOURCE_REVISION
        or approval.get("sha256") != RESNET18_SHA256
        or approval.get("expected_bytes") != RESNET18_ARTIFACT_BYTES
        or approval.get("external_spend") != 0
    ):
        raise PretrainedResourceError(
            "Ticket 34 requires its exact fold-0 CPU resource approval record"
        )
    scope_details = approval.get("scope_details")
    if not isinstance(scope_details, Mapping):
        raise PretrainedResourceError("Ticket 34 resource approval is missing its scoped use details")
    experiment = scope_details.get("experiment")
    if not isinstance(experiment, str) or "ticket34-run-plan.json" not in experiment:
        raise PretrainedResourceError("Ticket 34 approval does not point to the frozen run plan")
    required_scope_phrases = {
        "held_out_data": ("prediction and evaluation only", "no fitting"),
        "report_supervision": (
            "Ticket 30 audit v4",
            "ticket16-astra-medium",
            "61 image-backed admitted studies",
        ),
        "image_data": ("Do not access or download ImageNet source images",),
        "acquisition": ("exact existing local checkpoint", "No transfer or payment"),
        "processing": ("offline CPU Slurm", "no runtime network", "no GPU"),
        "reuse_and_sharing": (
            "Ticket 32 frozen-feature cache",
            "read-only",
            "both Ticket 34 arms",
            "zero writes",
            "only for Ticket 34",
            "separate new Ticket 34 cache",
        ),
        "release": ("does not authorize", "blocked"),
    }
    for field_name, phrases in required_scope_phrases.items():
        value = scope_details.get(field_name)
        if not isinstance(value, str) or any(phrase not in value for phrase in phrases):
            raise PretrainedResourceError(
                f"Ticket 34 approval does not preserve its {field_name} restrictions"
            )
    release_eligibility = approval.get("release_eligibility")
    if not isinstance(release_eligibility, Mapping) or any(
        release_eligibility.get(field_name) != "blocked"
        for field_name in (
            "final_candidate_freeze",
            "winner_publication",
            "redistribution_or_release",
        )
    ):
        raise PretrainedResourceError("Ticket 34 approval must keep all release actions blocked")
    return approval


def _check_cluster_allocation() -> None:
    required = {
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
        "SLURM_JOB_PARTITION": os.environ.get("SLURM_JOB_PARTITION"),
        "SLURM_JOB_NUM_NODES": os.environ.get("SLURM_JOB_NUM_NODES"),
        "SLURM_NTASKS": os.environ.get("SLURM_NTASKS"),
        "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK"),
    }
    if not required["SLURM_JOB_ID"] or required["SLURM_JOB_PARTITION"] != "cluster1-cpu":
        raise PreflightError("Ticket 34 must run inside its cluster1-cpu Slurm allocation")
    if (
        required["SLURM_JOB_NUM_NODES"] != "1"
        or required["SLURM_NTASKS"] != "1"
        or required["SLURM_CPUS_PER_TASK"] != "16"
        or os.environ.get("SLURM_MEM_PER_NODE", "").strip().upper() not in {"64G", "65536"}
        or os.environ.get("SLURM_TIMELIMIT", "").strip() not in {"360", "06:00", "06:00:00"}
        or os.environ.get("SLURM_JOB_EXCLUSIVE", "0").strip().lower() in {"1", "true", "yes"}
    ):
        raise PreflightError("Slurm allocation differs from the frozen Ticket 34 CPU envelope")


def _load_plan(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Ticket34Error("the frozen Ticket 34 run plan could not be read") from exc
    if not isinstance(payload, Mapping):
        raise Ticket34Error("the frozen Ticket 34 run plan must be an object")
    if (
        payload.get("schema_version") != "rsna-ticket34-one-fold-report-supervision-plan-v1"
        or payload.get("ticket") != 34
        or payload.get("scientific_identity")
        != "ticket34-fold0-ticket16-astra-medium-vs-reference-only-resnet18-v1"
    ):
        raise Ticket34Error("the Ticket 34 run plan identity is not the frozen fold-0 pilot")
    lineage = payload.get("evidence_lineage")
    if not isinstance(lineage, Mapping) or (
        lineage.get("fold_manifest_checksum") != _FOLD_MANIFEST_CHECKSUM
        or lineage.get("integration_manifest_checksum") != _INTEGRATION_MANIFEST_CHECKSUM
        or lineage.get("source_fingerprint") != _SOURCE_FINGERPRINT
        or lineage.get("admitted_label_artifact_sha256") != TICKET30_AUDITED_ARTIFACT_SHA256
    ):
        raise Ticket34Error("the Ticket 34 run plan evidence lineage changed")
    return payload


def _validate_frozen_execution_policy(
    plan: Mapping[str, object],
    config: TrainingConfig,
) -> None:
    matched = plan.get("matched_training_configuration")
    if not isinstance(matched, Mapping):
        raise Ticket34Error("the Ticket 34 plan is missing its matched training configuration")
    cache_reuse_policy = matched.get("prior_weights_and_feature_cache_reuse")
    if (
        not isinstance(cache_reuse_policy, str)
        or _TICKET32_FEATURE_CACHE_IDENTITY not in cache_reuse_policy
        or "58 fallback imports" not in cache_reuse_policy
        or "separate new Ticket 34 cache" not in cache_reuse_policy
    ):
        raise Ticket34Error("the Ticket 34 plan changed its approved Ticket 32 cache reuse policy")
    preprocessing = config.preprocess_config().cache_config()
    observed_training = {
        "model_id": config.model_id,
        "encoder_id": config.encoder_id,
        "encoder_weights_sha256": config.encoder_weights_sha256,
        "pooling_id": config.pooling_id,
        "image_size": list(config.image_size),
        "image_representation": config.preprocess_config().representation_id,
        "preprocessing_schema": preprocessing.get("schema_version"),
        "max_series": config.max_series,
        "max_slices": config.max_slices,
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "schedule": config.schedule,
        "loss": config.loss,
        "batch_size": config.batch_size,
        "successful_optimizer_steps": config.successful_optimizer_steps,
        "initialization_seed": config.seed,
        "reference_batch_order_seed": config.batch_order_seed,
        "selection_policy": config.selection_policy,
        "augmentation": config.augmentation,
        "device": config.device,
        "weak_sampling_policy": TICKET34_WEAK_SAMPLING_POLICY,
        "weak_batch_seed_offset": TICKET34_WEAK_BATCH_SEED_OFFSET,
    }
    planned_training = {
        "model_id": matched.get("model_id"),
        "encoder_id": matched.get("encoder_id"),
        "encoder_weights_sha256": matched.get("encoder_weights_sha256"),
        "pooling_id": matched.get("pooling_id"),
        "image_size": matched.get("image_size"),
        "image_representation": matched.get("preprocessing_representation"),
        "preprocessing_schema": matched.get("preprocessing_schema"),
        "max_series": matched.get("packing", {}).get("max_series")
        if isinstance(matched.get("packing"), Mapping)
        else None,
        "max_slices": matched.get("packing", {}).get("max_slices")
        if isinstance(matched.get("packing"), Mapping)
        else None,
        "optimizer": matched.get("optimizer"),
        "learning_rate": matched.get("learning_rate"),
        "schedule": matched.get("schedule"),
        "loss": matched.get("loss"),
        "batch_size": matched.get("batch_size"),
        "successful_optimizer_steps": matched.get("successful_optimizer_steps"),
        "initialization_seed": matched.get("initialization_seed"),
        "reference_batch_order_seed": matched.get("reference_batch_order_seed"),
        "selection_policy": matched.get("selection_policy"),
        "augmentation": matched.get("augmentation"),
        "device": matched.get("device"),
        "weak_sampling_policy": matched.get("weak_sampling_policy"),
        "weak_batch_seed_offset": matched.get("weak_batch_seed_offset"),
    }
    if planned_training != observed_training:
        raise Ticket34Error("constructed training configuration differs from the frozen Ticket 34 plan")
    if (
        matched.get("encoder_frozen") is not True
        or matched.get("same_initial_state_required") is not True
        or matched.get("identical_reference_exposure_sequence_required") is not True
    ):
        raise Ticket34Error("the frozen Ticket 34 plan does not preserve its paired controls")
    arms = plan.get("arms")
    reference_arm = arms.get("reference_only") if isinstance(arms, Mapping) else None
    report_arm = arms.get("report_supervision") if isinstance(arms, Mapping) else None
    if (
        not isinstance(reference_arm, Mapping)
        or not isinstance(report_arm, Mapping)
        or reference_arm.get("training_design")
        != f"{TICKET34_ADDITIVE_DESIGN} control"
        or report_arm.get("training_design")
        != f"{TICKET34_ADDITIVE_DESIGN} treatment"
    ):
        raise Ticket34Error("the frozen Ticket 34 arm definitions changed")

    execution = plan.get("execution")
    if not isinstance(execution, Mapping):
        raise Ticket34Error("the Ticket 34 plan is missing its bounded execution request")
    planned_resources = {
        "partition": execution.get("partition"),
        "nodes": execution.get("nodes"),
        "cpus": execution.get("cpus"),
        "memory_gib": execution.get("memory_gib"),
        "wall_time_limit": execution.get("wall_time_limit"),
        "exclusive": execution.get("exclusive"),
        "gpu": execution.get("gpu"),
        "external_spend": execution.get("external_spend"),
        "network_during_execution": execution.get("network_during_execution"),
        "new_weight_download": execution.get("new_weight_download"),
    }
    expected_resources = {
        "partition": "cluster1-cpu",
        "nodes": 1,
        "cpus": 16,
        "memory_gib": 64,
        "wall_time_limit": "06:00:00",
        "exclusive": False,
        "gpu": False,
        "external_spend": 0,
        "network_during_execution": False,
        "new_weight_download": False,
    }
    if planned_resources != expected_resources:
        raise Ticket34Error("the Ticket 34 plan exceeds its frozen CPU-only resource envelope")
    expected_cache_roots = {
        "reference_only": "artifacts/private/ticket32-resnet18/frozen-feature-cache",
        "report_supervision": "artifacts/private/ticket34-resnet18-supervision/frozen-feature-cache/report-supervision",
    }
    reused_cache = execution.get("reused_feature_cache")
    if (
        execution.get("run_root") != "artifacts/private/ticket34-resnet18-supervision"
        or execution.get("feature_cache_roots") != expected_cache_roots
        or not isinstance(reused_cache, Mapping)
        or reused_cache.get("ticket") != 32
        or reused_cache.get("identity_fingerprint") != _TICKET32_FEATURE_CACHE_IDENTITY
        or reused_cache.get("read_only") is not True
        or reused_cache.get("studies_shared_by_both_arms") != 58
        or reused_cache.get("required_reference_hits") != 58
        or reused_cache.get("required_reference_materializations") != 0
        or reused_cache.get("required_treatment_fallback_imports") != 58
        or reused_cache.get("required_treatment_materializations") != 61
        or execution.get("weights_path")
        != "artifacts/private/ticket31-resnet18/weights/resnet18-f37072fd.pth"
        or execution.get("approval_record")
        != ".scratch/rsna-leaderboard/resource-approvals/ticket34-resnet18-imagenet1k-v1.json"
    ):
        raise Ticket34Error("the Ticket 34 plan changed its approved artifact or cache locations")

    comparison = plan.get("comparison_policy")
    bootstrap = comparison.get("paired_bootstrap") if isinstance(comparison, Mapping) else None
    if (
        not isinstance(comparison, Mapping)
        or not isinstance(bootstrap, Mapping)
        or comparison.get("held_out_predictions")
        != "exactly the same 20 fold-0 gold studies in both arms"
        or comparison.get("primary_metric")
        != "fold-0 macro ROC-AUC over targets with both classes; one-fold pilot only"
        or bootstrap.get("unit") != "study"
        or bootstrap.get("resamples") != 2000
        or bootstrap.get("seed") != 20260925
        or bootstrap.get("confidence_level") != 0.95
    ):
        raise Ticket34Error("the Ticket 34 comparison policy differs from the frozen pilot")


def _validate_weights(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise PretrainedResourceError("the already acquired ResNet-18 weights must be a regular file")
    if path.stat().st_size != RESNET18_ARTIFACT_BYTES or _sha256_file(path) != RESNET18_SHA256:
        raise PretrainedResourceError("the existing ResNet-18 weights fail exact size or SHA-256 verification")


def _build_image_scoped_supervision(
    fold_manifest: FoldManifest,
    *,
    integration_study_uids: Sequence[str],
    ticket30_artifact: Path,
    plan: Mapping[str, object],
) -> tuple[AdmittedFoldSupervision, tuple[str, ...], dict[str, object]]:
    fold = 0
    gold_training = {
        item.study_instance_uid for item in fold_manifest.gold_memberships if item.fold != fold
    }
    held_out_gold = {
        item.study_instance_uid for item in fold_manifest.gold_memberships if item.fold == fold
    }
    eligible_report_training = {
        item.study_instance_uid
        for item in fold_manifest.report_only_memberships
        if item.fold != fold
    }
    integration_uids = set(integration_study_uids)
    if not gold_training.issubset(integration_uids) or not held_out_gold.issubset(integration_uids):
        raise Ticket34Error("the image integration manifest is missing a fold-0 gold study")
    image_report_training = eligible_report_training.intersection(integration_uids)
    image_scope = tuple(
        sorted((gold_training | held_out_gold | eligible_report_training).intersection(integration_uids))
    )
    admitted = admit_ticket30_fold_supervision(fold_manifest, fold, ticket30_artifact)
    image_admitted = admitted.for_image_studies(image_scope)
    selected_uids = tuple(
        sorted(set(image_admitted.training_study_uids) | held_out_gold)
    )
    admitted_weak = set(image_admitted.admitted_report_only_study_uids)
    all_source_admitted = set(admitted.admitted_report_only_study_uids)
    no_call_image_reports = image_report_training - all_source_admitted
    admitted_target_calls = sum(
        admitted.by_study[uid].supervised_entry_count for uid in admitted_weak
    )
    plan_cohort = plan.get("cohort")
    if not isinstance(plan_cohort, Mapping):
        raise Ticket34Error("the Ticket 34 run plan is missing its frozen cohort")
    observed = {
        "integration_cohort_size": len(integration_uids),
        "gold_training_studies_with_images": len(gold_training),
        "held_out_gold_studies_with_images": len(held_out_gold),
        "report_only_training_studies_with_images": len(image_report_training),
        "report_only_studies_with_at_least_one_admitted_call": len(admitted_weak),
        "admitted_target_calls_in_image_cohort": admitted_target_calls,
        "image_report_only_studies_without_admitted_calls": len(no_call_image_reports),
    }
    for name, value in observed.items():
        if plan_cohort.get(name) != value:
            raise Ticket34Error(f"observed Ticket 34 cohort count {name} differs from its frozen plan")
    if len(image_admitted.reference_study_uids) != len(gold_training):
        raise Ticket34Error("the image-scoped report arm changed the gold training population")
    if not admitted_weak or not admitted_weak.issubset(image_report_training):
        raise Ticket34Error("the fold-0 treatment has no image-backed admitted report studies")

    source_counts = admitted.to_public_dict()["counts"]
    image_counts = image_admitted.to_public_dict()["counts"]
    if not isinstance(source_counts, Mapping) or not isinstance(image_counts, Mapping):
        raise Ticket34Error("image-scoped supervision counts are malformed")
    cohort_evidence: dict[str, object] = {
        "schema_version": "rsna-ticket34-fold0-image-cohort-v1",
        "fold": fold,
        "fold_manifest_checksum": fold_manifest.manifest_checksum,
        "ticket30_admitted_artifact_sha256": admitted.source_artifact_digest,
        "integration_study_count": len(integration_uids),
        "gold_training_studies_with_images": len(gold_training),
        "held_out_gold_studies_with_images": len(held_out_gold),
        "eligible_report_only_training_studies": len(eligible_report_training),
        "eligible_report_only_training_studies_with_images": len(image_report_training),
        "eligible_report_only_training_studies_without_images": len(
            eligible_report_training - integration_uids
        ),
        "source_admitted_report_only_training_studies": source_counts.get(
            "admitted_report_only_studies"
        ),
        "source_missing_report_only_training_studies": source_counts.get(
            "missing_report_only_studies"
        ),
        "image_backed_admitted_report_only_studies": len(admitted_weak),
        "image_backed_report_only_studies_without_admitted_calls": len(no_call_image_reports),
        "admitted_report_only_training_studies_without_images": len(
            image_admitted.excluded_image_report_only_study_uids
        ),
        "image_scoped_training_studies": image_counts.get("training_studies"),
        "image_scoped_supervised_entries": image_counts.get("supervised_entries"),
        "image_scoped_report_supervised_entries": admitted_target_calls,
        "image_scoped_masked_report_target_entries_per_epoch": (
            len(admitted_weak) * 12 - admitted_target_calls
        ),
        "raw_report_text_read": False,
        "held_out_report_labels_used_for_fitting": False,
        "image_backed_report_only_uid_set_sha256": _uid_set_sha256(
            tuple(image_report_training)
        ),
        "image_backed_admitted_report_uid_set_sha256": _uid_set_sha256(
            tuple(admitted_weak)
        ),
        "selected_pixel_study_count": len(selected_uids),
        "selected_pixel_study_uid_set_sha256": _uid_set_sha256(selected_uids),
    }
    return image_admitted, selected_uids, cohort_evidence


def _verify_manifests(
    integration_manifest_path: Path,
    fold_manifest_path: Path,
    source_path: Path,
) -> tuple[IntegrationManifest, FoldManifest]:
    fold_manifest = load_fold_manifest(fold_manifest_path)
    integration_manifest = load_integration_manifest(integration_manifest_path)
    if fold_manifest.superseded:
        raise Ticket34Error("the superseded fold manifest cannot run Ticket 34")
    if (
        fold_manifest.manifest_checksum != _FOLD_MANIFEST_CHECKSUM
        or integration_manifest.manifest_checksum != _INTEGRATION_MANIFEST_CHECKSUM
        or integration_manifest.archive_fingerprint != _SOURCE_FINGERPRINT
        or fold_manifest.source_version != integration_manifest.source_version
        or fold_manifest.source_manifest_checksum != integration_manifest.source_manifest_checksum
    ):
        raise Ticket34Error("the Ticket 34 image or fold lineage differs from the frozen plan")
    if not source_path.exists():
        raise Ticket34Error("the declared real-DICOM source is missing")
    return integration_manifest, fold_manifest


def _write_run_resources() -> dict[str, object]:
    return dict(_REQUESTED_RESOURCES)


def _require_new_outputs(output_root: Path, preflight_path: Path) -> None:
    paths = (
        preflight_path,
        output_root / "cohort.json",
        output_root / "comparison.json",
        output_root / "execution.json",
        output_root / "reference-only",
        output_root / "report-supervision",
        output_root / "series-cache",
        output_root / "frozen-feature-cache",
    )
    occupied = [path for path in paths if path.exists() and (not path.is_dir() or any(path.iterdir()))]
    if occupied:
        raise Ticket34Error(
            "refusing to overwrite existing Ticket 34 run artifacts: "
            + ", ".join(str(path) for path in occupied)
        )


def _run_pilot(
    args: argparse.Namespace,
    *,
    approval: Mapping[str, object],
    started_at: float,
) -> dict[str, object]:
    _check_cluster_allocation()
    plan_path = args.output_root / "ticket34-run-plan.json"
    plan = _load_plan(plan_path)
    if args.output_root.resolve() != _RUN_ROOT.resolve():
        raise Ticket34Error("the run output root must match the exact Ticket 34 approved scope")
    if args.weights.resolve() != (
        _PROJECT_ROOT / "artifacts/private/ticket31-resnet18/weights/resnet18-f37072fd.pth"
    ).resolve():
        raise PretrainedResourceError("Ticket 34 may use only the already acquired checkpoint at its frozen path")
    _validate_weights(args.weights)
    _require_new_outputs(args.output_root, args.preflight_evidence)
    integration_manifest, fold_manifest = _verify_manifests(
        args.integration_manifest,
        args.fold_manifest,
        args.source,
    )
    config: TrainingConfig = image_model_training_config(
        TICKET31_RESNET18_VARIANT,
        preprocessing_config=integration_manifest.preprocess_config,
        packing_config=integration_manifest.packing_config,
        pooling_id=ANATOMICAL_POOLING_ID,
    )
    _validate_frozen_execution_policy(plan, config)
    image_supervision, selected_uids, cohort_evidence = _build_image_scoped_supervision(
        fold_manifest,
        integration_study_uids=integration_manifest.study_uids,
        ticket30_artifact=args.ticket30_artifact,
        plan=plan,
    )
    atomic_json_write(args.output_root / "cohort.json", cohort_evidence)

    preflight = run_cpu_preflight(
        output_path=args.preflight_evidence,
        require_isolated_python=True,
        require_slurm=True,
        required_slurm_resources=_RESOURCE_REQUIREMENTS,
    )
    require_cpu_preflight(preflight, require_slurm=True, require_isolated_python=True)
    if not preflight.passed:
        raise PreflightError("Ticket 34 CPU preflight did not pass")

    loaded = tuple(
        iter_real_studies(
            integration_manifest,
            args.source,
            require_complete_series=True,
            study_uids=selected_uids,
        )
    )
    actual_loaded = tuple(item.study.study_instance_uid for item in loaded)
    expected_loaded = tuple(
        item.study_instance_uid
        for item in integration_manifest.studies
        if item.study_instance_uid in set(selected_uids)
    )
    if actual_loaded != expected_loaded:
        raise Ticket34Error("the real image loader did not return the exact planned study cohort")
    studies = {item.study.study_instance_uid: item.study for item in loaded}

    series_cache = SeriesCache(args.output_root / "series-cache")
    reference_feature_cache_root = _TICKET32_FEATURE_CACHE_ROOT
    report_feature_cache_root = args.output_root / "frozen-feature-cache/report-supervision"
    common_training_args = {
        "config": config,
        "source_version": integration_manifest.source_version,
        "source_fingerprint": integration_manifest.archive_fingerprint,
        "integration_manifest_checksum": integration_manifest.manifest_checksum,
        "series_cache": series_cache,
        "ticket34_additive_design": True,
        "preflight": cast(CpuPreflightResult, preflight),
        "require_preflight": True,
        "requested_resources": _write_run_resources(),
        "pretrained_weights_path": args.weights,
        "resource_approval_path": args.resource_approval,
    }
    reference_result: TrainingResult = train_one_fold(
        studies,
        fold_manifest,
        0,
        args.output_root / "reference-only",
        feature_cache_root=reference_feature_cache_root,
        read_only_feature_cache=True,
        **common_training_args,
    )
    reference_cache = json.loads(
        (args.output_root / "reference-only/configuration.json").read_text(encoding="utf-8")
    ).get("feature_cache_execution", {})
    if (
        not isinstance(reference_cache, Mapping)
        or reference_cache.get("identity_fingerprint") != _TICKET32_FEATURE_CACHE_IDENTITY
        or reference_cache.get("read_only") is not True
        or reference_cache.get("entries") != 58
        or reference_cache.get("cache_hits") != 58
        or reference_cache.get("materialized_entries") != 0
    ):
        raise Ticket34Error(
            "the reference arm did not reuse all 58 exact Ticket 32 cache entries read-only"
        )

    report_result: TrainingResult = train_one_fold(
        studies,
        fold_manifest,
        0,
        args.output_root / "report-supervision",
        supervision=image_supervision,
        feature_cache_root=report_feature_cache_root,
        fallback_feature_cache_root=_TICKET32_FEATURE_CACHE_ROOT,
        **common_training_args,
    )
    if (
        reference_result.successful_optimizer_steps != config.successful_optimizer_steps
        or report_result.successful_optimizer_steps != config.successful_optimizer_steps
    ):
        raise Ticket34Error("a Ticket 34 arm did not complete the frozen fixed-step budget")

    report_cache = json.loads(
        (args.output_root / "report-supervision/configuration.json").read_text(encoding="utf-8")
    ).get("feature_cache_execution", {})
    if (
        not isinstance(reference_cache, Mapping)
        or not isinstance(report_cache, Mapping)
        or reference_cache.get("identity_fingerprint") == report_cache.get("identity_fingerprint")
        or report_cache.get("read_only") is not False
        or report_cache.get("fallback_cache_imports") != 58
        or report_cache.get("materialized_entries") != 61
    ):
        raise Ticket34Error(
            "Ticket 34 arms did not use the exact read-only Ticket 32 cache for all shared studies"
        )
    comparison = compare_supervision_pilot(
        args.output_root / "reference-only",
        args.output_root / "report-supervision",
        fold_manifest,
        expected_ticket30_artifact_sha256=TICKET30_AUDITED_ARTIFACT_SHA256,
        image_study_uids=integration_manifest.study_uids,
        bootstrap_resamples=2000,
        bootstrap_seed=20260925,
        confidence_level=0.95,
    )
    comparison["approval_scope"] = {
        "scope": approval["scope"],
        "approval_sha256": _sha256_file(args.resource_approval),
        "rights_status": approval["rights_status"],
        "external_spend": approval["external_spend"],
        "release_eligibility": approval.get("release_eligibility"),
    }
    comparison["feature_cache_reuse"] = {
        "source_ticket": 32,
        "source_root": str(_TICKET32_FEATURE_CACHE_ROOT),
        "source_identity_fingerprint": _TICKET32_FEATURE_CACHE_IDENTITY,
        "source_read_only": True,
        "reference_arm_exact_cache_hits": reference_cache.get("cache_hits"),
        "report_arm_fallback_cache_imports": report_cache.get("fallback_cache_imports"),
        "report_arm_encoder_computations": report_cache.get("materialized_entries"),
    }
    atomic_json_write(args.output_root / "comparison.json", comparison)
    return {
        "status": "completed",
        "started_monotonic": started_at,
        "completed_at_unix_seconds": time.time(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "fold": 0,
        "resource_scope": TICKET34_RESOURCE_SCOPE,
        "reference_run_id": reference_result.run_id,
        "report_run_id": report_result.run_id,
        "reference_optimizer_steps": reference_result.successful_optimizer_steps,
        "report_optimizer_steps": report_result.successful_optimizer_steps,
        "comparison_path": str(args.output_root / "comparison.json"),
        "cohort_path": str(args.output_root / "cohort.json"),
        "preflight_path": str(args.preflight_evidence),
        "feature_cache_roots": {
            "reference_only": str(reference_feature_cache_root),
            "report_supervision": str(report_feature_cache_root),
            "ticket32_read_only_source": str(_TICKET32_FEATURE_CACHE_ROOT),
        },
    }


def _execution_cost_record(
    status: str,
    *,
    started_at: float,
    started_usage: resource.struct_rusage,
    detail: Mapping[str, object] | None = None,
) -> dict[str, object]:
    ended_usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "schema_version": "rsna-ticket34-execution-cost-v1",
        "status": status,
        "fold": 0,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "wall_seconds": time.perf_counter() - started_at,
        "cpu_user_seconds": ended_usage.ru_utime - started_usage.ru_utime,
        "cpu_system_seconds": ended_usage.ru_stime - started_usage.ru_stime,
        "peak_rss_kb": ended_usage.ru_maxrss,
        "detail": dict(detail or {}),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        approval = _validate_resource_approval(args.resource_approval)
    except (PretrainedResourceError, OSError) as exc:
        print(f"Ticket 34 resource gate: {exc}", file=sys.stderr)
        return 2

    started_at = time.perf_counter()
    started_usage = resource.getrusage(resource.RUSAGE_SELF)
    try:
        detail = _run_pilot(args, approval=approval, started_at=started_at)
    except Exception as exc:
        record = _execution_cost_record(
            "failed",
            started_at=started_at,
            started_usage=started_usage,
            detail={"error_type": type(exc).__name__, "error": str(exc)},
        )
        try:
            atomic_json_write(args.output_root / "execution.json", record)
        except OSError:
            pass
        print(f"Ticket 34 run failed: {exc}", file=sys.stderr)
        return 2
    record = _execution_cost_record(
        "completed",
        started_at=started_at,
        started_usage=started_usage,
        detail=detail,
    )
    atomic_json_write(args.output_root / "execution.json", record)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
