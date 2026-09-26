"""Freeze Ticket 35 membership or run one approved CPU materialization batch."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Mapping, Sequence, cast

import torch

from .archive import atomic_json_write
from .model import ModelConfig, StudyModel, restore_study_model
from .real_inference import IntegrationManifest
from .ticket35 import (
    Ticket35Error,
    build_ticket35_batch_integration_manifest,
    capture_ticket35_storage_ledger,
    estimate_planned_batch_storage,
    freeze_ticket35_membership,
    process_ticket35_studies,
    sha256_file,
    validate_ticket35_membership,
    write_ticket35_batch_evidence,
)


_TICKET35_INFERENCE_APPROVAL_SCHEMA = "rsna-ticket35-all-batches-inference-approval-v1"
_TICKET35_INFERENCE_SCOPE = "ticket35-all-batches-real-inference"


def _parse_root(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or re.fullmatch(r"[A-Za-z0-9._-]+", label) is None or not raw_path:
        raise argparse.ArgumentTypeError("storage roots must use LABEL=/absolute/path")
    path = Path(raw_path)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("storage root paths must be absolute")
    return label, path


def _read_membership(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Ticket35Error("could not read the frozen membership plan") from exc
    return validate_ticket35_membership(value)


def _approve_cpu_batch(
    path: Path,
    membership: Mapping[str, object],
    batch_index: int,
    *,
    inventory_path: Path,
    reference_integration_path: Path,
    source_path: Path,
    cache_root: Path,
) -> Mapping[str, object]:
    try:
        approval = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Ticket35Error("the exact Ticket 35 CPU approval record cannot be read") from exc
    if not isinstance(approval, dict):
        raise Ticket35Error("the exact Ticket 35 CPU approval record must be an object")
    resources = approval.get("resources")
    if not isinstance(resources, dict):
        raise Ticket35Error("Ticket 35 CPU approval is missing its exact resource request")
    storage_gate = approval.get("storage_gate")
    if not isinstance(storage_gate, dict):
        raise Ticket35Error("Ticket 35 CPU approval is missing its storage gate")
    batches = cast(Sequence[Mapping[str, object]], membership.get("batches", ()))
    if batch_index < 0 or batch_index >= len(batches):
        raise Ticket35Error("approved batch index is outside the frozen membership plan")
    planned_batch = batches[batch_index]
    expected_batch = {
        "report_only_studies": len(cast(Sequence[object], planned_batch.get("study_uids", ()))),
        "series": planned_batch.get("series_count"),
        "source_entries_and_effective_slices": planned_batch.get("effective_frame_count"),
        "uncompressed_source_bytes": planned_batch.get("uncompressed_source_bytes"),
        "compressed_source_bytes": planned_batch.get("compressed_source_bytes"),
        "224px_float32_pixel_cache_bytes": planned_batch.get("pixel_cache_bytes"),
    }
    batch_summary = approval.get("batch_summary")
    if not isinstance(batch_summary, dict) or any(
        batch_summary.get(key) != value for key, value in expected_batch.items()
    ):
        raise Ticket35Error("CPU approval batch summary differs from the frozen batch")
    expected_paths = {
        "inventory_path": str(inventory_path.resolve()),
        "reference_integration_manifest_path": str(reference_integration_path.resolve()),
        "source_archive_path": str(source_path.resolve()),
        "cache_root": str(cache_root.resolve()),
    }
    if (
        approval.get("schema_version") != "rsna-ticket35-cpu-preparation-approval-v1"
        or approval.get("decision") != "approved"
        or approval.get("approval_status") != "approved_by_project_owner"
        or approval.get("membership_checksum") != membership.get("manifest_checksum")
        or approval.get("batch_indices") != [batch_index]
        or approval.get("scope") != "ticket35-image-preparation-single-batch"
        or approval.get("model_use") is not False
        or approval.get("feature_extraction_or_prediction") is not False
        or approval.get("training") is not False
        or approval.get("external_spend_usd") != 0
        or approval.get("gpu_count") != 0
        or any(approval.get(name) != value for name, value in expected_paths.items())
        or approval.get("source_version")
        != cast(Mapping[str, object], membership["source_identity"]).get("source_version")
        or approval.get("archive_fingerprint")
        != cast(Mapping[str, object], membership["source_identity"]).get("archive_fingerprint")
        or approval.get("archive_size_bytes")
        != cast(Mapping[str, object], membership["source_identity"]).get("archive_size_bytes")
        or storage_gate.get("soft_budget_bytes") != 2_000_000_000_000
    ):
        raise Ticket35Error("CPU approval does not match this exact Ticket 35 preparation request")
    expected_resources = {
        "partition": "cluster1-cpu",
        "nodes": 1,
        "cpus_per_task": 16,
        "memory_gib": 64,
        "walltime_seconds": 21600,
        "maximum_concurrent_jobs": 1,
        "gpu_count": 0,
    }
    if any(resources.get(name) != value for name, value in expected_resources.items()):
        raise Ticket35Error("CPU approval does not match the frozen 16-CPU, 64-GiB, six-hour request")
    if not isinstance(approval.get("approved_by"), str) or not approval["approved_by"].strip():
        raise Ticket35Error("approved CPU scope must record who approved it")
    if not isinstance(approval.get("approved_at"), str) or not approval["approved_at"].strip():
        raise Ticket35Error("approved CPU scope must record when it was approved")
    return cast(Mapping[str, object], approval)


def _approve_inference_batch(
    path: Path,
    membership: Mapping[str, object],
    membership_path: Path,
    batch_index: int,
    *,
    inventory_path: Path,
    reference_integration_path: Path,
    source_path: Path,
    cache_root: Path,
    checkpoint_path: Path,
    storage_roots: Sequence[tuple[str, Path]],
) -> Mapping[str, object]:
    try:
        approval = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Ticket35Error("the exact Ticket 35 inference approval record cannot be read") from exc
    if not isinstance(approval, dict):
        raise Ticket35Error("the exact Ticket 35 inference approval record must be an object")
    batches = cast(Sequence[Mapping[str, object]], membership.get("batches", ()))
    if batch_index < 0 or batch_index >= len(batches):
        raise Ticket35Error("approved batch index is outside the frozen membership plan")
    expected_paths = {
        "membership_path": str(membership_path.resolve()),
        "inventory_path": str(inventory_path.resolve()),
        "reference_integration_manifest_path": str(reference_integration_path.resolve()),
        "source_archive_path": str(source_path.resolve()),
        "cache_root": str(cache_root.resolve()),
    }
    expected_roots = [
        {"label": label, "path": str(root.resolve())}
        for label, root in storage_roots
    ]
    source_identity = cast(Mapping[str, object], membership["source_identity"])
    resources = approval.get("resources")
    storage_gate = approval.get("storage_gate")
    model = approval.get("model")
    if not isinstance(resources, dict) or not isinstance(storage_gate, dict) or not isinstance(model, dict):
        raise Ticket35Error("Ticket 35 inference approval is missing exact resource or model details")
    if (
        approval.get("schema_version") != _TICKET35_INFERENCE_APPROVAL_SCHEMA
        or approval.get("decision") != "approved"
        or approval.get("approval_status") != "approved_by_project_owner"
        or approval.get("scope") != _TICKET35_INFERENCE_SCOPE
        or approval.get("membership_checksum") != membership.get("manifest_checksum")
        or approval.get("membership_file_sha256") != sha256_file(membership_path)
        or approval.get("batch_indices") != list(range(len(batches)))
        or approval.get("batch_count") != len(batches)
        or approval.get("image_ready_study_count")
        != cast(Mapping[str, object], membership["cohort"]).get("image_ready_report_study_count")
        or approval.get("series_count")
        != cast(Mapping[str, object], membership["cohort"]).get("image_ready_series_count")
        or approval.get("effective_frame_count")
        != cast(Mapping[str, object], membership["cohort"]).get("image_ready_effective_frame_count")
        or any(approval.get(name) != value for name, value in expected_paths.items())
        or approval.get("storage_roots") != expected_roots
        or approval.get("source_version") != source_identity.get("source_version")
        or approval.get("archive_fingerprint") != source_identity.get("archive_fingerprint")
        or approval.get("archive_size_bytes") != source_identity.get("archive_size_bytes")
        or approval.get("representation_id") != "ticket31-resnet18-224-whole-field-v1"
        or approval.get("soft_budget_bytes") != 2_000_000_000_000
        or approval.get("external_spend_usd") != 0
        or approval.get("gpu_count") != 0
        or approval.get("model_use") is not True
        or approval.get("feature_extraction_or_prediction") is not True
        or approval.get("training") is not False
        or approval.get("scientific_evaluation") is not False
        or approval.get("candidate_promotion") is not False
        or approval.get("private_predictions_only") is not True
        or storage_gate.get("fresh_campaign_wide_ledger_in_same_job") is not True
        or storage_gate.get("archive_counted_once") is not True
        or storage_gate.get("soft_budget_bytes") != 2_000_000_000_000
        or storage_gate.get("stop_before_dicom_access_if_projection_exceeds_budget") is not True
    ):
        raise Ticket35Error("inference approval does not match this exact Ticket 35 batch request")
    expected_resources = {
        "partition": "cluster1-cpu",
        "nodes": 1,
        "tasks": 1,
        "cpus_per_task": 16,
        "memory_gib": 64,
        "walltime_seconds": 21600,
        "maximum_concurrent_jobs": 1,
        "gpu_count": 0,
    }
    if any(resources.get(name) != value for name, value in expected_resources.items()):
        raise Ticket35Error("inference approval differs from the frozen CPU-only resource envelope")
    expected_model = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": "17220c47d15e786e870eb0d46aa088314973b7929094ede697585c148211e872",
        "checkpoint_bytes": 45615081,
        "checkpoint_kind": "final_selected",
        "checkpoint_config_fingerprint": "bb8ce4047dac2589959f717005eabca4417a390df3fa1e089dc17cd1b7c950dc",
        "checkpoint_run_record_path": str(
            checkpoint_path.resolve().parents[1] / "run.json"
        ),
        "checkpoint_run_record_sha256": "7f5a7bf3265420542fe011c351388f877c1265bdb482e03b83fe5e4907914aea",
        "fold": 0,
        "device": "cpu",
        "model_id": "resnet18-frozen-slice-features-v1",
        "encoder_id": "torchvision-resnet18-imagenet1k-v1",
        "encoder_weights_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
        "representation_id": "ticket31-resnet18-224-whole-field-v1",
        "image_size": [224, 224],
        "pooling_id": "plane-stratified-series-mean-v1",
    }
    if any(model.get(name) != value for name, value in expected_model.items()):
        raise Ticket35Error("inference approval does not bind the selected checkpoint and representation")
    if not isinstance(approval.get("approved_by"), str) or not approval["approved_by"].strip():
        raise Ticket35Error("approved inference scope must record who approved it")
    if not isinstance(approval.get("approved_at"), str) or not approval["approved_at"].strip():
        raise Ticket35Error("approved inference scope must record when it was approved")
    return cast(Mapping[str, object], approval)


def _check_inference_allocation(batch_index: int) -> None:
    expected = {
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
        "SLURM_JOB_PARTITION": os.environ.get("SLURM_JOB_PARTITION"),
        "SLURM_JOB_NUM_NODES": os.environ.get("SLURM_JOB_NUM_NODES"),
        "SLURM_NTASKS": os.environ.get("SLURM_NTASKS"),
        "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK"),
    }
    if not expected["SLURM_JOB_ID"] or expected["SLURM_JOB_PARTITION"] != "cluster1-cpu":
        raise Ticket35Error("real Ticket 35 inference must run inside its cluster1-cpu Slurm allocation")
    if (
        expected["SLURM_JOB_NUM_NODES"] != "1"
        or expected["SLURM_NTASKS"] != "1"
        or expected["SLURM_CPUS_PER_TASK"] != "16"
        or os.environ.get("SLURM_MEM_PER_NODE", "").strip().upper() not in {"64G", "65536"}
        or os.environ.get("SLURM_TIMELIMIT", "").strip() not in {"360", "06:00", "06:00:00"}
    ):
        raise Ticket35Error("Slurm allocation differs from the approved Ticket 35 CPU-only envelope")
    array_index = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_index is not None and array_index != str(batch_index):
        raise Ticket35Error("batch index does not match the Slurm array task index")


def _load_approved_model(
    checkpoint_path: Path,
    approval: Mapping[str, object],
    manifest: IntegrationManifest,
) -> tuple[StudyModel, Mapping[str, object]]:
    model_approval = cast(Mapping[str, object], approval["model"])
    try:
        checkpoint_bytes = checkpoint_path.stat().st_size
    except OSError as exc:
        raise Ticket35Error("approved model checkpoint is not available") from exc
    if (
        checkpoint_bytes != model_approval["checkpoint_bytes"]
        or sha256_file(checkpoint_path) != model_approval["checkpoint_sha256"]
    ):
        raise Ticket35Error("model checkpoint size or checksum differs from the approved artifact")
    run_record_path = Path(cast(str, model_approval["checkpoint_run_record_path"]))
    if sha256_file(run_record_path) != model_approval["checkpoint_run_record_sha256"]:
        raise Ticket35Error("model run record checksum differs from the approved artifact")
    try:
        # The project-produced checkpoint is loaded only after its approved SHA-256 matches.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise Ticket35Error("approved model checkpoint could not be loaded") from exc
    if not isinstance(checkpoint, dict):
        raise Ticket35Error("approved model checkpoint has an unsupported structure")
    if (
        checkpoint.get("schema_version") != "rsna-knee-checkpoint-v2"
        or checkpoint.get("fold") != model_approval["fold"]
        or checkpoint.get("checkpoint_kind") != model_approval["checkpoint_kind"]
        or checkpoint.get("config_fingerprint") != model_approval["checkpoint_config_fingerprint"]
    ):
        raise Ticket35Error("checkpoint metadata differs from the approved model identity")
    config_value = checkpoint.get("config")
    if not isinstance(config_value, Mapping):
        raise Ticket35Error("approved checkpoint is missing its training configuration")
    raw_model_config = config_value.get("model")
    if not isinstance(raw_model_config, Mapping):
        raise Ticket35Error("approved checkpoint is missing its model configuration")
    expected_model_config = {
        "schema_version": "rsna-knee-model-config-v1",
        "model_id": model_approval["model_id"],
        "encoder_id": model_approval["encoder_id"],
        "encoder_weights_sha256": model_approval["encoder_weights_sha256"],
        "pooling_id": model_approval["pooling_id"],
        "image_size": model_approval["image_size"],
        "input_channels": 1,
        "num_targets": 12,
        "head_hidden_size": 32,
        "channels": [64, 128, 256],
    }
    if (
        dict(raw_model_config) != expected_model_config
        or model_approval.get("model_config") != expected_model_config
    ):
        raise Ticket35Error("checkpoint model configuration differs from the approved architecture")
    preprocess_value = config_value.get("preprocessing")
    packing_value = config_value.get("packing")
    if preprocess_value != manifest.preprocess_config.cache_config():
        raise Ticket35Error("checkpoint preprocessing differs from this Ticket 35 batch manifest")
    if packing_value != manifest.packing_config.to_dict():
        raise Ticket35Error("checkpoint packing differs from this Ticket 35 batch manifest")
    state_dict = checkpoint.get("model_state_dict")
    try:
        model = restore_study_model(ModelConfig.from_dict(raw_model_config), state_dict)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Ticket35Error("approved checkpoint could not reconstruct its study model") from exc
    model.to(torch.device("cpu"))
    model.eval()
    return model, {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": model_approval["checkpoint_sha256"],
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_kind": model_approval["checkpoint_kind"],
        "checkpoint_config_fingerprint": model_approval["checkpoint_config_fingerprint"],
        "fold": model_approval["fold"],
        "device": "cpu",
        "model_id": model_approval["model_id"],
        "encoder_id": model_approval["encoder_id"],
        "encoder_weights_sha256": model_approval["encoder_weights_sha256"],
        "representation_id": model_approval["representation_id"],
        "pipeline_verification_only": True,
        "scientific_evaluation": False,
        "candidate_promotion": False,
    }


def _freeze(args: argparse.Namespace) -> int:
    manifest = freeze_ticket35_membership(
        args.fold_manifest,
        args.inventory,
        args.integration_manifest,
        args.label_artifact,
        args.output,
        expected_label_artifact_sha256=args.expected_label_sha256,
        batch_size=args.batch_size,
    )
    cohort = cast(Mapping[str, object], manifest["cohort"])
    print(
        json.dumps(
            {
                "status": "frozen",
                "manifest_checksum": manifest["manifest_checksum"],
                "admitted_report_studies": cohort["admitted_report_study_count"],
                "image_ready_report_studies": cohort["image_ready_report_study_count"],
                "additional_studies": cohort["additional_vs_existing_manifest_count"],
                "batches": len(cast(list[object], manifest["batches"])),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


def _prepare(args: argparse.Namespace) -> int:
    membership = _read_membership(args.membership)
    inference = args.command == "prepare-infer"
    if inference:
        _check_inference_allocation(args.batch_index)
        approval = _approve_inference_batch(
            args.approval,
            membership,
            args.membership,
            args.batch_index,
            inventory_path=args.inventory,
            reference_integration_path=args.reference_integration_manifest,
            source_path=args.source_archive,
            cache_root=args.cache_root,
            checkpoint_path=args.model_checkpoint,
            storage_roots=args.storage_root,
        )
    else:
        approval = _approve_cpu_batch(
            args.approval,
            membership,
            args.batch_index,
            inventory_path=args.inventory,
            reference_integration_path=args.reference_integration_manifest,
            source_path=args.source_archive,
            cache_root=args.cache_root,
        )
    ledger = capture_ticket35_storage_ledger(args.storage_root, args.source_archive)
    archive_bytes = int(args.source_archive.stat().st_size)
    if ledger.get("source_archive_bytes") != archive_bytes:
        raise Ticket35Error("measured archive bytes differ from the current source archive file")
    measured_value = ledger.get("measured_roots_logical_bytes")
    if not isinstance(measured_value, int) or isinstance(measured_value, bool):
        raise Ticket35Error("storage ledger has invalid measured root bytes")
    measured_bytes = measured_value
    estimate = estimate_planned_batch_storage(
        membership,
        args.batch_index,
        current_measured_bytes=measured_bytes,
        source_archive_bytes=archive_bytes,
    )
    ledger["membership_checksum"] = membership["manifest_checksum"]
    ledger["batch_index"] = args.batch_index
    ledger["batch_estimate"] = estimate
    atomic_json_write(args.ledger_output, ledger)
    args.ledger_output.chmod(0o600)
    selection_path = args.selection_output
    integration_manifest = build_ticket35_batch_integration_manifest(
        args.inventory,
        args.reference_integration_manifest,
        membership,
        args.batch_index,
        args.batch_manifest_output,
        current_measured_bytes=measured_bytes,
        source_archive_bytes=archive_bytes,
        selection_output_path=selection_path,
    )
    model: StudyModel | None = None
    model_provenance: Mapping[str, object] | None = None
    if inference:
        model, model_provenance = _load_approved_model(args.model_checkpoint, approval, integration_manifest)
    batches = cast(Sequence[Mapping[str, object]], membership["batches"])
    selected_batch = batches[args.batch_index]
    selected_uids_value = selected_batch.get("study_uids")
    if not isinstance(selected_uids_value, list) or not all(
        isinstance(uid, str) for uid in selected_uids_value
    ):
        raise Ticket35Error("frozen batch has invalid study identifiers")
    batch_uids = tuple(selected_uids_value)
    evidence = process_ticket35_studies(
        integration_manifest,
        args.source_archive,
        args.cache_root,
        study_uids=batch_uids,
        model=model,
        require_slurm=True,
    )
    if inference and evidence.get("prediction_count") != evidence.get("study_count"):
        raise Ticket35Error("model inference did not produce exactly one prediction per batch study")
    study_metrics = cast(Mapping[str, Mapping[str, object]], membership["study_metrics"])
    supervised_by_fold = {
        str(fold): sum(
            int(cast(Mapping[str, int], study_metrics[uid]["supervised_target_entries_by_fold"]).get(str(fold), 0))
            for uid in batch_uids
        )
        for fold in range(3)
    }
    evidence.update(
        {
            "ticket35_membership_checksum": membership["manifest_checksum"],
            "batch_index": args.batch_index,
            "storage_estimate": estimate,
            "storage_ledger_path": str(args.ledger_output),
            "storage_ledger_checksum": sha256_file(args.ledger_output),
            "integration_manifest_path": str(args.batch_manifest_output),
            "integration_manifest_checksum": integration_manifest.manifest_checksum,
            "selection_path": str(selection_path),
            "supervised_target_entry_count_by_fold": supervised_by_fold,
            "supervised_target_entry_count_across_folds": sum(supervised_by_fold.values()),
            "approval_record_path": str(args.approval.resolve()),
            "approval_record_sha256": sha256_file(args.approval),
            "model_provenance": model_provenance,
        }
    )
    write_ticket35_batch_evidence(args.evidence_output, evidence)
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "batch_index": args.batch_index,
                "studies": evidence["study_count"],
                "series": evidence["series_count"],
                "slices": evidence["decoded_slice_count"],
                "elapsed_seconds": evidence["elapsed_seconds"],
                "cache_hits": cast(Mapping[str, object], evidence["cache"])["hits"],
                "cache_misses": cast(Mapping[str, object], evidence["cache"])["misses"],
                "predictions": evidence["prediction_count"],
                "projected_peak_bytes": estimate["projected_peak_bytes"],
                "evidence": str(args.evidence_output),
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="freeze fold-safe image membership")
    freeze.add_argument("--fold-manifest", type=Path, required=True)
    freeze.add_argument("--inventory", type=Path, required=True)
    freeze.add_argument("--integration-manifest", type=Path, required=True)
    freeze.add_argument("--label-artifact", type=Path, required=True)
    freeze.add_argument("--expected-label-sha256", required=True)
    freeze.add_argument("--batch-size", type=int, default=32)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(handler=_freeze)

    prepare = commands.add_parser("prepare", help="run one specifically approved CPU batch")
    prepare.add_argument("--membership", type=Path, required=True)
    prepare.add_argument("--batch-index", type=int, required=True)
    prepare.add_argument("--approval", type=Path, required=True)
    prepare.add_argument("--inventory", type=Path, required=True)
    prepare.add_argument("--reference-integration-manifest", type=Path, required=True)
    prepare.add_argument("--source-archive", type=Path, required=True)
    prepare.add_argument("--cache-root", type=Path, required=True)
    prepare.add_argument("--storage-root", type=_parse_root, action="append", required=True)
    prepare.add_argument("--ledger-output", type=Path, required=True)
    prepare.add_argument("--selection-output", type=Path, required=True)
    prepare.add_argument("--batch-manifest-output", type=Path, required=True)
    prepare.add_argument("--evidence-output", type=Path, required=True)
    prepare.set_defaults(handler=_prepare)

    prepare_infer = commands.add_parser(
        "prepare-infer",
        help="run one exactly approved CPU batch with private real-model predictions",
    )
    prepare_infer.add_argument("--membership", type=Path, required=True)
    prepare_infer.add_argument("--batch-index", type=int, required=True)
    prepare_infer.add_argument("--approval", type=Path, required=True)
    prepare_infer.add_argument("--inventory", type=Path, required=True)
    prepare_infer.add_argument("--reference-integration-manifest", type=Path, required=True)
    prepare_infer.add_argument("--source-archive", type=Path, required=True)
    prepare_infer.add_argument("--cache-root", type=Path, required=True)
    prepare_infer.add_argument("--storage-root", type=_parse_root, action="append", required=True)
    prepare_infer.add_argument("--ledger-output", type=Path, required=True)
    prepare_infer.add_argument("--selection-output", type=Path, required=True)
    prepare_infer.add_argument("--batch-manifest-output", type=Path, required=True)
    prepare_infer.add_argument("--evidence-output", type=Path, required=True)
    prepare_infer.add_argument("--model-checkpoint", type=Path, required=True)
    prepare_infer.set_defaults(handler=_prepare)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, Ticket35Error, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
