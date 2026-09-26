"""Freeze Ticket 35 membership or run one approved CPU materialization batch."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Mapping, Sequence, cast

from .archive import atomic_json_write
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
    _approve_cpu_batch(
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
        require_slurm=True,
    )
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

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, Ticket35Error, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
