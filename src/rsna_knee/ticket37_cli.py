"""Offline command line for the bounded ticket-37 Qwen3 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Mapping, cast

from .ticket37 import (
    Qwen3TransformersGenerator,
    Ticket37Error,
    compare_ticket37_admissions_with_ticket30,
    download_qwen3_snapshot,
    load_ticket37_approval,
    load_ticket37_inputs,
    make_qwen3_snapshot_manifest,
    plan_ticket37_sample,
    run_ticket37_pilot,
    sha256_file,
    ticket37_sample_plan_artifacts,
    verify_qwen3_snapshot,
    write_ticket37_approval,
    write_ticket37_json,
)


def _read_json(path: str | Path, description: str) -> Mapping[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Ticket37Error(f"could not read {description}") from exc
    if not isinstance(payload, Mapping):
        raise Ticket37Error(f"{description} must be a JSON object")
    return cast(Mapping[str, object], payload)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _outside_repository(path: str | Path, description: str) -> Path:
    resolved = Path(path).resolve()
    if resolved == _repo_root() or _repo_root() in resolved.parents:
        raise Ticket37Error(f"{description} must be outside the RSNA repository")
    return resolved


def _validate_source_identities(
    *,
    manifest: object,
    corpus: object,
    audit_public_path: str | Path,
    policy: Mapping[str, object],
    ticket30_private_path: str | Path,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    audit = _read_json(audit_public_path, "ticket-30 public audit")
    provenance = audit.get("provenance")
    private = _read_json(ticket30_private_path, "ticket-30 private admissions")
    lineage = private.get("lineage")
    fold_identity = lineage.get("fold_identity") if isinstance(lineage, Mapping) else None
    existing_policy = private.get("policy")
    if not isinstance(provenance, Mapping) or not isinstance(fold_identity, Mapping):
        raise Ticket37Error("ticket-30 source identity record is incomplete")
    if (
        getattr(manifest, "manifest_checksum", None) != provenance.get("fold_manifest_checksum")
        or getattr(manifest, "manifest_checksum", None) != fold_identity.get("fold_manifest_checksum")
    ):
        raise Ticket37Error("ticket-37 fold manifest differs from the audited version-2 manifest")
    if (
        getattr(corpus, "record_set_sha256", None) != provenance.get("report_set_sha256")
        or getattr(corpus, "source_version", None) != provenance.get("report_source_version")
        or getattr(corpus, "record_set_sha256", None) != fold_identity.get("report_set_sha256")
    ):
        raise Ticket37Error("ticket-37 report corpus differs from the audited local source")
    if (
        not isinstance(existing_policy, Mapping)
        or existing_policy.get("policy_sha256") != policy.get("policy_sha256")
    ):
        raise Ticket37Error("ticket-37 admission policy differs from ticket 30")
    return audit, private


def _current_peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports ru_maxrss in KiB; the cluster CPU nodes run Linux.
    return int(usage.ru_maxrss * 1024)


def _cmd_approve(args: argparse.Namespace) -> int:
    output = _outside_repository(args.output, "ticket-37 approval artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    path = write_ticket37_approval(
        output,
        approval_reference=args.approval_reference,
        approved_by=args.approved_by,
        approved_on=args.approved_on,
    )
    print(json.dumps({"status": "approved-for-bounded-pilot", "artifact": str(path)}))
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    approval = load_ticket37_approval(args.approval)
    del approval
    target = _outside_repository(args.output, "Qwen3 model snapshot")
    download_qwen3_snapshot(target)
    manifest = make_qwen3_snapshot_manifest(target)
    print(
        json.dumps(
            {
                "model_id": manifest["model_id"],
                "model_revision": manifest["model_revision"],
                "snapshot_sha256": manifest["snapshot_sha256"],
                "total_size_bytes": manifest["total_size_bytes"],
            }
        )
    )
    return 0


def _cmd_pilot(args: argparse.Namespace) -> int:
    approval = load_ticket37_approval(args.approval)
    output_root = _outside_repository(args.output_dir, "ticket-37 pilot output")
    private_root = output_root / "private"
    public_root = output_root / "public"
    private_root.mkdir(parents=True, exist_ok=True)
    public_root.mkdir(parents=True, exist_ok=True)

    manifest, corpus = load_ticket37_inputs(
        fold_manifest_path=args.fold_manifest,
        reports_path=args.reports,
    )
    policy = _read_json(args.frozen_policy, "frozen ticket-30 admission policy")
    audit_public, prior_private = _validate_source_identities(
        manifest=manifest,
        corpus=corpus,
        audit_public_path=args.ticket30_public_audit,
        policy=policy,
        ticket30_private_path=args.ticket30_private_admissions,
    )
    del audit_public

    model_root = _outside_repository(args.model_dir, "Qwen3 model directory")
    expected_manifest = _read_json(
        model_root / "ticket37-model-manifest.json",
        "pinned Qwen3 local snapshot manifest",
    )
    verified_manifest = verify_qwen3_snapshot(model_root, expected_manifest)
    sample = plan_ticket37_sample(
        manifest=manifest,
        corpus=corpus,
        report_only_per_fold=args.report_only_per_fold,
    )
    plan_private, plan_public = ticket37_sample_plan_artifacts(
        manifest=manifest,
        corpus=corpus,
        sample=sample,
        model_manifest=verified_manifest,
        approval=approval,
    )
    plan_private_path = write_ticket37_json(private_root / "run-plan.json", plan_private)
    write_ticket37_json(public_root / "run-plan.json", plan_public, mode=0o644)

    # The sample and exact source identities are immutable on disk before the
    # first report is passed to the model.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    load_started = time.perf_counter_ns()
    generator = Qwen3TransformersGenerator.from_local_snapshot(model_root, verified_manifest)
    model_load_elapsed_ns = time.perf_counter_ns() - load_started
    result = run_ticket37_pilot(
        manifest=manifest,
        corpus=corpus,
        sample=sample,
        generator=generator,
        model_manifest=verified_manifest,
        frozen_policy=policy,
        max_elapsed_seconds=args.max_elapsed_seconds,
        model_load_elapsed_ns=model_load_elapsed_ns,
        peak_rss_bytes=_current_peak_rss_bytes(),
    )
    # Include inference and serialization buffers in the process high-water mark.
    peak_rss = _current_peak_rss_bytes()
    cast(dict[str, object], result.private["runtime"])["process_peak_rss_bytes"] = peak_rss
    cast(dict[str, object], result.public["runtime"])["process_peak_rss_bytes"] = peak_rss
    result.private["approval_fingerprint"] = plan_private["approval_fingerprint"]
    result.private["run_plan_sha256"] = sha256_file(plan_private_path)
    result.private["ticket30_admission_artifact_sha256"] = sha256_file(
        args.ticket30_private_admissions
    )
    result.public["run_plan_sha256"] = result.private["run_plan_sha256"]
    result.public["ticket30_public_audit_sha256"] = sha256_file(args.ticket30_public_audit)

    if isinstance(result.private.get("admission"), Mapping):
        comparison = compare_ticket37_admissions_with_ticket30(
            ticket37_private=cast(Mapping[str, object], result.private["admission"]),
            ticket30_private=prior_private,
        )
        result.private["existing_output_comparison"] = comparison
        result.public["existing_output_comparison"] = comparison

    private_path = write_ticket37_json(private_root / "pilot-results.json", result.private)
    public_path = write_ticket37_json(public_root / "pilot-summary.json", result.public, mode=0o644)
    print(
        json.dumps(
            {
                "status": result.public["status"],
                "processed_report_count": result.public["processed_report_count"],
                "expected_report_count": result.public["expected_report_count"],
                "branch_disposition": result.public["branch_disposition"],
                "private_artifact": str(private_path),
                "public_artifact": str(public_path),
            }
        )
    )
    return 0 if result.public["status"] == "completed" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded ticket-37 Qwen3 report pilot")
    commands = parser.add_subparsers(dest="command", required=True)

    approve = commands.add_parser("approve", help="write the exact bounded-pilot approval record")
    approve.add_argument("--output", required=True)
    approve.add_argument("--approval-reference", required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--approved-on", required=True)
    approve.set_defaults(handler=_cmd_approve)

    download = commands.add_parser("download", help="download only the pinned public Qwen3 files")
    download.add_argument("--approval", required=True)
    download.add_argument("--output", required=True)
    download.set_defaults(handler=_cmd_download)

    pilot = commands.add_parser("pilot", help="run the approved local CPU pilot through Slurm")
    pilot.add_argument("--approval", required=True)
    pilot.add_argument("--model-dir", required=True)
    pilot.add_argument("--fold-manifest", required=True)
    pilot.add_argument("--reports", required=True)
    pilot.add_argument("--frozen-policy", required=True)
    pilot.add_argument("--ticket30-public-audit", required=True)
    pilot.add_argument("--ticket30-private-admissions", required=True)
    pilot.add_argument("--output-dir", required=True)
    pilot.add_argument("--report-only-per-fold", type=int, default=2)
    pilot.add_argument("--max-elapsed-seconds", type=float, default=19_800)
    pilot.set_defaults(handler=_cmd_pilot)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return cast(int, args.handler(args))
    except Ticket37Error as exc:
        print(f"ticket37: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
