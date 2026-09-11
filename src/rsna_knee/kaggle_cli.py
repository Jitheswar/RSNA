"""Prepare, execute, and record the approval-gated ticket-12 Kaggle path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .kaggle import (
    DEFAULT_KAGGLE_ADAPTER_PATH,
    load_kaggle_delivery,
    prepare_kaggle_delivery,
    promote_kaggle_candidate,
    record_kaggle_verification,
    run_kaggle_baseline,
)


def _uid_list(path: Path) -> tuple[str, ...]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read expected study list {path}") from exc
    if isinstance(value, dict):
        value = value.get("study_uids")
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError("expected study list must be a JSON array or {study_uids: [...]}")
    if len(set(value)) != len(value):
        raise ValueError("expected study list contains duplicate identifiers")
    return tuple(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="write an immutable owner-approval request")
    prepare.add_argument("--candidate", type=Path, required=True)
    prepare.add_argument("--local-evidence", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--notebook", type=Path, default=None)
    prepare.add_argument("--adapter", type=Path, default=DEFAULT_KAGGLE_ADAPTER_PATH)
    prepare.add_argument("--local-submission", type=Path, default=None)
    prepare.add_argument("--expected-uids", type=Path, default=None)
    prepare.add_argument("--request-id", default=None)

    run = subparsers.add_parser("run", help="run the shared adapter in a Kaggle notebook")
    run.add_argument("--candidate", type=Path, required=True)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--evidence", type=Path, required=True)
    run.add_argument("--private-evidence", type=Path, default=None)
    run.add_argument("--request-checksum", default=None)
    run.add_argument("--notebook-sha256", default=None)
    run.add_argument("--adapter-sha256", default=None)
    run.add_argument("--request-root", type=Path, default=None)
    run.add_argument("--approval", type=Path, default=None)
    run.add_argument("--notebook-path", type=Path, default=None)
    run.add_argument("--adapter-path", type=Path, default=None)
    run.add_argument("--notebook-started-at", type=float, default=None)
    run.add_argument("--startup-seconds", type=float, default=None)
    run.add_argument("--imports-seconds", type=float, default=None)
    run.add_argument("--device", default="cpu")
    run.add_argument("--allow-local-smoke", action="store_true")

    verify = subparsers.add_parser("verify", help="record owner-approved accepted Kaggle evidence")
    verify.add_argument("--request", type=Path, required=True)
    verify.add_argument("--candidate", type=Path, required=True)
    verify.add_argument("--approval", type=Path, required=True)
    verify.add_argument("--execution", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--submission", type=Path, default=None)
    verify.add_argument("--private-execution", type=Path, default=None)
    verify.add_argument("--expected-uids", type=Path, default=None)

    promote = subparsers.add_parser("promote", help="append a Kaggle-verified promotion pointer")
    promote.add_argument("--request", type=Path, required=True)
    promote.add_argument("--candidate", type=Path, required=True)
    promote.add_argument("--verification", type=Path, required=True)
    promote.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            expected = _uid_list(args.expected_uids) if args.expected_uids is not None else None
            request = prepare_kaggle_delivery(
                args.candidate,
                args.local_evidence,
                args.output,
                delivery_notebook_path=args.notebook,
                delivery_adapter_path=args.adapter,
                local_submission_path=args.local_submission,
                expected_study_uids=expected,
                request_id=args.request_id,
            )
            print(json.dumps(dict(request.payload), indent=2, sort_keys=True))
            return 0
        if args.command == "run":
            stage_timings = None
            if args.startup_seconds is not None or args.imports_seconds is not None:
                if args.startup_seconds is None or args.imports_seconds is None:
                    raise ValueError("--startup-seconds and --imports-seconds must be supplied together")
                stage_timings = {
                    "startup_setup": args.startup_seconds,
                    "imports": args.imports_seconds,
                }
            inference = run_kaggle_baseline(
                args.candidate,
                args.input,
                args.output,
                evidence_path=args.evidence,
                private_evidence_path=args.private_evidence,
                delivery_request_checksum=args.request_checksum,
                delivery_notebook_sha256=args.notebook_sha256,
                delivery_adapter_sha256=args.adapter_sha256,
                delivery_request_root=args.request_root,
                owner_approval_path=args.approval,
                delivery_notebook_path=args.notebook_path,
                delivery_adapter_path=args.adapter_path,
                notebook_stage_timings=stage_timings,
                notebook_started_at=args.notebook_started_at,
                device=args.device,
                allow_local_smoke=args.allow_local_smoke,
            )
            print(json.dumps(dict(inference.evidence), indent=2, sort_keys=True))
            return 0
        if args.command == "verify":
            expected = _uid_list(args.expected_uids) if args.expected_uids is not None else None
            verification = record_kaggle_verification(
                args.request,
                args.candidate,
                args.approval,
                args.execution,
                args.output,
                submission_path=args.submission,
                private_evidence_path=args.private_execution,
                expected_study_uids=expected,
            )
            print(json.dumps(dict(verification.payload), indent=2, sort_keys=True))
            return 0
        promotion = promote_kaggle_candidate(
            args.request,
            args.candidate,
            args.verification,
            args.output,
        )
        print(json.dumps(dict(promotion.payload), indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ticket-12 Kaggle path failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
