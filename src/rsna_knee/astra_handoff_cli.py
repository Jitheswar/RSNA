"""Prepare and validate the manual Ticket 16 Astra Medium handoff.

This command never switches models or calls Codex.  The operator performs that
model switch manually, saves the exact response, and then uses ``record`` to
run the local validator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .astra_handoff import (
    ASTRA_BATCH_SIZE,
    ASTRA_OUTPUT_JSON_SCHEMA,
    AstraHandoffError,
    astra_medium_experiment_approval,
    build_astra_batch_plan,
    completed_astra_batch_ids,
    load_astra_approval,
    load_astra_batch_plan,
    load_astra_prompt,
    make_astra_prompt_artifact,
    pending_astra_batches,
    record_astra_output,
    write_astra_approval,
    write_astra_batch_plan,
    write_astra_prompt,
    write_astra_public_evidence,
)
from .folds import load_fold_manifest
from .report_extraction import load_report_corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    approval = subparsers.add_parser("write-approval", help="write the explicit Astra identity record")
    approval.add_argument("--output", type=Path, required=True)

    prompt = subparsers.add_parser("write-prompt", help="write an owner-supplied prompt artifact")
    prompt.add_argument("--prompt-file", type=Path, required=True)
    prompt.add_argument("--output", type=Path, required=True)
    prompt.add_argument("--prompt-id", default="ticket16-owner-supplied-prompt")
    prompt.add_argument("--prompt-version", default="owner-supplied-v1")
    prompt.add_argument("--approved-by")
    prompt.add_argument("--approval-reference")
    prompt.add_argument("--approved-on")

    prepare = subparsers.add_parser("prepare", help="prepare private batches; never invokes Astra")
    prepare.add_argument("--fold-manifest", type=Path, required=True)
    prepare.add_argument("--reports", type=Path, required=True)
    prepare.add_argument("--approval", type=Path, required=True)
    prepare.add_argument("--prompt", type=Path, required=True)
    prepare.add_argument("--public-plan", type=Path, required=True)
    prepare.add_argument("--private-dir", type=Path, required=True)
    prepare.add_argument("--batch-size", type=int, default=ASTRA_BATCH_SIZE)

    pending = subparsers.add_parser("pending", help="list batches not yet completed")
    pending.add_argument("--private-plan", type=Path, required=True)
    pending.add_argument("--private-dir", type=Path, required=True)

    record = subparsers.add_parser("record", help="save and validate one verbatim Astra response")
    record.add_argument("--private-plan", type=Path, required=True)
    record.add_argument("--private-dir", type=Path, required=True)
    record.add_argument("--batch-id", required=True)
    record.add_argument("--raw-output", type=Path, required=True)
    record.add_argument("--attempt", type=int, default=1)

    publish = subparsers.add_parser("publish", help="write text-free public handoff status")
    publish.add_argument("--private-plan", type=Path, required=True)
    publish.add_argument("--private-dir", type=Path, required=True)
    publish.add_argument("--public-output", type=Path, required=True)

    schema = subparsers.add_parser("schema", help="print the exact Astra output JSON schema")
    schema.add_argument("--output", type=Path)
    return parser


def _write_or_print(payload: object, path: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "write-approval":
            approval = astra_medium_experiment_approval()
            path = write_astra_approval(args.output, approval)
            print(json.dumps({"approval": str(path), "fingerprint": approval.fingerprint}, sort_keys=True))
            return 0

        if args.command == "write-prompt":
            prompt = make_astra_prompt_artifact(
                args.prompt_file.read_text(encoding="utf-8"),
                prompt_id=args.prompt_id,
                prompt_version=args.prompt_version,
            )
            approval_fields = (args.approved_by, args.approval_reference, args.approved_on)
            if any(value is not None for value in approval_fields):
                if not all(value is not None for value in approval_fields):
                    raise AstraHandoffError(
                        "--approved-by, --approval-reference and --approved-on must be supplied together"
                    )
                prompt = prompt.approve(
                    approved_by=args.approved_by,
                    approval_reference=args.approval_reference,
                    approved_on=args.approved_on,
                )
            path = write_astra_prompt(args.output, prompt)
            print(json.dumps({"prompt": str(path), "status": prompt.status, "fingerprint": prompt.fingerprint}, sort_keys=True))
            return 0

        if args.command == "prepare":
            approval = load_astra_approval(args.approval)
            prompt = load_astra_prompt(args.prompt)
            manifest = load_fold_manifest(args.fold_manifest)
            corpus = load_report_corpus(args.reports)
            plan = build_astra_batch_plan(
                manifest,
                corpus,
                approval,
                prompt,
                batch_size=args.batch_size,
            )
            public, private = write_astra_batch_plan(
                plan,
                public_path=args.public_plan,
                private_dir=args.private_dir,
            )
            print(
                json.dumps(
                    {
                        "public_plan": str(public),
                        "private_plan": str(private),
                        "batch_count": plan.batch_count,
                        "unique_study_count": plan.unique_study_count,
                        "plan_fingerprint": plan.plan_fingerprint,
                    },
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "pending":
            plan = load_astra_batch_plan(args.private_plan)
            completed = completed_astra_batch_ids(plan, args.private_dir)
            pending_batches = pending_astra_batches(plan, args.private_dir)
            print(
                json.dumps(
                    {
                        "plan_fingerprint": plan.plan_fingerprint,
                        "completed_batch_ids": list(completed),
                        "pending_batch_ids": [batch.batch_id for batch in pending_batches],
                    },
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "record":
            plan = load_astra_batch_plan(args.private_plan)
            raw_output = args.raw_output.read_text(encoding="utf-8")
            audit = record_astra_output(
                plan,
                args.batch_id,
                raw_output,
                args.private_dir,
                attempt=args.attempt,
            )
            print(json.dumps(audit.to_public_dict(), sort_keys=True))
            return 0

        if args.command == "publish":
            plan = load_astra_batch_plan(args.private_plan)
            path = write_astra_public_evidence(plan, args.private_dir, args.public_output)
            print(json.dumps({"public_evidence": str(path)}, sort_keys=True))
            return 0

        if args.command == "schema":
            _write_or_print(ASTRA_OUTPUT_JSON_SCHEMA, args.output)
            return 0

        raise AstraHandoffError(f"unknown command {args.command}")
    except (OSError, RuntimeError, ValueError, AstraHandoffError) as exc:
        print(f"ticket-16 Astra handoff failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
