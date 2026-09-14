"""Command-line entrypoint for the ticket-16 local model comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .folds import load_fold_manifest
from .local_report_extraction import (
    LocalModelApprovalError,
    LocalReportExtractionError,
    compare_report_extractors,
    load_local_model_approval,
    run_local_report_extraction,
    write_local_model_comparison_reports,
    write_local_model_proposal,
    LocalLanguageModelExtractor,
)
from .report_extraction import (
    ReportExtractionError,
    load_report_corpus,
    run_report_extraction,
)


LOCAL_SMOKE_MAX_STUDIES = 128
LOCAL_SMOKE_MAX_REPORTS = 32


def _check_execution_context(*, require_slurm: bool, allow_local_smoke: bool) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if require_slurm and (not job_id or partition != "cluster1-cpu"):
        raise RuntimeError(
            "ticket-16 local model comparison must run in a cluster1-cpu Slurm allocation"
        )
    if not job_id and not allow_local_smoke:
        raise RuntimeError(
            "refusing local model comparison outside Slurm; use --allow-local-smoke only with a tiny fixture"
        )


def _check_local_smoke_size(manifest: object, corpus: object) -> None:
    memberships = getattr(manifest, "memberships", ())
    report_count = getattr(corpus, "report_count", 0)
    if len(memberships) > LOCAL_SMOKE_MAX_STUDIES:
        raise RuntimeError(
            f"local smoke is limited to {LOCAL_SMOKE_MAX_STUDIES} studies; use Slurm for the supplied fold manifest"
        )
    if report_count > LOCAL_SMOKE_MAX_REPORTS:
        raise RuntimeError(
            f"local smoke is limited to {LOCAL_SMOKE_MAX_REPORTS} reports; use Slurm for the supplied report corpus"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-manifest", type=Path)
    parser.add_argument("--reports", type=Path)
    parser.add_argument("--model-approval", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--public-output", type=Path)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--write-proposal", type=Path)
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("--allow-local-smoke", action="store_true")
    parser.add_argument("--skip-reproducibility-check", action="store_true")
    return parser


def _require_comparison_arguments(args: argparse.Namespace) -> None:
    required = (
        "fold_manifest",
        "reports",
        "model_approval",
        "model_dir",
        "public_output",
        "private_output",
    )
    missing = [name.replace("_", "-") for name in required if getattr(args, name) is None]
    if missing:
        raise RuntimeError("comparison requires: " + ", ".join(f"--{name}" for name in missing))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.write_proposal is not None:
            write_local_model_proposal(args.write_proposal)
            print(json.dumps({"proposal": str(args.write_proposal)}, sort_keys=True))
            return 0
        _check_execution_context(
            require_slurm=args.require_slurm,
            allow_local_smoke=args.allow_local_smoke,
        )
        _require_comparison_arguments(args)
        approval = load_local_model_approval(args.model_approval)
        # Fail before opening the report corpus or loading optional model code.
        approval.require_approved()
        manifest = load_fold_manifest(args.fold_manifest)
        corpus = load_report_corpus(args.reports)
        if args.allow_local_smoke and not os.environ.get("SLURM_JOB_ID"):
            _check_local_smoke_size(manifest, corpus)
        extractor = LocalLanguageModelExtractor.from_local_directory(
            approval,
            args.model_dir,
        )
        baseline = run_report_extraction(
            manifest,
            corpus,
            verify_reproducibility=not args.skip_reproducibility_check,
        )
        local_model = run_local_report_extraction(
            manifest,
            corpus,
            extractor,
            verify_reproducibility=not args.skip_reproducibility_check,
        )
        comparison = compare_report_extractors(baseline, local_model)
        write_local_model_comparison_reports(
            comparison,
            public_path=args.public_output,
            private_path=args.private_output,
        )
    except (
        OSError,
        RuntimeError,
        LocalModelApprovalError,
        LocalReportExtractionError,
        ReportExtractionError,
        ValueError,
    ) as exc:
        print(f"ticket-16 local model comparison failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(comparison.to_public_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
