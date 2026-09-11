"""Freeze, execute, or benchmark a locally validated ticket-11 candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .candidate import (
    CandidateError,
    benchmark_candidate_execution,
    freeze_candidate,
    run_candidate_inference,
)


def _common_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--private-evidence", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--recover-damaged-inputs", action="store_true")
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("--allow-local-smoke", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="publish one immutable candidate bundle")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--checkpoint", type=Path, action="append", required=True)
    freeze.add_argument("--oof-evidence", type=Path, required=True)
    freeze.add_argument("--oof-comparison", type=Path, default=None)
    freeze.add_argument("--code-root", type=Path, default=None)
    freeze.add_argument("--cluster-requirements", type=Path, default=None)
    freeze.add_argument("--offline-requirements", type=Path, default=None)
    freeze.add_argument("--notebook", type=Path, default=None)
    freeze.add_argument("--candidate-id", required=True)
    freeze.add_argument("--code-revision", required=True)
    freeze.add_argument("--source-version", default=None)
    freeze.add_argument("--source-fingerprint", default=None)
    freeze.add_argument("--fold-manifest-checksum", default=None)

    run = subparsers.add_parser("run", help="run one candidate on a manifest/source pair")
    _common_execution_arguments(run)

    benchmark = subparsers.add_parser("benchmark", help="run and project whole-pipeline cost")
    _common_execution_arguments(benchmark)
    benchmark.add_argument("--expected-study-count", type=int, default=1300)
    benchmark.add_argument("--long-tail-studies", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze":
            kwargs = {
                "code_root": args.code_root,
                "cluster_requirements_path": args.cluster_requirements,
                "offline_requirements_path": args.offline_requirements,
                "notebook_path": args.notebook,
            }
            kwargs = {key: value for key, value in kwargs.items() if value is not None}
            bundle = freeze_candidate(
                args.output,
                model_checkpoints=args.checkpoint,
                oof_evidence_path=args.oof_evidence,
                candidate_id=args.candidate_id,
                code_revision=args.code_revision,
                source_version=args.source_version,
                source_fingerprint=args.source_fingerprint,
                fold_manifest_checksum=args.fold_manifest_checksum,
                oof_comparison_path=args.oof_comparison,
                **kwargs,
            )
            print(json.dumps(dict(bundle.manifest), indent=2, sort_keys=True))
            return 0
        if args.command == "run":
            execution = run_candidate_inference(
                args.candidate,
                args.manifest,
                args.source,
                args.output,
                evidence_path=args.evidence,
                private_evidence_path=args.private_evidence,
                cache_root=args.cache_root,
                recover_damaged_inputs=args.recover_damaged_inputs,
                require_slurm=args.require_slurm,
                allow_local_smoke=args.allow_local_smoke,
            )
        else:
            execution = benchmark_candidate_execution(
                args.candidate,
                args.manifest,
                args.source,
                args.output,
                evidence_path=args.evidence,
                private_evidence_path=args.private_evidence,
                expected_study_count=args.expected_study_count,
                long_tail_studies=args.long_tail_studies,
                cache_root=args.cache_root,
                recover_damaged_inputs=args.recover_damaged_inputs,
                require_slurm=args.require_slurm,
                allow_local_smoke=args.allow_local_smoke,
            )
        print(json.dumps(dict(execution.evidence), indent=2, sort_keys=True))
        return 0
    except (CandidateError, OSError, RuntimeError, ValueError) as exc:
        print(f"ticket-11 candidate failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
