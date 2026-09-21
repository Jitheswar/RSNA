"""Bounded Ticket 17 training adapter for clean report supervision."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from .archive import atomic_json_write
from .folds import FOLD_IDS, load_fold_manifest
from .real_inference import iter_real_studies, load_integration_manifest
from .report_extraction import load_report_corpus, run_report_extraction
from .supervision import SupervisionAdmissionError, admit_fold_supervision
from .training import (
    PreflightError,
    TrainingError,
    TrainingDataError,
    _public_run_environment,
    require_cpu_preflight,
    run_cpu_preflight,
    train_one_fold,
)
from .series_cache import SeriesCache


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=FOLD_IDS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--source-fingerprint", default=None)
    parser.add_argument("--preflight-evidence", type=Path, default=None)
    parser.add_argument("--evidence", type=Path, default=None)
    parser.add_argument("--supervision-public", type=Path, default=None)
    parser.add_argument("--supervision-private", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--execution-attempt-id", default=None)
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("--allow-local-smoke", action="store_true")
    return parser


def _check_context(*, require_slurm: bool, allow_local_smoke: bool) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if require_slurm and (not job_id or partition != "cluster1-cpu"):
        raise PreflightError("ticket-17 training requires a cluster1-cpu Slurm allocation")
    if not job_id and not allow_local_smoke:
        raise PreflightError(
            "refusing weak-label training outside Slurm; use --allow-local-smoke for a tiny fixture"
        )


def _validate_inputs(args: argparse.Namespace, fold_manifest: Any, integration_manifest: Any) -> None:
    if not args.source.exists():
        raise TrainingDataError(f"real-DICOM source does not exist: {args.source}")
    if integration_manifest.source_version != args.source_version:
        raise TrainingDataError("source version disagrees with the integration manifest")
    if fold_manifest.source_version != args.source_version:
        raise TrainingDataError("source version disagrees with the fold manifest")
    if integration_manifest.source_manifest_checksum != fold_manifest.source_manifest_checksum:
        raise TrainingDataError("integration and fold manifests have different source checksums")
    if (
        args.source_fingerprint is not None
        and args.source_fingerprint != integration_manifest.archive_fingerprint
    ):
        raise TrainingDataError("source fingerprint disagrees with the integration manifest")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _check_context(
            require_slurm=args.require_slurm,
            allow_local_smoke=args.allow_local_smoke,
        )
        fold_manifest = load_fold_manifest(args.fold_manifest)
        integration_manifest = load_integration_manifest(args.integration_manifest)
        _validate_inputs(args, fold_manifest, integration_manifest)
        if args.allow_local_smoke and not os.environ.get("SLURM_JOB_ID"):
            if len(fold_manifest.memberships) > 128:
                raise TrainingDataError("local smoke is limited to 128 fold studies")

        preflight = run_cpu_preflight(
            output_path=args.preflight_evidence,
            require_isolated_python=not args.allow_local_smoke,
            require_slurm=args.require_slurm,
        )
        require_cpu_preflight(
            preflight,
            require_slurm=not args.allow_local_smoke,
            require_isolated_python=not args.allow_local_smoke,
        )

        report_corpus = load_report_corpus(args.reports)
        if args.allow_local_smoke and not os.environ.get("SLURM_JOB_ID"):
            if report_corpus.report_count > 32:
                raise TrainingDataError("local smoke is limited to 32 reports")
        report_extraction = run_report_extraction(
            fold_manifest,
            report_corpus,
            verify_reproducibility=True,
        )
        admitted = admit_fold_supervision(fold_manifest, args.fold, report_extraction)
        public_supervision = args.supervision_public or Path(
            f"{args.output}.supervision.public.json"
        )
        private_supervision = args.supervision_private or Path(
            f"{args.output}.supervision.private.json"
        )
        atomic_json_write(public_supervision, admitted.to_public_dict())
        atomic_json_write(private_supervision, admitted.to_private_dict())

        studies = {}
        for loaded in iter_real_studies(
            integration_manifest,
            args.source,
            require_complete_series=True,
            allow_unverified_directory=args.allow_local_smoke,
        ):
            studies[loaded.study.study_instance_uid] = loaded.study
        missing_images = sorted(set(admitted.training_study_uids) - set(studies))
        if missing_images:
            raise TrainingDataError(
                "integration manifest does not cover admitted training studies: "
                f"{missing_images[:3]}"
            )
        cache = SeriesCache(args.cache_root) if args.cache_root is not None else None
        result = train_one_fold(
            studies,
            fold_manifest,
            args.fold,
            args.output,
            source_version=args.source_version,
            source_fingerprint=args.source_fingerprint or integration_manifest.archive_fingerprint,
            integration_manifest_checksum=integration_manifest.manifest_checksum,
            series_cache=cache,
            supervision=admitted,
            preprocess_config=integration_manifest.preprocess_config,
            packing_config=integration_manifest.packing_config,
            preflight=preflight,
            allow_local_smoke=args.allow_local_smoke,
            resume_from=args.resume_from,
            execution_attempt_id=args.execution_attempt_id,
        )
        if args.evidence is not None:
            atomic_json_write(
                args.evidence,
                {
                    **result.to_public_dict(),
                    "environment": _public_run_environment(),
                    "ticket": "17",
                    "source_version": args.source_version,
                    "fold_manifest_checksum": fold_manifest.manifest_checksum,
                    "report_extraction_semantic_digest": report_extraction.semantic_digest,
                    "supervision_admission": admitted.to_public_dict(),
                    "preflight_passed": preflight.passed,
                },
            )
        print(json.dumps(result.to_public_dict(), indent=2, sort_keys=True))
    except (OSError, RuntimeError, ValueError, TrainingError, SupervisionAdmissionError) as exc:
        print(f"ticket-17 weak-label training failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
