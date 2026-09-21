"""Bounded Ticket 17 training adapter for clean report supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, cast

from .archive import atomic_json_write
from .folds import FOLD_IDS, FoldManifest, load_fold_manifest
from .real_inference import iter_real_studies, load_integration_manifest
from .report_extraction import (
    REPORT_CORPUS_SCHEMA_VERSION,
    REPORT_EXTRACTION_SCHEMA_VERSION,
    REPORT_TRAINING_POLICY_VERSION,
    FrozenDeterministicExtractor,
    ReportCorpus,
    ReportExtractionRun,
    load_report_corpus,
    run_report_extraction,
)
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
    parser.add_argument("--frozen-report-artifact", type=Path, default=None)
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


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SupervisionAdmissionError(f"frozen report artifact {field_name} must be an object")
    return cast(Mapping[str, object], value)


def _validate_frozen_report_artifact(
    path: Path,
    manifest: FoldManifest,
    corpus: ReportCorpus,
    extraction: ReportExtractionRun,
) -> str:
    """Prove that the replayed labels equal the stored Ticket 15 artifact."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SupervisionAdmissionError(
            f"could not read frozen Ticket 15 report artifact: {path}"
        ) from exc
    root = _mapping(payload, "root")
    if root.get("schema_version") != REPORT_EXTRACTION_SCHEMA_VERSION or root.get("ticket") != "15":
        raise SupervisionAdmissionError("frozen report artifact is not a Ticket 15 extraction artifact")
    input_payload = _mapping(root.get("input"), "input")
    if (
        input_payload.get("schema_version") != REPORT_CORPUS_SCHEMA_VERSION
        or input_payload.get("source_version") != corpus.source_version
        or input_payload.get("record_set_sha256") != corpus.record_set_sha256
        or input_payload.get("report_count") != corpus.report_count
    ):
        raise SupervisionAdmissionError("frozen report artifact report-set identity disagrees")
    fold_context = _mapping(root.get("fold_context"), "fold_context")
    if (
        fold_context.get("fold_manifest_checksum") != manifest.manifest_checksum
        or fold_context.get("repeat_name") != manifest.repeat_name
        or fold_context.get("report_source_version") != corpus.source_version
    ):
        raise SupervisionAdmissionError("frozen report artifact fold/source identity disagrees")
    extractor = _mapping(root.get("extractor"), "extractor")
    frozen_extractor = FrozenDeterministicExtractor().to_public_dict()
    expected_extractor_fields = {
        "version": frozen_extractor["extractor_version"],
        "configuration_fingerprint": frozen_extractor["configuration_fingerprint"],
        "ruleset_fingerprint": frozen_extractor["ruleset_fingerprint"],
        "code_fingerprint": frozen_extractor["code_fingerprint"],
    }
    for field_name, expected in expected_extractor_fields.items():
        if extractor.get(field_name) != expected:
            raise SupervisionAdmissionError(
                f"frozen report artifact extractor {field_name} disagrees"
            )
    if extractor.get("training_policy") != REPORT_TRAINING_POLICY_VERSION:
        raise SupervisionAdmissionError("frozen report artifact training policy disagrees")
    reproducibility = _mapping(root.get("reproducibility"), "reproducibility")
    if (
        reproducibility.get("verified") is not True
        or reproducibility.get("semantic_digest") != extraction.semantic_digest
    ):
        raise SupervisionAdmissionError("frozen report artifact reproducibility identity disagrees")
    provenance = _mapping(root.get("provenance"), "provenance")
    if provenance.get("upstream_input_manifest") != extraction.evaluation.upstream_input_manifest:
        raise SupervisionAdmissionError("frozen report artifact upstream lineage disagrees")
    partitions = root.get("fold_partitions")
    if not isinstance(partitions, list) or len(partitions) != len(extraction.folds):
        raise SupervisionAdmissionError("frozen report artifact fold partitions are incomplete")
    for stored, actual in zip(partitions, extraction.folds):
        partition = _mapping(stored, "fold_partitions")
        stored_partitions = partition.get("partitions")
        if not isinstance(stored_partitions, dict):
            raise SupervisionAdmissionError(
                "frozen report artifact fold partition payload is incomplete"
            )
        stored_training_labels = stored_partitions.get("training_labels")
        stored_evaluation_labels = stored_partitions.get("held_out_evaluation")
        eligible_training_uids = partition.get("eligible_training_report_uids")
        if (
            partition.get("fold") != actual.fold
            or partition.get("fold_manifest_checksum") != manifest.manifest_checksum
            or not isinstance(stored_training_labels, list)
            or len(stored_training_labels) != len(actual.training_labels)
            or not isinstance(eligible_training_uids, list)
            or len(eligible_training_uids) != len(actual.eligible_training_report_uids)
            or not isinstance(stored_evaluation_labels, list)
            or len(stored_evaluation_labels) != len(actual.evaluation_labels)
        ):
            raise SupervisionAdmissionError("frozen report artifact fold partition disagrees")
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise SupervisionAdmissionError("could not hash frozen report artifact") from exc


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _check_context(
            require_slurm=args.require_slurm,
            allow_local_smoke=args.allow_local_smoke,
        )
        if not args.allow_local_smoke and args.frozen_report_artifact is None:
            raise SupervisionAdmissionError(
                "real Ticket 17 training requires the stored Ticket 15 report artifact"
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
        frozen_report_artifact_sha256 = None
        if args.frozen_report_artifact is not None:
            frozen_report_artifact_sha256 = _validate_frozen_report_artifact(
                args.frozen_report_artifact,
                fold_manifest,
                report_corpus,
                report_extraction,
            )
        admitted = admit_fold_supervision(fold_manifest, args.fold, report_extraction)
        public_supervision = args.supervision_public or Path(
            f"{args.output}.supervision.public.json"
        )
        private_supervision = args.supervision_private or Path(
            f"{args.output}.supervision.private.json"
        )
        studies = {}
        for loaded in iter_real_studies(
            integration_manifest,
            args.source,
            require_complete_series=True,
            allow_unverified_directory=args.allow_local_smoke,
        ):
            studies[loaded.study.study_instance_uid] = loaded.study
        admitted_for_images = admitted.for_image_studies(tuple(studies))
        atomic_json_write(public_supervision, admitted_for_images.to_public_dict())
        atomic_json_write(private_supervision, admitted_for_images.to_private_dict())
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
            supervision=admitted_for_images,
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
                    "frozen_report_artifact_sha256": frozen_report_artifact_sha256,
                    "supervision_admission": admitted_for_images.to_public_dict(),
                    "full_fold_supervision_admission": admitted.to_public_dict(),
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
