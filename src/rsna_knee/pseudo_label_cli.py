"""Slurm-only entrypoint for the bounded Ticket 21 pseudo-label trial."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .archive import atomic_json_write
from .folds import FOLD_IDS, load_fold_manifest
from .pseudo_label_experiment import run_ticket21_pseudo_label_experiment
from .real_inference import iter_real_studies, load_integration_manifest
from .series_cache import SeriesCache
from .training import PreflightError, run_cpu_preflight


def _parser() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=FOLD_IDS, default=0)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-fingerprint", default=None)
    parser.add_argument("--preflight-evidence", type=Path, required=True)
    parser.add_argument("--compute-evidence", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--cache-root", type=Path, default=None)
    return parser


def _require_ticket21_slurm_context(code_revision: str) -> None:
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "cluster1-cpu":
        raise PreflightError("Ticket 21 real-data work requires a cluster1-cpu Slurm allocation")
    if os.environ.get("RSNA_CODE_REVISION") != code_revision:
        raise PreflightError("Ticket 21 code revision environment does not match its frozen argument")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _require_ticket21_slurm_context(args.code_revision)
        repository_root = Path(__file__).resolve().parents[2]
        output_root = args.output_root.resolve()
        if output_root.is_relative_to(repository_root):
            raise ValueError("Ticket 21 output must be outside the repository")
        if not args.source.exists():
            raise FileNotFoundError("Ticket 21 DICOM source does not exist")
        for path in (args.integration_manifest, args.fold_manifest):
            if not path.is_file():
                raise FileNotFoundError(f"Ticket 21 input manifest does not exist: {path}")
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError(f"refusing to overwrite Ticket 21 output: {output_root}")

        fold_manifest = load_fold_manifest(args.fold_manifest)
        integration_manifest = load_integration_manifest(args.integration_manifest)
        source_fingerprint = args.source_fingerprint or integration_manifest.archive_fingerprint
        if source_fingerprint != integration_manifest.archive_fingerprint:
            raise ValueError("Ticket 21 source fingerprint disagrees with the integration manifest")
        if fold_manifest.source_version != integration_manifest.source_version:
            raise ValueError("Ticket 21 fold and integration manifests have different source versions")
        if fold_manifest.source_fingerprint is not None and (
            fold_manifest.source_fingerprint != source_fingerprint
        ):
            raise ValueError("Ticket 21 source fingerprint disagrees with the frozen folds")
        if fold_manifest.source_manifest_checksum != integration_manifest.source_manifest_checksum:
            raise ValueError("Ticket 21 fold and integration manifests have different source checksums")

        expected_gold = {
            item.study_instance_uid for item in fold_manifest.gold_memberships
        }
        if not expected_gold.issubset(set(integration_manifest.study_uids)):
            raise ValueError("Ticket 21 integration manifest does not cover every frozen gold study")
        if len(args.code_revision) != 40 or any(
            character not in "0123456789abcdef" for character in args.code_revision
        ):
            raise ValueError("Ticket 21 requires a full lowercase Git revision")

        preflight = run_cpu_preflight(
            output_path=args.preflight_evidence,
            require_isolated_python=True,
            require_slurm=True,
            required_slurm_resources={
                "partition": "cluster1-cpu",
                "nodes": 1,
                "ntasks": 1,
                "cpus_per_task": 16,
                "memory_gb": 64,
                "wall_time_minutes": 360,
                "exclusive": False,
            },
        )

        load_started = time.perf_counter()
        studies = {
            loaded.study.study_instance_uid: loaded.study
            for loaded in iter_real_studies(
                integration_manifest,
                args.source,
                require_complete_series=True,
            )
        }
        input_loading_seconds = time.perf_counter() - load_started
        cache = SeriesCache(args.cache_root) if args.cache_root is not None else None
        evidence = run_ticket21_pseudo_label_experiment(
            studies,
            fold_manifest,
            integration_manifest,
            output_root,
            fold=args.fold,
            code_revision=args.code_revision,
            source_fingerprint=source_fingerprint,
            preflight=preflight,
            input_loading_seconds=input_loading_seconds,
            series_cache=cache,
        )
        atomic_json_write(
            args.compute_evidence,
            {
                "schema_version": "rsna-knee-ticket21-cli-cost-v1",
                "input_loading_seconds": input_loading_seconds,
                "source_studies_loaded": len(studies),
                "preflight_passed": preflight.passed,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
                "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
                "memory_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
                "exclusive": os.environ.get("SLURM_JOB_EXCLUSIVE", "0"),
            },
        )
        print(json.dumps(evidence, indent=2, sort_keys=True))
    except (OSError, RuntimeError, ValueError, PreflightError) as exc:
        print(f"Ticket 21 pseudo-label experiment failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
