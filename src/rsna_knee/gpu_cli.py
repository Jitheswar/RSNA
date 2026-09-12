"""Prepare, submit, and execute the approval-gated ticket-13 GPU path."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from .gpu import (
    GPU_DEFAULT_PARTITION,
    GpuResourceRequest,
    prepare_gpu_experiment,
    run_approved_gpu_experiment,
    submit_gpu_experiment,
)
from .training import TrainingConfig, TrainingDataError, run_cpu_preflight


def _tuple_ints(value: str, *, count: int, field_name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be comma-separated integers") from exc
    if len(values) != count:
        raise ValueError(f"{field_name} must contain {count} values")
    return values


def _read_json_object(path: Path) -> Mapping[str, object]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read JSON object {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON value is not an object: {path}")
    return value


def _environment_values(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError("--env values must use KEY=VALUE")
        key, value = item.split("=", 1)
        if not key or not value:
            raise ValueError("--env values require non-empty keys and values")
        result[key] = value
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="run the bounded CPU preflight")
    preflight.add_argument("--output", type=Path, required=True)

    prepare = subparsers.add_parser("prepare", help="write an immutable GPU request")
    prepare.add_argument("--preflight", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--experiment-id", default=None)
    prepare.add_argument("--identity-json", type=Path, default=None)
    prepare.add_argument("--partition", default=GPU_DEFAULT_PARTITION)
    prepare.add_argument("--gres", default="gpu:1")
    prepare.add_argument("--cpus-per-task", type=int, default=16)
    prepare.add_argument("--memory-gb", type=int, default=64)
    prepare.add_argument("--wall-time-minutes", type=int, default=360)
    prepare.add_argument("--exclusive", action="store_true")
    prepare.add_argument("--smoke-test", action="store_true")
    prepare.add_argument("--steps", type=int, default=None)
    prepare.add_argument("--image-size", default="128,128")
    prepare.add_argument("--channels", default="8,16,32")
    prepare.add_argument("--max-slices", type=int, default=8)
    prepare.add_argument("--max-series", type=int, default=16)
    prepare.add_argument("--fold", type=int, default=None)
    prepare.add_argument("--source-version", default=None)
    prepare.add_argument("--source-fingerprint", default=None)
    prepare.add_argument("--fold-manifest-checksum", default=None)
    prepare.add_argument("--integration-manifest-checksum", default=None)
    prepare.add_argument("--no-retry", action="store_true")

    submit = subparsers.add_parser("submit", help="submit an exact approved request through sbatch")
    submit.add_argument("--request", type=Path, required=True)
    submit.add_argument("--approval", type=Path, required=True)
    submit.add_argument("--script", type=Path, default=Path("slurm/run_ticket13_gpu_experiment.sbatch"))
    submit.add_argument("--sbatch", default="sbatch")
    submit.add_argument("--no-wait", action="store_true")
    submit.add_argument("--env", action="append", default=[])

    run = subparsers.add_parser("run", help="run one approved GPU experiment in its Slurm allocation")
    run.add_argument("--request", type=Path, required=True)
    run.add_argument("--approval", type=Path, required=True)
    run.add_argument("--preflight", type=Path, default=None)
    run.add_argument("--integration-manifest", type=Path, required=True)
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--fold-manifest", type=Path, required=True)
    run.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--evidence", type=Path, required=True)
    run.add_argument("--source-version", required=True)
    run.add_argument("--source-fingerprint", default=None)
    run.add_argument("--cache-root", type=Path, default=None)
    return parser


def _prepare(args: argparse.Namespace) -> int:
    steps = args.steps if args.steps is not None else (1 if args.smoke_test else 100)
    image_size = _tuple_ints(args.image_size, count=2, field_name="--image-size")
    channels = _tuple_ints(args.channels, count=3, field_name="--channels")
    identity = _read_json_object(args.identity_json) if args.identity_json is not None else None
    config = TrainingConfig(
        image_size=(image_size[0], image_size[1]),
        channels=(channels[0], channels[1], channels[2]),
        max_slices=args.max_slices,
        max_series=args.max_series,
        successful_optimizer_steps=steps,
        device="cuda",
    )
    resource = GpuResourceRequest(
        partition=args.partition,
        gres=args.gres,
        cpus_per_task=args.cpus_per_task,
        memory_gb=args.memory_gb,
        wall_time_minutes=args.wall_time_minutes,
        exclusive=args.exclusive,
        smoke_test=args.smoke_test,
    )
    request = prepare_gpu_experiment(
        args.output,
        preflight_path=args.preflight,
        resource=resource,
        training_config=config,
        experiment_id=args.experiment_id,
        experiment_identity=identity,
        fold=args.fold,
        source_version=args.source_version,
        source_fingerprint=args.source_fingerprint,
        fold_manifest_checksum=args.fold_manifest_checksum,
        integration_manifest_checksum=args.integration_manifest_checksum,
        retry_allowed=not args.no_retry,
    )
    print(json.dumps(dict(request.payload), indent=2, sort_keys=True))
    return 0


def _preflight(args: argparse.Namespace) -> int:
    result = run_cpu_preflight(
        output_path=args.output,
        require_isolated_python=True,
        require_slurm=True,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.passed else 1


def _run(args: argparse.Namespace) -> int:
    from .folds import load_fold_manifest
    from .real_inference import iter_real_studies, load_integration_manifest
    from .series_cache import SeriesCache

    if not args.source.exists():
        raise TrainingDataError(f"real-DICOM source does not exist: {args.source}")
    fold_manifest = load_fold_manifest(args.fold_manifest)
    if fold_manifest.superseded:
        raise TrainingDataError("GPU training refuses a superseded fold manifest")
    integration_manifest = load_integration_manifest(args.integration_manifest)
    if integration_manifest.source_version != args.source_version:
        raise TrainingDataError("source version disagrees with the integration manifest")
    if args.source_fingerprint is not None and args.source_fingerprint != integration_manifest.archive_fingerprint:
        raise TrainingDataError("source fingerprint disagrees with the integration manifest")
    required_gold = {item.study_instance_uid for item in fold_manifest.gold_memberships}
    if not required_gold.issubset(set(integration_manifest.study_uids)):
        raise TrainingDataError("integration manifest does not cover every frozen gold study")
    studies = {
        loaded.study.study_instance_uid: loaded.study
        for loaded in iter_real_studies(
            integration_manifest,
            args.source,
            require_complete_series=True,
            allow_unverified_directory=False,
        )
    }
    cache = SeriesCache(args.cache_root) if args.cache_root is not None else None
    result = run_approved_gpu_experiment(
        args.request,
        args.approval,
        preflight_path=args.preflight,
        output_dir=args.output,
        evidence_path=args.evidence,
        studies=studies,
        fold_manifest=fold_manifest,
        fold=args.fold,
        source_version=args.source_version,
        source_fingerprint=args.source_fingerprint or integration_manifest.archive_fingerprint,
        fold_manifest_checksum=fold_manifest.manifest_checksum,
        integration_manifest_checksum=integration_manifest.manifest_checksum,
        series_cache=cache,
        previous_retries=int(os.environ.get("RSNA_TICKET13_PREVIOUS_RETRIES", "0")),
    )
    print(json.dumps(dict(result.evidence), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            return _preflight(args)
        if args.command == "prepare":
            return _prepare(args)
        if args.command == "submit":
            result = submit_gpu_experiment(
                args.request,
                args.approval,
                script_path=args.script,
                environment=_environment_values(args.env),
                sbatch_binary=args.sbatch,
                wait=not args.no_wait,
            )
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            return 0 if result.status != "failed" else 1
        return _run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ticket-13 GPU path failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
