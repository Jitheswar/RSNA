"""Bounded real-series cache benchmark used by the ticket-06 Slurm adapter."""

from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

from .archive import atomic_json_write
from .real_inference import (
    _check_execution_context,
    _validate_local_smoke_scope,
    _verify_archive_source_identity,
    iter_real_studies,
    load_integration_manifest,
)
from .series_cache import CacheBenchmark, SeriesCache


def _empty_totals() -> dict[str, int | float]:
    return {
        "series_count": 0,
        "worker_count": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "bytes_read": 0,
        "bytes_written": 0,
        "peak_temporary_bytes": 0,
        "elapsed_seconds": 0.0,
    }


def _add_benchmark(total: dict[str, int | float], result: CacheBenchmark) -> None:
    total["series_count"] = int(total["series_count"]) + result.series_count
    total["worker_count"] = max(int(total["worker_count"]), result.worker_count)
    total["cache_hits"] = int(total["cache_hits"]) + result.cache_hits
    total["cache_misses"] = int(total["cache_misses"]) + result.cache_misses
    total["bytes_read"] = int(total["bytes_read"]) + result.bytes_read
    total["bytes_written"] = int(total["bytes_written"]) + result.bytes_written
    total["peak_temporary_bytes"] = max(
        int(total["peak_temporary_bytes"]), result.peak_temporary_bytes
    )
    total["elapsed_seconds"] = float(total["elapsed_seconds"]) + result.elapsed_seconds


def run_cache_benchmark(
    manifest_path: str | Path,
    source_path: str | Path,
    cache_root: str | Path,
    output_path: str | Path,
    *,
    workers: int = 1,
    require_slurm: bool = False,
    allow_local_smoke: bool = False,
) -> dict[str, object]:
    """Measure bounded materialization and sampling reuse for admitted series."""

    if workers <= 0 or workers > 16:
        raise ValueError("workers must be in the authorized range 1..16")
    _check_execution_context(require_slurm=require_slurm, allow_local_smoke=allow_local_smoke)
    manifest = load_integration_manifest(manifest_path)
    source = Path(source_path)
    if allow_local_smoke:
        _validate_local_smoke_scope(manifest, source)
    _verify_archive_source_identity(source, manifest)

    cache = SeriesCache(cache_root)
    narrow_config = replace(manifest.preprocess_config, max_slices=8)
    wide_config = replace(manifest.preprocess_config, max_slices=16)
    narrow_totals = _empty_totals()
    wide_totals = _empty_totals()
    study_count = 0
    accepted_series_count = 0
    rejected_series_count = 0
    excluded_series_count = 0
    key_equivalence_count = 0
    started = time.perf_counter()

    for loaded in iter_real_studies(
        manifest,
        source,
        allow_unverified_directory=allow_local_smoke,
    ):
        study_count += 1
        accepted = tuple(item.series for item in loaded.series_results if item.series is not None)
        accepted_series_count += len(accepted)
        rejected_series_count += sum(item.status == "rejected" for item in loaded.series_results)
        excluded_series_count += sum(
            item.status == "excluded_non_diagnostic" for item in loaded.series_results
        )
        if not accepted:
            continue
        _add_benchmark(narrow_totals, cache.benchmark(accepted, narrow_config, workers=workers))
        _add_benchmark(wide_totals, cache.benchmark(accepted, wide_config, workers=workers))
        key_equivalence_count += sum(
            cache.cache_key(series, narrow_config) == cache.cache_key(series, wide_config)
            for series in accepted
        )

    if accepted_series_count == 0:
        raise RuntimeError("manifest produced no accepted series for cache benchmarking")

    evidence: dict[str, object] = {
        "schema_version": "rsna-knee-series-cache-benchmark-v1",
        "ticket": "06",
        "status": "completed_cache_mechanics_benchmark",
        "scope": "aggregate-only cache mechanics; no identifiers or paths in this record",
        "policies": {
            "preprocessing": manifest.preprocess_config.cache_config(),
            "sampling_variants": [8, 16],
            "sampling_is_post_materialization": True,
            "model_architecture_in_cache_identity": False,
        },
        "execution": {
            "mode": "slurm_cluster1_cpu" if os.environ.get("SLURM_JOB_ID") else "local_smoke",
            "workers": workers,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "counts": {
            "studies": study_count,
            "accepted_series": accepted_series_count,
            "rejected_series": rejected_series_count,
            "excluded_non_diagnostic_series": excluded_series_count,
            "sampling_key_equivalence": key_equivalence_count,
        },
        "runs": {
            "eight_slice_materialization": narrow_totals,
            "sixteen_slice_reuse": wide_totals,
        },
        "limitations": [
            "NFS throughput and whole-project peak storage require the authorized cluster run.",
            "This record does not authorize cache expansion or protected-artifact cleanup.",
        ],
    }
    atomic_json_write(output_path, evidence)
    return evidence


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("--allow-local-smoke", action="store_true")
    args = parser.parse_args(argv)
    requested_workers = args.workers
    if requested_workers is None:
        try:
            requested_workers = max(1, min(16, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
        except ValueError:
            requested_workers = 1
    try:
        evidence = run_cache_benchmark(
            args.manifest,
            args.source,
            args.cache_root,
            args.output,
            workers=requested_workers,
            require_slurm=args.require_slurm,
            allow_local_smoke=args.allow_local_smoke,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ticket-06 cache benchmark failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


__all__ = ["run_cache_benchmark", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
