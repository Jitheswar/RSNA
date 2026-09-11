"""CLI for the bounded synthetic CPU inference demonstration."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from .inference import InferenceEngine, infer_studies
from .model import HierarchicalStudyModel, ModelConfig
from .preprocessing import PreprocessConfig
from .synthetic import make_synthetic_study
from .targets import SUBMISSION_COLUMNS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/submission.csv"))
    parser.add_argument("--evidence", type=Path, default=None)
    parser.add_argument(
        "--require-slurm",
        action="store_true",
        help="fail unless this process is running in a cluster1-cpu Slurm allocation",
    )
    parser.add_argument(
        "--allow-local-smoke",
        action="store_true",
        help="explicitly permit the tiny development smoke test outside Slurm",
    )
    return parser


def _check_execution_context(require_slurm: bool, allow_local_smoke: bool) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if require_slurm and (not job_id or partition != "cluster1-cpu"):
        raise RuntimeError("CPU inference must run in a cluster1-cpu Slurm allocation")
    if not job_id and not allow_local_smoke:
        raise RuntimeError(
            "refusing to run the inference demonstration outside Slurm; "
            "use --allow-local-smoke only for a tiny development check"
        )


def _configure_threads() -> int:
    requested = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    try:
        thread_count = max(1, min(16, int(requested)))
    except ValueError:
        thread_count = 1
    torch.set_num_threads(thread_count)
    return thread_count


def run_demo(output_path: Path, evidence_path: Path | None = None) -> dict[str, object]:
    """Run two private-free synthetic studies and return auditable evidence."""

    thread_count = _configure_threads()
    preprocess = PreprocessConfig()
    model_config = ModelConfig()
    torch.manual_seed(20260911)
    model = HierarchicalStudyModel(model_config)
    engine = InferenceEngine(model, preprocess)
    studies = (
        make_synthetic_study("synthetic-study-001", series_count=2, slices_per_series=6),
        make_synthetic_study("synthetic-study-002", series_count=1, slices_per_series=4, seed=20260912),
    )
    started = time.perf_counter()
    batch = infer_studies(engine, studies, output_path)
    elapsed_seconds = time.perf_counter() - started

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    scores = np.concatenate([result.scores for result in batch.results])
    evidence: dict[str, object] = {
        "ticket": "02",
        "status": "passed_for_synthetic_cpu_scope",
        "scope": "in-memory synthetic pydicom MR datasets; no competition data",
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "python_prefix": sys.prefix,
            "python_base_prefix": sys.base_prefix,
            "python_isolated_venv": sys.prefix != sys.base_prefix,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "pydicom_version": version("pydicom") if _has_distribution("pydicom") else None,
            "torch_num_threads": thread_count,
        },
        "input_studies": len(batch.results),
        "input_series": sum(result.valid_series_count for result in batch.results),
        "output_path": str(output_path),
        "output_header": rows[0],
        "output_rows": len(rows) - 1,
        "output_columns": len(rows[0]),
        "exact_submission_schema": rows[0] == list(SUBMISSION_COLUMNS),
        "scores_finite": bool(np.isfinite(scores).all()),
        "scores_bounded": bool(((scores >= 0.0) & (scores <= 1.0)).all()),
        "fallback_studies": sum(result.used_fallback for result in batch.results),
        "elapsed_seconds": elapsed_seconds,
        "real_dicom_compatibility_verified": False,
        "full_training_preflight_verified": False,
    }
    if evidence_path is not None:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def _has_distribution(name: str) -> bool:
    try:
        version(name)
    except Exception:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _check_execution_context(args.require_slurm, args.allow_local_smoke)
        evidence = run_demo(args.output, args.evidence)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ticket-02 inference failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
