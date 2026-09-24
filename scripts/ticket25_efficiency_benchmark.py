"""Benchmark namespace-scoped Kaggle mount discovery with the frozen candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, cast

import rsna_knee.kaggle as kaggle_module
from rsna_knee.candidate import load_candidate
from rsna_knee.inference import InferenceError
from rsna_knee.kaggle import KaggleInputLayout, discover_kaggle_inputs, run_kaggle_baseline
from rsna_knee.submission import validate_submission_csv


REPO = Path(__file__).resolve().parents[1]
PRIVATE = REPO / "artifacts/private"
BASELINE_BUNDLE = PRIVATE / "ticket-24-kaggle-bundle-20260924-v6"
APPROVAL = PRIVATE / "ticket-24-kaggle-approval-dataset-20260924-v6/owner-approval.json"
INPUT = PRIVATE / "ticket-12-representative-input-20260912-v12"
BASELINE_OUTPUT = PRIVATE / "ticket-24-v6-validation/representative-158/submission.csv"
BASELINE_EXECUTION = PRIVATE / "ticket-24-v6-kernel-evidence/kaggle-execution.json"
EXPECTED_CANDIDATE_MANIFEST = "cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc"
EXPECTED_STUDIES = 158
EXPECTED_SERIES = 965
EXPECTED_SLICES = 31_486
SYNTHETIC_UNRELATED_FILES = 31_486
MAX_PEAK_RSS_BYTES = 8 * 1024**3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link_tree(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir() or destination.exists():
        raise ValueError(f"unsafe or existing benchmark tree: {destination}")
    destination.mkdir(parents=True)
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"benchmark source contains a symlink: {path}")
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir()
        elif path.is_file():
            os.link(path, target)


def _mounted_files(root: Path, filename: str, *, max_depth: int = 5) -> tuple[Path, ...]:
    """Mirror the v6 notebook's original shallow metadata search."""

    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"benchmark mount root is not a regular directory: {root}")
    found: list[Path] = []
    frontier: list[tuple[Path, int]] = [(root, 0)]
    payload_names = {"candidate", "code", "test_series", "train", "test"}
    while frontier:
        directory, depth = frontier.pop(0)
        children = sorted(directory.iterdir(), key=lambda path: str(path))
        found_here = False
        for child in children:
            if child.is_symlink():
                continue
            if child.is_file() and child.name == filename:
                found.append(child)
                found_here = True
        if found_here or depth >= max_depth:
            continue
        for child in children:
            if child.is_symlink() or not child.is_dir() or child.name in payload_names:
                continue
            frontier.append((child, depth + 1))
    return tuple(found)


def _make_mount(root: Path) -> tuple[Path, Path]:
    delivery = root / "datasets/jitheswar/rsna-knee-ticket24-hidden-delivery-v6"
    delivery.parent.mkdir(parents=True)
    for name in ("candidate", "code", "runtime"):
        _link_tree(BASELINE_BUNDLE / name, delivery / name)
    for name in ("request.json", "ticket24_hidden_test_delivery_v6.ipynb"):
        os.link(BASELINE_BUNDLE / name, delivery / name)

    approval = root / "datasets/jitheswar/rsna-knee-ticket24-approval-v6"
    approval.mkdir(parents=True)
    os.link(APPROVAL, approval / "owner-approval.json")

    competition = root / "competitions/rsna-knee-abnormality-detection"
    competition.mkdir(parents=True)
    (competition / "test.csv").write_text("synthetic hidden-mount probe\n", encoding="utf-8")
    (competition / "test_series.csv").write_text("synthetic hidden-mount probe\n", encoding="utf-8")
    (competition / "test_series").mkdir()

    # Empty entries reproduce the directory count without duplicating private
    # DICOM pixels or reading any competition data during the lookup probe.
    noise_root = competition / "train_images"
    for index in range(SYNTHETIC_UNRELATED_FILES):
        study = index // 200
        local_index = index % 200
        series = local_index // 34
        slice_index = local_index % 34
        directory = noise_root / f"study-{study:03d}" / f"series-{series:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"slice-{slice_index:04d}.dcm").touch(exist_ok=False)
    return root, competition


def _discover(
    input_root: Path,
    request: Mapping[str, Any],
    *,
    namespace_scoped: bool,
) -> tuple[float, KaggleInputLayout]:
    resolver = kaggle_module._input_namespace_root
    if not namespace_scoped:
        def legacy_root(root: Path, namespace: str) -> Path:
            del namespace
            return root

        kaggle_module._input_namespace_root = legacy_root
    try:
        started = time.perf_counter()
        layout = discover_kaggle_inputs(
            input_root,
            expected_request_checksum=str(request["request_checksum"]),
            expected_candidate_manifest_checksum=EXPECTED_CANDIDATE_MANIFEST,
        )
        return time.perf_counter() - started, layout
    finally:
        kaggle_module._input_namespace_root = resolver


def _fixed_projection(baseline: dict[str, Any], direct_discovery_seconds: float) -> dict[str, Any]:
    stages = baseline["timing"]["stages"]
    detailed = baseline["timing"]["detailed"]
    outer_loop = float(stages["dicom_read_decode"])
    inner_decode = float(detailed["dicom_read_decode_seconds"])
    preprocessing = float(stages["preprocessing"])
    inference = float(stages["inference_ensembling"])
    loop_overhead = max(0.0, outer_loop - inner_decode - preprocessing - inference)
    scaling_seconds_per_study = (
        inner_decode + preprocessing + inference + loop_overhead + float(stages["csv_creation"])
    ) / int(baseline["workload"]["study_count"])
    input_discovery_seconds = float(detailed["input_discovery_seconds"])
    fixed_seconds = sum(
        float(stages[name])
        for name in ("startup_setup", "imports", "model_metadata_loading")
    ) + float(detailed["adapter_preflight_seconds"]) + input_discovery_seconds
    optimized_fixed_seconds = fixed_seconds - input_discovery_seconds + direct_discovery_seconds
    target_studies = int(baseline["timing"]["projection"]["target_studies"])
    scaling_at_target = scaling_seconds_per_study * target_studies
    return {
        "target_studies": target_studies,
        "quality_of_workload_sample": "Ticket 24 actual Kaggle hidden input: 3 studies, 15 series, 557 slices",
        "method": "non-overlapping measured stages; fixed Kaggle setup plus observed 3-study scaling",
        "baseline_fixed_seconds": fixed_seconds,
        "optimized_fixed_seconds": optimized_fixed_seconds,
        "scaling_seconds_per_study": scaling_seconds_per_study,
        "projected_scaling_seconds_at_target": scaling_at_target,
        "baseline_projected_seconds_at_target": fixed_seconds + scaling_at_target,
        "optimized_projected_seconds_at_target": optimized_fixed_seconds + scaling_at_target,
        "projected_savings_seconds_at_target": fixed_seconds - optimized_fixed_seconds,
        "old_projection_corrected_for_nested_loop_timing": True,
        "hidden_scale_execution_verified": False,
    }


def run(output_root: Path) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite benchmark artifacts: {output_root}")
    required = (BASELINE_BUNDLE, APPROVAL, INPUT, BASELINE_OUTPUT, BASELINE_EXECUTION)
    for path in required:
        if path.is_symlink() or not path.exists():
            raise FileNotFoundError(f"required Ticket 24 input is missing or unsafe: {path}")
    output_root.mkdir(parents=True)

    candidate = load_candidate(BASELINE_BUNDLE / "candidate")
    if candidate.manifest_checksum != EXPECTED_CANDIDATE_MANIFEST:
        raise AssertionError("benchmark is not bound to the selected Ticket 11 candidate")
    request = json.loads((BASELINE_BUNDLE / "request.json").read_text(encoding="utf-8"))
    previous_execution = json.loads(BASELINE_EXECUTION.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="mount-probe-", dir=output_root) as temporary_name:
        input_mount, competition_root = _make_mount(Path(temporary_name) / "kaggle-input")
        started = time.perf_counter()
        bootstrap_baseline = _mounted_files(input_mount, "request.json")
        baseline_bootstrap_seconds = time.perf_counter() - started
        started = time.perf_counter()
        bootstrap_optimized = _mounted_files(input_mount / "datasets", "request.json")
        optimized_bootstrap_seconds = time.perf_counter() - started
        if bootstrap_baseline != bootstrap_optimized or len(bootstrap_optimized) != 1:
            raise AssertionError("namespace-scoped bootstrap lookup changed the selected request")

        baseline_discovery_seconds, baseline_layout = _discover(
            input_mount,
            request,
            namespace_scoped=False,
        )
        optimized_discovery_seconds, optimized_layout = _discover(
            input_mount,
            request,
            namespace_scoped=True,
        )
        for field in (
            "request_checksum",
            "candidate_manifest_checksum",
            "approval_path",
            "competition_root",
        ):
            if getattr(baseline_layout, field) != getattr(optimized_layout, field):
                raise AssertionError(f"namespace-scoped discovery changed {field}")
        if optimized_layout.competition_root != competition_root:
            raise AssertionError("optimized discovery selected a different competition mount")

    output_path = output_root / "submission.csv"
    evidence_path = output_root / "kaggle-execution.json"
    private_evidence_path = output_root / "private-execution.json"
    inference_started = time.perf_counter()
    result = run_kaggle_baseline(
        BASELINE_BUNDLE / "candidate",
        INPUT,
        output_path,
        evidence_path=evidence_path,
        private_evidence_path=private_evidence_path,
        allow_local_smoke=True,
        device="cpu",
    )
    local_inference_wall_seconds = time.perf_counter() - inference_started
    evidence = cast(Mapping[str, Any], result.evidence)
    workload = evidence["workload"]
    resources = evidence["resources"]
    details = evidence["timing"]["detailed"]
    stages = evidence["timing"]["stages"]
    output = evidence["output"]
    fallback = evidence["fallback"]
    if (
        workload.get("study_count") != EXPECTED_STUDIES
        or workload.get("series_count") != EXPECTED_SERIES
        or workload.get("slice_count") != EXPECTED_SLICES
    ):
        raise AssertionError("representative benchmark input counts differ from Ticket 24")
    if output.get("decode_failure_count") != 0 or fallback.get("numerator") != 0:
        raise AssertionError("representative inference had decode failures or fallback predictions")
    if int(resources["peak_ram_bytes"]) > MAX_PEAK_RSS_BYTES:
        raise AssertionError("representative inference exceeded the 8 GiB actual-memory bound")
    with BASELINE_OUTPUT.open(newline="", encoding="utf-8") as handle:
        baseline_rows = list(csv.reader(handle))
    expected_uids = tuple(row[0] for row in baseline_rows[1:])
    validate_submission_csv(output_path, expected_uids)
    if output_path.read_bytes() != BASELINE_OUTPUT.read_bytes():
        raise AssertionError("namespace-scoped candidate output differs from the Ticket 24 v6 CSV")

    projection = _fixed_projection(previous_execution, optimized_discovery_seconds)
    summary: dict[str, Any] = {
        "schema_version": "rsna-knee-ticket25-efficiency-benchmark-v1",
        "status": "passed_local_representative_and_synthetic_mount_probe",
        "optimization": "Search delivery and approval metadata under Kaggle's datasets namespace and competition mounts under competitions; do not walk the competition payload for metadata.",
        "quality_comparator": {
            "candidate_manifest_sha256": candidate.manifest_checksum,
            "baseline_local_csv_sha256": _sha256(BASELINE_OUTPUT),
            "optimized_local_csv_sha256": _sha256(output_path),
            "csv_byte_identical": True,
            "reference_oof_macro_auc": 0.464313,
            "oof_auc_effect": 0.0,
            "oof_recomputed": False,
            "reason": "The change only narrows metadata search roots. Candidate weights, decoding, preprocessing, forward passes, ensemble and score writing are unchanged; representative CSV bytes match.",
        },
        "mount_probe": {
            "synthetic_unrelated_empty_file_count": SYNTHETIC_UNRELATED_FILES,
            "private_dicom_pixels_copied": False,
            "bootstrap_baseline_seconds": baseline_bootstrap_seconds,
            "bootstrap_dataset_namespace_seconds": optimized_bootstrap_seconds,
            "adapter_discovery_baseline_seconds": baseline_discovery_seconds,
            "adapter_discovery_namespaced_seconds": optimized_discovery_seconds,
            "selected_paths_unchanged": True,
            "measurements_are_kaggle_host_runtime": False,
        },
        "representative_inference": {
            "hardware_partition": os.environ.get("SLURM_JOB_PARTITION", "unavailable"),
            "slurm_cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or None,
            "studies": workload["study_count"],
            "series": workload["series_count"],
            "slices": workload["slice_count"],
            "long_tail": workload["long_tail"],
            "observed_wall_seconds": local_inference_wall_seconds,
            "adapter_whole_notebook_seconds": evidence["timing"]["whole_notebook_seconds"],
            "stage_seconds": stages,
            "detailed_stage_seconds": {
                key: details[key]
                for key in (
                    "dicom_read_decode_seconds",
                    "pixel_decode_seconds",
                    "preprocessing_resize_seconds",
                    "preprocessing_packing_seconds",
                    "model_load_seconds",
                    "model_inference_seconds",
                    "csv_generation_validation_seconds",
                )
            },
            "observed_peak_ram_bytes": resources["peak_ram_bytes"],
            "peak_vram_bytes": resources["peak_vram_bytes"],
            "decode_failures": output["decode_failure_count"],
            "fallbacks": fallback["numerator"],
            "local_contract_passed": output["accepted_by_local_contract"],
        },
        "notebook_runtime_projection": projection,
        "package": {
            "model_size_bytes": resources["model_size_bytes"],
            "candidate_bundle_size_bytes": resources["candidate_bundle_size_bytes"],
            "delivery_package_size_bytes": 196_517_501,
            "ensemble_model_count": resources["ensemble_model_count"],
            "adapter_sha256": _sha256(REPO / "src/rsna_knee/kaggle.py"),
        },
        "limitations": [
            "No Ticket 25 notebook was uploaded or executed on Kaggle; the namespaced discovery speedup is measured on a Slurm CPU node using a synthetic directory payload.",
            "The official hidden input had only 3 studies. Its full-scale projection remains an estimate, not a 1,300-study run.",
            "The OOF metric is inherited from the frozen candidate and was not recomputed; this experiment does not alter prediction semantics.",
            "The Ticket 24 stored projection double-counted the study loop with its nested decode, preprocessing and inference stages; this report recomputes it without overlap.",
        ],
    }
    summary_path = output_root / "public-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "bootstrap_baseline_seconds": round(baseline_bootstrap_seconds, 6),
        "bootstrap_namespaced_seconds": round(optimized_bootstrap_seconds, 6),
        "discovery_baseline_seconds": round(baseline_discovery_seconds, 6),
        "discovery_namespaced_seconds": round(optimized_discovery_seconds, 6),
        "representative_studies": workload["study_count"],
        "representative_wall_seconds": round(local_inference_wall_seconds, 3),
        "csv_byte_identical": True,
        "public_summary": str(summary_path.resolve().relative_to(REPO)),
    }, sort_keys=True))
    return summary


def verify_existing(output_root: Path) -> dict[str, Any]:
    summary_path = output_root / "public-summary.json"
    output_path = output_root / "submission.csv"
    if output_root.is_symlink() or summary_path.is_symlink() or output_path.is_symlink():
        raise ValueError("existing benchmark output contains a symlink")
    if not summary_path.is_file() or not output_path.is_file():
        raise FileNotFoundError("existing benchmark summary or submission is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "rsna-knee-ticket25-efficiency-benchmark-v1":
        raise ValueError("existing benchmark summary has an unexpected schema")
    if summary.get("status") != "passed_local_representative_and_synthetic_mount_probe":
        raise ValueError("existing benchmark summary does not record a passed run")
    quality = summary.get("quality_comparator")
    if not isinstance(quality, dict) or quality.get("optimized_local_csv_sha256") != _sha256(output_path):
        raise ValueError("existing benchmark CSV checksum differs from its summary")
    print(json.dumps({
        "status": summary["status"],
        "observed_wall_seconds": summary["representative_inference"]["observed_wall_seconds"],
        "baseline_discovery_seconds": summary["mount_probe"]["adapter_discovery_baseline_seconds"],
        "namespaced_discovery_seconds": summary["mount_probe"]["adapter_discovery_namespaced_seconds"],
        "csv_byte_identical": quality["csv_byte_identical"],
        "public_summary": str(summary_path.resolve().relative_to(REPO)),
    }, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify_existing:
            verify_existing(args.output_root)
        else:
            run(args.output_root)
    except (FileNotFoundError, FileExistsError, InferenceError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
