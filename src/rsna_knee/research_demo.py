"""Generate one public-safe synthetic example from the frozen research candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, UID

from .candidate import CandidateBundle, load_candidate, run_candidate_inference
from .dicom import DecoderPolicy
from .domain import DicomSlice
from .geometry import GeometryPolicy
from .inventory import identify_archive
from .preprocessing import PackingConfig, PreprocessConfig
from .real_inference import IntegrationManifest, IntegrationSeries, IntegrationStudy, write_integration_manifest
from .synthetic import make_synthetic_study
from .targets import SUBMISSION_COLUMNS


EXAMPLE_SEED = 20260926
EXAMPLE_SOURCE_VERSION = "synthetic-ticket26-frozen-example-v1"
DEFAULT_CANDIDATE = Path("artifacts/private/ticket-11-local-candidate-20260911-v1")
@dataclass(frozen=True)
class _SeriesExample:
    series_uid: str
    label: str
    is_synthetic_localizer: bool
    input_shape: tuple[int, int]
    model_shape: tuple[int, int]
    ordered_slices: tuple[tuple[str, float], ...]


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"candidate {field} must be an object")
    return cast(Mapping[str, object], value)


def _count(values: Mapping[str, object], name: str, default: int) -> int:
    value = values.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"candidate count {name!r} must be an integer")
    try:
        return int(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"candidate count {name!r} must be an integer") from exc


def _array(value: object, field: str, length: int) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"candidate {field} must contain {length} values")
    return tuple(value)


def _candidate_policies(
    bundle: CandidateBundle,
) -> tuple[DecoderPolicy, PreprocessConfig, PackingConfig]:
    inference = bundle.inference_config
    decoder = DecoderPolicy.from_dict(inference.get("decoder"))
    preprocessing = _mapping(inference.get("preprocessing"), "preprocessing")
    geometry = GeometryPolicy.from_dict(preprocessing.get("geometry_policy"))
    image_size_values = _array(preprocessing.get("image_size"), "image_size", 2)
    percentiles = _array(preprocessing.get("normalization_percentiles"), "normalization_percentiles", 2)
    preprocess = PreprocessConfig(
        image_size=(int(str(image_size_values[0])), int(str(image_size_values[1]))),
        geometry_policy=geometry,
        normalization_domain=str(preprocessing.get("normalization_domain")),
        normalization_percentiles=(float(str(percentiles[0])), float(str(percentiles[1]))),
        use_voi_lut=preprocessing.get("use_voi_lut") is True,
        interpolation=str(preprocessing.get("interpolation")),
        padding_fill_value=float(str(preprocessing.get("padding_fill_value"))),
        output_dtype=str(preprocessing.get("output_dtype")),
    )
    packing_payload = _mapping(inference.get("packing"), "packing")
    packing = PackingConfig(
        max_slices=int(str(packing_payload.get("max_slices"))),
        max_series=int(str(packing_payload.get("max_series"))),
    )
    return decoder, preprocess, packing


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pixel_array(item: DicomSlice) -> np.ndarray:
    pixels = np.asarray(item.pixel_array)
    if pixels.ndim != 2 or pixels.size == 0:
        raise ValueError("the synthetic example must contain non-empty 2D slices")
    if not np.issubdtype(pixels.dtype, np.integer):
        pixels = np.rint(pixels).astype(np.int32)
    minimum = int(pixels.min())
    maximum = int(pixels.max())
    if minimum < 0 or maximum > 65_535:
        raise ValueError("synthetic pixels are outside the supported unsigned 16-bit range")
    return pixels.astype(np.uint16, copy=False)


def _write_fixture_slice(
    item: DicomSlice,
    destination: Path,
    *,
    localizer: bool,
) -> None:
    pixels = _pixel_array(item)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = UID(item.sop_instance_uid)
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(destination), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = UID(item.sop_instance_uid)
    dataset.StudyInstanceUID = UID(str(item.study_instance_uid))
    dataset.SeriesInstanceUID = UID(str(item.series_instance_uid))
    dataset.Modality = "MR"
    dataset.ImagePositionPatient = list(item.image_position or (0.0, 0.0, 0.0))
    dataset.ImageOrientationPatient = list(item.image_orientation or (1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    dataset.InstanceNumber = int(item.instance_number or 1)
    dataset.SeriesDescription = "SYNTHETIC LOCALIZER" if localizer else "SYNTHETIC DIAGNOSTIC"
    dataset.ImageType = ["ORIGINAL", "PRIMARY", "LOCALIZER"] if localizer else ["ORIGINAL", "PRIMARY"]
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = item.photometric_interpretation
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.RescaleSlope = float(item.rescale_slope)
    dataset.RescaleIntercept = float(item.rescale_intercept)
    dataset.PixelData = pixels.tobytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_as(destination, enforce_file_format=True)


def _build_synthetic_source(
    temporary_root: Path,
    bundle: CandidateBundle,
) -> tuple[Path, Path, tuple[_SeriesExample, ...], str]:
    decoder, preprocess, packing = _candidate_policies(bundle)
    study = make_synthetic_study(
        "ticket-26-generated-example",
        series_count=3,
        slices_per_series=12,
        seed=EXAMPLE_SEED,
    )
    ordered_source_series = tuple(sorted(study.series, key=lambda item: item.series_instance_uid))
    source_root = temporary_root / "synthetic-source"
    archive_path = temporary_root / "synthetic-source.zip"
    integration_series: list[IntegrationSeries] = []
    series_examples: list[_SeriesExample] = []

    for series_index, series in enumerate(ordered_source_series):
        label = f"Series {chr(ord('A') + series_index)}"
        localizer = series_index == len(ordered_source_series) - 1
        ordered_slices = tuple(
            sorted(
                series.slices,
                key=lambda item: (
                    item.image_position[2] if item.image_position is not None else 0.0,
                    item.sop_instance_uid,
                ),
            )
        )
        aliases = tuple(
            (
                f"Slice {slice_index + 1:02d}",
                float(item.image_position[2]) if item.image_position is not None else float(slice_index),
            )
            for slice_index, item in enumerate(ordered_slices)
        )
        first_shape = _pixel_array(series.slices[0]).shape
        if any(_pixel_array(item).shape != first_shape for item in series.slices):
            raise ValueError("each synthetic series must have consistent image dimensions")
        entries: list[str] = []
        dimensions: set[str] = set()
        for item in series.slices:
            relative = PurePosixPath("synthetic") / label.lower().replace(" ", "-") / f"{item.sop_instance_uid}.dcm"
            target = source_root.joinpath(*relative.parts)
            _write_fixture_slice(item, target, localizer=localizer)
            entries.append(relative.as_posix())
            rows, columns = _pixel_array(item).shape
            dimensions.add(f"{rows}x{columns}")
        integration_series.append(
            IntegrationSeries(
                study_instance_uid=study.study_instance_uid,
                series_instance_uid=series.series_instance_uid,
                entries=tuple(entries),
                diagnostic_status="unknown",
                expected_transfer_syntaxes=(str(ExplicitVRLittleEndian),),
                expected_dimensions=tuple(sorted(dimensions)),
                expected_frame_counts=(1,),
                expected_entry_count=len(entries),
            )
        )
        series_examples.append(
            _SeriesExample(
                series_uid=series.series_instance_uid,
                label=label,
                is_synthetic_localizer=localizer,
                input_shape=(int(first_shape[0]), int(first_shape[1])),
                model_shape=preprocess.image_size,
                ordered_slices=aliases,
            )
        )

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source_file in sorted(source_root.rglob("*.dcm")):
            archive.write(source_file, source_file.relative_to(source_root).as_posix())
    archive_identity = identify_archive(
        archive_path,
        source_version=EXAMPLE_SOURCE_VERSION,
    )
    manifest_path = temporary_root / "integration-manifest.json"
    manifest = IntegrationManifest(
        source_version=EXAMPLE_SOURCE_VERSION,
        archive_fingerprint=archive_identity.fingerprint,
        geometry_policy=preprocess.geometry_policy,
        decoder_policy=decoder,
        preprocess_config=preprocess,
        studies=(IntegrationStudy(study.study_instance_uid, tuple(integration_series)),),
        packing_config=packing,
    )
    write_integration_manifest(manifest, manifest_path)
    return archive_path, manifest_path, tuple(series_examples), archive_identity.fingerprint


def _read_json(path: Path, description: str) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, description)


def _summarize_series(
    private_study: Mapping[str, object],
    examples: tuple[_SeriesExample, ...],
) -> list[dict[str, object]]:
    source_series_value = private_study.get("series")
    if not isinstance(source_series_value, list):
        raise ValueError("candidate private evidence has no series list")
    source_series: dict[str, Mapping[str, object]] = {}
    for raw_item in source_series_value:
        item = _mapping(raw_item, "private series evidence")
        uid = str(item.get("SeriesInstanceUID", ""))
        source_series[uid] = item

    prediction = _mapping(private_study.get("prediction"), "private prediction")
    rejected_value = prediction.get("rejected_series")
    if not isinstance(rejected_value, list):
        raise ValueError("candidate private evidence has no rejection list")
    rejection_reasons: dict[str, str] = {}
    for raw_item in rejected_value:
        item = _mapping(raw_item, "series rejection")
        rejection_reasons[str(item.get("series_instance_uid", ""))] = str(item.get("reason", "rejected"))

    accepted_examples = [
        example
        for example in examples
        if str(source_series.get(example.series_uid, {}).get("status")) == "accepted"
        and example.series_uid not in rejection_reasons
    ]
    accepted_examples.sort(key=lambda item: item.series_uid)
    sampled_value = prediction.get("sampled_source_indices")
    if not isinstance(sampled_value, list):
        raise ValueError("candidate private evidence has no sampling record")
    sampled_indices: dict[str, tuple[int, ...]] = {}
    active_arrays = [value for value in sampled_value if isinstance(value, list) and value]
    if len(active_arrays) != len(accepted_examples):
        raise ValueError("candidate sampling record does not match admitted series")
    for example, raw_indices in zip(accepted_examples, active_arrays):
        sampled_indices[example.series_uid] = tuple(int(str(index)) for index in cast(list[object], raw_indices))

    summaries: list[dict[str, object]] = []
    for example in examples:
        series_evidence = source_series.get(example.series_uid)
        if series_evidence is None:
            raise ValueError("candidate evidence omitted a generated series")
        indices = sampled_indices.get(example.series_uid, ())
        if any(index < 0 or index >= len(example.ordered_slices) for index in indices):
            raise ValueError("candidate selected an out-of-range synthetic slice")
        selected = [example.ordered_slices[index] for index in indices]
        status = str(series_evidence.get("status", "unknown"))
        rejection = rejection_reasons.get(example.series_uid)
        if example.is_synthetic_localizer and status == "excluded_non_diagnostic":
            rejection = "synthetic SeriesDescription/ImageType localizer marker"
        summaries.append(
            {
                "label": example.label,
                "status": status,
                "decoded_slices": int(str(series_evidence.get("decoded_slice_count", 0))),
                "expected_slices": int(str(series_evidence.get("expected_slice_count", 0))),
                "selected": selected,
                "not_sampled": [
                    pair
                    for index, pair in enumerate(example.ordered_slices)
                    if index not in set(indices)
                ],
                "rejection": rejection,
                "warnings": series_evidence.get("warnings", []),
                "preprocess_shape": (
                    f"{example.input_shape[0]}×{example.input_shape[1]}"
                    f" → {example.model_shape[0]}×{example.model_shape[1]}"
                ),
            }
        )
    return summaries


def _render_report(
    bundle: CandidateBundle,
    candidate_evidence: Mapping[str, object],
    private_evidence: Mapping[str, object],
    examples: tuple[_SeriesExample, ...],
    scores: Mapping[str, float],
    source_fingerprint: str,
) -> str:
    studies_value = private_evidence.get("studies")
    if not isinstance(studies_value, list) or len(studies_value) != 1:
        raise ValueError("the demonstration must contain exactly one synthetic study")
    private_study = _mapping(studies_value[0], "synthetic study evidence")
    series_summaries = _summarize_series(private_study, examples)
    counts = _mapping(candidate_evidence.get("counts"), "candidate counts")
    policies = _mapping(candidate_evidence.get("policies"), "candidate policies")
    preprocessing = _mapping(policies.get("preprocessing"), "preprocessing policy")
    geometry = _mapping(preprocessing.get("geometry_policy"), "geometry policy")
    packing = _mapping(policies.get("packing"), "packing policy")
    decoder = _mapping(policies.get("decoder"), "decoder policy")
    provenance = _mapping(bundle.manifest.get("provenance"), "candidate provenance")
    candidate_environment = _mapping(bundle.manifest.get("environment"), "candidate environment")
    promotion = _mapping(bundle.manifest.get("promotion"), "candidate promotion metadata")
    model = _mapping(bundle.manifest.get("model"), "candidate model")
    model_configuration = _mapping(model.get("configuration"), "model configuration")
    candidate_execution = _mapping(candidate_evidence.get("candidate_execution"), "candidate execution")
    environment = _mapping(candidate_evidence.get("environment"), "execution environment")

    member_rows: list[str] = []
    members_value = model.get("members")
    if not isinstance(members_value, list):
        raise ValueError("candidate has no model members")
    for raw_member in members_value:
        member = _mapping(raw_member, "candidate model member")
        member_rows.append(
            "| {fold} | `{run_id}` | `{sha256}` |".format(
                fold=member.get("fold"),
                run_id=member.get("run_id"),
                sha256=member.get("sha256"),
            )
        )

    series_rows: list[str] = []
    selected_total = 0
    for summary in series_summaries:
        selected_items = cast(list[tuple[str, float]], summary["selected"])
        selected_total += len(selected_items)
        selected_text = ", ".join(f"{label} (z={position:g} mm)" for label, position in selected_items) or "none"
        not_sampled = cast(list[tuple[str, float]], summary["not_sampled"])
        not_sampled_text = ", ".join(label for label, _ in not_sampled) or "none"
        decision = str(summary["status"])
        if summary["rejection"]:
            decision += f" ({summary['rejection']})"
        series_rows.append(
            "| {label} | {shape} | {decoded}/{expected} | {decision} | {selected} | {omitted} |".format(
                label=summary["label"],
                shape=summary["preprocess_shape"],
                decoded=summary["decoded_slices"],
                expected=summary["expected_slices"],
                decision=decision,
                selected=selected_text,
                omitted=not_sampled_text,
            )
        )

    score_rows = [f"| {name} | {scores[name]:.6f} |" for name in SUBMISSION_COLUMNS[1:]]
    failed = (
        len(series_summaries) != 3
        or _count(counts, "input_studies", 0) != 1
        or _count(counts, "fallback_studies", -1) != 0
        or _count(counts, "rejected_series", -1) != 0
        or _count(counts, "usable_series", -1) != 2
        or _count(counts, "excluded_non_diagnostic_series", -1) != 1
        or _count(counts, "expected_slices", -1) != 36
        or _count(counts, "decoded_slices", -1) != 36
        or sum(str(summary["status"]) == "accepted" for summary in series_summaries) != 2
        or sum(str(summary["status"]) == "excluded_non_diagnostic" for summary in series_summaries) != 1
        or selected_total != 16
        or not bool(counts.get("scores_finite"))
        or not bool(counts.get("scores_bounded"))
    )
    if failed:
        raise ValueError("synthetic example did not satisfy the expected one-study inference checks")

    configuration_path = bundle.root / "configuration" / "inference.json"
    configuration_sha = _sha256_file(configuration_path)
    source_code_sha = _sha256_file(Path(__file__).resolve())
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    excluded_series = _count(counts, "excluded_non_diagnostic_series", 0)
    expected_slices = _count(counts, "expected_slices", 0)
    decoded_slices = _count(counts, "decoded_slices", 0)
    wall_seconds = float(str(candidate_execution.get("whole_pipeline_seconds", 0.0)))
    geometry_version = str(geometry.get("policy_version", "unknown"))
    promotion_status = str(promotion.get("status", "unknown"))

    return f"""# Ticket 26: Frozen synthetic research demonstration

Generated: {generated_at}

Scope: one deterministic synthetic knee-like DICOM example, processed with the selected frozen candidate. No patient data, competition MRI, reports, or patient-derived identifiers are included in this report.

## Demonstration trace

```text
3 generated DICOM series / 36 slices
  -> pinned pydicom decoder: {decoded_slices}/{expected_slices} slices decoded
  -> geometry ordering and per-slice intensity normalization
  -> 2 diagnostic series retained; {excluded_series} synthetic localizer excluded
  -> {selected_total} model slices packed at {packing.get('max_slices')} per series
  -> three frozen fold models; mean-probability aggregation
  -> one row with twelve finite scores in [0, 1]
```

The source slices were generated from fixed gradients and seeded noise, written to a temporary explicit-VR little-endian DICOM archive, and deleted after inference. One generated series carries an explicit `SYNTHETIC LOCALIZER` marker to make the exclusion path visible. The example is deliberately small and is not representative clinical imaging.

## Series, preprocessing, and slice selection

The model uses the candidate's fixed per-slice preprocessing configuration: geometry policy `{geometry_version}`, minimum {geometry.get('minimum_usable_slices')} usable slices, valid-pixel min–max normalization, nearest-neighbor resizing to 128×128, and float32 model input. The packer admits up to {packing.get('max_series')} series and samples at most {packing.get('max_slices')} geometry-ordered slices from each series. Decoder policy: pydicom {decoder.get('pydicom_version')}, explicit VR little endian.

| Synthetic series | Input → model image size | Decoded slices | Decision | Geometry-ordered slices selected | Not sampled or excluded |
| --- | --- | ---: | --- | --- | --- |
{chr(10).join(series_rows)}

No series failed preprocessing. The localizer was decoded but excluded by its synthetic metadata marker. Within each retained series, slices omitted from the packed tensor were omitted by the frozen eight-slice sampling cap; they were not decode failures.

## Model scores

These values are uncalibrated model confidence scores for this synthetic input. They are not calibrated probabilities of disease, diagnostic conclusions, or validated risk estimates.

| Target | Score |
| --- | ---: |
{chr(10).join(score_rows)}

## Frozen model and provenance

| Field | Value |
| --- | --- |
| Candidate | `{bundle.candidate_id}` |
| Candidate manifest SHA-256 | `{bundle.manifest_checksum}` |
| Candidate promotion metadata | `{promotion_status}` (bundle metadata; the later Ticket 24 delivery record is separate) |
| Model configuration | `{json.dumps(dict(model_configuration), sort_keys=True)}` |
| Aggregation | `{model.get('aggregation')}` across {len(members_value)} selected fold checkpoints |
| Inference configuration SHA-256 | `{configuration_sha}` |
| Candidate shared-code digest | `{candidate_environment.get('shared_code_digest', 'unavailable')}` |
| Candidate source revision | `{provenance.get('code_revision')}` |
| Fold manifest SHA-256 | `{provenance.get('fold_manifest_checksum')}` |
| OOF evidence SHA-256 | `{provenance.get('oof_evidence_sha256')}` |
| Synthetic input archive fingerprint | `{source_fingerprint}` |
| Demo implementation SHA-256 | `{source_code_sha}` |
| Execution | `{environment.get('execution')}; Python {environment.get('python_version')}; NumPy {environment.get('numpy_version')}; PyTorch {environment.get('torch_version')}` |
| Whole pipeline wall time | {wall_seconds:.3f} seconds; one synthetic study only |
| Reports / network | Reports not required; runtime network download disabled |

| Fold | Run identity | Checkpoint SHA-256 |
| ---: | --- | --- |
{chr(10).join(member_rows)}

## Interpretation and limitations

- This verifies one synthetic execution path and the output contract. The generated pixel patterns do not represent patient anatomy or disease.
- The Ticket 11 candidate's development evidence is three-fold OOF on 58 reference-labelled studies; repeated model selection means it is not an untouched internal test. Study-only splitting does not establish patient-level separation.
- The scores are uncalibrated, and this demonstration provides no clinical validation, diagnostic accuracy claim, or clinical-use approval. It is for research and education only.
- The report contains aggregate processing details, synthetic slice labels, and model hashes only. Per-slice DICOM identifiers and temporary input files were not retained.
- Ticket 24 records the separately approved Kaggle delivery of this frozen candidate. That delivery evidence does not turn this synthetic demonstration into full-scale runtime or clinical evidence.
"""


def generate_report(
    candidate_root: str | Path,
    report_path: str | Path,
    *,
    require_slurm: bool = False,
    allow_local_smoke: bool = False,
) -> dict[str, object]:
    """Run the frozen model on a synthetic example and publish a Markdown report."""

    if require_slurm == allow_local_smoke:
        raise ValueError("choose exactly one execution mode: --require-slurm or --allow-local-smoke")
    report = Path(report_path).expanduser().resolve()
    repository_root = Path(__file__).resolve().parents[2]
    try:
        report.relative_to(repository_root)
    except ValueError:
        pass
    else:
        raise ValueError("the Ticket 26 report must be written outside the RSNA repository")
    if report.suffix.casefold() != ".md":
        raise ValueError("the Ticket 26 report path must use the .md extension")
    if report.exists() or report.is_symlink():
        raise FileExistsError(f"refusing to overwrite an existing report: {report}")

    bundle = load_candidate(candidate_root)
    report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ticket26-frozen-demo-") as temporary_name:
        temporary_root = Path(temporary_name)
        source, manifest, examples, source_fingerprint = _build_synthetic_source(temporary_root, bundle)
        output = temporary_root / "submission.csv"
        public_evidence_path = temporary_root / "public-evidence.json"
        private_evidence_path = temporary_root / "private-evidence.json"
        result = run_candidate_inference(
            bundle.root,
            manifest,
            source,
            output,
            evidence_path=public_evidence_path,
            private_evidence_path=private_evidence_path,
            require_slurm=require_slurm,
            allow_local_smoke=allow_local_smoke,
        )
        public_evidence = _read_json(public_evidence_path, "public candidate evidence")
        private_evidence = _read_json(private_evidence_path, "private synthetic evidence")
        with output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1 or tuple(rows[0].keys()) != SUBMISSION_COLUMNS:
            raise ValueError("synthetic inference did not produce the exact one-row submission schema")
        scores = {name: float(rows[0][name]) for name in SUBMISSION_COLUMNS[1:]}
        score_values = np.asarray(tuple(scores.values()), dtype=np.float64)
        if not np.isfinite(score_values).all() or np.any(score_values < 0.0) or np.any(score_values > 1.0):
            raise ValueError("synthetic inference returned a non-finite or out-of-range score")
        markdown = _render_report(
            bundle,
            public_evidence,
            private_evidence,
            examples,
            scores,
            source_fingerprint,
        )

    with report.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown)
    private_studies = private_evidence.get("studies")
    if not isinstance(private_studies, list) or len(private_studies) != 1:
        raise ValueError("the demonstration must contain exactly one private synthetic study record")
    selected_model_slices = sum(
        len(cast(list[tuple[str, float]], summary["selected"]))
        for summary in _summarize_series(
            _mapping(private_studies[0], "private study"),
            examples,
        )
    )
    counts = _mapping(result.evidence.get("counts"), "candidate counts")
    return {
        "status": "completed_synthetic_frozen_example",
        "candidate_id": bundle.candidate_id,
        "candidate_manifest_checksum": bundle.manifest_checksum,
        "report_path": str(report),
        "report_sha256": _sha256_file(report),
        "input_studies": 1,
        "manifest_series": 3,
        "usable_series": _count(counts, "usable_series", 0),
        "decoded_slices": _count(counts, "decoded_slices", 0),
        "selected_model_slices": selected_model_slices,
        "fallback_studies": _count(counts, "fallback_studies", 0),
        "scores_finite": bool(counts.get("scores_finite")),
        "scores_bounded": bool(counts.get("scores_bounded")),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--report", type=Path, required=True, help="new Markdown report path outside the repository")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--require-slurm",
        action="store_true",
        help="require the bounded cluster1-cpu Slurm allocation",
    )
    mode.add_argument(
        "--allow-local-smoke",
        action="store_true",
        help="permit this one-study synthetic smoke run outside Slurm",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = generate_report(
            args.candidate_root,
            args.report,
            require_slurm=args.require_slurm,
            allow_local_smoke=args.allow_local_smoke,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ticket-26 research demonstration failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
