"""Private real-training cohort cache and fast checkpoint memorization check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Mapping, cast

import numpy as np
import torch

from rsna_knee.evaluation import roc_auc_score
from rsna_knee.folds import load_fold_manifest
from rsna_knee.image_diagnostic import (
    COHORT_SIZE,
    FOLD_ID,
    _check_inputs,
    _evaluate,
    _example_tensors,
    _example_summary,
    _pixel_and_geometry_summary,
    _verify_manifest_identity,
    load_diagnostic_plan,
)
from rsna_knee.inventory import identify_archive
from rsna_knee.model import ModelConfig, build_study_model
from rsna_knee.real_inference import iter_real_studies, load_integration_manifest
from rsna_knee.targets import TARGET_COLUMNS
from rsna_knee.training import (
    SupervisionProvenance,
    TrainingExample,
    build_fold_training_examples,
    load_training_checkpoint,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as target:
        json.dump(value, target, sort_keys=True, separators=(",", ":"))
        target.write("\n")


def prepare(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    output.chmod(0o700)
    plan_path = Path(args.plan)
    integration_path = Path(args.integration_manifest)
    fold_path = Path(args.fold_manifest)
    selection_path = Path(args.selection_report)
    archive_path = Path(args.source_archive)
    plan = load_diagnostic_plan(plan_path)
    integration = load_integration_manifest(integration_path)
    folds = load_fold_manifest(fold_path)
    selected, _ = _verify_manifest_identity(
        plan, integration, folds, integration_path, fold_path, selection_path
    )
    source = cast(Mapping[str, object], plan["source"])
    if archive_path.stat().st_size != source["source_archive_bytes"]:
        raise RuntimeError("source archive size differs from frozen plan")
    if identify_archive(archive_path, source_version=integration.source_version).fingerprint != integration.archive_fingerprint:
        raise RuntimeError("source archive fingerprint differs from frozen plan")
    loaded = tuple(
        iter_real_studies(
            integration,
            archive_path,
            require_complete_series=True,
            study_uids=selected,
        )
    )
    pixel_summary, _ = _pixel_and_geometry_summary(loaded)
    studies_by_uid = {item.study.study_instance_uid: item.study for item in loaded}
    examples = build_fold_training_examples(
        studies_by_uid,
        folds,
        FOLD_ID,
        preprocess_config=integration.preprocess_config,
        packing_config=integration.packing_config,
        allow_partial_inputs=False,
        selected_training_study_uids=selected,
    )
    example_summary = _example_summary(examples, expected_ids=selected)
    checks = _check_inputs(pixel_summary, example_summary)
    if len(examples) != COHORT_SIZE or not all(checks.values()):
        raise RuntimeError("frozen real training cohort failed the input checks")
    if any(item.provenance.source_kind != "reference" or not item.provenance.held_out_excluded for item in examples):
        raise RuntimeError("probe cache includes non-reference or held-out supervision")
    arrays = {
        "uids": np.asarray([item.study_instance_uid for item in examples]),
        "images": np.stack([item.images for item in examples]),
        "slice_masks": np.stack([item.slice_mask for item in examples]),
        "series_masks": np.stack([item.series_mask for item in examples]),
        "targets": np.stack([item.targets for item in examples]),
        "supervision_masks": np.stack([item.supervision_mask for item in examples]),
    }
    cache_path = output / "training-only-cohort.npz"
    fd = os.open(cache_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as target:
        np.savez_compressed(
            target,
            uids=arrays["uids"],
            images=arrays["images"],
            slice_masks=arrays["slice_masks"],
            series_masks=arrays["series_masks"],
            targets=arrays["targets"],
            supervision_masks=arrays["supervision_masks"],
        )
    metadata: dict[str, object] = {
        "schema_version": "rsna-ticket29-private-memorization-probe-v1",
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": _sha256(plan_path),
        "integration_file_sha256": _sha256(integration_path),
        "fold_file_sha256": _sha256(fold_path),
        "selection_file_sha256": _sha256(selection_path),
        "cache_file_sha256": _sha256(cache_path),
        "training_studies": len(examples),
        "held_out_studies": 0,
        "target_order": list(TARGET_COLUMNS),
        "packed_valid_slices": example_summary["valid_slice_mask_count"],
        "real_decoded_slices": pixel_summary["decoded_slices_in_accepted_series"],
    }
    _write_new_json(output / "metadata.json", metadata)
    print(json.dumps({key: metadata[key] for key in (
        "plan_sha256", "cache_file_sha256", "training_studies",
        "held_out_studies", "packed_valid_slices", "real_decoded_slices"
    )}, sort_keys=True))
    return 0


def load_cache(directory: Path) -> tuple[dict[str, object], tuple[TrainingExample, ...]]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if _sha256(directory / "training-only-cohort.npz") != metadata["cache_file_sha256"]:
        raise RuntimeError("private probe cache checksum changed")
    with np.load(directory / "training-only-cohort.npz", allow_pickle=False) as arrays:
        examples = tuple(
            TrainingExample(
                study_instance_uid=str(arrays["uids"][index]),
                images=arrays["images"][index],
                slice_mask=arrays["slice_masks"][index],
                series_mask=arrays["series_masks"][index],
                targets=arrays["targets"][index],
                supervision_mask=arrays["supervision_masks"][index],
                provenance=SupervisionProvenance(fold=FOLD_ID),
            )
            for index in range(COHORT_SIZE)
        )
    if len(examples) != metadata["training_studies"]:
        raise RuntimeError("private probe cache membership changed")
    return metadata, examples


def check(args: argparse.Namespace) -> int:
    metadata, examples = load_cache(Path(args.cache))
    checkpoint_path = Path(args.checkpoint)
    payload = load_training_checkpoint(checkpoint_path)
    identity = payload.get("identity")
    if not isinstance(identity, Mapping) or identity.get("diagnostic_plan_sha256") != metadata["plan_sha256"]:
        raise RuntimeError("checkpoint and private probe cache use different frozen plans")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError("checkpoint configuration is missing")
    model = build_study_model(ModelConfig.from_dict(config.get("model")))
    load_training_checkpoint(checkpoint_path, model=model)
    result = _evaluate(model, examples)
    auc = result["supported_target_macro_auc"]
    if not isinstance(auc, float) or result["supported_targets"] != len(TARGET_COLUMNS):
        raise RuntimeError("training cohort AUC is undefined for a target")
    public = {
        "checkpoint_sha256": _sha256(checkpoint_path),
        "cache_sha256": metadata["cache_file_sha256"],
        "training_studies": len(examples),
        "masked_bce": result["training_cohort_mean_masked_bce"],
        "training_macro_auc": auc,
        "required_training_macro_auc": args.require_auc,
        "gate_passed": auc >= args.require_auc,
    }
    print(json.dumps(public, sort_keys=True))
    return 0 if public["gate_passed"] else 1


def analyze(args: argparse.Namespace) -> int:
    """Measure whether the saved image representation separates the eight studies."""

    metadata, examples = load_cache(Path(args.cache))
    checkpoint_path = Path(args.checkpoint)
    payload = load_training_checkpoint(checkpoint_path)
    identity = payload.get("identity")
    if not isinstance(identity, Mapping) or identity.get("diagnostic_plan_sha256") != metadata["plan_sha256"]:
        raise RuntimeError("checkpoint and private probe cache use different frozen plans")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError("checkpoint configuration is missing")
    model = build_study_model(ModelConfig.from_dict(config.get("model")))
    load_training_checkpoint(checkpoint_path, model=model)
    model.eval()
    embeddings: list[np.ndarray] = []
    logits: list[np.ndarray] = []

    def capture(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        embeddings.append(inputs[0].detach().cpu().numpy())

    hook = model.head.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            for start in range(0, len(examples), 2):
                images, slice_mask, series_mask, _, _ = _example_tensors(examples[start : start + 2])
                logits.append(model(images, slice_mask, series_mask).cpu().numpy())
    finally:
        hook.remove()
    feature_matrix = np.concatenate(embeddings)
    logit_matrix = np.concatenate(logits)
    centered = feature_matrix - feature_matrix.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    relative_rank = int(np.count_nonzero(singular_values > singular_values[0] * 1e-5))
    labels = np.stack([item.targets for item in examples]).astype(int)
    oracle_targets = np.where(labels == 1, 3.0, -3.0)
    design = np.column_stack((np.ones(len(examples)), feature_matrix))
    coefficients, *_ = np.linalg.lstsq(design, oracle_targets, rcond=None)
    oracle_logits = design @ coefficients
    oracle_aucs = [
        roc_auc_score(labels[:, index].tolist(), oracle_logits[:, index].tolist())
        for index in range(len(TARGET_COLUMNS))
    ]
    if any(value is None for value in oracle_aucs):
        raise RuntimeError("oracle head cannot evaluate all twelve targets")
    result = {
        "cache_sha256": metadata["cache_file_sha256"],
        "checkpoint_sha256": _sha256(checkpoint_path),
        "study_count": len(examples),
        "feature_width": feature_matrix.shape[1],
        "centered_feature_rank": relative_rank,
        "centered_feature_singular_values": singular_values.tolist(),
        "mean_feature_l2": float(np.linalg.norm(feature_matrix, axis=1).mean()),
        "max_pairwise_feature_l2": max(
            float(np.linalg.norm(feature_matrix[left] - feature_matrix[right]))
            for left in range(len(examples)) for right in range(left + 1, len(examples))
        ),
        "mean_target_logit_std": float(logit_matrix.std(axis=0).mean()),
        "oracle_linear_head_training_macro_auc": float(np.mean(cast(list[float], oracle_aucs))),
        "oracle_head_coefficient_l2": float(np.linalg.norm(coefficients)),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--plan", required=True)
    prepare_parser.add_argument("--integration-manifest", required=True)
    prepare_parser.add_argument("--fold-manifest", required=True)
    prepare_parser.add_argument("--selection-report", required=True)
    prepare_parser.add_argument("--source-archive", required=True)
    prepare_parser.add_argument("--output", required=True)
    check_parser = commands.add_parser("check")
    check_parser.add_argument("--cache", required=True)
    check_parser.add_argument("--checkpoint", required=True)
    check_parser.add_argument("--require-auc", type=float, default=0.85)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--cache", required=True)
    analyze_parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            return prepare(args)
        if args.command == "check":
            return check(args)
        return analyze(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"memorization probe failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
