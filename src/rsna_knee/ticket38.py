"""Private, preregistered OOF analysis for the frozen Ticket 38 DINOv2 branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence, cast

import numpy as np
import torch

from .archive import atomic_json_write
from .candidate import _checkpoint_contract
from .dinov2 import (
    DINOV2_FEATURE_DIM,
    DINOV2_SOURCE_REVISION,
    DINOV2_WEIGHTS_SHA256,
    Dinov2StudyModel,
)
from .evaluation import predict_training_examples
from .feature_cache import FeatureCacheError, FrozenFeatureCache
from .folds import load_fold_manifest
from .image_model_comparison import compare_image_model_predictions
from .image_variant import TICKET38_DINOV2_VARIANT, image_model_training_config
from .model import build_study_model
from .oof import (
    OOFArtifactError,
    OOFRunRecord,
    aggregate_oof_run_records,
    load_oof_run_record,
    write_oof_reports,
)
from .real_inference import iter_real_studies, load_integration_manifest
from .series_cache import SeriesCache
from .training import (
    TrainingDataError,
    _attach_cached_frozen_features,
    build_fold_held_out_examples,
    load_training_checkpoint,
)


BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260911
PERMUTATION_RESAMPLES = 10000


class Ticket38AnalysisError(RuntimeError):
    """Raised when Ticket 38 run records cannot support its frozen comparison."""


def _run_paths(root: Path) -> tuple[Path, ...]:
    return tuple(root / f"fold-{fold}" / "run.json" for fold in range(3))


def _merge_predictions(records: Sequence[OOFRunRecord]) -> dict[str, tuple[float, ...]]:
    predictions: dict[str, tuple[float, ...]] = {}
    for record in records:
        overlap = set(predictions).intersection(record.predictions)
        if overlap:
            raise Ticket38AnalysisError("fold records contain duplicate held-out studies")
        predictions.update(record.predictions)
    return predictions


def _configuration_identity(record: OOFRunRecord) -> Mapping[str, object]:
    """Read private identity fields from configuration after OOF validation."""

    path = record.path.parent / "configuration.json"
    try:
        configuration = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Ticket38AnalysisError("validated OOF configuration is unreadable") from exc
    if not isinstance(configuration, Mapping):
        raise Ticket38AnalysisError("validated OOF configuration is not an object")
    identity = configuration.get("identity")
    if not isinstance(identity, Mapping):
        raise Ticket38AnalysisError("validated OOF configuration has no private identity")
    if configuration.get("run_id") != record.run_id:
        raise Ticket38AnalysisError("validated OOF configuration identity changed after aggregation")
    return cast(Mapping[str, object], identity)


def _require_dinov2_records(records: Sequence[OOFRunRecord]) -> None:
    for record in records:
        identity = _configuration_identity(record)
        if identity.get("source_version") != "rsna-2026-08-04":
            raise Ticket38AnalysisError("DINOv2 fold source version differs from the approved source")
        if identity.get("fold_manifest_checksum") is None:
            raise Ticket38AnalysisError("DINOv2 fold record is missing its fold identity")
        initialization = identity.get("initialization")
        if not isinstance(initialization, Mapping):
            raise Ticket38AnalysisError("DINOv2 fold record is missing encoder provenance")
        if (
            initialization.get("sha256") != DINOV2_WEIGHTS_SHA256
            or initialization.get("source_revision") != DINOV2_SOURCE_REVISION
        ):
            raise Ticket38AnalysisError("DINOv2 fold record does not pin its encoder artifact")
        if not isinstance(identity.get("frozen_feature_cache"), Mapping):
            raise Ticket38AnalysisError("DINOv2 fold record is missing frozen-feature cache identity")


def _cache_identity(record: OOFRunRecord) -> Mapping[str, object]:
    identity_record = _configuration_identity(record)
    evidence = identity_record.get("frozen_feature_cache")
    if not isinstance(evidence, Mapping):
        raise Ticket38AnalysisError("DINOv2 fold record has no feature-cache evidence")
    identity = evidence.get("identity")
    if not isinstance(identity, Mapping):
        raise Ticket38AnalysisError("DINOv2 feature-cache identity is malformed")
    return cast(Mapping[str, object], identity)


def _verify_ticket32_baseline(plan_path: Path, baseline_runs_root: Path) -> tuple[Path, ...]:
    """Require the exact Ticket 32 run IDs and digests frozen in the plan."""

    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Ticket38AnalysisError("frozen Ticket 38 plan is unreadable") from exc
    if not isinstance(plan, Mapping):
        raise Ticket38AnalysisError("frozen Ticket 38 plan is not an object")
    expected_plan_digest = plan.get("plan_sha256")
    canonical = json.dumps(
        {key: value for key, value in plan.items() if key != "plan_sha256"},
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != expected_plan_digest:
        raise Ticket38AnalysisError("frozen Ticket 38 plan digest does not match its contents")
    baseline = plan.get("baseline")
    if not isinstance(baseline, Mapping):
        raise Ticket38AnalysisError("frozen Ticket 38 plan has no Ticket 32 comparator")
    entries = baseline.get("ticket32_resnet18_runs")
    if not isinstance(entries, list) or len(entries) != 3:
        raise Ticket38AnalysisError("frozen Ticket 38 plan must name exactly three Ticket 32 runs")
    project_root = plan_path.resolve().parents[3]
    expected_paths: list[Path] = []
    for fold, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or entry.get("fold") != fold:
            raise Ticket38AnalysisError("frozen Ticket 32 comparator folds are invalid")
        relative = entry.get("run_path")
        expected_run_id = entry.get("run_id")
        expected_digest = entry.get("run_sha256")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise Ticket38AnalysisError("frozen Ticket 32 run path is invalid")
        planned_path = (project_root / relative).resolve()
        try:
            planned_path.relative_to(project_root)
        except ValueError as exc:
            raise Ticket38AnalysisError("frozen Ticket 32 run path escapes the project") from exc
        current_path = baseline_runs_root / f"fold-{fold}" / "run.json"
        if planned_path != current_path.resolve():
            raise Ticket38AnalysisError("Ticket 32 baseline directory differs from the frozen plan")
        if not current_path.is_file() or current_path.is_symlink():
            raise Ticket38AnalysisError("frozen Ticket 32 baseline run is missing or unsafe")
        digest = hashlib.sha256(current_path.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise Ticket38AnalysisError("Ticket 32 baseline run digest differs from the frozen plan")
        try:
            run = json.loads(current_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise Ticket38AnalysisError("frozen Ticket 32 baseline run is unreadable") from exc
        if not isinstance(run, Mapping) or run.get("run_id") != expected_run_id:
            raise Ticket38AnalysisError("Ticket 32 baseline run ID differs from the frozen plan")
        expected_paths.append(current_path)
    return tuple(expected_paths)


def _markdown_comparison(
    comparison: Mapping[str, object],
    *,
    candidate_name: str,
    baseline_name: str,
) -> str:
    delta = cast(Mapping[str, object], comparison["delta"])
    interval = cast(
        Mapping[str, object],
        cast(Mapping[str, object], comparison["paired_study_bootstrap"])["macro_auc"],
    )
    base = cast(Mapping[str, object], comparison["baseline"])
    candidate = cast(Mapping[str, object], comparison["candidate"])
    rows = [
        "# Ticket 38 paired OOF comparison",
        "",
        f"- Cohort: `{comparison['prediction_count']}` matched reference studies",
        f"- Fold manifest checksum: `{comparison['fold_manifest_checksum']}`",
        f"- Baseline: {baseline_name} (pooled macro AUC `{base['pooled_macro_auc']}`)",
        f"- Candidate: {candidate_name} (pooled macro AUC `{candidate['pooled_macro_auc']}`)",
        f"- Candidate minus baseline: `{delta['pooled_macro_auc']}`",
        "- Paired 95% study-bootstrap interval: "
        f"`{interval['lower']}` to `{interval['upper']}` "
        f"({interval['defined_resamples']} defined, {interval['undefined_resamples']} undefined "
        f"of {interval['resamples']})",
        "",
        "## Fold results",
        "",
        "| Fold | Baseline macro AUC | Candidate macro AUC | Delta | Held-out studies |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in cast(Sequence[Mapping[str, object]], delta["folds"]):
        rows.append(
            f"| {fold['fold']} | {fold['baseline_macro_auc']} | {fold['candidate_macro_auc']} "
            f"| {fold['delta_macro_auc']} | {fold['prediction_count']} |"
        )
    rows.extend(
        [
            "",
            "## Per-target support and effects",
            "",
            "| Target | Positive | Negative | Baseline AUC | Candidate AUC | Delta | Paired 95% interval |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for target in cast(Sequence[Mapping[str, object]], comparison["per_target"]):
        target_interval = cast(Mapping[str, object], target["paired_study_bootstrap"])
        rows.append(
            f"| {target['target']} | {target['positive_count']} | {target['negative_count']} "
            f"| {target['baseline_auc']} | {target['candidate_auc']} | {target['delta_auc']} "
            f"| {target_interval['lower']} to {target_interval['upper']} "
            f"({target_interval['undefined_resamples']} undefined) |"
        )
    rows.extend(
        [
            "",
            "Undefined one-class AUCs and bootstrap replicates remain undefined. This is repeated "
            "development cross-validation for model selection, not an untouched test or a clinical "
            "generalization estimate.",
            "",
        ]
    )
    return "\n".join(rows)


def analyze_ticket38(
    *,
    plan_path: str | Path,
    fold_manifest_path: str | Path,
    integration_manifest_path: str | Path,
    source_path: str | Path,
    dinov2_runs_root: str | Path,
    resnet_runs_root: str | Path,
    series_cache_root: str | Path,
    feature_cache_root: str | Path,
    output_root: str | Path,
) -> Mapping[str, object]:
    """Validate complete OOF evidence, run frozen-feature ablation and write reports."""

    fold_manifest = load_fold_manifest(fold_manifest_path)
    if fold_manifest.superseded or len(fold_manifest.gold_memberships) != 58:
        raise Ticket38AnalysisError("Ticket 38 requires the corrected 58-study reference cohort")
    dinov2_paths = _run_paths(Path(dinov2_runs_root))
    resnet_paths = _verify_ticket32_baseline(Path(plan_path), Path(resnet_runs_root))
    try:
        dinov2 = aggregate_oof_run_records(
            dinov2_paths,
            fold_manifest,
            expected_gold_count=58,
            bootstrap_resamples=BOOTSTRAP_RESAMPLES,
            bootstrap_seed=BOOTSTRAP_SEED,
            permutation_resamples=PERMUTATION_RESAMPLES,
            permutation_seed=BOOTSTRAP_SEED,
        )
        resnet = aggregate_oof_run_records(
            resnet_paths,
            fold_manifest,
            expected_gold_count=58,
            bootstrap_resamples=BOOTSTRAP_RESAMPLES,
            bootstrap_seed=BOOTSTRAP_SEED,
            permutation_resamples=PERMUTATION_RESAMPLES,
            permutation_seed=BOOTSTRAP_SEED,
        )
        dinov2_records = tuple(load_oof_run_record(path) for path in dinov2_paths)
        resnet_records = tuple(load_oof_run_record(path) for path in resnet_paths)
    except OOFArtifactError as exc:
        raise Ticket38AnalysisError("Ticket 38 or Ticket 32 OOF artifacts failed validation") from exc
    _require_dinov2_records(dinov2_records)
    if dinov2.fold_manifest_checksum != resnet.fold_manifest_checksum:
        raise Ticket38AnalysisError("DINOv2 and ResNet OOF folds do not match")
    dinov2_predictions = _merge_predictions(dinov2_records)
    resnet_predictions = _merge_predictions(resnet_records)
    comparison = compare_image_model_predictions(
        resnet_predictions,
        dinov2_predictions,
        fold_manifest,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )

    cache_identity = _cache_identity(dinov2_records[0])
    if any(_cache_identity(record) != cache_identity for record in dinov2_records[1:]):
        raise Ticket38AnalysisError("DINOv2 folds used different frozen-feature cache identities")
    integration = load_integration_manifest(integration_manifest_path)
    gold_uids = tuple(
        sorted(item.study_instance_uid for item in fold_manifest.gold_memberships)
    )
    if any(
        record.integration_manifest_checksum != integration.manifest_checksum
        or record.source_fingerprint != integration.archive_fingerprint
        or record.fold_manifest_checksum != fold_manifest.manifest_checksum
        for record in dinov2_records
    ):
        raise Ticket38AnalysisError("DINOv2 run lineage differs from the admitted manifests")
    studies = {
        loaded.study.study_instance_uid: loaded.study
        for loaded in iter_real_studies(
            integration,
            Path(source_path),
            require_complete_series=True,
            study_uids=gold_uids,
        )
    }
    if set(studies) != set(gold_uids):
        raise Ticket38AnalysisError("DINOv2 offline replay did not load exactly the 58 gold studies")
    series_cache = SeriesCache(series_cache_root, read_only=True)
    training_config = image_model_training_config(
        TICKET38_DINOV2_VARIANT,
        preprocessing_config=integration.preprocess_config,
        packing_config=integration.packing_config,
    )
    feature_cache = FrozenFeatureCache(
        feature_cache_root,
        cache_identity,
        feature_width=DINOV2_FEATURE_DIM,
        read_only=True,
    )
    full_ablation_predictions: dict[str, tuple[float, ...]] = {}
    zero_ablation_predictions: dict[str, tuple[float, ...]] = {}
    checkpoint_contracts: list[dict[str, object]] = []
    for record in dinov2_records:
        checkpoint_path = record.path.parent / "checkpoints" / "final.pt"
        checkpoint_payload, model_config, preprocess_config, packing_config = _checkpoint_contract(
            checkpoint_path
        )
        if model_config != training_config.model_config():
            raise Ticket38AnalysisError("candidate checkpoint contract changed the DINOv2 model identity")
        if preprocess_config != training_config.preprocess_config() or packing_config != training_config.packing_config():
            raise Ticket38AnalysisError("candidate checkpoint contract changed DINOv2 preprocessing or packing")
        candidate_model = cast(Dinov2StudyModel, build_study_model(model_config))
        checkpoint = load_training_checkpoint(
            checkpoint_path,
            model=candidate_model,
            expected_config=training_config,
            expected_fold=record.fold,
            expected_identity=_configuration_identity(record),
            expected_checkpoint_kind="final_selected",
        )
        if checkpoint.get("checkpoint_kind") != "final_selected":
            raise Ticket38AnalysisError("DINOv2 checkpoint did not reload as final_selected")
        examples = build_fold_held_out_examples(
            studies,
            fold_manifest,
            record.fold,
            preprocess_config=training_config.preprocess_config(),
            packing_config=training_config.packing_config(),
            series_cache=series_cache,
        )
        try:
            examples = _attach_cached_frozen_features(
                examples,
                model=candidate_model,
                cache=feature_cache,
                device=torch.device("cpu"),
            )
        except (FeatureCacheError, TrainingDataError) as exc:
            raise Ticket38AnalysisError("held-out DINOv2 feature cache failed its read-only replay") from exc
        if any(example.features is None for example in examples):
            raise Ticket38AnalysisError("held-out DINOv2 cache replay omitted features")
        full = predict_training_examples(candidate_model, examples, device="cpu")
        zeroed = tuple(
            replace(example, features=np.zeros_like(cast(np.ndarray, example.features)))
            for example in examples
        )
        ablated = predict_training_examples(candidate_model, zeroed, device="cpu")
        for uid, scores in full.items():
            if uid in full_ablation_predictions or uid in zero_ablation_predictions:
                raise Ticket38AnalysisError("DINOv2 held-out replay duplicated a study")
            full_ablation_predictions[uid] = tuple(float(value) for value in scores)
            zero_ablation_predictions[uid] = tuple(float(value) for value in ablated[uid])
        checkpoint_contracts.append(
            {
                "fold": record.fold,
                "run_id": record.run_id,
                "checkpoint_kind": checkpoint_payload.get("checkpoint_kind"),
                "model_id": model_config.model_id,
                "source_revision": model_config.encoder_source_revision,
                "candidate_checkpoint_contract": "accepted",
                "checkpoint_reload": "accepted",
                "read_only_cache_hits": feature_cache.hits,
                "read_only_cache_misses": feature_cache.misses,
            }
        )
    if set(full_ablation_predictions) != set(dinov2_predictions):
        raise Ticket38AnalysisError("offline checkpoint replay does not cover the complete OOF cohort")
    if any(
        not np.allclose(full_ablation_predictions[uid], dinov2_predictions[uid], atol=1e-6, rtol=0.0)
        for uid in dinov2_predictions
    ):
        raise Ticket38AnalysisError("offline DINOv2 checkpoint replay differs from saved OOF predictions")
    ablation_comparison = compare_image_model_predictions(
        zero_ablation_predictions,
        full_ablation_predictions,
        fold_manifest,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    comparison_delta = cast(Mapping[str, object], comparison["delta"])
    ablation_delta = cast(Mapping[str, object], ablation_comparison["delta"])

    output = Path(output_root)
    public_dir = output / "public"
    private_dir = output / "private"
    public_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    private_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_oof_reports(
        dinov2,
        public_path=public_dir / "dinov2-oof-evidence.json",
        comparison_path=public_dir / "dinov2-oof-comparison.md",
        private_path=private_dir / "dinov2-oof-evidence.json",
        source_run_paths=dinov2_paths,
    )
    write_oof_reports(
        resnet,
        public_path=public_dir / "resnet-oof-evidence.json",
        comparison_path=public_dir / "resnet-oof-comparison.md",
        private_path=private_dir / "resnet-oof-evidence.json",
        source_run_paths=resnet_paths,
    )
    atomic_json_write(private_dir / "resnet-vs-dinov2-comparison.json", comparison)
    atomic_json_write(private_dir / "dinov2-zero-feature-ablation.json", ablation_comparison)
    atomic_json_write(
        private_dir / "candidate-checkpoint-lifecycle.json",
        {
            "schema_version": "rsna-ticket38-candidate-checkpoint-lifecycle-v1",
            "folds": checkpoint_contracts,
            "offline_oof_replay_prediction_count": len(full_ablation_predictions),
            "saved_prediction_replay_tolerance": 1e-6,
            "read_only_feature_cache": True,
            "candidate_bundle_freeze": False,
            "candidate_promotion": False,
            "reason": "Ticket 38 authorization excludes candidate promotion and final-candidate freeze",
        },
    )
    atomic_json_write(
        public_dir / "resnet-vs-dinov2-comparison.json",
        comparison,
    )
    comparison_markdown = _markdown_comparison(
        comparison,
        candidate_name="Frozen DINOv2 ViT-S/14",
        baseline_name="Ticket 32 frozen ResNet-18",
    )
    (public_dir / "resnet-vs-dinov2-comparison.md").write_text(
        comparison_markdown, encoding="utf-8"
    )
    atomic_json_write(
        public_dir / "dinov2-zero-feature-ablation.json",
        ablation_comparison,
    )
    ablation_markdown = _markdown_comparison(
        ablation_comparison,
        candidate_name="DINOv2 class-token features",
        baseline_name="Zeroed DINOv2 features with unchanged head",
    )
    (public_dir / "dinov2-zero-feature-ablation.md").write_text(
        ablation_markdown, encoding="utf-8"
    )
    return {
        "prediction_count": dinov2.prediction_count,
        "dinov2_pooled_macro_auc": dinov2.pooled_macro_auc,
        "resnet_pooled_macro_auc": resnet.pooled_macro_auc,
        "candidate_minus_resnet_macro_auc": comparison_delta["pooled_macro_auc"],
        "zero_feature_ablation_delta": ablation_delta["pooled_macro_auc"],
        "checkpoint_lifecycle_folds": len(checkpoint_contracts),
        "read_only_feature_cache_hits": feature_cache.hits,
        "read_only_feature_cache_misses": feature_cache.misses,
        "public_outputs": [
            "public/dinov2-oof-evidence.json",
            "public/dinov2-oof-comparison.md",
            "public/resnet-vs-dinov2-comparison.json",
            "public/resnet-vs-dinov2-comparison.md",
            "public/dinov2-zero-feature-ablation.json",
            "public/dinov2-zero-feature-ablation.md",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--integration-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dinov2-runs-root", type=Path, required=True)
    parser.add_argument("--resnet-runs-root", type=Path, required=True)
    parser.add_argument("--series-cache-root", type=Path, required=True)
    parser.add_argument("--feature-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = analyze_ticket38(
            plan_path=args.plan,
            fold_manifest_path=args.fold_manifest,
            integration_manifest_path=args.integration_manifest,
            source_path=args.source,
            dinov2_runs_root=args.dinov2_runs_root,
            resnet_runs_root=args.resnet_runs_root,
            series_cache_root=args.series_cache_root,
            feature_cache_root=args.feature_cache_root,
            output_root=args.output_root,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Ticket 38 analysis failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Ticket38AnalysisError", "analyze_ticket38", "main"]
