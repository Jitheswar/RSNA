"""Read-only aggregation of atomic fold training records into OOF evidence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from .archive import atomic_json_write
from .evaluation import OOFEvaluation, evaluate_oof_predictions
from .folds import FOLD_IDS, FoldManifest, validate_fold_manifest_against_source
from .inventory import SourceManifest


class OOFArtifactError(ValueError):
    """Raised when fold run records cannot form one complete OOF repeat."""


@dataclass(frozen=True)
class OOFRunRecord:
    """The immutable, identifier-bearing portion of one fold's run record."""

    path: Path
    fold: int
    run_id: str
    config_fingerprint: str
    predictions: Mapping[str, tuple[float, ...]]
    held_out_study_uids: tuple[str, ...] = ()
    training_study_uids: tuple[str, ...] = ()
    fold_manifest_checksum: str | None = None
    source_version: str | None = None
    source_fingerprint: str | None = None
    integration_manifest_checksum: str | None = None


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OOFArtifactError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _nonempty(value: object, field: str) -> str:
    if value is None:
        raise OOFArtifactError(f"{field} must be non-empty")
    text = str(value).strip()
    if not text:
        raise OOFArtifactError(f"{field} must be non-empty")
    return text


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise OOFArtifactError(f"{field} must be an integer")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise OOFArtifactError(f"{field} must be an integer") from exc
    if parsed < 0:
        raise OOFArtifactError(f"{field} must be non-negative")
    return parsed


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, field)


def _uid_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise OOFArtifactError(f"{field} must be an array")
    result = tuple(_nonempty(item, field) for item in value)
    if len(set(result)) != len(result):
        raise OOFArtifactError(f"{field} must not contain duplicates")
    return result


def _read_object(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OOFArtifactError(f"could not read OOF run record {path}") from exc
    return _mapping(payload, "OOF run record")


def _load_artifact_contract(
    run_path: Path,
    payload: Mapping[str, object],
    *,
    fold: int,
    run_id: str,
    config_fingerprint: str,
    prediction_count: int,
) -> Mapping[str, object]:
    configuration_path = run_path.parent / "configuration.json"
    if not configuration_path.is_file():
        raise OOFArtifactError(f"completed OOF run is missing configuration.json: {run_path.parent}")
    configuration = _read_object(configuration_path)
    if configuration.get("run_id") != run_id or configuration.get("fold") != fold:
        raise OOFArtifactError(f"configuration identity disagrees with run record: {run_path}")
    if configuration.get("scientific_configuration_fingerprint") != config_fingerprint:
        raise OOFArtifactError(f"configuration fingerprint disagrees with run record: {run_path}")
    run_config = _mapping(payload.get("config"), "run.config")
    configuration_config = _mapping(configuration["config"], "configuration.config")
    if configuration_config != run_config:
        raise OOFArtifactError(f"full configuration disagrees with run record: {run_path}")
    for field in (
        "config",
        "environment",
        "resources",
        "inference_configuration",
        "supervision",
        "preflight",
    ):
        if not isinstance(configuration.get(field), Mapping):
            raise OOFArtifactError(f"configuration is missing object field {field}: {configuration_path}")
    environment = _mapping(configuration["environment"], "configuration.environment")
    if "code_revision" not in environment:
        raise OOFArtifactError(f"configuration.environment.code_revision is missing: {configuration_path}")
    if environment.get("code_revision") is not None:
        _nonempty(environment.get("code_revision"), "configuration.environment.code_revision")
    for field in ("environment_fingerprint", "python_executable", "python_version"):
        _nonempty(environment.get(field), f"configuration.environment.{field}")
    supervision = _mapping(configuration["supervision"], "configuration.supervision")
    if supervision.get("kind") != "reference":
        raise OOFArtifactError(f"OOF configuration is not reference-only: {configuration_path}")
    if supervision.get("held_out_inputs_excluded") is not True:
        raise OOFArtifactError(f"OOF configuration does not prove held-out exclusion: {configuration_path}")
    if supervision.get("reports_used") is not False:
        raise OOFArtifactError(f"OOF configuration does not prove report exclusion: {configuration_path}")
    preflight = _mapping(configuration["preflight"], "configuration.preflight")
    if preflight.get("passed") is not True:
        raise OOFArtifactError(f"OOF run does not contain a passed CPU preflight: {run_path}")
    identity = _mapping(configuration.get("identity"), "configuration.identity")
    for field in (
        "source_version",
        "source_fingerprint",
        "cache_fingerprint",
        "fold_manifest_checksum",
        "integration_manifest_checksum",
        "data_fingerprint",
        "label_source",
        "model_source",
        "training_schema_version",
    ):
        _nonempty(identity.get(field), f"configuration.identity.{field}")

    checkpoint = _mapping(payload.get("checkpoint"), "checkpoint")
    if checkpoint.get("final_present") is not True or checkpoint.get("latest_present") is not True:
        raise OOFArtifactError(f"completed OOF run is missing final/latest checkpoint evidence: {run_path}")
    final_sha256 = _nonempty(checkpoint.get("final_sha256"), "checkpoint.final_sha256")
    final_path = run_path.parent / "checkpoints" / "final.pt"
    latest_path = run_path.parent / "checkpoints" / "latest.pt"
    if not final_path.is_file() or not latest_path.is_file():
        raise OOFArtifactError(f"completed OOF run is missing checkpoint files: {run_path.parent}")
    try:
        actual_sha256 = hashlib.sha256(final_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise OOFArtifactError(f"could not read final checkpoint: {final_path}") from exc
    if actual_sha256 != final_sha256:
        raise OOFArtifactError(f"final checkpoint checksum disagrees with run record: {run_path}")

    metrics_path = run_path.parent / "metrics.json"
    if not metrics_path.is_file():
        raise OOFArtifactError(f"completed OOF run is missing metrics.json: {run_path.parent}")
    metrics = _read_object(metrics_path)
    if metrics.get("fold") != fold or metrics.get("prediction_count") != prediction_count:
        raise OOFArtifactError(f"metrics evidence disagrees with run record: {metrics_path}")
    return identity


def load_oof_run_record(path: str | Path) -> OOFRunRecord:
    """Load one completed fold record without changing it or its directory."""

    candidate = Path(path)
    run_path = candidate / "run.json" if candidate.is_dir() else candidate
    payload = _read_object(run_path)
    if payload.get("status") != "completed_fixed_budget_training":
        raise OOFArtifactError(f"OOF run is not completed fixed-budget training: {run_path}")
    if payload.get("reference_supervision_only") is not True:
        raise OOFArtifactError(f"OOF run is not reference-only: {run_path}")
    if payload.get("held_out_inputs_excluded_from_fitting") is not True:
        raise OOFArtifactError(f"OOF run does not prove held-out exclusion: {run_path}")
    if "held_out_study_uids" not in payload or "training_study_uids" not in payload:
        raise OOFArtifactError(f"OOF run does not record both fold partitions: {run_path}")
    fold = _required_int(payload.get("fold"), "fold")
    if fold not in FOLD_IDS:
        raise OOFArtifactError(f"OOF run fold must be one of {FOLD_IDS}")
    predictions_payload = _mapping(payload.get("predictions"), "predictions")
    predictions: dict[str, tuple[float, ...]] = {}
    for uid, values in predictions_payload.items():
        if uid in predictions:
            raise OOFArtifactError(f"OOF run contains duplicate prediction {uid}")
        if not isinstance(values, (list, tuple)):
            raise OOFArtifactError(f"prediction for {uid} must be an array")
        try:
            predictions[_nonempty(uid, "prediction study identifier")] = tuple(
                float(item) for item in values
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise OOFArtifactError(f"prediction for {uid} is not numeric") from exc
    run_id = _nonempty(payload.get("run_id"), "run_id")
    config_fingerprint = _nonempty(payload.get("config_fingerprint"), "config_fingerprint")
    identity = _load_artifact_contract(
        run_path,
        payload,
        fold=fold,
        run_id=run_id,
        config_fingerprint=config_fingerprint,
        prediction_count=len(predictions),
    )
    return OOFRunRecord(
        path=run_path.resolve(),
        fold=fold,
        run_id=run_id,
        config_fingerprint=config_fingerprint,
        predictions=predictions,
        held_out_study_uids=_uid_tuple(payload["held_out_study_uids"], "held_out_study_uids"),
        training_study_uids=_uid_tuple(payload["training_study_uids"], "training_study_uids"),
        fold_manifest_checksum=_optional_text(
            identity.get("fold_manifest_checksum", payload.get("fold_manifest_checksum")),
            "fold_manifest_checksum",
        ),
        source_version=_optional_text(
            identity.get("source_version", payload.get("source_version")),
            "source_version",
        ),
        source_fingerprint=_optional_text(
            identity.get("source_fingerprint", payload.get("source_fingerprint")),
            "source_fingerprint",
        ),
        integration_manifest_checksum=_optional_text(
            identity.get(
                "integration_manifest_checksum",
                payload.get("integration_manifest_checksum"),
            ),
            "integration_manifest_checksum",
        ),
    )


def _validate_run_against_manifest(record: OOFRunRecord, manifest: FoldManifest) -> None:
    expected_held_out = tuple(
        sorted(
            item.study_instance_uid
            for item in manifest.gold_memberships
            if item.fold == record.fold
        )
    )
    expected_training = tuple(
        sorted(
            item.study_instance_uid
            for item in manifest.gold_memberships
            if item.fold != record.fold
        )
    )
    if tuple(sorted(record.held_out_study_uids)) != expected_held_out:
        raise OOFArtifactError(f"fold {record.fold} held-out membership disagrees with the fold manifest")
    if tuple(sorted(record.training_study_uids)) != expected_training:
        raise OOFArtifactError(f"fold {record.fold} training membership disagrees with the fold manifest")
    if tuple(sorted(record.predictions)) != expected_held_out:
        raise OOFArtifactError(f"fold {record.fold} predictions disagree with held-out membership")
    if record.fold_manifest_checksum is not None and (
        record.fold_manifest_checksum != manifest.manifest_checksum
    ):
        raise OOFArtifactError(f"fold {record.fold} uses a different fold manifest")
    if record.source_version is not None and record.source_version != manifest.source_version:
        raise OOFArtifactError(f"fold {record.fold} uses a different source version")
    if (
        record.source_fingerprint is not None
        and manifest.source_fingerprint is not None
        and record.source_fingerprint != manifest.source_fingerprint
    ):
        raise OOFArtifactError(f"fold {record.fold} uses a different source fingerprint")


def _validate_common_run_identity(records: Sequence[OOFRunRecord]) -> None:
    for field, label in (
        ("fold_manifest_checksum", "fold manifest"),
        ("source_version", "source version"),
        ("source_fingerprint", "source fingerprint"),
        ("integration_manifest_checksum", "integration manifest"),
    ):
        values = tuple(getattr(record, field) for record in records)
        present = tuple(value for value in values if value is not None)
        if present and (len(present) != len(values) or len(set(present)) != 1):
            raise OOFArtifactError(f"OOF {label} identity mismatch across folds")


def aggregate_oof_run_records(
    run_paths: Sequence[str | Path],
    fold_manifest: FoldManifest,
    *,
    source_manifest: SourceManifest | None = None,
    expected_gold_count: int | None = None,
    bootstrap_resamples: int | None = None,
    bootstrap_seed: int = 20260911,
    bootstrap_unit: str = "auto",
    bootstrap_confidence_level: float = 0.95,
) -> OOFEvaluation:
    """Recompute an OOF comparison view from one atomic record per fold."""

    if expected_gold_count is not None and expected_gold_count <= 0:
        raise OOFArtifactError("expected_gold_count must be positive")
    expected_gold = len(fold_manifest.gold_memberships)
    if source_manifest is not None:
        try:
            validate_fold_manifest_against_source(fold_manifest, source_manifest)
        except ValueError as exc:
            raise OOFArtifactError("fold manifest does not reconcile with the source labels") from exc
    if expected_gold_count is not None and expected_gold != expected_gold_count:
        raise OOFArtifactError(
            f"fold manifest has {expected_gold} gold studies; expected {expected_gold_count}"
        )
    if not run_paths:
        raise OOFArtifactError("at least one OOF run record is required")
    records = tuple(load_oof_run_record(path) for path in run_paths)
    if len({record.path for record in records}) != len(records):
        raise OOFArtifactError("OOF aggregation received the same run record more than once")
    by_fold: dict[int, OOFRunRecord] = {}
    for record in records:
        if record.fold in by_fold:
            raise OOFArtifactError(f"duplicate OOF run for fold {record.fold}")
        by_fold[record.fold] = record
        _validate_run_against_manifest(record, fold_manifest)
    if set(by_fold) != set(FOLD_IDS):
        missing = sorted(set(FOLD_IDS) - set(by_fold))
        extra = sorted(set(by_fold) - set(FOLD_IDS))
        raise OOFArtifactError(f"OOF fold coverage mismatch: missing={missing}, extra={extra}")
    ordered = tuple(by_fold[fold] for fold in FOLD_IDS)
    if len({record.run_id for record in ordered}) != len(ordered):
        raise OOFArtifactError("OOF run identities must be unique across folds")
    if len({record.config_fingerprint for record in ordered}) != 1:
        raise OOFArtifactError("OOF configuration fingerprint mismatch across folds")
    _validate_common_run_identity(ordered)
    predictions: dict[str, tuple[float, ...]] = {}
    for record in ordered:
        for uid, values in record.predictions.items():
            if uid in predictions:
                raise OOFArtifactError(f"study has more than one OOF prediction: {uid}")
            predictions[uid] = values
    result = evaluate_oof_predictions(
        predictions,
        fold_manifest,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_unit=bootstrap_unit,
        bootstrap_confidence_level=bootstrap_confidence_level,
    )
    return replace(result, run_ids=tuple(record.run_id for record in ordered))


def _atomic_text_write(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-",
        dir=destination.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def _run_directory(path: str | Path) -> Path:
    candidate = Path(path)
    return (candidate if candidate.is_dir() else candidate.parent).resolve()


def _reject_source_destinations(
    destinations: Sequence[Path], source_run_paths: Sequence[str | Path]
) -> None:
    if not source_run_paths:
        raise OOFArtifactError("source run paths are required to protect aggregation inputs")
    source_directories = tuple(_run_directory(path) for path in source_run_paths)
    for destination in destinations:
        resolved = destination.resolve()
        for source_directory in source_directories:
            try:
                resolved.relative_to(source_directory)
            except ValueError:
                continue
            raise OOFArtifactError(
                f"OOF report destination is inside a source run directory: {destination}"
            )


def _format_metric(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.6f}"


def render_oof_comparison(evaluation: OOFEvaluation) -> str:
    """Render an identifier-free Markdown comparison view."""

    lines = [
        "# Gold-only OOF comparison",
        "",
        f"- Repeat: `{evaluation.repeat_name}`",
        f"- Held-out predictions: `{evaluation.prediction_count}`",
        f"- Pooled macro ROC-AUC: `{_format_metric(evaluation.pooled_macro_auc)}`",
        f"- Mean fold macro ROC-AUC: `{_format_metric(evaluation.mean_fold_macro_auc)}`",
        f"- Fold macro standard deviation: `{_format_metric(evaluation.fold_macro_auc_std)}`",
        "",
        "| Target | Pooled AUC | Positives | Negatives | Fold 0 | Fold 1 | Fold 2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    fold_targets = {
        fold.fold: {item.target: item for item in fold.targets}
        for fold in evaluation.folds
    }
    for target in evaluation.pooled_targets:
        lines.append(
            "| "
            + " | ".join(
                (
                    target.target,
                    _format_metric(target.auc),
                    str(target.positive_count),
                    str(target.negative_count),
                    *(
                        _format_metric(fold_targets[fold][target.target].auc)
                        for fold in FOLD_IDS
                    ),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Undefined one-class AUCs remain undefined; no reduced-target macro is "
            "used as the primary metric.",
            "",
            "This is repeated development cross-validation for model selection. It "
            "is not an untouched internal test and does not establish clinical "
            "generalization.",
            "",
        ]
    )
    return "\n".join(lines)


def write_oof_reports(
    evaluation: OOFEvaluation,
    *,
    public_path: str | Path,
    source_run_paths: Sequence[str | Path],
    comparison_path: str | Path | None = None,
    private_path: str | Path | None = None,
) -> tuple[Path, ...]:
    """Write only regenerated aggregate views; input run records stay untouched."""

    destinations = [Path(public_path)]
    if comparison_path is not None:
        destinations.append(Path(comparison_path))
    if private_path is not None:
        destinations.append(Path(private_path))
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise OOFArtifactError("OOF report destinations must be distinct")
    _reject_source_destinations(destinations, source_run_paths)
    written: list[Path] = [
        atomic_json_write(public_path, evaluation.to_public_dict(), overwrite=True)
    ]
    if comparison_path is not None:
        written.append(_atomic_text_write(comparison_path, render_oof_comparison(evaluation)))
    if private_path is not None:
        written.append(atomic_json_write(private_path, evaluation.to_private_dict(), overwrite=True))
    return tuple(written)


def _parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=None)
    parser.add_argument("--run", dest="run_paths", action="append", required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, default=None)
    parser.add_argument("--comparison-output", type=Path, default=None)
    parser.add_argument("--expected-gold-count", type=int, default=58)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260911)
    parser.add_argument("--bootstrap-unit", choices=("auto", "study", "group"), default="auto")
    parser.add_argument("--bootstrap-confidence-level", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Aggregate complete fold records without reading or mutating their inputs."""

    from .folds import load_fold_manifest, load_inventory_source

    args = _parser().parse_args(argv)
    try:
        parsed_resamples = cast(int, args.bootstrap_resamples)
        if parsed_resamples < 0:
            raise OOFArtifactError("bootstrap resamples cannot be negative")
        inventory = (
            load_inventory_source(cast(Path, args.inventory))
            if args.inventory is not None
            else None
        )
        if inventory is not None and not inventory.full_corpus_reconciled:
            raise OOFArtifactError("source inventory is not a complete corpus reconciliation")
        evaluation = aggregate_oof_run_records(
            cast(Sequence[str | Path], args.run_paths),
            load_fold_manifest(cast(Path, args.fold_manifest)),
            source_manifest=inventory.source_manifest if inventory is not None else None,
            expected_gold_count=cast(int, args.expected_gold_count),
            bootstrap_resamples=parsed_resamples or None,
            bootstrap_seed=cast(int, args.bootstrap_seed),
            bootstrap_unit=cast(str, args.bootstrap_unit),
            bootstrap_confidence_level=cast(float, args.bootstrap_confidence_level),
        )
        comparison_path = cast(Path | None, args.comparison_output)
        if comparison_path is None:
            public_path = cast(Path, args.public_output)
            comparison_path = public_path.with_suffix(".md")
        write_oof_reports(
            evaluation,
            public_path=cast(Path, args.public_output),
            source_run_paths=cast(Sequence[str | Path], args.run_paths),
            private_path=cast(Path | None, args.private_output),
            comparison_path=comparison_path,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        import sys

        print(f"ticket-10 OOF aggregation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(evaluation.to_public_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OOFArtifactError",
    "OOFRunRecord",
    "aggregate_oof_run_records",
    "load_oof_run_record",
    "main",
    "render_oof_comparison",
    "write_oof_reports",
]
