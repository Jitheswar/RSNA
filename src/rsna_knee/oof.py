"""Read-only aggregation of atomic fold training records into OOF evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from .archive import atomic_json_write
from .evaluation import OOFEvaluation, evaluate_oof_predictions
from .folds import FOLD_IDS, FoldManifest, validate_fold_manifest_against_source
from .inventory import SourceManifest
from .report_extraction import FrozenDeterministicExtractor, REPORT_EXTRACTION_SCHEMA_VERSION
from .supervision import ADMISSION_SCHEMA_VERSION, MIXTURE_POLICY_VERSION, WEAK_LABEL_VERSION
from .ticket34 import TICKET30_SOURCE_ID
from .targets import TARGET_COLUMNS


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
    expected_supervision: str = "reference"
    config: Mapping[str, object] = field(default_factory=dict)
    counts: Mapping[str, object] = field(default_factory=dict)
    supervision: Mapping[str, object] = field(default_factory=dict, repr=False)
    identity: Mapping[str, object] = field(default_factory=dict)
    resources: Mapping[str, object] = field(default_factory=dict)
    environment: Mapping[str, object] = field(default_factory=dict)


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
    expected_supervision: str,
    expected_report_ticket: int,
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
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
    if expected_supervision == "reference":
        if supervision.get("kind") != "reference":
            raise OOFArtifactError(f"OOF configuration is not reference-only: {configuration_path}")
        if supervision.get("held_out_inputs_excluded") is not True:
            raise OOFArtifactError(
                f"OOF configuration does not prove held-out exclusion: {configuration_path}"
            )
        if supervision.get("reports_used") is not False:
            raise OOFArtifactError(
                f"OOF configuration does not prove report exclusion: {configuration_path}"
            )
    elif expected_supervision == "report":
        _validate_report_supervision_contract(
            supervision,
            fold=fold,
            configuration_path=configuration_path,
            expected_report_ticket=expected_report_ticket,
        )
    else:
        raise OOFArtifactError(f"unsupported OOF supervision arm {expected_supervision!r}")
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
    return identity, configuration, supervision


def _validate_report_supervision_contract(
    supervision: Mapping[str, object],
    *,
    fold: int,
    configuration_path: Path,
    expected_report_ticket: int = 18,
) -> None:
    if supervision.get("schema_version") != ADMISSION_SCHEMA_VERSION:
        raise OOFArtifactError("report-supervised run has an unsupported admission schema")
    if expected_report_ticket == 36:
        expected_source_schema = "rsna-knee-report-label-audit-v1"
        expected_weak_policy = "ticket30-fold-specific-wilson-admission-v4"
        expected_mixture_policy = "reference-precedence-plus-ticket30-report-only-v1"
    else:
        expected_source_schema = REPORT_EXTRACTION_SCHEMA_VERSION
        expected_weak_policy = WEAK_LABEL_VERSION
        expected_mixture_policy = MIXTURE_POLICY_VERSION
    if supervision.get("source_schema_version") != expected_source_schema:
        raise OOFArtifactError("report-supervised run has an unsupported extraction schema")
    if supervision.get("fold") != fold:
        raise OOFArtifactError("report supervision fold disagrees with the training run")
    if supervision.get("weak_label_version") != expected_weak_policy:
        raise OOFArtifactError("report-supervised run uses an unapproved weak-label policy")
    if supervision.get("mixture_policy") != expected_mixture_policy:
        raise OOFArtifactError("report-supervised run uses an unapproved mixture policy")
    if not _DIGEST_RE.fullmatch(str(supervision.get("source_artifact_digest", ""))):
        raise OOFArtifactError("report-supervised run has no valid source artifact digest")
    boundaries = _mapping(supervision.get("boundaries"), "report supervision boundaries")
    for field_name in (
        "held_out_inputs_excluded",
        "held_out_report_labels_evaluation_only",
        "reference_labels_take_precedence",
        "unknown_uncertain_failed_masked",
    ):
        if boundaries.get(field_name) is not True:
            raise OOFArtifactError(
                f"report supervision does not prove {field_name.replace('_', '-')}"
            )
    if boundaries.get("fallback_source") != "trusted_reference_labels_only":
        raise OOFArtifactError("report supervision does not prove reference-only fallback provenance")

    extractor = _mapping(supervision.get("extractor"), "report supervision extractor")
    if expected_report_ticket == 36:
        if supervision.get("source_id") != TICKET30_SOURCE_ID:
            raise OOFArtifactError("Ticket 36 report run does not use the frozen Ticket 16 source")
        for field_name in ("version", "configuration_fingerprint", "ruleset_fingerprint"):
            _nonempty(extractor.get(field_name), f"report supervision extractor.{field_name}")
    else:
        expected_extractor = FrozenDeterministicExtractor().to_public_dict()
        for field_name, expected in (
            ("version", expected_extractor["extractor_version"]),
            ("configuration_fingerprint", expected_extractor["configuration_fingerprint"]),
            ("ruleset_fingerprint", expected_extractor["ruleset_fingerprint"]),
        ):
            if extractor.get(field_name) != expected:
                raise OOFArtifactError("report-supervised run does not use the selected deterministic extractor")

    for field_name in (
        "training_study_uids",
        "held_out_study_uids",
        "eligible_report_only_study_uids",
        "admitted_report_only_study_uids",
        "missing_report_only_study_uids",
        "excluded_image_report_only_study_uids",
    ):
        _uid_tuple(supervision.get(field_name), f"report supervision {field_name}")
    resolved = supervision.get("resolved_studies")
    if not isinstance(resolved, list):
        raise OOFArtifactError("report supervision is missing resolved training rows")
    resolved_uids = tuple(
        _nonempty(_mapping(item, "resolved report supervision row").get("study_instance_uid"),
                  "resolved supervision study identifier")
        for item in resolved
    )
    if len(set(resolved_uids)) != len(resolved_uids):
        raise OOFArtifactError("report supervision repeats a resolved training study")
    counts = _mapping(supervision.get("counts"), "report supervision counts")
    if counts.get("training_studies") != len(resolved_uids):
        raise OOFArtifactError("report supervision training count disagrees with resolved rows")


def _validate_reference_run_supervision(
    run_supervision: Mapping[str, object],
    configuration_supervision: Mapping[str, object],
    payload: Mapping[str, object],
) -> None:
    """Reconcile the compact run summary with the full configuration contract."""

    for field in ("kind", "fold_manifest_checksum", "held_out_inputs_excluded"):
        if run_supervision.get(field) != configuration_supervision.get(field):
            raise OOFArtifactError("reference supervision evidence disagrees with configuration")
    if run_supervision.get("kind") != "reference":
        raise OOFArtifactError("OOF run is not reference-only")
    if "policy" in run_supervision and run_supervision.get("policy") != "reference-only-v1":
        raise OOFArtifactError("reference supervision policy is unsupported")
    if run_supervision.get("held_out_inputs_excluded") is not True:
        raise OOFArtifactError("reference supervision does not prove held-out exclusion")
    if run_supervision.get("reports_used", False) is not False:
        raise OOFArtifactError("reference supervision unexpectedly used reports")

    run_training = _uid_tuple(payload.get("training_study_uids"), "training_study_uids")
    run_held_out = _uid_tuple(payload.get("held_out_study_uids"), "held_out_study_uids")
    for field, run_partition in (
        ("training_studies", run_training),
        ("held_out_studies", run_held_out),
    ):
        if field in configuration_supervision:
            configuration_partition = _uid_tuple(
                configuration_supervision[field], f"configuration.supervision.{field}"
            )
            if configuration_partition != run_partition:
                raise OOFArtifactError("reference supervision partitions disagree with configuration")

    expected_counts = {
        "training_studies": len(run_training),
        "reference_studies": len(run_training),
        "weak_studies": 0,
        "supervised_entries": len(run_training) * len(TARGET_COLUMNS),
        "source_disagreements": 0,
    }
    for source, evidence in (
        ("run", run_supervision),
        ("configuration", configuration_supervision),
    ):
        if "counts" not in evidence:
            continue
        counts = _mapping(evidence["counts"], f"{source}.supervision.counts")
        for field, expected in expected_counts.items():
            if (
                _required_int(counts.get(field), f"{source}.supervision.counts.{field}")
                != expected
            ):
                raise OOFArtifactError(
                    "reference supervision counts disagree with its study partition"
                )


_DIGEST_RE = re.compile(r"[0-9a-f]{64}")


def load_oof_run_record(
    path: str | Path,
    *,
    expected_supervision: str = "reference",
    expected_report_ticket: int = 18,
) -> OOFRunRecord:
    """Load one completed fold record without changing its directory."""

    if expected_supervision not in {"reference", "report"}:
        raise OOFArtifactError("expected_supervision must be reference or report")

    candidate = Path(path)
    run_path = candidate / "run.json" if candidate.is_dir() else candidate
    payload = _read_object(run_path)
    if payload.get("status") != "completed_fixed_budget_training":
        raise OOFArtifactError(f"OOF run is not completed fixed-budget training: {run_path}")
    reference_only = payload.get("reference_supervision_only") is True
    if expected_supervision == "reference" and not reference_only:
        raise OOFArtifactError(f"OOF run is not reference-only: {run_path}")
    if expected_supervision == "report" and reference_only:
        raise OOFArtifactError(f"OOF run does not use report supervision: {run_path}")
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
    identity, configuration, configuration_supervision = _load_artifact_contract(
        run_path,
        payload,
        fold=fold,
        run_id=run_id,
        config_fingerprint=config_fingerprint,
        prediction_count=len(predictions),
        expected_supervision=expected_supervision,
        expected_report_ticket=expected_report_ticket,
    )
    run_supervision = _mapping(payload.get("supervision"), "run.supervision")
    if expected_supervision == "reference":
        _validate_reference_run_supervision(
            run_supervision,
            configuration_supervision,
            payload,
        )
    elif run_supervision != configuration_supervision:
        raise OOFArtifactError(f"supervision evidence disagrees with configuration: {run_path}")
    counts = _mapping(payload.get("counts"), "run.counts")
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
        expected_supervision=expected_supervision,
        config=_mapping(payload.get("config"), "run.config"),
        counts=counts,
        supervision=run_supervision,
        identity={
            field: identity[field]
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
            )
        },
        resources=_mapping(configuration.get("resources"), "configuration.resources"),
        environment=_mapping(configuration.get("environment"), "configuration.environment"),
    )


def _validate_run_against_manifest(record: OOFRunRecord, manifest: FoldManifest) -> None:
    expected_held_out = tuple(
        sorted(
            item.study_instance_uid
            for item in manifest.gold_memberships
            if item.fold == record.fold
        )
    )
    expected_training_gold = tuple(
        sorted(
            item.study_instance_uid
            for item in manifest.gold_memberships
            if item.fold != record.fold
        )
    )
    if tuple(sorted(record.held_out_study_uids)) != expected_held_out:
        raise OOFArtifactError(f"fold {record.fold} held-out membership disagrees with the fold manifest")
    actual_training = set(record.training_study_uids)
    if record.expected_supervision == "reference":
        if tuple(sorted(actual_training)) != expected_training_gold:
            raise OOFArtifactError(
                f"fold {record.fold} training membership disagrees with the fold manifest"
            )
    else:
        all_training = {
            item.study_instance_uid for item in manifest.memberships if item.fold != record.fold
        }
        expected_gold = set(expected_training_gold)
        if actual_training.intersection(expected_gold) != expected_gold:
            raise OOFArtifactError(
                f"fold {record.fold} report training omits reference training studies"
            )
        if not actual_training.issubset(all_training):
            raise OOFArtifactError(
                f"fold {record.fold} report training includes an unknown or held-out study"
            )
        _validate_report_supervision_against_manifest(record, manifest, expected_training_gold)
    if tuple(sorted(record.predictions)) != expected_held_out:
        raise OOFArtifactError(f"fold {record.fold} predictions disagree with held-out membership")
    if record.fold_manifest_checksum is not None and (
        record.fold_manifest_checksum != manifest.manifest_checksum
    ):
        raise OOFArtifactError(f"fold {record.fold} uses a different fold manifest")
    if record.supervision.get("fold_manifest_checksum") != manifest.manifest_checksum:
        raise OOFArtifactError(
            f"fold {record.fold} supervision uses a different fold manifest"
        )
    if record.source_version is not None and record.source_version != manifest.source_version:
        raise OOFArtifactError(f"fold {record.fold} uses a different source version")
    if (
        record.source_fingerprint is not None
        and manifest.source_fingerprint is not None
        and record.source_fingerprint != manifest.source_fingerprint
    ):
        raise OOFArtifactError(f"fold {record.fold} uses a different source fingerprint")


def _validate_report_supervision_against_manifest(
    record: OOFRunRecord, manifest: FoldManifest, expected_training_gold: Sequence[str]
) -> None:
    supervision = record.supervision
    training_uids = set(record.training_study_uids)
    report_training_uids = training_uids - set(expected_training_gold)
    if set(_uid_tuple(supervision.get("training_study_uids"), "supervision.training_study_uids")) != training_uids:
        raise OOFArtifactError("report supervision training identities disagree with the run")
    expected_held_out = {
        item.study_instance_uid for item in manifest.memberships if item.fold == record.fold
    }
    if set(_uid_tuple(supervision.get("held_out_study_uids"), "supervision.held_out_study_uids")) != expected_held_out:
        raise OOFArtifactError("report supervision held-out identities disagree with the fold manifest")
    expected_eligible = {
        item.study_instance_uid
        for item in manifest.report_only_memberships
        if item.fold != record.fold
    }
    eligible = set(
        _uid_tuple(
            supervision.get("eligible_report_only_study_uids"),
            "supervision.eligible_report_only_study_uids",
        )
    )
    if eligible != expected_eligible:
        raise OOFArtifactError("report supervision eligible cohort disagrees with the fold manifest")
    admitted = set(
        _uid_tuple(
            supervision.get("admitted_report_only_study_uids"),
            "supervision.admitted_report_only_study_uids",
        )
    )
    missing = set(
        _uid_tuple(
            supervision.get("missing_report_only_study_uids"),
            "supervision.missing_report_only_study_uids",
        )
    )
    image_excluded = set(
        _uid_tuple(
            supervision.get("excluded_image_report_only_study_uids"),
            "supervision.excluded_image_report_only_study_uids",
        )
    )
    if admitted != report_training_uids:
        raise OOFArtifactError("admitted report identities disagree with image training membership")
    if not admitted.issubset(eligible) or missing.intersection(admitted) or image_excluded.intersection(admitted):
        raise OOFArtifactError("report supervision cohort partitions overlap or escape eligibility")
    if eligible != admitted | missing | image_excluded:
        raise OOFArtifactError("eligible report cohort does not reconcile with admission exclusions")

    resolved_rows = supervision.get("resolved_studies")
    if not isinstance(resolved_rows, list):
        raise OOFArtifactError("report supervision has no resolved training rows")
    resolved_by_uid: dict[str, Mapping[str, object]] = {}
    for value in resolved_rows:
        row = _mapping(value, "resolved supervision row")
        uid = _nonempty(row.get("study_instance_uid"), "resolved supervision study identifier")
        if uid in resolved_by_uid:
            raise OOFArtifactError("report supervision repeats a resolved training study")
        resolved_by_uid[uid] = row
    if set(resolved_by_uid) != training_uids:
        raise OOFArtifactError("resolved report supervision does not cover training membership")

    gold_labels = {
        item.study_instance_uid: item.reference_labels for item in manifest.gold_memberships
    }
    for uid, row in resolved_by_uid.items():
        source_kind = row.get("source_kind")
        expected_kind = "reference" if uid in gold_labels else "weak"
        if source_kind not in {expected_kind, "mixed" if expected_kind == "reference" else expected_kind}:
            raise OOFArtifactError("report supervision source kind disagrees with cohort membership")
        targets = row.get("targets")
        mask = row.get("supervision_mask")
        if not isinstance(targets, (list, tuple)) or not isinstance(mask, (list, tuple)):
            raise OOFArtifactError("resolved report target vector or mask is missing")
        if len(targets) != len(TARGET_COLUMNS) or len(mask) != len(TARGET_COLUMNS):
            raise OOFArtifactError("resolved report target vector or mask has the wrong width")
        if any(value not in (None, 0, 1) for value in targets) or any(value not in (0, 1) for value in mask):
            raise OOFArtifactError("resolved report target vector or mask contains invalid values")
        if tuple(int(value is not None) for value in targets) != tuple(int(value) for value in mask):
            raise OOFArtifactError("resolved report targets disagree with their supervision mask")
        if uid in gold_labels:
            if tuple(targets) != tuple(gold_labels[uid]) or any(mask_value != 1 for mask_value in mask):
                raise OOFArtifactError("report supervision does not preserve reference-label precedence")


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
    permutation_resamples: int | None = None,
    permutation_seed: int = 20260911,
    expected_supervision: str = "reference",
    expected_report_ticket: int = 18,
    allow_superseded_fold_manifest: bool = False,
) -> OOFEvaluation:
    """Recompute an OOF view from one atomic record per fold and supervision arm."""

    if expected_supervision not in {"reference", "report"}:
        raise OOFArtifactError("expected_supervision must be reference or report")

    if fold_manifest.superseded and not allow_superseded_fold_manifest:
        raise OOFArtifactError(
            "the fold manifest was produced by the superseded version-1 generator; its OOF "
            "results are historical evidence only"
        )
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
    records = tuple(
        load_oof_run_record(
            path,
            expected_supervision=expected_supervision,
            expected_report_ticket=expected_report_ticket,
        )
        for path in run_paths
    )
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
        permutation_resamples=permutation_resamples,
        permutation_seed=permutation_seed,
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


def _table_row(values: Sequence[str]) -> str:
    return "| " + " | ".join(values) + " |"


def render_oof_comparison(evaluation: OOFEvaluation) -> str:
    """Render an identifier-free Markdown comparison view."""

    diagnostics = evaluation.diagnostics
    lines = [
        "# Gold-only OOF comparison",
        "",
        f"- Repeat: `{evaluation.repeat_name}`",
        f"- Fold manifest checksum: `{evaluation.fold_manifest_checksum}`",
        f"- Fold generation method: `{evaluation.fold_generation_method}`",
        f"- Fold manifest superseded: `{str(evaluation.fold_manifest_superseded).lower()}`",
        f"- Held-out predictions: `{evaluation.prediction_count}`",
        f"- Pooled macro ROC-AUC: `{_format_metric(evaluation.pooled_macro_auc)}`",
        "- Fold macro ROC-AUCs: "
        + ", ".join(f"`{_format_metric(fold.macro_auc)}`" for fold in evaluation.folds),
        f"- Mean fold macro ROC-AUC: `{_format_metric(evaluation.mean_fold_macro_auc)}`",
        f"- Fold macro standard deviation: `{_format_metric(evaluation.fold_macro_auc_std)}`",
    ]
    if diagnostics is not None:
        baseline = diagnostics.prevalence_baseline
        lines.extend(
            [
                "- Pair-weighted within-fold macro ROC-AUC: "
                f"`{_format_metric(diagnostics.within_fold_macro_auc)}`",
                f"- Cross-fold macro ROC-AUC: `{_format_metric(diagnostics.cross_fold_macro_auc)}`",
                "- Mean within-fold pair share: "
                f"`{_format_metric(diagnostics.mean_within_fold_pair_share)}`",
                "- Training-prevalence-only pooled macro ROC-AUC: "
                f"`{_format_metric(baseline.macro_auc)}`",
            ]
        )
        if diagnostics.permutation_test is not None:
            test = diagnostics.permutation_test
            lines.append(
                "- Within-fold permutation test: "
                f"p_upper `{test.p_upper:.6f}` over `{test.resamples}` resamples "
                f"(null mean `{test.null_mean:.6f}`, 95th percentile `{test.null_quantile_95:.6f}`); "
                f"measurable image signal: `{str(test.rejects_no_signal).lower()}`"
            )
    if evaluation.bootstrap is not None:
        macro = evaluation.bootstrap.macro_auc
        lines.append(
            f"- {evaluation.bootstrap.unit.title()} bootstrap pooled macro "
            f"{evaluation.bootstrap.confidence_level:.0%} interval: "
            f"`{_format_metric(macro.lower)}` to `{_format_metric(macro.upper)}` "
            f"({macro.undefined_resamples} undefined of {macro.resamples})"
        )
        within = evaluation.bootstrap.within_fold_macro_auc
        if within is not None:
            lines.append(
                f"- {evaluation.bootstrap.unit.title()} bootstrap within-fold macro "
                f"{evaluation.bootstrap.confidence_level:.0%} interval: "
                f"`{_format_metric(within.lower)}` to `{_format_metric(within.upper)}` "
                f"({within.undefined_resamples} undefined of {within.resamples})"
            )
    lines.extend(
        [
            "",
            _table_row(
                (
                    "Target",
                    "Pooled AUC",
                    "Within-fold AUC",
                    "Cross-fold AUC",
                    "Within share",
                    "Prevalence baseline",
                    "Positives",
                    "Negatives",
                    "Fold 0",
                    "Fold 1",
                    "Fold 2",
                )
            ),
            _table_row(("---", *(("---:",) * 10))),
        ]
    )
    fold_targets = {
        fold.fold: {item.target: item for item in fold.targets}
        for fold in evaluation.folds
    }
    decomposition = (
        {item.target: item for item in diagnostics.pair_decomposition}
        if diagnostics is not None
        else {}
    )
    baseline_targets = (
        {item.target: item for item in diagnostics.prevalence_baseline.targets}
        if diagnostics is not None
        else {}
    )
    for target in evaluation.pooled_targets:
        pairs = decomposition.get(target.target)
        lines.append(
            _table_row(
                (
                    target.target,
                    _format_metric(target.auc),
                    _format_metric(pairs.within_fold_auc if pairs else None),
                    _format_metric(pairs.cross_fold_auc if pairs else None),
                    _format_metric(
                        pairs.within_fold_pairs / pairs.pooled_pairs
                        if pairs and pairs.pooled_pairs
                        else None
                    ),
                    _format_metric(
                        baseline_targets[target.target].auc if target.target in baseline_targets else None
                    ),
                    str(target.positive_count),
                    str(target.negative_count),
                    *(
                        _format_metric(fold_targets[fold][target.target].auc)
                        for fold in FOLD_IDS
                    ),
                )
            )
        )
    if diagnostics is not None:
        lines.extend(
            [
                "",
                "## Fold score scale",
                "",
                "Correlation of fold mean score with fold training prevalence over fold x target "
                f"cells: Pearson `{_format_metric(diagnostics.mean_score_vs_training_prevalence_pearson)}`, "
                f"Spearman `{_format_metric(diagnostics.mean_score_vs_training_prevalence_spearman)}`. "
                "Fold-standardized pooled macro (diagnostic only): "
                f"`{_format_metric(diagnostics.fold_standardized_macro_auc)}`.",
                "",
                _table_row(
                    (
                        "Target",
                        *(f"Fold {fold} mean / SD" for fold in FOLD_IDS),
                        *(f"Fold {fold} train prev." for fold in FOLD_IDS),
                        "Between/within SD",
                    )
                ),
                _table_row(("---", *(("---:",) * 7))),
            ]
        )
        cells = {(item.fold, item.target): item for item in diagnostics.fold_score_scale}
        for index, target_name in enumerate(TARGET_COLUMNS):
            lines.append(
                _table_row(
                    (
                        target_name,
                        *(
                            f"{cells[(fold, target_name)].mean_score:.4f} / "
                            f"{cells[(fold, target_name)].sd_score:.4f}"
                            for fold in FOLD_IDS
                        ),
                        *(f"{cells[(fold, target_name)].training_prevalence:.4f}" for fold in FOLD_IDS),
                        _format_metric(diagnostics.between_fold_to_within_fold_sd_ratio[index]),
                    )
                )
            )
    lines.extend(
        [
            "",
            "Undefined one-class AUCs remain undefined; no reduced-target macro is "
            "used as the primary metric. Within- and cross-fold AUCs partition the pooled "
            "positive/negative pairs exactly; cross-fold pairs compare scores from different "
            "fold models. The prevalence baseline uses no images.",
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
    parser.add_argument("--permutation-resamples", type=int, default=10000)
    parser.add_argument("--permutation-seed", type=int, default=20260911)
    parser.add_argument(
        "--allow-superseded-fold-manifest",
        action="store_true",
        help="re-aggregate runs on a superseded version-1 manifest as historical evidence only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Aggregate complete fold records without reading or mutating their inputs."""

    from .folds import load_fold_manifest, load_inventory_source

    args = _parser().parse_args(argv)
    try:
        parsed_resamples = cast(int, args.bootstrap_resamples)
        if parsed_resamples < 0:
            raise OOFArtifactError("bootstrap resamples cannot be negative")
        permutation_resamples = cast(int, args.permutation_resamples)
        if permutation_resamples < 0:
            raise OOFArtifactError("permutation resamples cannot be negative")
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
            permutation_resamples=permutation_resamples or None,
            permutation_seed=cast(int, args.permutation_seed),
            allow_superseded_fold_manifest=cast(bool, args.allow_superseded_fold_manifest),
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
