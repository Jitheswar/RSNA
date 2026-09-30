"""Fold-0 report-supervision pilot boundaries for Ticket 34."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence, cast

import numpy as np

from .evaluation import FoldEvaluation, evaluate_fold_predictions, roc_auc_score
from .folds import FOLD_IDS, FoldManifest
from .pretrained import (
    RESNET18_ARTIFACT_ID,
    RESNET18_ARTIFACT_BYTES,
    RESNET18_DOWNLOAD_URL,
    RESNET18_SHA256,
    RESNET18_SOURCE_REVISION,
    TICKET34_RESOURCE_SCOPE,
)
from .supervision import (
    AdmittedFoldSupervision,
    ResolvedLabelVector,
    SupervisionAdmissionError,
)
from .targets import NUM_TARGETS, TARGET_COLUMNS


TICKET30_AUDIT_SCHEMA_VERSION = "rsna-knee-report-label-audit-v1"
TICKET30_ADMISSION_POLICY_VERSION = "training-side-wilson-lower-above-class-prevalence-v1"
TICKET30_ADMISSION_POLICY_SHA256 = "40fcb8e6b2d6e691adc7bd38aff697261107bb1360d2411eb85dcd44883bd580"
TICKET30_SOURCE_ID = "ticket16-astra-medium"
TICKET30_AUDITED_ARTIFACT_SHA256 = "33ae5e71499cf91fcc825d903e07d8519557e2bb2e1a40e64891f4c0fe00f6df"
TICKET34_WEAK_LABEL_VERSION = "ticket30-fold-specific-wilson-admission-v4"
TICKET34_MIXTURE_POLICY = "reference-precedence-plus-ticket30-report-only-v1"
TICKET34_ADDITIVE_DESIGN = "ticket34-additive-report-supervision-v1"
TICKET34_WEAK_SAMPLING_POLICY = "ticket34-report-only-cyclic-shuffle-v1"
TICKET34_WEAK_BATCH_SEED_OFFSET = 34_034_034
_BINARY_STATES = {"positive": 1, "negative": 0}
_MASKED_STATES = frozenset({"not-mentioned", "uncertain", "unknown", "extraction-failed"})
_VALID_STATES = frozenset((*_BINARY_STATES, *_MASKED_STATES))
_DIGEST_LENGTH = 64


class Ticket34Error(ValueError):
    """Raised when Ticket 34 inputs cannot form an admissible paired pilot."""


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Ticket34Error(f"{field_name} must be an object")
    return cast(Mapping[str, object], value)


def _digest(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Ticket34Error(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_audit(path: Path) -> Mapping[str, object]:
    if not path.is_file() or path.is_symlink():
        raise Ticket34Error("Ticket 30 admission artifact must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Ticket34Error("Ticket 30 admission artifact could not be read") from exc
    return _mapping(payload, "Ticket 30 admission artifact")


def _validate_target_row(row: object, *, study_uid: str) -> tuple[int | None, ...]:
    if not isinstance(row, list) or len(row) != NUM_TARGETS:
        raise Ticket34Error("every Ticket 30 study row must contain all 12 targets")
    values: list[int | None] = []
    for expected_target, raw_label in zip(TARGET_COLUMNS, row, strict=True):
        label = _mapping(raw_label, f"Ticket 30 target row for {study_uid}")
        if label.get("target") != expected_target:
            raise Ticket34Error("Ticket 30 target rows must use canonical target order")
        state = label.get("state")
        if state not in _VALID_STATES:
            raise Ticket34Error(f"Ticket 30 target {expected_target} has an unknown state")
        mask = label.get("supervision_mask")
        admitted = label.get("admitted")
        value = label.get("value")
        rejection = label.get("rejection_reason")
        if not isinstance(admitted, bool):
            raise Ticket34Error("Ticket 30 admitted flags must be booleans")
        if not isinstance(mask, int) or isinstance(mask, bool) or mask not in (0, 1):
            raise Ticket34Error("Ticket 30 supervision masks must be binary integers")
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1)
        ):
            raise Ticket34Error("Ticket 30 target values must be binary or missing")
        if state in _BINARY_STATES:
            if admitted and value != _BINARY_STATES[cast(str, state)]:
                raise Ticket34Error("Ticket 30 state and binary target value disagree")
            if value is not None and value != _BINARY_STATES[cast(str, state)]:
                raise Ticket34Error("Ticket 30 state and binary target value disagree")
        elif value is not None:
            raise Ticket34Error("masked Ticket 30 states must not carry a target value")
        if admitted:
            if state not in _BINARY_STATES or mask != 1 or rejection is not None:
                raise Ticket34Error("only admitted positive or negative calls may be supervised")
            values.append(cast(int, value))
        else:
            if mask != 0 or not isinstance(rejection, str) or not rejection.strip():
                raise Ticket34Error("rejected Ticket 30 calls must remain masked with a reason")
            if state in _MASKED_STATES and rejection != "state_is_always_masked":
                raise Ticket34Error("uncertain or absent report states must remain masked")
            values.append(None)
    return tuple(values)


def admit_ticket30_fold_supervision(
    manifest: FoldManifest,
    fold: int,
    artifact_path: str | Path,
    *,
    expected_artifact_sha256: str = TICKET30_AUDITED_ARTIFACT_SHA256,
) -> AdmittedFoldSupervision:
    """Convert the frozen Ticket 30 v4 audit into fold-scoped training labels.

    Only Ticket 16 Astra calls admitted by the frozen training-side policy are
    exposed. The adapter never opens report text. Weak rows carry an opaque
    row locator into the hashed audit artifact, not a raw-report reference.
    """

    if fold not in FOLD_IDS:
        raise Ticket34Error(f"fold must be one of {FOLD_IDS}")
    if manifest.superseded:
        raise Ticket34Error("superseded fold manifests cannot admit Ticket 30 labels")
    try:
        expected_digest = _digest(expected_artifact_sha256, "expected_artifact_sha256")
    except Ticket34Error:
        raise
    path = Path(artifact_path)
    payload = _read_audit(path)
    if _sha256(path) != expected_digest:
        raise Ticket34Error("Ticket 30 admission artifact SHA-256 does not match the frozen identity")
    if payload.get("schema_version") != TICKET30_AUDIT_SCHEMA_VERSION:
        raise Ticket34Error("Ticket 30 admission artifact schema is unsupported")
    if payload.get("status") != "completed_existing_artifact_audit":
        raise Ticket34Error("Ticket 30 admission artifact is not a completed audit")

    policy = _mapping(payload.get("policy"), "Ticket 30 admission policy")
    if (
        policy.get("schema_version") != "rsna-knee-report-admission-policy-v1"
        or policy.get("policy_version") != TICKET30_ADMISSION_POLICY_VERSION
        or policy.get("policy_sha256") != TICKET30_ADMISSION_POLICY_SHA256
        or policy.get("target_order") != list(TARGET_COLUMNS)
        or policy.get("held_out_evaluation_used_to_fit_policy") is not False
    ):
        raise Ticket34Error("Ticket 30 admission policy identity or held-out boundary disagrees")

    lineage = _mapping(payload.get("lineage"), "Ticket 30 lineage")
    fold_identity = _mapping(lineage.get("fold_identity"), "Ticket 30 fold identity")
    ticket16 = _mapping(lineage.get("ticket16"), "Ticket 16 source lineage")
    if (
        fold_identity.get("fold_manifest_checksum") != manifest.manifest_checksum
        or ticket16.get("fold_manifest_checksum") != manifest.manifest_checksum
    ):
        raise Ticket34Error("Ticket 30 admission lineage does not match the corrected fold manifest")
    if lineage.get("held_out_outputs_used_in_their_own_fold_policy_fit") is not False:
        raise Ticket34Error("Ticket 30 policy fitting is not proven to exclude held-out outputs")
    for field_name in ("private_report_execution_approved", "prompt_approved", "prompt_frozen"):
        if ticket16.get(field_name) is not True:
            raise Ticket34Error(f"Ticket 16 lineage does not prove {field_name}")
    if ticket16.get("prompt_tuning_or_adaptation") is not False:
        raise Ticket34Error("Ticket 16 prompt adaptation is not excluded")
    if ticket16.get("validation_failure_count") != 0:
        raise Ticket34Error("Ticket 16 source has validation failures")
    model_fingerprint = _digest(
        ticket16.get("model_identity_fingerprint"), "Ticket 16 model identity fingerprint"
    )
    prompt_fingerprint = _digest(ticket16.get("prompt_fingerprint"), "Ticket 16 prompt fingerprint")
    model_revision = ticket16.get("model_revision")
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise Ticket34Error("Ticket 16 source has no frozen model revision")
    experiment_id = ticket16.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise Ticket34Error("Ticket 16 source has no frozen experiment identity")

    source_rows = payload.get("fold_specific_training_admissions")
    if not isinstance(source_rows, list):
        raise Ticket34Error("Ticket 30 fold-specific admissions are missing")
    matching_sources = [
        _mapping(item, "Ticket 30 source admission")
        for item in source_rows
        if isinstance(item, Mapping) and item.get("source_id") == TICKET30_SOURCE_ID
    ]
    if len(matching_sources) != 1:
        raise Ticket34Error("Ticket 30 must contain exactly one frozen Ticket 16 source")
    source_folds = matching_sources[0].get("folds")
    if not isinstance(source_folds, list) or len(source_folds) != len(FOLD_IDS):
        raise Ticket34Error("Ticket 30 source admissions must contain all three folds")
    fold_rows: dict[int, Mapping[str, object]] = {}
    for raw_fold in source_folds:
        item = _mapping(raw_fold, "Ticket 30 fold admission")
        item_fold = item.get("fold")
        if not isinstance(item_fold, int) or isinstance(item_fold, bool) or item_fold not in FOLD_IDS:
            raise Ticket34Error("Ticket 30 source contains an invalid fold id")
        if item_fold in fold_rows:
            raise Ticket34Error("Ticket 30 source repeats a fold")
        fold_rows[item_fold] = item
    if set(fold_rows) != set(FOLD_IDS):
        raise Ticket34Error("Ticket 30 source admission folds are incomplete")

    selected = fold_rows[fold]
    calibration_folds = selected.get("calibration_folds")
    if calibration_folds != [value for value in FOLD_IDS if value != fold]:
        raise Ticket34Error("Ticket 30 policy calibration folds are not outside the consuming fold")
    expected_gold_training = {
        item.study_instance_uid for item in manifest.gold_memberships if item.fold != fold
    }
    expected_held_out = set(manifest.held_out_study_uids(fold))
    expected_report_training = {
        item.study_instance_uid
        for item in manifest.report_only_memberships
        if item.fold != fold
    }
    if selected.get("eligible_report_only_study_count") != len(expected_report_training):
        raise Ticket34Error("Ticket 30 eligible report-only count disagrees with the fold manifest")
    raw_studies = selected.get("studies")
    if not isinstance(raw_studies, list):
        raise Ticket34Error("Ticket 30 fold admission study rows are missing")
    membership_by_uid = {
        item.study_instance_uid: item for item in manifest.report_only_memberships
    }
    row_by_uid: dict[str, Mapping[str, object]] = {}
    values_by_uid: dict[str, tuple[int | None, ...]] = {}
    for raw_row in raw_studies:
        row = _mapping(raw_row, "Ticket 30 report-only study row")
        uid = row.get("study_instance_uid")
        if not isinstance(uid, str) or not uid.strip():
            raise Ticket34Error("Ticket 30 study rows require a study identifier")
        if uid in row_by_uid:
            raise Ticket34Error("Ticket 30 fold admission repeats a report-only study")
        if uid not in expected_report_training:
            raise Ticket34Error("Ticket 30 report-only row is held out or outside the fold manifest")
        membership = membership_by_uid[uid]
        if (
            row.get("assigned_fold") != membership.fold
            or row.get("consuming_fold") != fold
        ):
            raise Ticket34Error("Ticket 30 study assignment disagrees with the fold manifest")
        report_digest = _digest(row.get("report_sha256"), "Ticket 30 report digest")
        values = _validate_target_row(row.get("targets"), study_uid=uid)
        row_by_uid[uid] = {**row, "report_sha256": report_digest}
        values_by_uid[uid] = values
    if set(row_by_uid) != expected_report_training:
        raise Ticket34Error("Ticket 30 report-only rows do not cover the eligible training fold")

    admitted_uids = {
        uid for uid, values in values_by_uid.items() if any(value is not None for value in values)
    }
    admitted_target_calls = sum(
        sum(value is not None for value in values) for values in values_by_uid.values()
    )
    admitted_study_count = len(admitted_uids)
    if (
        selected.get("admitted_studies_with_any_call") != admitted_study_count
        or selected.get("admitted_target_call_count") != admitted_target_calls
    ):
        raise Ticket34Error("Ticket 30 admitted call counts do not reconcile with its study rows")

    resolved: list[ResolvedLabelVector] = []
    for uid in sorted(expected_gold_training):
        membership = next(item for item in manifest.gold_memberships if item.study_instance_uid == uid)
        resolved.append(
            ResolvedLabelVector(
                study_instance_uid=uid,
                targets=cast(tuple[int | None, ...], membership.reference_labels),
                supervision_mask=(1,) * NUM_TARGETS,
                source_kind="reference",
            )
        )
    for uid in sorted(admitted_uids):
        row = row_by_uid[uid]
        resolved.append(
            ResolvedLabelVector(
                study_instance_uid=uid,
                targets=values_by_uid[uid],
                supervision_mask=tuple(int(value is not None) for value in values_by_uid[uid]),
                source_kind="weak",
                source_report_ref=f"ticket30-audit-v4:{expected_digest}:{uid}",
                source_report_sha256=cast(str, row["report_sha256"]),
            )
        )
    resolved.sort(key=lambda item: item.study_instance_uid)
    return AdmittedFoldSupervision(
        fold=fold,
        fold_manifest_checksum=manifest.manifest_checksum,
        source_artifact_digest=expected_digest,
        source_schema_version=TICKET30_AUDIT_SCHEMA_VERSION,
        extractor_version=model_revision,
        extractor_configuration_fingerprint=model_fingerprint,
        extractor_ruleset_fingerprint=prompt_fingerprint,
        report_source_version=str(fold_identity.get("report_source_version", experiment_id)),
        weak_label_version=TICKET34_WEAK_LABEL_VERSION,
        mixture_policy=TICKET34_MIXTURE_POLICY,
        training_study_uids=tuple(sorted(expected_gold_training | admitted_uids)),
        held_out_study_uids=tuple(sorted(expected_held_out)),
        eligible_report_only_study_uids=tuple(sorted(expected_report_training)),
        admitted_report_only_study_uids=tuple(sorted(admitted_uids)),
        missing_report_only_study_uids=tuple(sorted(expected_report_training - admitted_uids)),
        resolved=tuple(resolved),
        source_id=TICKET30_SOURCE_ID,
    )


def _read_run_artifact(path: str | Path) -> tuple[Path, Mapping[str, object], Mapping[str, object]]:
    candidate = Path(path)
    run_path = candidate / "run.json" if candidate.is_dir() else candidate
    directory = run_path.parent
    if not run_path.is_file() or run_path.is_symlink():
        raise Ticket34Error(f"completed Ticket 34 run record is missing: {run_path}")
    configuration_path = directory / "configuration.json"
    if not configuration_path.is_file() or configuration_path.is_symlink():
        raise Ticket34Error(f"Ticket 34 run is missing configuration.json: {directory}")
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Ticket34Error(f"Ticket 34 run artifacts could not be read: {directory}") from exc
    return directory, _mapping(run, "run record"), _mapping(configuration, "run configuration")


def _string_set(value: object, field_name: str) -> set[str]:
    if not isinstance(value, (list, tuple)):
        raise Ticket34Error(f"{field_name} must be an array")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise Ticket34Error(f"{field_name} must contain non-empty identifiers")
        result.append(item)
    if len(set(result)) != len(result):
        raise Ticket34Error(f"{field_name} must not contain duplicate identifiers")
    return set(result)


def _positive_integer(value: object, field_name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Ticket34Error(f"{field_name} must be an integer of at least {minimum}")
    return value


def _validated_predictions(value: object, field_name: str) -> dict[str, tuple[float, ...]]:
    raw = _mapping(value, field_name)
    predictions: dict[str, tuple[float, ...]] = {}
    for uid, scores in raw.items():
        if not isinstance(uid, str) or not uid.strip():
            raise Ticket34Error(f"{field_name} contains an invalid study identity")
        if not isinstance(scores, (list, tuple)) or len(scores) != NUM_TARGETS:
            raise Ticket34Error(f"{field_name} predictions must contain all 12 targets")
        try:
            parsed = tuple(float(score) for score in scores)
        except (TypeError, ValueError, OverflowError) as exc:
            raise Ticket34Error(f"{field_name} contains a non-numeric score") from exc
        if any(not np.isfinite(score) or not 0.0 <= score <= 1.0 for score in parsed):
            raise Ticket34Error(f"{field_name} scores must be finite values in [0, 1]")
        predictions[uid] = parsed
    return predictions


def _interval(values: Sequence[float], *, resamples: int, seed: int, confidence: float) -> dict[str, object]:
    if values:
        alpha = (1.0 - confidence) / 2.0
        low, median, high = np.quantile(np.asarray(values, dtype=np.float64), [alpha, 0.5, 1.0 - alpha])
        bounds: dict[str, object] = {
            "lower": float(low),
            "median": float(median),
            "upper": float(high),
        }
    else:
        bounds = {"lower": None, "median": None, "upper": None}
    return {
        "method": "paired_study_percentile_bootstrap",
        **bounds,
        "defined_resamples": len(values),
        "undefined_resamples": resamples - len(values),
        "resamples": resamples,
        "seed": seed,
        "confidence_level": confidence,
    }


def _load_one_arm(
    run_path: str | Path,
    *,
    fold_manifest: FoldManifest,
    expected_fold: int,
) -> tuple[Path, Mapping[str, object], Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    directory, run, configuration = _read_run_artifact(run_path)
    if run.get("status") != "completed_fixed_budget_training":
        raise Ticket34Error("Ticket 34 comparison requires completed fixed-budget runs")
    if run.get("fold") != expected_fold or configuration.get("fold") != expected_fold:
        raise Ticket34Error("both run and configuration must identify the selected pilot fold")
    if run.get("run_id") != configuration.get("run_id"):
        raise Ticket34Error("run and configuration identities disagree")
    if run.get("config_fingerprint") != configuration.get("scientific_configuration_fingerprint"):
        raise Ticket34Error("run and configuration fingerprints disagree")
    if run.get("config") != configuration.get("config"):
        raise Ticket34Error("run and configuration training settings disagree")
    if run.get("semantic_fingerprint") != configuration.get("semantic_fingerprint"):
        raise Ticket34Error("run and configuration scientific identities disagree")
    if run.get("held_out_inputs_excluded_from_fitting") is not True:
        raise Ticket34Error("run does not prove held-out inputs were excluded from fitting")
    preflight = _mapping(configuration.get("preflight"), "configuration preflight")
    if preflight.get("passed") is not True:
        raise Ticket34Error("run does not carry a passed CPU preflight")
    identity = _mapping(configuration.get("identity"), "configuration identity")
    if identity.get("fold_manifest_checksum") != fold_manifest.manifest_checksum:
        raise Ticket34Error("run uses a different fold manifest")
    inference = _mapping(configuration.get("inference_configuration"), "inference configuration")
    if inference.get("checkpoint_selection") != "final_checkpoint":
        raise Ticket34Error("both arms must use the final-checkpoint selection policy")

    checkpoint = _mapping(run.get("checkpoint"), "run checkpoint")
    if checkpoint.get("final_present") is not True or checkpoint.get("latest_present") is not True:
        raise Ticket34Error("completed run is missing its final or resumable checkpoint")
    final_path = directory / "checkpoints" / "final.pt"
    latest_path = directory / "checkpoints" / "latest.pt"
    if not final_path.is_file() or final_path.is_symlink() or not latest_path.is_file() or latest_path.is_symlink():
        raise Ticket34Error("completed run checkpoint files are missing or not regular files")
    final_digest = _digest(checkpoint.get("final_sha256"), "final checkpoint SHA-256")
    if _sha256(final_path) != final_digest:
        raise Ticket34Error("final checkpoint SHA-256 does not match the run record")
    counts = _mapping(run.get("counts"), "run counts")
    return directory, run, configuration, identity, counts


def _validate_arm_training(
    run: Mapping[str, object],
    configuration: Mapping[str, object],
    identity: Mapping[str, object],
    counts: Mapping[str, object],
    *,
    report_arm: bool,
    fold_manifest: FoldManifest,
    fold: int,
    expected_gold_training: set[str],
    expected_held_out_gold: set[str],
    expected_report_training: set[str],
    expected_artifact_sha256: str,
) -> tuple[dict[str, tuple[float, ...]], Mapping[str, object] | None, Mapping[str, Mapping[str, int]]]:
    reference_only = run.get("reference_supervision_only")
    if not isinstance(reference_only, bool) or reference_only is report_arm:
        raise Ticket34Error("run supervision arm does not match its declared comparison role")
    training_uids = _string_set(run.get("training_study_uids"), "run training_study_uids")
    heldout_uids = _string_set(run.get("held_out_study_uids"), "run held_out_study_uids")
    if heldout_uids != expected_held_out_gold:
        raise Ticket34Error("run held-out identities do not match the selected fold's gold cohort")
    if not heldout_uids.isdisjoint(training_uids):
        raise Ticket34Error("held-out studies appear in the training partition")
    predictions = _validated_predictions(run.get("predictions"), "run predictions")
    if set(predictions) != expected_held_out_gold:
        raise Ticket34Error("both arms must have the same complete held-out prediction cohort")
    if _positive_integer(counts.get("training_studies"), "training study count") != len(training_uids):
        raise Ticket34Error("run training count does not match its training identities")
    if _positive_integer(counts.get("held_out_studies"), "held-out study count") != len(heldout_uids):
        raise Ticket34Error("run held-out count does not match its held-out identities")

    supervision_value = run.get("supervision")
    supervision: Mapping[str, object] | None
    if report_arm:
        supervision = _mapping(supervision_value, "Ticket 34 report supervision")
        if (
            supervision.get("kind") != "weak"
            or supervision.get("schema_version") != "rsna-knee-supervision-admission-v1"
            or supervision.get("fold") != fold
            or supervision.get("fold_manifest_checksum") != fold_manifest.manifest_checksum
            or supervision.get("source_artifact_digest") != expected_artifact_sha256
            or supervision.get("source_schema_version") != TICKET30_AUDIT_SCHEMA_VERSION
            or supervision.get("source_id") != TICKET30_SOURCE_ID
            or supervision.get("weak_label_version") != TICKET34_WEAK_LABEL_VERSION
            or supervision.get("mixture_policy") != TICKET34_MIXTURE_POLICY
        ):
            raise Ticket34Error("Ticket 34 report run is bound to a different admitted source or policy")
        expected_heldout_supervision = set(fold_manifest.held_out_study_uids(fold))
        if _string_set(supervision.get("held_out_study_uids"), "report supervision held-out studies") != expected_heldout_supervision:
            raise Ticket34Error("report supervision held-out lineage disagrees with the fold manifest")
        eligible = _string_set(
            supervision.get("eligible_report_only_study_uids"),
            "report supervision eligible report-only studies",
        )
        admitted = _string_set(
            supervision.get("admitted_report_only_study_uids"),
            "report supervision admitted report-only studies",
        )
        missing = _string_set(
            supervision.get("missing_report_only_study_uids"),
            "report supervision missing report-only studies",
        )
        image_excluded = _string_set(
            supervision.get("excluded_image_report_only_study_uids"),
            "report supervision image-excluded report-only studies",
        )
        if eligible != expected_report_training:
            raise Ticket34Error("report supervision eligible identities disagree with the fold manifest")
        if admitted.intersection(missing | image_excluded) or missing.intersection(image_excluded):
            raise Ticket34Error("report admission, missing and image-excluded sets overlap")
        if eligible != admitted | missing | image_excluded:
            raise Ticket34Error("report-only admissions and exclusions do not reconcile")
        if not admitted or not admitted.issubset(expected_report_training):
            raise Ticket34Error("report treatment has no eligible training-side report labels")
        resolved_rows = supervision.get("resolved_studies")
        if not isinstance(resolved_rows, list):
            raise Ticket34Error("report supervision is missing its resolved study identities")
        resolved_by_uid: dict[str, str] = {}
        for raw_row in resolved_rows:
            item = _mapping(raw_row, "resolved report supervision row")
            uid = item.get("study_instance_uid")
            source_kind = item.get("source_kind")
            if not isinstance(uid, str) or not uid.strip() or source_kind not in {"reference", "weak"}:
                raise Ticket34Error("resolved supervision contains an invalid study source")
            if uid in resolved_by_uid:
                raise Ticket34Error("resolved supervision repeats a study")
            resolved_by_uid[uid] = cast(str, source_kind)
        if set(resolved_by_uid) != training_uids or training_uids != expected_gold_training | admitted:
            raise Ticket34Error("report training identities do not match gold plus image-backed admitted rows")
        if {uid for uid, kind in resolved_by_uid.items() if kind == "weak"} != admitted:
            raise Ticket34Error("resolved weak rows do not match the image-backed admission set")
        if {uid for uid, kind in resolved_by_uid.items() if kind == "reference"} != expected_gold_training:
            raise Ticket34Error("report run must preserve every gold training study as reference-labeled")
        boundaries = _mapping(supervision.get("boundaries"), "report supervision boundaries")
        for field_name in (
            "held_out_inputs_excluded",
            "held_out_report_labels_evaluation_only",
            "reference_labels_take_precedence",
            "unknown_uncertain_failed_masked",
        ):
            if boundaries.get(field_name) is not True:
                raise Ticket34Error(f"report supervision does not prove {field_name}")
        if boundaries.get("fallback_source") != "trusted_reference_labels_only":
            raise Ticket34Error("report supervision does not use the reference-only fallback")
    else:
        supervision = _mapping(supervision_value, "reference supervision")
        if (
            supervision.get("kind") != "reference"
            or supervision.get("fold_manifest_checksum") != fold_manifest.manifest_checksum
            or supervision.get("held_out_inputs_excluded") is not True
            or supervision.get("reports_used") is not False
            or training_uids != expected_gold_training
        ):
            raise Ticket34Error("reference run does not contain only the fold's gold training studies")

    config = _mapping(run.get("config"), "run training configuration")
    optimization = _mapping(config.get("optimization"), "run optimization configuration")
    expected_steps = _positive_integer(
        optimization.get("successful_optimizer_steps"), "configured successful optimizer steps"
    )
    batch_size = _positive_integer(optimization.get("batch_size"), "configured batch size")
    successful = _positive_integer(counts.get("successful_optimizer_steps"), "successful optimizer steps")
    scheduler = _positive_integer(counts.get("scheduler_steps"), "scheduler steps")
    attempted = _positive_integer(counts.get("attempted_batches"), "attempted batches")
    skipped = _positive_integer(
        counts.get("skipped_all_masked_batches"), "skipped all-masked batches", allow_zero=True
    )
    if successful != expected_steps or scheduler != successful or attempted != successful or skipped != 0:
        raise Ticket34Error("paired arms must complete the same fixed successful-step budget")
    if counts.get("training_design") != TICKET34_ADDITIVE_DESIGN:
        raise Ticket34Error("both arms must use the declared Ticket 34 additive training design")
    exposures_raw = _mapping(counts.get("exposures_by_source"), "source-specific exposures")
    if not {"reference", "weak", "mixed"}.issubset(exposures_raw):
        raise Ticket34Error("source-specific exposures must separate reference, weak and mixed data")
    exposures: dict[str, dict[str, int]] = {}
    for source_kind, raw_counts in exposures_raw.items():
        if source_kind not in {"reference", "weak", "mixed", "pseudo"}:
            raise Ticket34Error("source-specific exposures contain an unknown source kind")
        group = _mapping(raw_counts, f"{source_kind} exposure counts")
        exposures[source_kind] = {
            "sample_exposures": _positive_integer(
                group.get("sample_exposures"), f"{source_kind} sample exposures", allow_zero=True
            ),
            "supervised_entry_exposures": _positive_integer(
                group.get("supervised_entry_exposures"),
                f"{source_kind} supervised-entry exposures",
                allow_zero=True,
            ),
        }
    sample_exposures = _positive_integer(counts.get("sample_exposures"), "sample exposures")
    supervised_exposures = _positive_integer(
        counts.get("supervised_entry_exposures"), "supervised-entry exposures"
    )
    if sum(item["sample_exposures"] for item in exposures.values()) != sample_exposures:
        raise Ticket34Error("source sample exposures do not reconcile with total exposure")
    if sum(item["supervised_entry_exposures"] for item in exposures.values()) != supervised_exposures:
        raise Ticket34Error("source supervised-entry exposures do not reconcile")
    expected_reference_samples = successful * batch_size
    expected_total_samples = expected_reference_samples * (2 if report_arm else 1)
    if (
        exposures["reference"]["sample_exposures"] != expected_reference_samples
        or exposures["reference"]["supervised_entry_exposures"] != expected_reference_samples * NUM_TARGETS
        or exposures["weak"]["sample_exposures"] != (expected_reference_samples if report_arm else 0)
        or (report_arm and exposures["weak"]["supervised_entry_exposures"] <= 0)
        or exposures["mixed"]["sample_exposures"] != 0
        or exposures["mixed"]["supervised_entry_exposures"] != 0
        or sample_exposures != expected_total_samples
    ):
        raise Ticket34Error("reference and report exposure counts do not match the additive policy")
    _digest(counts.get("gold_exposure_sequence_sha256"), "gold exposure sequence SHA-256")
    _digest(counts.get("initial_model_state_sha256"), "initial model state SHA-256")
    if report_arm:
        _digest(counts.get("weak_exposure_sequence_sha256"), "weak exposure sequence SHA-256")
        if counts.get("weak_sampling_policy") != TICKET34_WEAK_SAMPLING_POLICY:
            raise Ticket34Error("report treatment uses an undeclared weak-label sampling policy")
        seeds = _mapping(config.get("seeds"), "training seeds")
        batch_order_seed = _positive_integer(
            seeds.get("batch_order"), "batch-order seed", allow_zero=True
        )
        if counts.get("weak_sampling_seed") != batch_order_seed + TICKET34_WEAK_BATCH_SEED_OFFSET:
            raise Ticket34Error("report weak-label sampling seed differs from the frozen policy")

    approval = _mapping(identity.get("pretrained_resource_approval"), "pretrained resource approval")
    if (
        approval.get("artifact_id") != RESNET18_ARTIFACT_ID
        or approval.get("source_revision") != RESNET18_SOURCE_REVISION
        or approval.get("download_url") != RESNET18_DOWNLOAD_URL
        or approval.get("expected_bytes") != RESNET18_ARTIFACT_BYTES
        or approval.get("scope") != TICKET34_RESOURCE_SCOPE
        or approval.get("decision") != "approved"
        or approval.get("rights_status") != "eligible"
        or approval.get("external_spend") != 0
        or approval.get("sha256") != RESNET18_SHA256
    ):
        raise Ticket34Error("run is not bound to the exact approved Ticket 34 pretrained resource scope")
    resource_request = _mapping(configuration.get("resources"), "requested resources")
    if resource_request.get("requested_partition") != "cluster1-cpu":
        raise Ticket34Error("Ticket 34 run must use the approved CPU partition")
    if identity.get("source_version") != fold_manifest.source_version:
        raise Ticket34Error("run source version disagrees with the fold manifest")
    return predictions, supervision, cast(dict[str, Mapping[str, int]], exposures)


def compare_supervision_pilot(
    reference_run_path: str | Path,
    report_run_path: str | Path,
    fold_manifest: FoldManifest,
    *,
    expected_ticket30_artifact_sha256: str = TICKET30_AUDITED_ARTIFACT_SHA256,
    image_study_uids: Sequence[str],
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 20260925,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Validate and compare the matched one-fold Ticket 34 run artifacts."""

    fold = 0
    if fold_manifest.superseded:
        raise Ticket34Error("superseded fold manifests cannot support the Ticket 34 pilot")
    expected_artifact = _digest(
        expected_ticket30_artifact_sha256, "expected Ticket 30 artifact SHA-256"
    )
    if bootstrap_resamples <= 0 or bootstrap_seed < 0:
        raise Ticket34Error("bootstrap resamples must be positive and seed non-negative")
    if not 0.0 < confidence_level < 1.0:
        raise Ticket34Error("bootstrap confidence level must be between zero and one")

    reference_directory, reference, reference_configuration, reference_identity, reference_counts = _load_one_arm(
        reference_run_path, fold_manifest=fold_manifest, expected_fold=fold
    )
    report_directory, report, report_configuration, report_identity, report_counts = _load_one_arm(
        report_run_path, fold_manifest=fold_manifest, expected_fold=fold
    )
    if reference_directory == report_directory:
        raise Ticket34Error("reference and report arms must use distinct run directories")
    if reference.get("run_id") == report.get("run_id"):
        raise Ticket34Error("reference and report arms must have distinct run identities")
    if reference.get("config_fingerprint") != report.get("config_fingerprint"):
        raise Ticket34Error("training configuration differs between supervision arms")
    if reference.get("config") != report.get("config"):
        raise Ticket34Error("training settings differ between supervision arms")

    comparable_identity_fields = (
        "source_version",
        "source_fingerprint",
        "fold_manifest_checksum",
        "integration_manifest_checksum",
        "configuration_fingerprint",
        "training_schema_version",
        "model_source",
        "architecture",
        "initialization",
        "preprocessing",
        "packing",
        "optimization",
        "loss",
        "augmentation",
        "budget",
        "seeds",
        "pretrained_resource_approval",
    )
    for field_name in comparable_identity_fields:
        if reference_identity.get(field_name) != report_identity.get(field_name):
            raise Ticket34Error(f"paired training identity differs for {field_name}")
    for field_name in ("source_fingerprint", "integration_manifest_checksum"):
        _digest(reference_identity.get(field_name), field_name)
    if reference_identity.get("fold_manifest_checksum") != fold_manifest.manifest_checksum:
        raise Ticket34Error("paired runs do not use the requested fold manifest")
    if reference_configuration.get("preflight") != report_configuration.get("preflight"):
        raise Ticket34Error("paired runs do not share the same CPU preflight evidence")

    expected_gold_training = {
        item.study_instance_uid for item in fold_manifest.gold_memberships if item.fold != fold
    }
    expected_held_out_gold = {
        item.study_instance_uid for item in fold_manifest.gold_memberships if item.fold == fold
    }
    expected_report_training = {
        item.study_instance_uid
        for item in fold_manifest.report_only_memberships
        if item.fold != fold
    }
    image_uids = _string_set(image_study_uids, "image-backed study identities")
    manifest_uids = {item.study_instance_uid for item in fold_manifest.memberships}
    if not image_uids.issubset(manifest_uids):
        raise Ticket34Error("image cohort contains studies outside the frozen fold manifest")
    if not expected_gold_training.issubset(image_uids):
        raise Ticket34Error("image cohort is missing a fold-0 gold training study")
    if not expected_held_out_gold.issubset(image_uids):
        raise Ticket34Error("image cohort is missing a fold-0 held-out gold study")
    reference_predictions, _, reference_exposures = _validate_arm_training(
        reference,
        reference_configuration,
        reference_identity,
        reference_counts,
        report_arm=False,
        fold_manifest=fold_manifest,
        fold=fold,
        expected_gold_training=expected_gold_training,
        expected_held_out_gold=expected_held_out_gold,
        expected_report_training=expected_report_training,
        expected_artifact_sha256=expected_artifact,
    )
    report_predictions, report_supervision, report_exposures = _validate_arm_training(
        report,
        report_configuration,
        report_identity,
        report_counts,
        report_arm=True,
        fold_manifest=fold_manifest,
        fold=fold,
        expected_gold_training=expected_gold_training,
        expected_held_out_gold=expected_held_out_gold,
        expected_report_training=expected_report_training,
        expected_artifact_sha256=expected_artifact,
    )
    assert report_supervision is not None
    eligible_report_images = expected_report_training.intersection(image_uids)
    admitted_report_images = _string_set(
        report_supervision.get("admitted_report_only_study_uids"),
        "image-backed admitted report-only studies",
    )
    missing_report = _string_set(
        report_supervision.get("missing_report_only_study_uids"),
        "report-only studies without admitted calls",
    )
    excluded_image_report = _string_set(
        report_supervision.get("excluded_image_report_only_study_uids"),
        "admitted report-only studies without images",
    )
    source_admitted_reports = admitted_report_images | excluded_image_report
    if admitted_report_images != eligible_report_images.intersection(source_admitted_reports):
        raise Ticket34Error("image-backed report admissions do not match the frozen image cohort")
    if excluded_image_report != source_admitted_reports - eligible_report_images:
        raise Ticket34Error("image-excluded report admissions do not reconcile with image membership")
    image_report_no_call = eligible_report_images - source_admitted_reports
    if not image_report_no_call.issubset(missing_report):
        raise Ticket34Error("image-backed reports without admitted calls are not recorded as masked")
    if reference_counts.get("gold_exposure_sequence_sha256") != report_counts.get(
        "gold_exposure_sequence_sha256"
    ):
        raise Ticket34Error("reference exposure sequence differs between arms")
    if reference_counts.get("initial_model_state_sha256") != report_counts.get(
        "initial_model_state_sha256"
    ):
        raise Ticket34Error("initial model state differs between arms")
    if reference_exposures["reference"] != report_exposures["reference"]:
        raise Ticket34Error("reference exposure counts differ between arms")

    try:
        reference_evaluation = evaluate_fold_predictions(reference_predictions, fold_manifest, fold)
        report_evaluation = evaluate_fold_predictions(report_predictions, fold_manifest, fold)
    except ValueError as exc:
        raise Ticket34Error(f"paired prediction evaluation failed: {exc}") from exc
    reference_metrics = {item.target: item for item in reference_evaluation.targets}
    report_metrics = {item.target: item for item in report_evaluation.targets}
    memberships = {item.study_instance_uid: item for item in fold_manifest.gold_memberships}
    ordered_uids = tuple(sorted(expected_held_out_gold))
    labels = np.asarray([memberships[uid].reference_labels for uid in ordered_uids], dtype=np.int8)
    reference_scores = np.asarray([reference_predictions[uid] for uid in ordered_uids], dtype=np.float64)
    report_scores = np.asarray([report_predictions[uid] for uid in ordered_uids], dtype=np.float64)
    rng = np.random.default_rng(bootstrap_seed)
    macro_deltas: list[float] = []
    target_deltas: list[list[float]] = [[] for _ in TARGET_COLUMNS]
    for _ in range(bootstrap_resamples):
        sample = rng.integers(0, len(ordered_uids), size=len(ordered_uids))
        deltas: list[float | None] = []
        for target_index in range(NUM_TARGETS):
            reference_auc = roc_auc_score(
                labels[sample, target_index], reference_scores[sample, target_index]
            )
            report_auc = roc_auc_score(
                labels[sample, target_index], report_scores[sample, target_index]
            )
            delta = (
                None
                if reference_auc is None or report_auc is None
                else float(report_auc - reference_auc)
            )
            deltas.append(delta)
            if delta is not None:
                target_deltas[target_index].append(delta)
        if all(delta is not None for delta in deltas):
            macro_deltas.append(float(np.mean([cast(float, delta) for delta in deltas])))

    per_target: list[dict[str, object]] = []
    for index, target in enumerate(TARGET_COLUMNS):
        reference_metric = reference_metrics[target]
        report_metric = report_metrics[target]
        delta = (
            None
            if reference_metric.auc is None or report_metric.auc is None
            else report_metric.auc - reference_metric.auc
        )
        per_target.append(
            {
                "target": target,
                "positive_count": reference_metric.positive_count,
                "negative_count": reference_metric.negative_count,
                "reference_auc": reference_metric.auc,
                "report_auc": report_metric.auc,
                "delta_auc_report_minus_reference": delta,
                "paired_study_bootstrap": _interval(
                    target_deltas[index],
                    resamples=bootstrap_resamples,
                    seed=bootstrap_seed,
                    confidence=confidence_level,
                ),
            }
        )
    supervision_counts = _mapping(report_supervision.get("counts"), "report supervision counts")
    report_weak_exposures = report_exposures["weak"]
    return {
        "schema_version": "rsna-ticket34-one-fold-supervision-comparison-v1",
        "fold": fold,
        "fold_manifest_checksum": fold_manifest.manifest_checksum,
        "image_integration_manifest_checksum": reference_identity["integration_manifest_checksum"],
        "source_fingerprint": reference_identity["source_fingerprint"],
        "ticket30_admitted_artifact_sha256": expected_artifact,
        "admitted_source_id": TICKET30_SOURCE_ID,
        "cohort": {
            "held_out_studies": len(ordered_uids),
            "gold_training_studies": len(expected_gold_training),
            "eligible_report_only_training_studies": len(expected_report_training),
            "image_backed_admitted_report_only_studies": _positive_integer(
                supervision_counts.get("admitted_report_only_studies"),
                "image-backed admitted report-only study count",
                allow_zero=True,
            ),
            "missing_report_only_studies": _positive_integer(
                supervision_counts.get("missing_report_only_studies"),
                "missing report-only study count",
                allow_zero=True,
            ),
            "image_excluded_report_only_studies": _positive_integer(
                supervision_counts.get("image_excluded_report_only_studies"),
                "image-excluded report-only study count",
                allow_zero=True,
            ),
            "image_membership": {
                "integration_studies": len(image_uids),
                "gold_training_studies_with_images": len(expected_gold_training),
                "held_out_gold_studies_with_images": len(expected_held_out_gold),
                "eligible_report_only_training_studies_with_images": len(eligible_report_images),
                "eligible_report_only_training_studies_without_images": len(
                    expected_report_training - image_uids
                ),
                "image_backed_admitted_report_only_studies": len(admitted_report_images),
                "image_backed_report_only_studies_without_admitted_calls": len(
                    image_report_no_call
                ),
                "admitted_report_only_studies_without_images": len(excluded_image_report),
                "image_backed_report_only_uid_set_sha256": hashlib.sha256(
                    "\n".join(sorted(eligible_report_images)).encode("utf-8")
                ).hexdigest(),
                "image_backed_admitted_report_uid_set_sha256": hashlib.sha256(
                    "\n".join(sorted(admitted_report_images)).encode("utf-8")
                ).hexdigest(),
            },
        },
        "reference_macro_auc": reference_evaluation.macro_auc,
        "report_macro_auc": report_evaluation.macro_auc,
        "delta_macro_auc_report_minus_reference": (
            None
            if reference_evaluation.macro_auc is None or report_evaluation.macro_auc is None
            else report_evaluation.macro_auc - reference_evaluation.macro_auc
        ),
        "paired_study_bootstrap": {
            "unit": "study",
            "macro_auc_delta": _interval(
                macro_deltas,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
                confidence=confidence_level,
            ),
            "macro_requires_all_12_targets_defined": True,
        },
        "per_target": per_target,
        "exposures": {
            "reference_arm": dict(reference_exposures["reference"]),
            "report_arm_reference": dict(report_exposures["reference"]),
            "report_arm_weak": {
                **report_weak_exposures,
                "masked_target_exposures": (
                    report_weak_exposures["sample_exposures"] * NUM_TARGETS
                    - report_weak_exposures["supervised_entry_exposures"]
                ),
                "sampling_policy": report_counts.get("weak_sampling_policy"),
                "sampling_seed": report_counts.get("weak_sampling_seed"),
                "sequence_sha256": report_counts.get("weak_exposure_sequence_sha256"),
            },
            "same_reference_exposure_sequence_sha256": reference_counts.get(
                "gold_exposure_sequence_sha256"
            ),
            "same_initial_model_state_sha256": reference_counts.get(
                "initial_model_state_sha256"
            ),
        },
        "resource_request": dict(_mapping(reference_configuration.get("resources"), "reference resources")),
        "one_fold_limit": True,
        "limitations": [
            "This fold-0 pilot cannot establish a complete three-fold gain.",
            "Study bootstrap uncertainty does not rule out repeated-patient leakage because reliable grouping is unavailable.",
            "Undefined target AUCs and bootstrap resamples are reported without imputation.",
        ],
    }


__all__ = [
    "TICKET30_ADMISSION_POLICY_SHA256",
    "TICKET30_ADMISSION_POLICY_VERSION",
    "TICKET30_AUDIT_SCHEMA_VERSION",
    "TICKET30_AUDITED_ARTIFACT_SHA256",
    "TICKET30_SOURCE_ID",
    "TICKET34_MIXTURE_POLICY",
    "TICKET34_ADDITIVE_DESIGN",
    "TICKET34_RESOURCE_SCOPE",
    "TICKET34_WEAK_SAMPLING_POLICY",
    "TICKET34_WEAK_BATCH_SEED_OFFSET",
    "TICKET34_WEAK_LABEL_VERSION",
    "Ticket34Error",
    "admit_ticket30_fold_supervision",
    "compare_supervision_pilot",
]
