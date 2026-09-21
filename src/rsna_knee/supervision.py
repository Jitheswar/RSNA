"""Fold-scoped admission of report labels into image training.

The module owns the supervision seam for Ticket 17.  It consumes a complete,
frozen Ticket 15 extraction result and returns target vectors that the image
training path can use.  Image loading, preprocessing, packing and model
training are deliberately outside this module; a report can therefore never
be discovered by an imaging dataset as a side effect.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence, cast

from .folds import ADAPTATION_EXCLUSIONS, FOLD_IDS, FoldManifest
from .report_extraction import (
    REPORT_CORPUS_SCHEMA_VERSION,
    REPORT_EXTRACTION_SCHEMA_VERSION,
    REPORT_TRAINING_POLICY_VERSION,
    FoldReportExtraction,
    FrozenDeterministicExtractor,
    ReportExtractionRun,
    ReportLabelRecord,
    ReportPartition,
)
from .report_labels import LabelState, RuleOriginKind
from .targets import NUM_TARGETS, TARGET_COLUMNS


ADMISSION_SCHEMA_VERSION = "rsna-knee-supervision-admission-v1"
WEAK_LABEL_VERSION = REPORT_TRAINING_POLICY_VERSION
MIXTURE_POLICY_VERSION = "reference-precedence-plus-report-only-v1"
_SUPERVISED_STATES = frozenset({LabelState.POSITIVE, LabelState.NEGATIVE})


class SupervisionAdmissionError(ValueError):
    """Raised when derived supervision cannot be proven fold-clean."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_digest(value: object, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise SupervisionAdmissionError(f"{field_name} must be a SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise SupervisionAdmissionError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _tuple_of_strings(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise SupervisionAdmissionError(f"{field_name} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise SupervisionAdmissionError(f"{field_name} must not contain duplicates")
    return tuple(sorted(result))


def _reference_vector(values: Sequence[int | None]) -> tuple[int | None, ...]:
    result = tuple(values)
    if len(result) != NUM_TARGETS:
        raise SupervisionAdmissionError(
            f"reference labels must contain exactly {NUM_TARGETS} targets"
        )
    if any(value not in (None, 0, 1) for value in result):
        raise SupervisionAdmissionError("reference labels must contain only 0, 1 or missing values")
    return result


@dataclass(frozen=True)
class SupervisionDisagreement:
    """A weak call that was overridden by a reference label."""

    target: str
    reference_value: int
    weak_value: int
    source_report_ref: str
    source_report_sha256: str

    def __post_init__(self) -> None:
        if self.target not in TARGET_COLUMNS:
            raise SupervisionAdmissionError(f"unknown disagreement target {self.target!r}")
        if self.reference_value not in (0, 1) or self.weak_value not in (0, 1):
            raise SupervisionAdmissionError("supervision disagreements must be binary")
        if self.reference_value == self.weak_value:
            raise SupervisionAdmissionError("a disagreement must contain different values")
        if not self.source_report_ref.strip():
            raise SupervisionAdmissionError("supervision disagreement requires a report reference")
        _require_digest(self.source_report_sha256, "source_report_sha256")

    def to_public_dict(self) -> dict[str, object]:
        return {"target": self.target}

    def to_private_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "reference_value": self.reference_value,
            "weak_value": self.weak_value,
            "source_report_ref": self.source_report_ref,
            "source_report_sha256": self.source_report_sha256,
        }


@dataclass(frozen=True)
class ResolvedLabelVector:
    """One study's binary/unknown vector after reference precedence."""

    study_instance_uid: str
    targets: tuple[int | None, ...]
    supervision_mask: tuple[int, ...]
    source_kind: str
    source_report_ref: str | None = None
    source_report_sha256: str | None = None
    disagreements: tuple[SupervisionDisagreement, ...] = ()

    def __post_init__(self) -> None:
        if not self.study_instance_uid.strip():
            raise SupervisionAdmissionError("resolved supervision requires a study identifier")
        if len(self.targets) != NUM_TARGETS or len(self.supervision_mask) != NUM_TARGETS:
            raise SupervisionAdmissionError("resolved supervision vectors must contain 12 targets")
        if self.source_kind not in {"reference", "weak", "mixed"}:
            raise SupervisionAdmissionError("resolved supervision has an unknown source kind")
        for target, mask in zip(self.targets, self.supervision_mask):
            if target not in (None, 0, 1):
                raise SupervisionAdmissionError("resolved targets must contain only 0, 1 or missing values")
            if mask not in (0, 1) or mask != int(target is not None):
                raise SupervisionAdmissionError("resolved targets and supervision masks disagree")
        if self.source_kind in {"weak", "mixed"}:
            if not self.source_report_ref or not self.source_report_sha256:
                raise SupervisionAdmissionError(
                    "weak or mixed supervision requires report provenance"
                )
            _require_digest(self.source_report_sha256, "source_report_sha256")
        elif self.source_report_ref is not None or self.source_report_sha256 is not None:
            raise SupervisionAdmissionError("reference supervision must not carry report provenance")

    @property
    def supervised_entry_count(self) -> int:
        return sum(self.supervision_mask)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "source_kind": self.source_kind,
            "supervised_entry_count": self.supervised_entry_count,
            "disagreement_count": len(self.disagreements),
        }

    def to_private_dict(self) -> dict[str, object]:
        return {
            "study_instance_uid": self.study_instance_uid,
            "targets": list(self.targets),
            "supervision_mask": list(self.supervision_mask),
            "source_kind": self.source_kind,
            "source_report_ref": self.source_report_ref,
            "source_report_sha256": self.source_report_sha256,
            "supervised_entry_count": self.supervised_entry_count,
            "disagreements": [item.to_private_dict() for item in self.disagreements],
        }


def _validate_weak_record(record: ReportLabelRecord) -> None:
    if record.partition is not ReportPartition.TRAINING_LABELS:
        raise SupervisionAdmissionError(
            "only the report extraction training-label partition may supervise training"
        )
    if tuple(item.target for item in record.interpretation.labels) != TARGET_COLUMNS:
        raise SupervisionAdmissionError("weak labels must use canonical target order")
    for label in record.interpretation.labels:
        expected_mask = int(label.state in _SUPERVISED_STATES)
        if label.supervision_mask != expected_mask or (label.value is None) != (expected_mask == 0):
            raise SupervisionAdmissionError(
                f"weak label state and mask disagree for {label.target}"
            )
        if expected_mask:
            if label.confidence != 1.0:
                raise SupervisionAdmissionError(
                    f"weak label {label.target} is not an admissible confident binary call"
                )
            if label.rule_origin.kind is not RuleOriginKind.OFFICIAL_DEFINITION:
                raise SupervisionAdmissionError(
                    f"weak label {label.target} does not have an authoritative rule origin"
                )
            if label.rule_origin.derived_from_report or label.rule_origin.participant_interpretation:
                raise SupervisionAdmissionError(
                    f"weak label {label.target} has contaminated rule provenance"
                )


def merge_reference_and_weak_labels(
    reference_labels: Sequence[int | None],
    weak_record: ReportLabelRecord,
) -> ResolvedLabelVector:
    """Merge one weak record with references, giving references precedence."""

    _validate_weak_record(weak_record)
    reference = _reference_vector(reference_labels)
    weak_values = weak_record.values
    resolved: list[int | None] = []
    disagreements: list[SupervisionDisagreement] = []
    has_reference = any(value is not None for value in reference)
    has_weak = False
    for index, (reference_value, weak_value, label) in enumerate(
        zip(reference, weak_values, weak_record.interpretation.labels)
    ):
        if weak_value is not None:
            has_weak = True
            if reference_value is not None and reference_value != weak_value:
                disagreements.append(
                    SupervisionDisagreement(
                        target=TARGET_COLUMNS[index],
                        reference_value=reference_value,
                        weak_value=weak_value,
                        source_report_ref=weak_record.source_report_ref,
                        source_report_sha256=weak_record.report_sha256,
                    )
                )
        resolved.append(reference_value if reference_value is not None else weak_value)
    source_kind = "mixed" if has_reference and has_weak else "reference" if has_reference else "weak"
    return ResolvedLabelVector(
        study_instance_uid=weak_record.study_instance_uid,
        targets=tuple(resolved),
        supervision_mask=tuple(int(value is not None) for value in resolved),
        source_kind=source_kind,
        source_report_ref=weak_record.source_report_ref if source_kind != "reference" else None,
        source_report_sha256=weak_record.report_sha256 if source_kind != "reference" else None,
        disagreements=tuple(disagreements),
    )


def _membership_digest(manifest: FoldManifest) -> str:
    return _digest(
        [
            {
                "study_instance_uid": item.study_instance_uid,
                "cohort": item.cohort,
                "fold": item.fold,
                "reference_labels": list(item.reference_labels),
            }
            for item in manifest.memberships
        ]
    )


def _validate_upstream_lineage(manifest: FoldManifest, run: ReportExtractionRun) -> None:
    if manifest.superseded:
        raise SupervisionAdmissionError("superseded fold manifests cannot admit weak labels")
    if manifest.fold_ids != FOLD_IDS:
        raise SupervisionAdmissionError("weak-label admission requires the canonical three-fold manifest")
    if run.schema_version != REPORT_EXTRACTION_SCHEMA_VERSION:
        raise SupervisionAdmissionError("upstream report extraction schema is unsupported")
    if not run.reproducibility_verified:
        raise SupervisionAdmissionError(
            "weak-label admission requires a reproducibility-verified report extraction run"
        )
    evaluation = run.evaluation
    if evaluation.fold_manifest_checksum != manifest.manifest_checksum:
        raise SupervisionAdmissionError("weak-label lineage does not match the fold manifest")
    if evaluation.repeat_name != manifest.repeat_name:
        raise SupervisionAdmissionError("weak-label lineage has a different fold repeat")
    upstream = evaluation.upstream_input_manifest
    fold_input = upstream.get("fold_manifest")
    corpus_input = upstream.get("report_corpus")
    extractor_input = upstream.get("extractor")
    exclusions = upstream.get("held_out_adaptation_exclusions")
    if not isinstance(fold_input, Mapping) or not isinstance(corpus_input, Mapping):
        raise SupervisionAdmissionError("weak-label upstream lineage is incomplete")
    if not isinstance(extractor_input, Mapping) or not isinstance(exclusions, list):
        raise SupervisionAdmissionError("weak-label upstream lineage is incomplete")
    if fold_input.get("manifest_checksum") != manifest.manifest_checksum:
        raise SupervisionAdmissionError("weak-label upstream fold checksum is contaminated or missing")
    if fold_input.get("schema_version") != manifest.schema_version:
        raise SupervisionAdmissionError("weak-label upstream fold schema is contaminated or missing")
    if fold_input.get("repeat_name") != manifest.repeat_name:
        raise SupervisionAdmissionError("weak-label upstream repeat is contaminated or missing")
    if fold_input.get("eligible_study_set_sha256") != _membership_digest(manifest):
        raise SupervisionAdmissionError("weak-label upstream study-set lineage is incomplete")
    if corpus_input.get("source_version") != run.report_corpus.source_version:
        raise SupervisionAdmissionError("weak-label report source lineage is incomplete")
    if corpus_input.get("schema_version") != REPORT_CORPUS_SCHEMA_VERSION:
        raise SupervisionAdmissionError("weak-label report schema lineage is incomplete")
    if corpus_input.get("record_set_sha256") != run.report_corpus.record_set_sha256:
        raise SupervisionAdmissionError("weak-label report-set lineage is incomplete")
    extractor = FrozenDeterministicExtractor()
    expected_extractor = {
        "version": extractor.extractor_version,
        "code_fingerprint": extractor.to_public_dict()["code_fingerprint"],
        "configuration_fingerprint": extractor.configuration_fingerprint,
        "ruleset_fingerprint": extractor.ruleset_fingerprint,
        "checkpoint_hashes": [],
        "upstream_adaptation_inputs": [],
        "report_derived_rule_development": False,
    }
    if dict(extractor_input) != expected_extractor:
        raise SupervisionAdmissionError("weak-label extractor lineage is incomplete or contaminated")
    if (
        evaluation.extractor_version != expected_extractor["version"]
        or evaluation.configuration_fingerprint != expected_extractor["configuration_fingerprint"]
        or evaluation.ruleset_fingerprint != expected_extractor["ruleset_fingerprint"]
        or evaluation.code_fingerprint != expected_extractor["code_fingerprint"]
        or evaluation.report_source_version != run.report_corpus.source_version
    ):
        raise SupervisionAdmissionError("weak-label evaluation lineage is incomplete or contaminated")
    if tuple(exclusions) != ADAPTATION_EXCLUSIONS:
        raise SupervisionAdmissionError("weak-label held-out exclusions are incomplete")


@dataclass(frozen=True)
class AdmittedFoldSupervision:
    """Fold-scoped resolved labels ready for image-example construction."""

    fold: int
    fold_manifest_checksum: str
    source_artifact_digest: str
    source_schema_version: str
    extractor_version: str
    extractor_configuration_fingerprint: str
    extractor_ruleset_fingerprint: str
    report_source_version: str
    weak_label_version: str
    mixture_policy: str
    training_study_uids: tuple[str, ...]
    held_out_study_uids: tuple[str, ...]
    eligible_report_only_study_uids: tuple[str, ...]
    admitted_report_only_study_uids: tuple[str, ...]
    missing_report_only_study_uids: tuple[str, ...]
    resolved: tuple[ResolvedLabelVector, ...]
    source_disagreements: tuple[SupervisionDisagreement, ...] = ()

    def __post_init__(self) -> None:
        if self.fold not in FOLD_IDS:
            raise SupervisionAdmissionError(f"fold must be one of {FOLD_IDS}")
        _require_digest(self.fold_manifest_checksum, "fold_manifest_checksum")
        _require_digest(self.source_artifact_digest, "source_artifact_digest")
        _require_digest(self.extractor_configuration_fingerprint, "extractor_configuration_fingerprint")
        _require_digest(self.extractor_ruleset_fingerprint, "extractor_ruleset_fingerprint")
        for name, values in (
            ("training_study_uids", self.training_study_uids),
            ("held_out_study_uids", self.held_out_study_uids),
            ("eligible_report_only_study_uids", self.eligible_report_only_study_uids),
            ("admitted_report_only_study_uids", self.admitted_report_only_study_uids),
            ("missing_report_only_study_uids", self.missing_report_only_study_uids),
        ):
            if tuple(sorted(values)) != values or len(set(values)) != len(values):
                raise SupervisionAdmissionError(f"{name} must be unique and sorted")
        if set(self.training_study_uids).intersection(self.held_out_study_uids):
            raise SupervisionAdmissionError("admitted training and held-out studies overlap")
        eligible = set(self.eligible_report_only_study_uids)
        if not set(self.admitted_report_only_study_uids).issubset(eligible):
            raise SupervisionAdmissionError("admitted weak studies are outside the eligible set")
        if set(self.admitted_report_only_study_uids).intersection(self.missing_report_only_study_uids):
            raise SupervisionAdmissionError("admitted and missing weak studies overlap")
        if eligible != set(self.admitted_report_only_study_uids).union(self.missing_report_only_study_uids):
            raise SupervisionAdmissionError("eligible weak studies do not reconcile with missing reports")
        resolved_by_uid = {item.study_instance_uid: item for item in self.resolved}
        if len(resolved_by_uid) != len(self.resolved):
            raise SupervisionAdmissionError("resolved supervision repeats a study")
        if set(resolved_by_uid) != set(self.training_study_uids):
            raise SupervisionAdmissionError("resolved supervision does not cover training studies")
        if set(self.admitted_report_only_study_uids) != {
            item.study_instance_uid for item in self.resolved if item.source_kind in {"weak", "mixed"}
        }:
            raise SupervisionAdmissionError("weak supervision rows do not reconcile with admitted studies")

    @property
    def schema_version(self) -> str:
        return ADMISSION_SCHEMA_VERSION

    @property
    def by_study(self) -> dict[str, ResolvedLabelVector]:
        return {item.study_instance_uid: item for item in self.resolved}

    @property
    def reference_study_uids(self) -> tuple[str, ...]:
        return tuple(item.study_instance_uid for item in self.resolved if item.source_kind == "reference")

    @property
    def weak_study_count(self) -> int:
        return len(self.admitted_report_only_study_uids)

    @property
    def supervised_entry_count(self) -> int:
        return sum(item.supervised_entry_count for item in self.resolved)

    def provenance_for(self, study_instance_uid: str) -> dict[str, object]:
        item = self.by_study.get(study_instance_uid)
        if item is None:
            raise SupervisionAdmissionError(
                f"study {study_instance_uid} is not in the admitted training supervision"
            )
        return {
            "source_kind": item.source_kind,
            "source_id": (
                f"report-extraction:{self.source_artifact_digest}"
                if item.source_kind in {"weak", "mixed"}
                else f"fold-manifest:{self.fold_manifest_checksum}"
            ),
            "fold": self.fold,
            "held_out_excluded": True,
            "eligible_study_uids": list(self.training_study_uids),
            "excluded_study_uids": list(self.held_out_study_uids),
            "weak_label_version": self.weak_label_version if item.source_kind in {"weak", "mixed"} else None,
            "mixture_policy": self.mixture_policy,
            "source_report_ref": item.source_report_ref,
            "source_report_sha256": item.source_report_sha256,
            "source_disagreement_count": len(item.disagreements),
        }

    def to_public_dict(self) -> dict[str, object]:
        reference_count = len(self.reference_study_uids)
        return {
            "schema_version": self.schema_version,
            "fold": self.fold,
            "fold_manifest_checksum": self.fold_manifest_checksum,
            "source_artifact_digest": self.source_artifact_digest,
            "source_schema_version": self.source_schema_version,
            "weak_label_version": self.weak_label_version,
            "mixture_policy": self.mixture_policy,
            "extractor": {
                "version": self.extractor_version,
                "configuration_fingerprint": self.extractor_configuration_fingerprint,
                "ruleset_fingerprint": self.extractor_ruleset_fingerprint,
            },
            "counts": {
                "training_studies": len(self.training_study_uids),
                "reference_studies": reference_count,
                "weak_studies": self.weak_study_count,
                "eligible_report_only_studies": len(self.eligible_report_only_study_uids),
                "admitted_report_only_studies": len(self.admitted_report_only_study_uids),
                "missing_report_only_studies": len(self.missing_report_only_study_uids),
                "supervised_entries": self.supervised_entry_count,
                "source_disagreements": len(self.source_disagreements),
            },
            "boundaries": {
                "held_out_inputs_excluded": True,
                "held_out_report_labels_evaluation_only": True,
                "reference_labels_take_precedence": True,
                "unknown_uncertain_failed_masked": True,
                "fallback_source": "trusted_reference_labels_only",
            },
        }

    def to_private_dict(self) -> dict[str, object]:
        payload = self.to_public_dict()
        payload["training_study_uids"] = list(self.training_study_uids)
        payload["held_out_study_uids"] = list(self.held_out_study_uids)
        payload["eligible_report_only_study_uids"] = list(self.eligible_report_only_study_uids)
        payload["admitted_report_only_study_uids"] = list(self.admitted_report_only_study_uids)
        payload["missing_report_only_study_uids"] = list(self.missing_report_only_study_uids)
        payload["resolved_studies"] = [item.to_private_dict() for item in self.resolved]
        payload["source_disagreements"] = [
            item.to_private_dict() for item in self.source_disagreements
        ]
        return payload


def _fold_report_result(run: ReportExtractionRun, fold: int) -> FoldReportExtraction:
    try:
        result = run.folds[fold]
    except (IndexError, TypeError) as exc:
        raise SupervisionAdmissionError(f"report extraction has no fold {fold}") from exc
    if result.fold != fold:
        raise SupervisionAdmissionError("report extraction folds are not in canonical order")
    return result


def admit_fold_supervision(
    manifest: FoldManifest,
    fold: int,
    report_extraction: ReportExtractionRun,
) -> AdmittedFoldSupervision:
    """Admit clean Ticket 15 report labels for one image-training fold."""

    if fold not in FOLD_IDS:
        raise SupervisionAdmissionError(f"fold must be one of {FOLD_IDS}")
    if not isinstance(report_extraction, ReportExtractionRun):
        raise SupervisionAdmissionError("weak-label admission requires a ReportExtractionRun")
    _validate_upstream_lineage(manifest, report_extraction)
    fold_result = _fold_report_result(report_extraction, fold)
    expected_training = set(manifest.training_study_uids(fold))
    expected_held_out = set(manifest.held_out_study_uids(fold))
    expected_eligible_weak = {
        item.study_instance_uid
        for item in manifest.report_only_memberships
        if item.study_instance_uid in expected_training
    }
    expected_evaluation_gold = {
        item.study_instance_uid
        for item in manifest.gold_memberships
        if item.study_instance_uid in expected_held_out
    }
    if set(fold_result.fold_training_study_uids) != expected_training:
        raise SupervisionAdmissionError("report extraction training study identities do not match the fold")
    if set(fold_result.fold_held_out_study_uids) != expected_held_out:
        raise SupervisionAdmissionError("report extraction held-out study identities do not match the fold")
    if set(fold_result.eligible_training_report_uids) != expected_eligible_weak:
        raise SupervisionAdmissionError("report extraction eligible weak-study identities do not match the fold")
    if set(fold_result.eligible_evaluation_gold_uids) != expected_evaluation_gold:
        raise SupervisionAdmissionError("report extraction evaluation identities do not match the fold")
    corpus_uids = set(report_extraction.report_corpus.by_study)
    expected_excluded = expected_held_out.intersection(corpus_uids)
    if set(fold_result.excluded_held_out_report_uids) != expected_excluded:
        raise SupervisionAdmissionError("held-out report exclusion lineage does not match the fold")
    if set(record.study_instance_uid for record in fold_result.evaluation_labels).intersection(
        set(record.study_instance_uid for record in fold_result.training_labels)
    ):
        raise SupervisionAdmissionError("report training and evaluation partitions overlap")

    training_records = {record.study_instance_uid: record for record in fold_result.training_labels}
    report_only_memberships = {
        item.study_instance_uid: item for item in manifest.report_only_memberships
    }
    if not set(training_records).issubset(expected_eligible_weak):
        raise SupervisionAdmissionError("report training labels include a held-out or unknown study")
    for record in training_records.values():
        if record.study_instance_uid not in report_only_memberships:
            raise SupervisionAdmissionError("only report-only studies may supply weak training labels")
        source_report = report_extraction.report_corpus.by_study[record.study_instance_uid]
        if source_report.source_report_ref != record.source_report_ref:
            raise SupervisionAdmissionError("weak label source reference disagrees with the report corpus")
        if source_report.report_sha256 != record.report_sha256:
            raise SupervisionAdmissionError("weak label source digest disagrees with the report corpus")
        _validate_weak_record(record)

    missing = expected_eligible_weak.difference(training_records)
    resolved: list[ResolvedLabelVector] = []
    disagreements: list[SupervisionDisagreement] = []
    for uid in sorted(expected_training):
        membership = next(item for item in manifest.memberships if item.study_instance_uid == uid)
        weak_record = training_records.get(uid)
        if membership.cohort == "gold":
            if any(value is None for value in membership.reference_labels):
                raise SupervisionAdmissionError("gold training labels must be complete")
            if weak_record is None:
                resolved_item = ResolvedLabelVector(
                    study_instance_uid=uid,
                    targets=cast(tuple[int | None, ...], membership.reference_labels),
                    supervision_mask=(1,) * NUM_TARGETS,
                    source_kind="reference",
                )
            else:
                resolved_item = merge_reference_and_weak_labels(
                    membership.reference_labels,
                    weak_record,
                )
        elif weak_record is not None:
            resolved_item = merge_reference_and_weak_labels(
                (None,) * NUM_TARGETS,
                weak_record,
            )
        else:
            continue
        resolved.append(resolved_item)
        disagreements.extend(resolved_item.disagreements)

    return AdmittedFoldSupervision(
        fold=fold,
        fold_manifest_checksum=manifest.manifest_checksum,
        source_artifact_digest=report_extraction.semantic_digest,
        source_schema_version=report_extraction.schema_version,
        extractor_version=report_extraction.evaluation.extractor_version,
        extractor_configuration_fingerprint=report_extraction.evaluation.configuration_fingerprint,
        extractor_ruleset_fingerprint=report_extraction.evaluation.ruleset_fingerprint,
        report_source_version=report_extraction.report_corpus.source_version,
        weak_label_version=WEAK_LABEL_VERSION,
        mixture_policy=MIXTURE_POLICY_VERSION,
        training_study_uids=tuple(item.study_instance_uid for item in resolved),
        held_out_study_uids=tuple(sorted(expected_held_out)),
        eligible_report_only_study_uids=tuple(sorted(expected_eligible_weak)),
        admitted_report_only_study_uids=tuple(sorted(training_records)),
        missing_report_only_study_uids=tuple(sorted(missing)),
        resolved=tuple(resolved),
        source_disagreements=tuple(disagreements),
    )


__all__ = [
    "ADMISSION_SCHEMA_VERSION",
    "MIXTURE_POLICY_VERSION",
    "WEAK_LABEL_VERSION",
    "AdmittedFoldSupervision",
    "ResolvedLabelVector",
    "SupervisionAdmissionError",
    "SupervisionDisagreement",
    "admit_fold_supervision",
    "merge_reference_and_weak_labels",
]
