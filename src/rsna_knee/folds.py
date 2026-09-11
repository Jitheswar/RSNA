"""Deterministic, immutable study-fold generation for reference evaluation.

Fold membership is created from the reconciled training source manifest before
any report-derived supervision is admitted.  The module does not inspect
reports or images; keeping that boundary here makes it possible to freeze the
evaluation unit without importing training or inference orchestration.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from .inventory import SourceManifest, SourceSeries, SourceStudy
from .targets import TARGET_COLUMNS


FOLD_SCHEMA_VERSION = "rsna-knee-fold-manifest-v1"
PUBLIC_FOLD_SCHEMA_VERSION = "rsna-knee-fold-evidence-v1"
FOLD_COUNT = 3
FOLD_IDS: tuple[int, ...] = tuple(range(FOLD_COUNT))
FOLD_GENERATION_METHOD = (
    "gold:iterative-multilabel-greedy-v1;"
    "report-only:stable-study-or-reliable-group-identifier-hash-v1;"
    "groups:unsplit-v1"
)
ADAPTATION_EXCLUSIONS: tuple[str, ...] = ("images", "metadata", "reports")
_GROUPING_STATUSES = frozenset({"reliable", "unavailable", "untrustworthy"})
_SUPPORT_STATUSES = frozenset({"both_classes", "one_class_unavoidable"})


class FoldFreezeError(ValueError):
    """Raised when a fold manifest cannot be safely generated or validated."""


class FoldManifestIntegrityError(FoldFreezeError):
    """Raised when a persisted fold manifest is not self-consistent."""


class ImmutableArtifactExistsError(FoldFreezeError):
    """Raised when an immutable artifact path already contains an artifact."""


@dataclass(frozen=True)
class FoldGenerationConfig:
    """Named deterministic controls for one fold-generation repeat."""

    seed_name: str
    seed: int
    repeat_name: str = "repeat-1"
    repeat_justification: str | None = None

    def __post_init__(self) -> None:
        if not self.seed_name.strip() or not self.repeat_name.strip():
            raise FoldFreezeError("seed_name and repeat_name must be non-empty")
        if self.repeat_name.strip() != "repeat-1" and not (self.repeat_justification or "").strip():
            raise FoldFreezeError("additional repeats require a justification")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GroupingAssessment:
    """Evidence about whether a patient/group key is safe to use.

    A key is used only when ``status`` is ``reliable``.  Supplying a key with
    an ``untrustworthy`` status is intentionally rejected by
    :func:`freeze_folds`; it is never silently downgraded to study splitting.
    """

    status: str
    examined_fields: tuple[str, ...] = ()
    field_name: str | None = None
    scope: str | None = None
    provenance: str | None = None
    reason: str = ""
    study_to_group: tuple[tuple[str, str], ...] = ()
    missing_key_count: int | None = None
    within_study_consistent: bool | None = None
    cross_study_reuse_count: int | None = None
    issuer_scope_verified: bool | None = None
    contradiction_count: int | None = None

    def __post_init__(self) -> None:
        if self.status not in _GROUPING_STATUSES:
            raise FoldFreezeError(f"grouping status must be one of {sorted(_GROUPING_STATUSES)}")
        if any(not field.strip() for field in self.examined_fields):
            raise FoldFreezeError("examined grouping fields must be non-empty")
        if len(set(self.examined_fields)) != len(self.examined_fields):
            raise FoldFreezeError("examined grouping fields must be unique")
        if not self.reason.strip():
            raise FoldFreezeError("grouping assessment requires a reason")
        if self.status == "reliable":
            if not self.field_name or not self.scope or not self.provenance:
                raise FoldFreezeError(
                    "a reliable grouping assessment requires field_name, scope and provenance"
                )
            if (
                self.missing_key_count != 0
                or self.within_study_consistent is not True
                or self.cross_study_reuse_count is None
                or self.issuer_scope_verified is not True
                or self.contradiction_count != 0
            ):
                raise FoldFreezeError(
                    "a reliable grouping assessment requires complete zero-contradiction audit results"
                )
        for field_name, count in (
            ("missing_key_count", self.missing_key_count),
            ("cross_study_reuse_count", self.cross_study_reuse_count),
            ("contradiction_count", self.contradiction_count),
        ):
            if count is not None and (isinstance(count, bool) or count < 0):
                raise FoldFreezeError(f"grouping {field_name} must be a non-negative integer")
        for field_name, value in (
            ("within_study_consistent", self.within_study_consistent),
            ("issuer_scope_verified", self.issuer_scope_verified),
        ):
            if value is not None and not isinstance(value, bool):
                raise FoldFreezeError(f"grouping {field_name} must be boolean or null")
        if len({study_uid for study_uid, _ in self.study_to_group}) != len(self.study_to_group):
            raise FoldFreezeError("grouping assessment contains duplicate study identifiers")
        if any(not study_uid.strip() or not group_key.strip() for study_uid, group_key in self.study_to_group):
            raise FoldFreezeError("grouping assessment contains an empty study or group identifier")

    @classmethod
    def no_reliable_key(
        cls,
        *,
        examined_fields: Sequence[str] = (),
        reason: str = (
            "No reliable patient key was established from authoritative source metadata; "
            "the baseline therefore uses StudyInstanceUID study-level separation."
        ),
    ) -> "GroupingAssessment":
        """Create the explicit study-only limitation required by the protocol."""

        return cls(
            status="unavailable",
            examined_fields=tuple(examined_fields),
            reason=reason,
        )

    @classmethod
    def reliable_key(
        cls,
        study_to_group: Mapping[str, str],
        *,
        field_name: str,
        scope: str,
        provenance: str,
        examined_fields: Sequence[str] = (),
        reason: str = "The declared grouping key passed the documented scope and consistency checks.",
        missing_key_count: int = 0,
        within_study_consistent: bool = True,
        cross_study_reuse_count: int = 0,
        issuer_scope_verified: bool = True,
        contradiction_count: int = 0,
    ) -> "GroupingAssessment":
        """Create an explicitly attested grouping assessment."""

        return cls(
            status="reliable",
            examined_fields=tuple(examined_fields),
            field_name=field_name.strip(),
            scope=scope.strip(),
            provenance=provenance.strip(),
            reason=reason,
            study_to_group=tuple(
                sorted((key.strip(), value.strip()) for key, value in study_to_group.items())
            ),
            missing_key_count=missing_key_count,
            within_study_consistent=within_study_consistent,
            cross_study_reuse_count=cross_study_reuse_count,
            issuer_scope_verified=issuer_scope_verified,
            contradiction_count=contradiction_count,
        )

    @classmethod
    def untrustworthy_key(
        cls,
        *,
        field_name: str,
        examined_fields: Sequence[str],
        reason: str,
        study_to_group: Mapping[str, str] | None = None,
        missing_key_count: int | None = None,
        within_study_consistent: bool | None = None,
        cross_study_reuse_count: int | None = None,
        issuer_scope_verified: bool | None = None,
        contradiction_count: int | None = None,
    ) -> "GroupingAssessment":
        """Record contradictory evidence; generation must remain blocked."""

        return cls(
            status="untrustworthy",
            examined_fields=tuple(examined_fields),
            field_name=field_name.strip(),
            reason=reason,
            study_to_group=tuple(
                sorted(
                    (key.strip(), value.strip())
                    for key, value in (study_to_group or {}).items()
                )
            ),
            missing_key_count=missing_key_count,
            within_study_consistent=within_study_consistent,
            cross_study_reuse_count=cross_study_reuse_count,
            issuer_scope_verified=issuer_scope_verified,
            contradiction_count=contradiction_count,
        )

    @property
    def reliable(self) -> bool:
        return self.status == "reliable"

    @property
    def repeated_patient_leakage_cannot_be_ruled_out(self) -> bool:
        return not self.reliable

    @property
    def group_by_study(self) -> dict[str, str]:
        return dict(self.study_to_group)

    def to_private_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "examined_fields": list(self.examined_fields),
            "field_name": self.field_name,
            "scope": self.scope,
            "provenance": self.provenance,
            "reason": self.reason,
            "study_to_group": {key: value for key, value in self.study_to_group},
            "missing_key_count": self.missing_key_count,
            "within_study_consistent": self.within_study_consistent,
            "cross_study_reuse_count": self.cross_study_reuse_count,
            "issuer_scope_verified": self.issuer_scope_verified,
            "contradiction_count": self.contradiction_count,
            "repeated_patient_leakage_cannot_be_ruled_out": (
                self.repeated_patient_leakage_cannot_be_ruled_out
            ),
        }

    def to_public_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "examined_fields": list(self.examined_fields),
            "field_name": self.field_name,
            "scope": self.scope,
            "missing_key_count": self.missing_key_count,
            "within_study_consistent": self.within_study_consistent,
            "cross_study_reuse_count": self.cross_study_reuse_count,
            "issuer_scope_verified": self.issuer_scope_verified,
            "contradiction_count": self.contradiction_count,
            "repeated_patient_leakage_cannot_be_ruled_out": (
                self.repeated_patient_leakage_cannot_be_ruled_out
            ),
            "limitation": (
                "Repeated-patient leakage cannot be ruled out because no reliable patient key "
                "was established."
                if self.repeated_patient_leakage_cannot_be_ruled_out
                else (
                    "Fold allocation respects the attested grouping key; "
                    "patient-grouped leakage is checked at generation."
                )
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GroupingAssessment":
        examined_fields_value = payload.get("examined_fields", ())
        if not isinstance(examined_fields_value, (list, tuple)):
            raise FoldFreezeError("grouping examined_fields must be a list")
        mapping_value = payload.get("study_to_group", {})
        if not isinstance(mapping_value, Mapping):
            raise FoldFreezeError("grouping study_to_group must be an object")
        return cls(
            status=str(payload.get("status", "")),
            examined_fields=tuple(str(item) for item in examined_fields_value),
            field_name=_optional_string(payload.get("field_name")),
            scope=_optional_string(payload.get("scope")),
            provenance=_optional_string(payload.get("provenance")),
            reason=str(payload.get("reason", "")),
            study_to_group=tuple(sorted((str(key), str(value)) for key, value in mapping_value.items())),
            missing_key_count=_optional_nonnegative_int(payload.get("missing_key_count")),
            within_study_consistent=_optional_bool(payload.get("within_study_consistent")),
            cross_study_reuse_count=_optional_nonnegative_int(payload.get("cross_study_reuse_count")),
            issuer_scope_verified=_optional_bool(payload.get("issuer_scope_verified")),
            contradiction_count=_optional_nonnegative_int(payload.get("contradiction_count")),
        )


@dataclass(frozen=True)
class FoldMembership:
    """One immutable study-to-fold assignment."""

    study_instance_uid: str
    cohort: str
    fold: int
    reference_labels: tuple[int | None, ...]
    group_key: str | None = None

    def __post_init__(self) -> None:
        if not self.study_instance_uid.strip():
            raise FoldFreezeError("fold membership requires a StudyInstanceUID")
        if self.cohort not in {"gold", "report_only"}:
            raise FoldFreezeError("fold membership cohort must be gold or report_only")
        if self.fold not in FOLD_IDS:
            raise FoldFreezeError(f"fold must be one of {FOLD_IDS}")
        if len(self.reference_labels) != len(TARGET_COLUMNS):
            raise FoldFreezeError("fold membership reference vector must contain all targets")
        if any(label not in {None, 0, 1} for label in self.reference_labels):
            raise FoldFreezeError("fold membership labels may contain only 0, 1 or missing values")
        if self.cohort == "gold" and any(label is None for label in self.reference_labels):
            raise FoldFreezeError("gold fold membership requires complete reference labels")
        if self.cohort == "report_only" and any(label is not None for label in self.reference_labels):
            raise FoldFreezeError("report-only fold membership cannot contain reference labels")
        if self.group_key is not None and not self.group_key.strip():
            raise FoldFreezeError("fold membership group_key must be non-empty when present")


@dataclass(frozen=True)
class TargetFoldCounts:
    """Gold positive/negative counts and the support conclusion for one target."""

    positive: int
    negative: int
    positive_group_count: int
    negative_group_count: int
    support_status: str
    support_reason: str

    def __post_init__(self) -> None:
        if min(self.positive, self.negative, self.positive_group_count, self.negative_group_count) < 0:
            raise FoldFreezeError("fold target counts cannot be negative")
        if self.support_status not in _SUPPORT_STATUSES:
            raise FoldFreezeError("unknown fold target support status")
        if not self.support_reason.strip():
            raise FoldFreezeError("fold target support requires a reason")


@dataclass(frozen=True)
class FoldSummary:
    """Aggregate membership and gold support for one declared fold."""

    fold: int
    study_count: int
    gold_study_count: int
    report_only_study_count: int
    target_counts: tuple[tuple[str, TargetFoldCounts], ...]

    def __post_init__(self) -> None:
        if self.fold not in FOLD_IDS:
            raise FoldFreezeError(f"fold summary must use one of {FOLD_IDS}")
        if self.study_count != self.gold_study_count + self.report_only_study_count:
            raise FoldFreezeError("fold summary cohort counts do not add up")
        if tuple(target for target, _ in self.target_counts) != TARGET_COLUMNS:
            raise FoldFreezeError("fold summary must contain targets in competition order")

    @property
    def target_counts_by_name(self) -> dict[str, TargetFoldCounts]:
        return dict(self.target_counts)


@dataclass(frozen=True)
class FoldManifest:
    """Self-checking private fold manifest.

    The object is immutable in memory and its checksum covers every private
    membership row, grouping assignment, generation parameter, and support
    count.  Public evidence is derived from this object without membership
    identifiers.
    """

    schema_version: str
    source_split: str
    source_version: str
    source_manifest_checksum: str
    source_fingerprint: str | None
    fold_count: int
    fold_ids: tuple[int, ...]
    repeat_name: str
    repeat_justification: str | None
    generation_method: str
    seed_name: str
    seed: int
    grouping: GroupingAssessment
    memberships: tuple[FoldMembership, ...]
    fold_summaries: tuple[FoldSummary, ...]
    adaptation_exclusions: tuple[str, ...]
    manifest_checksum: str
    input_inventory_reconciled: bool | None = None

    def __post_init__(self) -> None:
        if self.schema_version != FOLD_SCHEMA_VERSION:
            raise FoldManifestIntegrityError("unexpected fold manifest schema version")
        if self.source_split != "train":
            raise FoldManifestIntegrityError("fold manifests must be generated from the train split")
        if not self.source_version.strip():
            raise FoldManifestIntegrityError("fold manifest requires a source version")
        if not self.source_manifest_checksum.strip():
            raise FoldManifestIntegrityError("fold manifest requires a source manifest checksum")
        if self.source_fingerprint is not None and not self.source_fingerprint.strip():
            raise FoldManifestIntegrityError("source fingerprint must be non-empty when present")
        if self.input_inventory_reconciled is not None and not isinstance(
            self.input_inventory_reconciled, bool
        ):
            raise FoldManifestIntegrityError("input inventory reconciliation status must be boolean or null")
        if self.fold_count != FOLD_COUNT or self.fold_ids != FOLD_IDS:
            raise FoldManifestIntegrityError("ticket 04 requires exactly folds 0, 1 and 2")
        if not self.repeat_name.strip():
            raise FoldManifestIntegrityError("fold manifest requires a repeat name")
        if self.repeat_name != "repeat-1" and not (self.repeat_justification or "").strip():
            raise FoldManifestIntegrityError("additional repeats require a justification")
        if not self.generation_method.strip() or not self.seed_name.strip():
            raise FoldManifestIntegrityError("fold generation metadata is incomplete")
        if self.adaptation_exclusions != ADAPTATION_EXCLUSIONS:
            raise FoldManifestIntegrityError(
                "fold adaptation must exclude held-out images, metadata and reports"
            )
        self._validate_membership_shape()
        self._validate_summary_shape()
        if self.manifest_checksum and self.manifest_checksum != _json_sha256(self._checksum_payload()):
            raise FoldManifestIntegrityError("fold manifest checksum does not match its contents")

    def _validate_membership_shape(self) -> None:
        uids = [item.study_instance_uid for item in self.memberships]
        if len(set(uids)) != len(uids):
            raise FoldManifestIntegrityError("fold manifest contains duplicate study membership")
        if self.grouping.reliable:
            mapping = self.grouping.group_by_study
            if set(mapping) != set(uids):
                raise FoldManifestIntegrityError(
                    "reliable grouping evidence must cover exactly every assigned study"
                )
            for item in self.memberships:
                if item.group_key != mapping[item.study_instance_uid]:
                    raise FoldManifestIntegrityError(
                        f"fold membership group join disagrees for {item.study_instance_uid}"
                    )
            groups: dict[str, set[int]] = defaultdict(set)
            for item in self.memberships:
                groups[item.group_key or ""].add(item.fold)
            split_groups = [key for key, folds in groups.items() if len(folds) != 1]
            if split_groups:
                raise FoldManifestIntegrityError(
                    f"reliable groups are split across folds: {split_groups[:3]}"
                )
        elif any(item.group_key is not None for item in self.memberships):
            raise FoldManifestIntegrityError("study-level manifests must not persist untrusted group keys")

    def _validate_summary_shape(self) -> None:
        if tuple(item.fold for item in self.fold_summaries) != self.fold_ids:
            raise FoldManifestIntegrityError("fold summaries must contain each fold exactly once in order")
        if any(
            target_counts.target_counts_by_name.keys() != set(TARGET_COLUMNS)
            for target_counts in self.fold_summaries
        ):
            raise FoldManifestIntegrityError("fold summaries must contain every competition target")
        self._validate_summary_values()

    def _validate_summary_values(self) -> None:
        group_by_study = {
            item.study_instance_uid: (
                item.group_key or item.study_instance_uid
                if self.grouping.reliable
                else item.study_instance_uid
            )
            for item in self.memberships
        }
        total_positive = [0] * len(TARGET_COLUMNS)
        total_negative = [0] * len(TARGET_COLUMNS)
        total_positive_groups = [set[str]() for _ in TARGET_COLUMNS]
        total_negative_groups = [set[str]() for _ in TARGET_COLUMNS]
        for item in self.gold_memberships:
            group_key = group_by_study[item.study_instance_uid]
            for index, label in enumerate(item.reference_labels):
                if label == 1:
                    total_positive[index] += 1
                    total_positive_groups[index].add(group_key)
                elif label == 0:
                    total_negative[index] += 1
                    total_negative_groups[index].add(group_key)
        for index, target in enumerate(TARGET_COLUMNS):
            possible = (
                total_positive[index] >= FOLD_COUNT
                and total_negative[index] >= FOLD_COUNT
                and len(total_positive_groups[index]) >= FOLD_COUNT
                and len(total_negative_groups[index]) >= FOLD_COUNT
            )
            if possible and any(
                summary.target_counts_by_name[target].positive == 0
                or summary.target_counts_by_name[target].negative == 0
                for summary in self.fold_summaries
            ):
                raise FoldManifestIntegrityError(
                    f"manifest has feasible but missing class support for {target}"
                )
            reason = (
                "Both classes are present in this fold."
                if possible
                else (
                    "Both-class support is mathematically unavailable under unsplit groups: "
                    f"positive studies={total_positive[index]}, negative studies={total_negative[index]}, "
                    f"positive groups={len(total_positive_groups[index])}, "
                    f"negative groups={len(total_negative_groups[index])}."
                )
            )
            for summary in self.fold_summaries:
                actual_fold = tuple(
                    item for item in self.gold_memberships if item.fold == summary.fold
                )
                counts = summary.target_counts_by_name[target]
                expected = _target_fold_counts(
                    actual_fold,
                    target_index=index,
                    group_by_study=group_by_study,
                    support_possible=possible,
                    unavoidable_reason=reason,
                )
                if counts != expected:
                    raise FoldManifestIntegrityError(
                        f"fold summary counts do not match memberships for fold {summary.fold}, {target}"
                    )

    def _checksum_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_split": self.source_split,
            "source_version": self.source_version,
            "source_manifest_checksum": self.source_manifest_checksum,
            "source_fingerprint": self.source_fingerprint,
            "fold_count": self.fold_count,
            "fold_ids": list(self.fold_ids),
            "repeat_name": self.repeat_name,
            "repeat_justification": self.repeat_justification,
            "generation_method": self.generation_method,
            "seed_name": self.seed_name,
            "seed": self.seed,
            "grouping": self.grouping.to_private_dict(),
            "memberships": [_membership_to_dict(item) for item in self.memberships],
            "fold_summaries": [_summary_to_dict(item) for item in self.fold_summaries],
            "adaptation_exclusions": list(self.adaptation_exclusions),
            "input_inventory_reconciled": self.input_inventory_reconciled,
        }

    def to_private_dict(self) -> dict[str, object]:
        """Return the identifier-bearing manifest for a private path."""

        payload = self._checksum_payload()
        payload["ticket"] = "04"
        payload["manifest_checksum"] = self.manifest_checksum
        return payload

    def to_public_dict(self) -> dict[str, object]:
        """Return generation evidence without study, series, or group identifiers."""

        return {
            "schema_version": PUBLIC_FOLD_SCHEMA_VERSION,
            "ticket": "04",
            "source": {
                "split": self.source_split,
                "source_version": self.source_version,
                "source_manifest_checksum": self.source_manifest_checksum,
                "source_fingerprint": self.source_fingerprint,
                "input_inventory_reconciled": self.input_inventory_reconciled,
            },
            "generation": {
                "fold_count": self.fold_count,
                "fold_ids": list(self.fold_ids),
                "repeat_name": self.repeat_name,
                "repeat_justification": self.repeat_justification,
                "method": self.generation_method,
                "seed_name": self.seed_name,
                "seed": self.seed,
            },
            "grouping": self.grouping.to_public_dict(),
            "counts": {
                "eligible_training_studies": len(self.memberships),
                "gold_studies": len(self.gold_memberships),
                "report_only_studies": len(self.report_only_memberships),
            },
            "pooled_gold_target_counts": self.pooled_gold_target_counts,
            "folds": [_summary_to_dict(item) for item in self.fold_summaries],
            "adaptation_exclusion": {
                "unit": "study",
                "held_out_inputs": list(self.adaptation_exclusions),
                "training_partition": "all studies not assigned to the held-out fold",
            },
            "manifest_checksum": self.manifest_checksum,
        }

    @classmethod
    def from_private_dict(cls, payload: Mapping[str, object]) -> "FoldManifest":
        """Parse and verify a private JSON manifest."""

        memberships_value = payload.get("memberships")
        summaries_value = payload.get("fold_summaries")
        grouping_value = payload.get("grouping")
        exclusions_value = payload.get("adaptation_exclusions")
        if not isinstance(memberships_value, list):
            raise FoldManifestIntegrityError("private fold manifest memberships must be a list")
        if not isinstance(summaries_value, list):
            raise FoldManifestIntegrityError("private fold manifest summaries must be a list")
        if not isinstance(grouping_value, Mapping):
            raise FoldManifestIntegrityError("private fold manifest grouping must be an object")
        if not isinstance(exclusions_value, list):
            raise FoldManifestIntegrityError("private fold manifest adaptation exclusions must be a list")
        candidate = cls(
            schema_version=str(payload.get("schema_version", "")),
            source_split=str(payload.get("source_split", "")),
            source_version=str(payload.get("source_version", "")),
            source_manifest_checksum=str(payload.get("source_manifest_checksum", "")),
            source_fingerprint=_optional_string(payload.get("source_fingerprint")),
            fold_count=_required_int(payload.get("fold_count"), "fold_count"),
            fold_ids=_required_int_tuple(payload.get("fold_ids"), "fold_ids"),
            repeat_name=str(payload.get("repeat_name", "")),
            repeat_justification=_optional_string(payload.get("repeat_justification")),
            generation_method=str(payload.get("generation_method", "")),
            seed_name=str(payload.get("seed_name", "")),
            seed=_required_int(payload.get("seed"), "seed"),
            grouping=GroupingAssessment.from_dict(grouping_value),
            memberships=tuple(_membership_from_dict(item) for item in memberships_value),
            fold_summaries=tuple(_summary_from_dict(item) for item in summaries_value),
            adaptation_exclusions=tuple(str(item) for item in exclusions_value),
            manifest_checksum=str(payload.get("manifest_checksum", "")),
            input_inventory_reconciled=_optional_bool(payload.get("input_inventory_reconciled")),
        )
        if not candidate.manifest_checksum:
            raise FoldManifestIntegrityError("private fold manifest has no checksum")
        return candidate

    @property
    def study_to_fold(self) -> dict[str, int]:
        return {item.study_instance_uid: item.fold for item in self.memberships}

    @property
    def gold_memberships(self) -> tuple[FoldMembership, ...]:
        return tuple(item for item in self.memberships if item.cohort == "gold")

    @property
    def report_only_memberships(self) -> tuple[FoldMembership, ...]:
        return tuple(item for item in self.memberships if item.cohort == "report_only")

    @property
    def pooled_gold_target_counts(self) -> dict[str, dict[str, int | str]]:
        """Return pooled reference support, including the group counts."""

        counts: dict[str, dict[str, int | str]] = {}
        for index, target in enumerate(TARGET_COLUMNS):
            positives = tuple(
                item for item in self.gold_memberships if item.reference_labels[index] == 1
            )
            negatives = tuple(
                item for item in self.gold_memberships if item.reference_labels[index] == 0
            )
            positive_groups = {
                item.group_key or item.study_instance_uid for item in positives
            }
            negative_groups = {
                item.group_key or item.study_instance_uid for item in negatives
            }
            counts[target] = {
                "positive": len(positives),
                "negative": len(negatives),
                "positive_group_count": len(positive_groups),
                "negative_group_count": len(negative_groups),
                "support_status": (
                    "both_classes" if positives and negatives else "one_class_unavoidable"
                ),
            }
        return counts

    def held_out_study_uids(self, fold: int) -> tuple[str, ...]:
        _validate_fold_id(fold)
        return tuple(item.study_instance_uid for item in self.memberships if item.fold == fold)

    def training_study_uids(self, fold: int) -> tuple[str, ...]:
        _validate_fold_id(fold)
        return tuple(item.study_instance_uid for item in self.memberships if item.fold != fold)

    def assert_adaptation_input_is_training_only(
        self,
        fold: int,
        study_instance_uids: Sequence[str],
    ) -> None:
        """Reject an adaptation input set containing any held-out study."""

        known = set(self.study_to_fold)
        unknown = sorted(set(study_instance_uids) - known)
        if unknown:
            raise FoldFreezeError(
                f"fold {fold} adaptation input contains unknown studies: {unknown[:3]}"
            )
        held_out = set(self.held_out_study_uids(fold))
        overlap = sorted(held_out.intersection(study_instance_uids))
        if overlap:
            raise FoldFreezeError(
                f"fold {fold} adaptation input contains held-out studies: {overlap[:3]}"
            )


@dataclass(frozen=True)
class InventorySource:
    """Training source metadata recovered from ticket-03's private report."""

    source_manifest: SourceManifest
    source_version: str
    source_fingerprint: str | None
    full_corpus_reconciled: bool


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_nonnegative_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FoldFreezeError("grouping count fields must be non-negative integers or null")
    return value


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise FoldFreezeError("grouping boolean fields must be boolean or null")
    return value


def _validate_fold_id(fold: int) -> None:
    if fold not in FOLD_IDS:
        raise FoldFreezeError(f"fold must be one of {FOLD_IDS}")


def _membership_to_dict(item: FoldMembership) -> dict[str, object]:
    return {
        "StudyInstanceUID": item.study_instance_uid,
        "cohort": item.cohort,
        "fold": item.fold,
        "reference_labels": list(item.reference_labels),
        "group_key": item.group_key,
    }


def _target_count_to_dict(item: TargetFoldCounts) -> dict[str, object]:
    return {
        "positive": item.positive,
        "negative": item.negative,
        "positive_group_count": item.positive_group_count,
        "negative_group_count": item.negative_group_count,
        "support_status": item.support_status,
        "support_reason": item.support_reason,
    }


def _summary_to_dict(item: FoldSummary) -> dict[str, object]:
    return {
        "fold": item.fold,
        "study_count": item.study_count,
        "gold_study_count": item.gold_study_count,
        "report_only_study_count": item.report_only_study_count,
        "gold_target_counts": {
            target: _target_count_to_dict(counts) for target, counts in item.target_counts
        },
    }


def _target_fold_counts(
    gold_memberships: Sequence[FoldMembership],
    *,
    target_index: int,
    group_by_study: Mapping[str, str],
    support_possible: bool,
    unavoidable_reason: str,
) -> TargetFoldCounts:
    """Build one independent support record for generation and validation."""

    positives = tuple(item for item in gold_memberships if item.reference_labels[target_index] == 1)
    negatives = tuple(item for item in gold_memberships if item.reference_labels[target_index] == 0)
    has_both_classes = bool(positives and negatives)
    return TargetFoldCounts(
        positive=len(positives),
        negative=len(negatives),
        positive_group_count=len(
            {group_by_study[item.study_instance_uid] for item in positives}
        ),
        negative_group_count=len(
            {group_by_study[item.study_instance_uid] for item in negatives}
        ),
        support_status=(
            "both_classes" if has_both_classes else "one_class_unavoidable"
        ),
        support_reason=(
            "Both classes are present in this fold."
            if has_both_classes
            else unavoidable_reason
        ),
    )


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FoldManifestIntegrityError(f"private fold manifest {field} must be an integer")
    return value


def _required_int_tuple(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise FoldManifestIntegrityError(f"private fold manifest {field} must be a list")
    return tuple(_required_int(item, field) for item in value)


def _reference_labels(value: object, field: str) -> tuple[int | None, ...]:
    if not isinstance(value, list):
        raise FoldManifestIntegrityError(f"private fold manifest {field} must be a list")
    labels: list[int | None] = []
    for item in value:
        if item is None:
            labels.append(None)
        elif isinstance(item, bool) or not isinstance(item, int) or item not in {0, 1}:
            raise FoldManifestIntegrityError(f"private fold manifest {field} contains an invalid label")
        else:
            labels.append(item)
    return tuple(labels)


def _membership_from_dict(value: object) -> FoldMembership:
    if not isinstance(value, Mapping):
        raise FoldManifestIntegrityError("private fold manifest membership must be an object")
    return FoldMembership(
        study_instance_uid=str(value.get("StudyInstanceUID", "")),
        cohort=str(value.get("cohort", "")),
        fold=_required_int(value.get("fold"), "membership.fold"),
        reference_labels=_reference_labels(value.get("reference_labels"), "membership.reference_labels"),
        group_key=_optional_string(value.get("group_key")),
    )


def _target_count_from_dict(value: object) -> TargetFoldCounts:
    if not isinstance(value, Mapping):
        raise FoldManifestIntegrityError("private fold manifest target count must be an object")
    return TargetFoldCounts(
        positive=_required_int(value.get("positive"), "target_count.positive"),
        negative=_required_int(value.get("negative"), "target_count.negative"),
        positive_group_count=_required_int(
            value.get("positive_group_count"), "target_count.positive_group_count"
        ),
        negative_group_count=_required_int(
            value.get("negative_group_count"), "target_count.negative_group_count"
        ),
        support_status=str(value.get("support_status", "")),
        support_reason=str(value.get("support_reason", "")),
    )


def _summary_from_dict(value: object) -> FoldSummary:
    if not isinstance(value, Mapping):
        raise FoldManifestIntegrityError("private fold manifest summary must be an object")
    target_counts_value = value.get("gold_target_counts")
    if not isinstance(target_counts_value, Mapping):
        raise FoldManifestIntegrityError("private fold manifest gold_target_counts must be an object")
    return FoldSummary(
        fold=_required_int(value.get("fold"), "fold_summary.fold"),
        study_count=_required_int(value.get("study_count"), "fold_summary.study_count"),
        gold_study_count=_required_int(
            value.get("gold_study_count"), "fold_summary.gold_study_count"
        ),
        report_only_study_count=_required_int(
            value.get("report_only_study_count"), "fold_summary.report_only_study_count"
        ),
        target_counts=tuple(
            (target, _target_count_from_dict(target_counts_value.get(target)))
            for target in TARGET_COLUMNS
        ),
    )


def _write_immutable_json(path: str | Path, payload: Mapping[str, object], *, mode: int) -> Path:
    """Publish complete JSON exactly once, refusing to overwrite an artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.partial-",
            dir=destination.parent,
        )
        temporary_path = Path(temporary_name)
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, destination)
        except FileExistsError as exc:
            raise ImmutableArtifactExistsError(f"immutable artifact already exists: {destination}") from exc
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return destination


def write_fold_reports(
    manifest: FoldManifest,
    *,
    private_path: str | Path,
    public_path: str | Path,
) -> tuple[Path, Path]:
    """Publish the private manifest and identifier-free evidence once."""

    private_destination = Path(private_path)
    public_destination = Path(public_path)
    if private_destination.resolve() == public_destination.resolve():
        raise FoldFreezeError("private and public fold reports must use different paths")
    for destination in (private_destination, public_destination):
        if os.path.lexists(destination):
            raise ImmutableArtifactExistsError(f"immutable artifact already exists: {destination}")
    private = _write_immutable_json(private_destination, manifest.to_private_dict(), mode=0o600)
    public = _write_immutable_json(public_destination, manifest.to_public_dict(), mode=0o644)
    return private, public


def load_fold_manifest(path: str | Path) -> FoldManifest:
    """Load a private manifest and verify its checksum and support counts."""

    try:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FoldManifestIntegrityError(f"could not read fold manifest {path}") from exc
    if not isinstance(payload, Mapping):
        raise FoldManifestIntegrityError("private fold manifest root must be an object")
    return FoldManifest.from_private_dict(cast(Mapping[str, object], payload))


def verify_fold_manifest(
    path: str | Path,
    *,
    source_manifest: SourceManifest | None = None,
) -> FoldManifest:
    """Load a manifest and optionally verify its source membership joins."""

    manifest = load_fold_manifest(path)
    if source_manifest is not None:
        validate_fold_manifest_against_source(manifest, source_manifest)
    return manifest


def _read_json_object(path: str | Path, *, description: str) -> Mapping[str, object]:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FoldFreezeError(f"could not read {description}: {path}") from exc
    if not isinstance(payload, Mapping):
        raise FoldFreezeError(f"{description} must contain a JSON object")
    return cast(Mapping[str, object], payload)


def _inventory_bool(value: object, field: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise FoldFreezeError(f"inventory {field} must be true, false, or null")
    return value


def load_inventory_source(path: str | Path) -> InventorySource:
    """Recover source studies and labels from a private ticket-03 report."""

    payload = _read_json_object(path, description="private inventory report")
    source_value = payload.get("source")
    if not isinstance(source_value, Mapping):
        raise FoldFreezeError("private inventory report has no source object")
    studies_value = source_value.get("studies")
    series_value = source_value.get("series")
    if not isinstance(studies_value, list) or not isinstance(series_value, list):
        raise FoldFreezeError("private inventory source must contain studies and series lists")
    studies: list[SourceStudy] = []
    for value in studies_value:
        if not isinstance(value, Mapping):
            raise FoldFreezeError("private inventory study entry must be an object")
        labels = _reference_labels(value.get("reference_labels"), "inventory.reference_labels")
        report_present = value.get("report_present")
        if not isinstance(report_present, bool):
            raise FoldFreezeError("private inventory report_present must be boolean")
        studies.append(
            SourceStudy(
                study_instance_uid=str(value.get("StudyInstanceUID", "")),
                report_present=report_present,
                reference_labels=labels,
            )
        )
    series = []
    for value in series_value:
        if not isinstance(value, Mapping):
            raise FoldFreezeError("private inventory series entry must be an object")
        series.append(
            SourceSeries(
                study_instance_uid=str(value.get("StudyInstanceUID", "")),
                series_instance_uid=str(value.get("SeriesInstanceUID", "")),
                fluid_sensitive=_inventory_bool(value.get("Fluid_Sensitive"), "Fluid_Sensitive"),
                fat_suppression=_inventory_bool(value.get("Fat_Suppression"), "Fat_Suppression"),
                anatomical_plane=_optional_string(value.get("Anatomical_Plane")),
            )
        )
    source_split = str(payload.get("split", ""))
    source_version = str(payload.get("source_version", "")).strip()
    if not source_version:
        raise FoldFreezeError("private inventory report has no source version")
    archive_value = payload.get("archive")
    if archive_value is not None and not isinstance(archive_value, Mapping):
        raise FoldFreezeError("private inventory archive must be an object")
    source_fingerprint = (
        _optional_string(archive_value.get("fingerprint"))
        if isinstance(archive_value, Mapping)
        else None
    )
    reconciliation_value = payload.get("reconciliation")
    full_corpus_reconciled = False
    if reconciliation_value is not None:
        if not isinstance(reconciliation_value, Mapping):
            raise FoldFreezeError("private inventory reconciliation must be an object")
        reconciled_value = reconciliation_value.get("full_corpus_reconciled", False)
        if not isinstance(reconciled_value, bool):
            raise FoldFreezeError("private inventory full_corpus_reconciled must be boolean")
        full_corpus_reconciled = reconciled_value
    return InventorySource(
        source_manifest=SourceManifest(source_split, tuple(studies), tuple(series)),
        source_version=source_version,
        source_fingerprint=source_fingerprint,
        full_corpus_reconciled=full_corpus_reconciled,
    )


def load_grouping_assessment(path: str | Path) -> GroupingAssessment:
    """Load an explicit private grouping audit without inferring any key."""

    return GroupingAssessment.from_dict(
        _read_json_object(path, description="grouping evidence")
    )


def run_fold_freeze(
    inventory_path: str | Path,
    output_dir: str | Path,
    *,
    grouping_path: str | Path | None = None,
    seed_name: str = "ticket-04-fold-freeze-v1",
    seed: int = 20260911,
    repeat_name: str = "repeat-1",
    repeat_justification: str | None = None,
) -> dict[str, object]:
    """Freeze folds from ticket-03's private inventory and publish both reports."""

    source = load_inventory_source(inventory_path)
    if not source.full_corpus_reconciled:
        raise FoldFreezeError(
            "refusing to publish canonical folds from an unreconciled ticket-03 inventory"
        )
    if grouping_path is None:
        raise FoldFreezeError(
            "explicit grouping evidence is required; record reliable-key or study-only V02 evidence"
        )
    grouping = load_grouping_assessment(grouping_path)
    manifest = freeze_folds(
        source.source_manifest,
        source_version=source.source_version,
        source_fingerprint=source.source_fingerprint,
        input_inventory_reconciled=source.full_corpus_reconciled,
        grouping=grouping,
        seed_name=seed_name,
        seed=seed,
        repeat_name=repeat_name,
        repeat_justification=repeat_justification,
    )
    validate_fold_manifest_against_source(manifest, source.source_manifest)
    destination = Path(output_dir)
    write_fold_reports(
        manifest,
        private_path=destination / "private" / "fold-manifest.json",
        public_path=destination / "public" / "fold-evidence.json",
    )
    public = manifest.to_public_dict()
    public["input_inventory"] = {
        "source_studies": source.source_manifest.study_count,
        "source_gold_studies": len(source.source_manifest.gold_study_uids),
        "source_report_only_studies": len(source.source_manifest.report_only_study_uids),
        "source_manifest_checksum": source_manifest_checksum(source.source_manifest),
        "full_corpus_reconciled": source.full_corpus_reconciled,
    }
    return public


def _check_execution_context(*, require_slurm: bool, allow_local_smoke: bool) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if require_slurm and (not job_id or partition != "cluster1-cpu"):
        raise RuntimeError("ticket-04 fold freeze must run in a cluster1-cpu Slurm allocation")
    if not job_id and not allow_local_smoke:
        raise RuntimeError(
            "refusing to freeze competition folds outside Slurm; "
            "use --allow-local-smoke only with a tiny fixture inventory"
        )


def _fold_parser() -> object:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grouping-evidence", type=Path, default=None)
    parser.add_argument("--seed-name", default="ticket-04-fold-freeze-v1")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--repeat-name", default="repeat-1")
    parser.add_argument("--repeat-justification", default=None)
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("--allow-local-smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = cast(Any, _fold_parser()).parse_args(argv)
    try:
        _check_execution_context(
            require_slurm=args.require_slurm,
            allow_local_smoke=args.allow_local_smoke,
        )
        summary = run_fold_freeze(
            args.inventory,
            args.output_dir,
            grouping_path=args.grouping_evidence,
            seed_name=args.seed_name,
            seed=args.seed,
            repeat_name=args.repeat_name,
            repeat_justification=args.repeat_justification,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ticket-04 fold freeze failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


def source_manifest_checksum(source_manifest: SourceManifest) -> str:
    """Hash the source membership, descriptors, and labels used to build folds."""

    payload = {
        "split": source_manifest.split,
        "studies": [
            {
                "StudyInstanceUID": item.study_instance_uid,
                "report_present": item.report_present,
                "reference_labels": list(item.reference_labels),
            }
            for item in sorted(source_manifest.studies, key=lambda value: value.study_instance_uid)
        ],
        "series": [
            {
                "StudyInstanceUID": item.study_instance_uid,
                "SeriesInstanceUID": item.series_instance_uid,
                "Fluid_Sensitive": item.fluid_sensitive,
                "Fat_Suppression": item.fat_suppression,
                "Anatomical_Plane": item.anatomical_plane,
            }
            for item in sorted(
                source_manifest.series,
                key=lambda value: (value.study_instance_uid, value.series_instance_uid),
            )
        ],
    }
    return _json_sha256(payload)


@dataclass(frozen=True)
class _StudyGroup:
    key: str
    studies: tuple[SourceStudy, ...]

    @property
    def gold_studies(self) -> tuple[SourceStudy, ...]:
        return tuple(item for item in self.studies if item.has_reference_labels)

    @property
    def has_gold(self) -> bool:
        return bool(self.gold_studies)

    def positive_count(self, target_index: int) -> int:
        return sum(
            int(item.reference_labels[target_index] == 1)
            for item in self.gold_studies
        )

    def negative_count(self, target_index: int) -> int:
        return sum(
            int(item.reference_labels[target_index] == 0)
            for item in self.gold_studies
        )


def _stable_digest(config: FoldGenerationConfig, value: str) -> str:
    material = (
        f"{config.seed_name}|{config.seed}|{config.repeat_name}|{value}"
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _build_groups(
    source_manifest: SourceManifest,
    grouping: GroupingAssessment,
) -> tuple[_StudyGroup, ...]:
    if grouping.status == "untrustworthy":
        raise FoldFreezeError(
            "fold freeze is blocked by contradictory or untrustworthy grouping evidence"
        )
    studies = tuple(sorted(source_manifest.studies, key=lambda item: item.study_instance_uid))
    if source_manifest.partially_labeled_study_uids:
        raise FoldFreezeError(
            "partially reference-labeled studies must be resolved before folds are frozen"
        )
    if grouping.reliable:
        group_by_study = grouping.group_by_study
        study_uids = {item.study_instance_uid for item in studies}
        if set(group_by_study) != study_uids:
            missing = sorted(study_uids - set(group_by_study))
            extra = sorted(set(group_by_study) - study_uids)
            raise FoldFreezeError(
                "reliable grouping evidence must cover every training study exactly; "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        groups: dict[str, list[SourceStudy]] = defaultdict(list)
        for study in studies:
            groups[group_by_study[study.study_instance_uid]].append(study)
    else:
        groups = {study.study_instance_uid: [study] for study in studies}
    return tuple(
        _StudyGroup(key=key, studies=tuple(value))
        for key, value in sorted(groups.items())
    )


def _support_feasibility(groups: Sequence[_StudyGroup]) -> tuple[tuple[bool, str], ...]:
    results: list[tuple[bool, str]] = []
    gold_groups = tuple(group for group in groups if group.has_gold)
    for target_index, target in enumerate(TARGET_COLUMNS):
        positives = sum(group.positive_count(target_index) for group in gold_groups)
        negatives = sum(group.negative_count(target_index) for group in gold_groups)
        positive_groups = sum(group.positive_count(target_index) > 0 for group in gold_groups)
        negative_groups = sum(group.negative_count(target_index) > 0 for group in gold_groups)
        possible = (
            positives >= FOLD_COUNT
            and negatives >= FOLD_COUNT
            and positive_groups >= FOLD_COUNT
            and negative_groups >= FOLD_COUNT
        )
        if possible:
            reason = "At least three positive and three negative gold studies/groups are available."
        else:
            reason = (
                f"Both-class support is mathematically unavailable under unsplit groups: "
                f"positive studies={positives}, negative studies={negatives}, "
                f"positive groups={positive_groups}, negative groups={negative_groups}."
            )
        results.append((possible, reason))
    return tuple(results)


def _find_support_assignments(
    groups: Sequence[_StudyGroup],
    feasible: Sequence[tuple[bool, str]],
) -> dict[str, int] | None:
    """Find a deterministic whole-group assignment satisfying support coverage."""

    requirements = tuple(
        (target_index, required_positive)
        for target_index, (possible, _) in enumerate(feasible)
        if possible
        for required_positive in (True, False)
    )
    if not requirements:
        return {}
    supporting_groups = {
        requirement: tuple(
            group
            for group in groups
            if (
                group.positive_count(requirement[0]) > 0
                if requirement[1]
                else group.negative_count(requirement[0]) > 0
            )
        )
        for requirement in requirements
    }
    assignments: dict[str, int] = {}

    def is_covered(requirement: tuple[int, bool], fold: int) -> bool:
        return any(
            assignments.get(group.key) == fold
            for group in supporting_groups[requirement]
        )

    def search() -> dict[str, int] | None:
        unmet = [
            (requirement, fold)
            for requirement in requirements
            for fold in FOLD_IDS
            if not is_covered(requirement, fold)
        ]
        if not unmet:
            return dict(assignments)
        options: list[tuple[int, int, tuple[int, bool], int]] = []
        for requirement, fold in unmet:
            candidate_count = sum(
                assignments.get(group.key) in (None, fold)
                for group in supporting_groups[requirement]
            )
            options.append((candidate_count, len(supporting_groups[requirement]), requirement, fold))
        _, _, requirement, destination = min(
            options,
            key=lambda item: (item[0], item[1], item[2][0], item[2][1], item[3]),
        )
        candidate_groups = [
            group
            for group in supporting_groups[requirement]
            if assignments.get(group.key) in (None, destination)
        ]
        unmet_before = set(unmet)
        candidate_groups.sort(
            key=lambda group: (
                -sum(
                    int(
                        (other in unmet_before)
                        and (
                            group.positive_count(other[0]) > 0
                            if other[1]
                            else group.negative_count(other[0]) > 0
                        )
                    )
                    for other in requirements
                ),
                group.key,
            )
        )
        for group in candidate_groups:
            previous = assignments.get(group.key)
            if previous is None:
                assignments[group.key] = destination
            result = search()
            if result is not None:
                return result
            if previous is None:
                del assignments[group.key]
        return None

    return search()


def _allocate_gold_groups(
    groups: Sequence[_StudyGroup],
    *,
    config: FoldGenerationConfig,
) -> dict[str, int]:
    gold_groups = tuple(group for group in groups if group.has_gold)
    if not gold_groups:
        return {}
    feasible = _support_feasibility(groups)
    assignments = _find_support_assignments(gold_groups, feasible)
    if assignments is None:
        raise FoldFreezeError(
            "grouping constraints prevent both-class support for a feasible target; "
            "revise the fold design before freezing it"
        )
    total_positive = [sum(group.positive_count(index) for group in gold_groups) for index in range(len(TARGET_COLUMNS))]
    total_negative = [sum(group.negative_count(index) for group in gold_groups) for index in range(len(TARGET_COLUMNS))]
    positive_group_totals = [
        sum(group.positive_count(index) > 0 for group in gold_groups)
        for index in range(len(TARGET_COLUMNS))
    ]
    negative_group_totals = [
        sum(group.negative_count(index) > 0 for group in gold_groups)
        for index in range(len(TARGET_COLUMNS))
    ]
    ordered_groups = sorted(
        gold_groups,
        key=lambda group: (
            -sum(
                (1.0 / positive_group_totals[index])
                for index in range(len(TARGET_COLUMNS))
                if group.positive_count(index) > 0
            )
            -sum(
                (1.0 / negative_group_totals[index])
                for index in range(len(TARGET_COLUMNS))
                if group.negative_count(index) > 0
            ),
            -sum(
                int(group.positive_count(index) > 0) + int(group.negative_count(index) > 0)
                for index in range(len(TARGET_COLUMNS))
            ),
            -len(group.gold_studies),
            _stable_digest(config, group.key),
            group.key,
        ),
    )
    fold_gold_counts = [0] * FOLD_COUNT
    fold_positive = [[0] * len(TARGET_COLUMNS) for _ in FOLD_IDS]
    fold_negative = [[0] * len(TARGET_COLUMNS) for _ in FOLD_IDS]
    assigned_groups = {group.key for group in gold_groups if group.key in assignments}
    for group in gold_groups:
        if group.key not in assigned_groups:
            continue
        fold = assignments[group.key]
        fold_gold_counts[fold] += len(group.gold_studies)
        for index in range(len(TARGET_COLUMNS)):
            fold_positive[fold][index] += group.positive_count(index)
            fold_negative[fold][index] += group.negative_count(index)
    target_positive = [value / FOLD_COUNT for value in total_positive]
    target_negative = [value / FOLD_COUNT for value in total_negative]
    target_gold = len(tuple(item for group in gold_groups for item in group.gold_studies)) / FOLD_COUNT

    for group in ordered_groups:
        if group.key in assigned_groups:
            continue
        candidate_scores: list[tuple[float, int]] = []
        for fold in FOLD_IDS:
            support_gain = 0
            balance = 0.0
            for index in range(len(TARGET_COLUMNS)):
                new_positive = fold_positive[fold][index] + group.positive_count(index)
                new_negative = fold_negative[fold][index] + group.negative_count(index)
                if feasible[index][0]:
                    support_gain += int(fold_positive[fold][index] == 0 and group.positive_count(index) > 0)
                    support_gain += int(fold_negative[fold][index] == 0 and group.negative_count(index) > 0)
                balance += ((new_positive - target_positive[index]) / max(1.0, target_positive[index])) ** 2
                balance += ((new_negative - target_negative[index]) / max(1.0, target_negative[index])) ** 2
            new_gold_count = fold_gold_counts[fold] + len(group.gold_studies)
            balance += 0.05 * ((new_gold_count - target_gold) / max(1.0, target_gold)) ** 2
            candidate_scores.append((balance - 100.0 * support_gain, fold))
        _, selected_fold = min(candidate_scores, key=lambda value: (value[0], value[1]))
        assignments[group.key] = selected_fold
        fold_gold_counts[selected_fold] += len(group.gold_studies)
        for index in range(len(TARGET_COLUMNS)):
            fold_positive[selected_fold][index] += group.positive_count(index)
            fold_negative[selected_fold][index] += group.negative_count(index)
    return assignments


def _summarize_memberships(
    memberships: Sequence[FoldMembership],
    groups: Sequence[_StudyGroup],
    feasibility: Sequence[tuple[bool, str]],
) -> tuple[FoldSummary, ...]:
    group_by_study = {
        study.study_instance_uid: group.key
        for group in groups
        for study in group.studies
    }
    summaries: list[FoldSummary] = []
    for fold in FOLD_IDS:
        fold_memberships = tuple(item for item in memberships if item.fold == fold)
        gold = tuple(item for item in fold_memberships if item.cohort == "gold")
        target_counts: list[tuple[str, TargetFoldCounts]] = []
        for index, target in enumerate(TARGET_COLUMNS):
            possible, reason = feasibility[index]
            target_counts.append(
                (
                    target,
                    _target_fold_counts(
                        gold,
                        target_index=index,
                        group_by_study=group_by_study,
                        support_possible=possible,
                        unavoidable_reason=reason,
                    ),
                )
            )
        summaries.append(
            FoldSummary(
                fold=fold,
                study_count=len(fold_memberships),
                gold_study_count=len(gold),
                report_only_study_count=len(fold_memberships) - len(gold),
                target_counts=tuple(target_counts),
            )
        )
    return tuple(summaries)


def freeze_folds(
    source_manifest: SourceManifest,
    *,
    source_version: str,
    source_fingerprint: str | None = None,
    input_inventory_reconciled: bool | None = None,
    grouping: GroupingAssessment | None = None,
    seed_name: str = "ticket-04-fold-freeze-v1",
    seed: int = 20260911,
    repeat_name: str = "repeat-1",
    repeat_justification: str | None = None,
) -> FoldManifest:
    """Generate one deterministic three-fold manifest for every train study."""

    if source_manifest.split != "train":
        raise FoldFreezeError("ticket 04 only freezes folds for the train split")
    if not source_version.strip():
        raise FoldFreezeError("source_version must be non-empty")
    seed_name = seed_name.strip()
    repeat_name = repeat_name.strip()
    config = FoldGenerationConfig(
        seed_name=seed_name,
        seed=seed,
        repeat_name=repeat_name,
        repeat_justification=repeat_justification.strip() if repeat_justification else None,
    )
    if source_fingerprint is not None and not source_fingerprint.strip():
        raise FoldFreezeError("source_fingerprint must be non-empty when present")
    if input_inventory_reconciled is not None and not isinstance(input_inventory_reconciled, bool):
        raise FoldFreezeError("input_inventory_reconciled must be boolean or null")
    if grouping is None:
        grouping = GroupingAssessment.no_reliable_key()
    groups = _build_groups(source_manifest, grouping)
    feasibility = _support_feasibility(groups)
    group_assignments = _allocate_gold_groups(
        groups,
        config=config,
    )
    for group in groups:
        if group.key not in group_assignments:
            group_assignments[group.key] = int(
                _stable_digest(config, group.key),
                16,
            ) % FOLD_COUNT
    memberships = tuple(
        FoldMembership(
            study_instance_uid=study.study_instance_uid,
            cohort="gold" if study.has_reference_labels else "report_only",
            fold=group_assignments[group.key],
            reference_labels=study.reference_labels,
            group_key=group.key if grouping.reliable else None,
        )
        for group in groups
        for study in group.studies
    )
    memberships = tuple(sorted(memberships, key=lambda item: item.study_instance_uid))
    candidate = FoldManifest(
        schema_version=FOLD_SCHEMA_VERSION,
        source_split=source_manifest.split,
        source_version=source_version.strip(),
        source_manifest_checksum=source_manifest_checksum(source_manifest),
        source_fingerprint=source_fingerprint.strip() if source_fingerprint is not None else None,
        fold_count=FOLD_COUNT,
        fold_ids=FOLD_IDS,
        repeat_name=config.repeat_name,
        repeat_justification=config.repeat_justification,
        generation_method=FOLD_GENERATION_METHOD,
        seed_name=config.seed_name,
        seed=config.seed,
        grouping=grouping,
        memberships=memberships,
        fold_summaries=_summarize_memberships(memberships, groups, feasibility),
        adaptation_exclusions=ADAPTATION_EXCLUSIONS,
        manifest_checksum="",
        input_inventory_reconciled=input_inventory_reconciled,
    )
    return replace(candidate, manifest_checksum=_json_sha256(candidate._checksum_payload()))


def validate_fold_manifest_against_source(
    manifest: FoldManifest,
    source_manifest: SourceManifest,
) -> None:
    """Verify exact membership, cohort, and reference-label joins."""

    if source_manifest.split != "train":
        raise FoldFreezeError("fold validation requires a train source manifest")
    if source_manifest_checksum(source_manifest) != manifest.source_manifest_checksum:
        raise FoldManifestIntegrityError("fold manifest source checksum does not match the source manifest")
    if source_manifest.partially_labeled_study_uids:
        raise FoldFreezeError(
            "partially reference-labeled studies cannot be joined to a frozen fold manifest"
        )
    expected_by_uid = source_manifest.studies_by_uid
    actual_by_uid = {item.study_instance_uid: item for item in manifest.memberships}
    if set(actual_by_uid) != set(expected_by_uid):
        raise FoldManifestIntegrityError(
            "fold membership does not contain exactly one record for every source study"
        )
    for study_uid, source_study in expected_by_uid.items():
        membership = actual_by_uid[study_uid]
        expected_cohort = "gold" if source_study.has_reference_labels else "report_only"
        if membership.cohort != expected_cohort:
            raise FoldManifestIntegrityError(
                f"fold cohort join disagrees for {study_uid}: "
                f"expected {expected_cohort}, got {membership.cohort}"
            )
        if membership.reference_labels != source_study.reference_labels:
            raise FoldManifestIntegrityError(f"fold label join disagrees for {study_uid}")


def assert_comparable_fold_manifests(
    left: FoldManifest,
    right: FoldManifest,
) -> None:
    """Require exact frozen-manifest identity before comparing fold results."""

    if left.manifest_checksum != right.manifest_checksum:
        raise FoldFreezeError(
            "fold manifests differ; invalidate affected comparisons and use one frozen manifest"
        )


__all__ = [
    "ADAPTATION_EXCLUSIONS",
    "FOLD_COUNT",
    "FOLD_GENERATION_METHOD",
    "FOLD_IDS",
    "FOLD_SCHEMA_VERSION",
    "PUBLIC_FOLD_SCHEMA_VERSION",
    "FoldFreezeError",
    "FoldManifestIntegrityError",
    "ImmutableArtifactExistsError",
    "FoldGenerationConfig",
    "GroupingAssessment",
    "FoldMembership",
    "TargetFoldCounts",
    "FoldSummary",
    "FoldManifest",
    "InventorySource",
    "source_manifest_checksum",
    "freeze_folds",
    "write_fold_reports",
    "load_fold_manifest",
    "verify_fold_manifest",
    "load_inventory_source",
    "load_grouping_assessment",
    "run_fold_freeze",
    "validate_fold_manifest_against_source",
    "assert_comparable_fold_manifests",
    "main",
]
