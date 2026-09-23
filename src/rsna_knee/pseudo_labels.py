"""Fold-scoped provenance and admission for MRI teacher pseudo-labels."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence, cast

from .folds import FOLD_IDS, FoldManifest
from .targets import NUM_TARGETS, TARGET_COLUMNS


PSEUDO_LABEL_SCHEMA_VERSION = "rsna-knee-fold-pseudo-labels-v1"
PSEUDO_LABEL_VERSION = "mri-teacher-hard-labels-confidence-0.10-0.90-v1"
_SHA256_SIZE = 64
_GIT_REVISION_SIZE = 40


class PseudoLabelAdmissionError(ValueError):
    """Raised when teacher outputs cannot be proven clean for a consuming fold."""


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


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_SIZE
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PseudoLabelAdmissionError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _study_ids(values: Sequence[str], field: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise PseudoLabelAdmissionError(f"{field} must contain non-empty study identities")
    if len(set(result)) != len(result):
        raise PseudoLabelAdmissionError(f"{field} must not contain duplicate study identities")
    return tuple(sorted(result))


@dataclass(frozen=True)
class PseudoLabelPolicy:
    """Predeclared hard-label rule; uncertain scores remain unlabelled."""

    version: str = PSEUDO_LABEL_VERSION
    negative_threshold: float = 0.10
    positive_threshold: float = 0.90
    input_modality: str = "MRI-only"
    reports_used: bool = False
    confidence_weighting: bool = False

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise PseudoLabelAdmissionError("pseudo-label policy requires a version")
        if not (
            math.isfinite(self.negative_threshold)
            and math.isfinite(self.positive_threshold)
            and 0.0 <= self.negative_threshold < 0.5 < self.positive_threshold <= 1.0
        ):
            raise PseudoLabelAdmissionError("pseudo-label confidence thresholds are invalid")
        if self.input_modality != "MRI-only" or self.reports_used or self.confidence_weighting:
            raise PseudoLabelAdmissionError(
                "ticket 21 policy must use MRI-only, unweighted hard labels"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "negative_threshold": self.negative_threshold,
            "positive_threshold": self.positive_threshold,
            "negative_rule": "score <= negative_threshold becomes 0",
            "positive_rule": "score >= positive_threshold becomes 1",
            "abstain_rule": (
                "scores strictly between thresholds remain unknown and have zero loss mask"
            ),
            "input_modality": self.input_modality,
            "reports_used": self.reports_used,
            "confidence_weighting": self.confidence_weighting,
            "thresholds_are_not_a_calibration_claim": True,
            "target_columns": list(TARGET_COLUMNS),
            "reference_precedence": True,
            "unknown_uncertain_failed_masked": True,
        }


PSEUDO_LABEL_POLICY = PseudoLabelPolicy()


@dataclass(frozen=True)
class PseudoLabelTeacherProvenance:
    """Identity and fold membership of the trained MRI teacher."""

    fold: int
    fold_manifest_checksum: str
    repeat_name: str
    training_study_uids: tuple[str, ...]
    held_out_study_uids: tuple[str, ...]
    source_fingerprint: str
    code_revision: str
    code_sha256: str
    configuration_fingerprint: str
    checkpoint_sha256: str
    upstream_adaptation_study_uids: tuple[str, ...] = ()
    upstream_adaptation_artifact_sha256s: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.fold not in FOLD_IDS:
            raise PseudoLabelAdmissionError(f"teacher fold must be one of {FOLD_IDS}")
        _require_sha256(self.fold_manifest_checksum, "fold_manifest_checksum")
        _require_sha256(self.source_fingerprint, "source_fingerprint")
        _require_sha256(self.code_sha256, "teacher code hash")
        _require_sha256(self.configuration_fingerprint, "teacher configuration fingerprint")
        _require_sha256(self.checkpoint_sha256, "teacher checkpoint hash")
        if (
            not isinstance(self.code_revision, str)
            or len(self.code_revision) != _GIT_REVISION_SIZE
            or any(character not in "0123456789abcdef" for character in self.code_revision)
        ):
            raise PseudoLabelAdmissionError("teacher code revision must be a full lowercase Git SHA")
        if not self.repeat_name.strip():
            raise PseudoLabelAdmissionError("teacher provenance requires a fold repeat")
        for field, values in (
            ("teacher training studies", self.training_study_uids),
            ("teacher held-out studies", self.held_out_study_uids),
            ("teacher upstream adaptation studies", self.upstream_adaptation_study_uids),
        ):
            if tuple(sorted(values)) != values or len(set(values)) != len(values):
                raise PseudoLabelAdmissionError(f"{field} must be unique and sorted")
            if any(not value.strip() for value in values):
                raise PseudoLabelAdmissionError(f"{field} must contain non-empty identities")
        if set(self.training_study_uids).intersection(self.held_out_study_uids):
            raise PseudoLabelAdmissionError("teacher training includes a held-out study")
        if set(self.upstream_adaptation_study_uids).intersection(self.held_out_study_uids):
            raise PseudoLabelAdmissionError("teacher adaptation includes a held-out study")
        if not set(self.upstream_adaptation_study_uids).issubset(self.training_study_uids):
            raise PseudoLabelAdmissionError(
                "teacher adaptation studies are outside its declared training cohort"
            )
        artifact_hashes = tuple(self.upstream_adaptation_artifact_sha256s)
        if tuple(sorted(artifact_hashes)) != artifact_hashes:
            raise PseudoLabelAdmissionError("teacher adaptation artifact hashes must be sorted")
        for digest in artifact_hashes:
            _require_sha256(digest, "teacher adaptation artifact hash")

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "fold_manifest_checksum": self.fold_manifest_checksum,
            "repeat_name": self.repeat_name,
            "training_study_uids": list(self.training_study_uids),
            "training_study_set_sha256": _digest(list(self.training_study_uids)),
            "held_out_study_uids": list(self.held_out_study_uids),
            "held_out_study_set_sha256": _digest(list(self.held_out_study_uids)),
            "source_fingerprint": self.source_fingerprint,
            "code_revision": self.code_revision,
            "code_sha256": self.code_sha256,
            "configuration_fingerprint": self.configuration_fingerprint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "upstream_adaptation_study_uids": list(self.upstream_adaptation_study_uids),
            "upstream_adaptation_artifact_sha256s": list(
                self.upstream_adaptation_artifact_sha256s
            ),
        }


@dataclass(frozen=True)
class PseudoLabelPrediction:
    """One teacher probability vector before the fixed confidence rule."""

    study_instance_uid: str
    scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.study_instance_uid.strip():
            raise PseudoLabelAdmissionError("pseudo-label predictions require a study identity")
        if len(self.scores) != NUM_TARGETS:
            raise PseudoLabelAdmissionError(
                f"teacher scores must contain exactly {NUM_TARGETS} targets"
            )
        if any(
            not math.isfinite(score) or score < 0.0 or score > 1.0
            for score in self.scores
        ):
            raise PseudoLabelAdmissionError("teacher scores must be finite probabilities")

    def to_dict(self) -> dict[str, object]:
        return {"study_instance_uid": self.study_instance_uid, "scores": list(self.scores)}


@dataclass(frozen=True)
class PseudoLabelGeneration:
    """Private, checksummed teacher output plus the exact candidate partition."""

    teacher: PseudoLabelTeacherProvenance
    eligible_study_uids: tuple[str, ...]
    excluded_study_uids: tuple[str, ...]
    predictions: tuple[PseudoLabelPrediction, ...]
    policy: PseudoLabelPolicy = PSEUDO_LABEL_POLICY

    def __post_init__(self) -> None:
        for field, values in (
            ("eligible pseudo-label studies", self.eligible_study_uids),
            ("excluded pseudo-label studies", self.excluded_study_uids),
        ):
            if tuple(sorted(values)) != values or len(set(values)) != len(values):
                raise PseudoLabelAdmissionError(f"{field} must be unique and sorted")
            if any(not value.strip() for value in values):
                raise PseudoLabelAdmissionError(f"{field} must contain non-empty identities")
        if not self.eligible_study_uids:
            raise PseudoLabelAdmissionError("pseudo-label generation requires eligible studies")
        if set(self.eligible_study_uids).intersection(self.excluded_study_uids):
            raise PseudoLabelAdmissionError("eligible and excluded pseudo-label studies overlap")
        if self.policy != PSEUDO_LABEL_POLICY:
            raise PseudoLabelAdmissionError("ticket 21 pseudo-label policy is frozen")
        prediction_uids = tuple(item.study_instance_uid for item in self.predictions)
        if tuple(sorted(prediction_uids)) != prediction_uids:
            raise PseudoLabelAdmissionError("pseudo-label predictions must be sorted by study identity")
        if len(set(prediction_uids)) != len(prediction_uids):
            raise PseudoLabelAdmissionError("pseudo-label predictions contain duplicate studies")
        if set(prediction_uids) != set(self.eligible_study_uids):
            raise PseudoLabelAdmissionError(
                "pseudo-label predictions must cover exactly the eligible studies"
            )

    @property
    def output_sha256(self) -> str:
        return _digest(
            {
                "policy": self.policy.to_dict(),
                "predictions": [item.to_dict() for item in self.predictions],
            }
        )

    @property
    def semantic_digest(self) -> str:
        return _digest(
            {
                "schema_version": PSEUDO_LABEL_SCHEMA_VERSION,
                "teacher": self.teacher.to_dict(),
                "eligible_study_uids": list(self.eligible_study_uids),
                "excluded_study_uids": list(self.excluded_study_uids),
                "generation_policy": self.policy.to_dict(),
                "output_sha256": self.output_sha256,
            }
        )

    def to_private_dict(self) -> dict[str, object]:
        return {
            "schema_version": PSEUDO_LABEL_SCHEMA_VERSION,
            "teacher": self.teacher.to_dict(),
            "eligible_study_uids": list(self.eligible_study_uids),
            "excluded_study_uids": list(self.excluded_study_uids),
            "generation_policy": self.policy.to_dict(),
            "predictions": [item.to_dict() for item in self.predictions],
            "output_sha256": self.output_sha256,
            "semantic_digest": self.semantic_digest,
        }


@dataclass(frozen=True)
class PseudoTrainingLabelVector:
    """One reference or admitted pseudo-label vector for the student."""

    study_instance_uid: str
    targets: tuple[int | None, ...]
    supervision_mask: tuple[int, ...]
    source_kind: str
    source_teacher_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.study_instance_uid.strip():
            raise PseudoLabelAdmissionError("student labels require a study identity")
        if len(self.targets) != NUM_TARGETS or len(self.supervision_mask) != NUM_TARGETS:
            raise PseudoLabelAdmissionError("student label vectors must contain twelve targets")
        if self.source_kind not in {"reference", "pseudo"}:
            raise PseudoLabelAdmissionError("student labels have an unsupported source kind")
        for target, mask in zip(self.targets, self.supervision_mask, strict=True):
            if target not in (None, 0, 1) or mask not in (0, 1) or mask != int(target is not None):
                raise PseudoLabelAdmissionError("student labels and masks disagree")
        if self.source_kind == "pseudo":
            _require_sha256(self.source_teacher_checkpoint_sha256, "teacher checkpoint hash")
        elif self.source_teacher_checkpoint_sha256 is not None:
            raise PseudoLabelAdmissionError("reference labels must not carry teacher provenance")

    @property
    def supervised_entry_count(self) -> int:
        return sum(self.supervision_mask)


@dataclass(frozen=True)
class AdmittedFoldPseudoLabels:
    """Fold-scoped labels consumed by student example construction."""

    fold: int
    fold_manifest_checksum: str
    source_artifact_digest: str
    teacher: PseudoLabelTeacherProvenance
    generation: PseudoLabelGeneration
    training_study_uids: tuple[str, ...]
    held_out_study_uids: tuple[str, ...]
    eligible_pseudo_study_uids: tuple[str, ...]
    excluded_pseudo_study_uids: tuple[str, ...]
    admitted_pseudo_study_uids: tuple[str, ...]
    abstained_pseudo_study_uids: tuple[str, ...]
    resolved: tuple[PseudoTrainingLabelVector, ...]

    @property
    def schema_version(self) -> str:
        return PSEUDO_LABEL_SCHEMA_VERSION

    @property
    def by_study(self) -> dict[str, PseudoTrainingLabelVector]:
        return {item.study_instance_uid: item for item in self.resolved}

    @property
    def reference_study_uids(self) -> tuple[str, ...]:
        return tuple(item.study_instance_uid for item in self.resolved if item.source_kind == "reference")

    @property
    def pseudo_study_count(self) -> int:
        return len(self.admitted_pseudo_study_uids)

    @property
    def supervised_entry_count(self) -> int:
        return sum(item.supervised_entry_count for item in self.resolved)

    def provenance_for(self, study_instance_uid: str) -> dict[str, object]:
        item = self.by_study.get(study_instance_uid)
        if item is None:
            raise PseudoLabelAdmissionError(
                "study is not in the admitted fold pseudo-label training set"
            )
        is_pseudo = item.source_kind == "pseudo"
        return {
            "source_kind": item.source_kind,
            "source_id": (
                f"pseudo-label-set:{self.source_artifact_digest}"
                if is_pseudo
                else f"fold-manifest:{self.fold_manifest_checksum}"
            ),
            "fold": self.fold,
            "held_out_excluded": True,
            "eligible_study_uids": list(self.training_study_uids),
            "excluded_study_uids": list(self.held_out_study_uids),
            "weak_label_version": None,
            "mixture_policy": self.generation.policy.version,
            "source_report_ref": None,
            "source_report_sha256": None,
            "source_disagreement_count": 0,
            "pseudo_label_version": self.generation.policy.version if is_pseudo else None,
            "teacher_checkpoint_sha256": (
                self.teacher.checkpoint_sha256 if is_pseudo else None
            ),
        }

    def to_public_dict(self) -> dict[str, object]:
        pseudo_entries = sum(
            item.supervised_entry_count for item in self.resolved if item.source_kind == "pseudo"
        )
        return {
            "kind": "pseudo",
            "schema_version": self.schema_version,
            "fold": self.fold,
            "repeat_name": self.teacher.repeat_name,
            "fold_manifest_checksum": self.fold_manifest_checksum,
            "source_artifact_digest": self.source_artifact_digest,
            "pseudo_label_output_sha256": self.generation.output_sha256,
            "teacher": {
                "source_fingerprint": self.teacher.source_fingerprint,
                "code_revision": self.teacher.code_revision,
                "code_sha256": self.teacher.code_sha256,
                "configuration_fingerprint": self.teacher.configuration_fingerprint,
                "checkpoint_sha256": self.teacher.checkpoint_sha256,
                "training_study_set_sha256": _digest(
                    list(self.teacher.training_study_uids)
                ),
                "excluded_study_set_sha256": _digest(
                    list(self.teacher.held_out_study_uids)
                ),
                "adaptation_artifact_sha256s": list(
                    self.teacher.upstream_adaptation_artifact_sha256s
                ),
            },
            "policy": self.generation.policy.to_dict(),
            "counts": {
                "training_studies": len(self.training_study_uids),
                "reference_studies": len(self.reference_study_uids),
                "eligible_pseudo_studies": len(self.eligible_pseudo_study_uids),
                "admitted_pseudo_studies": len(self.admitted_pseudo_study_uids),
                "abstained_pseudo_studies": len(self.abstained_pseudo_study_uids),
                "excluded_pseudo_studies": len(self.excluded_pseudo_study_uids),
                "held_out_studies": len(self.held_out_study_uids),
                "reference_supervised_entries": sum(
                    item.supervised_entry_count
                    for item in self.resolved
                    if item.source_kind == "reference"
                ),
                "pseudo_supervised_entries": pseudo_entries,
            },
            "boundaries": {
                "held_out_inputs_excluded": True,
                "held_out_labels_evaluation_only": True,
                "teacher_training_reference_only": True,
                "reports_used": False,
                "reference_labels_take_precedence": True,
                "uncertain_scores_masked": True,
            },
        }

    def to_private_dict(self) -> dict[str, object]:
        payload = self.to_public_dict()
        payload.update(
            {
                "training_study_uids": list(self.training_study_uids),
                "held_out_study_uids": list(self.held_out_study_uids),
                "eligible_pseudo_study_uids": list(self.eligible_pseudo_study_uids),
                "excluded_pseudo_study_uids": list(self.excluded_pseudo_study_uids),
                "admitted_pseudo_study_uids": list(self.admitted_pseudo_study_uids),
                "abstained_pseudo_study_uids": list(self.abstained_pseudo_study_uids),
                "teacher_provenance": self.teacher.to_dict(),
                "generation": self.generation.to_private_dict(),
                "resolved_studies": [
                    {
                        "study_instance_uid": item.study_instance_uid,
                        "targets": list(item.targets),
                        "supervision_mask": list(item.supervision_mask),
                        "source_kind": item.source_kind,
                        "source_teacher_checkpoint_sha256": item.source_teacher_checkpoint_sha256,
                    }
                    for item in self.resolved
                ],
            }
        )
        return payload


def create_pseudo_label_generation(
    teacher: PseudoLabelTeacherProvenance,
    predictions: Mapping[str, Sequence[float]],
    *,
    eligible_study_uids: Sequence[str],
    excluded_study_uids: Sequence[str],
) -> PseudoLabelGeneration:
    """Freeze scores, settings and study identities into a deterministic artifact."""

    eligible = _study_ids(eligible_study_uids, "eligible_study_uids")
    excluded = _study_ids(excluded_study_uids, "excluded_study_uids")
    ordered_predictions = tuple(
        PseudoLabelPrediction(uid, tuple(float(score) for score in predictions[uid]))
        for uid in sorted(predictions)
    )
    return PseudoLabelGeneration(
        teacher=teacher,
        eligible_study_uids=eligible,
        excluded_study_uids=excluded,
        predictions=ordered_predictions,
    )


def admit_fold_pseudo_labels(
    manifest: FoldManifest,
    fold: int,
    generation: PseudoLabelGeneration,
) -> AdmittedFoldPseudoLabels:
    """Admit only report-only training studies scored by a clean fold teacher."""

    if fold not in FOLD_IDS:
        raise PseudoLabelAdmissionError(f"fold must be one of {FOLD_IDS}")
    if manifest.superseded or manifest.fold_ids != FOLD_IDS:
        raise PseudoLabelAdmissionError("pseudo-label admission requires the canonical frozen folds")
    teacher = generation.teacher
    if teacher.fold != fold:
        raise PseudoLabelAdmissionError("teacher fold does not match the consuming fold")
    if teacher.fold_manifest_checksum != manifest.manifest_checksum:
        raise PseudoLabelAdmissionError("teacher lineage does not match the frozen fold manifest")
    if teacher.repeat_name != manifest.repeat_name:
        raise PseudoLabelAdmissionError("teacher lineage has a different fold repeat")
    expected_gold_training = tuple(
        sorted(item.study_instance_uid for item in manifest.gold_memberships if item.fold != fold)
    )
    expected_held_out = tuple(sorted(manifest.held_out_study_uids(fold)))
    if teacher.training_study_uids != expected_gold_training:
        raise PseudoLabelAdmissionError(
            "teacher training cohort must be exactly the consuming fold's gold training studies"
        )
    if teacher.held_out_study_uids != expected_held_out:
        raise PseudoLabelAdmissionError("teacher held-out exclusion lineage is incomplete")
    if set(teacher.upstream_adaptation_study_uids).intersection(expected_held_out):
        raise PseudoLabelAdmissionError("teacher adaptation contains a held-out study")
    if manifest.source_fingerprint is not None and teacher.source_fingerprint != manifest.source_fingerprint:
        raise PseudoLabelAdmissionError("teacher source fingerprint disagrees with the fold manifest")

    expected_candidates = {
        item.study_instance_uid
        for item in manifest.report_only_memberships
        if item.fold != fold
    }
    eligible = set(generation.eligible_study_uids)
    excluded = set(generation.excluded_study_uids)
    if eligible.intersection(expected_held_out) or excluded.intersection(expected_held_out):
        raise PseudoLabelAdmissionError("pseudo-label generation includes a held-out study")
    if eligible.intersection(teacher.training_study_uids):
        raise PseudoLabelAdmissionError("pseudo-label generation includes a teacher training study")
    if eligible | excluded != expected_candidates or eligible.intersection(excluded):
        raise PseudoLabelAdmissionError(
            "eligible and excluded pseudo-label studies do not partition the fold's report-only training cohort"
        )
    prediction_by_uid = {item.study_instance_uid: item for item in generation.predictions}
    if set(prediction_by_uid) != eligible:
        raise PseudoLabelAdmissionError(
            "pseudo-label predictions do not match the eligible training study identities"
        )

    resolved: list[PseudoTrainingLabelVector] = []
    for membership in sorted(
        (item for item in manifest.gold_memberships if item.fold != fold),
        key=lambda item: item.study_instance_uid,
    ):
        if any(value not in (0, 1) for value in membership.reference_labels):
            raise PseudoLabelAdmissionError("teacher cohort contains incomplete reference labels")
        values = tuple(cast(int, value) for value in membership.reference_labels)
        resolved.append(
            PseudoTrainingLabelVector(
                study_instance_uid=membership.study_instance_uid,
                targets=values,
                supervision_mask=(1,) * NUM_TARGETS,
                source_kind="reference",
            )
        )

    admitted_uids: list[str] = []
    abstained_uids: list[str] = []
    for uid in sorted(eligible):
        scores = prediction_by_uid[uid].scores
        targets = tuple(
            1
            if score >= generation.policy.positive_threshold
            else 0
            if score <= generation.policy.negative_threshold
            else None
            for score in scores
        )
        if not any(value is not None for value in targets):
            abstained_uids.append(uid)
            continue
        admitted_uids.append(uid)
        resolved.append(
            PseudoTrainingLabelVector(
                study_instance_uid=uid,
                targets=targets,
                supervision_mask=tuple(int(value is not None) for value in targets),
                source_kind="pseudo",
                source_teacher_checkpoint_sha256=teacher.checkpoint_sha256,
            )
        )

    resolved.sort(key=lambda item: item.study_instance_uid)
    training_uids = tuple(item.study_instance_uid for item in resolved)
    if set(training_uids).intersection(expected_held_out):
        raise PseudoLabelAdmissionError("admitted student supervision includes a held-out study")
    return AdmittedFoldPseudoLabels(
        fold=fold,
        fold_manifest_checksum=manifest.manifest_checksum,
        source_artifact_digest=generation.semantic_digest,
        teacher=teacher,
        generation=generation,
        training_study_uids=training_uids,
        held_out_study_uids=expected_held_out,
        eligible_pseudo_study_uids=tuple(sorted(eligible)),
        excluded_pseudo_study_uids=tuple(sorted(excluded)),
        admitted_pseudo_study_uids=tuple(admitted_uids),
        abstained_pseudo_study_uids=tuple(abstained_uids),
        resolved=tuple(resolved),
    )


__all__ = [
    "PSEUDO_LABEL_POLICY",
    "PSEUDO_LABEL_SCHEMA_VERSION",
    "PSEUDO_LABEL_VERSION",
    "AdmittedFoldPseudoLabels",
    "PseudoLabelAdmissionError",
    "PseudoLabelGeneration",
    "PseudoLabelPolicy",
    "PseudoLabelPrediction",
    "PseudoLabelTeacherProvenance",
    "PseudoTrainingLabelVector",
    "admit_fold_pseudo_labels",
    "create_pseudo_label_generation",
]
