"""Fold-clean deterministic report extraction and evaluation.

Ticket 14 owns the conservative interpretation rules.  This module owns the
next boundary: private report inputs are joined to an immutable fold manifest,
report-only records are admitted as training labels only from the fold's
training side, and gold reports in the held-out side are evaluated separately.
The public result contains aggregate evidence only; private audit records keep
identifiers, report references, hashes, rule ids and matched snippets, but
never persist the source report text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from math import isfinite
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence, cast

from .archive import atomic_json_write
from .folds import ADAPTATION_EXCLUSIONS, FOLD_IDS, FOLD_SCHEMA_VERSION, FoldManifest, FoldMembership
from .report_labels import (
    LabelState,
    REPORT_EXTRACTOR_CONFIGURATION,
    REPORT_EXTRACTOR_CONFIGURATION_FINGERPRINT,
    REPORT_EXTRACTOR_VERSION,
    ReportInterpretation,
    ReportInterpretationError,
    interpret_report,
    target_definitions,
    validate_rule_origin_for_folds,
    authority_register,
)
from .targets import TARGET_COLUMNS


REPORT_CORPUS_SCHEMA_VERSION = "rsna-knee-report-corpus-v1"
REPORT_LABEL_ARTIFACT_SCHEMA_VERSION = "rsna-knee-report-labels-v1"
REPORT_EVALUATION_SCHEMA_VERSION = "rsna-knee-report-evaluation-v1"
REPORT_EXTRACTION_SCHEMA_VERSION = "rsna-knee-report-extraction-v1"
REPORT_TRAINING_POLICY_VERSION = "report-only-confident-binary-v1"
REPORT_EVALUATION_POLICY_VERSION = "held-out-gold-frozen-extractor-v1"
LOCAL_SMOKE_MAX_STUDIES = 128
LOCAL_SMOKE_MAX_REPORTS = 32
REPORT_EXTRACTION_CODE_FINGERPRINT = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()
REPORT_RULESET_FINGERPRINT = hashlib.sha256(
    json.dumps(
        {
            "extractor_version": REPORT_EXTRACTOR_VERSION,
            "configuration": dict(REPORT_EXTRACTOR_CONFIGURATION),
            "authority_register": authority_register(),
        },
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


class ReportExtractionError(ValueError):
    """Raised when a report artifact or fold boundary is unsafe."""


class ReportSourceError(ReportExtractionError):
    """Raised when private report inputs do not reconcile with frozen folds."""


class ReportArtifactError(ReportExtractionError):
    """Raised when a persisted report artifact is incomplete or inconsistent."""


class ReportPartition(StrEnum):
    """The two intentionally separate destinations for extracted labels."""

    TRAINING_LABELS = "admitted_training_labels"
    HELD_OUT_EVALUATION = "held_out_evaluation"


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReportExtractionError(f"{field_name} must be a non-empty string")
    return value


def _validate_fold(fold: int) -> None:
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in FOLD_IDS:
        raise ReportExtractionError(f"fold must be one of {FOLD_IDS}")


def _validate_unique_strings(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    result = tuple(_require_text(value, field_name) for value in values)
    if len(set(result)) != len(result):
        raise ReportExtractionError(f"{field_name} must not contain duplicates")
    return result


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _resolve_code_revision() -> str:
    configured = os.environ.get("RSNA_GIT_REVISION", "").strip()
    if configured:
        return configured
    repository = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    revision = completed.stdout.strip()
    return revision if revision else "unavailable"


def _environment_fingerprint() -> str:
    package_versions: dict[str, str] = {}
    for package_name in ("rsna-knee", "numpy", "pydicom", "torch"):
        try:
            package_versions[package_name] = version(package_name)
        except PackageNotFoundError:
            package_versions[package_name] = "unavailable"
    return _json_sha256(
        {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": package_versions,
        }
    )


def _execution_identity() -> tuple[str, str]:
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    partition = os.environ.get("SLURM_JOB_PARTITION", "").strip()
    if job_id:
        return "slurm", f"slurm:{partition or 'unknown'}:{job_id}"
    return "local", "local-process"


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ReportArtifactError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ReportRecord:
    """One private source report held in memory during a local/cluster run."""

    study_instance_uid: str
    source_report_ref: str
    report_text: str = field(repr=False)

    def __post_init__(self) -> None:
        _require_text(self.study_instance_uid, "study_instance_uid")
        _require_text(self.source_report_ref, "source_report_ref")
        if not isinstance(self.report_text, str):
            raise ReportExtractionError("report_text must be a string")

    @property
    def report_sha256(self) -> str:
        """Return the source digest without retaining it as another input field."""

        return hashlib.sha256(self.report_text.encode("utf-8")).hexdigest()

    @property
    def text(self) -> str:
        """Compatibility alias for callers that call the private text ``text``."""

        return self.report_text


@dataclass(frozen=True)
class ReportCorpus:
    """Private report inputs with one immutable record per study."""

    source_version: str
    records: tuple[ReportRecord, ...]

    def __post_init__(self) -> None:
        _require_text(self.source_version, "source_version")
        if not isinstance(self.records, tuple):
            object.__setattr__(self, "records", tuple(self.records))
        if any(not isinstance(record, ReportRecord) for record in self.records):
            raise ReportSourceError("report corpus contains a non-ReportRecord")
        study_uids = _validate_unique_strings(
            (record.study_instance_uid for record in self.records),
            "report study identifiers",
        )
        refs = _validate_unique_strings(
            (record.source_report_ref for record in self.records),
            "source report references",
        )
        if len(study_uids) != len(self.records) or len(refs) != len(self.records):
            raise ReportSourceError("report corpus records must be unique")

    @property
    def by_study(self) -> dict[str, ReportRecord]:
        return {record.study_instance_uid: record for record in self.records}

    @property
    def report_count(self) -> int:
        return len(self.records)

    @property
    def record_set_sha256(self) -> str:
        """Hash the private report identities and content digests as one input set."""

        return _json_sha256(
            [
                {
                    "study_instance_uid": record.study_instance_uid,
                    "source_report_ref": record.source_report_ref,
                    "report_sha256": record.report_sha256,
                }
                for record in sorted(self.records, key=lambda item: item.study_instance_uid)
            ]
        )

    def to_public_dict(self) -> dict[str, object]:
        """Return source counts and policy, never identifiers or report text."""

        return {
            "schema_version": REPORT_CORPUS_SCHEMA_VERSION,
            "source_version": self.source_version,
            "report_count": self.report_count,
            "record_set_sha256": self.record_set_sha256,
            "raw_report_text_persisted": False,
            "source_report_references_private": True,
        }

    def to_private_dict(self) -> dict[str, object]:
        """Return audit metadata without copying private report text."""

        return {
            "schema_version": REPORT_CORPUS_SCHEMA_VERSION,
            "source_version": self.source_version,
            "report_count": self.report_count,
            "record_set_sha256": self.record_set_sha256,
            "raw_report_text_persisted": False,
            "reports": [
                {
                    "study_instance_uid": record.study_instance_uid,
                    "source_report_ref": record.source_report_ref,
                    "report_sha256": record.report_sha256,
                }
                for record in sorted(self.records, key=lambda item: item.study_instance_uid)
            ],
        }


def _object_payload(path: str | Path, description: str) -> Mapping[str, object]:
    candidate = Path(path)
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportSourceError(f"could not read {description}: {candidate}") from exc
    if not isinstance(payload, Mapping):
        raise ReportSourceError(f"{description} must be a JSON object")
    return cast(Mapping[str, object], payload)


def load_report_corpus(path: str | Path) -> ReportCorpus:
    """Load a private corpus input; the input file must contain source text."""

    payload = _object_payload(path, "report corpus")
    if payload.get("schema_version") != REPORT_CORPUS_SCHEMA_VERSION:
        raise ReportSourceError("report corpus has an unexpected schema version")
    raw_records = payload.get("reports")
    if not isinstance(raw_records, list):
        raise ReportSourceError("report corpus reports must be an array")
    records: list[ReportRecord] = []
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise ReportSourceError(f"report corpus record {index} must be an object")
        report_text = raw.get("report_text", raw.get("text"))
        record = ReportRecord(
            study_instance_uid=_require_text(
                raw.get("study_instance_uid"),
                f"report {index} study_instance_uid",
            ),
            source_report_ref=_require_text(
                raw.get("source_report_ref"),
                f"report {index} source_report_ref",
            ),
            report_text=report_text if isinstance(report_text, str) else "__invalid__",
        )
        if not isinstance(report_text, str):
            raise ReportSourceError(f"report {index} report_text must be a string")
        declared_digest = raw.get("report_sha256")
        if declared_digest is not None and declared_digest != record.report_sha256:
            raise ReportSourceError(f"report {index} report_sha256 disagrees with report_text")
        records.append(record)
    return ReportCorpus(
        source_version=_require_text(payload.get("source_version"), "report source_version"),
        records=tuple(records),
    )


@dataclass(frozen=True)
class FrozenDeterministicExtractor:
    """Immutable configuration for the ticket-14 deterministic comparator."""

    extractor_version: str = REPORT_EXTRACTOR_VERSION
    configuration_fingerprint: str = REPORT_EXTRACTOR_CONFIGURATION_FINGERPRINT
    ruleset_fingerprint: str = REPORT_RULESET_FINGERPRINT

    def __post_init__(self) -> None:
        if self.extractor_version != REPORT_EXTRACTOR_VERSION:
            raise ReportExtractionError("unsupported deterministic extractor version")
        if self.configuration_fingerprint != REPORT_EXTRACTOR_CONFIGURATION_FINGERPRINT:
            raise ReportExtractionError("deterministic extractor configuration fingerprint mismatch")
        if self.ruleset_fingerprint != REPORT_RULESET_FINGERPRINT:
            raise ReportExtractionError("deterministic extractor ruleset fingerprint mismatch")
        _validate_digest(self.configuration_fingerprint, "configuration_fingerprint")
        _validate_digest(self.ruleset_fingerprint, "ruleset_fingerprint")
        for definition in target_definitions():
            validate_rule_origin_for_folds(definition.rule_origin, FOLD_IDS)

    @property
    def is_frozen(self) -> bool:
        """Whether this immutable object can be used for held-out evaluation."""

        return True

    @property
    def fingerprint(self) -> str:
        """Hash the complete version/configuration/ruleset identity."""

        return _json_sha256(
            {
                "extractor_version": self.extractor_version,
                "configuration_fingerprint": self.configuration_fingerprint,
                "ruleset_fingerprint": self.ruleset_fingerprint,
                "code_fingerprint": REPORT_EXTRACTION_CODE_FINGERPRINT,
            }
        )

    def to_public_dict(self) -> dict[str, object]:
        """Return the reproducible extractor identity without private data."""

        return {
            "extractor_version": self.extractor_version,
            "configuration_fingerprint": self.configuration_fingerprint,
            "ruleset_fingerprint": self.ruleset_fingerprint,
            "code_fingerprint": REPORT_EXTRACTION_CODE_FINGERPRINT,
            "fingerprint": self.fingerprint,
            "configuration": dict(REPORT_EXTRACTOR_CONFIGURATION),
            "frozen": True,
        }

    def extract(self, report: ReportRecord, *, fold: int) -> ReportInterpretation:
        """Extract one report for exactly one consuming fold."""

        if not isinstance(report, ReportRecord):
            raise ReportExtractionError("extractor input must be a ReportRecord")
        _validate_fold(fold)
        try:
            interpretation = interpret_report(
                report.report_text,
                source_report_ref=report.source_report_ref,
                consuming_folds=(fold,),
            )
        except ReportInterpretationError as exc:
            raise ReportExtractionError(f"could not interpret report for fold {fold}") from exc
        if interpretation.report_sha256 != report.report_sha256:
            raise ReportArtifactError("interpretation report digest disagrees with its source record")
        return interpretation


def freeze_deterministic_extractor() -> FrozenDeterministicExtractor:
    """Create the only deterministic extractor admitted by ticket 15."""

    return FrozenDeterministicExtractor()


@dataclass(frozen=True)
class ReportLabelRecord:
    """One fold-scoped interpretation with an explicit artifact partition."""

    study_instance_uid: str
    fold: int
    partition: ReportPartition
    interpretation: ReportInterpretation

    def __post_init__(self) -> None:
        _require_text(self.study_instance_uid, "study_instance_uid")
        _validate_fold(self.fold)
        if not isinstance(self.partition, ReportPartition):
            try:
                object.__setattr__(self, "partition", ReportPartition(str(self.partition)))
            except ValueError as exc:
                raise ReportArtifactError("unknown report label partition") from exc
        if not isinstance(self.interpretation, ReportInterpretation):
            raise ReportArtifactError("report label record requires a ReportInterpretation")
        if self.interpretation.consuming_folds != (self.fold,):
            raise ReportArtifactError("report label interpretation must name exactly its consuming fold")
        for label in self.interpretation.labels:
            validate_rule_origin_for_folds(label.rule_origin, (self.fold,))
            if label.state in {LabelState.POSITIVE, LabelState.NEGATIVE}:
                if label.confidence is None:
                    raise ReportArtifactError(
                        "confident report labels must carry a confidence value"
                    )
            elif label.confidence is not None:
                raise ReportArtifactError("abstained report labels cannot carry confidence")

    @property
    def source_report_ref(self) -> str:
        return self.interpretation.source_report_ref

    @property
    def report_sha256(self) -> str:
        return self.interpretation.report_sha256

    @property
    def values(self) -> tuple[int | None, ...]:
        return self.interpretation.values

    @property
    def supervision_mask(self) -> tuple[int, ...]:
        return self.interpretation.supervision_mask

    @property
    def state_counts(self) -> dict[str, int]:
        counts = {state.value: 0 for state in LabelState}
        for label in self.interpretation.labels:
            counts[label.state.value] += 1
        return counts

    @property
    def supervised_entry_count(self) -> int:
        return sum(self.supervision_mask)

    def semantic_dict(self) -> dict[str, object]:
        """Return the private, text-free content used for reproducibility hashes."""

        return {
            "study_instance_uid": self.study_instance_uid,
            "fold": self.fold,
            "partition": self.partition.value,
            "report_sha256": self.report_sha256,
            "source_report_ref": self.source_report_ref,
            "interpretation": self.interpretation.to_private_dict(),
        }

    def to_public_dict(self) -> dict[str, object]:
        """Return one identifier-free label record."""

        return {
            "fold": self.fold,
            "partition": self.partition.value,
            "states": [item.state.value for item in self.interpretation.labels],
            "values": list(self.values),
            "supervision_mask": list(self.supervision_mask),
            "rule_ids": [item.rule_id for item in self.interpretation.labels],
        }

    def to_private_dict(self) -> dict[str, object]:
        """Return local audit evidence without source report text."""

        return {
            "schema_version": REPORT_LABEL_ARTIFACT_SCHEMA_VERSION,
            "artifact_kind": self.partition.value,
            "study_instance_uid": self.study_instance_uid,
            "fold": self.fold,
            "source_report_ref": self.source_report_ref,
            "report_sha256": self.report_sha256,
            "interpretation": self.interpretation.to_private_dict(),
        }


def _metric_count(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReportArtifactError(f"{field_name} must be a non-negative integer")


def _metric_float(value: float | None, field_name: str) -> None:
    if value is None:
        return
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ReportArtifactError(f"{field_name} must be null or within [0, 1]")


@dataclass(frozen=True)
class TargetReportMetrics:
    """Accuracy, coverage, abstention and uncertainty for one target."""

    target: str
    report_count: int
    gold_positive_count: int
    gold_negative_count: int
    predicted_positive_count: int
    predicted_negative_count: int
    correct_count: int
    accuracy: float | None
    coverage: float | None
    not_mentioned_count: int
    uncertain_count: int
    unknown_count: int
    extraction_failed_count: int
    abstention_count: int
    failure_count: int
    mean_confidence: float | None

    def __post_init__(self) -> None:
        if self.target not in TARGET_COLUMNS:
            raise ReportArtifactError(f"unknown report-evaluation target {self.target!r}")
        count_fields = (
            "report_count",
            "gold_positive_count",
            "gold_negative_count",
            "predicted_positive_count",
            "predicted_negative_count",
            "correct_count",
            "not_mentioned_count",
            "uncertain_count",
            "unknown_count",
            "extraction_failed_count",
            "abstention_count",
            "failure_count",
        )
        for field_name in count_fields:
            _metric_count(cast(int, getattr(self, field_name)), field_name)
        state_total = (
            self.predicted_positive_count
            + self.predicted_negative_count
            + self.not_mentioned_count
            + self.uncertain_count
            + self.unknown_count
            + self.extraction_failed_count
        )
        if state_total != self.report_count:
            raise ReportArtifactError(f"{self.target} state counts do not add to report_count")
        if self.gold_positive_count + self.gold_negative_count != self.report_count:
            raise ReportArtifactError(f"{self.target} reference counts do not add to report_count")
        if self.correct_count > self.confident_count:
            raise ReportArtifactError(f"{self.target} correct count exceeds confident predictions")
        if self.abstention_count != (
            self.not_mentioned_count
            + self.uncertain_count
            + self.unknown_count
            + self.extraction_failed_count
        ):
            raise ReportArtifactError(f"{self.target} abstention counts are inconsistent")
        if self.failure_count != self.extraction_failed_count:
            raise ReportArtifactError(f"{self.target} failure count is inconsistent")
        _metric_float(self.accuracy, f"{self.target} accuracy")
        _metric_float(self.coverage, f"{self.target} coverage")
        _metric_float(self.mean_confidence, f"{self.target} mean_confidence")
        if (self.confident_count == 0) != (self.accuracy is None):
            raise ReportArtifactError(f"{self.target} accuracy definedness is inconsistent")
        if (self.report_count == 0) != (self.coverage is None):
            raise ReportArtifactError(f"{self.target} coverage definedness is inconsistent")
        if (self.confident_count == 0) != (self.mean_confidence is None):
            raise ReportArtifactError(f"{self.target} confidence definedness is inconsistent")

    @property
    def confident_count(self) -> int:
        return self.predicted_positive_count + self.predicted_negative_count

    @property
    def abstention_rate(self) -> float | None:
        return self.abstention_count / self.report_count if self.report_count else None

    @property
    def failure_rate(self) -> float | None:
        return self.failure_count / self.report_count if self.report_count else None

    @property
    def state_counts(self) -> dict[str, int]:
        return {
            LabelState.POSITIVE.value: self.predicted_positive_count,
            LabelState.NEGATIVE.value: self.predicted_negative_count,
            LabelState.NOT_MENTIONED.value: self.not_mentioned_count,
            LabelState.UNCERTAIN.value: self.uncertain_count,
            LabelState.UNKNOWN.value: self.unknown_count,
            LabelState.EXTRACTION_FAILED.value: self.extraction_failed_count,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "report_count": self.report_count,
            "gold_positive_count": self.gold_positive_count,
            "gold_negative_count": self.gold_negative_count,
            "predicted_positive_count": self.predicted_positive_count,
            "predicted_negative_count": self.predicted_negative_count,
            "confident_count": self.confident_count,
            "correct_count": self.correct_count,
            "accuracy": self.accuracy,
            "coverage": self.coverage,
            "abstention_count": self.abstention_count,
            "abstention_rate": self.abstention_rate,
            "failure_count": self.failure_count,
            "failure_rate": self.failure_rate,
            "uncertainty": {
                "not_mentioned": self.not_mentioned_count,
                "uncertain": self.uncertain_count,
                "unknown": self.unknown_count,
                "extraction_failed": self.extraction_failed_count,
            },
            "mean_confidence": self.mean_confidence,
        }


@dataclass(frozen=True)
class FoldReportMetrics:
    """Held-out report metrics for one frozen fold."""

    fold: int
    report_count: int
    missing_report_count: int
    targets: tuple[TargetReportMetrics, ...]

    def __post_init__(self) -> None:
        _validate_fold(self.fold)
        _metric_count(self.report_count, "report_count")
        _metric_count(self.missing_report_count, "missing_report_count")
        if tuple(item.target for item in self.targets) != TARGET_COLUMNS:
            raise ReportArtifactError("fold report metrics must use competition target order")
        if any(item.report_count != self.report_count for item in self.targets):
            raise ReportArtifactError("fold target report counts disagree")

    @property
    def macro_accuracy(self) -> float | None:
        values = [item.accuracy for item in self.targets]
        return sum(cast(float, value) for value in values) / len(values) if all(
            value is not None for value in values
        ) else None

    @property
    def mean_coverage(self) -> float | None:
        values = [item.coverage for item in self.targets]
        return sum(cast(float, value) for value in values) / len(values) if all(
            value is not None for value in values
        ) else None

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "report_count": self.report_count,
            "missing_report_count": self.missing_report_count,
            "macro_accuracy": self.macro_accuracy,
            "mean_coverage": self.mean_coverage,
            "targets": [item.to_dict() for item in self.targets],
        }


@dataclass(frozen=True)
class LatencySummary:
    """Measured extraction latency for the reports processed by a run."""

    report_count: int
    total_elapsed_ns: int
    minimum_elapsed_ns: int
    maximum_elapsed_ns: int
    mean_latency_ms: float
    p95_latency_ms: float

    def __post_init__(self) -> None:
        _metric_count(self.report_count, "latency report_count")
        for field_name in (
            "total_elapsed_ns",
            "minimum_elapsed_ns",
            "maximum_elapsed_ns",
        ):
            _metric_count(cast(int, getattr(self, field_name)), f"latency {field_name}")
        if self.report_count == 0:
            if any(
                value != 0
                for value in (
                    self.total_elapsed_ns,
                    self.minimum_elapsed_ns,
                    self.maximum_elapsed_ns,
                )
            ):
                raise ReportArtifactError("empty latency summary must have zero nanoseconds")
        elif self.minimum_elapsed_ns > self.maximum_elapsed_ns:
            raise ReportArtifactError("latency minimum exceeds maximum")
        if (
            not isfinite(self.mean_latency_ms)
            or not isfinite(self.p95_latency_ms)
            or self.mean_latency_ms < 0.0
            or self.p95_latency_ms < 0.0
        ):
            raise ReportArtifactError("latency values cannot be negative")

    @classmethod
    def from_durations(cls, durations_ns: Sequence[int]) -> "LatencySummary":
        values = tuple(durations_ns)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ReportArtifactError("latency durations cannot be negative")
        if not values:
            return cls(0, 0, 0, 0, 0.0, 0.0)
        ordered = sorted(values)
        percentile_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
        return cls(
            report_count=len(values),
            total_elapsed_ns=sum(values),
            minimum_elapsed_ns=min(values),
            maximum_elapsed_ns=max(values),
            mean_latency_ms=sum(values) / len(values) / 1_000_000.0,
            p95_latency_ms=ordered[percentile_index] / 1_000_000.0,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "report_count": self.report_count,
            "total_elapsed_seconds": self.total_elapsed_ns / 1_000_000_000.0,
            "minimum_latency_ms": self.minimum_elapsed_ns / 1_000_000.0,
            "maximum_latency_ms": self.maximum_elapsed_ns / 1_000_000.0,
            "mean_latency_ms": self.mean_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
        }


@dataclass(frozen=True)
class FoldReportExtraction:
    """Private fold output with training and evaluation partitions separated."""

    fold: int
    fold_manifest_checksum: str
    fold_training_study_uids: tuple[str, ...]
    fold_held_out_study_uids: tuple[str, ...]
    eligible_training_report_uids: tuple[str, ...]
    eligible_evaluation_gold_uids: tuple[str, ...]
    training_labels: tuple[ReportLabelRecord, ...]
    evaluation_labels: tuple[ReportLabelRecord, ...]
    missing_training_report_uids: tuple[str, ...]
    missing_evaluation_report_uids: tuple[str, ...]
    excluded_held_out_report_uids: tuple[str, ...]
    latency_samples_ns: tuple[int, ...]
    evaluation_metrics: FoldReportMetrics
    latency: LatencySummary

    def __post_init__(self) -> None:
        _validate_fold(self.fold)
        _validate_text_digest(self.fold_manifest_checksum, "fold_manifest_checksum")
        if self.evaluation_metrics.fold != self.fold:
            raise ReportArtifactError("fold report metrics use a different fold")
        for field_name, values in (
            ("fold_training_study_uids", self.fold_training_study_uids),
            ("fold_held_out_study_uids", self.fold_held_out_study_uids),
            ("eligible_training_report_uids", self.eligible_training_report_uids),
            ("eligible_evaluation_gold_uids", self.eligible_evaluation_gold_uids),
            ("missing_training_report_uids", self.missing_training_report_uids),
            ("missing_evaluation_report_uids", self.missing_evaluation_report_uids),
            ("excluded_held_out_report_uids", self.excluded_held_out_report_uids),
        ):
            _validate_unique_strings(values, field_name)
        for field_name, values in (
            ("fold_training_study_uids", self.fold_training_study_uids),
            ("fold_held_out_study_uids", self.fold_held_out_study_uids),
            ("eligible_training_report_uids", self.eligible_training_report_uids),
            ("eligible_evaluation_gold_uids", self.eligible_evaluation_gold_uids),
            ("missing_training_report_uids", self.missing_training_report_uids),
            ("missing_evaluation_report_uids", self.missing_evaluation_report_uids),
            ("excluded_held_out_report_uids", self.excluded_held_out_report_uids),
        ):
            if tuple(sorted(values)) != values:
                raise ReportArtifactError(f"{field_name} must be sorted")
        if set(self.fold_training_study_uids).intersection(self.fold_held_out_study_uids):
            raise ReportArtifactError("fold training and held-out study sets overlap")
        if not set(self.eligible_training_report_uids).issubset(self.fold_training_study_uids):
            raise ReportArtifactError("eligible training reports are outside the fold training set")
        if not set(self.eligible_evaluation_gold_uids).issubset(self.fold_held_out_study_uids):
            raise ReportArtifactError("eligible evaluation reports are outside the held-out set")
        if not set(self.excluded_held_out_report_uids).issubset(self.fold_held_out_study_uids):
            raise ReportArtifactError("excluded reports are outside the held-out set")
        training_uids = set()
        for record in self.training_labels:
            if record.fold != self.fold or record.partition is not ReportPartition.TRAINING_LABELS:
                raise ReportArtifactError("training label partition is inconsistent")
            if record.study_instance_uid in training_uids:
                raise ReportArtifactError("training labels contain duplicate study identifiers")
            training_uids.add(record.study_instance_uid)
        evaluation_uids = set()
        for record in self.evaluation_labels:
            if record.fold != self.fold or record.partition is not ReportPartition.HELD_OUT_EVALUATION:
                raise ReportArtifactError("evaluation label partition is inconsistent")
            if record.study_instance_uid in evaluation_uids:
                raise ReportArtifactError("evaluation labels contain duplicate study identifiers")
            evaluation_uids.add(record.study_instance_uid)
        if training_uids.intersection(evaluation_uids):
            raise ReportArtifactError("training and evaluation report partitions overlap")
        if not training_uids.issubset(self.eligible_training_report_uids):
            raise ReportArtifactError("training labels are outside the eligible training report set")
        if not evaluation_uids.issubset(self.eligible_evaluation_gold_uids):
            raise ReportArtifactError("evaluation labels are outside the eligible gold report set")
        if training_uids.union(self.missing_training_report_uids) != set(
            self.eligible_training_report_uids
        ) or training_uids.intersection(self.missing_training_report_uids):
            raise ReportArtifactError("training report eligibility and missing IDs disagree")
        if evaluation_uids.union(self.missing_evaluation_report_uids) != set(
            self.eligible_evaluation_gold_uids
        ) or evaluation_uids.intersection(self.missing_evaluation_report_uids):
            raise ReportArtifactError("evaluation report eligibility and missing IDs disagree")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.latency_samples_ns
        ):
            raise ReportArtifactError("latency samples must be non-negative integers")
        if len(self.latency_samples_ns) != self.latency.report_count:
            raise ReportArtifactError("latency samples do not match extracted records")
        if LatencySummary.from_durations(self.latency_samples_ns) != self.latency:
            raise ReportArtifactError("latency summary does not match its samples")
        if self.latency.report_count != len(self.training_labels) + len(self.evaluation_labels):
            raise ReportArtifactError("fold latency count does not match extracted records")

    @property
    def training_supervised_entry_count(self) -> int:
        return sum(record.supervised_entry_count for record in self.training_labels)

    @property
    def evaluation_abstention_entry_count(self) -> int:
        return sum(
            sum(1 - value for value in record.supervision_mask)
            for record in self.evaluation_labels
        )

    def semantic_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "fold_manifest_checksum": self.fold_manifest_checksum,
            "fold_training_study_uids": list(self.fold_training_study_uids),
            "fold_held_out_study_uids": list(self.fold_held_out_study_uids),
            "eligible_training_report_uids": list(self.eligible_training_report_uids),
            "eligible_evaluation_gold_uids": list(self.eligible_evaluation_gold_uids),
            "training_labels": [
                record.semantic_dict() for record in sorted(
                    self.training_labels, key=lambda item: item.study_instance_uid
                )
            ],
            "evaluation_labels": [
                record.semantic_dict() for record in sorted(
                    self.evaluation_labels, key=lambda item: item.study_instance_uid
                )
            ],
            "missing_training_report_uids": list(self.missing_training_report_uids),
            "missing_evaluation_report_uids": list(self.missing_evaluation_report_uids),
            "excluded_held_out_report_uids": list(self.excluded_held_out_report_uids),
        }

    def to_public_dict(self) -> dict[str, object]:
        """Return counts and metrics without study/report identifiers."""

        return {
            "fold": self.fold,
            "training_label_count": len(self.training_labels),
            "eligible_training_report_count": len(self.eligible_training_report_uids),
            "eligible_evaluation_gold_count": len(self.eligible_evaluation_gold_uids),
            "fold_training_study_count": len(self.fold_training_study_uids),
            "fold_held_out_study_count": len(self.fold_held_out_study_uids),
            "training_supervised_entry_count": self.training_supervised_entry_count,
            "held_out_evaluation_report_count": len(self.evaluation_labels),
            "held_out_evaluation_abstention_entry_count": self.evaluation_abstention_entry_count,
            "missing_training_report_count": len(self.missing_training_report_uids),
            "missing_evaluation_report_count": len(self.missing_evaluation_report_uids),
            "excluded_held_out_report_count": len(self.excluded_held_out_report_uids),
            "evaluation_metrics": self.evaluation_metrics.to_dict(),
            "latency": self.latency.to_dict(),
        }

    def to_private_dict(self) -> dict[str, object]:
        """Return identifier-bearing audit records but never source report text."""

        return {
            "schema_version": REPORT_LABEL_ARTIFACT_SCHEMA_VERSION,
            "fold": self.fold,
            "fold_manifest_checksum": self.fold_manifest_checksum,
            "fold_training_study_uids": list(self.fold_training_study_uids),
            "fold_held_out_study_uids": list(self.fold_held_out_study_uids),
            "eligible_training_report_uids": list(self.eligible_training_report_uids),
            "eligible_evaluation_gold_uids": list(self.eligible_evaluation_gold_uids),
            "partitions": {
                "training_labels": [
                    item.to_private_dict()
                    for item in sorted(self.training_labels, key=lambda value: value.study_instance_uid)
                ],
                "held_out_evaluation": [
                    item.to_private_dict()
                    for item in sorted(self.evaluation_labels, key=lambda value: value.study_instance_uid)
                ],
            },
            "missing_training_report_uids": list(self.missing_training_report_uids),
            "missing_evaluation_report_uids": list(self.missing_evaluation_report_uids),
            "excluded_held_out_report_uids": list(self.excluded_held_out_report_uids),
            "evaluation_metrics": self.evaluation_metrics.to_dict(),
            "latency": self.latency.to_dict(),
        }


def _validate_text_digest(value: str, field_name: str) -> None:
    _require_text(value, field_name)
    _validate_digest(value, field_name)


@dataclass(frozen=True)
class ReportExtractionEvaluation:
    """Aggregate held-out report evaluation for one frozen fold repeat."""

    extractor_version: str
    configuration_fingerprint: str
    ruleset_fingerprint: str
    fold_manifest_checksum: str
    repeat_name: str
    report_source_version: str
    code_fingerprint: str
    code_revision: str
    environment_fingerprint: str
    execution_kind: str
    execution_identity: str
    upstream_input_manifest: Mapping[str, object]
    folds: tuple[FoldReportMetrics, ...]
    pooled_targets: tuple[TargetReportMetrics, ...]
    latency: LatencySummary
    reproducibility_verified: bool
    semantic_digest: str

    def __post_init__(self) -> None:
        if self.extractor_version != REPORT_EXTRACTOR_VERSION:
            raise ReportArtifactError("evaluation uses an unsupported extractor version")
        _validate_text_digest(self.configuration_fingerprint, "configuration_fingerprint")
        _validate_text_digest(self.ruleset_fingerprint, "ruleset_fingerprint")
        _validate_text_digest(self.fold_manifest_checksum, "fold_manifest_checksum")
        _require_text(self.repeat_name, "repeat_name")
        _require_text(self.report_source_version, "report_source_version")
        _validate_text_digest(self.code_fingerprint, "code_fingerprint")
        _require_text(self.code_revision, "code_revision")
        _validate_text_digest(self.environment_fingerprint, "environment_fingerprint")
        if self.execution_kind not in {"local", "slurm"}:
            raise ReportArtifactError("execution_kind must be local or slurm")
        _require_text(self.execution_identity, "execution_identity")
        if not isinstance(self.upstream_input_manifest, Mapping):
            raise ReportArtifactError("upstream_input_manifest must be an object")
        if not isinstance(self.reproducibility_verified, bool):
            raise ReportArtifactError("reproducibility_verified must be boolean")
        if not isinstance(self.latency, LatencySummary):
            raise ReportArtifactError("report evaluation requires a LatencySummary")
        if tuple(item.fold for item in self.folds) != FOLD_IDS:
            raise ReportArtifactError("report evaluation must contain folds 0, 1 and 2 in order")
        if tuple(item.target for item in self.pooled_targets) != TARGET_COLUMNS:
            raise ReportArtifactError("pooled report metrics must use competition target order")
        _validate_text_digest(self.semantic_digest, "semantic_digest")

    @property
    def pooled_report_count(self) -> int:
        return sum(item.report_count for item in self.folds)

    @property
    def pooled_missing_report_count(self) -> int:
        return sum(item.missing_report_count for item in self.folds)

    @property
    def macro_accuracy(self) -> float | None:
        values = [item.accuracy for item in self.pooled_targets]
        return sum(cast(float, value) for value in values) / len(values) if all(
            value is not None for value in values
        ) else None

    @property
    def defined_target_macro_accuracy(self) -> float | None:
        values = [item.accuracy for item in self.pooled_targets if item.accuracy is not None]
        return sum(values) / len(values) if values else None

    @property
    def mean_coverage(self) -> float | None:
        values = [item.coverage for item in self.pooled_targets]
        return sum(cast(float, value) for value in values) / len(values) if all(
            value is not None for value in values
        ) else None

    def to_public_dict(self) -> dict[str, object]:
        """Return the aggregate-only evaluation record safe for public storage."""

        return {
            "schema_version": REPORT_EVALUATION_SCHEMA_VERSION,
            "ticket": "15",
            "status": "completed_synthetic_or_private_local_report_evaluation",
            "extractor": {
                "version": self.extractor_version,
                "configuration_fingerprint": self.configuration_fingerprint,
                "ruleset_fingerprint": self.ruleset_fingerprint,
                "code_fingerprint": self.code_fingerprint,
                "fingerprint": _json_sha256(
                    {
                        "extractor_version": self.extractor_version,
                        "configuration_fingerprint": self.configuration_fingerprint,
                        "ruleset_fingerprint": self.ruleset_fingerprint,
                        "code_fingerprint": self.code_fingerprint,
                    }
                ),
                "frozen_before_held_out_evaluation": True,
                "evaluation_policy": REPORT_EVALUATION_POLICY_VERSION,
                "training_policy": REPORT_TRAINING_POLICY_VERSION,
            },
            "fold_context": {
                "fold_manifest_checksum": self.fold_manifest_checksum,
                "repeat_name": self.repeat_name,
                "report_source_version": self.report_source_version,
                "fold_count": len(self.folds),
            },
            "provenance": {
                "code_revision": self.code_revision,
                "code_fingerprint": self.code_fingerprint,
                "environment_fingerprint": self.environment_fingerprint,
                "execution_kind": self.execution_kind,
                "execution_identity_private": True,
                "upstream_input_manifest": dict(self.upstream_input_manifest),
            },
            "boundaries": {
                "training_labels_are_report_only": True,
                "held_out_gold_reports_are_evaluation_only": True,
                "held_out_reports_excluded_from_training": True,
                "report_derived_rules_cross_fold": False,
                "reference_labels_take_precedence": True,
                "unknown_uncertain_failed_masked": True,
            },
            "privacy": {
                "raw_report_text_persisted": False,
                "source_report_references_private": True,
                "public_study_identifiers": False,
            },
            "counts": {
                "held_out_evaluation_reports": self.pooled_report_count,
                "missing_held_out_gold_reports": self.pooled_missing_report_count,
                "defined_target_accuracy_count": sum(
                    item.accuracy is not None for item in self.pooled_targets
                ),
            },
            "pooled": {
                "macro_accuracy": self.macro_accuracy,
                "defined_target_macro_accuracy": self.defined_target_macro_accuracy,
                "mean_coverage": self.mean_coverage,
                "targets": [item.to_dict() for item in self.pooled_targets],
            },
            "folds": [item.to_dict() for item in self.folds],
            "latency": self.latency.to_dict(),
            "reproducibility": {
                "verified": self.reproducibility_verified,
                "semantic_digest": self.semantic_digest,
                "repeatability_method": "same frozen extractor, manifest and private report digests",
            },
        }

    def to_private_dict(self) -> dict[str, object]:
        """Return the aggregate record plus the private execution identity."""

        payload = self.to_public_dict()
        provenance = cast(dict[str, object], payload["provenance"])
        provenance["execution_identity"] = self.execution_identity
        return payload


@dataclass(frozen=True)
class ReportExtractionRun:
    """Complete ticket-15 result, including private training/evaluation records."""

    evaluation: ReportExtractionEvaluation
    folds: tuple[FoldReportExtraction, ...]
    semantic_digest: str
    report_corpus: ReportCorpus

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation, ReportExtractionEvaluation):
            raise ReportArtifactError("report extraction run requires an evaluation")
        if not isinstance(self.report_corpus, ReportCorpus):
            raise ReportArtifactError("report extraction run requires a ReportCorpus")
        if tuple(item.fold for item in self.folds) != FOLD_IDS:
            raise ReportArtifactError("report extraction run must contain folds 0, 1 and 2 in order")
        if self.evaluation.semantic_digest != self.semantic_digest:
            raise ReportArtifactError("run and evaluation semantic digests disagree")
        if self.evaluation.latency.report_count != sum(item.latency.report_count for item in self.folds):
            raise ReportArtifactError("run and fold latency counts disagree")
        if any(
            item.fold_manifest_checksum != self.evaluation.fold_manifest_checksum
            for item in self.folds
        ):
            raise ReportArtifactError("run and fold manifest checksums disagree")
        if tuple(item.evaluation_metrics for item in self.folds) != self.evaluation.folds:
            raise ReportArtifactError("run and evaluation fold metrics disagree")

    @property
    def schema_version(self) -> str:
        return REPORT_EXTRACTION_SCHEMA_VERSION

    @property
    def reproducibility_verified(self) -> bool:
        return self.evaluation.reproducibility_verified

    def to_public_dict(self) -> dict[str, object]:
        payload = self.evaluation.to_public_dict()
        payload["schema_version"] = REPORT_EXTRACTION_SCHEMA_VERSION
        payload["input"] = self.report_corpus.to_public_dict()
        payload["fold_partitions"] = [item.to_public_dict() for item in self.folds]
        return payload

    def to_private_dict(self) -> dict[str, object]:
        payload = self.evaluation.to_private_dict()
        payload["schema_version"] = REPORT_EXTRACTION_SCHEMA_VERSION
        payload["input"] = self.report_corpus.to_private_dict()
        payload["fold_partitions"] = [item.to_private_dict() for item in self.folds]
        return payload


def _validate_manifest_for_report_extraction(manifest: FoldManifest) -> None:
    if not isinstance(manifest, FoldManifest):
        raise ReportSourceError("report extraction requires a FoldManifest")
    if manifest.superseded:
        raise ReportSourceError("superseded fold manifests cannot admit report-derived evidence")
    if manifest.fold_ids != FOLD_IDS:
        raise ReportSourceError("report extraction requires the canonical three-fold manifest")


def _membership_map(manifest: FoldManifest) -> dict[str, FoldMembership]:
    return {item.study_instance_uid: item for item in manifest.memberships}


def validate_fold_report_boundaries(
    manifest: FoldManifest,
    fold: int,
    *,
    training_study_uids: Sequence[str],
    evaluation_study_uids: Sequence[str],
) -> None:
    """Validate training/evaluation report IDs against one frozen fold.

    Training IDs may be a subset of the fold's training partition. Evaluation
    IDs may be a subset of held-out gold IDs because reports can be unavailable.
    The two sets are always disjoint, and every held-out report is excluded
    from training even when it is not itself an evaluation record.
    """

    _validate_manifest_for_report_extraction(manifest)
    _validate_fold(fold)
    training = set(_validate_unique_strings(training_study_uids, "training report studies"))
    evaluation = set(_validate_unique_strings(evaluation_study_uids, "evaluation report studies"))
    if training.intersection(evaluation):
        raise ReportExtractionError("training and evaluation report studies overlap")
    membership = _membership_map(manifest)
    unknown = sorted((training | evaluation) - set(membership))
    if unknown:
        raise ReportSourceError(f"report boundaries contain unknown studies: {unknown[:3]}")
    held_out = set(manifest.held_out_study_uids(fold))
    if training.intersection(held_out):
        raise ReportExtractionError(
            f"fold {fold} training report inputs contain held-out studies: "
            f"{sorted(training.intersection(held_out))[:3]}"
        )
    non_report_only_training = sorted(
        uid for uid in training if membership[uid].cohort != "report_only"
    )
    if non_report_only_training:
        raise ReportExtractionError(
            f"fold {fold} training report labels must be report-only studies: "
            f"{non_report_only_training[:3]}"
        )
    non_gold_evaluation = sorted(
        uid for uid in evaluation if membership[uid].cohort != "gold"
    )
    if non_gold_evaluation:
        raise ReportExtractionError(
            f"fold {fold} evaluation reports are not gold studies: {non_gold_evaluation[:3]}"
        )
    wrong_fold_evaluation = sorted(evaluation - held_out)
    if wrong_fold_evaluation:
        raise ReportExtractionError(
            f"fold {fold} evaluation reports are not held out: {wrong_fold_evaluation[:3]}"
        )
    manifest.assert_adaptation_input_is_training_only(fold, tuple(training))


def _target_metrics(
    records: Sequence[ReportLabelRecord],
    manifest: FoldManifest,
) -> tuple[TargetReportMetrics, ...]:
    membership = _membership_map(manifest)
    metrics: list[TargetReportMetrics] = []
    for index, target in enumerate(TARGET_COLUMNS):
        gold_positive = 0
        gold_negative = 0
        predicted_positive = 0
        predicted_negative = 0
        correct = 0
        not_mentioned = 0
        uncertain = 0
        unknown = 0
        extraction_failed = 0
        confidence_values: list[float] = []
        for record in records:
            item = membership.get(record.study_instance_uid)
            if item is None:
                raise ReportSourceError(
                    f"report evaluation references an unknown study: {record.study_instance_uid}"
                )
            reference = item.reference_labels[index]
            if reference == 1:
                gold_positive += 1
            elif reference == 0:
                gold_negative += 1
            else:
                raise ReportSourceError("report evaluation requires complete gold reference labels")
            label = record.interpretation.labels[index]
            if label.state is LabelState.POSITIVE:
                predicted_positive += 1
                confidence_values.append(float(cast(float, label.confidence)))
                correct += label.value == reference
            elif label.state is LabelState.NEGATIVE:
                predicted_negative += 1
                confidence_values.append(float(cast(float, label.confidence)))
                correct += label.value == reference
            elif label.state is LabelState.NOT_MENTIONED:
                not_mentioned += 1
            elif label.state is LabelState.UNCERTAIN:
                uncertain += 1
            elif label.state is LabelState.UNKNOWN:
                unknown += 1
            elif label.state is LabelState.EXTRACTION_FAILED:
                extraction_failed += 1
            else:
                raise ReportArtifactError(f"unsupported report label state for {target}")
        report_count = len(records)
        confident = predicted_positive + predicted_negative
        metrics.append(
            TargetReportMetrics(
                target=target,
                report_count=report_count,
                gold_positive_count=gold_positive,
                gold_negative_count=gold_negative,
                predicted_positive_count=predicted_positive,
                predicted_negative_count=predicted_negative,
                correct_count=correct,
                accuracy=correct / confident if confident else None,
                coverage=confident / report_count if report_count else None,
                not_mentioned_count=not_mentioned,
                uncertain_count=uncertain,
                unknown_count=unknown,
                extraction_failed_count=extraction_failed,
                abstention_count=report_count - confident,
                failure_count=extraction_failed,
                mean_confidence=sum(confidence_values) / len(confidence_values)
                if confidence_values
                else None,
            )
        )
    return tuple(metrics)


def evaluate_fold_report_labels(
    manifest: FoldManifest,
    fold: int,
    records: Sequence[ReportLabelRecord],
    *,
    missing_report_count: int = 0,
) -> FoldReportMetrics:
    """Evaluate only the declared held-out gold report records for one fold."""

    _validate_manifest_for_report_extraction(manifest)
    _validate_fold(fold)
    _metric_count(missing_report_count, "missing_report_count")
    record_uids: list[str] = []
    for record in records:
        if record.fold != fold or record.partition is not ReportPartition.HELD_OUT_EVALUATION:
            raise ReportExtractionError("fold evaluation received a non-evaluation report record")
        record_uids.append(record.study_instance_uid)
    expected = set(
        item.study_instance_uid
        for item in manifest.gold_memberships
        if item.fold == fold
    )
    if not set(record_uids).issubset(expected) or len(set(record_uids)) != len(record_uids):
        raise ReportExtractionError("fold evaluation reports do not match held-out gold membership")
    metrics = _target_metrics(records, manifest)
    return FoldReportMetrics(
        fold=fold,
        report_count=len(records),
        missing_report_count=missing_report_count,
        targets=metrics,
    )


def _build_fold_report_extraction(
    manifest: FoldManifest,
    corpus: ReportCorpus,
    fold: int,
    extractor: FrozenDeterministicExtractor,
) -> FoldReportExtraction:
    _validate_fold(fold)
    records_by_study = corpus.by_study
    held_out = set(manifest.held_out_study_uids(fold))
    expected_training = {
        item.study_instance_uid
        for item in manifest.report_only_memberships
        if item.fold != fold
    }
    expected_evaluation = {
        item.study_instance_uid
        for item in manifest.gold_memberships
        if item.fold == fold
    }
    training_uids = tuple(sorted(expected_training.intersection(records_by_study)))
    evaluation_uids = tuple(sorted(expected_evaluation.intersection(records_by_study)))
    validate_fold_report_boundaries(
        manifest,
        fold,
        training_study_uids=training_uids,
        evaluation_study_uids=evaluation_uids,
    )
    training_labels: list[ReportLabelRecord] = []
    evaluation_labels: list[ReportLabelRecord] = []
    durations: list[int] = []
    for uid in training_uids:
        started = time.perf_counter_ns()
        interpretation = extractor.extract(records_by_study[uid], fold=fold)
        durations.append(time.perf_counter_ns() - started)
        training_labels.append(
            ReportLabelRecord(uid, fold, ReportPartition.TRAINING_LABELS, interpretation)
        )
    for uid in evaluation_uids:
        started = time.perf_counter_ns()
        interpretation = extractor.extract(records_by_study[uid], fold=fold)
        durations.append(time.perf_counter_ns() - started)
        evaluation_labels.append(
            ReportLabelRecord(uid, fold, ReportPartition.HELD_OUT_EVALUATION, interpretation)
        )
    excluded_held_out = tuple(sorted(held_out.intersection(records_by_study)))
    missing_training = tuple(sorted(expected_training - set(training_uids)))
    missing_evaluation = tuple(sorted(expected_evaluation - set(evaluation_uids)))
    metrics = evaluate_fold_report_labels(
        manifest,
        fold,
        tuple(evaluation_labels),
        missing_report_count=len(missing_evaluation),
    )
    result = FoldReportExtraction(
        fold=fold,
        fold_manifest_checksum=manifest.manifest_checksum,
        fold_training_study_uids=tuple(sorted(manifest.training_study_uids(fold))),
        fold_held_out_study_uids=tuple(sorted(manifest.held_out_study_uids(fold))),
        eligible_training_report_uids=tuple(sorted(expected_training)),
        eligible_evaluation_gold_uids=tuple(sorted(expected_evaluation)),
        training_labels=tuple(training_labels),
        evaluation_labels=tuple(evaluation_labels),
        missing_training_report_uids=missing_training,
        missing_evaluation_report_uids=missing_evaluation,
        excluded_held_out_report_uids=excluded_held_out,
        evaluation_metrics=metrics,
        latency_samples_ns=tuple(durations),
        latency=LatencySummary.from_durations(durations),
    )
    if any(uid in held_out for uid in (item.study_instance_uid for item in result.training_labels)):
        raise ReportArtifactError("held-out report was admitted to training labels")
    return result


def build_fold_report_extraction(
    manifest: FoldManifest,
    corpus: ReportCorpus,
    fold: int,
    *,
    extractor: FrozenDeterministicExtractor | None = None,
) -> FoldReportExtraction:
    """Build one clean fold's training-label and held-out-evaluation partitions."""

    _validate_manifest_for_report_extraction(manifest)
    if not isinstance(corpus, ReportCorpus):
        raise ReportSourceError("report extraction requires a ReportCorpus")
    _validate_fold(fold)
    unknown = sorted(set(corpus.by_study) - set(manifest.study_to_fold))
    if unknown:
        raise ReportSourceError(f"report corpus contains unknown studies: {unknown[:3]}")
    selected_extractor = extractor or freeze_deterministic_extractor()
    if not isinstance(selected_extractor, FrozenDeterministicExtractor) or not selected_extractor.is_frozen:
        raise ReportExtractionError("held-out report extraction requires a frozen deterministic extractor")
    return _build_fold_report_extraction(manifest, corpus, fold, selected_extractor)


def _upstream_input_manifest(
    manifest: FoldManifest,
    corpus: ReportCorpus,
    extractor: FrozenDeterministicExtractor,
) -> dict[str, object]:
    membership_digest = _json_sha256(
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
    return {
        "fold_manifest": {
            "schema_version": FOLD_SCHEMA_VERSION,
            "manifest_checksum": manifest.manifest_checksum,
            "repeat_name": manifest.repeat_name,
            "eligible_study_set_sha256": membership_digest,
        },
        "report_corpus": {
            "schema_version": REPORT_CORPUS_SCHEMA_VERSION,
            "source_version": corpus.source_version,
            "record_set_sha256": corpus.record_set_sha256,
        },
        "extractor": {
            "version": extractor.extractor_version,
            "code_fingerprint": REPORT_EXTRACTION_CODE_FINGERPRINT,
            "configuration_fingerprint": extractor.configuration_fingerprint,
            "ruleset_fingerprint": extractor.ruleset_fingerprint,
            "checkpoint_hashes": [],
            "upstream_adaptation_inputs": [],
            "report_derived_rule_development": False,
        },
        "held_out_adaptation_exclusions": list(ADAPTATION_EXCLUSIONS),
    }


def _build_run_once(
    manifest: FoldManifest,
    corpus: ReportCorpus,
    extractor: FrozenDeterministicExtractor,
) -> tuple[tuple[FoldReportExtraction, ...], LatencySummary, str]:
    folds = tuple(
        build_fold_report_extraction(manifest, corpus, fold, extractor=extractor)
        for fold in FOLD_IDS
    )
    semantic_payload = {
        "fold_manifest_checksum": manifest.manifest_checksum,
        "repeat_name": manifest.repeat_name,
        "report_source_version": corpus.source_version,
        "extractor": extractor.to_public_dict(),
        "folds": [item.semantic_dict() for item in folds],
    }
    return (
        folds,
        LatencySummary.from_durations(
            tuple(sample for item in folds for sample in item.latency_samples_ns)
        ),
        _json_sha256(semantic_payload),
    )


def run_report_extraction(
    manifest: FoldManifest,
    corpus: ReportCorpus,
    *,
    extractor: FrozenDeterministicExtractor | None = None,
    verify_reproducibility: bool = True,
) -> ReportExtractionRun:
    """Generate fold-specific labels and evaluate frozen held-out gold reports.

    The optional repeatability pass reruns the exact same immutable inputs and
    compares text-free semantic artifact hashes.  Timing is reported from the
    first pass only because wall-clock measurements are not deterministic.
    """

    _validate_manifest_for_report_extraction(manifest)
    if not isinstance(corpus, ReportCorpus):
        raise ReportSourceError("report extraction requires a ReportCorpus")
    selected_extractor = extractor or freeze_deterministic_extractor()
    if not isinstance(selected_extractor, FrozenDeterministicExtractor) or not selected_extractor.is_frozen:
        raise ReportExtractionError("held-out report extraction requires a frozen deterministic extractor")
    unknown = sorted(set(corpus.by_study) - set(manifest.study_to_fold))
    if unknown:
        raise ReportSourceError(f"report corpus contains unknown studies: {unknown[:3]}")
    folds, latency, semantic_digest = _build_run_once(manifest, corpus, selected_extractor)
    reproducibility_verified = False
    if verify_reproducibility:
        _, _, repeated_digest = _build_run_once(manifest, corpus, selected_extractor)
        if repeated_digest != semantic_digest:
            raise ReportArtifactError("deterministic report extraction changed across repeatability runs")
        reproducibility_verified = True
    pooled_records = tuple(
        record
        for fold_result in folds
        for record in fold_result.evaluation_labels
    )
    pooled_uids = [record.study_instance_uid for record in pooled_records]
    if len(set(pooled_uids)) != len(pooled_uids):
        raise ReportArtifactError("a gold report received more than one held-out evaluation record")
    pooled_targets = _target_metrics(pooled_records, manifest)
    execution_kind, execution_identity = _execution_identity()
    evaluation = ReportExtractionEvaluation(
        extractor_version=selected_extractor.extractor_version,
        configuration_fingerprint=selected_extractor.configuration_fingerprint,
        ruleset_fingerprint=selected_extractor.ruleset_fingerprint,
        fold_manifest_checksum=manifest.manifest_checksum,
        repeat_name=manifest.repeat_name,
        report_source_version=corpus.source_version,
        code_fingerprint=REPORT_EXTRACTION_CODE_FINGERPRINT,
        code_revision=_resolve_code_revision(),
        environment_fingerprint=_environment_fingerprint(),
        execution_kind=execution_kind,
        execution_identity=execution_identity,
        upstream_input_manifest=_upstream_input_manifest(
            manifest,
            corpus,
            selected_extractor,
        ),
        folds=tuple(item.evaluation_metrics for item in folds),
        pooled_targets=pooled_targets,
        latency=latency,
        reproducibility_verified=reproducibility_verified,
        semantic_digest=semantic_digest,
    )
    return ReportExtractionRun(
        evaluation=evaluation,
        folds=folds,
        semantic_digest=semantic_digest,
        report_corpus=corpus,
    )


def write_report_extraction_reports(
    result: ReportExtractionRun,
    *,
    public_path: str | Path,
    private_path: str | Path,
) -> tuple[Path, Path]:
    """Atomically write distinct public aggregate and private audit artifacts."""

    if not isinstance(result, ReportExtractionRun):
        raise ReportArtifactError("report writer requires a ReportExtractionRun")
    public = Path(public_path)
    private = Path(private_path)
    if public.resolve() == private.resolve():
        raise ReportArtifactError("public and private report destinations must be distinct")
    atomic_json_write(public, result.to_public_dict())
    try:
        atomic_json_write(private, result.to_private_dict())
    except BaseException:
        try:
            public.unlink()
        except FileNotFoundError:
            pass
        raise
    return public, private


def _check_execution_context(*, require_slurm: bool, allow_local_smoke: bool) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if require_slurm and (not job_id or partition != "cluster1-cpu"):
        raise RuntimeError("ticket-15 report extraction must run in a cluster1-cpu Slurm allocation")
    if not job_id and not allow_local_smoke:
        raise RuntimeError(
            "refusing report extraction outside Slurm; use --allow-local-smoke only with a tiny fixture"
        )


def _check_local_smoke_size(manifest: FoldManifest, corpus: ReportCorpus) -> None:
    if len(manifest.memberships) > LOCAL_SMOKE_MAX_STUDIES:
        raise RuntimeError(
            "local smoke is limited to "
            f"{LOCAL_SMOKE_MAX_STUDIES} studies; use Slurm for the supplied fold manifest"
        )
    if corpus.report_count > LOCAL_SMOKE_MAX_REPORTS:
        raise RuntimeError(
            "local smoke is limited to "
            f"{LOCAL_SMOKE_MAX_REPORTS} reports; use Slurm for the supplied report corpus"
        )


def _parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("--allow-local-smoke", action="store_true")
    parser.add_argument("--skip-reproducibility-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run ticket-15 extraction from private reports and a private fold manifest."""

    from .folds import load_fold_manifest

    args = _parser().parse_args(argv)
    try:
        _check_execution_context(
            require_slurm=args.require_slurm,
            allow_local_smoke=args.allow_local_smoke,
        )
        manifest = load_fold_manifest(args.fold_manifest)
        corpus = load_report_corpus(args.reports)
        if args.allow_local_smoke and not os.environ.get("SLURM_JOB_ID"):
            _check_local_smoke_size(manifest, corpus)
        result = run_report_extraction(
            manifest,
            corpus,
            verify_reproducibility=not args.skip_reproducibility_check,
        )
        write_report_extraction_reports(
            result,
            public_path=args.public_output,
            private_path=args.private_output,
        )
    except (OSError, RuntimeError, ReportExtractionError, ValueError) as exc:
        print(f"ticket-15 report extraction failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_public_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FrozenDeterministicExtractor",
    "FoldReportExtraction",
    "FoldReportMetrics",
    "LatencySummary",
    "REPORT_CORPUS_SCHEMA_VERSION",
    "REPORT_EVALUATION_POLICY_VERSION",
    "REPORT_EVALUATION_SCHEMA_VERSION",
    "REPORT_EXTRACTION_SCHEMA_VERSION",
    "REPORT_EXTRACTION_CODE_FINGERPRINT",
    "REPORT_LABEL_ARTIFACT_SCHEMA_VERSION",
    "REPORT_RULESET_FINGERPRINT",
    "REPORT_TRAINING_POLICY_VERSION",
    "ReportArtifactError",
    "ReportCorpus",
    "ReportExtractionError",
    "ReportExtractionEvaluation",
    "ReportExtractionRun",
    "ReportLabelRecord",
    "ReportPartition",
    "ReportRecord",
    "ReportSourceError",
    "TargetReportMetrics",
    "build_fold_report_extraction",
    "evaluate_fold_report_labels",
    "freeze_deterministic_extractor",
    "load_report_corpus",
    "main",
    "run_report_extraction",
    "validate_fold_report_boundaries",
    "write_report_extraction_reports",
]
