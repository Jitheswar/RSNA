"""Conservative, auditable report-to-target interpretation primitives.

This module is deliberately narrower than a clinical NLP system.  It records
the competition authority used to define each target and recognizes only a
small, explicit English vocabulary suitable for public synthetic tests.  A
real report mention that cannot be mapped to an authoritative criterion is
kept unknown and never becomes training supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import re
from typing import Iterable, Sequence

from .targets import TARGET_COLUMNS


REPORT_INTERPRETATION_SCHEMA_VERSION = "rsna-knee-report-interpretation-v1"
REPORT_EXTRACTOR_VERSION = "deterministic-synthetic-english-v1"
AUTHORITY_RETRIEVED_ON = "2026-09-14"

DATA_DESCRIPTION_URL = "https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data"
LABEL_DESCRIPTION_URL = (
    "https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343"
)
HOST_CLARIFICATION_URL = (
    "https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826"
)
PARENT_SPEC_PATH = ".scratch/rsna-2026-fresh/spec.md"


class ReportInterpretationError(ValueError):
    """Raised when report-label provenance or shape cannot be trusted."""


def _validate_fold_ids(folds: Iterable[int]) -> tuple[int, ...]:
    values = tuple(folds)
    if any(isinstance(fold, bool) or not isinstance(fold, int) or fold < 0 for fold in values):
        raise ReportInterpretationError("fold ids must be non-negative integers")
    if len(set(values)) != len(values):
        raise ReportInterpretationError("fold ids must be unique")
    return values


class LabelState(StrEnum):
    """The preserved outcome of one target's report interpretation."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NOT_MENTIONED = "not-mentioned"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"
    EXTRACTION_FAILED = "extraction-failed"


class RuleOriginKind(StrEnum):
    """Where an interpretation rule came from."""

    OFFICIAL_DEFINITION = "official-competition-definition"
    PROJECT_SAFETY_POLICY = "project-safety-policy"
    INDEPENDENT_SYNTHETIC = "independent-synthetic-case"
    PATIENT_REPORT = "patient-report-example"
    PARTICIPANT_INTERPRETATION = "participant-interpretation"


@dataclass(frozen=True)
class AuthoritySource:
    """A dated public source supporting a target interpretation."""

    source_id: str
    title: str
    url: str
    publisher: str
    retrieved_on: str
    role: str

    def __post_init__(self) -> None:
        for name, value in (
            ("source_id", self.source_id),
            ("title", self.title),
            ("url", self.url),
            ("publisher", self.publisher),
            ("retrieved_on", self.retrieved_on),
            ("role", self.role),
        ):
            if not value.strip():
                raise ReportInterpretationError(f"authority source {name} must be non-empty")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.retrieved_on) is None:
            raise ReportInterpretationError("authority source retrieved_on must be YYYY-MM-DD")

    def to_dict(self) -> dict[str, str]:
        """Return identifier-free authority metadata for an audit register."""

        return {
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "publisher": self.publisher,
            "retrieved_on": self.retrieved_on,
            "role": self.role,
        }


@dataclass(frozen=True)
class RuleOrigin:
    """Provenance and fold scope for one report interpretation rule.

    A report-derived origin is allowed only when it is explicitly fold-scoped.
    It can never be silently promoted to a shared rule.  ``source_report_refs``
    are private provenance and are intentionally excluded from public output.
    """

    rule_id: str
    kind: RuleOriginKind
    source_id: str
    source_url: str
    retrieved_on: str
    derived_from_report: bool = False
    participant_interpretation: bool = False
    shared_across_folds: bool = True
    allowed_folds: tuple[int, ...] | None = None
    source_report_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.rule_id.strip():
            raise ReportInterpretationError("rule origin rule_id must be non-empty")
        if not self.source_id.strip() or not self.source_url.strip():
            raise ReportInterpretationError("rule origin source metadata must be non-empty")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.retrieved_on) is None:
            raise ReportInterpretationError("rule origin retrieved_on must be YYYY-MM-DD")
        _validate_fold_ids(self.allowed_folds or ())
        if len(set(self.source_report_refs)) != len(self.source_report_refs):
            raise ReportInterpretationError("rule origin source_report_refs must be unique")

    def to_public_dict(self) -> dict[str, object]:
        """Return rule provenance without report or study references."""

        return {
            "rule_id": self.rule_id,
            "kind": self.kind.value,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "retrieved_on": self.retrieved_on,
            "derived_from_report": self.derived_from_report,
            "participant_interpretation": self.participant_interpretation,
            "shared_across_folds": self.shared_across_folds,
            "allowed_folds": list(self.allowed_folds) if self.allowed_folds is not None else None,
        }

    def to_private_dict(self) -> dict[str, object]:
        """Return provenance including private report references."""

        result = self.to_public_dict()
        result["source_report_refs"] = list(self.source_report_refs)
        return result


@dataclass(frozen=True)
class PatternRule:
    """One public, human-auditable phrase rule used by a target definition."""

    rule_id: str
    pattern: str
    rationale: str

    def __post_init__(self) -> None:
        if not self.rule_id.strip() or not self.pattern.strip() or not self.rationale.strip():
            raise ReportInterpretationError("pattern rules require id, pattern and rationale")
        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise ReportInterpretationError(f"invalid report pattern {self.rule_id!r}") from exc


@dataclass(frozen=True)
class TargetDefinition:
    """Authoritative target meaning plus the deliberately narrow rule surface."""

    target: str
    official_description: str
    positive_criterion: str
    negative_criterion: str
    sources: tuple[AuthoritySource, ...]
    rule_origin: RuleOrigin
    mention_pattern: str
    positive_patterns: tuple[PatternRule, ...]
    negative_patterns: tuple[PatternRule, ...]
    unresolved_interpretations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.target not in TARGET_COLUMNS:
            raise ReportInterpretationError(f"unknown competition target {self.target!r}")
        for name, value in (
            ("official_description", self.official_description),
            ("positive_criterion", self.positive_criterion),
            ("negative_criterion", self.negative_criterion),
            ("mention_pattern", self.mention_pattern),
        ):
            if not value.strip():
                raise ReportInterpretationError(f"{self.target} {name} must be non-empty")
        if not self.sources:
            raise ReportInterpretationError(f"{self.target} must have an authority source")
        if self.rule_origin.kind is not RuleOriginKind.OFFICIAL_DEFINITION:
            raise ReportInterpretationError(f"{self.target} must use an official definition origin")
        try:
            re.compile(self.mention_pattern)
        except re.error as exc:
            raise ReportInterpretationError(f"invalid mention pattern for {self.target!r}") from exc

    def to_dict(self) -> dict[str, object]:
        """Return the public target authority and rule register."""

        return {
            "target": self.target,
            "official_description": self.official_description,
            "positive_criterion": self.positive_criterion,
            "negative_criterion": self.negative_criterion,
            "sources": [source.to_dict() for source in self.sources],
            "rule_origin": self.rule_origin.to_public_dict(),
            "mention_pattern": self.mention_pattern,
            "positive_rules": [
                {"rule_id": rule.rule_id, "rationale": rule.rationale}
                for rule in self.positive_patterns
            ],
            "negative_rules": [
                {"rule_id": rule.rule_id, "rationale": rule.rationale}
                for rule in self.negative_patterns
            ],
            "unresolved_interpretations": list(self.unresolved_interpretations),
        }


@dataclass(frozen=True)
class TargetLabel:
    """One target state, its safe loss projection and provenance."""

    target: str
    state: LabelState
    rule_id: str
    rule_origin: RuleOrigin
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.target not in TARGET_COLUMNS:
            raise ReportInterpretationError(f"unknown target label {self.target!r}")
        if not self.rule_id.strip():
            raise ReportInterpretationError("target label rule_id must be non-empty")
        if any("\n" in item or "\r" in item for item in self.evidence):
            raise ReportInterpretationError("target label evidence must be single-line snippets")

    @property
    def value(self) -> int | None:
        """Return a binary target only when the state is confidently supervisable."""

        if self.state is LabelState.POSITIVE:
            return 1
        if self.state is LabelState.NEGATIVE:
            return 0
        return None

    @property
    def supervision_mask(self) -> int:
        """Return one only for explicit positive or negative evidence."""

        return int(self.value is not None)

    def to_public_dict(self) -> dict[str, object]:
        """Return safe aggregate label evidence without report snippets."""

        return {
            "target": self.target,
            "state": self.state.value,
            "value": self.value,
            "supervision_mask": self.supervision_mask,
            "rule_id": self.rule_id,
            "rule_origin": self.rule_origin.to_public_dict(),
        }

    def to_private_dict(self) -> dict[str, object]:
        """Return local audit evidence, including matched synthetic/report phrases."""

        result = self.to_public_dict()
        result["evidence"] = list(self.evidence)
        result["rule_origin"] = self.rule_origin.to_private_dict()
        return result


@dataclass(frozen=True)
class ReportInterpretation:
    """A complete twelve-target interpretation with no raw report retention."""

    extractor_version: str
    report_sha256: str
    labels: tuple[TargetLabel, ...]
    source_report_ref: str | None = None
    consuming_folds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.extractor_version != REPORT_EXTRACTOR_VERSION:
            raise ReportInterpretationError("unsupported report extractor version")
        if re.fullmatch(r"[0-9a-f]{64}", self.report_sha256) is None:
            raise ReportInterpretationError("report_sha256 must be a lowercase SHA-256 digest")
        if tuple(item.target for item in self.labels) != TARGET_COLUMNS:
            raise ReportInterpretationError("interpretation labels must follow competition target order")
        _validate_fold_ids(self.consuming_folds)
        for label in self.labels:
            validate_rule_origin_for_folds(label.rule_origin, self.consuming_folds)
        if self.source_report_ref is not None and not self.source_report_ref.strip():
            raise ReportInterpretationError("source_report_ref must be non-empty when supplied")

    @property
    def values(self) -> tuple[int | None, ...]:
        """Return the binary vector with unknown states represented as ``None``."""

        return tuple(item.value for item in self.labels)

    @property
    def supervision_mask(self) -> tuple[int, ...]:
        """Return the zero-loss mask for unknown, uncertain and failed states."""

        return tuple(item.supervision_mask for item in self.labels)

    def to_public_dict(self) -> dict[str, object]:
        """Return identifier-free state and mask evidence."""

        return {
            "schema_version": REPORT_INTERPRETATION_SCHEMA_VERSION,
            "extractor_version": self.extractor_version,
            "target_order": list(TARGET_COLUMNS),
            "states": [item.state.value for item in self.labels],
            "values": list(self.values),
            "supervision_mask": list(self.supervision_mask),
            "rule_ids": [item.rule_id for item in self.labels],
            "consuming_folds": list(self.consuming_folds),
        }

    def to_private_dict(self) -> dict[str, object]:
        """Return local audit evidence without retaining the raw report."""

        result = self.to_public_dict()
        result["report_sha256"] = self.report_sha256
        result["source_report_ref"] = self.source_report_ref
        result["labels"] = [item.to_private_dict() for item in self.labels]
        return result


@dataclass(frozen=True)
class _TargetEvent:
    state: LabelState
    rule_id: str
    evidence: str


_DATA_SOURCE = AuthoritySource(
    source_id="kaggle-data-description-2026-09-14",
    title="RSNA Knee Abnormality Detection — Data",
    url=DATA_DESCRIPTION_URL,
    publisher="Kaggle / RSNA competition host",
    retrieved_on=AUTHORITY_RETRIEVED_ON,
    role="official target names and dataset label meanings",
)
_LABEL_SOURCE = AuthoritySource(
    source_id="kaggle-host-label-description-2026-09-14",
    title="Knee Abnormality Detection AI Challenge Overview — Label Description",
    url=LABEL_DESCRIPTION_URL,
    publisher="RSNA competition host",
    retrieved_on=AUTHORITY_RETRIEVED_ON,
    role="official annotator criteria and borderline-negative policy",
)
_HOST_CLARIFICATION = AuthoritySource(
    source_id="kaggle-host-report-clarification-2026-09-14",
    title="Possible inconsistencies between MRI reports and provided labels",
    url=HOST_CLARIFICATION_URL,
    publisher="RSNA competition host",
    retrieved_on=AUTHORITY_RETRIEVED_ON,
    role="image-label authority and report-disagreement clarification",
)
_OFFICIAL_ORIGIN = RuleOrigin(
    rule_id="official-label-description-v1",
    kind=RuleOriginKind.OFFICIAL_DEFINITION,
    source_id=_LABEL_SOURCE.source_id,
    source_url=_LABEL_SOURCE.url,
    retrieved_on=AUTHORITY_RETRIEVED_ON,
)
_POLICY_ORIGIN = RuleOrigin(
    rule_id="report-unknown-and-mask-policy-v1",
    kind=RuleOriginKind.PROJECT_SAFETY_POLICY,
    source_id="parent-spec-report-supervision-policy-2026-09-14",
    source_url=PARENT_SPEC_PATH,
    retrieved_on=AUTHORITY_RETRIEVED_ON,
)


def _rule(rule_id: str, pattern: str, rationale: str) -> PatternRule:
    return PatternRule(rule_id=rule_id, pattern=pattern, rationale=rationale)


def _definition(
    target: str,
    official_description: str,
    positive_criterion: str,
    negative_criterion: str,
    mention_pattern: str,
    positive_patterns: tuple[PatternRule, ...],
    negative_patterns: tuple[PatternRule, ...],
    unresolved_interpretations: tuple[str, ...],
) -> TargetDefinition:
    return TargetDefinition(
        target=target,
        official_description=official_description,
        positive_criterion=positive_criterion,
        negative_criterion=negative_criterion,
        sources=(_DATA_SOURCE, _LABEL_SOURCE, _HOST_CLARIFICATION),
        rule_origin=_OFFICIAL_ORIGIN,
        mention_pattern=mention_pattern,
        positive_patterns=positive_patterns,
        negative_patterns=negative_patterns,
        unresolved_interpretations=unresolved_interpretations,
    )


_TARGET_DEFINITIONS: tuple[TargetDefinition, ...] = (
    _definition(
        "ACL",
        "Anterior cruciate ligament injury, operationalized by the host as a high-grade tear.",
        "High-grade partial or full-thickness ACL tear: discontinuity or more than half of fibers disrupted.",
        "Mild signal change, degeneration, thickening, borderline wording or a low-grade change is negative.",
        r"\b(?:acl|anterior cruciate ligament)\b",
        (
            _rule(
                "acl-high-grade-tear-v1",
                r"\b(?:high[- ]grade\s+)?(?:partial|complete|full[- ]thickness)\s+tear\s+of\s+(?:the\s+)?(?:acl|anterior cruciate ligament)\b",
                "requires a direct tear phrase and the host's high-grade/full-thickness criterion",
            ),
            _rule(
                "acl-discontinuity-v1",
                r"\b(?:acl|anterior cruciate ligament)\s+(?:is\s+)?(?:completely\s+)?(?:torn|ruptured|discontinuous)\b",
                "direct discontinuity or rupture language maps to the host criterion",
            ),
        ),
        (
            _rule(
                "acl-low-grade-change-v1",
                r"\b(?:mild|low[- ]grade)\s+(?:acl|anterior cruciate ligament)\b.{0,40}\b(?:signal|degeneration|thickening|sprain|injury)\b",
                "the host explicitly grades mild signal, degeneration and thickening negative",
            ),
        ),
        (
            "partial tear without high-grade/discontinuity wording",
            "isolated ACL injury or instability without the defined tear criterion",
        ),
    ),
    _definition(
        "MCL",
        "Medial collateral ligament injury, operationalized by the host as a high-grade acute tear.",
        "High-grade partial or complete acute MCL tear with disrupted fibers and adjacent edema.",
        "Low-grade sprain, chronic change, remote injury, borderline wording or isolated edema is negative.",
        r"\b(?:mcl|medial collateral ligament)\b",
        (
            _rule(
                "mcl-high-grade-tear-v1",
                r"\b(?:high[- ]grade\s+)?(?:partial|complete)\s+(?:acute\s+)?tear\s+of\s+(?:the\s+)?(?:mcl|medial collateral ligament)\b",
                "requires a direct acute/high-grade tear phrase",
            ),
            _rule(
                "mcl-fiber-disruption-v1",
                r"\b(?:mcl|medial collateral ligament)\b.{0,45}\b(?:fibers?|fibre)\s+(?:are\s+)?(?:disrupted|torn|ruptured)\b",
                "fiber disruption is part of the host's high-grade tear criterion",
            ),
        ),
        (
            _rule(
                "mcl-low-grade-or-chronic-v1",
                r"\b(?:low[- ]grade|mild|chronic|remote)\b.{0,45}\b(?:mcl|medial collateral ligament)\b",
                "the host grades low-grade sprains and chronic/remote changes negative",
            ),
        ),
        (
            "MCL injury without tear grade or acute fiber-disruption wording",
            "isolated ligament edema or a nonspecific sprain",
        ),
    ),
    _definition(
        "Medial Meniscus",
        "Medial meniscus tear.",
        "Definite surface-contacting abnormal signal on at least two images or a qualifying displaced/truncated fragment.",
        "Intrasubstance degeneration that does not reach the surface, borderline wording or no tear is negative.",
        r"\bmedial menisc(?:us|al)\b",
        (
            _rule(
                "medial-meniscus-tear-v1",
                r"\bmedial menisc(?:us|al)\b.{0,65}\btear\b",
                "a direct medial-meniscus tear is the supported report phrase",
            ),
            _rule(
                "medial-meniscus-surface-signal-v1",
                r"\bmedial menisc(?:us|al)\b.{0,65}\b(?:signal|line)\b.{0,35}\b(?:reaches?|contacts?)\s+(?:the\s+)?surface\b",
                "surface-contacting signal matches the host criterion",
            ),
        ),
        (
            _rule(
                "medial-meniscus-intrasubstance-v1",
                r"\bmedial menisc(?:us|al)\b.{0,75}\bintrasubstance degeneration\b.{0,45}\b(?:does not reach|not reaching)\s+(?:the\s+)?surface\b",
                "the host explicitly excludes intrasubstance degeneration without surface contact",
            ),
        ),
        (
            "tear grade or surface contact not stated",
            "degenerative signal without an explicit surface-contacting tear",
        ),
    ),
    _definition(
        "Lateral Meniscus",
        "Lateral meniscus tear.",
        "Definite surface-contacting abnormal signal on at least two images or a qualifying displaced/truncated fragment.",
        "Intrasubstance degeneration that does not reach the surface, borderline wording or no tear is negative.",
        r"\blateral menisc(?:us|al)\b",
        (
            _rule(
                "lateral-meniscus-tear-v1",
                r"\blateral menisc(?:us|al)\b.{0,65}\btear\b",
                "a direct lateral-meniscus tear is the supported report phrase",
            ),
            _rule(
                "lateral-meniscus-surface-signal-v1",
                r"\blateral menisc(?:us|al)\b.{0,65}\b(?:signal|line)\b.{0,35}\b(?:reaches?|contacts?)\s+(?:the\s+)?surface\b",
                "surface-contacting signal matches the host criterion",
            ),
        ),
        (
            _rule(
                "lateral-meniscus-intrasubstance-v1",
                r"\blateral menisc(?:us|al)\b.{0,75}\bintrasubstance degeneration\b.{0,45}\b(?:does not reach|not reaching)\s+(?:the\s+)?surface\b",
                "the host explicitly excludes intrasubstance degeneration without surface contact",
            ),
        ),
        (
            "tear grade or surface contact not stated",
            "degenerative signal without an explicit surface-contacting tear",
        ),
    ),
    _definition(
        "Medial OA",
        "Osteoarthritis of the medial tibiofemoral compartment.",
        "Moderate/large area of high-grade medial-compartment cartilage loss, over half thickness, with or without marrow change.",
        "Mild, low-grade, borderline or nonspecific medial cartilage change is negative.",
        r"\bmedial\b.{0,30}\b(?:compartment|tibiofemoral|oa|osteoarthritis|cartilage)\b",
        (
            _rule(
                "medial-oa-moderate-large-v1",
                r"\b(?:moderate|large)\s+(?:medial\s+)?(?:tibiofemoral\s+)?(?:compartment\s+)?(?:osteoarthritis|oa)\b",
                "moderate/large compartment OA is directly supported by the host description",
            ),
            _rule(
                "medial-oa-high-grade-cartilage-loss-v1",
                r"\bhigh[- ]grade\s+cartilage loss\b.{0,55}\bmedial\s+(?:tibiofemoral\s+)?compartment\b",
                "high-grade medial compartment cartilage loss matches the host criterion",
            ),
        ),
        (
            _rule(
                "medial-oa-borderline-v1",
                r"\b(?:mild|low[- ]grade|small)\b.{0,45}\bmedial\b.{0,30}\b(?:oa|osteoarthritis|cartilage)\b",
                "borderline/mild changes are conservatively negative under the host policy",
            ),
        ),
        (
            "chondropathy, cartilage wear or severity below the host threshold",
            "high-grade loss without a compartment or severity statement",
        ),
    ),
    _definition(
        "Lateral OA",
        "Osteoarthritis of the lateral tibiofemoral compartment.",
        "Moderate/large area of high-grade lateral-compartment cartilage loss, over half thickness, with or without marrow change.",
        "Mild, low-grade, borderline or nonspecific lateral cartilage change is negative.",
        r"\blateral\b.{0,30}\b(?:compartment|tibiofemoral|oa|osteoarthritis|cartilage)\b",
        (
            _rule(
                "lateral-oa-moderate-large-v1",
                r"\b(?:moderate|large)\s+(?:lateral\s+)?(?:tibiofemoral\s+)?(?:compartment\s+)?(?:osteoarthritis|oa)\b",
                "moderate/large compartment OA is directly supported by the host description",
            ),
            _rule(
                "lateral-oa-high-grade-cartilage-loss-v1",
                r"\bhigh[- ]grade\s+cartilage loss\b.{0,55}\blateral\s+(?:tibiofemoral\s+)?compartment\b",
                "high-grade lateral compartment cartilage loss matches the host criterion",
            ),
        ),
        (
            _rule(
                "lateral-oa-borderline-v1",
                r"\b(?:mild|low[- ]grade|small)\b.{0,45}\blateral\b.{0,30}\b(?:oa|osteoarthritis|cartilage)\b",
                "borderline/mild changes are conservatively negative under the host policy",
            ),
        ),
        (
            "chondropathy, cartilage wear or severity below the host threshold",
            "high-grade loss without a compartment or severity statement",
        ),
    ),
    _definition(
        "PF OA",
        "Patellofemoral compartment osteoarthritis.",
        "Moderate/large area of high-grade patellofemoral cartilage loss, over half thickness, with or without marrow change.",
        "Mild, low-grade, borderline or nonspecific patellofemoral cartilage change is negative.",
        r"\b(?:pf|patellofemoral)\b.{0,35}\b(?:compartment|oa|osteoarthritis|cartilage|chondropathy)\b",
        (
            _rule(
                "pf-oa-moderate-large-v1",
                r"\b(?:moderate|large)\s+(?:pf|patellofemoral)(?:\s+compartment)?\s+(?:osteoarthritis|oa)\b",
                "moderate/large patellofemoral OA is directly supported by the host description",
            ),
            _rule(
                "pf-oa-high-grade-cartilage-loss-v1",
                r"\bhigh[- ]grade\s+cartilage loss\b.{0,55}\b(?:pf|patellofemoral)\b",
                "high-grade patellofemoral cartilage loss matches the host criterion",
            ),
        ),
        (
            _rule(
                "pf-oa-borderline-v1",
                r"\b(?:mild|low[- ]grade|small)\b.{0,45}\b(?:pf|patellofemoral)\b.{0,35}\b(?:oa|osteoarthritis|cartilage|chondropathy)\b",
                "borderline/mild changes are conservatively negative under the host policy",
            ),
        ),
        (
            "patellar chondropathy or cartilage wear without the host OA threshold",
            "high-grade loss without a patellofemoral compartment or severity statement",
        ),
    ),
    _definition(
        "Effusion",
        "Joint effusion / excess fluid.",
        "Moderate or large fluid amount distending the joint.",
        "Trace, small, mild, borderline or absent fluid is negative; generic fluid mention is not enough.",
        r"\b(?:joint\s+)?effusion\b",
        (
            _rule(
                "effusion-moderate-large-v1",
                r"\b(?:moderate|large)\s+(?:joint\s+)?effusion\b|\b(?:joint\s+)?effusion\b.{0,25}\b(?:moderate|large)\b",
                "requires the host's moderate/large fluid threshold",
            ),
        ),
        (
            _rule(
                "effusion-small-v1",
                r"\b(?:trace|small|mild|minimal)\s+(?:joint\s+)?effusion\b",
                "small or mild fluid is below the host threshold",
            ),
        ),
        (
            "suprapatellar fluid without a moderate/large joint-distension statement",
            "fluid amount or distension severity not stated",
        ),
    ),
    _definition(
        "Synovitis",
        "Inflammation of the joint lining.",
        "Explicit synovial inflammation or synovial lining thickening.",
        "Explicitly normal/non-thickened synovium or no synovitis is negative; absence of mention is not negative.",
        r"\b(?:synovitis|synovial(?:\s+lining|\s+membrane)?)\b",
        (
            _rule(
                "synovitis-explicit-v1",
                r"\b(?:synovitis|synovial\s+(?:lining|membrane)?)\b.{0,35}\b(?:inflammation|thickening|inflamed|thickened)\b|\bsynovitis\b",
                "direct inflammation/thickening or the target name is supported",
            ),
        ),
        (
            _rule(
                "synovitis-normal-lining-v1",
                r"\b(?:synovial\s+(?:lining|membrane)|synovium)\b.{0,35}\b(?:normal|not thickened|without thickening)\b",
                "explicitly normal or non-thickened lining is negative",
            ),
        ),
        (
            "minimal/borderline synovitis and inflammatory wording without a clear finding",
            "effusion does not imply synovitis",
        ),
    ),
    _definition(
        "Baker's",
        "Baker (popliteal) cyst.",
        "Moderate or large fluid collection in the characteristic posterior-knee location.",
        "Small, mild, borderline or absent Baker/popliteal cyst is negative.",
        r"\b(?:baker(?:'s|s)?|popliteal)(?:\s+cyst)?\b",
        (
            _rule(
                "bakers-cyst-moderate-large-v1",
                r"\b(?:moderate|large)\s+(?:baker(?:'s|s)?|popliteal)\s+cyst\b|\b(?:baker(?:'s|s)?|popliteal)\s+cyst\b.{0,25}\b(?:moderate|large)\b",
                "requires the host's moderate/large characteristic collection threshold",
            ),
        ),
        (
            _rule(
                "bakers-cyst-small-v1",
                r"\b(?:small|mild|trace|minimal)\s+(?:baker(?:'s|s)?|popliteal)\s+cyst\b",
                "small or mild cyst is below the host threshold",
            ),
        ),
        (
            "posterior fluid collection without size/location confirmation",
            "a generic cyst or fluid collection is not enough",
        ),
    ),
    _definition(
        "Contusion",
        "Bone contusion / bone bruise.",
        "Impact-related bone marrow edema-like signal without a discrete fracture line.",
        "No impact-related contusion, or an ambiguous marrow-edema phrase without a contusion interpretation, is not positive.",
        r"\b(?:bone\s+contusion|bone\s+bruise|bone\s+marrow\s+edema(?:[- ]like)?|marrow\s+edema)\b",
        (
            _rule(
                "contusion-direct-v1",
                r"\b(?:bone\s+contusion|bone\s+bruise)\b",
                "direct bone contusion/bruise language is supported",
            ),
            _rule(
                "contusion-impact-edema-v1",
                r"\b(?:bone\s+)?marrow\s+edema(?:[- ]like)?\b.{0,55}\b(?:from|due to)\s+impact\b",
                "impact-related marrow edema matches the host description",
            ),
        ),
        (
            _rule(
                "contusion-negated-v1",
                r"\b(?:no|without|absence of)\b.{0,35}\b(?:bone\s+contusion|bone\s+bruise)\b",
                "explicit absence is negative",
            ),
        ),
        (
            "nonspecific marrow edema, stress reaction, or osteochondral injury",
            "marrow edema without impact attribution or contusion wording",
        ),
    ),
    _definition(
        "Fracture",
        "Fracture.",
        "Acute cortical break or fracture line.",
        "No acute fracture/cortical break is negative; chronic, osteochondral, subchondral or insufficiency wording remains unresolved.",
        r"\b(?:fracture|fractures|cortical break)\b",
        (
            _rule(
                "fracture-acute-line-v1",
                r"\bacute\s+(?:cortical\s+break|fracture(?:\s+line)?)\b|\b(?:acute|new)\s+fracture\s+line\b",
                "requires an acute break/line matching the host definition",
            ),
        ),
        (
            _rule(
                "fracture-negated-v1",
                r"\b(?:no|without|absence of)\b.{0,35}\b(?:acute\s+)?fracture\b|\b(?:no|without)\s+cortical break\b",
                "explicit absence of an acute fracture is negative",
            ),
        ),
        (
            "osteochondral, subchondral, insufficiency, chronic or healed fracture",
            "fracture wording without the supported acute cortical-break criterion",
        ),
    ),
)


_HISTORICAL_PATTERN = re.compile(
    r"\b(?:history of|prior|previous|remote|old|chronic|healed)\b",
    re.IGNORECASE,
)
_UNCERTAIN_PATTERN = re.compile(
    r"\b(?:possible|probable|suspected|question of|cannot exclude|may represent|equivocal|indeterminate)\b",
    re.IGNORECASE,
)
_NEGATION_BEFORE_PATTERN = re.compile(
    r"\b(?:no|without|negative for|absence of|absent|unremarkable|normal|intact|preserved|not identified|not seen)\b[^.;:]{0,55}$",
    re.IGNORECASE,
)
_NEGATION_AFTER_PATTERN = re.compile(
    r"^\s*(?:is|are|appears?|looks?)?\s*(?:normal|intact|preserved|unremarkable|absent|not identified|not seen)\b",
    re.IGNORECASE,
)
_SENTENCE_BREAKS = re.compile(r"[.!?;:\n]")


def validate_rule_origin_for_folds(origin: RuleOrigin, consuming_folds: Iterable[int]) -> None:
    """Reject report-derived or participant rules that are not fold-clean.

    Official definitions and the project masking policy are shared-safe.  A
    future report-derived rule may be used only for an explicit, non-shared
    fold scope with an allowlist covering every consuming fold.
    """

    folds = _validate_fold_ids(consuming_folds)
    report_derived = origin.derived_from_report or origin.kind is RuleOriginKind.PATIENT_REPORT
    if report_derived:
        if origin.shared_across_folds:
            raise ReportInterpretationError(
                f"report-derived rule {origin.rule_id!r} cannot be shared across folds"
            )
        if origin.allowed_folds is None:
            raise ReportInterpretationError(
                f"report-derived rule {origin.rule_id!r} requires an allowed fold scope"
            )
    if origin.participant_interpretation or origin.kind is RuleOriginKind.PARTICIPANT_INTERPRETATION:
        raise ReportInterpretationError(
            f"participant interpretation {origin.rule_id!r} is not an authoritative rule"
        )
    if origin.source_report_refs and origin.shared_across_folds:
        raise ReportInterpretationError(
            f"rule {origin.rule_id!r} contains report examples but is marked shared"
        )
    if origin.shared_across_folds and origin.kind not in {
        RuleOriginKind.OFFICIAL_DEFINITION,
        RuleOriginKind.PROJECT_SAFETY_POLICY,
    }:
        raise ReportInterpretationError(
            f"shared rule {origin.rule_id!r} is not from an authoritative source"
        )
    if origin.allowed_folds is not None and not set(folds).issubset(origin.allowed_folds):
        raise ReportInterpretationError(
            f"rule {origin.rule_id!r} is not clean for all consuming folds"
        )


def target_definitions() -> tuple[TargetDefinition, ...]:
    """Return the immutable authoritative register in competition order."""

    return _TARGET_DEFINITIONS


def authority_register() -> tuple[dict[str, object], ...]:
    """Return a serializable public authority/rule register."""

    return tuple(item.to_dict() for item in _TARGET_DEFINITIONS)


def _normalize_report(report: str) -> str:
    normalized = report.casefold().replace("’", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", normalized).strip()


def _local_context(text: str, start: int, end: int) -> tuple[str, str, str]:
    left_breaks = [match.start() for match in _SENTENCE_BREAKS.finditer(text[:start])]
    right_match = _SENTENCE_BREAKS.search(text, end)
    left = left_breaks[-1] + 1 if left_breaks else 0
    right = right_match.start() if right_match is not None else len(text)
    sentence = text[left:right].strip()
    sentence_start = text.find(sentence, left, right) if sentence else left
    if sentence_start < 0:
        sentence_start = left
    local_start = max(sentence_start, start - 80)
    local_end = min(right, end + 80)
    return text[local_start:local_end], text[local_start:start], text[end:local_end]


def _has_local_negation(before: str, after: str) -> bool:
    return bool(_NEGATION_BEFORE_PATTERN.search(before) or _NEGATION_AFTER_PATTERN.search(after))


def _find_events(text: str, definition: TargetDefinition) -> tuple[_TargetEvent, ...]:
    mention_pattern = re.compile(definition.mention_pattern, re.IGNORECASE)
    events: list[_TargetEvent] = []
    for match in mention_pattern.finditer(text):
        local, before, after = _local_context(text, match.start(), match.end())
        if _HISTORICAL_PATTERN.search(local):
            events.append(
                _TargetEvent(LabelState.UNKNOWN, "historical-finding-v1", match.group(0))
            )
            continue
        if _UNCERTAIN_PATTERN.search(local):
            events.append(
                _TargetEvent(LabelState.UNCERTAIN, "uncertain-finding-v1", match.group(0))
            )
            continue
        matching_negative = next(
            (rule for rule in definition.negative_patterns if re.search(rule.pattern, local)),
            None,
        )
        matching_positive = next(
            (rule for rule in definition.positive_patterns if re.search(rule.pattern, local)),
            None,
        )
        if matching_negative is not None or _has_local_negation(before, after):
            rule_id = matching_negative.rule_id if matching_negative is not None else "explicit-negation-v1"
            events.append(_TargetEvent(LabelState.NEGATIVE, rule_id, match.group(0)))
        elif matching_positive is not None:
            events.append(_TargetEvent(LabelState.POSITIVE, matching_positive.rule_id, match.group(0)))
        else:
            events.append(_TargetEvent(LabelState.UNKNOWN, "unsupported-mention-v1", match.group(0)))
    return tuple(events)


def _label_from_events(
    definition: TargetDefinition,
    events: tuple[_TargetEvent, ...],
) -> TargetLabel:
    if not events:
        return TargetLabel(
            target=definition.target,
            state=LabelState.NOT_MENTIONED,
            rule_id="absence-is-not-negative-v1",
            rule_origin=_POLICY_ORIGIN,
        )
    states = {event.state for event in events}
    if LabelState.POSITIVE in states and (
        LabelState.NEGATIVE in states
        or LabelState.UNKNOWN in states
        or LabelState.UNCERTAIN in states
    ):
        return TargetLabel(
            target=definition.target,
            state=LabelState.EXTRACTION_FAILED,
            rule_id="conflicting-evidence-v1",
            rule_origin=_POLICY_ORIGIN,
            evidence=tuple(event.evidence for event in events),
        )
    if LabelState.NEGATIVE in states and (
        LabelState.UNKNOWN in states or LabelState.UNCERTAIN in states
    ):
        return TargetLabel(
            target=definition.target,
            state=LabelState.EXTRACTION_FAILED,
            rule_id="conflicting-evidence-v1",
            rule_origin=_POLICY_ORIGIN,
            evidence=tuple(event.evidence for event in events),
        )
    event = events[0]
    origin = definition.rule_origin if event.state in {LabelState.POSITIVE, LabelState.NEGATIVE} else _POLICY_ORIGIN
    return TargetLabel(
        target=definition.target,
        state=event.state,
        rule_id=event.rule_id,
        rule_origin=origin,
        evidence=tuple(item.evidence for item in events),
    )


def interpret_report(
    report: str,
    *,
    source_report_ref: str | None = None,
    consuming_folds: Sequence[int] = (),
) -> ReportInterpretation:
    """Interpret one report conservatively into the twelve target states.

    This function retains no raw report text in its result.  Only exact,
    criterion-backed positive/negative matches receive a value and mask one.
    Empty input is an extraction failure; an ordinary report with no supported
    target mention is ``not-mentioned`` rather than negative.
    """

    if not isinstance(report, str):
        raise ReportInterpretationError("report must be a string")
    folds = _validate_fold_ids(consuming_folds)
    for definition in _TARGET_DEFINITIONS:
        validate_rule_origin_for_folds(definition.rule_origin, folds)
    digest = sha256(report.encode("utf-8")).hexdigest()
    normalized = _normalize_report(report)
    if not normalized:
        labels = tuple(
            TargetLabel(
                target=definition.target,
                state=LabelState.EXTRACTION_FAILED,
                rule_id="empty-report-v1",
                rule_origin=_POLICY_ORIGIN,
            )
            for definition in _TARGET_DEFINITIONS
        )
    else:
        labels = tuple(
            _label_from_events(definition, _find_events(normalized, definition))
            for definition in _TARGET_DEFINITIONS
        )
    return ReportInterpretation(
        extractor_version=REPORT_EXTRACTOR_VERSION,
        report_sha256=digest,
        labels=labels,
        source_report_ref=source_report_ref,
        consuming_folds=folds,
    )


__all__ = [
    "AUTHORITY_RETRIEVED_ON",
    "DATA_DESCRIPTION_URL",
    "HOST_CLARIFICATION_URL",
    "LABEL_DESCRIPTION_URL",
    "LabelState",
    "PatternRule",
    "ReportInterpretation",
    "ReportInterpretationError",
    "REPORT_EXTRACTOR_VERSION",
    "REPORT_INTERPRETATION_SCHEMA_VERSION",
    "RuleOrigin",
    "RuleOriginKind",
    "TargetDefinition",
    "TargetLabel",
    "authority_register",
    "interpret_report",
    "target_definitions",
    "validate_rule_origin_for_folds",
]
