# 14: Establish auditable report-to-target interpretation

**What to build:** Establish authoritative target interpretation and clean synthetic cases without inspecting held-out reports for rule development.

**Blocked by:** None (dependency-free; implementation authorization is still required).

**Status:** resolved

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [x] Record source/date and supported clinical-positive/negative criteria per target using official competition definitions or host clarifications.
- [x] Test negation, uncertainty, historical findings and absence of mention with independently authored synthetic examples; unsupported interpretations remain unknown with zero loss mask.
- [x] Track rule origin so shared rules are clean for every consuming fold; no patient-report examples or unverified participant interpretation become universal rules.
- [x] Document gaps and optional clinical-review needs without blocking gold-only mechanics or inventing unsupported clinical criteria.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V11 — Official clinical target definitions

- [x] For every target used in report-derived supervision, retain a dated authoritative competition definition or attributable host clarification that supports the implemented positive/negative criteria. Map severity, negation, uncertainty and historical/absent mention handling to that evidence; independent synthetic examples must exercise the mapping. Unsupported or ambiguous interpretations remain unknown with zero loss mask and local audit evidence. Participant interpretations alone do not satisfy authority.

**Blocking consequence:** Block report-label rules for any target/interpretation lacking authoritative support; do not invent a clinical rule. Gold-reference mechanics can proceed without interpreting report text. If complete twelve-target clinical support is unavailable, document masked/unsupported targets and do not claim a complete trustworthy extractor.

## Completion evidence

- [x] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.

## Comments

### 2026-09-14 — Synthetic authority and masking implementation

Implemented the ticket-14 public seam in `src/rsna_knee/report_labels.py`.
The immutable authority register covers all twelve targets using the official
Kaggle data description, the competition-host label description, and the host
report clarification, each recorded with retrieval date `2026-09-14`.

`interpret_report()` requires a private source-report reference and preserves
positive, negative, not-mentioned, uncertain, unknown and extraction-failed
states. Only explicit criterion-backed positive and negative states receive a
binary value and supervision mask one. Historical, uncertain, unsupported,
absent, empty and contradictory cases remain masked, including negated
osteochondral-fracture and nonspecific-marrow phrases. The result records
extractor version/configuration fingerprint, rule confidence, report digest,
fold context and rule provenance without retaining raw report text; public
serialization excludes private report references and evidence snippets.

`validate_rule_origin_for_folds()` rejects shared patient-report and participant
interpretation rules. A future report-derived rule must be explicitly
fold-scoped with an allowed-fold set covering every consumer, private
report/study references, source folds, and no overlap with a consuming fold's
held-out source fold. No patient or held-out report was inspected, and no
report-derived label was admitted to training.

Added 38 public synthetic tests covering all twelve positive and negative
criteria, negation, uncertainty, historical findings, unsupported
osteochondral wording, absent mention, empty input, conflicting evidence,
public/private provenance and fold-scope enforcement. Focused tests passed and
mypy reported no issues for the new module. Evidence is recorded in
`docs/evidence/ticket-14-report-interpretation.md`.

The demonstrated scope is the authority register and synthetic deterministic
English seam. Multilingual coverage, real-report accuracy/coverage, clinical
review, fold-specific report evaluation and weak-label training remain open
for ticket 15; V11 is not a claim that a complete trustworthy real-world
extractor has been validated.
