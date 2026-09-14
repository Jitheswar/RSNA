# Ticket 14 — Auditable report-to-target interpretation

Status: implementation evidence recorded; no private or held-out report was inspected.

Date: 2026-09-14

## Scope

Ticket 14 establishes the authority register, conservative label-state model,
fold-clean rule provenance, and public synthetic behavior needed before any
report-derived supervision is admitted. It does not run on the competition
corpus, tune rules from gold reports, create weak-label training artifacts, or
claim that the small deterministic vocabulary is a complete multilingual
report extractor. Those activities belong to ticket 15 and remain gated.

The implementation is in [`src/rsna_knee/report_labels.py`](../../src/rsna_knee/report_labels.py),
with public behavior tests in [`tests/test_report_labels.py`](../../tests/test_report_labels.py).

## Authority and date record

The register records all sources with retrieval date `2026-09-14`. The live
Kaggle rendering does not expose a stable publication date in the retrieved
HTML, so the implementation records the verifiable retrieval date rather than
inventing a publication date.

| Source | Role |
| --- | --- |
| [Official Kaggle data description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data) | Confirms the twelve target names and their dataset-level meanings. |
| [Competition-host label description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343) | Supplies the annotator criteria, severity/size thresholds, and borderline-negative policy. |
| [Competition-host report clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826) | Confirms that image-derived consensus labels are authoritative when reports disagree and that absence of a report finding is not treated as a confirmed negative by this project. |

The implementation paraphrases the source criteria into a per-target
`positive_criterion`, `negative_criterion`, and unresolved-interpretation list.
It does not treat participant discussions or patient reports as authority.

| Target | Supported positive interpretation | Supported negative interpretation |
| --- | --- | --- |
| ACL | High-grade partial/full-thickness tear, discontinuity, or rupture. | Mild signal/degeneration/thickening or explicit intact/normal wording. |
| MCL | High-grade acute partial/complete tear with fiber disruption. | Low-grade sprain or chronic/remote change; explicit intact/normal wording. |
| Medial Meniscus | Definite medial meniscus tear or surface-contacting signal. | Intrasubstance degeneration without surface contact; explicit normal/no-tear wording. |
| Lateral Meniscus | Definite lateral meniscus tear or surface-contacting signal. | Intrasubstance degeneration without surface contact; explicit normal/no-tear wording. |
| Medial OA | Moderate/large medial-compartment OA or high-grade cartilage loss. | Mild/low-grade/borderline medial cartilage/OA wording. |
| Lateral OA | Moderate/large lateral-compartment OA or high-grade cartilage loss. | Mild/low-grade/borderline lateral cartilage/OA wording. |
| PF OA | Moderate/large patellofemoral OA or high-grade cartilage loss. | Mild/low-grade/borderline patellofemoral cartilage/chondropathy wording. |
| Effusion | Moderate/large joint effusion distending the joint. | Trace/small/mild effusion or explicit absence. |
| Synovitis | Explicit synovitis, synovial inflammation, or lining thickening. | Explicit normal/non-thickened synovium or no synovitis. |
| Baker's | Moderate/large Baker or popliteal cyst in the characteristic location. | Small/mild/borderline cyst or explicit absence. |
| Contusion | Impact-related bone contusion/bruise or marrow edema-like signal. | Explicit absence; nonspecific marrow edema remains unsupported rather than negative. |
| Fracture | Acute cortical break or acute fracture line. | Explicit absence of acute fracture; chronic/osteochondral/subchondral/insufficiency wording remains unsupported. |

The register keeps one official source origin for each target rule. The
project safety origin is used only for abstention/masking policies such as
“not mentioned is not negative,” historical findings, unsupported wording,
contradictory evidence, and empty reports.

## State and masking contract

`interpret_report()` requires a private `source_report_ref`, records the
extractor version/configuration fingerprint, and emits one ordered record for
every target. It preserves:

- `positive` → value `1`, supervision mask `1`;
- `negative` → value `0`, supervision mask `1`;
- `not-mentioned` → no value, mask `0`;
- `uncertain` → no value, mask `0`;
- `unknown` → no value, mask `0`;
- `extraction-failed` → no value, mask `0`.

An empty report is an extraction failure. A non-empty report with no supported
mention is not silently converted to a negative. Contradictory positive and
negative evidence is also failed and masked. A negated unsupported phrase such
as an osteochondral fracture or nonspecific marrow edema remains `unknown`,
not a confident negative.

The result contains a SHA-256 report digest and required private source-report
reference, but never retains the raw report text. Public serialization omits
the reference and matched evidence snippets; private serialization is the
local audit form for a future approved fold-specific run.

## Fold-clean rule provenance

`validate_rule_origin_for_folds()` permits shared rules only from the official
definition or project safety-policy origins. A report-derived rule must be
non-shared, name an explicit allowed-fold set, carry private report and study
references, record its source folds, avoid the consuming fold's held-out source
folds, and cover every consuming fold. Participant interpretations are
rejected. This prevents a patient-report example or an unverified discussion
interpretation from becoming a universal rule while leaving a later
fold-specific ticket a precise provenance seam.

## Synthetic verification

The focused suite passed with 38 tests:

- one authority record covers all twelve target columns and all sources carry
  the retrieval date;
- twelve independently authored positive report sentences map to positive
  values and mask one;
- twelve independently authored negative report sentences map to negative
  values and mask one;
- possible, historical, osteochondral, nonspecific-marrow, absent, empty, and
  contradictory cases remain unknown/uncertain/failed with mask zero;
- public output does not include raw report text or a private source reference;
- shared report-derived origins are rejected and correctly fold-scoped origins
  are accepted only for their declared fold and source-fold boundary;
- every admitted deterministic positive/negative label records rule confidence,
  while abstentions remain unconfident and masked, and the interpretation
  carries a reproducible configuration fingerprint.

Commands:

```text
PYTHONPATH=src .venv/bin/python -m pytest tests/test_report_labels.py -q
PYTHONPATH=src .venv/bin/python -m mypy src/rsna_knee/report_labels.py
```

Observed result: `38 passed`; mypy reported no issues.

## Limitations and unresolved review needs

- The matcher is intentionally English-only and narrow. It is a synthetic
  mechanics seam, not evidence of coverage over the competition's multilingual
  reports.
- Direct report language can disagree with independently image-derived labels;
  reference labels remain authoritative. No report-derived label was admitted
  to training here.
- Chondropathy versus OA, minimal synovitis, nonspecific marrow edema,
  osteochondral/subchondral/insufficiency fracture, bilateral-study wording,
  and fluid-size thresholds outside the explicit criteria require clinical
  review or abstention. The current implementation masks these unsupported
  interpretations.
- No real report, held-out report, report-only cohort, fold manifest, Slurm job,
  or clinical-review process was used. No accuracy, coverage, latency, or
  weak-supervision benefit claim is made.

## Gate disposition

The authority-register and independent-synthetic stage of V11 is demonstrated
for all twelve targets. The full report-extraction and fold-specific evaluation
gate remains open for ticket 15; gold-only mechanics remain valid and do not
depend on this module. No approval-dependent compute, upload, submission, or
private-data action was performed.
