# 15: Evaluate fold-specific deterministic report extraction

**What to build:** Generate and evaluate auditable deterministic report labels under clean fold-specific development and evaluation boundaries.

**Blocked by:** 10: Produce complete three-fold OOF evidence; 14: Establish auditable report-to-target interpretation

**Status:** resolved

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [x] Represent positive, negative, not_mentioned, uncertain and extraction_failed distinctly; only admissible confident positives/negatives have supervision mask one.
- [x] Record private source-report reference, extractor/configuration version, rule evidence, fold context and confidence; source text never leaves the cluster by default.
- [x] Develop each fold's rules only from its training-side reports or independently clean resources; materially report-derived rules cannot cross into an affected held-out fold.
- [x] Freeze the extractor before evaluation on held-out gold reports; separate evaluation outputs from training labels and do not tune from those report contents.
- [x] Report target accuracy where defined, coverage, failure/abstention, uncertainty handling, reproducibility, auditability and latency; unsupported targets are explicitly masked.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V11 — Official clinical target definitions

- [ ] For every target used in report-derived supervision, retain a dated authoritative competition definition or attributable host clarification that supports the implemented positive/negative criteria. Map severity, negation, uncertainty and historical/absent mention handling to that evidence; independent synthetic examples must exercise the mapping. Unsupported or ambiguous interpretations remain unknown with zero loss mask and local audit evidence. Participant interpretations alone do not satisfy authority.

**Blocking consequence:** Block report-label rules for any target/interpretation lacking authoritative support; do not invent a clinical rule. Gold-reference mechanics can proceed without interpreting report text. If complete twelve-target clinical support is unavailable, document masked/unsupported targets and do not claim a complete trustworthy extractor.

### V15 — Derived-supervision provenance and isolation

- [ ] For each extractor/rule/teacher/pseudo-label/student artifact, resolve versioned code/configuration, source and checkpoint hashes, fold/repeat, eligible/excluded study-set identities and all upstream adaptation inputs. Verify zero prohibited held-out input influence for the consuming fold and no unresolved lineage. Enforce reference precedence and zero-mask unknown/uncertain/failed states. Frozen evaluation-only outputs must be distinguishable from admitted training labels.

**Blocking consequence:** Reject artifacts with ambiguous or contaminated lineage from that fold's training and model-selection evidence. Invalidate and rerun materially affected comparisons after repair; an apparent AUC gain cannot override the gate. Clean reference-only work remains possible.

## Completion evidence

- [x] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.

## Comments

### 2026-09-14 — Fold-clean deterministic extraction implementation

Implemented the ticket-15 boundary in `src/rsna_knee/report_extraction.py`.
`FrozenDeterministicExtractor` is an immutable wrapper around the ticket-14
official-rule register and records both the configuration and complete ruleset
fingerprints. `ReportCorpus` accepts private report text only as an input; the
generated private audit artifact retains source references, report digests,
fold context, rule evidence and confidence but never copies source text.

`run_report_extraction()` builds one partition per canonical fold. Only
report-only studies from that fold's training side become
`admitted_training_labels`; reports from held-out gold studies are emitted as
`held_out_evaluation` records and are never admitted to training. Missing gold
reports are counted explicitly. The boundary validator rejects unknown studies,
gold training labels, held-out training reports, wrong-fold evaluation reports,
and overlapping partitions. Evaluation compares confident report values with
the frozen manifest's reference labels and leaves all abstentions masked.

The public result reports per-target accuracy where defined, coverage,
failure/abstention, not-mentioned/uncertain/unknown/extraction-failed counts,
confidence, fold metrics, latency, semantic repeatability, privacy and
provenance. JSON publication is atomic and refuses overwrites. The bounded
adapter is `slurm/run_ticket15_report_extraction.sbatch`; it uses one
non-exclusive `cluster1-cpu` node, two CPUs, 8 GB and one hour. CLI and package
exports are documented in `README.md`.

Added seven public synthetic tests in `tests/test_report_extraction.py`.
The fixture contains twelve gold and six report-only studies. The complete
synthetic report cohort produced four held-out gold reports per fold, five/two/
five training report-label records for folds 0/1/2, 100% defined-target
accuracy and coverage, and a verified repeated semantic digest. Separate tests
cover unsupported/uncertain masking, missing reports, boundary rejection,
private/public serialization and corpus loading. Focused report tests passed;
the full suite and typecheck are recorded in the external implementation
report requested by the owner.

The implementation demonstrates the clean synthetic/local mechanics only. No
private competition report was inspected or processed, no report-derived
training run was admitted, and no Slurm allocation was submitted. Real-report
accuracy, coverage, latency and weak-supervision benefit remain pending the
authorized private execution; competition inference remains report-free.

## Answer

### Closure decision — 2026-09-14

Ticket 15 is resolved for the implemented, synthetic-testable extraction
boundary. The public API now joins a checksum-covered canonical three-fold
manifest to private reports, admits only report-only training records from the
fold's training side, and evaluates held-out gold records in a separate
partition with a frozen deterministic extractor. The artifact writer keeps
aggregate evidence public and identifier-bearing audit evidence private; it
does not persist source report text.

The five acceptance criteria are covered by the implementation and seven
synthetic tests. The fixture contains twelve gold and six report-only studies;
each fold evaluates four gold reports, training-label counts are 5/2/5 for
folds 0/1/2, and all defined synthetic target metrics have 100% accuracy and
100% coverage. Repeated text-free semantic artifacts produce the same digest.

V11 and V15 remain unchecked here because no private competition reports were
processed and no report-derived training run or cluster execution was
authorized or performed in this implementation turn. The authoritative rule
register and synthetic masking mechanics from ticket 14 are reused, but they
are not a claim of real-report clinical accuracy or complete production gate
evidence. The detailed implementation and verification record is maintained
outside the repository at `/home/ch.jitheswar/ticket-15-report.md`.
