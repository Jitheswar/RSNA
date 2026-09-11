# 15: Evaluate fold-specific deterministic report extraction

**What to build:** Generate and evaluate auditable deterministic report labels under clean fold-specific development and evaluation boundaries.

**Blocked by:** 10: Produce complete three-fold OOF evidence; 14: Establish auditable report-to-target interpretation

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Represent positive, negative, not_mentioned, uncertain and extraction_failed distinctly; only admissible confident positives/negatives have supervision mask one.
- [ ] Record private source-report reference, extractor/configuration version, rule evidence, fold context and confidence; source text never leaves the cluster by default.
- [ ] Develop each fold's rules only from its training-side reports or independently clean resources; materially report-derived rules cannot cross into an affected held-out fold.
- [ ] Freeze the extractor before evaluation on held-out gold reports; separate evaluation outputs from training labels and do not tune from those report contents.
- [ ] Report target accuracy where defined, coverage, failure/abstention, uncertainty handling, reproducibility, auditability and latency; unsupported targets are explicitly masked.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V11 — Official clinical target definitions

- [ ] For every target used in report-derived supervision, retain a dated authoritative competition definition or attributable host clarification that supports the implemented positive/negative criteria. Map severity, negation, uncertainty and historical/absent mention handling to that evidence; independent synthetic examples must exercise the mapping. Unsupported or ambiguous interpretations remain unknown with zero loss mask and local audit evidence. Participant interpretations alone do not satisfy authority.

**Blocking consequence:** Block report-label rules for any target/interpretation lacking authoritative support; do not invent a clinical rule. Gold-reference mechanics can proceed without interpreting report text. If complete twelve-target clinical support is unavailable, document masked/unsupported targets and do not claim a complete trustworthy extractor.

### V15 — Derived-supervision provenance and isolation

- [ ] For each extractor/rule/teacher/pseudo-label/student artifact, resolve versioned code/configuration, source and checkpoint hashes, fold/repeat, eligible/excluded study-set identities and all upstream adaptation inputs. Verify zero prohibited held-out input influence for the consuming fold and no unresolved lineage. Enforce reference precedence and zero-mask unknown/uncertain/failed states. Frozen evaluation-only outputs must be distinguishable from admitted training labels.

**Blocking consequence:** Reject artifacts with ambiguous or contaminated lineage from that fold's training and model-selection evidence. Invalidate and rerun materially affected comparisons after repair; an apparent AUC gain cannot override the gate. Clean reference-only work remains possible.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.
