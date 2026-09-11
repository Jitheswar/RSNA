# 14: Establish auditable report-to-target interpretation

**What to build:** Establish authoritative target interpretation and clean synthetic cases without inspecting held-out reports for rule development.

**Blocked by:** None (dependency-free; implementation authorization is still required).

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Record source/date and supported clinical-positive/negative criteria per target using official competition definitions or host clarifications.
- [ ] Test negation, uncertainty, historical findings and absence of mention with independently authored synthetic examples; unsupported interpretations remain unknown with zero loss mask.
- [ ] Track rule origin so shared rules are clean for every consuming fold; no patient-report examples or unverified participant interpretation become universal rules.
- [ ] Document gaps and optional clinical-review needs without blocking gold-only mechanics or inventing unsupported clinical criteria.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V11 — Official clinical target definitions

- [ ] For every target used in report-derived supervision, retain a dated authoritative competition definition or attributable host clarification that supports the implemented positive/negative criteria. Map severity, negation, uncertainty and historical/absent mention handling to that evidence; independent synthetic examples must exercise the mapping. Unsupported or ambiguous interpretations remain unknown with zero loss mask and local audit evidence. Participant interpretations alone do not satisfy authority.

**Blocking consequence:** Block report-label rules for any target/interpretation lacking authoritative support; do not invent a clinical rule. Gold-reference mechanics can proceed without interpreting report text. If complete twelve-target clinical support is unavailable, document masked/unsupported targets and do not claim a complete trustworthy extractor.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.
