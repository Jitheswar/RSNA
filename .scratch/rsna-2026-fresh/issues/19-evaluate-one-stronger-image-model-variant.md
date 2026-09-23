# 19: Evaluate one stronger image-model variant

**What to build:** Evaluate one narrowly scoped image architecture or input-policy change against the established supervision-controlled baseline.

**Blocked by:** 18: Measure the controlled benefit of report supervision

**Status:** ready-for-agent

**Authorization:** Initially planning-only. The owner's direct instruction to implement Ticket 19 on 2026-09-23 separately authorized this implementation and the bounded CPU experiment below. GPU execution and the other approval-gated actions remain unauthorized unless separately approved.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Define one proposal such as resolution, slice count, encoder or plane aggregation; keep supervision, folds and other comparison settings fixed where required to isolate the change.
- [ ] Treat numerical preprocessing/sampling changes as versioned scientific settings and preserve cache compatibility only when semantics allow it.
- [ ] Record OOF/per-target evidence, uncertainty, train cost, full inference runtime, VRAM and package size, accepting negative results.
- [ ] Any new pretrained resource must satisfy V13 before download/use and any GPU execution must satisfy V10 and specific approval; neither is silently authorized by this ticket.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.

## Comments

- 2026-09-23 — Owner explicitly instructed implementation. A CPU-only three-fold run was submitted to `cluster1-cpu` as Slurm array `7101`, with two concurrent tasks maximum. No pretrained resource or GPU is used.
