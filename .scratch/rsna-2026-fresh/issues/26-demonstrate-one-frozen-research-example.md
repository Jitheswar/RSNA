# 26: Demonstrate one frozen research example

**What to build:** Create a small later academic demonstration of one frozen example and model with visible processing and provenance.

**Blocked by:** 24: Select and deliver the final candidate

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Use synthetic or explicitly permitted examples after competition-rights and export checks; do not introduce public patient MRI/report uploads.
- [ ] Show selected series/slices, preprocessing/rejections, twelve model scores, model/configuration/candidate provenance and known limitations.
- [ ] Reuse tested Python inference and choose a minimal appropriate interface; MongoDB, Express and other services are not mandatory infrastructure.
- [ ] Describe scores as uncalibrated model confidence and the application as research/education; no clinical diagnostic or validated-risk claims.
- [ ] Keep external deployment and restricted-data publication behind applicable approval and eligibility gates.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.
