# 27: Produce the research and reproduction report

**What to build:** Produce a local research report and reproduction guide grounded in preserved experiment and candidate evidence.

**Blocked by:** 24: Select and deliver the final candidate

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Describe domain/task, image-only baseline, actual controlled experiments, selection strategy and separately reproducible cluster/inference environments.
- [ ] Report pooled/per-target OOF, class counts, variation and uncertainty; disclose development-selection bias, lack of untouched internal test and any unverified patient separation.
- [ ] Tie reported results to source, configuration, fold/data/supervision identities and checkpoints while keeping identifiers, reports and restricted artifacts private.
- [ ] Explain resource costs, failure/recovery policies and evidence scope; do not claim clinical superiority from small CV or leaderboard gains.
- [ ] Deliver local documentation only; external publication and artifact transfer remain subject to rights and approval, and omitted optional experiments are not reported as executed.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.
