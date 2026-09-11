# 23: Compare fold ensembles with all-data refitting

**What to build:** Produce an evidence-based final delivery strategy using available validated methods without requiring advanced research to succeed.

**Blocked by:** 11: Freeze a locally validated offline candidate

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Compare eligible fold-model averaging against one/few all-data refit options using OOF selection evidence, training cost, runtime, memory, checkpoint size and packaging complexity.
- [ ] Define the refit training budget and admissible all-training supervision explicitly; proposed additional training requires its own approval where applicable.
- [ ] Do not run an expensive refit merely to complete a checklist when existing evidence rejects it; document the decision and measurement limits.
- [ ] Do not fabricate OOF performance for exact refitted weights; record final priors from all trusted training reference labels and preserve the existing candidate.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.
