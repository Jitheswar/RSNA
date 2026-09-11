# 25: Evaluate one late efficiency optimization

**What to build:** Optionally measure one efficiency improvement after a valid final delivery path exists.

**Blocked by:** 24: Select and deliver the final candidate

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Choose one measurable whole-notebook bottleneck and a fixed quality comparator; include setup, loading, decoding, preprocessing, forward passes, ensembles and output costs.
- [ ] Report OOF quality effects, observed/projected runtime, peak VRAM and model/package size; the primary metric and hard validity constraints remain dominant.
- [ ] Freeze any resulting candidate separately and request owner selection before replacing the selected final model.
- [ ] Keep this optional late experiment from delaying the existing valid submission or becoming an early Efficiency Prize workstream.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V16 — Whole-notebook runtime and inference cost

- [ ] Benchmark every stage from startup/setup through CSV creation on documented hardware and representative workload, including long-tail studies, model loading and ensembling. Record peak VRAM, package/model size, input counts and observed versus projected times. A locally promoted candidate must project no more than 25,920 seconds at approximately 1,300 studies unless the owner explicitly approves the engineering-margin risk; observed Kaggle evidence must state which workload actually ran and remain within the 32,400-second hard limit. An example-only run is not hidden-scale proof.

**Blocking consequence:** Block normal promotion above the 7h12m engineering target until optimized or the specific margin risk is approved. Execution beyond nine hours is ineligible and cannot be waived by internal approval. Block full-scale compliance claims when evidence is only a tiny example; keep unknown hidden runtime explicit and preserve the previous verified candidate.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.
