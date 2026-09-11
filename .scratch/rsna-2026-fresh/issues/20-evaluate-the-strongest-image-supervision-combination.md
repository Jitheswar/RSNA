# 20: Evaluate the strongest image/supervision combination

**What to build:** Test whether individually supported image and supervision choices improve the combined MRI-only system.

**Blocked by:** 19: Evaluate one stronger image-model variant

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Use a named fixed comparison and fold-safe supervision lineage; do not infer combined gains from two independent experiment results.
- [ ] Report pooled/per-target OOF evidence, variation, uncertainty and whole-pipeline costs against existing eligible baselines.
- [ ] If supervision or image changes did not earn adoption, preserve the stronger simpler system and record that decision rather than forcing a combination.
- [ ] Freeze any promoted result separately; advanced experiments cannot overwrite or block the working competition path.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.
