# 13: Execute an approved GPU experiment safely

**What to build:** Run one specifically approved GPU experiment with verified Slurm resources and bounded failure recovery.

**Blocked by:** 09: Resume an interrupted experiment safely

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Require passed CPU preflight and explicit approval before every GPU training request, including short smoke tests; prepare requested resources and evidence before asking.
- [ ] Use sbatch on the appropriate verified partition; interactive srun debugging needs approval and shared GPU memory is not assumed to equal nominal A40 capacity.
- [ ] Record actual hardware, memory, driver, peak VRAM, full run metadata and checkpoint recovery for the intended shape/configuration.
- [ ] Permit one unchanged retry only for clearly transient infrastructure failure covered by experiment approval; user cancellation, OOM, code bugs, NaN, incompatible resume and deterministic loader failures are excluded.
- [ ] Stop after a failed retry; resource, model, batch-size or scientific changes require diagnosis and applicable new approval.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V10 — Slurm GPU resource syntax, sharing and A40 assumptions

- [ ] Record the intended partition, Slurm/GRES request semantics, actual device model, driver compatibility, memory assigned/available and sharing/exclusivity behavior for the specific resource. Treat the user-reported approximately 45 GB A40 capacity as nominal, not guaranteed allocatable memory. Following CPU gates and explicit approval, a bounded GPU smoke test must run the intended model/data shape through forward, backward and checkpoint operations within that exact request. Record observed peak VRAM and compare it to the available allocation.

**Blocking consequence:** Block full GPU training or claims of GPU compatibility for an unverified resource. Do not assume cluster1-gpu-add provides a whole A40 or silently move to a larger/multi-GPU request. Resource or model/batch changes require the applicable new experiment/request approval and repeated smoke verification.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.
