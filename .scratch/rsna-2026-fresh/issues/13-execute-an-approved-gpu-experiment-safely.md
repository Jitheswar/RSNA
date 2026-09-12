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

## Comments

### 2026-09-12 — Approval-gated execution implementation

Implemented the ticket-13 seam in `src/rsna_knee/gpu.py` and
`src/rsna_knee/gpu_cli.py`. Request preparation requires a passed,
checksum-covered CPU preflight and freezes the CUDA configuration, exact
single-GPU Slurm resources, input identities and retry policy. Submission
requires a separate owner approval for that exact request checksum and uses
`sbatch` only. The in-allocation runner verifies Slurm/GRES visibility,
records actual CUDA/nvidia-smi hardware and assigned memory, delegates to the
existing checkpointed trainer, reloads the final checkpoint, and permits only
one unchanged transient-infrastructure retry. Durable failure metadata and
classification prevent an in-job retry from becoming a second wrapper retry;
OOM, cancellation, numerical, semantic and loader failures stop.

Added bounded CPU-preflight and GPU Slurm adapters, shared-training resource
metadata support, and synthetic tests for approval, resource, hardware,
checkpoint and retry boundaries. Public implementation evidence is recorded
in `docs/evidence/ticket-13-gpu-experiment.md`.

The GPU adapter uses a separate Python 3.11 CUDA environment rather than the
repository's CPU-only `.venv`, validates the pinned wheel/runtime contract,
requires reported Slurm CPU, memory, wall-time, GPU-count and sharing facts,
and caps smoke requests at 60 minutes. Direct CUDA training also requires the
approval-gated execution capability.

No real preflight, owner-approved GPU request, `sbatch` submission, GPU
allocation, A40 measurement, private-data training, or compatibility claim was
performed. The acceptance checklist and V10 therefore remain intentionally
unchecked.

### 2026-09-13 — Final verification and handoff

Hardened the implementation after review: public training seams are CPU-only;
the private GPU seams require the high-level approval/preflight path and an
active Slurm job; assigned GPU memory is explicit and must be observed before
training; `nvidia-smi` is bound to the CUDA-visible device; production study
loading is lazy until allocation checks pass; and a checksum-keyed submission
record prevents reuse of one approval for duplicate initial submissions. The
CPU preflight now verifies its exact bounded Slurm envelope; private GPU
authorization independently reloads the request, approval and preflight and
rechecks the live allocation; execution is consumed before lazy loading; and
public retry evidence redacts raw exception text. Submission output must also
contain a parseable scheduler job ID.

Final checks passed: 192 repository tests (35 focused GPU/training tests), mypy for 36 source files, Bash
syntax checks for both Ticket 13 adapters, and `git diff --check`. The code is
committed on `master` as `9c371cc`, `eda51a3`, `4fde09a`, `8ae96c8`, and
`3fe25a9`. No real
CPU preflight, owner approval, GPU allocation, smoke run, private-data
training, or V10 verification was performed; all future execution gates remain
unchecked.
