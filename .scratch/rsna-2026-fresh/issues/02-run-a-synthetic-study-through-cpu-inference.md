# 02: Run a synthetic study through CPU inference

**What to build:** Demonstrate a minimal synthetic DICOM study reaching twelve scores and valid competition-format output in a reproducible CPU environment.

**Blocked by:** None (dependency-free; implementation authorization is still required).

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [x] Use verified user-space Python 3.11 and modern pinned dependencies, with a thin CPU Slurm adapter; significant computation never runs on the login node.
- [x] Preserve study and series structure through deterministic preprocessing, masked hierarchical pooling and a twelve-output head; inference imports independently of training and report orchestration.
- [x] Test synthetic spatial ordering, shape/mask behavior and output mapping, exact thirteen-column schema, finite bounded scores and no extra CSV index.
- [x] Record environment identity and successful CPU inference evidence. This first slice does not claim real-DICOM compatibility or completion of the later full training preflight.

## Completion evidence

- [x] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.

## Comments

### 2026-09-11 — Implementation evidence

Implemented the synthetic CPU tracer bullet in `src/rsna_knee/`. It creates
in-memory pydicom MR fixtures and maps them through a decoder-independent
record, then provides geometry-validated spatial ordering, deterministic
128×128 series preparation,
post-cache eight-slice sampling, explicit slice/series masks, a shared small
2D encoder with masked hierarchical pooling, a twelve-output head,
training-independent inference, all-invalid study fallback, and strict
competition CSV validation. The thin adapter is
`slurm/run_cpu_inference.sbatch` and is bounded to one `cluster1-cpu` node,
four CPUs, 8 GB, and 30 minutes without exclusive allocation.

Slurm job `6896` completed successfully on 2026-09-11 using Python 3.11.16,
NumPy 2.2.6, PyTorch 2.7.1+cpu, and pydicom 3.0.1. It processed two synthetic
studies and produced two rows with the exact thirteen-column schema, finite
scores in `[0, 1]`, and zero fallbacks. The detailed evidence is recorded in
[`docs/evidence/ticket-02-cpu-inference.md`](../../../docs/evidence/ticket-02-cpu-inference.md).

The implementation does not claim real-DICOM compatibility, full corpus
coverage, training completion, Kaggle execution, upload, or competition
submission. The available cluster Python 3.11.9 module was not used because
its interpreter could not import `_ctypes`; the separate user-space `mise`
Python environment passed the runtime checks.
