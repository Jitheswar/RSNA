# 09: Resume an interrupted experiment safely

**What to build:** Resume the same scientific experiment after interruption while rejecting semantic changes disguised as resumes.

**Blocked by:** 08: Train and evaluate one gold-only fold

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Save model, optimizer, scheduler, AMP scaler when applicable, epoch/global step, best metric, RNG states, configuration, fold and checkpoint version.
- [ ] Checkpoint at every epoch boundary and approximately every fifteen minutes during long epochs; test bounded replay and record any measured exception rather than replaying an entire long epoch silently.
- [ ] Fault-inject interrupted writes and reload the last complete persistent checkpoint; separate latest-resumable from final/selected checkpoints.
- [ ] Compare semantic fingerprints covering data, split, labels, preprocessing, architecture, initialization, seeds, optimizer, schedule, loss, augmentation and budget; mismatches require a new run with parent provenance.
- [ ] Separate scientific run identity from execution attempts and logs; compatible operational restarts preserve identity without overwriting earlier evidence.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.

## Comments

### 2026-09-11 — Resumable checkpoint and execution-attempt implementation

Implemented the resumable training seam in `src/rsna_knee/training.py`.
Complete `latest_resumable` checkpoints now retain model, optimizer, scheduler,
optional AMP scaler, epoch/global step, best metric, loss/progress counters,
RNG state, configuration, fold, semantic fingerprint, deterministic sampler
continuation and checkpoint version. Epoch boundaries and the 15-minute timer
are checkpoint triggers; timing and uncheckpointed-batch bounds are recorded.
Complete ticket-08 v1 checkpoints are accepted through a validated in-memory
migration, retaining their original run identity while publishing v2 state on
the next checkpoint.

Latest checkpoints are atomically published and copied to immutable history.
Corrupt latest pointers resolve to the highest-progress complete history entry,
while `final_selected` checkpoints are separate and rejected by the resume
path. Semantic changes across data, split, labels, preprocessing, architecture,
initialization, seeds, optimizer, schedule, loss, augmentation or budget are
rejected. `fork_training_run` provides the explicit new-run path with parent
checkpoint provenance.

Scientific run identity is preserved across compatible resumes. Each execution
attempt receives its own log and immutable start plus completion/failure record,
including conservative one-retry transient-failure classification. The new
bounded adapter is `slurm/run_ticket09_resume_training.sbatch`.

Public synthetic evidence is recorded in
`docs/evidence/ticket-09-training-resume.md`. No private Slurm restart,
real-corpus rollback measurement, GPU retry, competition result or
approval-dependent gate is claimed by this implementation pass.

### 2026-09-11 — Verification and commit record

Implementation committed as `d7d74cb` (`feat: resume interrupted training safely`).
Verification passed with 112 tests, mypy across 26 source files, Slurm shell
syntax validation, and a clean staged diff. The public evidence remains
synthetic; no private execution, real-corpus rollback measurement, GPU retry,
competition result, or approval-dependent gate is marked verified.
