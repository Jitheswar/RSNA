# 08: Train and evaluate one gold-only fold

**What to build:** Complete one fixed-budget image-only training run and held-out evaluation with reproducible experiment evidence.

**Blocked by:** 06: Reuse and recover incremental series caches; 07: Recover damaged inputs without losing study rows

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Use the agreed shared 8/16/32-channel encoder, hierarchical masked means, twelve logits, masked BCE, Adam at 0.001 with constant schedule, batch size two and 100 successful optimizer steps.
- [ ] Normalize BCE over supervised entries only; all-masked batches advance neither optimizer nor scheduler, and persistent no-progress fails explicitly.
- [ ] Verify finite gradients and learning on a deterministic tiny overfit fixture without requiring competitive real-data AUC; evaluate the final checkpoint, not a validation-selected epoch.
- [ ] Persist configuration, seeds, model/source identity, data/cache/fold fingerprints, environment, resources, Slurm attempt identity, checkpoint, metrics, logs and inference configuration.
- [ ] Use reference supervision only; confirm held-out inputs do not enter fitting or learned normalization and complete the CPU preflight before real training.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V07 — KLEF software environment

- [ ] Record the actual interpreter and module identity, OS, resolved dependency versions and environment fingerprint. In a bounded cluster1-cpu verification job, demonstrate imports of the core, decoder, metric and training dependencies; known ROC-AUC results; finite model loss and gradients; checkpoint save/reload; and inference import without report/training orchestration dependencies. Confirm that the interpreter is user-environment Python 3.11 rather than the system stack. Any required user-space runtime repair must have preserved provenance and pass the same checks.

**Blocking consequence:** Block real training and GPU-job requests until CPU preflight passes. Dependency installation or a successful resolver check alone does not qualify. Diagnose in user space without modifying shared system software; a changed effective environment must be recorded and dependent compatibility checks repeated.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.

## Comments

### 2026-09-11 — Fixed-budget training implementation

Implemented the reference-only training/evaluation seam in
`src/rsna_knee/training.py` and `src/rsna_knee/evaluation.py`. The default
configuration uses the shared 8/16/32-channel encoder, hierarchical masked
means, twelve logits, masked BCE, Adam at 0.001, a constant schedule, batch
size two, and 100 successful optimizer steps. All-masked batches do not
advance optimizer or scheduler state and persistent no-progress fails
explicitly.

Fold example construction admits only gold rows outside the selected held-out
fold and records reference-only provenance. The final checkpoint is saved
atomically with complete optimizer/scheduler/RNG/configuration state and is
reloaded before held-out evaluation. Configuration, data/cache/fold
fingerprints, environment, resource envelope, Slurm attempt identity, logs,
metrics, and inference configuration are persisted in the private experiment
directory. The CPU preflight covers the pinned Python 3.11 environment,
dependency imports, known ROC-AUC values, finite loss/gradients,
checkpoint save/reload, and inference import isolation.

The bounded adapter is `slurm/run_ticket08_train_fold.sbatch`; it requests one
`cluster1-cpu` node, 16 CPUs, 64 GB, six hours, and no exclusive allocation.
Public synthetic scope is recorded in
`docs/evidence/ticket-08-training.md`. At this implementation commit, the full
suite passed 101 tests and mypy passed. The execution limitation recorded here
was later superseded by the reconciliation comment below; the historical
implementation evidence is retained as written.

### 2026-09-11 — Reconciled real fold-0 execution evidence

A completed bounded cluster1-cpu Slurm run now exists for the real corpus. The
identifier-free evidence records reference-only supervision, 32 training
studies, 26 held-out fold-0 gold studies, 100 successful optimizer and
scheduler steps, zero all-masked skips, no learned normalization, a passing
CPU preflight, and a reloadable final checkpoint.

The private run artifact contains one twelve-score prediction vector for each
held-out study. Independent replay against the frozen fold manifest confirmed
the canonical target order, exact held-out identity and label-count joins,
finite scores, and non-constant values in every target column. The saved and
recomputed macro ROC-AUC is `0.43171391044456064`. The current suite passes 112
tests.

This closes the earlier “no private run / no real-corpus AUC” documentation
caveat. Complete three-fold OOF coverage, V09 shared-storage evidence, and all
V14 three-fold allocation/support requirements remain unresolved; no claim of
competitive performance, GPU compatibility, promotion, upload, or final
selection is made.

### 2026-09-11 — Superseded for comparison by the version-2 fold correction

The real fold-0 run above used the version-1 fold manifest `ae81f327…d51ad3e7`,
whose defective gold allocation held out 26 studies and trained on 32. The
owner approved the version-2 fold correction. This run's metric
(`0.43171391044456064`) is therefore historical evidence only and is not
comparable with any version-2 result. The artifacts are retained unchanged.
The corrected three-fold rerun with the same configuration and 100-step budget
is recorded under ticket 10.
