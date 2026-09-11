# Ticket 09 interrupted-training recovery evidence

Recorded 2026-09-11 (Asia/Kolkata). This is public-safe implementation and
synthetic-fixture evidence; it contains no competition identifiers, reports,
raw paths, private manifests, or real-corpus predictions.

## Implemented seam

`rsna_knee.training.train_fixed_budget` can continue a fixed-budget run from
an atomically published `latest_resumable` checkpoint. The checkpoint schema
stores model, optimizer, scheduler, optional AMP scaler, epoch, global step,
best metric, loss/progress counters, Python/NumPy/PyTorch RNG state, scientific
configuration, fold, semantic fingerprint, deterministic sampler state, and
checkpoint version.

The latest pointer is written only after a complete same-directory temporary
write. Complete latest checkpoints are also retained in an immutable history,
so a damaged pointer can resolve to the highest-progress valid history entry.
Epoch boundaries and the configured 15-minute interval trigger checkpoints;
each event records elapsed and overdue timing plus uncheckpointed batch count.
The final checkpoint is marked `final_selected` and is never accepted as a
resume source.
Complete ticket-08 v1 checkpoints are validated and normalized in memory on
load; their original run ID is retained and the next durable checkpoint uses
the v2 schema.

Semantic fingerprints cover data/source and cache identity, split/fold,
reference labels, preprocessing, packing, architecture, initialization,
seeds, optimizer, schedule, masked BCE loss, augmentation, and successful-step
budget. Slurm/resource details, execution IDs, host details, and checkpoint
cadence are operational metadata. Compatible resumes preserve the scientific
run ID and create a new `attempts/attempt-NNNN/` log and completion record.
Incompatible changes fail before training; `fork_training_run` records the
parent run/checkpoint and starts a new deterministic initialization. Retry
classification allows at most one explicitly transient infrastructure retry
and does not submit retries automatically.

## Demonstrated scope

The public tests cover uninterrupted versus interrupted/resumed model-state
equivalence, distinct execution attempts, complete AMP/progress/RNG metadata,
atomic write fault injection, epoch and timed checkpoint records, fallback to
the last complete history checkpoint, semantic mismatch rejection, explicit
fork provenance, retry classification, and high-level fold resume.

The bounded adapter is
`slurm/run_ticket09_resume_training.sbatch`; it retains the ticket-08
one-node `cluster1-cpu`, 16-CPU, 64-GB, six-hour, non-exclusive envelope.

## Fold-manifest supersession

This ticket's evidence is synthetic and contains no real-corpus metric. Since
the version-2 fold correction, the training CLI refuses a superseded version-1
fold manifest unless `--allow-superseded-fold-manifest` is given, and that
option exists only to replay historical runs. The ticket-09 resume adapter does
not forward the option, so it cannot resume a run on a superseded manifest.

## Verification limits

No private ticket-09 Slurm restart, real-corpus rollback measurement, or
GPU-training retry was run in this implementation pass. V07/V09/V14 remain
execution-dependent, and this evidence does not establish a real-data
15-minute guarantee, competition performance, candidate promotion, upload, or
final selection.
