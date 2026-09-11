# Ticket 04 immutable fold-freeze evidence

Recorded 2026-09-11 (Asia/Kolkata). This is public-safe implementation
evidence; it contains no competition identifiers, reports, raw paths, patient
keys, or private manifest contents.

## Demonstrated scope

The synthetic source-manifest seam demonstrates:

- exact one-fold membership for every gold and report-only training study;
- deterministic multilabel-aware gold allocation over all twelve targets;
- stable identifier-hash allocation for report-only studies;
- whole-group allocation across gold and report-only cohorts when a grouping
  assessment is explicitly reliable;
- explicit study-level splitting and the repeated-patient leakage limitation
  when no reliable key is available;
- per-fold gold positive/negative counts for all twelve targets, including a
  documented mathematically unavoidable one-class case;
- checksum-covered private membership and generation metadata;
- public evidence without study, series, report, or group identifiers;
- immutable, atomic publication and checksum-verified reload; and
- a fold-level admission check that rejects held-out studies from adaptation
  inputs and declares held-out images, metadata, and reports as excluded.

The bounded adapter is `slurm/run_ticket04_freeze_folds.sbatch`. It uses one
`cluster1-cpu` node, two CPUs, 8 GB, and one hour without exclusive allocation.
The core itself does not inspect report text or DICOM pixels.

## Tests

The ticket-specific synthetic fold tests pass in the pinned Python 3.11
environment. They cover deterministic regeneration, target support, reliable
group joins, partial-label rejection, privacy, immutable publication,
checksum failure, and held-out adaptation rejection.

## Unresolved gates and limitations

- No private competition fold manifest was generated in this evidence pass.
  The owner must run the adapter against the private ticket-03 inventory before
  treating the resulting artifact as the canonical competition manifest.
- V02 remains unresolved until legitimate grouping fields are inspected across
  the indexed corpus. The default study-only path permits the baseline but does
  not support patient-independent generalization claims.
- V14 remains pending real-corpus membership and label-join verification. This
  implementation does not claim that a synthetic demonstration verifies the
  private fold-freeze gate.
- A later reliable patient key, changed source version, label join, or fold
  generation parameter requires a new immutable manifest and invalidates
  affected comparisons.
- No report adaptation, training, upload, submission, or protected deletion
  was performed.
