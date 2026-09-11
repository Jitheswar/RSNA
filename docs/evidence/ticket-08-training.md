# Ticket 08 fixed-budget gold-only training evidence

Recorded 2026-09-11 (Asia/Kolkata). This is public-safe implementation,
synthetic-fixture, and one real-corpus fold-0 execution evidence; it contains
no competition identifiers, reports, raw paths, private manifests, or
per-study prediction rows.

## Implemented seam

`rsna_knee.training` now provides a fixed mechanics configuration and a
reference-only packed-study training boundary. The default configuration is
the agreed shared `(8, 16, 32)` slice encoder, 128×128 grayscale inputs,
hierarchical masked means, twelve logits, masked BCE, Adam at `0.001`, a
constant schedule, batch size two, and 100 successful optimizer steps.

The training loop averages BCE over supervised entries only. Fully masked
batches are recorded and skip both optimizer and scheduler progress; a
consecutive no-progress limit raises an explicit failure. It validates finite
losses and gradients, writes complete atomic latest/final checkpoints with
optimizer, scheduler, RNG, configuration and progress state, and supports
configuration-checked reload.

`build_fold_training_examples` selects only gold memberships outside the
declared held-out fold. It uses fixed series-local preprocessing, never fits
normalization statistics, excludes reports, and records reference-label and
fold provenance. `train_one_fold` reloads the final checkpoint before scoring
the held-out gold rows. `rsna_knee.evaluation` implements rank ROC-AUC with
average ties and retains one-class support as undefined.

The real-data CLI requires matching source versions, archive fingerprints,
source-manifest checksums, selected gold-study coverage, preprocessing and
packing policies, and integration-manifest checksums before fitting.

## Demonstrated scope

The public suite currently contains 112 passing tests. Ticket-specific tests
demonstrate masked-entry normalization, all-masked no-progress failure,
deterministic finite learning on a tiny fixture, complete checkpoint metadata and semantic
compatibility checks, gold-only fold partitioning, final-checkpoint held-out
evaluation, exact AUC behavior, and the CPU preflight. The preflight also
passed locally in the pinned isolated Python 3.11 environment, including
dependency imports, finite model gradients, checkpoint save/reload, known AUC
values, and a fresh inference import without training orchestration.

The bounded adapter is `slurm/run_ticket08_train_fold.sbatch`. It requests one
`cluster1-cpu` node, 16 CPUs, 64 GB, six hours, and no exclusive allocation.

## Real fold-0 execution evidence

> **Superseded for comparison (2026-09-11).** This run used the version-1 fold
> manifest `ae81f327…d51ad3e7`, whose defective gold allocation held out 26
> studies in fold 0 and trained on 32. Its metric is kept as historical
> evidence only. The corrected three-fold rerun on the version-2 manifest is
> recorded in [`ticket-10-oof.md`](ticket-10-oof.md).

A completed bounded cluster1-cpu Slurm run is now recorded for the real
corpus. It used reference supervision only, trained on 32 studies, evaluated
26 fold-0 held-out gold studies, performed 100 successful optimizer and
scheduler steps, skipped zero all-masked batches, used no learned
normalization, and produced a finite final loss. The CPU preflight passed for
the same run, and the final checkpoint is present and reloadable through the
current compatibility loader.

The held-out artifact contains exactly one prediction vector of twelve scores
per held-out study. Replaying the evaluation against the frozen fold manifest
confirmed the canonical target order, exact held-out identity join, matching
per-target label counts, finite scores, and non-constant scores in every target
column. The saved and recomputed macro ROC-AUC is
`0.43171391044456064`.

The stored final checkpoint uses the supported legacy v1 checkpoint schema;
the current loader migrates it in memory as a `final_selected` checkpoint.
This is compatibility metadata, not a model or score-direction change.

## Verification limits

The earlier implementation-pass limitation that no private ticket-08 Slurm
allocation or real-corpus held-out AUC existed is superseded by the fold-0
evidence above. This single fold does not establish complete three-fold OOF
coverage, the shared-storage throughput/concurrency/peak-use gate V09, or all
three-fold support and allocation requirements of V14. It also does not claim
competitive performance, GPU compatibility, promotion, upload, or final
selection.
