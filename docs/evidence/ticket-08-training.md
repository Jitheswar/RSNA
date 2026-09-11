# Ticket 08 fixed-budget gold-only training evidence

Recorded 2026-09-11 (Asia/Kolkata). This is public-safe implementation and
synthetic-fixture evidence; it contains no competition identifiers, reports,
raw paths, private manifests, or real-corpus predictions.

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

The public suite contains 101 passing tests. Ticket-specific tests demonstrate
masked-entry normalization, all-masked no-progress failure, deterministic
finite learning on a tiny fixture, complete checkpoint metadata and semantic
compatibility checks, gold-only fold partitioning, final-checkpoint held-out
evaluation, exact AUC behavior, and the CPU preflight. The preflight also
passed locally in the pinned isolated Python 3.11 environment, including
dependency imports, finite model gradients, checkpoint save/reload, known AUC
values, and a fresh inference import without training orchestration.

The bounded adapter is `slurm/run_ticket08_train_fold.sbatch`. It requests one
`cluster1-cpu` node, 16 CPUs, 64 GB, six hours, and no exclusive allocation.

## Verification limits

No private ticket-08 Slurm training allocation or real-corpus held-out AUC was
run in this implementation pass. V07 remains an execution gate for the
cluster environment, and V09/V14 remain dependent on authorized private cache
and canonical-fold evidence. The adapter is prepared for that run but does
not claim real-data training, competitive performance, GPU compatibility,
promotion, upload, or final selection.
