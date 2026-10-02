# 63: Measure shared full-backbone training and checkpoint delivery cost

Type: task
Status: claimed
Labels: ready-for-agent
Parent: [RSNA campaign](../spec.md)

## Purpose

Measure the complete shared CPU training path before a prospective longer
three-fold comparison. Ticket 61 established per-update feasibility; Ticket
62 independently admitted every original packed pixel. The remaining costs
include complete training-example construction/fingerprints, repeated staged
startup, mixed reference/report batches, durable checkpoints and fresh-process
exact resume. These costs must be measured rather than inferred from a small
four-report throughput pilot.

## Fixed scope

One nonexclusive 16-CPU/64-GiB/90-minute job, one launch, no requeue, GPU,
network, new weights, spend, held-out prediction or candidate selection.
Forecast is 16 times the actual complete-cohort verified cache read cost plus
600 seconds. Five fresh processes under one allocation: admit all 4,228 pixel
entries and all original fold memberships; interrupt fold-0 training at one
successful update; resume to three updates; train the same three updates
continuously; independently compare complete model/Adam/progress/sampler
states and unchanged normalization buffers.

Use the original fold-specific warm report head after 7,500 updates, admitted
ImageNet encoder, original ordered gold/report pools, two draws per source,
fixed 224-pixel packing, Adam encoder 1e-5/head 1e-4, and no augmentation.
Three diagnostic updates are never evaluated for AUC or promoted. Admit every
target/mask from the existing audited source and reject empty supervision.
Checkpoint every diagnostic update to measure serialization and reloading.

## Requirements

- [x] Freeze exact source, inputs, complete cohort, fresh storage, runtime,
  standing resource decision and accepted one-attempt allocation.
- [ ] Measure all-fold construction/fingerprints and actual complete pixel I/O.
- [ ] Verify fresh-process interrupted/resumed and continuous real-MRI state
  equality, actual full-backbone changes, fixed normalization and terminal.
- [ ] Retain stage startup, step/checkpoint intervals, file sizes, reload times
  and RSS for a separate complete fitting/final-delivery decision.

## Comments

October 2: The complete pixel job passes exact source replay and terminal
reconciliation. Shared lifecycle tests pass, including exact sampler exposure
and checkpoint resume. Standing owner authorization applies. This ticket
admits a bounded cost experiment only; a 7,500-update three-fold regime is a
prospective scenario, not an authorized full fit or an accuracy result.

October 2: Accepted CPU job **8018**, one attempt, actual allocation agrees.
Plan `40f5268a019a9d247cb3ced8d0d7a62c419bd7b1e544a1370af7f884d65f7e39`,
complete forecast **3,629.069 seconds**, 90-minute server bound. Five fresh
processes use the immutable source snapshot. Terminal observer is active.
No second request; complete experiment/checkpoint metrics remain pending.
