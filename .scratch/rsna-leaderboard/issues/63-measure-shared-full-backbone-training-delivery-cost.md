# 63: Measure shared full-backbone training and checkpoint delivery cost

Type: task
Status: resolved
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
- [x] Measure all-fold construction/fingerprints and actual complete pixel I/O.
- [x] Verify fresh-process interrupted/resumed and continuous real-MRI state
  equality, actual full-backbone changes, fixed normalization and terminal.
- [x] Retain stage startup, step/checkpoint intervals, file sizes, reload times
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


## Answer

Job **8018 completed `0:0`, 2,170 allocated CPU-node seconds**. Independent
terminal/source/state reconciliation passes. All 4,228 pixel entries and every
original training target/mask are admitted. Fresh-process resumed and continuous
three-update states match exactly, including model, Adam, scheduler, progress
and sampler; every encoder stage changes and normalization buffers stay fixed.
No held-out predictions or predictive gain are inferred.

Complete pixel read **126.443 s**; all-fold example construction **97.390–100.800
s**, fingerprinting **185.434–192.671 s**. Maximum fresh-stage startup **372.921 s**,
maximum diagnostic step/prior-checkpoint interval **8.745448 s**, maximum
checkpoint **135,071,169 bytes**, peak RSS **16,871,268,352 bytes**. Actual
checkpoint reload **0.259566–0.479378 s**. These measured complete costs support
separate Ticket 64 admission, rather than a training-only extrapolation.
[Evidence](../../../docs/evidence/ticket-63-shared-full-backbone-training-cost.md).

October 2: Independent verification at **06:06:37 UTC** binds exact original
supervision/exposures, admitted initial weights, retained terminal and source.
One consumed attempt remains closed. Ticket 64 prospectively registers the
longer matched regime and independently bounds its entire delivery route.
