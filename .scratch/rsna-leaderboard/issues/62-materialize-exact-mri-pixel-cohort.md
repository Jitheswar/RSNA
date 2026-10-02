# 62: Materialize exact MRI pixels for full-backbone training

Type: task
Status: claimed
Labels: ready-for-agent
Parent: [RSNA campaign](../spec.md)

## Purpose

The verified full-backbone CPU route needs bounded, random-access pixel I/O
for all folds. Retain every admitted selected slice exactly as FP32, storing
only valid slices with explicit ascending packed indices. Reconstruct the
same 16-series/8-slice padded bags without repeated source decoding or fitted
transforms. This enables the complete matched whole-backbone comparison;
it does not establish an accuracy gain.

## Fixed first stage

All **4,228** admitted studies and **187,272** selected valid slices from the
existing frozen-feature/activation cohort. Preserve original MRI identity,
preprocessing/packing, source-series/slice ordering, masks and anatomical
descriptors. Cache contains only pixels and metadata; no labels, fitted model,
augmentation or transformed target. Source prepared series and archive are
read-only. Output payload **37,586,239,488 bytes**, plus bounded metadata.

One nonexclusive CPU job, 16 CPUs/64 GiB/three hours, one materialization
attempt, no requeue, GPU or external spend. Current CoAt scan keeps its job;
at most two concurrent CPU jobs. Freeze fresh storage measurement and source
snapshot. Forecast uses the prior actual full-cohort source/encoding run with
2× cost plus 900 seconds; removing encoder inference must not justify weaker
source checks. Archive gold studies follow the previously admitted reader.

## Requirements

- [x] Implement immutable pixel entries with complete array/source identities,
  ordered valid indices and exact padded reconstruction; reject corruption,
  wrong source membership, identity or slice mapping.
- [x] Freeze and launch one exact storage/resource-admitted complete cohort.
- [x] Independently verify every file/array/source/mask/descriptor identity,
  exact cohort/terminal/job/resources, and previously admitted real-MRI pixel
  replay; capture read/reconstruction costs for the next training plan.
- [ ] Freeze complete three-fold matched-reference training, staged checkpoint,
  I/O/evaluation/final-fit/delivery cost separately before fitting.

## Comments

October 2: [Whole-backbone CPU pilot](../../../docs/evidence/ticket-61-full-backbone-cpu-cost.md)
is independently verified. The long 7,500-step duration needs bounded staged
jobs; the current stage admits pixels only, without selecting a fitted
candidate or authorizing a full campaign. CPU can proceed while the exact
TPU environment request remains queued. Standing owner authorization applies.

October 2: **8017 completed `0:0`, 2,080 CPU-node seconds**. All 4,228 studies /
187,272 selected slices match a fresh native-source packing process exactly;
37,586,239,488 pixel bytes, complete masks/descriptors/source mappings, and the
prior four-report full input hash agree. Verified whole-cohort cache read and
reconstruction took 189.317 seconds. Source replay uses the same frozen
preprocessing algorithm in a separate process. No labels or fitting occurred.
[Evidence](../../../docs/evidence/ticket-62-exact-mri-pixel-cohort.md).

The shared CPU trainer now supports file-backed inputs, complete encoder/head
optimizer groups, and native prediction/reconstruction. Exact interrupted /
resumed versus continuous training passes focused lifecycle checks. A separate
[Ticket 63](63-measure-shared-full-backbone-training-delivery-cost.md) measures
actual mixed-source startup, staged real-MRI checkpoint delivery and cost.
Complete longer comparison/final-fit/delivery admission remains pending.
