# 57: Materialize deterministic ResNet prefix activations

Type: task
Status: resolved
Labels: ready-for-agent
Parent: [RSNA leaderboard campaign](../spec.md)
Blocked by: 56

## Purpose and scope

Ticket 56 demonstrated exact native/cached last-block adaptation parity and
plausible CPU cost. Prepare and measure the complete activation cohort so the
next experiment can adapt fold-specific encoders with admitted report supervision.
This supplies the complete unfitted inputs for the next CPU adaptation comparison.

Use the existing **4,228 studies / 187,272 valid slices**, 224px whole-field
preprocessing, at most 16 series and eight slices, anatomy descriptors and
the exact admitted ImageNet ResNet-18 weights. All encoder parameters and
normalization buffers remain frozen in eval mode. Store only each valid slice's
FP32 activation before layer4, shape **(N,256,14,14)**, with exact masks,
descriptors, source identities and checksums. No labels, fitted statistics,
candidate head, adjacent-slice context or training enter feature extraction.
Frozen encoding of held-out images is prediction preparation, not adaptation.

Reuse the complete Ticket 49 source/membership and real archive/cache readers.
Verify every packed mask, descriptor and source identity against Ticket 36.
Do not interpret an inventory or partial directory as a completed activation bank.
Seal an immutable manifest only after every expected study and slice is verified.
Keep all identifier-bearing outputs private and preserve existing caches.

## Resource and completion gates

- Freeze source, runtime, exact inputs and resource decision before launching.
- Refresh the 2 TB campaign storage observation. Activation payload projects to
  **37,586,239,488 bytes**; include metadata, checkpoints and temporary writes.
- At most two nonexclusive CPU jobs, each one node, **16 CPUs / 64 GiB / 3 hours**,
  explicit watchdog and no requeue. Assign disjoint immutable partitions; account
  for every accepted attempt. Standing zero-spend owner authorization applies.
- No new GPU request, model download, scoring request, publication or final selection.
- Record actual extraction/read/write timing, peak memory, bytes and terminal
  controller evidence. Require complete source/array/hash verification before
  downstream fitting. Failures require diagnosis and a distinct recovery identity.

## Requirements

- [x] Freeze the exact extraction plan and resource decision.
- [x] Complete all expected activation entries through bounded Slurm jobs.
- [x] Independently verify the complete cohort and native encoder parity.
- [x] Seal manifest and measured costs; specify a separate three-fold adaptation
  hypothesis, seeds, fixed budget, baseline, controls and checkpoints before fitting.

## Comments

The four-report cost scenario estimates 2.305 total CPU-node hours including
three possible 1,500-update folds, with a 2x allowance. Random-read I/O, final
evaluation and full-cohort scaling are unmeasured; this ticket supplies that
missing evidence. No full training or predictive gain has occurred.

## Answer

Resolved October 1 at **07:39:36 UTC**. Extraction jobs **7983 / 7984** and
independent verifier **7985** all completed `0:0`. All **4,228 studies / 187,272
slices**, source identities, masks/descriptors, explicit packed indices and
hashes pass; actual frozen layer4 reconstruction has maximum error **0.0**.
Sealed manifest SHA-256
`21bfeb43e24e8e55797961aeb6f0ad6b313cafad4cdb22ee2aeb2f2870d84e40`.
Files occupy **37,617,892,848 bytes**. Total **2,558 CPU-node seconds**, zero
GPU/spend. [Measured costs and verification](../../../docs/evidence/ticket-57-frozen-resnet-prefix-cohort.md).

[Ticket 58](58-test-report-supervised-layer4-adaptation.md) fixes the separate
three-fold matched warm-head/adaptation hypothesis before any new held-out
predictions. Its full-cache random-read/memory/batch-cost probe is separate;
full fitting and any predictive benefit remain unverified.
