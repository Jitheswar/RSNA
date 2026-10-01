# 57: Materialize deterministic ResNet prefix activations

Type: task
Status: ready-for-agent
Labels: ready-for-agent
Parent: [RSNA leaderboard campaign](../spec.md)
Blocked by: 56

## Purpose and scope

Ticket 56 demonstrated exact native/cached last-block adaptation parity and
plausible CPU cost. Prepare and measure the complete activation cohort so the
next experiment can adapt fold-specific encoders with admitted report supervision.
This is the next useful CPU action while exact D4 submission 56739912 is pending.

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

- [ ] Freeze the exact extraction plan and resource decision.
- [ ] Complete all expected activation entries through bounded Slurm jobs.
- [ ] Independently verify the complete cohort and native encoder parity.
- [ ] Seal manifest and measured costs; specify a separate three-fold adaptation
  hypothesis, seeds, fixed budget, baseline, controls and checkpoints before fitting.

## Comments

The four-report cost scenario estimates 2.305 total CPU-node hours including
three possible 1,500-update folds, with a 2x allowance. Random-read I/O, final
evaluation and full-cohort scaling are unmeasured; this ticket supplies that
missing evidence. No full training or predictive gain has occurred.
