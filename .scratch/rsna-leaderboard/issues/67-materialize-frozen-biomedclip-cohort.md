# 67: Materialize the frozen BiomedCLIP cohort

**Type:** task
**Status:** claimed
**Assignee:** campaign agent
**Blocked by:** 66

## Scope

After the successful independent eight-study CPU admission, extract one
unfitted BiomedCLIP representation for the exact 4,228 studies /187,272 valid
slices already admitted by Ticket 62. Preserve whole-field pixels, ordered
valid indices, masks and descriptors; FP32 microbatch one, pinned official
vision weights, 512-D learned projection and explicit L2 normalization.
No text tower, label use, fitting, AUC, public submission or promotion.

Freeze four partitions balanced by valid-slice count. Derive complete cost
from actual maximum slice inference and complete verified pixel-bank I/O,
including startup, serialization and independent validation allowance. Protect
Ticket 64's original route/storage/review reserves and October 11 cutoff.
At most two pending/active owner CPU jobs, 16 CPUs /64 GiB /three hours per
partition, one attempt each, no retry/requeue, zero accelerator use or spend.
Fit scheduling handoffs preserve every live job and scientific stage.

Independently validate every feature's source/bytes/shape/norm/padding/masks,
cohort completeness and actual terminals. Replay the fixed eight original-MRI
pilot outputs and compute unfitted anatomical pooled features. Separate paired
head fitting and later exact anchor/delivery comparison need their own decision.

## Acceptance

- [x] Actual successful Ticket 66 and full pixel-source integrity checked.
- [x] Complete storage/calendar/runtime/partition/source admission frozen.
- [ ] Four sole extraction attempts complete with durable exact CPU terminals.
- [ ] All 4,228 feature outputs and fixed native outputs independently verified.
- [ ] Full measured I/O, cost, feature identity and next decision recorded.

## Comments

- Actual successful native pilot and all-source pixel proof verified. Immutable
  cohort plan `e9ae761facc5ca3345925f38c36682ac18ff94e6e92d5c018e8a3eb7a9f5256a`;
  four partitions each 1,057 studies, 46,824/46,816/46,816/46,816 slices.
  Protected full route plus review projects October 10 05:05 UTC, storage
  1.849 TB below 2 TB. Four 3-hour extraction allocations plus one 1-hour
  independent verification, one bank job plus unchanged registered fitting
  within two global owner CPU slots.
- Persistent combined scheduler is live, PID 293555, waiting for a CPU slot;
  both 8036/8039 still run. It advances the exact unchanged Ticket 64 stage
  schedule itself and restores the original coordinator on exit. Do not
  manually restart that coordinator during this ownership interval.
- Fourteen focused guards and five launcher checks pass. Actual extraction,
  whole-bank validation and measured cohort cost remain pending.
- [Admission evidence](../../../docs/evidence/ticket-67-biomedclip-feature-cohort.md).
