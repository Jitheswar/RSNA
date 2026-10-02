# 67: Materialize the frozen BiomedCLIP cohort

**Type:** task
**Status:** resolved
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
- [x] Four sole extraction attempts complete with durable exact CPU terminals.
- [x] All 4,228 feature outputs and fixed native outputs independently verified.
- [x] Full measured I/O, cost, feature identity and next decision recorded.

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

- October2 10:06UTC: **8036** completed `0:0`, exactly2,000 updates,
  6,857 CPU-node seconds, checkpoint
  `e9fdceb3af12099a71e250b5aa9028abf5816e17c81919a005b3c2658d7a0668`.
  Sole first extraction **8041** is RUNNING, first64 studies at216.537 process
  seconds; **8039** continues. Two-slot limit preserved. Matched-head Ticket68
  controller waits for complete independent bank proof without taking slots.

- October2 11:12UTC: first8041 completed0:0/1901 CPU-node seconds, second
  8042 completed0:0/1831 seconds; both1057 studies, respectively46824/46816
  valid slices. Third8057 runs beside unchanged full8054(fold0/stage2).
  Independent all4228 feature proof and fourth partition remain pending.

- October2 11:38UTC: third8057 completed0:0/1896 CPU-node seconds,
  1057 studies/46816 slices; three-part total3171 studies/140456 slices/5628
  seconds. Fourth8058 runs beside unchanged full8054. Independent complete
  bank proof remains pending; no scientific schedule change or scoring.

## Answer

All four sole partitions8041/8042/8057/8058 completed0:0, respectively
1901/1831/1896/1905 CPU-node seconds. Independent verifier8071 completed0:0
/352 seconds. Total7885 CPU-node seconds,4228 studies/187272 valid slices,
1178253654 bytes for feature plus pooled files. Extraction read124.714s,
inference7258.738s, writes20.288s; independent pixel read171.753s.

Every feature byte/semantic array/source/mask/descriptor/index/unit norm/padding
and complete cohort was checked. Fixed344 native pilot outputs and fresh
official vision replays match exactly. This is not all4228 native vision replay.
No labels, fitting, accelerator seconds, AUC, promotion or scoring.

ManifestSHA `b5c136b5d0d44e7e9dcecb040c8e087bbac78258811e112c965bc2fc7ac4f798`;
independentSHA `1a3424090bc59cc227d64be4dd1658f4a87f7c4b0c480f3322fe39611c803fe9`;
terminal verificationSHA `f7348df1d46d8fb0ef8a58e6947c5baa6fac1b23f1679483b6a96097d77413f3`.
Completion auditSHA `d4a18be222866aaf4e8191e961c8e282222109fb59d430869262f08c003d4ebc`.
Ticket67 controller exited/restored original scheduling; Ticket68 subsequently
admitted its exact matched plan and now owns the combined unchanged schedule.
Preserve all frozen sources and consumed intents.
[Completed evidence](../../../docs/evidence/ticket-67-biomedclip-feature-cohort.md).
