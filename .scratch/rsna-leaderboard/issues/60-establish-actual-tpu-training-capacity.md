# 60: Establish actual TPU capacity before full-backbone report training

Type: task
Status: claimed
Labels: ready-for-agent
Parent: [RSNA campaign](../spec.md)

## Question

Can the account's unused 72,000-second TPU quota provide a real, offline
PyTorch/XLA accelerator for whole-backbone MRI training? A quota field does
not establish allocation. Fresh upstream issue 1197 reports that scripted TPU
requests may silently execute on CPU; this account has not reproduced it.

## Fixed first stage

One private internet-off notebook, no datasets/competition/model attachments,
no MRI/reports/labels/checkpoints. Request `enable_gpu=false`, `enable_tpu=true`,
`machine_shape=TpuV5E8`, a 600-second session limit and 540-second code watchdog.
One attempt, zero GPU and zero external spend. Charge up to the full 600 TPU
quota seconds if allocation cannot be independently reconciled. Preserve
GPU ledger and existing CPU job 8014.

The notebook records torch/torch_xla/libtpu versions, actual XLA device type,
local/global counts, hardware metadata where available, and a synchronized
FP32 matrix calculation with a known exact result. Missing imports, CPU runtime,
GPU presence, wrong TPU counts or wrong numerical output fail the notebook;
no CPU fallback, dependency installation, automatic retry or fitting is allowed.
Capture receipt before raising, preserve exact remote source/settings and
terminal logs, then independently verify the actual output and quota observations.

## Further scope

A successful environment stage permits preparation of a separate fixed real-MRI
whole-backbone update benchmark using existing admitted weights and report
labels/masks. Freeze inputs, learning policy, compilation/steady-step timing,
checkpoint/replay gates and complete prospective cost before launching it.
No full training campaign, GPU scoring retry, candidate promotion or final
selection follows from either the quota or this first environment test.

## Requirements

- [x] Preserve primary-source TPU quota and scripted-launch concern.
- [x] Freeze and launch exactly one bounded fail-on-CPU environment request.
- [x] Independently inspect actual allocation, calculation, terminal identity,
  remote privacy/network settings and quota accounting.
- [ ] If actual TPU exists, measure real whole-backbone training and delivery
  feasibility under a separate complete resource plan.
- [ ] Record concrete training availability/disposition with no first-place claim.

## Comments

October 2: [Research](../../../docs/research/competitive-next-route-2026-10-02.md)
finds no newly verified complete public route above 0.943. Whole-backbone
report training is a materially different direction from the inconclusive
1,500-step layer4 continuation. This first stage tests allocation only and
does not use the competition leaderboard as a training signal. Standing
owner authorization applies; no new exact numeric owner approval is claimed.

October 2, 01:52 UTC: one request accepted as kernel **136722939 /v1**,
`jitheswar/rsna-tpu-environment-admission-v1`, currently **QUEUED**. Exact
remote source/private/offline/shape/empty inputs verified; local missing-XLA
CPU run fails with its receipt preserved. The accepted response's `/code/`
prefix was reconciled read-only with no second push. TPU ledger reserves 600
seconds, GPU ledger unchanged. See [execution evidence](../../../docs/evidence/ticket-60-tpu-environment.md).
Actual TPU execution and any dependent training feasibility remain unverified.

October2,12:40UTC: independent exact-version verification passes terminal COMPLETE. Actual XLA TPU, eight global and eight addressable devices, no CUDA; synchronized FP32 16x16 result has every entry16 and sum4096. Conservatively charge the complete600-second bound, releasing its reservation; notebook71.683-second elapsed is not allocation accounting. Fresh typed quota12:45UTC shows free GPU57178.249 and TPU71883.65 seconds, paid scaling off. This establishes environment availability only; the separate real-MRI training/checkpoint pilot remains unlaunched. [Evidence](../../../docs/evidence/ticket-60-tpu-environment.md).
