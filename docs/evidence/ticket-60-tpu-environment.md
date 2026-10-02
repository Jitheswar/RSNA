# Ticket 60 — actual TPU environment admission

One private, offline environment probe was accepted on October 2 at
**01:48:42 UTC**, handle
`jitheswar/rsna-tpu-environment-admission-v1`, numeric kernel **136722939**,
version **1**. Latest observation **01:52:11 UTC: QUEUED**. Actual TPU hardware,
framework availability and the synchronized calculation remain **unverified**.
No MRI, reports, labels, checkpoints or model attachments were uploaded.
No training, inference, competition submission or external spend occurs.

## Fixed request and authorization

[Research](../research/competitive-next-route-2026-10-02.md) found no newly
verified complete public route above 0.943. The account's nominal unused
72,000-second TPU allowance provides a concrete allocation question; the
upstream silent-CPU report is not a verified failure on this account.

The [prospective issue](../../.scratch/rsna-leaderboard/issues/60-establish-actual-tpu-training-capacity.md)
and [agent decision](../../.scratch/rsna-leaderboard/resource-approvals/ticket60-tpu-environment-20261002.json)
freeze one request: `enable_gpu=false`, `enable_tpu=true`, `TpuV5E8`, private,
internet off, no attached inputs, server limit **600 seconds**, code watchdog
**540 seconds**, one attempt, no automatic retry or dependency installation.
Standing owner authorization applies; no new numeric owner approval is claimed.

Request SHA-256:
`5fa413ef63f1dd2cf5ddb4a88f2d33077b6ffc28f9af6769a1b17c7b67d3113f`.
Private root: `artifacts/private/ticket60-tpu-environment-v1/`.
The [TPU ledger](../../.scratch/rsna-leaderboard/tpu-campaign-ledger.json)
reserves **600 seconds**. This is a conservative campaign bound, not proof of
Kaggle charging semantics or successful hardware allocation. GPU ledger remains
unchanged at **129,637 charged /136,800 cap /zero reserved**.

## Verified request, pending actual execution

- Fresh quota before launch has TPU used/reserved zero, nominal allowance
  72,000 seconds and paid scaling off. The malformed GPU literal is preserved.
- Canonical remote version and numeric kernel identity match the response.
  Exact cell types and executable source match the frozen notebook; private,
  offline, GPU-off, TPU-on, requested shape and all empty input lists pass.
- The push response used `/code/` before its canonical handle. The launch
  receipt check initially rejected that prefix. Read-only exact remote identity
  reconciliation recovered the accepted response; **no second push occurred**.
  The launcher now normalizes only that literal prefix and refuses an existing
  launch intent, including ambiguous previous attempts.
- Executing the generated code on the local CPU runtime produces exit **1**,
  `ENVIRONMENT_ADMISSION_FAILED`, `ModuleNotFoundError: torch_xla`, after saving
  its receipt. This verifies rejection of the actual missing-XLA CPU runtime;
  it is neither a Kaggle result nor proof of positive TPU execution.
- Three preparation/launch/verification scripts compile. No project source
  model implementation or MRI training configuration changed.

Actual allocation must be verified using the terminal receipt: XLA device type
TPU, eight global devices, addressable devices, no CUDA, installed versions,
and a synchronized FP32 16x16 calculation whose every entry is 16 and sum 4096.
The observer is `rsna-ticket60-tpu-environment-v1-observer.service`; it checks
the exact saved version without launch or retry. On terminal status, run
`/home/ch.jitheswar/.venvs/kaggle-cli-2.2.4/bin/python scripts/verify_ticket60_tpu_environment.py`.
Preserve any failure/unknown allocation and conservatively charge the full
600-second bound if accounting is unresolved. A successful admission permits
planning a separate real-MRI whole-backbone throughput/checkpoint-delivery
experiment; it does not justify a full training campaign or model promotion.

CPU CoAt job **8014** continues independently and must not be restarted for a
polling timeout. Neither public nor private first place is established.
