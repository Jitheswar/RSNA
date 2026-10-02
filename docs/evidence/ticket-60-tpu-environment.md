# Ticket 60 — verified actual TPU environment

Independent verification on **October2,12:40:29UTC** passes for the sole private
offline probe, kernel **136722939/v1**, terminal **COMPLETE**. It verifies actual
XLA **TPU**, **eight global and eight addressable devices**, no CUDA devices,
and synchronized FP32 16×16 matrix multiplication: every entry16, sum4096.
This permits preparation of the separately admitted real-MRI training pilot.
It does not establish MRI training throughput, model gain or either rank.

The exact remote executable source, private/offline settings, `TpuV5E8`, GPU-off,
TPU-on and empty dataset/model/competition attachments match the frozen request.
No MRI, reports, labels, weights, fitting, model inference or scoring occurred.
RequestSHA `5fa413ef63f1dd2cf5ddb4a88f2d33077b6ffc28f9af6769a1b17c7b67d3113f`.
Handle `jitheswar/rsna-tpu-environment-admission-v1`.

Actual runtime: Python3.12.13, Torch2.8.0+cpu, torch_xla2.8.0, libtpu0.0.17,
NumPy2.5.0 and timm1.0.27. Calculation synchronization took0.332seconds.
Notebook elapsed71.683seconds does not establish the entire allocation duration.
The [TPU ledger](../../.scratch/rsna-leaderboard/tpu-campaign-ledger.json)
therefore charges the full **600seconds**, releases its reservation and retains
conservative campaign remaining71400seconds. No external spend.

Retained ignored root `artifacts/private/ticket60-tpu-environment-v1/`:

- Independent verificationSHA
  `eaad4dd0ed158c9478bc749195593f31b1a3f315d1b3b53031e374289b9e254c`.
- Actual runtime receiptSHA
  `dfe06f642a3efe9d37357072ab7cddc3474192748248f266a7db232303dbc5c1`.
- Fresh typed terminal quotaSHA
  `34aa09c491a78537ff93228f16146d9f038f471e7faa9c0eb6a33cd067e1b8bb`.

Authenticated typed timedelta quota at **12:45:03UTC** reports GPU allowance108000,
used50821.751, reserved0, **free57178.249seconds**; TPU allowance72000,
used116.35, reserved0, **free71883.65seconds**. Paid scaling is off for both.
The SDK refresh field is `2026-10-03T00:00:00`, without timezone. Earlier malformed
serialized duration literals are preserved and never parsed; quota observations
are distinct from conservative campaign accounting and require refresh at launch.

The successful verifier checked the GPU ledger unchanged since the consumed
launch intent, before any subsequent GPU amendment. Do not rerun this consumed
verifier after unrelated GPU ledger changes. Inspect its retained proof instead.
The probe observer has exited successfully; never push or restart its intent.

The sole request was accepted01:48:42UTC after the initial quota/source checks.
Its response `/code/` prefix was reconciled read-only; no second push occurred.
Local missing-XLA execution failed as intended; actual remote output supplies the
positive hardware proof. The server/code bounds were600/540seconds, one attempt,
no retry, no package installation. Standing owner autonomy applies, without a
new exact numeric owner approval.

[Ticket60](../../.scratch/rsna-leaderboard/issues/60-establish-actual-tpu-training-capacity.md)
remains claimed: real whole-backbone MRI update, compile/steady timing,
checkpoint/replay and delivery feasibility need a separate frozen resource plan.
The actual TPU runtime differs from the admitted CPU medical runtime; hardware
proof supplies no weight, optimizer or prediction equivalence.
