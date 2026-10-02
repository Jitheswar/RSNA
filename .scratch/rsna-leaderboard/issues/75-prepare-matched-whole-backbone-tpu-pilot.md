# 75: Prepare the matched whole-backbone TPU mechanics pilot

Type: task
Status: claimed
Labels: ready-for-agent
Parent: [RSNA campaign](../spec.md)

## Scope

Follow the fixed [pilot design](../../../docs/research/tpu-whole-backbone-pilot-design-2026-10-02.md)
after the genuine eight-device environment proof. Preserve fold0, original
38-reference/2731-report ordered pools, first16 global batches,2+2 real draws,
four empty padding ranks, FP32/highest backend precision, unchanged original
full-backbone model, warm head, Adam rates and all state/native replay gates.
These disposable mechanics states do not count toward T64's7500 additional
updates/fold or establish predictive gain.

No resource, attachment upload or TPU launch is admitted by this issue or the
preflight. Fresh combined calendar/storage, account quota, sole accelerator
vacancy and owner CPU slot admission must include existing T73 and T74 reserves.
T74's prospective additional7200 CPU-node seconds cannot be silently omitted.
Preserve all frozen live routes; arrange any new admission dependencies before
creating outstanding pilot reservations. No competition submission or retry.

## CPU preflight already established

The new independent replicated_masked_loss module admits original four-study
FP32 supervision on CPU, maps ranks0/1 reference and2/3 weak, and maps4–7 to
zero supervision. Its tensor-only replica loss is8*N_rank/D_global for a mean
collective; it does not average local means or count padding as real exposure.
It performs no device scalar read or variable-length index selection.

Unequal-mask loss/gradient checks pass. The original full-backbone synthetic
fixture also matches gradients and one ordinary Adam update, with normalization
buffers unchanged. A separate synthetic16-update test matches ordinary global
BCE Adam/model state within the predeclared engineering tolerance and replays
replicated training exactly after a fresh checkpoint load. These are CPU
mechanics tests, not actual MPI/PJRT, real224-MRI or TPU proof.

## Attachment envelope clarification

The real frozen pools and original sampler select exactly32 reference+32 report
studies in the first16 batches,64 distinct studies. Uncompressed FP32 pixels
alone are1,644,167,168 bytes. Three original full-checkpoint upper bounds add
405,213,507 bytes, exceeding2GB even before the initial state or metadata.
Retain CPU reference states0/3/8/16 on the cluster. Attach only the initial state
and the step3 CPU reference alongside the original64 MRI and targets/masks.
Using the conservative135,071,169-byte bound for each attached state gives
1,914,309,506 bytes before code/metadata, leaving85,690,494 bytes within2GB.
Hash and enforce the actual complete bundle before upload; no lossy conversion,
fewer studies/updates or dropped replay gate. Steps8/16 remain retained for the
independent CPU verification. The complete retained-assets envelope remains8GB.

Private sampler/envelope receipt:
artifacts/private/ticket75-tpu-loss-preflight-v1/sampler-and-input-envelope.json,
SHA256 fbb5131d28acce6fa7d642a57357546355d90a7016529b1903cfece0491a928c.
It contains indices and original cohort/source hashes, not a native-MRI or
resource-admission claim.

## Acceptance

- [x] Prove correct unequal-mask four-real/eight-rank loss and gradient reduction.
- [x] Prove ordinary Adam and fresh CPU checkpoint replay in synthetic preflight.
- [x] Verify original pool/sampler selection and feasible lossless attachment layout.
- [x] Pin real64-study source/cache selection and original-pool order without held-out overlap.
- [ ] Freeze real64-MRI source/targets/masks, initial state and CPU reference request.
- [x] Admit the complete calendar/storage route with joint T74 ownership and retained completed-fit evidence.
- [x] Freeze the one bounded CPU reference request and start its exact waiting controller.
- [ ] Obtain an actual free CPU slot and complete the original64-MRI reference preparation.
- [ ] Verify private pinned attachment, fresh quota/vacancy and exact TPU runtime/source.
- [ ] Run the one fixed TPU pilot, preserving all compile/precision/state/replay failures.
- [ ] Independently reproduce native CPU delivery and account the entire allocation.
- [ ] If mechanics pass, prepare a separate full-training feasibility proposal.

## Comments

October3 IST: real source-selection metadata b08d52294efac12435fb49717ffac8b6f51bfb53c22fb1dc7658fb834e2ea8ce pinned under artifacts/private/ticket75-real-mri-selection-v1. Complete original T73/old scientific input closure and seven metadata hashes checked;64 training studies/no held-out overlap. Original source/cache hashes, pool order, full-pool input fingerprints and supervision admission retained. No MRI/target arrays or initial state exported, no CPU reference request/resource admission.17new schema/selection/bundle guards and mypy90 PASS. New tpu_pilot_inputs and freeze script now metadata-hash-bound; preserve them. CPU worker and combined T73/T74/pilot route ownership remain pending. Evidence docs/evidence/ticket-75-real-mri-selection-contracts.md.

October3 IST: implemented CPU reference executable and17 fail-closed admission guards;53 combined admission/input/loss tests PASS, Python compilation and mypy90 PASS. Worker remains unfrozen/unlaunched, no request/reservation/upload. It implements original64 native-source replay and staged3/8/16 shared-trainer reference with model/Adam/scheduler/mask-exposure/normalization checks, initial/step3 attached and8/16 local, actual2GB/8GB gates. Fresh storage plus unchanged old future envelope exceeds2TB through completed-fit double counting: explicit proved completed-fit credits and joint T74 future reservation ownership required before admission. Preserve live T73 and frozen T74 waiter. Evidence docs/evidence/ticket-75-cpu-reference-worker.md. No scored submission used.

October3 IST: joint route cd3a801db6d26287644247fde8e4f9559fd1a79ba5a8ee143a72d08caa99f7db admitted under standing authorization. Genuine8034 completed-control checkpoints retained; credit only230566485483 bytes duplicate future control allowance. Entire original calendar retained plus7200 T74 and28800 pilot, finishOct9 10:22UTC/storage1784433500971B. Frozen T74 v2 waiter PID1199029 actually live, originalv1 stopped and preserved, same scientific protocol/plan. CPU request f9977c944803733e6565cc5b23c30b4d27b04128ff7ef51c335d3d6a65be4428 frozen; exact waiting controller PID1203487 live, no CPU launch intent/acceptance yet because both authorized slots occupied. No TPU quota reservation/accelerator admission or scoring.82tests and mypy90 PASS. Evidence docs/evidence/ticket-75-joint-resource-admission.md. New route/CPU source bindings now immutable; preserve them.

October3 IST: added lossless restricted-load CPU reference/RNG packet contract and post-success local converter. Reproduced original NumPy RNG checkpoint restricted-loader failure and proved converted weights_only=True loading with original bytes unchanged. Synthetic fresh step8→16 exact model/Adam/scheduler/sampler and Python/NumPy/Torch RNG replay, including Gaussian caches and ordered model metadata, passes.18state tests/54combined tests and mypy91 PASS. Actual converter guard correctly rejects pending CPU preparation and creates no output; controller PID1203487 genuinely live with both CPU slots occupied. No real packets/export, final TPU attachment/upload/launch or score. All frozen worker/controllers remain unchanged. Evidence docs/evidence/ticket-75-portable-reference-state.md. Portable module/converter not yet frozen by an execution request; actual TPU adapter and independent host/reference gates remain pending.

October3 IST: implemented fixed CPU global→rank input mapping with original224/16×8 geometry, exact2+2 ordering, four fully empty padding ranks/common supervised-entry denominator, and complete16-mask prefix source accounting. Added complete256-observation gates for continuous16/staged8+fresh8, actual graph/compile/execute/fallback evidence, rank7 failures, steady ordinary-Adam graph churn and fixed max-rank timing.16input+22telemetry guards;92combined tests and mypy93 PASS. Synthetic observations explicitly do not prove actual TPU/full-phase budget/MRI/state parity. No new allocation/upload/scoring; actual T73/T74v2/CPU waiting handles remain live. New modules editable until future TPU request freeze. Actual TPU worker/supervisor and hostCPU2.8/fullbundle/quota/independent-delivery work remain pending. Evidence docs/evidence/ticket-75-replica-input-and-telemetry-gates.md.

October3 IST: implemented non-launching pinned XLA update adapter: deferred device initialization, explicit admitted worker/exact API source/runtime/highest FP32/eight-rank guards, original full-model/Adam/scheduler scope, exactly one mean gradient reduction, phase graph hashes and raw metric/fallback inventories, timed world finite/normalization guards and mandatory first3 reduced-gradient export.16new CPU optimizer/parser/admission checks plus existingpilot guards=108PASS, mypy93/compilation PASS. No actual XLA update claimed. Actual CPU reference PID1203487 live/no accepted job yet,8128/8127 RUNNING. All frozen routes preserved. New update adapter remains editable until future TPU request freeze; actual hostCPU2.8/full-worker fresh-process/checkpoint-RNG/bundle/quota/run/delivery remain pending. Evidence docs/evidence/ticket-75-pinned-xla-update-adapter.md. No daily submission used.
