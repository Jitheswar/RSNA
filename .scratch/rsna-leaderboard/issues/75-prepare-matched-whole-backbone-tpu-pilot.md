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
- [ ] Freeze real64-MRI source/targets/masks, initial state and CPU reference request.
- [ ] Admit the complete route and schedule one bounded CPU preparation in a free slot.
- [ ] Verify private pinned attachment, fresh quota/vacancy and exact TPU runtime/source.
- [ ] Run the one fixed TPU pilot, preserving all compile/precision/state/replay failures.
- [ ] Independently reproduce native CPU delivery and account the entire allocation.
- [ ] If mechanics pass, prepare a separate full-training feasibility proposal.
