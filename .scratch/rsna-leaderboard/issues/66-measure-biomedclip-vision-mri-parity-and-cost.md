# 66: Measure BiomedCLIP vision MRI parity and cost

**Type:** task
**Status:** resolved
**Assignee:** campaign agent
**Blocked by:** 65

## Scope

Prepare an isolated CPU runtime using official OpenCLIP 2.23.0, its recommended
Transformers 4.35.2, and the already admitted Torch/TorchVision/timm environment.
Pin dependency-wheel publisher digests before acquisition; preserve existing
environments and Ticket 64 workers. Then separately freeze one offline 16-CPU,
64-GiB pilot with a 900-second code watchdog and 960-second allocation.

Select exactly eight report-only pixel studies by ascending SHA-256 of UID,
without label values. Preserve the original whole-field 224 pixels, valid-slot
order, masks and descriptors. Strict-load the 151 actual vision tensors into
the official vision wrapper and official factory-built reference, without
constructing text towers or fetching text assets. Use FP32 microbatch one and
explicit L2-normalized 512-D projected embeddings. Replay original MRI packing,
require exact pixel identity and <=1e-6 reference-output difference, finite
nonconstant unit vectors, repeatability, measured load/inference cost and RSS.
Do not fit, inspect AUC, choose alternate layers, submit or promote a candidate.

Respect at most two active/pending owner CPU jobs. Any temporary interruption
of the fit coordinator must preserve running jobs, every scientific update and
the frozen stage schedule; resume scheduling immediately after admitting the
single pilot. No retries or accelerator use. Full extraction requires measured
cost and a separate exact decision.

## Acceptance

- [x] Pinned isolated runtime acquired and original official source bytes verified.
- [x] Selected cohort, native/cache contracts, source/runtime/resource plan frozen.
- [x] Sole CPU pilot launched within owner concurrency and terminal reconciled.
- [x] Strict state coverage, original-MRI/output parity and cost independently checked.
- [x] Exact completion, continuation action and limits documented; no gain inferred.

## Comments

- October 2: six exact runtime wheels acquired; failed first import identifies
  absent transitive wcwidth, resolved by one separately admitted pinned addition.
  Offline imports pass and official vision source is byte-identical. No model
  forward yet. Eight label-blind report studies /344 slices and 14,643 runtime/
  input/source identities are frozen. Eleven selection/handoff tests pass.
- Persistent `rsna-ticket66-cpu-handoff-v1.service` is confirmed live waiting
  on 8035/8036. Both fits run; only future fit scheduling is temporarily held.
  The controller admits the one pilot after a successful stage frees a slot,
  resumes the exact fit coordinator immediately and retains actual terminal.
  Pilot plus independent original-DICOM replay share one 16-minute allocation;
  neither has executed yet. Preserve all frozen sources/consumed intents.
- Context: [admission evidence](../../../docs/evidence/ticket-66-biomedclip-mri-admission.md).

## Answer

Sole pilot **8037** completed `0:0`, **217 CPU-node seconds**. Strict official
151-tensor coverage, all original pixels/masks/descriptors and all **344**
projected outputs from eight studies independently replay exactly, maximum
error **0**. Median slice inference **0.038754 s**, maximum **0.069421 s**,
vision load **3.133 s**, peak RSS **3,109,085,184 bytes**. All outputs are
finite, normalized and image-dependent. The consumed handoff restored the
unchanged fit scheduler immediately after pilot admission; fold-1 stage 1
**8039** starts after successful **8035**, alongside **8036**. No fitting,
AUC, submission, promotion or accelerator use follows from this result.

Ticket 67 separately admits the complete cached-pixel feature cohort, protecting
existing fitting, storage and review reserves. See [completed evidence](../../../docs/evidence/ticket-66-biomedclip-mri-admission.md).
