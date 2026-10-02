# 66: Measure BiomedCLIP vision MRI parity and cost

**Type:** task
**Status:** claimed
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
- [ ] Sole CPU pilot launched within owner concurrency and terminal reconciled.
- [ ] Strict state coverage, original-MRI/output parity and cost independently checked.
- [ ] Exact completion, continuation action and limits documented; no gain inferred.

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
