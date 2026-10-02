# 65: Admit pinned BiomedCLIP assets

**Type:** task
**Status:** resolved
**Assignee:** campaign agent

## Scope

After Ticket 64's completed control-state audit, acquire the single pinned
BiomedCLIP checkpoint for prospective private biomedical image-representation
research. Verify current official external-model rules, public access, the exact
Microsoft weight license, full checkpoint digest, restricted CPU tensor loading
and actual vision namespace. This asset stage does not execute the proposed
eight-study MRI pilot, install dependencies, fit a model, alter Ticket 64's
frozen workers, occupy a scheduler slot, submit predictions or publish weights.

Apply the standing owner authorization through a concrete resource decision.
One acquisition attempt, 900-second watchdog, one CPU thread, 4 GiB observed
peak RSS, zero GPU/TPU use and zero external spend. Reserve existing Ticket 64
checkpoint/delivery storage before adding the 783,705,670-byte joint checkpoint
and at most 16 MB supporting metadata. Reject digest mismatch or unsupported
restricted loading; preserve failure and do not retry automatically.

## Acceptance

- [x] Fresh official rules and pinned license/source evidence retained.
- [x] Exact resource decision and storage admission frozen before acquisition.
- [x] Complete bytes and digest independently verified.
- [x] Actual tensor-only loading, namespaces, tensor shapes and memory recorded.
- [x] Asset result explicitly leaves strict model loading, MRI output parity,
  measured throughput and predictive gain for a separately admitted next stage.

## Comments

- October 2: claim before acquisition. Both CPU training slots remain occupied
  by 8032 and 8035; their frozen comparison and verification watchers stay active.

## Answer

The sole acquisition completed in **28.471 seconds**, peak RSS
**1,169,944,576 bytes**. Independent full SHA-256 and byte checks match the
publisher declaration. Restricted loading yields a flat state dictionary,
**352 tensors**, including **151 finite FP32 vision tensors /344,767,488
bytes**, with no prefix or unrestricted fallback. Projection `[512,768]` and
positions `[1,197,768]` are observed shapes, not strict model coverage.

Six boundary tests pass. Fresh rules and exact MIT notice/config/card are
retained. Projected storage including retained Ticket 64 reserves and this
asset is **1,834,491,103,747 bytes**, below 2 TB. No scheduler job, MRI/label
read, install, model forward, fitting, submission, GPU/TPU use or spend.
Next is separately admitted isolated-runtime strict vision loading and a fixed
eight-study original-MRI parity/cost pilot; predictive gain remains untested.
Context: [asset evidence](../../../docs/evidence/ticket-65-biomedclip-assets.md).
