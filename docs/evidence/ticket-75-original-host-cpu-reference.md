# Ticket75 original host CPU reference preparation

Previous goal turn classification: progress. Commit8648ea5 implemented fixed
complete numerical gates and passed234 pilot checks. This turn starts from the
clean worktree and re-polls actual jobs8132/8133 and active T73/T74v3/corrected
reference controllers. They remain live. The real CPU reference remains waiting
for a free owner slot; no timeout or missing receipt permits a duplicate launch.

The new non-launching ticket75_host_cpu_reference script implements the first
three original CPU updates inside a future admitted offline worker. It uses the
shared original masked_bce_loss (masked-index combined supervised-entry mean),
original full-backbone model, ordinary Adam groups/defaults/rates and constant
scheduler. It does not substitute the proposed TPU tensor-safe loss into the
host reference, so the cross-backend gate measures that difference honestly.

Before loading or fitting, it validates exactly three original four-study
16-series×8-slice batches at the model's configured image size, all CPU FP32
pixels/descriptors/targets and boolean masks, valid real MRI for every draw and
nonempty finite bounded supervision. Unknown unsupervised NaNs remain masked
rather than being relabeled. It checks complete initial state/config/metadata
and requires the original initialized-model tensor hash after strict loading.
The original initial host RNG is restored after construction and initial replay.

Each update validates original Adam/scheduler position, computes the original
loss, backpropagates, checks loss/full gradients/model/moments/normalization
before and after update, captures every original parameter gradient and records
loss and actual elapsed interval. Final output retains full CPU-converted step3
model/Adam, all three gradient/loss records, scheduler and host RNG, plus initial
and step3 fixed eight-study batch-two logits. These fields fit the complete fixed
first-three parity gate. It preserves the initial source tensors and normalization
buffers; no registered fit updates or competition candidate are credited.

The production entry point requires actual Python3.12.13/Torch2.8.0+cpu/NumPy2.5.0,
no CUDA, FP32 default and disabled CPU autocast, exact installed optimizer/ordinary
Adam API source hashes, original224-MRI configuration and16 CPU threads before
fitting. The CPU-autocast query matches the pinned
[Torch2.8 autocast implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/amp/autocast_mode.py).
It rejects a host reference result over the fixed180-second phase bound and
restores the prior thread setting. This result-level check does not interrupt a
hung operation; the still-pending supervisor must enforce the actual180-second
process deadline and whole3540-second session watchdog. Source/attachment and
resource admission remain required before calling this entry point.

Validation:9 new host checks plus234 existing pilot checks pass together
(243total). One test runs actual three-update full-backbone CPU forward/backward
with original16×8 packing and synthetic32-pixel inputs, unequal masks and absent
NaNs, and validates all output gradient/moment/loss/logit inventories. It checks
unchanged initial tensors and normalization, changed model weights, scheduler
step3 and observed runtime. Seven malformed-input/config cases fail before model
load/update; the actual local CPU2.7 environment is rejected by the pinned-runtime
production entry point before fitting. These mechanics tests report observed
CPU2.7 and leave pinned CPU2.8 runtime/native MRI/actual TPU proof false. They are
not actual real224-MRI reference execution or independent CPU2.8-versus-CPU2.7
parity. Mypy passes96 source modules; script/tests compile and diff checks pass.

The script remains editable until a separately admitted request freezes it.
Actual private reference preparation/derived packets, lossless source-bound input
loader, rank-state capture/CPU conversion/global writer, fresh PJRT supervisor,
complete private attachment/quota/vacancy admission, pinned hostCPU2.8 execution,
TPU pilot and independent native/license/runtime delivery remain pending. No
frozen live route or original scientific source was modified, no new allocation,
upload or daily scoring attempt occurred. First place remains unproven and the
original campaign goal stays active.
