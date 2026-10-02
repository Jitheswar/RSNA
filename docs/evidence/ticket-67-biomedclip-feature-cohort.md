# Ticket 67: frozen BiomedCLIP feature cohort admission

The complete unfitted feature bank is admitted and its persistent scheduler
is live. Sole first partition **8041** is **RUNNING**, with 64 studies observed
complete after 216.537 process seconds. It was admitted only after fit **8036**
completed its fixed 2,000-update boundary successfully. Fit **8039** continues;
exactly two owner CPU jobs are live. Actual full-cohort outputs and independent
validation remain pending; no accuracy or rank result follows.

## Evidence and fixed representation

[Ticket 66](ticket-66-biomedclip-mri-admission.md) sole **8037 /0:0**, 217
CPU-node seconds, independently reproduced all eight original-MRI studies
/344 projected outputs exactly. Preparation rehashed every admitted runtime,
source and input, checked the successful terminal/result/native replay receipts,
and verified Ticket 62's complete original-source pixel proof and exact
**4,228 studies /187,272 valid slices**.

Pinned BiomedCLIP revision `9f341de24bfb00180f1b847274256e9b65a3a32e`, strict
151 FP32 vision tensors, original whole-field 224-pixel grayscale repeated RGB,
pinned CLIP mean/std, microbatch one, learned **512-D projection with explicit
L2 normalization**. No text tower, fitted preprocessing or target values.
All masks, descriptors, valid-slot ordering and original source identities
are retained. Every output carries byte and semantic-array digests.

| Frozen admission | SHA-256 |
| --- | --- |
| Cohort plan | `e9ae761facc5ca3345925f38c36682ac18ff94e6e92d5c018e8a3eb7a9f5256a` |
| Feature identity | `7ba6b06160d908c6e03c2edb38bae73b0d90e50ab27c0eef2bbd1ce6cf3ac230` |

Private root `artifacts/private/ticket67-biomedclip-cohort-v1/`; **14,741**
frozen source/runtime/admission files. Snapshot and matching four repository
scripts must remain immutable after preparation. The resource decision is an
agent decision under standing owner authority, with no new numeric owner
approval claimed.

## Complete cost and scheduling

Four label-blind partitions, each **1,057 studies**, are balanced by valid-slice
count: **46,824 /46,816 /46,816 /46,816**. Conservative forecast per partition
is `2 × (measured load + slices × maximum slice time + studies × maximum
verified cache-read time) + 600 seconds`: **7,363.519 /7,362.408 s**, below
**10,740 s** code watchdog and **10,800 s** allocation. Cache inputs are
exact independently verified original MRI pixels, avoiding repeat decoding.
Each partition has one attempt; final independent validation has one
**3,600 s** allocation and **3,540 s** watchdog. Every job uses 16 CPUs /
64 GiB, nonexclusive, no requeue, no accelerator, no external spend.

CPU-node reserve **46,800 s /13 h**, additional calendar reserve **64,800 s
/18 h**, including five one-hour queue allowances. The complete prior Ticket
64 **523,800 s** route and one review day are retained conservatively;
projected completion **October 10 05:05:15 UTC**, before October 11 science
freeze. Fresh apparent storage plus protected prior checkpoints, recovery
margin and 2 GB new bank allowance totals **1,848,889,097,881 bytes /2 TB**.
Maximum padded feature payload **1,108,344,832 bytes**; pooled matrix
**35,109,312 bytes**. No conditional additional full fit is authorized.

`rsna-ticket67-biomedclip-cohort-v1.service`, PID **293555**, owns combined
scheduling. It holds the original Ticket 64 fit-coordinator unit and itself
advances the exact existing registered stage schedule through frozen worker
launches. Every live fit and scientific update remains intact. One bank job
at a time, with at most **two pending+active owner CPU jobs** globally. Original
fit scheduling is restored on completion or controller exit. Do not manually
restart the old fit coordinator while this service owns scheduling, and do
not relaunch any consumed bank/controller intent. Separate Ticket 64 final
verification remains active and unchanged.

## Independent completion gate

All four exact successful CPU terminals and partition identities must pass.
A fresh verification process checks every feature file's bytes, source/pixel
identity, finite FP32 shape, valid-slice unit norms, invalid zeros, masks,
descriptors and indices, and exact cohort membership. It compares the fixed
eight bags against independently replayed original-DICOM pilot bytes and
re-infers all **344** slices. It produces unfitted **2,076-D anatomical pooled
vectors** for every study and a complete immutable feature manifest.

The existing all-study native pixel proof and fixed eight-study native vision
proof remain distinct: this does **not** claim a fresh original-DICOM vision
replay of all 4,228 studies. Actual materialization/verification CPU cost and
successful final terminal are required before any separate paired head-fit
admission. No labels, training, AUC, submission or promotion are authorized.

**14 focused tests pass**, covering partition membership/order, feature padding,
unit vectors, two-slot ownership, failure restoration, consumed-intent refusal
and exact CPU terminal identity/resources/exit. All four scripts parse and all
five frozen launcher hashes/bash syntax pass. GPU/TPU ledgers remain unchanged.
