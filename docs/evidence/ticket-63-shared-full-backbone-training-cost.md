# Shared full-backbone training: complete cost and exact staged replay

Ticket 63 passes its cost/reproducibility gate. Job **8018** completed `0:0`
in **2,170 allocated CPU-node seconds**, using one nonexclusive 16-CPU/64-GiB/
90-minute allocation, one attempt, no requeue, GPU, external spend, network,
new weights or held-out prediction. Independent verification at **06:06:37 UTC
on October 2** binds original supervision, admitted weights, executed source,
resource decision, actual accepted job and retained successful terminal.

All **4,228** file-backed MRI pixel entries were admitted without retaining
complete pixel arrays. Original fold training pools contain **38/39/39**
reference studies and **2,731/2,846/2,749** admitted report studies. Every
target/mask and the actual mixed-batch source exposures were independently
reconstructed from the original audited admission. All runs start from the
fold-specific 7,500-update report head and original ImageNet encoder.

Five fresh processes perform complete admission, one-update interruption,
resume to three updates, continuous three updates and independent state
comparison. Resumed and continuous **model, Adam, scheduler, progress and
sampler states match exactly**. Stem, layers 1–4 and head change; normalization
running buffers remain fixed. Strict fresh checkpoint reload passes. These
three diagnostic updates establish mechanics and cost, with no AUC or gain.

| Actual cost | Observation |
| --- | --- |
| Complete pixel read/admission | 126.443 seconds |
| Per-fold construction | 97.390–100.800 seconds |
| Per-fold fingerprinting | 185.434–192.671 seconds |
| Maximum fresh training startup | 372.921 seconds |
| Maximum update/prior-checkpoint interval | 8.745448 seconds |
| Maximum checkpoint file | 135,071,169 bytes |
| Strict checkpoint reload | 0.259566–0.479378 seconds |
| Peak process RSS | 16,871,268,352 bytes |

The maximum interval includes serialization at every diagnostic update. Future
fits also checkpoint at short reference-epoch boundaries, so projecting only
500-step periodic files would undercount storage. Complete route admission
must include all such histories, repeated startup, final held-out inference,
original-source replay and conditional final-fit/offline delivery reserves.

A separate deterministic regression exposed planned pauses publishing stale
step-0 state at a step-2 pause between ordinary checkpoint boundaries. The
shared trainer now publishes the exact planned stopping state before raising
`TrainingInterrupted`. Unexpected failures still replay only uncheckpointed
updates. **27 relevant lifecycle/training tests pass**, including exact
resumed-versus-continuous results; typing passes **84 source files**. This
correction changes operational checkpoint publication, preserving sampling,
optimizer choices and scientific duration.

Private root: `artifacts/private/ticket63-full-backbone-training-cost-v1/`.
Plan SHA-256:
`40f5268a019a9d247cb3ced8d0d7a62c419bd7b1e544a1370af7f884d65f7e39`.
Cost result SHA-256:
`624894c3366b482fdff1c02eb7c1de94fce6496d7c74c3eea86a2c8af22862e6`.
Independent receipt: `terminal-verification.json`; exact terminal persists
under `controller-8018/` after Slurm purges its controller record. Preserve
the consumed request; do not rerun it.

Disposition: **eligible for separately registered complete fitting admission**.
[Ticket 64](../../.scratch/rsna-leaderboard/issues/64-test-long-full-backbone-report-adaptation.md)
prospectively registers the 7,500-additional-update three-fold matched
comparison. No candidate promotion or first-place evidence follows from cost.
