# Complete public frontier inference — September 30, 2026

The corrected visible dry run of the public D4 recipe completed on exactly
two T4s. Independent verification checked **30 prediction artifacts**, exact
study/target ordering, final CSV equivalence, finite output, all four CoAt
success receipts and **zero reported fallback events**. Code execution took
**340.948 seconds** for three visible test studies. This establishes complete
visible inference, with no measured leaderboard improvement or hidden-runtime
claim. The [upstream scored source](https://www.kaggle.com/code/yamadan96/rsna-knee-d4-public0946?scriptVersionId=353578144)
has its author's public score **0.943**; this account's completed earlier
baseline scored **0.936**.

All twenty DINO predictions, five A5 folds, the RadImageNet branch, four Raptor
views and four CoAt readers ran. CoAt used residual-gated epochs 4/6/8,
Global96 epochs 16/23/18, Repair-v1 epochs 12/7/11 and the pinned D4 SWA/adapter.
The unchanged recipe gives each CoAt reader equal weight and mixes its family
with Raptor at 40%/60%. The exact 23 upstream code cells, all attachments,
private internet-off settings, pinned container and Apache attribution were
verified against the remote source.

The first execution failed before CoAt because our compiled-cell wrapper did
not register cell text for `inspect.getsource`. A regression test exercises
the actual generated execution loop and reproduced that exact error before
repair. The wrapper now registers unchanged cell text in `linecache` under
compatible virtual filenames. All **14 focused checks** pass; additional
checks exported all eight actual publisher helpers with exact source text.
The initial failure consumed **382 GPU seconds**; the corrected run consumed
**726**, both charged from rounded normalized quota deltas for both GPUs.
Original failed source, output and terminal evidence remain preserved.

Device-wide two-second sampling observed peaks of **4.66 / 4.96 GB** GPU
memory and **22.39 GB** cgroup memory, with 164 samples and no sampler errors.
These are observed peaks; polling can miss peaks and cgroup scope depends on
the container hierarchy. Parent-only measurements are recorded separately.
Three-study memory/runtime does not establish the hidden cohort's requirements.

Private exact evidence is under
`artifacts/private/public-frontier-adoption/gpu-inference-preflight-v3/`:
`request.json`, `accepted-launch.json`, `remote-source-validation.json`,
`kernel-terminal.json`, `actual-helper-source-export-check.json`,
`validation.json`, `visible-run-validation-and-accounting.json` and `output/`.
The verified CSV SHA-256 is
`8f0c5e7b2b561538dae55439a4d2e9173141096da3defd9bc0a4bac351eef37a`.
This was canonical notebook version **2**, a visible preflight with a
three-study guard, and is unsuitable for hidden scoring itself.

A separate full-workload bundle is prepared at
`artifacts/private/public-frontier-adoption/scored-baseline-v1/`. It keeps the
model recipe and strict checks and changes only the cohort guard and execution
envelope. Its saved version **3** requires separate terminal/source/output
verification. The visible save reserves **2,400 GPU seconds** within the
current cap. Its scoring proposal reserves **57,600 GPU seconds**, one attempt,
with a 28,000-second watchdog inside an eight-hour two-T4 reservation.

The current campaign cap is 20 GPU hours. Hidden scoring needs a separate
budget decision: the concrete proposal raises that cap to **36 GPU hours**,
adding **16 GPU hours**, with zero external spend and paid scaling disabled.
Worst-case current charges plus the visible save and one scoring reservation
are **128,834 seconds**, below the proposed 129,600-second cap. No extension,
scoring request, final candidate selection or first-place result is implied
by preparation. [The exact-submission telemetry audit](../research/exact-submission-hidden-allocation-2026-09-30.md)
did not justify recovering the previous baseline's conservative hidden charge.


## Full-workload saved version verified

Saved version **3** completed and passed independent checks of all 30 prediction
artifacts, strict four-reader receipts and exact output. The visible CSV is
byte-identical to preflight version 2. Code elapsed **238.288 seconds**, normalized
quota delta **256.763 wall seconds**, charged **514 GPU seconds**. All GPU requests
are terminal; capacity is **69,348 charged / 72,000**, with **2,652 seconds left**.

The immutable `competition-submission-proposal.json` in the saved bundle binds
version **3**, output `submission.csv`, exact source/validation/accounting hashes,
one scoring attempt and the proposed budget extension. Its SHA-256 is
`3490b9fcba9381e4778c70f4e2a4597ff660c7eb7e74dc3f8b0f446ea763317c`.
With measured save accounting, charges plus one scoring reservation would be
**126,948 seconds**, within the proposed **129,600-second** cap. No competition
submission or cap change has occurred. [Ticket 51](../../.scratch/rsna-leaderboard/issues/51-score-verified-public-frontier.md)
records the pending budget decision.
