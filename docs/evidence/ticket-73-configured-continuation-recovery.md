# Ticket 73: Configuration and checkpoint recovery

The distinct execution route v4 is frozen under plan
`9844fd6a7e62f3f1aaf5d0678784f197b8373d702e14c19e669ccde5244a36c8`.
It restores the configuration prerequisite exposed by failed job 8111 without
changing the original three-fold scientific treatment or repeating completed
updates. No competition submission or candidate promotion is authorized here.

## Failure and actual regression

Job 8111 failed before the optimizer loop, FAILED/1:0, with 349 allocated CPU
node seconds. Its fresh training directory lacked the immutable original
configuration.json required when resuming a checkpoint. Its consumed intent,
trace and actual terminal remain preserved. The controller stopped; its route
and frozen sources remain unchanged. The historical 8072/8074 missing terminals
remain unknown and retain their conservative 39,600-second charge.

The real shared trainer reproduces the same missing-configuration error with a
small interrupted run. Carrying unchanged original configuration bytes into the
new artifact directory allows exact continuation: model tensors, loss history,
exposures and all Adam steps match uninterrupted training. The source checkpoint
remains unchanged. Publication uses an exclusive atomic link and refuses changed
source bytes or replacing a different destination configuration.

## Separately frozen execution

The new worker, controller and independent native verifier live in separate
Ticket 73 files. All three original configuration files, original scientific
identity, complete source snapshot, audited seeds and failure evidence are hash
bound. Configuration is copied before any stage is launched and rechecked before
calling the unchanged trainer. An AST comparison confirms the scientific worker
is identical after removing the added configuration handoff; the original
terminal/audit/control/runtime/final-analysis functions are unchanged.

Fold 0 inherits its independently audited 4,000-update checkpoint. Fold 2
inherits the actual completed 8081 1,000-update checkpoint. Fold 1 retains running
job 8113 in v3, continuing from 3,000 to 4,000. No changing live checkpoint is
copied. Its next stage requires an actual successful terminal, exact 4,000-step
worker result and predecessor checkpoint hash. A separately immutable input
closure binds the terminal, accepted identity, worker result and copied completed
state. Failure or a purged handle stops this route without retry.

There are 15 new stages / 13,500 updates, plus 1,000 updates in the existing live
job. All folds still finish the original 7,500 additional updates, with unchanged
matched controls, Adam, normalization, source pixels, labels, masks, sampling and
final-only statistics. Final verification still requires all 58 original MRI,
six final states, 116 predictions and independently recomputed registered
comparisons. Historical gaps remain explicit in the final disposition.

## Entire remaining resource route

The tracked decision is
`.scratch/rsna-leaderboard/resource-approvals/ticket73-configured-continuation-20261002.json`.
It uses standing zero-spend owner authorization, not a claim of new exact numeric
owner approval. Each new stage retains 16 CPUs, 64 GiB, 19,800 seconds, a
19,740-second watchdog, one launch, no requeue and no accelerator. At most two
owner CPU jobs are pending/running, coordinated through the shared launch lock.
The original v1/v2/v3 scheduler locks are held throughout the new controller.

Only unfinished work is projected anew; completed allocations remain charged.
The 15 new stage maxima total 297,000 CPU node seconds, and the live job retains
its complete 19,800-second bound. The two-slot fit critical path is 11 stage
bounds / 217,800 seconds: four fold-0 stages followed by seven fold-2 stages,
with the live job and four remaining fold-1 stages occupying the other slot.
The forecast retains all of the following:

- Conditional final fit: eight stages / 158,400 seconds; still not authorized.
- Independent original-MRI replay: 5,400 seconds.
- Offline delivery: 7,200 seconds.
- Queue transitions: 21,600 seconds.
- Other previously retained CPU delivery: 21,600 seconds.
- Other previously retained GPU delivery: 7,200 seconds; no new GPU launch authorization.
- Review: 86,400 seconds.

The conservative complete remaining route totals 525,600 seconds, finishing
October 8 at 19:55:50 UTC from admission, before October 11 00:00 UTC.
Existing plus all retained allowances and an extra 1 GB recovery envelope projects
1,972,567,098,522 bytes, below the 2 TB cap, with sufficient physical free space.
No calendar or storage margin was obtained by reducing the scientific budget.

## Actual launch and checks

The new coordinator is `rsna-ticket73-configured-continuation-v4.service`.
Its sole fold-0 stage-4 launch was accepted as **8119**, observed RUNNING with
existing **8113** also RUNNING. This is execution evidence, not fit completion or
predictive improvement. The other new stages and final replay remain pending.

The new recovery guards, real-trainer seam regression and preserved v3 guards
pass **45 tests**. They cover live-state copying prevention, failed/missing
terminals, wrong ancestry, immutable closure tampering, slot scheduling,
configuration byte identity and exact optimizer continuation. No daily scoring
slot has been used. Submission still requires verified gain and offline/runtime
validation; leaderboard first place has not been achieved.

Project type checks also pass across all 87 source files. At the follow-up
observation, existing8113 had recorded optimizer step3247. New8119 was still
loading with its exact configuration already present; optimizer progress beyond
4000 is not yet claimed.

Subsequent actual observation confirms8119 RUNNING after the configuration
guard, with optimizer step4008 recorded beyond its original4000-step
checkpoint. The private configuration-recovery-progress.json retains the raw
controller and counter observation. This proves the recovered training seam is
working in the actual full model; final stage/native success remains pending.
