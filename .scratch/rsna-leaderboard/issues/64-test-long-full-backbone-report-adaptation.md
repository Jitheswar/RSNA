# 64: Test long full-backbone report adaptation

Type: task
Status: claimed
Labels: ready-for-agent
Parent: [RSNA campaign](../spec.md)
Blocked by: 63

## Prospective hypothesis

Adapting every encoder stage for a substantial fixed duration learns MRI
discrimination beyond the inconclusive frozen-head and layer4-only regimes.
Compare against another equally long head continuation from identical admitted
fold-specific weights. Register these choices before any new held-out prediction.

## Fixed scientific choices

Use the original 4,228-study pixel/feature cohort, corrected three folds and
unchanged audited report admission. Per-fold original ordered training pools:
38/39/39 reference studies and 2,731/2,846/2,749 report studies. No held-out
study/target enters a training batch. Every selected study has nonempty admitted
supervision. Report labels remain distinguished from supplied reference labels.

Both arms begin with the exact same fold's Ticket 55 report head after 7,500
updates and the original admitted ImageNet encoder. Restart Adam for both arms.
Control: frozen encoder/head rate **1e-4**. Treatment: every encoder parameter
trains at **1e-5**, head at **1e-4**. Preserve all normalization running buffers,
FP32, native 224-pixel whole-field preprocessing, 16×8 packing and anatomical
pooling. No augmentation, adjacent-slice channels, new weights or reports.

Each arm performs **7,500 additional successful updates**, two reference and
two report draws per update, initialization seed **20261002**, batch seed
**20260912**, report offset **34034034**, masked BCE averaged over combined
supervised entries. Ordered source exposures and masks must match exactly.
Use the shared fixed-budget trainer and its model/Adam/scheduler/RNG/progress
checkpoint format. File-backed treatment pixels are verified on every read;
control features remain unfitted, source/packing-identical and independently
admitted. Verify the initial native/feature head agreement on the first actual
mixed-source batch, without inspecting its AUC.

Choose operational stage size only from Ticket 63's actual cost. Keep one
unchanged 7,500-step scientific budget through every resumed stage. Checkpoint
every 500 steps, each 900 seconds, at epoch boundaries and at every planned
stage pause. Planned pauses must publish the exact stopping step. Do not score
or select intermediate states, change duration, sweep rates or choose a fold
after seeing outcomes. One consumed launch per stage; failures retain evidence
and require diagnosed recovery under a distinct request.

## Evaluation and decision

Only final checkpoints produce 58 complete held-out predictions per arm, using
shared native model pooling and batch-two/16-thread CPU inference. Primary:
full-backbone minus matched head continuation, twelve-target macro ROC AUC,
2,000 paired study bootstrap resamples, seed **20261002**, 95% interval and all
fold/target directions/support. Secondary: each arm versus the Ticket 55 warm
parent and encoded-image ablation preserving masks/anatomical descriptors.
No weight search, target-selection search or new ensemble is admitted here.

All six final checkpoint identities, finite gradients/losses, exact source
exposures, changed treatment stem/layers1–4/head, unchanged treatment
normalization buffers and control encoder, fresh-process strict reload and
original-MRI prediction replay within **1e-6** are required. Repeat comparisons
independently before disposition. Positive primary lower bound and positive
directions in all folds plus positive image-signal controls enable provisional
delivery review only. Otherwise record inconclusive/rejected results.

Gold58 is repeatedly reused adaptive development, with missing attested patient
groups. Its AUC cannot establish hidden generalization, complementarity with the
scored 0.936 anchor, or either first-place rank. A positive result still needs
a separate fixed anchor comparison and actual offline candidate delivery check.

## Complete resource admission

Freeze an exact executable plan only after Ticket 63 independently succeeds
and actual accepted resources/terminal/source agree. Each nonexclusive CPU job
is at most 16 CPUs/64 GiB/six hours, watchdog below server limit, no automatic
requeue, at most two active or pending campaign CPU jobs. Bound and retain all
startup/fingerprint, training, checkpoint, evaluation, original-source replay,
storage and conditional final-fit/delivery costs. Use fresh apparent storage
under the two-terabyte cap, conservative 2× observed costs and actual calendar
headroom before the October 11 science freeze. No uncontrolled stage launch.

Prospective reserve: three full treatment folds, one bounded three-fold control
job, independent final/source replay, and conditional one additional full fit
plus offline delivery. The reserve does not authorize a final fit or GPU job;
their exact inputs/recipe/hardware need their own decision after evidence.
Preserve the existing GPU ledger/charges and the single queued TPU request.
Zero external spend. Do not substitute a shorter duration based on runtime or
intermediate metrics: if the complete route does not fit, do not launch it.

## Requirements

- [x] Complete prerequisite actual shared-trainer cost and exact stage-pause
  checkpoint publication.
- [x] Freeze complete executable scientific/resource plan and exact launch
  guards, including checkpoint/time/storage/calendar/delivery reserve.
- [ ] Complete all six matched final runs and exposure/state/terminal checks.
- [ ] Independently replay original MRI predictions and all registered
  comparisons, report uncertainty/cost/disposition.
- [ ] If provisional gate passes, separately admit fixed anchor comparison and
  offline delivery; actual leaderboard evidence remains necessary.

## Comments

October 2: Registered while the diagnostic cost job 8018 is running. No new
held-out prediction, full fitting request, learned candidate or GPU use exists.
Planned stage publication is verified with a regression that reproduces stale
step-0 checkpoints after a step-2 pause; shared trainer correction is underway.


October 2: Ticket 63 independently passes terminal/source/state verification.
The planned-pause bug was reproduced (pause at step 2 saved step 0), corrected,
and verified by **27 relevant lifecycle/training tests**. Five measured-cost
admission tests pass. Exact executable plan SHA-256
`557230fec672f6db9252b1895a4d174652fe217c0388a8c9598bd749183aa985` freezes
**1,000 updates per stage, eight stages per treatment fold**, with a final
500-update stage and unchanged scientific duration. Each treatment allocation
is nonexclusive **16 CPUs /64 GiB /5.5 hours**, watchdog 19,740 seconds.
Matched three-fold control has a two-hour allocation. Complete reserve includes
**5,400 seconds** independent original-source replay, **7,200 seconds** offline
delivery and one conditional, unauthorized final fit. Checkpoint allowance
**537,988,466,127 bytes**, including epoch-triggered histories, all six fits and
conditional fit; total route reserve **653,400 CPU-node seconds**, conservative
calendar **516,600 seconds** plus a review day before October 11. Both live slots
and fresh storage passed admission. GPU/TPU ledgers and zero spend are unchanged.

Accepted first-stage **8032** (fold 0/stage 0) and control **8033**; actual
allocations match their immutable launchers. Their launch intents are consumed.
Persistent `rsna-ticket64-comparison-coordinator-v1.service` only advances
registered stages after successful exact terminal/progress checks, retains two
CPU slots and stops on failures or ambiguous consumed launches. Seven operational
safety tests pass. No retry, intermediate metric selection, final-fit admission
or candidate promotion is performed by the coordinator.


Independent original-MRI verifier implementation is prepared, with **six
admission/shape checks** passing and syntax verified. No verifier request or
execution exists before all final fits; actual 116 final predictions, original
source replay and the registered paired comparisons remain pending. First-stage
and control optimizer updates are progressing with finite logged losses.


October 2: Control **8033 failed `1:0`**, 1,551 CPU-node seconds, after completing
fold 0's 7,500 updates. Prediction selector used all 1,522 fold members instead
of 20 gold studies. The exact `Selected source cohort incomplete` error was
reproduced at the actual call site in 0.07 s. All 58 gold pixels are available
and the cohort manifest hash matches. Original trace/source/intents/terminal
and trained checkpoint remain unchanged; v1 coordinator stopped correctly.

Distinct v2 execution plan
`41028974908d8ea8c4fe336f86de0263782100c76b9375dc7693bf60ac213eaf` restricts
final prediction to gold studies while retaining the original science/config/
checkpoint identity and all 7,500-update choices. Recovery control **8034**
reuses fold-0 final weights and trains only previously unstarted folds 1/2.
Actual gold-only prediction now passes all 20 fold-0 studies; retained checkpoint
SHA unchanged, no updates repeated and no partial-fold AUC inspected. Full
**8032** continues its original first stage. New v2 stage 1 requires its exact
successful terminal/step-1,000/hash, then resumes the original scientific run.
All future requests are distinct v2 launches, at most two CPU jobs, no retries.

Fresh complete conservative bounds **660,600 CPU-node seconds** and **523,800
calendar seconds** include the failed control's prior allocation; actual failed
charge 1,551 seconds is separately retained. Projected storage 1,792,099,252,356
bytes remains below 2 TB. GPU/TPU ledgers unchanged, spend zero. Only
`rsna-ticket64-comparison-coordinator-v2.service` advances; v1 is stopped.
Gold-cohort regression **2**, recovery scheduling **3**, updated independent
verifier admission/path/shape **8** tests pass. Final scientific comparison,
independent original-MRI replay and any leaderboard gain remain pending.
[Diagnosis and recovery](../../../docs/evidence/ticket-64-gold-prediction-recovery.md).


October 2: Recovery control **8034 completed `0:0`, 995 CPU-node seconds**,
all three 7,500-update heads and 58 gold predictions. Independent state audit
passes at 07:18:32 UTC: strict original identity/config/fold, initial weights,
unchanged encoders, changed heads, finite complete Adam/loss states and exact
original target/mask/source exposures; no AUC read. Failed **8033** keeps its
1,551-second charge. Full **8035** (fold 1/stage 0) is now live beside **8032**,
exact admitted resources, no third CPU job or intermediate scoring.

The reserved final analysis and independent MRI replay are now handled by
frozen `rsna-ticket64-final-verification-v1.service`, admission
`c7c3ece040a5504462d14876526b3cb3b5eba043f382c01d4dcc57c98fa077ec`.
It waits for all six exact finals, freezes their inputs, then performs analysis
and native replay within the same reserved 90-minute CPU allocation. It stops
on failed/ambiguous execution and never retries a consumed launch or promotes
results. Ten final-replay tests and seven watcher guards pass. Actual final
comparison/original MRI reproducibility and leaderboard advantage remain pending.
