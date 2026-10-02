# Ticket 64: gold-only final prediction recovery

The first control fit completed its original **7,500 updates**, then job
**8033 failed `1:0`** at final prediction admission. The helper used the fold
manifest's entire held-out membership, including report-only studies, where
this comparison requires only supplied reference labels. This was a prediction
cohort selector mistake; it does not attribute the separate hidden Kaggle error.

Exact retained trace: `final_predictions` → `load_bank` →
`ValueError: Selected source cohort incomplete`. Actual failed terminal:
**1,551 allocated CPU-node seconds**, one 16-CPU/64-GiB/two-hour nonexclusive
attempt, no requeue/GPU/spend. The v1 coordinator stopped on failure. The
failed source, controller, trace, consumed intent and full checkpoint remain
under `artifacts/private/ticket64-full-backbone-comparison-v1/`.

| Fold | Incorrect held-out selection | Required gold studies | Missing pixels in incorrect selection |
| --- | --- | --- | --- |
| 0 | 1,522 | 20 | 65 |
| 1 | 1,403 | 19 | 60 |
| 2 | 1,482 | 19 | 54 |

All **58** gold studies have admitted pixels. The frozen cohort manifest hash
matches. This excludes the alternatives of missing gold pixels or a changed
manifest. The exact call-site regression first failed in **0.07 s** with the
same error, using one gold and one excluded report study. The corrected v2
function intersects held-out membership with gold memberships before pixel
admission. Two regressions pass, including preserved failure in the unmodified
v1 function; the original mixed-cohort trace/source are not overwritten.

A distinct immutable recovery is admitted under standing owner authorization.
Execution plan SHA-256:
`41028974908d8ea8c4fe336f86de0263782100c76b9375dc7693bf60ac213eaf`.
Original scientific/checkpoint identity remains:
`557230fec672f6db9252b1895a4d174652fe217c0388a8c9598bd749183aa985`.
The new request preserves all scientific choices, original source pools,
7,500 updates per fit, sampling, rates, optimizer semantics and fixed buffers.
No completed update is repeated, no intermediate checkpoint is scored and no
candidate is selected by this recovery.

Accepted CPU recovery control **8034**, one nonexclusive 16-CPU/64-GiB/two-hour
allocation, actual resources match. It strictly reloads the prior fold-0 final
checkpoint, produces only that fold's **20** gold predictions and encoded-image
ablation, then trains the previously unstarted control folds 1 and 2. The
fold-0 actual call now passes, finite twelve-target predictions are complete,
and the original trained checkpoint hash is unchanged:
`5049961977732a1ef29d429f384170d19bf595edbea75c2d664c88eefaf48133`.
No partial-fold AUC or paired comparison is inspected.

Full-backbone fold-0/stage-0 **8032** continues under its original immutable
source and allocation. Its first 1,000 updates perform no final prediction,
so the diagnosed selector is not exercised in that stage. Recovery stage 1
requires this exact job's successful terminal, durable step 1,000 and matching
checkpoint hash. It resumes the original training directory using the original
scientific/config/checkpoint identity; new execution/terminal receipts live
under v2. Folds 1 and 2 use new v2 training directories. The original v1 stage
0 is inherited once and cannot be relaunched by the recovery coordinator.

Fresh complete admission retains **537,988,466,127 checkpoint bytes**, all six
fits and conditional final-fit/delivery reserves; observed project/external
model storage **1,249,110,786,229 bytes**, projected peak
**1,792,099,252,356 bytes**, below the two-terabyte cap. Conservative aggregate
route bound includes the prior control allocation: **660,600 CPU-node seconds**,
calendar **523,800 seconds** plus a review day before October 11. All new full
stages retain 1,000 updates (last 500), nonexclusive 16 CPUs/64 GiB/5.5 hours,
19,740-second watchdog. At most two owner CPU jobs; GPU/TPU ledgers and spend
unchanged. Conditional final fit and GPU delivery are not authorized here.

Only `rsna-ticket64-comparison-coordinator-v2.service` advances fits. It checks
exact control/stage terminals, published progress and predecessor state, stops
on ambiguous or failed execution, and never retries consumed launches. Three
recovery scheduling tests pass. The final verifier now binds inherited v1
checkpoints/first-stage receipts, v2 receipts and the exact resumed checkpoint
chain; eight admission/path/shape checks pass. Actual independent original-MRI
final replay and all-five comparison recomputation still await all six finals.
The objective remains both actual first-place ranks; no gain or rank is proven.
