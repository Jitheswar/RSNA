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


## Completed control and independent state audit

Recovery **8034 completed `0:0`**, **995 allocated CPU-node seconds**, peak RSS
**2,938,675,200 bytes**. All three control fits are final: 7,500 updates each,
58 gold predictions plus ablations; fold 0's completed weights were reused.
Independent state audit at **07:18:32 UTC** reloads all three checkpoints with
the exact original scientific identity, configuration and fold. Initial warm
states agree; every control encoder tensor is unchanged, each head changes,
Adam rate is 1e-4 with 7,500 finite-moment steps, and loss histories are finite.
Original audited targets/masks and deterministic source draws reconstruct all
30,000 sample exposures per fold exactly, without held-out contamination.
No AUC was read or comparative selection made.

Private receipt: `control/independent-state-validation.json`. Retained actual
terminal hash `6e7a9bfd137d380fdd772561df13e15132c44bc82d630a0a250834f57ad5cfeb`.
The prior failed control's 1,551 seconds remain recorded; combined control
attempt allocation is **2,546 CPU-node seconds**. Independent original-MRI
prediction replay remains necessary. The coordinator subsequently accepted
full fold-1/stage-0 **8035**, exact 16 CPUs/64 GiB/5.5-hour resources, alongside
original full fold-0/stage-0 **8032**. Neither treatment fold is final.

## Reserved final analysis and replay automation

`rsna-ticket64-final-verification-v1.service` runs a frozen read-only watcher.
Admission SHA-256:
`c7c3ece040a5504462d14876526b3cb3b5eba043f382c01d4dcc57c98fa077ec`.
Private root `postfit-controller-v1/` holds its three immutable source files,
source hashes, file lock and current phase. Actual service is running and its
phase is `wait_for_fits`; no independent replay job has been submitted.

After all six fits and every inherited/v2 stage have exact successful terminals,
it freezes final record/checkpoint identities and fresh storage. The existing
**5,400 CPU-node-second** replay reserve covers both the fixed final analysis
and independently recomputed original-MRI comparisons in one nonexclusive
16-CPU/64-GiB/90-minute job. No additional CPU route reserve, fitting, rate/
duration changes, intermediate metrics, GPU request, spend or candidate
promotion. Two owner CPU slots remain enforced. The watcher stops on failed
fit evidence or consumed ambiguous replay intent; an observation timeout
re-polls the same handle. It reconciles the actual final replay terminal/output
before stopping. Provisional results still require separate anchor/delivery
review and both real first-place ranks remain unproven.

Tests: **10** final replay admission/shape/analysis checks and **7** watcher
scheduling/one-attempt/readiness guards pass. The scientific fitting snapshots
remain unchanged. These checks prove operational guards; they do not prove
final MRI reproducibility or predictive advantage before the real replay.
