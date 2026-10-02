# 59: Diagnose the exact D4 hidden execution failure before another scoring attempt

Type: task
Status: claimed
Labels: ready-for-agent
Parent: [RSNA leaderboard campaign](../spec.md)
Blocked by: 51

## Observed failure

Authenticated exact submission **56739912**, saved script **354194017 / v3**,
is terminal with no score and a generic unhandled rerun error. Its successful
visible session processed only three studies. The hidden traceback and input
are unavailable in the exact submission response. Latest notebook logs remain
the visible session; do not label them hidden scoring logs or infer a root cause
from status `COMPLETE` alone.

## Diagnosis scope

Use the diagnosing-bugs feedback-loop discipline. Preserve exact source, terminal
records, conservative charges and both scored fallbacks. Establish an unattended
reproduction on accessible MRI or captured execution before ranking causes or
editing the recipe. A nearby fixture error does not establish this hidden error's
cause. Minimize a reproducible failure, record falsifiable alternatives, and
test one change at a time at the actual call-site seam.

The already registered **Ticket 53 v4** paired Gold58 development experiment
keeps all 23 D4 anchor cells unchanged and exercises 58 real studies and the
disk-cache branch. Its actual output/trace may supply an accessible failure
loop, but successful Gold58 execution would not reproduce the hidden symptom.
Its separate inference decision and reservation apply; this diagnosis issues no
duplicate code scoring request and no new GPU allocation.

Require a regression that fails before a repair and passes afterward, original
cohort replay, intact prediction identities/weights and strict runtime/schema
checks. Reject speculative tolerance relaxation, silent fallback removal or
prediction neutralization solely to obtain a score. Any recovered runtime needs
a new concrete complete-workload validation and resource decision before scoring.

## Requirements

- [x] Retain exact terminal source/error and reconcile unknown allocation.
- [x] Check the official error-debugging guidance and actual API/log scope.
- [ ] Build and minimize a red-capable loop on accessible data or a captured trace.
- [ ] Test ranked causes; independently verify the repair and original replay.
- [ ] Freeze a distinct justified delivery/runtime decision before rescoring.

## Comments

October 1: Phase 1 is in progress. The official exact response and latest
session-log capture are retained under
`artifacts/private/public-frontier-adoption/scoring-failure-diagnosis-v1/`.
The first log-observation JSON capture failed to serialize the SDK status object;
the partial v1 capture remains preserved, and v2 serializes the real response.
That capture error is unrelated to the hidden notebook failure.

Primary documentation: [Kaggle code competition debugging](https://www.kaggle.com/code-competition-debugging),
[official kernel-log API implementation](https://github.com/Kaggle/kaggle-cli/blob/main/src/kaggle/api/kaggle_api_extended.py).


October 1, after Ticket 53: the unchanged-anchor 58-study GPU execution passed
all four CoAt readers, 27 prediction artifacts and strict no-fallback checks.
Its native pixels/masks equal the admitted CPU cache. It supplies **no red
loop for the hidden failure**, so Phase 1 remains incomplete. No hypotheses
are ranked and no hidden-runtime repair is claimed from this successful run.

### October 1 — broader CPU probe complete; full source scan running

Fixed label-blind **256-study** probe **7992** completed `0:0`, **264 CPU-node
seconds**, zero GPU/spend. It includes all original acquisitions for those
studies, including series excluded from the learning cache: **48,870 DICOMs /
1,446 header series / 1,195 selected slot-series**, **31,604,773,004 raw bytes**.
Unchanged public DINO preprocessing completed with **zero decode failures and
zero failed/fallback events**, disk cache **2,080,899,072 bytes**, peak process
RSS **5,391,650,816 bytes**. All studies have 3–6 present slots. Independent
summary verification checks exact terminal/source/cohort identities, producer
SHA records and all persisted file sizes; it does not reread every source byte.
The hidden failure remains unreproduced. Result SHA-256
`14bf8e3c7adc62851051c2b721c2ae8e5c167399ed56b2a4d95360c109562197`.

Fresh authenticated **08:56:47 UTC** observation retains baseline **0.936**,
leader **rhoskeri / 0.961**, exact frontier generic error/no score. The official
owned-output download endpoint did not expose either checked submission file;
no hidden output or traceback was obtained. No duplicate scoring or GPU charge.

The measured probe supports a separate full-original-input scan: **4,407 MRI
studies**, sorted into disjoint **2,204 / 2,203** partitions, jobs **7995 / 7996**.
Each is one nonexclusive **16-CPU / 64-GiB / three-hour** job, 10,560-second
watchdog, one attempt, no requeue/GPU/fitting. Exact parent source fragments and
pixel rules remain unchanged; only cohort shape and resource checks generalize.
Prospective per-partition cost uses 2x measured time plus 900 seconds. Aggregate
raw/cache allowance fits the 2 TiB project ceiling. Direct measured project
storage before preparation: **603,970,359,296 bytes**.

Full request **v2**, plan SHA-256
`a2d7f3624f1798fc3ca591e9db6f83ca171bb97793cca43d19e2ca4c0c84c1b8`.
Unlaunched v1 is retained after a transcribed storage figure was incorrectly
labelled measured; v2 takes its own exact `du` measurement. Neither pixel rules
nor resource limits changed, and v1 never launched. Persistent observers use
actual accepted IDs. Both full partitions are active; no full-corpus result,
hidden failure attribution or repair is established. Inspect their immutable
outputs/terminal controllers before another action.
[Evidence](../../../docs/evidence/ticket-59-hidden-execution-diagnosis.md).


### October 2 — full DINO corpus verified; residual CoAt seam extended

Both **7995 / 7996** completed `0:0`, **4,974 CPU-node seconds**. Independent
summary verifies disjoint complete original **4,407-study / 819,078-DICOM /
569,154,778,424-byte** coverage against original metadata/central directory,
all **24,371** header series, actual fragments/events/source/decision/terminal
identities and zero decode/fallback events. Integrity job **8012** completed
`0:0`, **779 CPU-node seconds**; every retained payload independently matches
its producer SHA and original archive CRC. Exact terminal/source/plan and full
partition byte totals pass reconciliation. Validation SHA-256
**bcc777b9193de735cd7ac7e76d76b8ce2558468a222052e53631e42ceba50844**.

Exact residual CoAt CPU preparation pilot **8013** passed **64** fixed
label-blind original-slice-count order statistics, including both **67 / 632**
slice extremes. **148 CPU-node seconds**, native **61–94-window** bags,
zero failures/warnings/dense/sparse fallbacks. Actual notebook-patched
`_prepare_bag_task` and admitted source/OpenCV hashes are unchanged. Independently
checked execution records include producer pixel/sidecar hashes; pixels were not
independently replayed. No model, GPU, labels/reports or hidden input.

Measured cost supports separate full-corpus CoAt CPU probe **8014**, all **4,407
studies**, one **16-CPU /64-GiB /six-hour** attempt, two workers, 21,540-second
watchdog; **21,282.375-second** forecast =2x measured scaled cost +900 s. Source
MRI reused read-only, exact renderer/sampler preserved, only resource guards
generalized. Plan **abea8f3b42312c3fba1a43a15ea732af256a19fdaa22c59e5998538da83dc5fa**.
Actual observer retains terminal evidence. Full outcome pending.

Fresh authenticated **01:08:08 UTC**: Elyra **rank 2,005 /4,808, 0.936**;
leader **0.961**; exact failed frontier still no score/traceback. Phase 1 remains
incomplete: no matching hidden reproduction, cause, repair or rescoring. Ledger
unchanged; zero new GPU/spend. [Current handoff](../session-handoff-20261002.md),
[evidence](../../../docs/evidence/ticket-59-hidden-execution-diagnosis.md).


October 2, **06:06:33 UTC**: complete CoAt input job **8014** independently
passes exact terminal/source/summary verification, `COMPLETED 0:0`, **8,770
CPU-node seconds**, all **4,407** studies, windows **52–94**, **zero input
failures/sampler fallbacks/warnings**. Producer hashes checked; independent
pixel replay was not performed. No hidden error reproduced or ranked cause
established, and no scoring retry or GPU use. This ticket remains open.
