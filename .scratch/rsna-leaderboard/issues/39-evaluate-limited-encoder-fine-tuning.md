# 39: Evaluate limited encoder fine-tuning

**What to build:** A bounded fine-tuning comparison tests whether adapting the selected encoder improves complete reference OOF performance at an acceptable delivery cost.

**Blocked by:** 32: [Complete the three-fold ResNet-18 comparison](32-complete-three-fold-resnet18-comparison.md); 33: [Run and resume an approved pretrained GPU experiment](33-run-and-resume-approved-pretrained-gpu-experiment.md)

**Parent:** [RSNA leaderboard campaign with zero external spend](../spec.md)

**Status:** resolved

**Execution constraints:** Follow the approved spec: zero external spend; significant work through Slurm; existing CPU ceilings; one active or pending GPU request and 20 allocated GPU-hours across the first campaign, including failures and retries; 2 TB soft storage budget; exact resource and delivery approvals where required. Questions go in visible chat. Synthetic checks establish software behavior only. No project-imposed candidate or submission-count cap.

**Execution trigger:** Select only after learning competence, input/label integrity, approved GPU compatibility and a complete comparison budget are established.

## Acceptance criteria

- [x] Freeze one fine-tuning hypothesis, trainable encoder portion, learning schedule, reference/supervision policy, seed, checkpoint rule and all three fold budgets.
- [ ] If optional weak supervision or an alternative representation supplies this run, add the actual supplying ticket as a blocker before execution; do not gate gold-only fine-tuning on unused branches.
- [x] Use independent fold checkpoints and caches for adapted encoders; do not reuse another fold's fine-tuned features or fit to held-out images.
- [x] Run through the approved GPU and recovery interfaces, charging attempts to the shared budget and preserving an unchanged scientific identity across permitted retries.
- [x] Complete matched OOF comparison against the relevant frozen-encoder candidate, reporting uncertainty, target/fold behavior and any changed exposures.
- [ ] Measure full candidate inference cost, model size and actual memory; three fine-tuned folds do not inherit one-encoder inference timing.
- [x] Demonstrate checkpoint-to-offline-output reconstruction and failure on mismatched artifacts through the existing lifecycle tests.
- [x] Record the promotion/rejection decision and evidence in the final-freeze register; no incomplete fold set qualifies for promotion.

## Completion evidence

October 1: experimental comparison is complete and the branch closes as **inconclusive/no promotion**. Optional supplying branches were unused. Full hidden-scale three-fold delivery cost remains unverified; that delivery criterion was not passed and no deliverable candidate is admitted. The branch disposition does not waive the gate for a future proposed candidate.

Record the demonstration, focused boundary checks, exact artifact identities, measured scope, unresolved limitations and any required branch disposition. A ready-for-agent label authorizes preparation within the spec; it does not establish resource approval or successful scientific performance.



## Comments

### 2026-09-27 implementation update

The software path and exact private experiment plan are prepared. The selected hypothesis updates only ResNet-18 `layer4` and the fold-specific head, using gold-reference labels, the Ticket 32 cohort/preprocessing, head learning rate `1e-3`, layer4 learning rate `1e-5`, 100 optimizer steps per fold, and final-step checkpoints. Each fold starts from the same admitted ImageNet weights and fits its own checkpoint; no fine-tuned feature cache is used. Resource scope is zero external spend, one GPU, 16 CPUs, 64 GiB RAM, one 30-minute smoke, then sequential six-hour fold reservations with no retries.

The first smoke request (job `7786`) was canceled before data access after detecting a preprocessing mismatch; Slurm controller evidence confirms zero allocation seconds. The corrected request is job `7787`, bound to the frozen plan and exact Ticket 32 preprocessing. It is still pending because the requested GPU node is allocated; Slurm currently estimates a start on 2026-09-29 at 05:11 local time. No fold has run, so there is no Ticket 39 OOF result or promotion decision; the final-freeze and release gates remain blocked.

Focused model, request-approval, candidate, inference, and checkpoint lifecycle checks pass (111 tests). The synthetic Ticket 39 lifecycle now writes and reloads a final checkpoint, replays offline predictions, and rejects a changed encoder learning rate. The full suite has 430 passing tests and seven failures in the pre-existing `supervision_comparison` work; those failures are outside this ticket.

### 2026-09-27 — Bounded autonomous recovery

The owner-directed `tsk.txt` authorizes up to three total attempts per stage, including the original, with each retry using a new immutable request identity. The additive recovery approval is `artifacts/private/ticket39-resnet18/ticket39-recovery-approval-amendment-v1.json`; it supersedes only the original no-retry execution policy. Each individual request remains a one-attempt request with automatic Slurm requeue disabled. Recovery requires root-cause diagnosis, an operational/implementation-only repair plus focused verification where applicable, or a diagnosed transient infrastructure failure.

The shared 20 GPU-hour campaign cap, single-GPU/16-CPU/64-GiB request shape, 30-minute smoke ceiling, six-hour fold ceilings, zero external spend, and 18-hour reserve for the three folds are unchanged. Each submission must fit its full wall-time reservation while preserving all remaining fold reserves. The present headroom after the current smoke reservation and the original fold reserve is 1,202 seconds. The all-stages, three-attempt worst case is 199,800 GPU seconds for Ticket 39 plus 4,198 previously charged seconds, or 203,998 seconds total; that would exceed the current cap by 131,998 seconds. No increase is authorized; if a concrete retry cannot pass the live budget gate, preserve its evidence and stop for owner approval.

The active workflow plan is revision 3 (`artifacts/private/ticket39-resnet18/ticket39-run-plan.json`). The already queued smoke job `7787` remains unchanged as attempt 1 under immutable plan revision 2 (`ticket39-run-plan-v2.json`, SHA-256 `19ac1c8a6c5310615862e82e5384f1929f102e25a81a70bb08f88cbd35e5c18e`). It is still pending because `KLEF-G-NODE2` is unavailable. No fold request has been submitted.

### 2026-09-29 06:25 IST execution checkpoint

The corrected smoke job `7787` remains the sole pending GPU request. Fresh Slurm state is `PENDING` (`Reason=Resources`), with zero runtime, no node allocation, no restart, and automatic requeue disabled. The current estimated start is 2026-09-30 18:10:45 Asia/Kolkata; this remains a forecast only. Slurm accounting is unavailable (`slurmdbd` connection refused), so no allocation is charged or inferred. The request and reservation are unchanged; no fold request has been submitted. See `artifacts/private/ticket39-resnet18/ticket39-execution-status.json` and `scheduler-observations/job-7787.jsonl`.

### 2026-09-29 06:45 IST execution checkpoint

Job `7787` remains the sole pending GPU request (`PENDING/Resources`), without allocation, restarts, or runtime. Slurm estimates 2026-09-30 18:10:45 Asia/Kolkata; accounting remains unreachable, and no charge is inferred. No new request was submitted. See the execution-status JSON and scheduler-observation JSONL for the latest check.

### 2026-09-30 — current execution-state reconciliation

The preserved execution-status JSON supersedes the older queued-job comments: corrected smoke job **7816** and GPU fold jobs **7845 / 7849 / 7854** all completed and were reconciled. The fold-1 launcher failure **7847** allocated zero GPU seconds. No live Ticket 39 request remains. Scientific configuration and final-step selection are unchanged.

Final analysis is still incomplete. CPU attempts **7856 / 7858** failed (missing cache path, then CPU-versus-CUDA replay discrepancy). The latter maximum error **6.085634231567383e-05** exceeds the unchanged **1e-6** replay gate. No completed paired comparison or promotion is claimed. An exact one-A100, 1,800-second final-analysis attempt-3 amendment is prepared at `artifacts/private/ticket39-resnet18/ticket39-final-analysis-gpu-replay-proposal-v1.json`, SHA-256 `559fe96debdbe6bd18a240dda1d15bb079da5898ac43168ba62847467ea2d41d`; it awaits approval to change final-analysis resource scope. Ticket 40 remains explicitly excluded by the existing owner instruction.

## Answer — October 1

The actual owner approved the exact attempt-3 replay proposal. Job **7966** completed `0:0` on one A100 in **161 allocated GPU seconds**, with a separate 90-second queue wait; controller evidence is retained and the ledger reconciled before frontier scoring. All 58 OOF predictions and six baseline/candidate checkpoint hashes passed independent verification; all three same-device replay errors are zero under the unchanged 1e-6 gate.

Fine-tuned AUC **0.518355** versus short frozen **0.464313**, paired delta **+0.054042**, 95% interval **[+0.001902,+0.106931]**. Two folds improve and one declines. Image-information controls are inconclusive: within-fold permutation p_upper **0.052895**, and full-minus-zero-image interval crosses zero. The branch closes as **inconclusive/no promotion**; no final-delivery candidate is admitted. CPU held-out inference and model/VRAM costs are reported; full hidden-scale delivery runtime remains unverified. No optional weak/representation branch supplied this gold-only run. [Evidence and limits](../../../docs/evidence/ticket-39-fine-tuning.md).
