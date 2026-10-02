# 68: Test matched BiomedCLIP report heads

**Type:** task
**Status:** claimed
**Assignee:** campaign agent
**Blocked by:** 67

## Scope

Prepare the exact next paired experiment while Ticket 67 materializes inputs.
Actual resource admission and launch require its full independently verified
feature cohort and successful terminal. Preserve all existing frozen sources.

Train six medical heads: reference-only and report-supervision on each of the
three original corrected folds, matching Ticket 55's **7,500 total updates**,
original random head state, Adam 1e-3, batch size two per source, exact original
source admission/targets/masks and ordered seeds. Medical vectors use Ticket
67's fixed512/L2 representation; no learned head transfer, fitted normalization,
layer choice or learning-rate sweep. Reuse independently verified Ticket 55
ResNet heads as matched comparators without redundant fitting.

Primary: medical report versus matched ResNet report, mean12 ROC AUC and
2,000 paired study-bootstrap draws, fixed seed20261002/95% interval plus all
three fold directions. Fixed secondaries: medical reference versus ResNet
reference, medical report versus medical reference, two zero-image controls,
and one prospectively fixed half medical/ResNet report blend. Gold58 is reused
adaptive development evidence with missing patient groups; no first-rank proof.

Before any provisional review, independently audit every fitted target/mask,
original init/Adam/sampler/source exposure/checkpoint/terminal and all58 original
MRI medical+ResNet predictions/zero-image controls and all fixed comparisons.
Require exact frozen bytes and <=1e-6 numerical replay. Preserve registered
Ticket 64 fitting/delivery/review/storage reserves; at most two owner CPU jobs,
no retries/requeue, no accelerator use or spend. Fixed scored-anchor/offline
competition delivery needs a later separate decision; no automatic promotion.

## Acceptance

- [x] Exact matched scientific protocol and initial-state equivalence implemented.
- [x] Completed verified bank then full frozen storage/calendar/resource admission.
- [ ] Six sole medical fits complete and fixed comparisons computed.
- [ ] Independent original-MRI/source/checkpoint/exposure/terminal replay passes.
- [ ] Evidence and disposition recorded; anchor/delivery gates retained.

## Comments

- Four-script workflow frozen under waiting-controller admission
  `ae31e9e0a1cf6e52c75fdd5e79f5e0dd22717c316b93f5d426a61c2bdd51ef67`.
  Actual initial head matches all six historical heads; metadata preflight
  reconstructs original source exposures/targets and held-out masks exactly.
  Twenty-one focused guards pass.
- Persistent controller is live, PID324630, waiting for the complete Ticket67
  bank. No medical fitting resource decision, job, gain or submission exists.
  On complete successful bank receipt it performs fresh protected admission
  and combined unchanged-fit scheduling, then one fit and one native verifier.
- [Workflow evidence](../../../docs/evidence/ticket-68-biomedclip-matched-heads.md).

- October2 12:15UTC: complete bank/8071 successful terminal verified; exact
  planSHA `16ee8b86a75aa30a0c6a1f25e68ee65028a084914b7863584e1abb540c8bc666`
  admitted under standing owner autonomy. Tracked decision
  `../resource-approvals/ticket68-biomedclip-heads-20261002.json`. One30-minute
  fit plus one3-hour native verifier,16CPU/64GiB, max2 owner jobs/no retries,
  additional12600 CPU-node/19800 calendar seconds,1GB extra storage allowance,
  protected full route/review projects October9 19:14UTC before October11.
  Combined controller holds original coordinator and advances identical full
  stages. Existing8054/8072 occupy both slots; medical fit/verification still
  unlaunched. No medical predictive result, promotion or scoring yet.

October2,12:57UTC — sole medical fit **8073** completed **0:0**, **357 CPU-node seconds**, no restarts. Controller advances unchanged full stages while the sole original-MRI verifier awaits one of the two CPU slots. No predictive results inspected before native/terminal verification. Current full jobs8074(fold0/stage3,target4000) and8072(fold1/stage2,target3000); do not launch a third owner CPU job.
