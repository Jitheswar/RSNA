# Find a credible route to first place with zero external spend

Labels: wayfinder:map
Status: resolved

## Destination

A decision-complete experiment and delivery plan aimed at first place on the final private leaderboard, using zero external spend and reproducible, leakage-controlled evidence. The map is complete when the supervision, validation, model, compute, and candidate-selection decisions are sufficient to begin execution; achieving first place is the ambition of that execution, not a promise or the completion criterion of this planning map.

## Notes

- Owner confirmed final private first place, zero external spend, GPU availability dependent on the queue, and no qualified clinical reviewer on 2026-09-25.
- Owner requested resolving all remaining decisions in this continuation, overriding wayfinder's default one-ticket-per-session limit. Human decision tickets still require the owner's answers; research and read-only inventory can proceed independently.
- October 1 owner authorization supersedes the earlier question/approval workflow: handle justified Kaggle-compliant, zero-spend campaign decisions autonomously without asking again; preserve concrete plans and decision evidence.
- Implementation handoff: [RSNA leaderboard campaign with zero external spend](spec.md), published as `ready-for-agent` from the resolved decisions.
- Use wayfinder, grilling, and domain-modeling for decisions; research for external facts. This map is planning only. Research tickets may resolve concurrently; do not resolve another decision ticket while charting.
- Existing evidence is the starting point: [reproduction effort](../rsna-2026-fresh/map.md), [corrected OOF evaluation](../../docs/evidence/ticket-10-oof.md), and [completed submission evidence](../../docs/evidence/ticket-24-candidate-selection.md).
- Recorded baseline: public score 0.498; corrected reference OOF macro AUC 0.464313 over 58 studies, with no demonstrated image signal. The 4,349 report-only studies are potential supervision, not additional trusted reference labels. Current leaderboard rank and leading score require verification.
- Keep the previous frozen candidate and historical evidence intact. New methods may replace the current modeling approach if evidence justifies them.
- Carry forward the existing resource boundaries: Slurm for significant work; autonomous CPU at most one node, 16 CPUs, 64 GB, six hours, two concurrent jobs, no exclusive allocation; 2 TB soft storage limit. GPU training retains the existing requirement for a concrete experiment/resource proposal before approval. No GPU-hour entitlement is assumed from zero-cost access.
- No paid resources or clinical review. Research may investigate freely accessible pretrained resources; investigation does not establish resource eligibility or authorize their download/use. No competition reports are transmitted by this planning effort.
- Distinguish local development estimates, public leaderboard observations, and final private rank. Repeated use of the 58 reference studies is adaptive development, not an untouched test. No synthetic result substitutes for real MRI performance.

## Decisions so far

<!-- Resolved-ticket summaries and links only. Owner constraints are recorded in Notes. -->

- [Establish the current competition target and delivery constraints](issues/01-establish-current-competition-target.md): available official material supports the recorded metric and delivery envelope; live rank, leading score, and complete current rules remain unverified.
- [Establish reliable supervision options without clinical review](issues/02-establish-supervision-options.md): compare conservative report supervision using target-level reliability and fold isolation; existing learned outputs and a local multilingual extractor remain alternatives awaiting selection and admission.
- [Establish image-learning options for scarce queued GPU time](issues/03-establish-image-learning-options.md): diagnose basic learning before evaluating a small pretrained backbone, anatomical pooling, adjacent-slice context, or frozen features; none has demonstrated a competition gain yet.
- [Inventory usable compute and existing reusable artifacts](issues/04-inventory-realistic-compute.md): 158 studies have an existing image-training setup; full-corpus preparation, project storage headroom and future GPU access need bounded verification before expansion.
- [Choose validation and evidence requirements for model promotion](issues/05-choose-validation-policy.md): owner chose the existing three folds, explicit uncertainty and simple controls, with reproducible local evidence taking priority when public scores disagree.
- [Choose the first bounded campaign to establish useful image learning](issues/06-choose-first-learning-campaign.md): owner selected diagnostics followed by ResNet-18, existing-label audit before new extraction, and a 20 GPU-hour cap with conditional expansion and a CPU route.
- [Choose the candidate-selection and delivery schedule](issues/07-choose-delivery-and-selection-policy.md): owner requires completion before October 15 and no project submission-count cap; freeze modeling October 11, finish checks October 13, and select by October 14, all IST.
- Execution evidence, 2026-09-30: [full weak-cohort exposure](issues/44-test-full-weak-cohort-exposure.md) is resolved. Matched 1,500-step gold/report heads reach OOF AUC 0.6284/0.6607 and establish image signal against within-fold permutations and zero-feature controls. Added reports remain inconclusive (paired difference +0.0323, interval crossing zero). The prior near-chance result remains preserved as the earlier short-budget observation.
- Execution evidence, 2026-09-30: [public frontier asset validation](issues/46-validate-public-frontier-assets.md) is resolved. One private CPU preflight verified four pinned CoAt manifests, 107 files and eleven checkpoints, plus E13/DINO mounts and the output schema, with zero GPU use and spend. The corrected public D4 source reports its author's 0.943; this account has no score for it. Source-license and complete GPU inference gates remain, and baseline submission 56705536 is independently pending.

## Not yet specified

None remains within this planning destination. All seven tickets are resolved. The campaign and delivery tickets specify conditional routes for supervision, model expansion, ensembling and runtime reduction. Resource eligibility, actual throughput, storage headroom and future experiment outcomes remain execution facts to establish; their absence invokes the documented fallback or stop conditions rather than a claim that they are verified. Map completion means the plan is ready for execution, not that training or a competitive leaderboard result has been achieved.

## Out of scope

- Executing the training campaign, uploading or submitting models, and claiming an achieved leaderboard rank within this planning map.
- Additional spending, paid clinical review, fabricated results, held-out label leakage, or tuning against private test labels.
- Reopening the previous mechanics effort as if its inconclusive experiments established competitive performance.

- [Ticket 48](issues/48-measure-frozen-convnext-cpu-feasibility.md) resolved: frozen ConvNeXt CPU pilot completed all 32 studies/1,464 slices in 34.6 seconds, zero GPU; independently validated 768-dimensional features. Twofold prepared-cohort estimate 1.45 node hours passes the bounded extraction gate; separate extraction plan required. No performance or promotion claim.

- [Ticket 49](issues/49-materialize-frozen-convnext-cohort.md) resolved: all 4,228 studies/187,272 slices extracted on CPU and independently verified against prior source/mask/descriptor identities. Complete 768-dimensional feature manifest and unfitted `(4228,3100)` anatomical pooled bank are sealed; zero GPU/spend. Ticket 50 can freeze a separate matched-head plan.

- [Ticket 50](issues/50-test-frozen-convnext-supervision.md) resolved: six matched 1,500-step ConvNeXt heads completed with exact checkpoint and exposure replay, zero GPU/spend. Report OOF 0.659947 versus fixed ResNet 0.660737; fixed half blend 0.665014, interval crosses zero and two negative folds. Inconclusive/no promotion; admitted reports improve ConvNeXt versus its own gold-only control, with image signal verified.

- [Ticket 45](issues/45-evaluate-attributed-public-ensemble-baseline.md) resolved: exact code submission 56705536 completed at public score **0.936**. Full unknown hidden GPU exposure conservatively charged; prior 0.498 fallback preserved. Final private rank and final selection remain unproven. Ticket 47 first dry run failed at source introspection; a locally verified wrapper repair is undergoing separate bounded inference.

- [Ticket 47](issues/47-evaluate-complete-public-frontier-inference.md): corrected public frontier dry run passed independent verification of 30 prediction artifacts and all four CoAt readers, zero reported fallback, 340.948 code seconds. Full-workload saved version 3 is separately running; scoring preparation identifies a required 20-to-36 GPU-hour budget decision, with zero external spend. Author 0.943 is not this account's score.

- [Ticket 47](issues/47-evaluate-complete-public-frontier-inference.md) resolved: full-workload saved version 3 also passed, with byte-identical visible predictions. Campaign charged 69,348/72,000 GPU seconds; no active request. [Ticket 51](issues/51-score-verified-public-frontier.md) prepares one exact scoring comparison and requires the proposed 20-to-36 GPU-hour budget decision. No frontier score or final private rank established.

- [Ticket 41](issues/41-maintain-fallback-and-freeze-final-candidate-after-branch-decisions.md) phase A verified: both scored sources remain exact/terminal; all 40 historical candidate artifacts and 75 delivery payload files retain their digests. Final freeze remains open, including Ticket 39 unverified analysis and Ticket 51 budget decision. No final-selection readiness is inferred from fallback integrity.

- [Ticket 52](issues/52-test-longer-dinov2-gold-head-training.md) resolved: one 39-second CPU duration experiment, zero GPU/spend. DINOv2 1,500-step AUC 0.609957 improves over its short run but remains below matched ResNet 0.628390; paired interval crosses zero and two negative folds. Inconclusive/no promotion. All three source/final replays and 58 OOF outputs are exact; final freeze remains open.

- [Ticket 51](issues/51-score-verified-public-frontier.md) execution preparation completed: exact-source scoring/observation commands and recovery path, 45 passing boundary/accounting tests. Fresh 17:35 UTC read-only check verifies saved version 3, free quota, paid scaling off and four submission slots. Ledger unchanged; the separate 20-to-36 GPU-hour approval remains pending. No frontier scoring request issued.

- Ticket 51 observer repair verified before scoring: exact-submission URLs name raw downloads; listing URLs name saved code. Separate reference-matched evidence now drives status and source checks. Regression went red then green; 53 boundary/integration/accounting tests pass and live baseline replay is valid. Zero new GPU use/spend, unchanged campaign cap, pending exact budget decision.

- September 30 goal boundary audit: registered CPU work and the diagnosed scoring observer repair are complete, no live jobs/scoring requests, no new resource decision. The same exact budget approval has remained pending across at least three consecutive goal turns. The next substantive actions are Ticket 51’s prepared scoring reservation and Ticket 39’s separately prepared GPU analysis amendment; both need the stated owner decisions. Goal marked blocked pending those decisions, with first place unproven. Primary observations and audit: `artifacts/private/goal-blocker-audits/20260930T174126667474Z/`.

- October 1 owner decisions supersede the preceding approval blocker: exact Ticket 51 cap extension and Ticket 39 replay approved, followed by standing authorization for Kaggle-compliant, zero-spend campaign decisions without repeated questions. Current cap 36 GPU hours; old unknown hidden charges retained. [Decision evidence](resource-approvals/campaign-owner-autonomy-20261001.json).
- [Ticket 39](issues/39-evaluate-limited-encoder-fine-tuning.md) resolved: analysis 7966 completed,161 GPU seconds; 58 OOF outputs and all checkpoint hashes independently verified, exact three-fold GPU replay. AUC 0.518355 versus short frozen 0.464313, but image-signal controls remain inconclusive. No promotion; hidden-scale delivery cost unverified. [Result](../../docs/evidence/ticket-39-fine-tuning.md).
- [Ticket 51](issues/51-score-verified-public-frontier.md) claimed: accepted exact code submission 56739912/source 354194017, saved version 3. Persistent exact-reference observer running; initial state PENDING and no score. Charged 69,509, reserved 57,600 of 129,600 GPU seconds. Final private first place is unproven.

- October 1: persistent scoring observer and existing-thread continuation are both active for submission 56739912. One continuation waits for exact terminal accounting and parent completion; 62 focused checks pass. [Execution evidence](../../docs/evidence/ticket-51-scoring-execution.md).

- October 1 active-session continuation: [Ticket 53](issues/53-admit-public-meniscus-specialist.md) CPU admission completed in 42.493 seconds, zero GPU/spend. All eight public0033 payload files and 233 FP32 state tensors verified. One fixed 10% Medial Meniscus paired development request v3 is frozen against all 23 unchanged anchor cells; 12 cache/overlay/random-state tests pass. No GPU launch or predictive gain yet. Exact frontier 56739912 remains PENDING at 03:58:18 UTC, baseline remains 0.936 and leader 0.961. The primary session stopped the waiting background continuation coordinator to prevent overlapping agents; exact-scoring observer remains active. [Evidence](../../docs/evidence/ticket-53-public-meniscus.md).

- October 1, 04:35 UTC: actual CPU MRI preprocessing passed for all **58 studies / 273 selected slot-series** in **72.203 seconds**, no decode failures or fallback events, zero GPU/spend. Retained disk pixels/masks remain identical after cleanup. Current paired draft **v4** binds those hashes; all 23 anchor cells and the fixed 10% vote are unchanged. No GPU launch or measured predictive gain yet; exact accepted scoring 56739912 remains PENDING at 04:33:19 UTC. [Evidence](../../docs/evidence/ticket-53-public-meniscus.md).

- [Ticket 54](issues/54-test-longer-convnext-report-head.md) resolved: actual 7,500-step matched ConvNeXt heads completed on CPU, 256 allocated node seconds, zero GPU/spend. Report development AUC **0.691065** vs its short **0.659947**, delta **+0.031119**, interval **[-0.004120,+0.067897]**, one negative fold. All original prefixes and 116 final predictions replay exactly with frozen 16-thread settings. Report-minus-gold and image-ablation controls are positive; duration branch remains **inconclusive/no promotion**. No GPU ledger change, competition gain or final selection. [Evidence](../../docs/evidence/ticket-54-long-convnext-report-head.md).

- October 1 session transfer: [current handoff](session-handoff-20261001.md) replaces stale service/status snapshots. Exact submission **56739912** remains PENDING at **05:56:06 UTC**; observer running, old continuation coordinator stopped, no CPU jobs live. [Ticket 55](issues/55-test-matched-long-resnet-report-head.md) job **7977** failed on absent-report provenance after an exact first 1,500-step reproduction, **32 CPU-node seconds**, zero GPU/spend. Repair and four tests are complete; separate v2 recovery is **not frozen or launched**, with no long ResNet/blend result. [Evidence and next steps](../../docs/evidence/ticket-55-resnet-recovery.md). Ticket 53 paired v4 still awaits terminal scoring reconciliation and fresh GPU gates.

- October 1 resumed execution: [Ticket 55](issues/55-test-matched-long-resnet-report-head.md) resolved as **inconclusive/no promotion**. Training recovery **7978** and independent verifier **7981** completed; path-only verifier failure **7980** retained. All six original prefixes and **116** final predictions replay exactly; all seven comparisons reproduce. Report ResNet AUC **0.686730**; fixed half blend **0.705751**, delta **+0.019021**, interval **[-0.009373,+0.050577]**, one negative fold. **356 new CPU-node seconds**, zero GPU/spend. Exact frontier **56739912** is still **PENDING at 06:18:35 UTC**; fresh leaderboard leader **0.961**, confirmed fallback **0.936**. [Complete evidence](../../docs/evidence/ticket-55-resnet-recovery.md). Ticket 53 v4 remains the prepared next GPU comparison after terminal reconciliation.

- October 1, 06:45 UTC: [Ticket 56](issues/56-measure-report-supervised-encoder-adaptation-cost.md) resolved. Real report-supervised CPU layer4/head updates have exact native/cached parity; median update **1.315271 → 0.508122 s**, **28 CPU-node seconds**, zero GPU/spend. Registered scenario projects **2.305 node hours** and **37.586 GB** prefix payload, with warm-cache/scaling limits. [Ticket 57](issues/57-materialize-frozen-resnet-prefix-cohort.md) is the next CPU action: separately freeze complete unfitted activations and measure real cohort I/O before a three-fold adaptation decision. No predictive gain or full training yet. [Evidence](../../docs/evidence/ticket-56-cpu-adaptation-cost.md).


- October 1, 08:40 UTC, authoritative continuation: [Ticket 51](issues/51-score-verified-public-frontier.md) resolved as **terminal hidden-rerun error / no score**, exact 56739912/script 354194017/v3. Full unknown 57,600 GPU seconds charged; visible logs do not expose a hidden traceback. [Ticket 59](issues/59-diagnose-hidden-frontier-execution-failure.md) tracks the unreproduced failure; no speculative repair or scoring retry. Best confirmed account 0.936, leader 0.961 at 08:00:31 UTC, final private first place unavailable.
- [Ticket 53](issues/53-admit-public-meniscus-specialist.md) resolved as **reject registered overlay / no promotion**. Private paired v 1 completed 58 actual MRI studies/all 27 anchor prediction artifacts, zero fallbacks, exact native-cache agreement. Macro AUC 0.951479→0.951179, paired delta−0.000300, interval[−0.001226,+0.000348]; all subgroup fold deltas zero. Actual sole-run quota delta charges 2,528 GPU seconds. Concrete standing-authorized 36→38 h cap amendment retained. Ledger 129,637 charged/136,800 cap/zero reserved/7,163 remaining. Accessible success does not reproduce hidden failure. [Evidence](../../docs/evidence/ticket-53-public-meniscus.md).
- [Ticket 57](issues/57-materialize-frozen-resnet-prefix-cohort.md) resolved: all 4,228 studies/187,272 slices extracted by 7983/7984 and exhaustively verified by 7985, frozen reconstruction error 0.0. Unfitted 37.586 GB prefix payload,2,558 CPU-node seconds, zero GPU/spend. [Evidence](../../docs/evidence/ticket-57-frozen-resnet-prefix-cohort.md).
- [Ticket 58](issues/58-test-report-supervised-layer 4-adaptation.md) resolved as **inconclusive/no promotion**. Full-bank cost 7986, matched six-run fit 7989, independent original-DICOM verifier 7991 all completed 0:0, total 2,483 CPU-node seconds, zero GPU/spend. Adaptation 0.689232 vs matched head 0.691387, delta−0.002155, interval[−0.022304,+0.019717], one negative fold. All 116 final predictions/58 warm-parent outputs replay exactly; five comparisons and controls agree. No current CPU/GPU allocation or final selection. [Evidence](../../docs/evidence/ticket-58-report-layer 4-adaptation.md).

- October 1 full input continuation: [Ticket 59](issues/59-diagnose-hidden-frontier-execution-failure.md) remains in Phase 1. Broader unchanged-preprocessing probe **7992** passed all **256 studies / 48,870 DICOMs**, zero failed/fallback events, **264 CPU-node seconds**, zero GPU/spend. A distinct measured full-source scan of **4,407 studies** is now running as disjoint **7995 / 7996** partitions, each 16 CPUs / 64 GiB / three hours. Unlaunched v1 storage-record correction is preserved; executed v2 takes a direct measurement. No hidden-error attribution, ranked cause, repair or scoring retry. Fresh authenticated **08:56:47 UTC** retains **0.936** fallback and **0.961** leader. [Evidence](../../docs/evidence/ticket-59-hidden-execution-diagnosis.md).

- October 2 continuation: **7995 /7996** completed and independent summary verifies all **4,407 studies /819,078 original DICOMs**, zero decode/fallback events, **4,974 CPU-node seconds**. Integrity job **8012** checks every retained payload SHA and original archive CRC. Exact residual CoAt preparation pilot **8013** passes fixed **64** original-slice complexity order statistics/both extremes, **148 CPU-node seconds**, zero failures/warnings/fallbacks. Distinct all-study CPU CoAt scan **8014** is running, one six-hour/16-CPU/64-GiB attempt, unchanged native renderer/sampler and zero GPU/model/spend. Hidden error remains unreproduced. Latest authenticated **01:08:08 UTC**: **rank 2,005 /4,808, 0.936**, leader **0.961**, private rank unavailable. [Current handoff](session-handoff-20261002.md), [evidence](../../docs/evidence/ticket-59-hidden-execution-diagnosis.md).

- October 2 integrity completion: **8012** completed `0:0`, **779 verifier CPU-node seconds**. All **819,078 /569,154,778,424-byte** retained original DICOM payloads independently match SHA and archive CRC, with exact worker/plan/terminal reconciliation. Only full CoAt **8014** remains active. No source corruption or hidden cause found; ledger unchanged.

- October 2 execution lead: [research refresh](../../docs/research/competitive-next-route-2026-10-02.md) found no newly verified complete public route above 0.943. [Ticket 60](issues/60-establish-actual-tpu-training-capacity.md) claimed, one private offline TPU environment request **136722939 /v1** accepted and queued; actual hardware unverified. Exact source/settings pass and CPU negative check rejects missing XLA. Separate TPU ledger reserves 600 seconds, GPU ledger unchanged, no training or promotion. [Evidence](../../docs/evidence/ticket-60-tpu-environment.md). CoAt job **8014** continues, 962 COMPLETE interim records with zero preparation/sampler/sparse errors.

- October 2 full-backbone learning route: [Ticket 61](issues/61-prepare-fixed-shape-full-backbone-report-training.md) code and real-MRI CPU cost pilot **8016 /0:0** pass, 127 node seconds. Median whole-encoder update 5.955313 s, all stages change with fixed normalization; exact native initial logits and independent diagnostic checkpoint replay. 39 focused tests/83-source-file typing pass. CPU can advance independently of queued TPU; long 7,500-update route needs staged-job, cache/I/O and complete comparison/delivery admission. No predictive gain or promotion. [Evidence](../../docs/evidence/ticket-61-full-backbone-cpu-cost.md).

- October 2 whole-cohort preparation: [Ticket 62](issues/62-materialize-exact-mri-pixel-cohort.md) pixel materialization **8017 /0:0**, 2,080 CPU-node seconds; all 4,228 studies /187,272 slices independently replay native sources exactly, 37.586 GB payload, 189.317 s verified cache reads/reconstruction. Shared full-encoder CPU training, file-backed pixel inputs, exact checkpoint resume and native prediction pass 57 focused tests/84-file typing. [Ticket 63](issues/63-measure-shared-full-backbone-training-delivery-cost.md) actual mixed-source training/startup/checkpoint cost **8018** is running, one 90-minute/16-CPU/64-GiB job. Complete 7,500-update comparison remains separately unadmitted. CoAt **8014** continues; exact TPU request remains queued. No predictive gain/rank change. [Pixel evidence](../../docs/evidence/ticket-62-exact-mri-pixel-cohort.md), [medical encoder research](../../docs/research/medical-image-encoder-options-2026-10-02.md).


- [Ticket 62](issues/62-materialize-exact-mri-pixel-cohort.md) resolved: all
  4,228 exact pixel bags independently replay original source packing; measured
  whole-cohort I/O and separate complete Ticket 64 fitting/delivery admission
  are recorded. No accuracy result follows from materialization.
- [Ticket 63](issues/63-measure-shared-full-backbone-training-delivery-cost.md)
  resolved: job 8018 completed `0:0`, 2,170 CPU-node seconds. Fresh-process
  staged and continuous real-MRI model/Adam/scheduler/progress/sampler states
  agree exactly; all encoder stages change with fixed normalization. Actual
  startup/serialization/RSS establish costs for prospectively registered Ticket 64.

- [Ticket 65](issues/65-admit-pinned-biomedclip-assets.md) resolved: sole pinned
  BiomedCLIP acquisition passed independent full SHA/byte checks and restricted
  CPU loading, **28.471 s /1.17 GB peak RSS**, 151 finite FP32 vision tensors.
  Six boundary tests pass; fresh rules and exact MIT notice retained. No MRI
  inference, fitting, scheduler job, accelerator use or spend. Strict official
  vision loading and fixed eight-study pilot require their next separate
  admission; registered Ticket 64 continues.

- [Ticket 66](issues/66-measure-biomedclip-vision-mri-parity-and-cost.md) resolved:
  sole **8037 /0:0**, **217 CPU-node seconds**; all eight original MRI studies
  /344 outputs independently replay exactly, strict 151 FP32 vision tensors.
  Median slice **0.038754 s**, maximum **0.069421 s**, peak RSS **3.109 GB**.
  Exact fit scheduling restored immediately after pilot admission; **8036** and
  resumed fold-1 stage-1 **8039** continue. Eleven guards pass. No fitted
  accuracy or rank evidence. [Completed evidence](../../docs/evidence/ticket-66-biomedclip-mri-admission.md).

- [Ticket 67](issues/67-materialize-frozen-biomedclip-cohort.md) resolved:
  all4228 studies/187272 slices extracted in four sole successful partitions,
 8071 independently checks every byte/source/mask/norm plus fixed344 native
  outputs exactly. Total7885 CPU-node seconds,1178253654 feature/pooled bytes,
  no labels/fitting/accuracy. [Completed evidence](../../docs/evidence/ticket-67-biomedclip-feature-cohort.md).

- [Ticket 68](issues/68-test-matched-biomedclip-report-heads.md) claimed:
  complete bank then actual matched-head plan admitted,
  `16ee8b86a75aa30a0c6a1f25e68ee65028a084914b7863584e1abb540c8bc666`.
  Six7500-total-step random-init/source-matched heads plus all58 medical/parent
  native verification, one fit/one verifier/max2 CPU jobs/no retry. Combined
  controller owns unchanged scheduling;8054/8072 occupy both slots, no medical
  fit/result yet. Route/review projects October9 19:14UTC.
  [Workflow evidence](../../docs/evidence/ticket-68-biomedclip-matched-heads.md).

- [Ticket69](issues/69-prepare-portable-medical-inference.md) claimed:
  additive strict CPU medical inference adapter, explicit held-out-fold versus
  probability mean, one encoding per valid slice and eligible-reference fallback
  flags.26 synthetic tests and typing pass; real-MRI/package parity pending.
  Read-only audit confirms exact0.936 source is five cells, distinct from
  cached23-cell D4 lineage; exact scored-anchor Gold58 parity remains unproven.
  Bank completed and independently verified; unchanged8054/8072 fits run.
  [Evidence](../../docs/evidence/ticket-69-portable-medical-inference.md).

- [Ticket70](issues/70-establish-exact-scored-anchor-predictions.md) claimed:
  exact0.936 scored-source Gold58 request separately prepared and independently
  checked; pinned envelopev2 retains identical notebook while binding7 explicit
  dataset versions and saved container. Eight source tests/typing and ten new
  envelope guards pass. Historical weight-byte/MRI/runtime parity and resource
  admission/launch remain unproven. D4 final cache has different members/fusion; shared calibration
  payload and50 matching top-level definitions do not prove predictions.
  [Evidence](../../docs/evidence/ticket-70-exact-scored-anchor.md).

- [Ticket71](issues/71-compare-medical-heads-with-scored-anchor.md) claimed:
  fixed90% exact-anchor/10% report-medical OOF rank vote declared before medical
  results, all12 targets,2000 paired bootstrap20261002, all-fold directions plus
  zero-image controls. Diagnostics are fixed and cannot become alternate
  candidates. Blocked by actual68/70 receipts; no analysis/gain/promotion yet.

October2,12:40UTC — Ticket60 actual TPU environment independently verified
COMPLETE136722939/v1: eight global/addressable TPU devices, no CUDA, exact FP32
matrix result. Full600-second conservative charge, zero reserved. Retained proof
SHA `eaad4dd0ed158c9478bc749195593f31b1a3f315d1b3b53031e374289b9e254c`.
Fresh typed free quota12:45UTC GPU57178.249/TPU71883.65, paid scaling off. No MRI
training/delivery pilot or predictive gain; Ticket60 remains claimed for that
separate stage. [Evidence](../../docs/evidence/ticket-60-tpu-environment.md).

October2,12:57UTC — sole Ticket68 medical fit8073 completes0:0/357CPU-node seconds. Native MRI replay still awaits CPU slot; no interim predictive read. Full fold0/stage2 8054 completes0:0/6833seconds/exact3000 boundary; successor8074 runs,target4000. Controller owns two slots8074/8072; science unchanged.

October2,16:20UTC — native medical verifier8106 running through unchangedT68plan.
Full8072/8074 worker checkpoints complete but backend terminal ancestry missing
since observer/controller interruption14:04; consumed jobs never retried.
[Recovery72](issues/72-recover-interrupted-cpu-scheduling.md). T70v3 full MRI/runtime
proof envelope and launch/output lifecycle frozen, complete resource admission
f6f5628623c095d06a44c2b4f5cf3e516fb49d78e195c11d78233ddf05a1a452, unlaunched.
T71 fixed analysis/replay prepared f90bb2ad7a216eb3a235595775ae5d559569b707a0b8b6b5b20369b57cb2934d;
no final input/resource admission or predictive analysis.103 combined guards and
module typing pass. Portable medical adapter is excluded from new45-module
analysis closure; all frozen source snapshots must remain unchanged.

- October2,16:47UTC: [T68](issues/68-test-matched-biomedclip-report-heads.md) resolved with exact native proof/inconclusive_no_promotion; [T70](issues/70-establish-exact-scored-anchor-predictions.md) v3 failed before code due missing kernelspec, full bound charged; separate unchanged-code v4 admitted; [T71](issues/71-compare-medical-heads-with-scored-anchor.md) v2 mechanically binds v4 with identical fixed science. T72 missing full-fit terminals remains open.

- October2,16:49UTC: exact baseline v4 numeric136806200/v1 RUNNING/source+kernelspec PASS, one development attempt/no daily scoring slot consumed. Owner explicitly says preserve daily submissions; recorded in spec and handoff.

- October2,16:57UTC: [T72](issues/72-recover-interrupted-cpu-scheduling.md) bounded independent state audit prepared against immutable copies; complete prior route retained. No fitting, accuracy scoring, ancestry repair or daily submission.

- October2,16:58UTC: independent checkpoint audit8108 RUNNING on an authorized CPU node; exact bounded allocation recorded. Continuation waits for actual successful audit and separate admission. Baseline development136806200/v1 remains active; daily scoring slots preserved.

- October2,17:23UTC: [T70](issues/70-establish-exact-scored-anchor-predictions.md) resolved with exact original MRI/runtime/full-member/fusion proof; [T71](issues/71-compare-medical-heads-with-scored-anchor.md) fixed analysis fully admitted; [T72](issues/72-recover-interrupted-cpu-scheduling.md) audit passed, third original1000-update fold2 stage verified, separate full14500-update continuation admitted. No daily scoring attempt.

- October2,17:28UTC: [T71](issues/71-compare-medical-heads-with-scored-anchor.md) resolved, independently rejected fixed medical overlay (all fold deltas negative); daily slots preserved. [T72](issues/72-recover-interrupted-cpu-scheduling.md) continuation8111/8113 now RUNNING with exact bounded CPU resources and persistent observers. Final7500 additional updates/fold and native checks remain pending.

- October2,17:45UTC: continuation8111 FAILED before updates (missing immutable config); original coordinator stopped. Exact config handoff preserved live8113, now progressing beyond3000. Three real-trainer regression guards pass; [T73](issues/73-carry-configuration-through-continuation-recovery.md) tracks separate re-admission with full original science and honest failure accounting.

October2 configured continuation: Ticket73 v4 plan9844fd6a7e62f3f1aaf5d0678784f197b8373d702e14c19e669ccde5244a36c8 admitted; coordinator rsna-ticket73-configured-continuation-v4.service live, new fold0/stage4 job8119 RUNNING plus preserved v3 fold1/stage3 job8113 RUNNING. All3 unchanged original configurations carried before launch. Fold1 successor gates on actual8113 successful terminal plus immutable4000-step input closure; no live-checkpoint copy/repeatedupdates.15 new stages13500 updates+live1000, full7500perfold/control/native/science unchanged. Retain39600unknown historical charges+349actualfailed8111. Conservative entire unfinished route525600seconds→Oct8 19:55:50UTC,1.972567098522TB.45 guards pass. Source files/helper now frozen; do not edit/restart consumed intents. No scoring slots used; T71 reject remains binding. Evidence docs/evidence/ticket-73-configured-continuation-recovery.md.
