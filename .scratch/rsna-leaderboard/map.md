# Find a credible route to first place with zero external spend

Labels: wayfinder:map
Status: resolved

## Destination

A decision-complete experiment and delivery plan aimed at first place on the final private leaderboard, using zero external spend and reproducible, leakage-controlled evidence. The map is complete when the supervision, validation, model, compute, and candidate-selection decisions are sufficient to begin execution; achieving first place is the ambition of that execution, not a promise or the completion criterion of this planning map.

## Notes

- Owner confirmed final private first place, zero external spend, GPU availability dependent on the queue, and no qualified clinical reviewer on 2026-09-25.
- Owner requested resolving all remaining decisions in this continuation, overriding wayfinder's default one-ticket-per-session limit. Human decision tickets still require the owner's answers; research and read-only inventory can proceed independently.
- Ask all questions directly in visible chat and stop for the owner's answer; asynchronous question widgets are inaccessible to this owner.
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
