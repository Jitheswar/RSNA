# RSNA leaderboard campaign with zero external spend

Status: ready-for-agent
Labels: ready-for-agent
Created: 2026-09-25
Decision source: [Completed wayfinder map](map.md)

## Problem Statement

The owner wants first place on the final private leaderboard of the RSNA Knee Abnormality Detection competition, earned through real predictive performance and reproducible evidence. The current system can deliver a scored submission, but it has not established useful image discrimination. Its recorded public score is 0.498; corrected three-fold reference OOF macro AUC is 0.464313 across 58 studies, with a 95% bootstrap interval of approximately 0.404–0.524. Training-prevalence controls and within-fold diagnostics do not establish measurable image signal.

Previous experiments do not settle whether the limiting factor is input quality, training duration, image representation or supervision. The wider image experiment only increased a small CNN's channels; it did not evaluate transfer learning. Report supervision and distillation comparisons were inconclusive. A working pipeline and successful synthetic checks cannot substitute for evidence of learning from real MRI studies.

Resources constrain the route. External spending must remain zero. GPU access depends on the cluster queue, and there is no qualified clinical reviewer available. Only 58 of the 4,407 inventoried training studies have reference labels; the other 4,349 have reports. Existing image-training preparation covers 158 studies, including the 58 reference studies and 100 report-only studies. A complete header inventory and broad report extraction do not establish complete decoded image caches or reliable twelve-target supervision.

The owner requires delivery before October 15, 2026. The plan must produce credible experiments, preserve an executable fallback and leave enough time for offline delivery checks. First place is the competition objective, not a guaranteed result or an acceptance criterion that software checks alone can prove.

## Solution

Execute the agreed staged campaign through the existing study-training, evaluation and candidate-delivery workflow. First establish that the input and optimization paths work on real training studies. Then evaluate a small pretrained ResNet-18 representation with anatomical pooling and fold-specific prediction heads. Audit existing learned report outputs before spending resources on another extractor. Expand supervision, image preparation and model complexity only when their evidence and measured cost justify the next stage.

Use the corrected three-fold development split, simple controls, paired uncertainty and explicit provenance for every comparison. Treat promising but inconclusive models as provisional candidates. When public scores disagree with local validation, investigate the disagreement and prioritize reproducible local evidence. Preserve the historical baseline and all contrary results.

The first campaign has a 20 allocated GPU-hour budget, one GPU request at a time, and a CPU route while queued. There is no project-imposed numerical cap on candidate evaluations or scored submissions; official limits, the compute budget and the delivery schedule still apply. Freeze scientific changes by October 11, finish delivery checks by October 13 and complete final selection by October 14, all in IST.

## User Stories

1. As a competition owner, I want the campaign to aim at final private first place, so that development serves the actual competition objective.
2. As a competition owner, I want measured results distinguished from goals, so that progress is never presented as an invented rank or score.
3. As a competition owner, I want zero external spending, so that the campaign uses only existing resources and admissible free resources.
4. As a competition owner, I want delivery complete before October 15, so that queue delays and packaging problems have a bounded recovery window.
5. As a competition owner, I want no arbitrary submission-count cap, so that justified candidates can be evaluated within the official rules and remaining resources.
6. As a competition owner, I want the existing scored candidate preserved, so that an unsuccessful experiment does not remove the fallback.
7. As a competition owner, I want campaign decisions handled autonomously within the competition rules and zero-spend constraint, without repeated approval questions, as authorized on October 1.
8. As a researcher, I want the current near-chance baseline recorded accurately, so that improvements are compared with real evidence.
9. As a researcher, I want reference labels kept distinct from report-derived labels, so that weak supervision cannot masquerade as ground truth.
10. As a researcher, I want prepared image membership distinguished from header inventory, so that corpus size claims match the images actually used.
11. As a researcher, I want exact source and cache identities checked before experiments, so that comparisons use the intended data.
12. As a researcher, I want the corrected three-fold manifest reused, so that new comparisons remain comparable with corrected historical results.
13. As a researcher, I want reliable patient grouping and duplicate evidence respected when available, so that related studies cannot silently cross validation boundaries.
14. As a researcher, I want unresolved grouping limitations disclosed, so that study-level validation is not overstated.
15. As a researcher, I want study identity and target order traced through the loss, so that a data-join defect cannot look like an unlearnable task.
16. As a researcher, I want real pixel content, geometry, sampling and valid masks checked, so that the model receives informative MRI inputs.
17. As a researcher, I want a tiny training-only memorization diagnostic, so that optimization competence can be checked without consuming held-out labels.
18. As a researcher, I want gradient and parameter-change evidence, so that inert training is detected before a larger run.
19. As a researcher, I want bounded learning curves beyond the old short budget, so that insufficient training is not confused with a poor representation.
20. As a researcher, I want diagnostic success separated from generalization, so that fitting training images is never called validation improvement.
21. As a model developer, I want an admitted ResNet-18 initialization, so that transfer learning is evaluated before expensive model scaling.
22. As a model developer, I want frozen slice features as the first representation experiment, so that fold-specific heads can be compared at modest compute cost.
23. As a model developer, I want anatomical plane and series information preserved, so that pooling does not discard useful study structure.
24. As a model developer, I want explicit missing-plane and invalid-slice handling, so that heterogeneous studies still yield valid predictions.
25. As a model developer, I want new-resolution preprocessing versioned separately, so that old 128-pixel caches are not mistaken for the selected pretrained representation.
26. As a model developer, I want deterministic feature-cache identities, so that reusable features remain tied to the exact encoder and preprocessing policy.
27. As a model developer, I want held-out feature processing separated from adaptation, so that frozen prediction does not become validation leakage.
28. As a model developer, I want limited fine-tuning considered only after diagnostics and resource checks, so that queued GPU time tests a defensible hypothesis.
29. As a model developer, I want DINOv2 reserved as a controlled alternative, so that an unhelpful first representation does not trigger an unrestricted architecture sweep.
30. As a model developer, I want adjacent-slice context evaluated as a separate change, so that any benefit can be attributed to the intended input change.
31. As a supervision researcher, I want existing learned outputs audited first, so that usable prior work can be evaluated before another extraction run.
32. As a supervision researcher, I want extraction provenance and fold eligibility reconstructed, so that inadmissible outputs never reach training.
33. As a supervision researcher, I want uncertainty, missing mention and extraction failure retained separately, so that unsupported negatives are not invented.
34. As a supervision researcher, I want target-level reliability and coverage, so that a high aggregate accuracy cannot hide unsupported findings.
35. As a supervision researcher, I want new extractor coverage assessed separately from agreement, so that two agreeing labelers are not treated as independent ground truth.
36. As a supervision researcher, I want reference labels to remain authoritative, so that report-derived disagreements are handled consistently.
37. As a supervision researcher, I want unresolved clinical disagreements left uncertain, so that the absence of a clinical reviewer is reflected honestly.
38. As a supervision researcher, I want a prospective admission policy frozen before evaluation, so that thresholds are not fitted retrospectively to held-out outcomes.
39. As a supervision researcher, I want reference exposure matched across paired arms, so that weak-supervision benefit is not confounded by reduced reference training.
40. As a supervision researcher, I want one local multilingual extractor considered only if existing outputs are inadequate, so that additional extraction has a clear purpose and bounded cost.
41. As a data engineer, I want image preparation to expand in restartable batches, so that the report-only corpus becomes usable without an uncontrolled storage increase.
42. As a data engineer, I want a project storage ledger, so that shared filesystem free space is not mistaken for room inside the project's budget.
43. As a data engineer, I want original data and protected artifacts retained, so that cache expansion never destroys the ability to reproduce earlier runs.
44. As a evaluator, I want pooled and within-fold AUC reported together, so that fold-dependent score offsets remain visible.
45. As a evaluator, I want per-target support and paired uncertainty, so that rare labels and unstable estimates are not hidden by one macro score.
46. As a evaluator, I want undefined bootstrap outcomes counted, so that uncertainty is not made artificially favorable.
47. As a evaluator, I want constant and prevalence controls plus an image-information diagnostic, so that apparent learning is checked against simple alternatives.
48. As a evaluator, I want complete fold evidence before promotion, so that one favorable fold is not presented as full OOF performance.
49. As a evaluator, I want reused validation identified as development evidence, so that adaptive model selection is not described as an untouched test.
50. As a evaluator, I want public/local disagreement investigated before retuning, so that leaderboard feedback does not silently drive overfitting.
51. As a cluster user, I want significant work submitted through Slurm, so that the login node remains available for coordination.
52. As a cluster user, I want CPU work within the established ceilings, so that useful preparation can continue while GPUs queue.
53. As a cluster user, I want one active or pending GPU request at a time, so that the first campaign has a controlled resource footprint.
54. As a cluster user, I want every allocated GPU attempt counted toward 20 hours, so that failures and extraction work do not escape the budget.
55. As a cluster user, I want actual device and assigned-memory checks, so that an assumed A40 or whole-GPU allocation cannot cause an avoidable failure.
56. As a cluster user, I want measured smoke-run cost before a full proposal, so that fold budgets are grounded in the selected model's behavior.
57. As a cluster user, I want resumable, atomically saved checkpoints, so that an interruption does not erase usable progress or corrupt the prior checkpoint.
58. As a cluster user, I want retries tied to the existing transient-failure policy, so that deterministic failures do not consume repeated allocations.
59. As an experiment owner, I want immutable scientific configurations and separate attempt records, so that operational retries do not conceal changed experiments.
60. As an experiment owner, I want exact job proposals prepared before approval, so that I can review the real resource request rather than a vague plan.
61. As a delivery owner, I want frozen model and dependency identities, so that local evidence belongs to the exact notebook candidate.
62. As a delivery owner, I want MRI-only offline inference with complete study coverage, so that hidden-test execution respects the input and output contract.
63. As a delivery owner, I want explicit decode and fallback accounting, so that successful CSV creation does not hide unusable image inputs.
64. As a delivery owner, I want measured full-pipeline runtime, so that model-only timing does not conceal delivery overhead.
65. As a delivery owner, I want example execution, representative measurements and hidden scoring reported separately, so that each result supports only the claim it actually demonstrates.
66. As a delivery owner, I want simple ensembles admitted only with OOF and runtime evidence, so that extra models earn their inference cost.
67. As a delivery owner, I want operational optimization checked for prediction equivalence, so that a speed improvement cannot silently change the frozen science.
68. As a delivery owner, I want final choices based on local evidence, robustness and applicable official allowances, so that selection remains reproducible and valid.
69. As a delivery owner, I want late failures to reduce scope or invoke the fallback, so that they do not move the agreed delivery date.
70. As a competition owner, I want a final record of completed, inconclusive and unrun work, so that the campaign's outcome remains honest even if the ambition is not achieved.

## Implementation Decisions

### October 1 owner authorization and current resource boundary

The owner answered **“yes”** to the exact Ticket 51 20→36 GPU-hour extension and Ticket 39 30-minute GPU replay proposals, then instructed **“approve anything as long as we win 1st place and fits kaggle comp dont ask me”**. This supersedes earlier requirements to pause for routine campaign decisions. Preserve concrete scientific/resource plans and decision evidence; act autonomously when the action fits Kaggle rules and the existing $0 spending constraint. First place remains an objective requiring actual final private leaderboard evidence.

The current ledger cap is **129,600 GPU seconds (36 hours)**. The 57,600-second increase is committed to one exact Ticket 51 scoring attempt. Job 7966 completed Ticket 39 analysis before that scoring reservation; its full 161-second controller allocation is charged. Existing unknown hidden allocations retain their full conservative charges. One active or pending GPU request across backends, CPU/storage ceilings, frozen experiment identities and protection of prior artifacts remain applicable. Any subsequent resource change must have a concrete, justified campaign decision recorded under this owner authorization; it must not silently erase accounting or bypass official limits.

Authorization: [owner instructions](resource-approvals/campaign-owner-autonomy-20261001.json), [exact scoring approval](resource-approvals/ticket51-frontier-scoring-20261001.json), [applied cap amendment](resource-approvals/cap-amendment-20261001.json).

### Existing workflow and interfaces

- Extend the current inventory, geometry-aware study loading, series cache, fold training, supervision admission, OOF comparison, frozen-candidate and Kaggle-delivery modules. Keep the existing study-to-twelve-predictions contract as the public behavior; avoid building a parallel training or submission stack.
- The current image model, training configuration and candidate loader assume the compact architecture. Introduce versioned representation/model configuration and shared model construction where needed so that training, checkpoint reload, frozen candidates and offline inference reconstruct the same selected encoder and pooling behavior. Preserve loading of existing baseline artifacts.
- Carry anatomical descriptors and their unknown/missing states through packing and pooling without changing study identity, target order or label semantics. Existing slice and series masks remain authoritative. Unknown descriptors do not justify fabricated anatomical labels.
- Reuse the CPU fold-training interface and the separate approved GPU execution path. Do not add a device argument or alternate entry point that bypasses GPU authorization. Reuse OOF aggregation and paired comparisons rather than independently reimplementing the metric in a new model runner.
- Extend the frozen-candidate manifest to cover any added encoder weights, preprocessing, feature representation, anatomical pooling, model configuration and offline dependencies. A model that trains but cannot be reconstructed by the shared delivery path is not a deliverable candidate.

### Input readiness and scientific identity

- Begin diagnostics inside the existing reference image scope. Reconcile selected source membership, target order, source identity, corrected fold identity and input/cache checksums before use. The larger metadata inventory is a preparation source, not proof of image readiness.
- Establish retained and projected storage before expanding image preparation. Include source archives, extracted inputs, caches, temporary atomic-write duplication, environments and preserved candidates in the 2 TB decimal soft budget. Shared free space is not reserved capacity.
- Prepare additional eligible report-only images in deterministic, restartable batches. Record actual admitted studies, series, usable slices and supervised target entries. Preserve exclusions and failures; never silently replace missing studies with a claim of whole-corpus training.
- Use the corrected version-2 three-fold manifest for new development. If reliable grouping or duplicate evidence reveals contamination, correct the split and rerun affected comparisons. Keep historical results identifiable as superseded. Otherwise disclose the existing study-level grouping limitation.
- Every scientific run binds source and fold identities, admitted supervision, preprocessing and sampling, architecture and initialization, optimizer/schedule, loss, augmentation, seed, checkpoint-selection rule and budget. A semantic change creates a new run with parent provenance; a retry retains the same scientific identity.

### Ordered learning campaign

- First diagnose real-input alignment, nontrivial pixel content, geometry, masks, supervised loss, gradient flow, parameter updates and a tiny training-only memorization task. Freeze its subset and bounded numerical success criteria before execution. Success proves optimization competence only; failure leads to diagnosis or repair before architectural expansion.
- Inspect bounded learning curves beyond the previous 100-step budget. Do not infer convergence from the historical step count, or select a favorable held-out checkpoint after viewing all outcomes. Freeze the budget and checkpoint policy in each experiment proposal.
- The first transfer-learning route uses the official ResNet-18 ImageNet initialization identified as `ResNet18_Weights.IMAGENET1K_V1`, after exact resource admission. Begin with frozen slice features, anatomical pooling and small fold-specific heads trained on reference labels. Retain a simple pooling comparator.
- Define a fixed geometry-aware slice/series policy, keep the whole knee field with documented resize/padding, and version grayscale-to-encoder normalization. Treat the proposed 224-pixel representation as a new preprocessing identity; do not reuse the 128-pixel materialization as if it were equivalent.
- Feature-cache identity covers source membership, preprocessing and sampling, encoder revision/digest, output representation and dtype. A fixed external encoder may be reused across fold-specific heads. Any normalization fitted to data, pooling adaptation, dimensionality reduction or encoder fine-tuning remains training-fold-specific.
- Limited fine-tuning follows input integrity, optimization competence and a measured resource fit. DINOv2 ViT-S/14 is the controlled alternative representation if the first route is unhelpful. Adjacent-slice 2.5D is a subsequent isolated input change. None is an automatic parallel sweep or a demonstrated improvement.
- Complete the declared folds and matched comparisons before promotion. Uninformative results return the campaign to the appropriate input, supervision or representation question. Do not create a favorable result by changing the reported metric or suppressing failed runs.

### Report-derived supervision

- Audit existing learned report outputs before running another extractor. Require reconstructable source/prompt/output identity, eligible report cohorts, fold exclusions, adaptation lineage and applicable usage permissions. Presence of an output artifact is not training admission.
- Preserve positive, negative, not-mentioned, uncertain, unknown and extraction-failed states. Only admitted positive/negative evidence contributes to masked supervision. Reference labels take precedence. Missing mention is not a negative label, and model-stated confidence is not calibrated reliability.
- Freeze a prospective target-level admission policy using only eligible training-side information. Report support, coverage, positive/negative reliability, failures and uncertainty. Evaluate newly covered calls separately from shared calls. Agreement between extractors does not become reference truth.
- Without a clinical reviewer, do not claim to adjudicate whether a report/reference discrepancy reflects extraction error or a different clinical assessment. Unsupported targets remain masked. Preserve the existing official-definition-based interpretation boundary.
- Compare reference-only and admitted report-supervision arms with matched reference exposure, initialization and evaluation cohorts. Record weak-label exposure separately. Retain the deterministic route as a comparator and preserve prior inconclusive results.
- If existing outputs are ineligible or offer inadequate trustworthy coverage, prepare one local multilingual alternative with an exact resource review and measured extraction cost. Qwen3-8B remains the researched candidate, not an automatically approved download or a required successful branch. No fresh remote report transfer is included.

### Validation and promotion

- Primary development estimate is pooled twelve-target reference OOF macro AUC. Also report per-target class support and AUC, fold metrics, pair-weighted within-fold AUC, prevalence controls and score-offset diagnostics. Compare identical held-out cohorts.
- Use paired bootstrap uncertainty with 2,000 resamples for real comparisons, retaining undefined counts. Resample reliable patient groups when available; otherwise use studies and disclose the limitation. Small synthetic fixtures may use fewer resamples to verify metric behavior without pretending to establish scientific performance.
- Every adaptation source must exclude the consuming held-out fold, including report-derived rules, prompt examples, thresholds, fitted transforms, teacher/student labels and self-supervised adaptation. Frozen external inference is distinct from adaptation.
- Assess image information against constant and training-prevalence controls, within-fold discrimination and a declared image-ablation or shuffle diagnostic. Training memorization or a tiny positive macro delta alone is insufficient evidence of generalization.
- Distinguish a provisional candidate, a supported scientific improvement and a final submission choice. A wide interval may permit another bounded experiment; it remains inconclusive. Do not impose an invented universal significance threshold or claim an untouched test from repeatedly reused folds.
- When local and public scores disagree, investigate pipeline differences, cohort shift, fallbacks, fold score scales and uncertainty before retuning. Local reproducible evidence remains the primary selection basis. Avoid unrestricted per-target ensemble-weight search on 58 reference studies.

### Resource control and recovery

- External spend remains zero. Significant work runs through Slurm, not the login node. Autonomous CPU work remains within one nonexclusive CPU-partition node, 16 CPUs, 64 GB, six hours per job and at most two concurrent jobs.
- The first campaign is capped at 20 allocated GPU-hours across all attempts, including smoke checks, feature extraction, report extraction, training, failures and retries. Queue wait is recorded separately. Only one GPU request may be active or pending at a time.
- Plan a smoke of at most 30 minutes for a new full-run shape. Initial full-job ceilings are one GPU, 16 CPUs, 64 GB and six hours, with measured device, sharing, memory and runtime evidence. Historical A30X evidence does not establish an available A40 or guarantee the next allocation.
- Reserve sufficient remaining budget for every fold in a proposed comparison before starting it. If the estimate does not fit, narrow the planned experiment or retain frozen features and CPU heads. Do not spend the remaining budget on one fold and call it complete OOF.
- Prepare concrete immutable job/resource proposals and CPU preflight evidence before the existing GPU approval step. Preserve any already applicable explicit authorization; a planning decision alone does not authorize a new exact request.
- Continue diagnostics, provenance review, the storage ledger and bounded input preparation while queued. CPU feature extraction is conditional on measured throughput and restartable chunks within the CPU ceiling; it is not promised to finish merely because it avoids a GPU.
- Persist atomic resumable checkpoints with training state, progress, random state, configuration and attempt identity. Keep important artifacts on persistent storage. Resume unchanged science only; preserve prior complete checkpoints if a write fails.
- Retain the established single identical-request retry for an approved transient infrastructure failure. OOM, divergence, deterministic loader defects, changed configuration and user cancellation are not automatic retries. Every allocated attempt consumes the campaign budget.
- Stop GPU expansion at the budget or modeling cutoff. A later campaign needs a new concrete budget decision; there is no automatic extension or spending allowance.

### Candidate selection and delivery

- There is no project-imposed numerical cap on candidates or scored submissions. Each new candidate still needs a hypothesis, immutable identity, applicable local evidence and a reason to use the remaining time. Official submission limits remain binding and must be refreshed before dependent actions.
- Subject to the current official final-selection allowance, plan up to two final choices. Prefer the strongest admissible validated single model and a complementary simple equal-weight ensemble only if complete paired OOF and full runtime evidence justify it. Otherwise use the strongest admissible alternatives; do not manufacture a second method merely to fill a slot.
- Tie-break close or inconclusive alternatives using fold/target consistency, robustness, simpler execution and runtime headroom. Preserve the existing scored frozen candidate as the operational fallback; its availability does not establish competitive performance.
- Use the shared MRI-only inference path with internet disabled. Produce one row per expected study in exact target order, twelve finite bounded probabilities and the required submission filename. Never omit an all-invalid study; use the established eligible-reference fallback and report its use.
- Measure complete discovery, decoding, preprocessing, inference, ensemble and output-writing cost. Keep representative measurements, projections, notebook example runs and completed hidden scoring separate. Do not infer full hidden-scale timing or recovery from the earlier three-study notebook evidence.
- Freeze the code, weights, preprocessing, runtime dependencies and output contract as one candidate identity. Verify the exact offline bundle and actual code-competition submission state; a local pass or a submission request is not a completed Kaggle score.
- First optimize redundant discovery/decoding/cache work while demonstrating prediction equivalence. If the selected method is too slow, reduce scientific model/ensemble/sampling cost before the freeze and reevaluate, or deliver the verified fallback. Operational fixes after the freeze must preserve scientific identity and repeat affected checks.
- Keep exact uploads, scored submissions and final account selections within their existing concrete approval boundaries. Prepare a reviewable bundle first. No rules acceptance, publication or protected deletion is implied by this spec.

### Delivery schedule

All internal dates are in IST and refer to 2026.

| Deadline | Outcome |
| --- | --- |
| September 28 | Initial learning diagnosis, resource/source review, storage ledger and first concrete experiment proposal. |
| October 3 | Target first complete three-fold representation comparison. |
| October 8 | Review supervision benefit, model evidence and remaining measured compute/runtime cost. |
| October 11, 23:59 | Freeze scientific configurations, checkpoint identities and the delivery shortlist. |
| October 13, 23:59 | Finish offline packaging, representative pipeline checks and intended Kaggle delivery verification. |
| October 14, 23:59 | Complete explicit final selection and preserve reproducible delivery evidence. |

Missed milestones reduce scope or invoke the existing fallback; they do not move the owner's before-October-15 target. The official competition deadline is a separate external constraint to refresh before action.

## Testing Decisions

- **Primary testing seam: the existing experiment-to-frozen-candidate lifecycle.** Use a small admitted study fixture and corrected fold manifest, train through the existing fold interface, aggregate complete held-out predictions, freeze the candidate and run its offline study-to-submission interface. A test-only integration harness may compose these existing public calls; do not introduce a new production orchestration API solely to make tests convenient.
- A good test asserts observable outputs and failure behavior: expected studies and target order, provenance and cohort agreement, reproducible predictions, complete metrics, checkpoint reload and typed rejection of invalid inputs. Avoid asserting private call order, internal layer names or the implementation's own helper calculations.
- Reuse prior art from the current fold-training, OOF aggregation, paired image/supervision comparison, candidate freeze/load, real-DICOM inference and offline Kaggle adapter tests. Existing generated DICOM fixtures, immutable artifact fixtures and interrupted-run cases already exercise the appropriate interfaces.
- Extend the lifecycle for the new representation: new-format checkpoints reload into the same preprocessing/model behavior used during evaluation; anatomical missingness and padded inputs do not corrupt study outputs; the historical compact candidate remains readable. Use tiny local fixture encoders for routine software tests, with a separate bounded admitted-weight check for actual pretrained compatibility.
- At the same artifact boundaries, reject missing/duplicate held-out rows, changed source or fold identities, incompatible scientific configurations, altered checkpoints and stale preprocessing/feature caches. Reject held-out influence in weak-label or fitted-feature provenance. Confirm that errors leave previous immutable artifacts intact.
- Verify supervision through observable admission and comparison results: masked uncertainty contributes no supervision, reference labels win conflicts, unsupported coverage stays undefined, and paired arms preserve reference exposure. Do not use synthetic extraction accuracy as evidence of clinical reliability.
- Verify OOF behavior with controlled ranking examples: the expected macro delta and per-target results are reported, undefined bootstrap outcomes are counted, within-fold controls remain visible, and incomplete/mismatched cohorts cannot produce a promotion-ready comparison. Existing metric fixtures are the prior art.
- Exercise recovery and resource boundaries through the existing training/GPU adapters with deterministic local fixtures. Check that interruption resumes unchanged science, changed configurations are rejected, allocation usage counts toward the campaign cap and unapproved or incompatible resources fail before training. Mock scheduler/network boundaries rather than launching real jobs from automated tests.
- Exercise the frozen offline adapter with complete and damaged-study fixtures. Assert exact row coverage, explicit fallback accounting, no report dependency, bounded study streaming and parity with the local shared inference path. Reuse existing archive-versus-expanded-layout and namespaced-input cases when delivery packaging changes.
- Compare speed-only changes on the same frozen candidate and input. Require identical predictions where equivalence is the claim; scientific changes require new evaluation and candidate identity. Measure full pipeline cost separately from software correctness tests.
- Real-data scientific validation is a separate acceptance layer: training-only diagnostics, three-fold matched comparisons, target support, paired uncertainty, measured resource consumption and actual candidate delivery evidence. Synthetic fixture success cannot satisfy a real MRI performance claim, and example notebook success cannot satisfy hidden scoring completion.
- Scope test execution to the implementation being changed and its affected integration path. Preserve existing safety and artifact contracts; do not add a broad test framework, perform a large architecture benchmark, or run training as part of writing this spec.
- The lifecycle seam is synthesized from the existing interfaces and agreed evidence policy. The to-spec skill requires a visible owner check that this seam matches expectations; this narrow check does not reopen the resolved campaign decisions.

## Out of Scope

- Guaranteed first place, fabricated scores, hidden-label leakage, disguised weak labels, or claims of clinical validity without supporting evidence.
- Paid compute, paid APIs, paid datasets, paid clinical review, new external spending or automatic expansion beyond the first campaign's agreed GPU budget.
- A new fixed candidate/submission-count cap; the owner explicitly rejected one. Official limits still apply.
- A broad rewrite of the pipeline, a new experiment platform, or a separate inference implementation for Kaggle.
- Unbounded architecture or per-target ensemble-weight searches, automatic full 3D/large-transformer training, or a large self-supervised-pretraining campaign. These require evidence and a separately scoped follow-up beyond the selected initial route.
- Treating all report-only studies as confidently labeled or fully image-ready, automatic use of existing learned outputs, or fresh remote processing of competition reports.
- Creating clinical labels, adjudicating uncertain diagnoses without a reviewer, or using reports at competition inference.
- Automatically downloading unreviewed weights, accepting binding rules, publishing competition data/code, deleting protected originals, or bypassing existing exact-job and delivery approvals.
- Moving completion to October 18 or October 20; those proposed dates were superseded by the owner's before-October-15 requirement.

## Further Notes

This is the implementation spec for the resolved campaign, published to the local tracker as `ready-for-agent`. It synthesizes existing owner decisions; it does not claim that new training, weight admission, GPU allocations or competition actions have occurred. Resource eligibility, actual storage headroom, throughput and future experiment outcomes remain facts to establish during execution, with the conditional routes and stop rules above.

The reported 0.498 score is historical completed public-scoring evidence. Current leading score, current owner rank, final private performance and complete current rules were not established by the planning research. The dated official material supports a nine-hour offline notebook envelope and an October 22 UTC competition deadline, both requiring refresh before dependent delivery actions. The owner's internal completion target is earlier.

The unresolved absence of a clinical reviewer and the 58-study reference cohort limit what can be concluded. No hard statistical threshold has been selected to manufacture certainty. A final campaign report must retain completed, inconclusive, failed and unrun branches, with exact candidate and resource provenance.

Decision and evidence references:

- [Validation and evidence requirements](issues/05-choose-validation-policy.md).
- [First bounded learning campaign](issues/06-choose-first-learning-campaign.md).
- [Final schedule and selection policy](issues/07-choose-delivery-and-selection-policy.md).
- [Compute and artifact inventory](../../docs/research/leaderboard-compute-inventory.md).
- [Image-learning research](../../docs/research/leaderboard-image-learning-options.md).
- [Supervision research](../../docs/research/leaderboard-supervision-options.md).
- [Competition research and access limits](../../docs/research/leaderboard-competition-target.md).
- [Corrected historical OOF evidence](../../docs/evidence/ticket-10-oof.md).
- [Recorded scored-candidate evidence](../../docs/evidence/ticket-24-candidate-selection.md).
- [Earlier mechanics specification](../rsna-2026-fresh/spec.md), for established interfaces and operational constraints; this campaign's resolved decisions govern its model route, budget, validation and schedule.
