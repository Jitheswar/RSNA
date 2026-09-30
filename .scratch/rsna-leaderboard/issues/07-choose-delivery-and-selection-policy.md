# Choose the candidate-selection and delivery schedule

Parent: ../map.md
Type: grilling
Labels: wayfinder:grilling
Mode: HITL
Status: resolved
Assignee: root
Blocked by: 01, 06

## Question

What dated checkpoints, stop-training margin, candidate selection rules, and real-scale inference evidence will preserve a valid competitive submission despite uncertain queue access? Derive deadlines and permitted final submissions from the current contract, account for the chosen first campaign and resource limits, and define what evidence later models or ensembles must supply. Preserve an executable fallback and record remaining risks rather than promising a rank.

## Comments

### Original delivery proposal — superseded by the owner's corrections below

The official [evaluation page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation), retrieved through official search indexing again on September 25, records October 22, 2026 at 23:59 UTC as the final deadline (October 23 at 05:29 IST), subject to organizer updates. Direct extraction still returned no body. The previous dated contract records two final selections, but the current complete rules/allowances remain to be refreshed before dependent actions.

- Proposed internal schedule: review diagnostics and resource readiness by September 28; first complete comparison target October 3; supervision/model expansion review October 8; stop scientific model changes and freeze candidate identities by October 14 at 23:59 IST; finish offline/runtime and Kaggle delivery checks by October 18; complete explicit final selections by October 20 at 23:59 IST. Missed milestones reduce scope rather than consume the delivery margin.
- Proposed shortlist: at most six new scientifically distinct scored candidates in this campaign, with no score-directed retries of the same model; failed infrastructure attempts are logged separately. Subject to the freshly verified official limit, select at most two final candidates from reproducible local evidence. Prefer the strongest validated single model and a complementary simple average only if complete OOF and runtime evidence justify it; otherwise select the strongest admissible alternatives. Keep the existing frozen submission as the operational fallback.
- Proposed ensemble/optimization policy: no per-target weight search or unbounded architecture sweep on 58 labels. A simple equal-weight combination needs paired OOF comparison and full pipeline timing. First reduce redundant decode/cache work without changing predictions; if runtime still misses the limit, reduce ensemble/slice cost as a new evaluated candidate or use the verified fallback. Apply the established total GPU budget; future performance-dependent research beyond it requires a new campaign decision.

This original proposal is retained as discussion history. Its October 18/20 dates and six-candidate cap were rejected by the owner; the resolution below governs execution planning.

## Answer

Resolved by live exchange on 2026-09-25. The owner required delivery **before October 15** and rejected a hard cap on scored candidates. Complete the internal delivery plan by **October 14, 2026 at 23:59 IST**, with no project-imposed numerical limit on candidate evaluations or scored submissions. Official submission and final-selection limits still apply. The separately accepted first-campaign budget remains 20 allocated GPU-hours; removing the candidate-count cap does not remove that compute budget or the zero-spend constraint.

### Internal schedule (all dates in IST)

| Deadline | Required outcome | If the milestone is missed |
| --- | --- | --- |
| September 28 | Initial learning diagnosis, source/resource review, storage ledger and exact first experiment proposal ready | Continue the CPU route and narrow the first comparison; do not infer pretrained-resource eligibility. |
| October 3 | Target first complete three-fold representation comparison | Keep incomplete evidence explicitly incomplete; prioritize finishing one defensible comparison over starting more variants. |
| October 8 | Review supervision benefit, candidate evidence and remaining measured compute/runtime cost | Stop branches that cannot finish with the remaining budget and delivery margin. |
| October 11, 23:59 | Freeze scientific configuration, checkpoint identity and candidate shortlist for delivery | Use the strongest completed eligible candidate; queue delays do not move the delivery deadline. |
| October 13, 23:59 | Complete offline packaging, representative full-pipeline checks and the intended Kaggle delivery verification | Repair operational defects using the frozen science; retain the already working submission as fallback. |
| October 14, 23:59 | Complete explicit owner final selection and preserve reproducible delivery evidence | Report the actual achieved state and any missing evidence. Do not label an unchecked candidate delivered. |

The October 11 modeling cutoff is the implementation of the owner's earlier delivery requirement, leaving three days for verification and selection. The official competition deadline is separate and remains subject to organizer updates. Recheck official rules and account eligibility/selection state before dependent competition actions; the internal target does not assert those checks have passed.

### Candidate and submission policy

- No fixed project cap on the number of model candidates or scored submissions. Each new candidate still needs a declared hypothesis, immutable configuration, complete applicable local evidence and a reason for using the remaining time. Log all attempts, including failed deliveries. A new random seed or public-score-driven retry is not evidence of a new scientific improvement.
- Follow the accepted validation policy: investigate local/public disagreements, prioritize reproducible local evidence, and preserve uncertainty. No unrestricted per-target ensemble-weight search on the 58 reference studies. The absence of a numerical cap does not make repeated CV an untouched test.
- Subject to current official final-selection allowances, plan up to two final choices. Prefer the strongest eligible validated single model and a complementary simple equal-weight ensemble only when paired OOF evidence and measured full inference cost support it. Otherwise choose the strongest admissible alternatives; do not manufacture a second distinct method merely to fill a slot.
- Tie-break close/inconclusive candidates using fold/target consistency, robustness, simpler execution and runtime headroom. Do not declare a statistically inconclusive improvement proven. Preserve the existing scored frozen candidate as an executable fallback even if its discrimination remains near chance.

### Delivery evidence and runtime decisions

- Freeze code/configuration, weight/input manifests, runtime dependencies, schema and provenance. Require MRI-only offline inference, exact expected study coverage, finite twelve-target scores, explicit fallback/decode accounting, and the applicable notebook runtime limit.
- Measure the complete pipeline, including input discovery, decoding, preprocessing, ensemble computation and CSV writing. A short example run or projection is not evidence of full hidden-test execution. Record representative workload size and long-tail behavior separately from actual Kaggle scoring completion and any hidden-run diagnostics that are exposed.
- First optimize repeated discovery/decoding/cache work while checking prediction equivalence. If runtime remains excessive, reduce model/ensemble/slice cost as a new evaluated candidate before the scientific freeze, or use the eligible verified fallback. Post-freeze fixes must preserve the scientific identity and undergo the affected delivery checks.
- Later ensemble, pretraining or larger-model work is conditional on genuine evidence and remaining budget/time. It is not an automatic campaign extension. Missing resource rights, storage headroom or GPU access invokes the agreed bounded/CPU/fallback paths rather than silently increasing resources.
- Exact GPU jobs, uploads, competition submissions and final account selections retain their existing concrete execution approval boundaries. This ticket resolves the schedule and selection policy; no such external action was performed while planning.
