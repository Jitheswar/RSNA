# RSNA Knee Abnormality Detection 2026 delivery contract

**Ticket:** 01 — Verify the competition delivery contract
**Parent spec:** `.scratch/rsna-2026-fresh/spec.md`
**Review date:** 2026-09-11, Asia/Kolkata (official pages and the linked MIRA PDF reviewed on this date)
**Status:** dated research record; competition entry, rules acceptance, eligibility determination, upload, submission, and publication were not performed.

This record separates facts stated by official competition or host sources from source gaps, owner decisions, and engineering inferences. It supplies V12 evidence for reconciliation, but V12 is not complete as an operational delivery gate until the owner-confirmation and resource/publication gates below are resolved. It is not an acceptance of the competition rules on the owner's behalf.

## Source register and access evidence

| ID | Official source | Access/review evidence on 2026-09-11 | Scope used |
| --- | --- | --- | --- |
| K-EVAL | [Kaggle overview/evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation) | The direct page returned Kaggle's client-rendered HTML shell to the browser extractor, but its official indexed rendering was available and was reviewed on 2026-09-11. A direct HTTPS GET also returned the page shell. | Metric, exact submission header, timeline, notebook and external-data requirements. |
| K-DATA | [Kaggle data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data) | Same client-rendering limitation; the official indexed rendering was reviewed on 2026-09-11. | File schemas, `StudyInstanceUID`/folder mapping, test inputs, report absence, DICOM notes. |
| K-RULES | [Kaggle competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) | Same client-rendering limitation; the official indexed rendering exposed the current rules sections used below. | Entry and eligibility, teams, submission limits, data security, code sharing, external resources, winner license and obligations. |
| K-HOST-LABELS | [Kaggle host overview and label description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343) | Official post by Po-Hao “Howard” Chen, marked Competition Host; indexed content reviewed on 2026-09-11. | Host-authored clinical label descriptions and annotation scope. |
| K-HOST-TEST | [Kaggle host confirmation that reports are unavailable at hidden-test time](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734118) | Official host response in the competition discussion; indexed content reviewed on 2026-09-11. | Confirms the inference-time report restriction stated on K-DATA. |
| RSNA-HOST | [RSNA Knee MRI AI Challenge page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge) and [RSNA launch announcement](https://www.rsna.org/media/press/2026/2669) | Both official RSNA pages were fetched on 2026-09-11; the press page was readable through line 247. | Host identity, challenge scope, broad public timeline context, and host publication intent. |
| MIRA | [RSNA MIRA license link](http://rsna.org/mira-license) and its [resolved RSNA PDF](https://www.rsna.org/-/media/files/rsna/RSNAI/RSNA-MIRA-Dataset-License-2025.pdf) | The short link redirected to the official PDF. The PDF was fetched directly and read on 2026-09-11; it identifies itself as **V2.0, 2025-11-05**. The browser safety layer could not open the short link directly, so the resolved PDF is the evidence link. | Research-use, copying, distribution, publication, re-identification, and non-clinical-use terms. |

The Kaggle pages' client-rendering limitation does not make their official indexed content an unofficial summary. It does mean the next delivery gate must re-open the live pages, because K-EVAL says the organizers may update the timeline.

## Verified competition contract

### Target vector and order

K-EVAL gives the exact submission header. The twelve target names and order are:

1. `ACL`
2. `MCL`
3. `Medial Meniscus`
4. `Lateral Meniscus`
5. `Medial OA`
6. `Lateral OA`
7. `PF OA`
8. `Effusion`
9. `Synovitis`
10. `Baker's`
11. `Contusion`
12. `Fracture`

The exact header, including spelling, capitalization, spaces, and the apostrophe, is:

```text
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
```

K-DATA describes the same twelve columns as binary labels and describes the task as predicting the per-study probability of each finding. K-HOST-LABELS is the official host's clinical description of the labels: ACL is high-grade partial or full tear; MCL is high-grade acute partial or complete tear; the meniscus labels require definite surface-reaching signal or specified morphology; each OA label is compartment-specific high-grade cartilage loss; effusion is moderate or large joint fluid; synovitis is synovial inflammation/thickening; Baker's is a moderate or large popliteal cyst; contusion is bone-marrow edema-like injury without a discrete fracture line; and fracture is an acute cortical break or fracture line. Borderline findings were graded negative by the annotators. The labels are whole-examination labels for a single knee, with two subspecialist readers and third-reader adjudication described by the host.

These clinical definitions support later report-supervision design, but they do not authorize inventing report rules. Any ambiguous report interpretation remains an unresolved V11/V15 issue in the parent spec.

### Study and series identity mapping

The authoritative mapping in K-DATA is:

| Record | Official mapping |
| --- | --- |
| Training study | One `train.csv` row per training study. `StudyInstanceUID` is the study identifier and matches the folder name under `train_series/`. |
| Training series | `train_series.csv` links each `SeriesInstanceUID` to its parent `StudyInstanceUID`; the series identifier matches the child folder under `train_series/<StudyInstanceUID>/`. |
| Training image | A DICOM slice is stored under `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`. |
| Test study | `test.csv` supplies the test `StudyInstanceUID` values; the example file is replaced by the hidden test studies during scoring. |
| Test series | `test_series.csv` has the series-descriptor schema for the test studies, and `test_series/` has the corresponding DICOM layout. |
| Submission row | K-EVAL requires one prediction row for each row in the test set, with `StudyInstanceUID` as the first field. |

The prediction unit is therefore the study, not a series or slice. The exact linkage from the source test UID to the first submission column is verified. Preserving the expected UID set exactly once, rejecting extras or omissions, and retaining test-file order are implementation safeguards inferred from “for each row in the test set” and the UID-to-folder mapping; K-EVAL does not separately state a CSV row-order rule in the reviewed text.

K-DATA explicitly says that the `Report` field is not provided at testing, and K-HOST-TEST confirms that reports are unavailable for the hidden test set. RSNA's broader press wording describes images and reports as inputs used to “train and test,” but the specific Kaggle input contract and host clarification resolve the delivery question for this competition: reports can support training or research paths only; they cannot be an inference input for the competition test set.

### Metric

K-EVAL defines the main score as:

```text
Final Score = (1 / 12) * sum(AUC_i for i = 0..11)
```

It names this the macro-averaged ROC AUC: the average area under the ROC curve between predicted confidence scores and observed targets over the twelve targets. The parent spec's “macro ROC-AUC” wording agrees with this official definition. The Kaggle rules state that final standing is determined by the Private Leaderboard, while the Public Leaderboard is based on a representative public portion of the test data.

### Submission file

The official schema is:

```text
submission.csv
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
<uid_1>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
<uid_2>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
```

Verified requirements:

- The file must be named `submission.csv`.
- It has a header and one confidence score for each of the twelve targets for every test row.
- `StudyInstanceUID` is first, followed by the exact twelve target columns above.
- `sample_submission.csv` is an official template whose example scores are `0.5`.

The reviewed official text calls the values probabilities or confidence scores but does not explicitly state a numeric validator bound or explicitly prohibit an extra CSV index. The parent spec's finite scores in `[0, 1]`, exactly thirteen columns, no index column, no duplicate expected UID, and no missing/extra UID are retained as local validation invariants inferred from this schema, not misreported as additional Kaggle quotations.

### Notebook and runtime constraints

K-EVAL states that submissions must be made through Notebooks. For the Submit action to become available after a commit, the official conditions are:

- CPU Notebook runtime `<= 9 hours`.
- GPU Notebook runtime `<= 9 hours`.
- Internet access disabled.
- Freely and publicly available external data is allowed, including pretrained models.
- The submission file is named `submission.csv`.

The nine-hour limit corresponds to 32,400 seconds; that exact denominator also appears in the official efficiency-score formula. The page states a notebook runtime cap but does not enumerate its phases, so the parent spec's requirement to measure startup, setup, loading, decoding, preprocessing, inference, ensembling, and output generation is an engineering interpretation of the total notebook runtime constraint.

K-DATA lists the official competition inputs:

- training `train.csv`, including `Report` and the twelve reference-label columns where provided;
- training `train_series.csv` and `train_series/` DICOM;
- test `test.csv`, `test_series.csv`, and `test_series/` DICOM;
- `sample_submission.csv`.

K-DATA also records variable MRI intensities, orientations, resolutions, several DICOM transfer syntaxes, and an allowlist of 86 DICOM metadata tags. That is an input-format warning and inventory scope, not evidence that every decoder or geometry policy has passed the parent spec's V03–V06 gates.

External data and models are permitted only at the high level stated above. K-RULES adds that external data must be publicly available and equally accessible to all participants at no cost, or satisfy the host's reasonableness criteria; the host may assess excessive cost, access restrictions, and software/tool licensing. A specific model, dataset, API, or pretrained checkpoint therefore remains a V13 review item.

## Official dates and participation limits

K-EVAL states that the dates below are at **11:59 PM UTC** unless otherwise noted and that organizers may update the timeline. IST equivalents are shown only for planning and are not a substitute for the official UTC deadline.

| Event | Official deadline/date | IST planning equivalent | Evidence/status |
| --- | --- | --- | --- |
| Start date | July 30, 2026 | Date only; source does not state a start time | Verified on K-EVAL. |
| Entry deadline | October 15, 2026, 23:59 UTC | October 16, 2026, 05:29 IST | Verified; rules must be accepted before this deadline. |
| Team merger deadline | October 15, 2026, 23:59 UTC | October 16, 2026, 05:29 IST | Verified; last day to join or merge teams. |
| Final submission deadline | October 22, 2026, 23:59 UTC | October 23, 2026, 05:29 IST | Verified. RSNA's press release also says the competition runs through October 22. |
| Winners' requirement deadline | November 5, 2026, 23:59 UTC | November 6, 2026, 05:29 IST | Verified; winners submit training code, video, and method description to the host/Kaggle. |
| Parent-spec internal readiness target | October 14, 2026, 23:59 IST | Same | Local project decision, not an official Kaggle deadline. |

K-RULES and the official evaluation page give these limits:

- Entry requires a registered Kaggle account, agreement to the rules, and one unique account; multiple accounts are disallowed.
- A participant may join or form only one team. Maximum team size is five.
- A team merger is allowed through the team leader, subject to the merger deadline, maximum team size, and a combined submission count no greater than the number allowed for one team at that point.
- A team may submit at most five submissions per day.
- A team may select up to two Final Submissions for judging. Kaggle may automatically select a final submission if the user does not select one, but the parent project requires an explicit owner selection before calling anything final.
- The final ranking is based on the Private Leaderboard, subject to rule compliance; the public score is not final evidence.

No account, team, rule acceptance, or final-selection state was inspected or changed during this research.

## Data, publication, and resource rules

### Competition data and MIRA terms

K-RULES labels data access and use as “Commercial and Academic Research — MIRA license,” and says Competition Use and Commercial are checked. It also incorporates the MIRA license and requires reasonable measures to prevent people who have not formally agreed to the rules from accessing Competition Data. It prohibits transmitting, duplicating, publishing, redistributing, or otherwise making Competition Data available to a nonparticipant, and requires immediate notice to Kaggle of possible unauthorized access.

The linked RSNA MIRA agreement (V2.0, 2025-11-05) separately says that MIRA datasets may be viewed and used without charge for research, including research to produce or manufacture products for sale, while other commercial use, sale, or monetization is prohibited. It also prohibits distributing, publishing, or reproducing any portion of a MIRA dataset to others without prior written permission; prohibits sharing download links; prohibits modification, reverse engineering, derivative works, and re-identification; and limits use to non-clinical research use.

These are both official sources. Their interaction is not resolved here: the Kaggle rules' “Commercial and Academic Research” wording and the MIRA agreement's restrictions must be reconciled for the owner's actual use, publication, and any external transfer. The stricter handling is the safe project default until the owner obtains a written host/legal answer. This artifact contains no competition images, reports, raw identifiers, or private manifests.

### Code, open-source, and winner obligations

K-RULES states that private sharing of Competition Code outside a team is prohibited during the competition. If a participant publicly shares Competition Code, the sharing must be on the competition's Kaggle forum or associated notebooks, must respect third-party rights, and is treated under the rule's open-source terms. Code used to generate a submission is required to use an OSI-approved license that does not limit commercial use unless the specific rules say otherwise.

For a winner, the current rules state all of the following:

- Winner license type: `CC-BY-NC 4.0`.
- The winning submission and source code are to be licensed as specified in the winner-license section, with exceptions for generally commercially available software and incompatible-license input data or pretrained models.
- The winner must provide the final training code, inference code, model weights, documentation, and the computational environment or reproducible requirements.
- Model weights must be provided as a public Kaggle dataset linked to the inference/submission code.
- A detailed reproducible method description may be required, and the host may request a recorded call or panel discussion.
- K-EVAL adds the November 5 winners' requirement deadline for training code, video, and method description.

There is an unresolved wording conflict inside the official winner section: it names `CC-BY-NC 4.0` and also describes that license in language about OSI approval and no commercial-use limitation. This record does not resolve license semantics or decide whether a proposed pretrained resource is compatible. Any intended resource or publication path needs an exact license review and, where necessary, host confirmation before use.

### Eligibility and owner gates

K-RULES states that an eligible entrant must be a registered Kaggle account holder, be at least 18 or the age of majority in the jurisdiction, not reside in the listed restricted territories, and not be subject to applicable U.S. export controls or sanctions. An entrant representing an employer, educational institution, or other entity must have that entity's knowledge and consent and must comply with its policies. The sponsor may verify eligibility. Competition-entity personnel may participate under the stated rules but are not prize-eligible.

The owner's facts, account status, jurisdiction, age/majority status, employer or institution consent, sanctions status, team status, and rule-acceptance state are not known from this research. The owner must confirm them directly before entry or delivery. The agent must not accept binding rules, make an eligibility representation, create or merge a team, select a final submission, upload a notebook/data/model, or submit on the owner's behalf.

## Disposition by evidence type

### Verified facts

- Official target header and target order.
- Study/series/DICOM path mapping through `StudyInstanceUID`, `SeriesInstanceUID`, and `SOPInstanceUID`.
- Study-level output and report absence at hidden-test inference.
- Macro-averaged ROC-AUC formula over twelve targets.
- `submission.csv` filename and the twelve-column prediction layout.
- Notebook-only delivery, offline execution, separate CPU/GPU nine-hour caps, and high-level allowance for freely public external data/pretrained models.
- Official start, entry, team-merger, final-submission, and winners' requirement dates, including UTC convention.
- Maximum team size five, five submissions per day, and up to two final selections.
- Official first-party label descriptions, competition data security terms, external-data reasonableness rule, code-sharing rule, winner obligations, and the linked MIRA agreement's stated restrictions.

### Inaccessible or unresolved rules

- The Kaggle pages are client-rendered in this environment; the direct browser extraction returned no body lines even though official indexed renderings exposed the text. Recheck the live pages before any dependent action.
- The MIRA short link was not directly openable through the browser safety layer, but its HTTP redirect and official PDF were fetched and reviewed. If the redirect or PDF changes, repeat the license review.
- The official sources do not settle the interaction between the Kaggle “Commercial and Academic Research” flag and the MIRA Research Use Agreement.
- The official winner section's CC-BY-NC/open-source/commercial-use wording needs host or legal clarification for any winning bundle and pretrained resource.
- The exact hidden test UID set, row order, and final study count are not public; K-DATA says the example is replaced at scoring and gives only an approximate count.
- The reviewed pages do not state an explicit numeric score range, finite-value validator, or CSV-order requirement. Those remain local validation requirements and should not be presented as additional official rules.
- The current live timeline can change; the parent spec's internal readiness date is a project guardrail, not a competition rule.

### Owner-confirmation gates

Before competition delivery, owner confirmation is required for:

1. The owner has personally accepted the current Kaggle rules before the entry deadline and confirms the account/team/eligibility facts required by K-RULES.
2. Any employer, educational-institution, team, or legal-entity consent needed for participation and prize eligibility.
3. The exact external data, pretrained model, tool, version, license, access cost, public availability, and reasonableness under K-RULES/V13.
4. Whether the intended use is limited to competition/research and whether any publication, public Kaggle dataset, model-weight release, or demonstration has rights compatible with both K-RULES and MIRA.
5. The exact frozen notebook/model/data bundle to upload, if upload is later authorized.
6. The final submission choice, up to the allowed two selections, if a submission is later authorized.

## Reconciliation and bottleneck plan

The parent spec says that V12 blocks competition uploads/submissions or resource/publication actions when eligibility or rules are unverified, while V08, V13, V16, and V17 govern the exact bundle, external resources, runtime, and complete output. Reconcile the sources in this order:

| Bottleneck | Reconciliation action | Consequence |
| --- | --- | --- |
| Contract freshness | Before each dependent stage, re-open K-EVAL, K-DATA, K-RULES, K-HOST-TEST, and the MIRA link; record access date and any changed text. | A changed deadline, schema, input rule, or runtime rule pauses the affected ticket and requires a parent-spec/dependent-ticket reconciliation. No unofficial summary fills a gap. |
| Owner authority and eligibility | Obtain the owner's direct confirmation of rule acceptance, account eligibility, team/entity consent, and final-selection authority. | Until confirmed, local synthetic or gold-only mechanics work may continue within its own authorization, but entry, upload, submission, team, and publication actions remain blocked. |
| External resource rights | For every external dataset/model/tool, record exact version/hash, license, cost/access, public availability, intended use, and winner/publication compatibility. | Unresolved resources stay out of serious experiments and delivery; use competition data only or a separately approved clean comparator. |
| Delivery bundle | Validate the exact frozen notebook and model against the exact header, expected UID mapping, offline runtime, resource inputs, and failure accounting. | A local pass is only local evidence. Kaggle-verified status requires the exact approved bundle to complete in Kaggle; preserve the known-good candidate. |
| Deadline and selection safety | Recheck UTC dates immediately before the October 15 entry/team gate and October 22 final gate. Keep the parent target of October 14 23:59 IST as readiness margin. | Do not let a deadline approach convert an unresolved owner or license gate into an implicit approval. Final status remains owner-selected. |

The practical path is to finish local contract and offline packaging evidence before the October 14 internal target, then use the remaining time only for owner-approved eligibility, delivery, and optimization decisions. If any source changes, append a new dated evidence pass and reconcile the parent spec before dependent work; do not silently edit an existing candidate's contract.

### Parent-spec reconciliation snapshot

| Parent-spec decision/gate | 2026-09-11 evidence | Reconciliation result |
| --- | --- | --- |
| Exact target/header and macro ROC-AUC (Domain and competition contract; V08) | K-EVAL and K-DATA agree with the twelve names/order and the `1/12` macro ROC-AUC formula. | Confirmed; retain the parent schema and local validator safeguards. |
| `StudyInstanceUID` study mapping and MRI-only test path (V08/V12) | K-DATA maps study and series folders; K-DATA and K-HOST-TEST say hidden-test reports are absent. | Confirmed; report extraction stays out of competition inference. |
| Nine-hour offline notebook (V08/V16) | K-EVAL requires notebook delivery, CPU/GPU `<= 9 hours`, and no internet. | Confirmed; phase-by-phase timing remains a parent engineering check. |
| Entry/final dates and internal October 14 readiness margin (V12) | K-EVAL gives October 15 entry/team merger and October 22 final submission at 23:59 UTC; the October 14 IST date is local. | Confirmed with an added official November 5 winners' requirement deadline; recheck the live timeline before action. |
| Permitted external resources (V12/V13) | K-EVAL allows freely public external data/pretrained models; K-RULES adds equal-access/minimal-cost/reasonableness and licensing conditions. | Parent gate remains open until each specific resource is reviewed. |
| Publication, data security, and winner delivery (V12/V13) | K-RULES and MIRA provide restrictions and the unresolved commercial/license wording described above. | Source facts are recorded; V12/V13 remain open and publication/resource release is blocked until owner and, where needed, host clarification. |

## Research boundary

This artifact records public contract evidence only. It does not accept rules, assert that the owner is eligible, create a team, submit or upload anything, expose competition data, modify code, or modify the ticket or parent spec.
