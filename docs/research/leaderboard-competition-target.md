# Current competition target and delivery constraints

Researched 2026-09-25 for **Establish the current competition target and delivery constraints**. Research branch: `research/rsna-competition-target`.

## Finding

The available official indexed material supports retaining the existing scoring, deadline and delivery plan. It does **not** establish today's leading score, the owner's current rank, or that the full rules have remained unchanged. Final private first place remains the destination; no numeric public score can guarantee it. This investigation inspected public pages and repository records only, without account mutations, training, or data/model downloads.

## Official material accessible in this investigation

The [official evaluation page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation), retrieved through search indexing on September 25 (index marked approximately two weeks old), states:

- Score: unweighted mean of the twelve target ROC AUCs.
- Output: one confidence per target per test study, beginning with `StudyInstanceUID`; filename `submission.csv`.
- Delivery: notebooks, internet disabled, at most nine hours on either CPU or GPU.
- Resources: freely public external data and pretrained models are allowed at overview level.
- Entry and team merger deadline: October 15, 2026; final submission: October 22; winners' training code, video and method description: November 5. Each deadline is 23:59 UTC unless amended.
- An efficiency prize is a separate track, combining runtime and accuracy; it does not replace the main accuracy objective.

The October 22 deadline corresponds to October 23, 05:29 IST. The [RSNA announcement](https://www.rsna.org/media/press/2026/2669) independently identifies the 2026 knee challenge, multilingual reports and international MRI training material. Its broad wording about images and reports does not specify the actual hidden-test input contract.

The [host's clinical overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343) was accessible through indexing, attributed to Competition Host Po-Hao “Howard” Chen. It describes twelve binary findings and says borderline cases were annotated negative. This reinforces the distinction between reference labels and report-derived labels; a report vocabulary match is not automatically the reference assessment.

## Dated repository evidence retained, not newly verified

The [September 11 delivery contract](../../.scratch/rsna-2026-fresh/delivery-contract.md) previously reviewed official indexed rules/data/host material. It records:

| Topic | Recorded contract | Current verification status |
| --- | --- | --- |
| Ranking | Final standing uses the private leaderboard; public scores use a public subset. | Full current rules body inaccessible. Retain this target, without claiming current private standings. |
| Submission limits | Five submissions daily; up to two final submissions; maximum team size five. | Not independently refreshed here. |
| Test inputs | Reports unavailable at test time; example test CSV, series descriptors and DICOMs are replaced for scoring; approximately 1,300 test studies. | Direct data and linked host clarification could not be recovered authoritatively here. Retain MRI-only inference. |
| External resources | Additional equal-access, cost/reasonableness and licensing provisions apply beyond the overview allowance. | Full rules inaccessible; no particular resource cleared by this research. |
| Main metric, dates, runtime | Twelve-target mean AUC; October deadlines; nine-hour offline CPU/GPU notebooks. | Consistent with official indexed evaluation retrieved today. |

The [recorded candidate-selection evidence](../evidence/ticket-24-candidate-selection.md) is the source for this project's completed submission and public score of 0.498. It is historical repository evidence, not a fresh authenticated leaderboard observation. That public score is neither a final private result nor proof of useful discrimination. The [corrected local evaluation](../evidence/ticket-10-oof.md) separately records reference-label OOF AUC 0.464313 on 58 studies. Those are different evaluation populations and must remain separately named.

## Access limitations and unresolved facts

Direct browser extraction of the official [evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation), [rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules), [data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data), and [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard) pages returned titles with zero content lines. Targeted searches recovered evaluation text, but not the full rules, direct data description, or leaderboard table. Search indexing is a dated view of an official source, not a live account or table snapshot.

Consequently, today's leading score, leading team, owner's rank, current public/private partition, account eligibility and final-selection state remain unverified. Participant discussion scores are not substituted for a live leaderboard. Search results also surfaced reports of host guidance on external LLM processing, but the referenced announcement was not recovered; participant interpretations are not permissions. Existing report-use restrictions remain in force until that question is resolved from authoritative text.

## Implications for the decision map

1. Optimize reproducible reference-label discrimination toward the final private ranking. Do not use a guessed leading score as an acceptance threshold.
2. Budget experiments backward from October 22, with enough time for hidden-test execution and candidate selection; treat the date as subject to organizer updates.
3. Preserve MRI-only offline inference and full-run timing. Successful execution on example studies establishes mechanics only; hidden scoring must be evidenced separately.
4. Review each proposed free checkpoint, external dataset and report-label source against exact provenance, license and current rules before its use. Public availability alone does not settle eligibility.
5. Recheck current rules, submission allowances and the actual leaderboard through an accessible official account view before delivery decisions depend on them. The present access gap does not prevent local planning or justify changing the established contract.

These are planning implications drawn from the sources above, not newly executed experiments or a claim that first place is achievable within the available queue time.
