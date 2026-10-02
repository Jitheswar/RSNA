# Official requirements and public standing — October 2, 2026

Observed through authenticated Kaggle API reads and anonymous rendering of official Kaggle pages on **2026-10-02 at approximately 12:55 UTC / 18:25 IST**. This refresh establishes current requirements and public standing; it does not approve a new configuration, notebook launch, or submission.

## Public standing and exact scored identity

The authenticated public leaderboard download, generated **12:55:00 UTC**, contains **4,862 teams**. **Elyra**, team 16847991 containing `jitheswar`, is **public rank 2,042 with displayed score 0.936**. **rhoskeri**, team 16687899, is public rank 1 with **0.961**, a displayed score gap of **0.025**. The rendered leaderboard confirms approximately **30% public / 70% private** test allocation; final standings can differ. These observations establish no private rank. [Official public leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard).

The account's best completed scored submission remains **56705536**, attributed ensemble baseline version 2, linked to **script 354125287**, public score **0.936**, `privateScore=null`. Submission **56739912**, frontier script **354194017**, remains an unhandled hidden-rerun error with no public score. Its completed status is not successful scoring evidence. [Exact scored baseline](https://www.kaggle.com/code/jitheswar/rsna-public-ensemble-gpu-preflight-attributed?scriptVersionId=354125287), [failed frontier version](https://www.kaggle.com/code/jitheswar/rsna-public-frontier-gpu-preflight-attributed?scriptVersionId=354194017). These identities and statuses are preserved from the authenticated competition submissions API, rather than inferred from notebook titles.

## Timeline and internal freeze

The fresh official timeline remains:

| Milestone | Official UTC | IST equivalent |
| --- | --- | --- |
| Entry and team merger | October 15, 2026, 23:59 | October 16, 05:29 |
| Final submission | October 22, 2026, 23:59 | October 23, 05:29 |
| Winners' requirements | November 5, 2026, 23:59 | November 6, 05:29 |

The host may update this schedule. [Official timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline).

The project's **October 11, 23:59 IST science freeze** is an earlier internal milestone, followed by October 13 delivery checks and October 14 final selection. The spec explicitly preserves the owner's before-October-15 target. The later official deadline does not move those project commitments. [Internal delivery schedule](../../.scratch/rsna-leaderboard/spec.md).

## Runtime, submissions, and external resources

Competition submission uses committed notebooks: **CPU or GPU runtime at most nine hours, internet disabled, output named `submission.csv`**. The evaluation remains macro ROC AUC across twelve targets. These are submission-runtime conditions; they do not establish a nine-hour limit on local development training. [Official evaluation and code requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation).

Competition-specific rules section 2.2 allows **five submissions per day and two final selections**. The authenticated limits endpoint at this observation reports `numToday=0`, `numTotal=4`, `numAllowedNow=5`, `limitedByTotal=false`; this is a changing account observation, not additional authorization. Section 2.6 permits external data/models subject to equal accessibility and the host's reasonableness assessment. Section 2.8 requires winners to deliver training/inference code, weights, documentation, and environment; weights must be a public Kaggle dataset linked to inference. [Official rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules).

For OAI specifically, the host conditions competition eligibility on free, general access without escalated institutional review/sign-off or access to a legal team. The host separately declines to resolve OAI stewards' data-use terms. This is a conditional host statement, not blanket OAI permission; participant replies are not host rulings. Downloadable checkpoints, rehosts, or medical pretraining claims alone do not establish compliance with underlying resource terms. [Official host clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819).

## Evidence and scope

Ignored raw evidence is under `artifacts/private/competition-requirements-refresh-20261002T125421Z/`: authenticated leaderboard ZIP/CSV and summary, authenticated own submissions and live limits, rendered rules/timeline/evaluation/host clarification/leaderboard, browser response index, and SHA-256 manifest. The API CSV is authoritative for the quoted team count and own rank; the overview participation widget was briefly one team behind. Browser metadata responses were bounded below 4.8 MB. No competition data, model weights, fitting, launch, submission, selection, quota mutation, ledger edit, or external communication occurred.
