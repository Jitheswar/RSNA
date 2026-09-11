# Knee MRI competition and research planning

Status: needs-info

## Confirmed user requirements

- College partnership aiming for a working competition model and publishable research.
- User delivery target: October 15 (2026 inferred from project context).
- Training can run continuously; implementation must accommodate limited coding-agent availability.
- User says competition data may leave the workstation; competition-specific permissions remain to be verified.
- External GPU access is unknown. Plan around free GPU resources; exclude paid compute until the user sets a budget.
- Additional learning methods and application technologies are conditional on value.
- Demo audience: competition judges, college reviewers, and researchers; competition data exclusively.
- User confirms unrestricted workstation use.
- User is the sole current human contributor; involvement depends on coding-agent availability. Faculty and radiology review are uncertain and likely unavailable initially.
- Prioritize a reproducible submission by October 15; research and paper writing may continue afterward.
- Publication venue and requirements are deferred: develop the project first, then write from its results.
- Experiments must run as bounded jobs with checkpoints, metrics, logs, and explicit runtime/storage limits. User review precedes the next experiment batch.
- No implementation until the requirements interview is complete.
- User delegates technical choices for validation, noisy supervision, method selection, and milestone ordering to the agent, prioritizing accuracy on unseen studies.
- First demo is operated locally for the user's supervisor. AWS hosting is a possible later phase contingent on supervisor feedback, account access, budget, and dataset permissions.
- User delegates demo prediction behavior: browse clearly identified precomputed predictions for selected permitted competition studies; provide separate live inference with measured runtime. Demo latency does not dictate competition model selection.
- Initial demo shows MRI slices and twelve abnormality probabilities. Localization and generated reports are excluded from the initial deliverable following delegated scope selection.

## Proposed architecture pending final planning review

- One Python ML core shared by training, competition inference, and FastAPI serving; React provides study viewing and prediction display.
- Defer Express and MongoDB until concrete application requirements justify them.
- Model selection prioritizes validated performance within verified competition runtime and available compute constraints. Demo serving may use offline predictions from that model.

## Methodology decisions

- Audit patient identifiers, duplicates, and label distributions before fixing versioned, patient-grouped validation folds. Choose fold count from audit evidence and available compute.
- For each evaluation fold, exclude its MRI and reports from all training, report-supervision fitting, pseudo-label generation, and distillation. Official validation labels are used for scoring; tuning must be tracked and distinguished from independent evaluation.
- Use cross-validation where feasible and report per-label performance and uncertainty. Do not select a favorable split or claim a best possible model.
- Treat report-derived labels as noisy, preserve source/provenance, and represent ambiguous or unmentioned findings as unknown rather than automatically negative.
- Compare additional methods against a common baseline on the same folds, track compute and failed experiments, and retain additions only when evidence justifies them.
- First milestone: audited DICOM ingestion through baseline training, validation, MRI-only inference, and valid submission generation. Web demo follows.

## Open decisions

- Actual free GPU availability and eventual college GPU access.
- Potential expert review and publication requirements remain deferred.
- Research hypothesis, final architecture review, and exact validation folds after data audit.
- Hosting details are deferred until after the local supervisor demo.

## Evidence awaiting full rules review

Official competition overview distinguishes October 15 entry/team deadline from October 22 final submission. Keep the user's October 15 target unless changed.
Source: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/efficiency-prize-evaluation

Read-only archive inspection found 4,407 training studies with reports, including 58 fully labeled studies and 4,349 entirely unlabeled studies. Only nine labeled MCL positives were found. This makes validation and report-derived supervision central planning questions; it does not establish a method choice.
