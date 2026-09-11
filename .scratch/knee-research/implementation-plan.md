# Knee MRI implementation plan

Status: proposed — implementation paused pending shared-understanding review

## Outcome and priorities

Deliver a reproducible MRI-only competition entry by October 15, 2026, followed by a local supervisor demo and an evidence-based research write-up. Publication acceptance and leaderboard rank are not promised. Accuracy on unseen studies takes priority over extra application features.

## Boundaries and data flow

Preserve the source archive. Build a versioned manifest linking study, series, image instances, official targets, and training reports. Separate official labels from report-derived targets. Data audit establishes which identifiers can support patient grouping; do not assume study IDs are patient IDs.

Training path: permitted source data → audited manifest → frozen split assignments → fold-local image preprocessing and report supervision → training → out-of-fold predictions and metrics → selected model artifacts.

Competition path: test DICOM → the same versioned image preprocessing → MRI-only model → probabilities in the official submission schema. No report parser or report dependency in inference. Validate identifier alignment, label order, finite probabilities, row count, and runtime.

Demo path: selected permitted competition studies → local viewer and saved predictions, with an optional asynchronous live inference job. Display whether predictions were precomputed, model version, and measured live runtime. No generated reports or unvalidated localization.

## Data quality and validation

- Audit archive inventory and available disk before extraction; retain source files and record hashes without making unnecessary full dataset copies.
- Inspect transfer syntaxes and decoding support, orientation, image position, spacing, series descriptions, sequence coverage, duplicates, missing/corrupt data, and unusually large studies. Do not order slices by filename. Handle unsupported geometry and missing series explicitly.
- Audit possible patient overlap and duplicate examinations across labeled/unlabeled data. If patient linkage cannot be established, disclose that limitation and use verified study/duplicate grouping.
- Choose a feasible small fold count after checking grouped positive/negative support for every label. Freeze assignments and seeds. A label with no positive or negative cases has undefined fold AUC; report it explicitly rather than silently averaging it away.
- Record official metric, per-label results, pooled out-of-fold predictions, uncertainty, and fold variability. Cross-validation used repeatedly for tuning is not independent confirmation; document the selection process and budget.
- For each fold, keep validation MRI and reports out of all learning stages, including report-target development and pseudo-label generation. Fit learned preprocessing on training data only.

## Experiments and research

First establish a valid small baseline and executable submission. Then evaluate report-derived weak supervision, with positive/negative/unknown states and provenance. Negation, uncertainty, historical findings, and label mapping need explicit handling; unmapped language stays unknown.

Candidate research question: under scarce official labels, does uncertainty-aware report supervision improve MRI-only classification compared with official-label-only training? This is a proposed question, not a novelty claim. Review prior work before making publication claims.

Keep the baseline and report-supervision comparison on identical folds. Consider pseudo-labeling next only if performance and compute warrant it; use fold-local teachers. Distillation is conditional on a stronger teacher and a concrete inference-cost benefit. Ensembling must fit the competition runtime. Do not spend all free GPU time tuning one favorable split.

Each run records code revision, environment, data/split version, configuration, seeds, supervision source, metrics, runtime, hardware, checkpoint, and failure state. Keep runs and artifacts locally first with a simple exportable index; defer a hosted tracking service.

## Software and execution

Use one Python ML core for data interpretation, preprocessing, model loading, and inference, called by training commands, the Kaggle adapter, and FastAPI. React handles the demo viewer. Defer Express and MongoDB; store demo manifests and experiment metadata as local artifacts initially.

Use a pinned project Python environment installed in the user account and a committed dependency lock. Maintain explicit CPU and GPU setup paths; choose GPU packages only after detecting the actual remote NVIDIA driver and testing compatibility. A CUDA toolkit alone does not provide a GPU. Do not require system package installation or Docker.

Run training through independently launched bounded jobs with checkpoints, restart instructions, logs, explicit CPU/RAM/disk budgets, and human review before the next batch. Benchmark preprocessing workers rather than assuming all 128 threads improve throughput. Free GPU session termination requires durable checkpoint export and resumable jobs.

Test boundaries that can silently corrupt results: study/label alignment, slice ordering and geometry, split isolation, report uncertainty mapping, missing-series behavior, shared inference parity, and submission schema. Use a tiny end-to-end fixture before a full run. No app implementation is part of this planning phase.

## Milestones and gates

1. Access and data audit: verify rules, joining requirements, permitted compute/data transfer, archive schema, decoding feasibility, and actual free GPU access. Freeze feasible scope based on a measured accelerator smoke test.
2. End-to-end baseline: complete a small training run, generate validation predictions, and execute a valid offline submission. Estimate full inference cost.
3. Main experiments: compare official-label-only and report-supervised training on frozen folds, then conditionally add one method at a time. Preserve failed and negative findings.
4. Submission freeze by October 15: select artifacts, validate a clean offline run on the permitted Kaggle runtime, and preserve a reproducible fallback submission. Any later competition deadline does not move the internal target automatically.
5. Supervisor demo: local MRI viewer and clearly identified saved predictions; benchmark optional live CPU inference. AWS deployment waits for access, budget, and verified data-display permissions.
6. Research write-up: document methods, ablations, limitations, uncertainty, and compute. Revisit venue and expert review after results; additional confirmation experiments may be required.

## Unresolved external dependencies

- Free GPU allocation and account eligibility; college GPU access remains unknown. CPU-only work can prepare data and validate plumbing, but competitive training feasibility is not established.
- Full competition rules and data/demo permissions must be verified before restricted actions.
- Faculty/radiology review and publication requirements remain deferred by the user.
- Exact fold count, image resolution, architecture, and experiment count depend on data audit and measured GPU throughput, not guessed hardware capacity.
