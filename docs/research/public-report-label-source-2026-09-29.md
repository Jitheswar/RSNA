# Public report-label source audit — 2026-09-29

## Source and exact artifact

The authenticated [Kaggle dataset](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels) identifies `stevenleehans/rsna-knee-llm-report-labels` as public and lists a CC0-1.0 license. Its `llm_labels_v2.csv` file was downloaded for a local, read-only audit to `artifacts/private/public-label-audit/llm_labels_v2.csv`: 548,231 bytes, SHA-256 `3c082a987939528a55202028f655bd6f9dc5c9e28eff5058bbae7ef088a9f144`. No labels were admitted to training or copied into a candidate.

The dataset card says an LLM read the competition's `Report` column and produced probabilities. It does not identify the exact LLM, inference host, prompt, execution environment, or raw-report transfer policy for v2. The separate v4 blend explicitly includes another public GPT-5.6-Sol label table; v4 was not downloaded or audited. The card reports using the 58 gold studies to evaluate and adjust its v2 labels, including a Synovitis rule. Thus performance on those same 58 studies is development evidence, not an independent validation result.

## Local data checks

The v2 CSV has exactly 4,407 unique study rows, `StudyInstanceUID` plus the twelve expected target columns, all 58 corrected gold-fold studies, and no missing, nonfinite, or out-of-range target values. Exactly 19.23% of its 52,884 target cells equal `0.5`, which the dataset card describes as “not addressed.” Recomputing twelve-target macro ROC AUC against the corrected gold labels gives `0.8873495`, matching the card's rounded `0.8873`. Per-target AUC ranges from `0.7903` for Synovitis to `0.9933` for ACL. These checks establish file consistency and label correlation on the 58 reused studies; they do not establish image-model gain, independent clinical accuracy, or rule compliance.

## Admission decision

**Research only; not admitted to training or delivery.** The [competition overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview) allows freely public external data, but the recorded [full-rule review](../../.scratch/rsna-2026-fresh/delivery-contract.md) adds equal-access and data-security obligations. The card's CC0 license and public download do not by themselves establish whether the derived competition labels were produced and shared consistently with those obligations, or whether the particular label-generation service received competition reports. Exact provenance and a current rules interpretation remain open. The local dataset stays private while that review is unresolved.

If later admitted, evaluate it as a separately identified label-source branch with fold isolation, reference precedence, `0.5` masked as unaddressed, matched reference exposure, and a complete three-fold image-model comparison. Do not use its gold-58 label AUC as proof of a training benefit; the source was developed against that same set.
