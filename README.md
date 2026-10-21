# RSNA 2026 Knee Abnormality Detection: team Elyra

Our solution for the [RSNA 2026 Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
Kaggle competition: detect 12 findings on knee MRI (ACL, MCL, medial / lateral meniscus, medial / lateral / patellofemoral
OA, effusion, synovitis, Baker's cyst, contusion, fracture). The metric is macro ROC AUC.

## The problem

- 4,407 training studies come with free-text radiology reports in about nine languages, but only **58** have
  radiologist reference labels. The hidden test set is labelled by the same radiologist panel.
- Reports are not available at test time, so the model has to learn from the images alone.
- Reports and the panel often disagree. The panel is more liberal than the written definitions, many reports never
  mention a finding at all (synovitis is unaddressed in 62-92% of reports), and DICOM orientation and laterality
  vary between sites.

## How it works

1. **Clean volumes.** Every series is put into a canonical orientation, mirrored to a common knee side, and checked
   by a small laterality classifier (`scripts/v2/prep.py`, `laterality_clf.py`).
2. **Labels from reports.** LLMs grade each finding from the report. The grades are then calibrated against the 58
   reference studies, so they match how the panel reads (`llm_label.py`, `build_labels.py`, `xcal_labels.py`).
3. **Labels from images + reports.** A vision-language model reads three MRI montages together with the report
   (`mm_label.py`, `mm_local.py`). This moves the labels from "what the report says" towards "what the panel sees".
4. **Self-training.** Students train on a mix of the labels and out-of-fold image predictions (`make_students.sh`).
5. **Distillation from a public model.** The public R6 model was run on all training scans
   (`scripts/v2/r6_distill/`). Its predictions were rank-mapped onto our label distribution and mixed in at 0.4 / 0.6
   (`r6_labels.py`). On the 58 reference studies, label AUC rose from 0.941 to 0.950.
6. **Model.** A multi-series 2.5D network (`model.py`). A shared ConvNeXt-S encoder turns slices from up to 6
   series into tokens, a small transformer combines them across series, and one query per finding reads the result.
   Image size is 320, trained with 5-fold CV plus full-data seeds (`train.py`).
7. **Blend.** Our 18 models are rank-blended at 1/3 with two attributed public pipelines (Raptor CoAtNet, R6 with
   flip TTA). The weights were fixed before any submission (`notebooks/v2/build.py`, `leg.py`).

## Results (public leaderboard)

| Submission | Public AUC |
|---|---|
| Public baseline we started from | 0.943 |
| Our models alone (report + multimodal labels) | 0.948 |
| Our models alone (+ R6 distillation, 6-23 models) | 0.951 |
| **Final 1:** Raptor + R6-flip + our 18 models (1/3) | **0.958** |
| **Final 2:** the same with 12 of our models, plus a 4th public arm (S6, weight 0.2) | **0.958** |

The private leaderboard was not out at the time of writing.

Things that did not help: a second self-training round, mirror augmentation, an AUC-margin loss, self-consistency and
reasoning modes for the labeller, Raptor as a distillation teacher, an R6 weight of 0.8, and public label sets.
Gains also flattened after about 6 models.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/v2/` | The full pipeline: data prep, labelling, label building, training, evaluation, Kaggle inference and release |
| `scripts/v2/r6_distill/` | Running the public R6 model on training scans for distillation |
| `notebooks/v2/` | Kaggle inference notebooks and the builder that assembles them |
| `slurm/` | Cluster job files (packing, GPU queue worker) |
| `.claude/agents/mm-labeler.md` | The multimodal labelling agent (report + image montages -> 12 probabilities) |
| `docs/paper.md` | Write-up draft with method details and every number referenced |
| `docs/log.md`, `docs/research.md` | Run log and research notes from the competition |

Competition data, labels and checkpoints are not in this repository. The trained models are on Hugging Face:
[`Jitheswar/rsna-knee-v2-r6distilled`](https://huggingface.co/Jitheswar/rsna-knee-v2-r6distilled).

## Credits

The final blends use public work by other Kagglers, each attributed in the notebooks:

- **Raptor CoAtNet 384 / 224**: dreaddevelopment's recipe, with weights by nartaa (trained with OAI data, disclosed).
- **R6, R7 and compact S6 ConvNeXt pipelines**: heliosli, who also packaged the OrthoFoundation backbone weights.
- **S6fall weights**: nartaa.
- **Blend recipes**: goodpjw2008 (CoAtNet + R6 2.5D MIL). The original baseline is yamadan96 (`rsna-knee-d4-public0946`).

The full history before the cleanup is kept under the git tag `pre-cleanup`.
