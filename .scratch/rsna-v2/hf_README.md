---
license: cc-by-nc-4.0
tags: [medical-imaging, knee-mri, rsna, multi-label-classification]
private: true
---
# RSNA Knee Abnormality Detection: v2 R6-distilled ConvNeXt-S ensemble (team Elyra)

18 KneeNet students (ConvNeXt-S `convnext_small.fb_in22k_ft_in1k_384`, img 320, multi-series MIL, 12 findings).
Trained on report-derived + multimodal labels, distilled with the public R6 model's train-set predictions
(rank-mapped onto the base label marginals; weights 0.4 = `s_r6`, 0.6 = `s_r6b`).

| Family | Files | Notes |
|---|---|---|
| s_r6 | 5 folds + 2 full-data seeds | R6 weight 0.4 |
| s_r6b | 5 folds + 1 full-data seed | R6 weight 0.6 |
| s_r6b_s2 | 5 folds | R6 weight 0.6, seed 2 |

Scores (public LB, macro AUC): these 18 alone ~0.951; blended 1/3 in rank space with public Raptor + R6-flip = 0.958
(submission 57461662, final #1).

Inference: `kaggle_infer.py` (uses `model.py`, `data.py`, `prep.py`, `lat_clf.bin` laterality classifier; config per checkpoint in `manifest.json`).
Release tag: see `release_tag.txt`. Public third-party weights used in the blend (Raptor, R6, S6) are NOT included.

## Quick start (our 18 models alone, ~0.951 public LB)
```bash
hf download Jitheswar/rsna-knee-v2-r6distilled --local-dir rsna-knee-v2
cd rsna-knee-v2 && pip install -r requirements.txt
python kaggle_infer.py --comp /path/to/rsna-knee-abnormality-detection --models *.pt --out preds.csv --fp16
```
`--comp` must have the competition layout: `test.csv`, `test_series.csv` and the DICOM folders
(use `--split train` for `train.csv` / `train_series.csv`). Output: one row per study, 12 probability columns
(ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion,
Fracture), averaged over the 18 models. Needs a CUDA GPU (runs on a Kaggle T4).
Tested with torch 2.14, timm 1.0.19, pydicom 3.0.1.

### Fast option: 6 models, same public score (a third of the model compute)
More models stopped helping after about 6 (public LB, our models alone: 6 = 0.951, 12 = 0.951, 23 = 0.951).
The 6 `s_r6` models (5 folds + 1 full-data model) were scored on their own at **0.951**:
```bash
hf download Jitheswar/rsna-knee-v2-r6distilled --local-dir rsna-knee-v2 \
  --include "*.py" "lat_clf.bin" "manifest.json" "requirements.txt" "s_r6__fold*" "s_r6_full_s1__full__last.pt"
cd rsna-knee-v2 && pip install -r requirements.txt
python kaggle_infer.py --comp /path/to/rsna-knee-abnormality-detection \
  --models s_r6__fold*.pt s_r6_full_s1__full__last.pt --out preds.csv --fp16
```
All 18 give smoother predictions (slightly more stable on unseen data) at 3x the model compute (DICOM loading costs the same either way).

## License
CC BY-NC 4.0 (non-commercial): trained on RSNA competition data and on labels distilled from public non-commercial models.
Not a medical device; research use only.

## Inference notebook (final #1, Kaggle `jitheswar/rsna-v2-rehearsal` version 12, public LB 0.958)
`notebooks/final1_raptor_r6flip_ours18.ipynb` + `kernel-metadata.json`. Kaggle T4 notebook, no internet. Stages:
1. Raptor CoAtNet 384 + 224 (public, nartaa / dreaddevelopment, OAI-trained; attributed in the notebook).
2. R6 flip-TTA ConvNeXt (public, heliosli).
3. Fixed rank blend of the public arms, then our 18 models (this repo) rank-blended at 1/3.

Needs these Kaggle inputs: `nartaa/rsna-knee-publication-swa-weights-20261007`, `heliosli/rsna-knee-r6-frozen-fe055406-w1`,
`heliosli/rsna-orthofoundation-fold0-epoch7-20261004`, and our models (this repo's files; on Kaggle `jitheswar/rsna-v2-models-c`).
