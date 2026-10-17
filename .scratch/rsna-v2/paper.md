# Report-to-Reference Calibration for Knee MRI: Learning Multi-Finding Classifiers from Multilingual Radiology Reports

*Working draft (RSNA 2026 Knee Abnormality Detection). Numbers are from `.scratch/rsna-v2/STATUS.md`; every claim
below has a run, file or submission reference there. Priority: competition first, paper second.*

## Abstract (draft)
Clinical knee MRI comes with free-text reports, not image-level reference labels. We study the RSNA 2026 knee
challenge: 4,407 training studies with multilingual reports, 58 with labels from a two-radiologist reference panel,
and a hidden test set labelled by the same panel (12 findings, macro AUC). We identify three gaps between reports and
the panel — a *semantic* gap (the panel is more liberal than written definitions), *silence* (findings left unaddressed
by reports, 26-92% depending on finding and language) and an *anatomical* gap (inconsistent DICOM orientation and
laterality) — and close them with (1) canonical anatomical normalisation with a self-supervised laterality checker,
(2) report-to-reference calibration of LLM severity grades, (3) image-informed soft bootstrapping (students on a fixed
50/50 mix of report labels and out-of-fold image predictions) and (4) *multimodal* labels: a vision-language model
reads three MRI montages together with the report. Students improve reference-standard AUC from 0.881 (teachers) to
0.920, and full-coverage multimodal labels — a frontier model on 18% of studies, a local open-weight 27B VLM on the
rest, no API cost — raise it further (within-fold 0.927 -> 0.932) while leaving report-label agreement unchanged,
i.e. the labels move from the reports toward the image-reading panel. On the hidden test our models alone reach 0.947
and pre-fixed rank blends with attributed public pipelines 0.949. We document that most public stacks above 0.95 rely
on weights trained with controlled-access data (OAI), and report negative results (second self-training round, mirror
augmentation, AUC-margin loss, self-consistency and reasoning modes for the VLM labeller).

## 1 Introduction
Large imaging datasets come with reports, not reference labels. Here 4,407 training studies carry reports in about
nine languages and only 58 carry reference labels; the test set is labelled by the same panel. Three gaps separate
report text from the panel's image reading:
- **Semantic.** The panel is more liberal than the written definitions: a "small effusion" is panel-positive 10/22
  times, a grade-2 MCL sprain 7/10, and synovitis is panel-positive in 11/35 studies whose report never mentions it.
- **Silence.** Absence of mention is not absence of disease. Synovitis is unaddressed in 62-92% of reports in every
  language; Spanish reports leave 26-65% of findings unaddressed versus 0-15% for most other languages.
- **Anatomical.** DICOM orientation and laterality vary, so the same structure appears in different image positions.

We address the semantic gap with report-to-reference calibration, silence with image-informed soft bootstrapping, and
the anatomical gap with canonical reorientation, laterality normalisation and a laterality checker. Every component is
measured on our own pipeline and against a strong attributed public baseline, alone and blended at a pre-fixed weight.

## 2 Related work (to expand)
- Report-derived labels: CheXpert/MIMIC-CXR labellers; LLM labellers beat rule-based ones (fracture sensitivity 95% vs
  51%) and image models tolerate label noise; hedging language should become soft labels (MICCAI 2025 #0580;
  arXiv 2510.05664; GPT-4o MSK radiograph study 2026).
- When reports help training: image-text alignment helps well-represented findings but hurts rarely mentioned ones;
  combine self-supervision with auxiliary text objectives (arXiv 2510.24385).
- Noisy labels and self-training: soft bootstrapping (Reed et al. 2015), noisy student (Xie et al. 2020); confirmation
  bias and saturation after few rounds (Radhakrishnan et al., WACV 2024).
- Knee MRI: MRNet (Bien et al. 2018); CoPAS co-plane attention across sequences, 12 abnormalities (Nat. Commun. 2024);
  ECM-Knee multi-task (EClinicalMedicine 2025); 2.5D multi-series models and multiple-instance learning over slices.
- Calibration (isotonic, Beta) used to construct labels.

## 3 Data and task
4,407 training studies (24,371 series, 819k DICOM slices; all uncompressed Explicit VR LE; 19 sites), 58 with
reference labels (two MSK radiologists + adjudication, image-based, 12 findings: ACL, MCL, medial/lateral meniscus,
medial/lateral/patellofemoral OA, effusion, synovitis, Baker's cyst, contusion, fracture). Hidden test about 1,300
studies (public LB 30%, private 70%); macro AUC over the 12 findings. Reports are unavailable at test time. The gold
sample is trauma-enriched relative to the training pool (calibrated ACL prevalence 0.41 vs 0.16; fracture 0.31 vs 0.15).

## 4 Methods

### 4.1 Canonical volume preparation (`scripts/v2/prep.py`)
Slices are ordered by ImagePositionPatient projected on the series normal (InstanceNumber fallback), duplicates
removed. Each series is rotated/flipped (no cross-plane resampling) to a canonical LPS frame per plane. Laterality
comes from the DICOM tag, else the image-centre x position (|x| >= 20 mm), else weak geometry; left knees are mirrored
so every knee appears as a right knee. A self-supervised laterality checker (ResNet-18 on three posterior coronal
slices, trained to detect mirrored normalised slices; 99.0% held-out) re-flips studies at p > 0.99, or p > 0.5 when the
side came from weak geometry (29/4,407 training studies; 11/1,300 in a simulated test run). Square physical crop of
min(160 mm, FOV) centred on the foreground centroid, per-series 0.5/99.7 percentile window, 384 px uint8, depth <= 64.

### 4.2 Report labels and report-to-reference calibration
The prompt states the host definitions and asks, per finding, for an English evidence phrase, an ordinal severity
grade (finding-specific scale) and P(panel positive). Sources: GPT-6.1-Sol (high reasoning), GPT-6-Astra (categorical
states), open-weight Qwen3.8-27B and Gemma-4-31B (bf16, vLLM), and a public LLM label table (use of commercial LLM APIs
for report labels was explicitly allowed by the host). Each source is mapped to P(panel+ | grade) per finding on the
gold studies (isotonic over ordered grades with smoothing toward prevalence; categorical states: smoothed rates;
probabilities: isotonic) and calibrated sources are averaged. Fold-specific tables exclude the validation fold's gold
studies, so gold OOF numbers are leak-free.

### 4.3 Image model (`model.py`, `train.py`)
Up to six series per study by type priority (fat-suppressed fluid-sensitive first). 2.5D windows (slices i-1, i, i+1
as RGB) per plane (sagittal 20, coronal 12, axial 10; evenly spaced, jittered in training) through a shared
ImageNet-22k ConvNeXt-S encoder at 320 px, GeM pooling, projection to d = 384, learned series-type and slice-position
(relative, mm) embeddings, a 2-layer transformer over all window tokens of all series, and 12 learned finding queries
(cross-attention). An auxiliary head regresses the LLM severity grades (weight 0.3, masked when unaddressed).
Training: BCE on soft labels, AdamW (encoder 1e-4, head 5e-4, wd 0.05), warm-up + cosine, 10 epochs, EMA 0.998,
bf16, affine/intensity augmentation without flips. Variants: 384 px, DINOv2-B/14 (280 px, layer-wise lr decay 0.8),
BCE pos_weight, 24/16/14 windows.

### 4.4 Image-informed label refinement (students)
Teacher OOF predictions (each study predicted by the fold model that did not train on it) are mixed with the
calibrated report labels: y = 0.5 * y_report + 0.5 * p_teacher_OOF. The weight is fixed a priori; it is never fitted
on gold. Gold studies keep their reference labels when in a training fold. Students are re-initialised from ImageNet.

### 4.5 Multimodal (image + report) LLM labels
For each training study we render three montages (sagittal, coronal, axial; the best fat-suppressed fluid-sensitive
series; six slices from 15-85% depth; every knee displayed as a right knee, orientation stated in the prompt) and give
them, with the report, to a frontier multimodal LLM that returns P(panel positive) for the 12 findings (host
clarification, topic 733965: sending competition data to an LLM API is allowed). Studies are labelled in order of
report terseness (most unaddressed findings first). Training label where available: y = 0.5 * p_mm + 0.5 * p_OOF
(image-model ensemble); elsewhere the report mix of 4.4.

### 4.6 Ensembling and blending
Fold models of a family are averaged; families are combined by rank averaging. Under the 9 h Kaggle limit (2xT4),
families are selected greedily by OOF objective per second of measured inference cost. Blends with the attributed
public baseline use per-target rank averaging at a pre-fixed weight.

## 5 Experimental setup
Five-fold multilabel-stratified split (seed 42; gold spread 12/12/11/11/12). Metrics: OOF weak AUC (vs calibrated
report labels > 0.5, n = 4,407), OOF gold58 AUC (reference standard, with bootstrap CIs and paired bootstrap for
differences), public LB (hidden panel labels, about 390 studies; about +-0.004 noise for single scores). Compute:
2x A100-80GB (+ one A30) for training; Kaggle 2xT4 for inference.

## 6 Results

### 6.1 Label sources (gold58)
Raw probability AUC: GPT-6.1-Sol 0.894, public v2 0.887, pilkwang 0.870, Qwen3.8-27B 0.852, Gemma-4-31B 0.842,
GPT-5.6-Sol (public) 0.835. LOO log-loss: calibrated multi-source fusion 0.387-0.390, frontier alone 0.406, public
calibrated 0.422, public raw 0.573. Fused label AUC (fold-specific, leak-free) 0.855.

**Evaluating gold-calibrated labels on a tiny gold set.** Two traps: (i) a calibration map fit on all 58 gold studies
is in-sample for those studies (0.9145); (ii) leak-free fold-specific maps, pooled across folds, mix five
calibration scales and depress AUC (0.855; even a per-target Platt map, which cannot change within-target ranking,
drops raw 0.894 to 0.857). We therefore score labels by AUC over **within-fold positive/negative pairs only**, which
is immune to per-fold scales. On that metric (gold58, macro over 12 findings): calibrated report fusion 0.898 = raw
GPT-6.1-Sol probabilities 0.898; image-ensemble OOF 0.931; multimodal Opus 0.928, GPT-6.1-Sol 0.933;
0.5 report + 0.5 OOF 0.936; **0.5 multimodal + 0.5 OOF 0.949** (GPT 0.948); adding the report to the multimodal mix
lowers it (0.940). Gains over the report mix are broad: MCL +0.035, PF OA +0.033, effusion +0.031,
synovitis +0.025, fracture +0.016, lateral meniscus +0.013 (medial OA -0.008).

### 6.2 Image recipe (fold 0, single model)
| model | labels | OOF weak AUC (cal_v3) | public LB (1 fold) |
|---|---|---|---|
| ConvNeXt-S 320, uniform K=12, transformer | public | — | 0.896 |
| CoAtNet-2 384, bag of 12 windows + MIL (public recipe) | calibrated | 0.873 | 0.896 |
| ResNet-34 256, per-plane mean/max pooling | calibrated | 0.843 | — |
| ConvNeXt-S 320, all-slice transformer, per-plane K, aux grades | calibrated | 0.907 | 0.911 |
| ConvNeXt-S 384 / ConvNeXt-B 320 (same recipe) | calibrated | 0.908 / 0.902 | — |
Calibrated vs public labels with the same bag recipe (fold 0, epoch 4): weak 0.866 vs 0.791, gold12 0.909 vs 0.823.

### 6.3 Students vs teachers (5-fold OOF; headline)
| model | OOF weak AUC | gold58 AUC [95% CI] | public LB (5 folds) |
|---|---|---|---|
| teacher (calibrated report labels) | 0.9046 | 0.881 [0.840, 0.915] | 0.935 |
| student (0.5 report + 0.5 teacher OOF) | 0.9179 | 0.920 [0.892, 0.943] | **0.944** |
Paired bootstrap on gold58: +0.038 [+0.020, +0.061], P < 0.001. 11 of 12 findings improve on gold58 (ACL flat, 0.988 -> 0.984;
all 12 improve on weak AUC); largest gains on findings the reports under-call or omit: lateral OA 0.754 -> 0.865, synovitis 0.772 -> 0.851, Baker's cyst 0.908 -> 0.978,
contusion 0.892 -> 0.946, MCL 0.857 -> 0.896, lateral meniscus 0.799 -> 0.855 (gold58 per target).

### 6.4 Recipe refinements (students, fold 0; weak AUC / gold12)
| variant | weak | gold12 | notable per-target change |
|---|---|---|---|
| student baseline | 0.9175 | 0.958 | — |
| BCE pos_weight (rare findings) | 0.9197 (5-fold 0.9192) | 0.966 (gold58 0.9237) | MCL 0.897 -> 0.918, fracture 0.926 -> 0.940 |
| 24/16/14 windows | 0.9191 | 0.967 | PF OA 0.886 -> 0.904 |
| rank average of the two | 0.9222 | 0.975 | — |
| DINOv2-B/14 280 | 0.9105 | 0.925 | better LatMen/synovitis/contusion/fracture, ACL 0.923 |
| ConvNeXt-S 384 (5-fold) | 0.9179 | 0.9188 (gold58) | ensemble with 320: weak 0.9200, gold58 0.9212 |
| EfficientNetV2-S 384 | 0.9004 | 0.938 | hurts ensembles |
Inference windows (trained 20/12/10): 14/8/7 -0.0017, 30/18/15 +0.0006, 40/24/20 +0.0010 weak AUC.

### 6.5 Leaderboard (public 30%)
| submission | public AUC |
|---|---|
| attributed public baseline (yamadan96/rsna-knee-d4-public0946), unchanged | 0.943 |
| ours: 5-fold teachers / 5-fold students | 0.935 / 0.944 |
| blend, rank weight 0.5: baseline + teachers | 0.946 |
| blend, rank weight 0.5 / 0.65: baseline + students | 0.948 / 0.948 |
| ours: 25 students (5 ConvNeXt-S families x 5 folds) | 0.945 |
| ours: ensemble-teacher students s_mix1e (5-fold) / + 1 full-data model | 0.946 / 0.946 |
| ours: + 1 full-data fine-tune on multimodal labels (3 epochs, Kaggle T4) | 0.946 |
| blend, rank weight 0.65: baseline + s_mix1e 5-fold | 0.948 |
| blend, rank weight 0.65: baseline + s_mix1e 5-fold + full-data + mm fine-tune | 0.949 |
| A/B single full-data model vs its 2 mm-label fine-tunes (seeds 45/46) | 0.943 vs 0.944 |
| ours: s_mm2 5-fold / + full-data s_mm2 (full-coverage multimodal labels) | 0.946 / 0.947 |
| ours: 19 models (s_mm2 + seed 2 + 384 5-fold each, 3 full-data + 384 full-data) | 0.948 |
| attributed R7 stage alone (OAI-free half of the public 0.957 stack) | 0.949 |
| blend, rank weight 0.5: R7 + s_mm2 5-fold | 0.954 |
| blend, rank weight 0.4 / 0.5 / 0.6 / 0.7: R7 + 11 s_mm2 models (5-fold + 2 full-data + 4 folds @384) | **0.955** / **0.955** / **0.955** / 0.954 |
| blend, rank weight 0.5: R7 + 13 s_mm2 models (+ 384 f4 + seed-2 f0) | **0.955** |
| blend, rank weight 0.5: R7 + 19 models (ours-only 19 = 0.948) | **0.955** |
| blend 1/3 each: R7 + nartaa S6fall + 27 models | **0.957** |
| blend R7 .25 / ours .25 / S6 .5 | 0.956 |
| blend 1/3 each: R6 flip-TTA + S6fall + 34 models | 0.956 |
| blend 1/3 each: R7 + S6fall + 34 models | **0.957** |
| blend 0.5/0.5: nartaa S6fall (ImageNet init, OAI fallback removed) + 27 models -- no OAI lineage | 0.954 |
| ours, negative: ResNet50@224 / CoAtNet-1@224 families (5-fold OOF weak 0.898 / 0.911 vs ConvNeXt-S 0.921) | — |

Ensembles saturate: 6 -> 25 models moves the public score by <= 0.001, and a single full-data fine-tune on the
multimodal labels (coverage 17.7% of training studies) is invisible at ensemble level. The remaining label gain
(gold58 within-fold 0.936 -> 0.949) requires multimodal coverage of the full training set.


**Multimodal-label students (s_mm1, 5-fold, from scratch; labels 0.5 Opus mm + 0.5 ens5 OOF on 778 studies, ensemble-teacher
labels elsewhere).** Weak 0.9222 vs 0.9211 (s_mix1e), gold58 0.9178 vs 0.9178; rank-average of the two families 0.9234.
With 17.7% coverage, better labels on a minority of studies do not move a family; the pair gains from diversity.

**Full-coverage multimodal labels (s_mm2).** Local open-weight VLM labels (Qwen3.8-27B, vLLM, same montages + prompt;
gold58 alone 0.889, mixed with ens5 OOF 0.942 vs 0.936 for the report mix; Gemma-4-31B 0.937) fill the 82% of studies
without Opus labels (3,629 studies in ~65 min on 2 A100s). Students on these labels: gold58 within-fold 0.9318 vs
0.9271 (s_mix1e) and 0.9265 (s_mm1, 17.7% coverage); report-label AUC unchanged (0.9210 vs 0.9211), i.e. the labels
move toward the image-reading panel and away from the reports.


**Public-stack provenance (Oct 13).** Public notebooks reached 0.954-0.957 by blending an OAI-fine-tuned CoAtNet
("Raptor", acknowledging NIMH-NDA controlled access) with other pipelines. The host's accessibility criteria exclude
datasets gated by institution-specific approval, so we use neither those weights nor OAI. The OAI-free half of the
strongest stack (heliosli R7: ConvNeXt-S, fastMRI+ auxiliary task, public OrthoFoundation SSL weights) scores 0.949
alone; we evaluate it as an attributed baseline for blending.
R7's encoder initialisation, OrthoFoundation-L (ViT-L from DINOv3-L, DINO-style SSL; MIT licence, public weights;
arXiv 2601.18250), was itself pretrained on ~1.25 M knee images that include ~200 k OAI MRI slices and ~100 k OAI
radiographs. We never access OAI data; the competition rules explicitly allow freely available pre-trained models, so
we treat R7 as admissible but report it separately and keep an ours-only submission with no OAI lineage as a hedge.

**Blend weight is flat (Oct 14).** R7 + 11 s_mm2 models at our-leg rank weight 0.4 / 0.5 / 0.6 / 0.7 scored
0.955 / 0.955 / 0.955 / 0.954, and 13 models at 0.5 also 0.955: two equally strong, differently built pipelines
(R7 alone 0.949, ours alone 0.947) give +0.006 over either, and the gain is insensitive to the mixing weight. The
public split (~390 studies, +-0.004) cannot rank leg variants further; selection relies on gold58 within-fold AUC.

### 6.6 Negative results and ablations
- **Second self-training round** (same architecture, labels from round-1 student OOF): weak AUC unchanged
  (0.9178 vs 0.9179), gold58 0.9155 vs 0.9195 (-0.004 [-0.014, +0.006]); lower in 4/5 folds — confirmation drift.
- **Cross-calibrated fracture labels** (fracture <- report contusion): better as a report label on gold
  (0.788 -> 0.899, leak-free) but the image student's fracture AUC fell (0.929 -> 0.897): the image model already
  ranked fractures better than the reports, and contusion-driven targets blurred fracture vs contusion.
  Lesson: report-label AUC is not a safe yardstick for label changes; the reference-panel direction must decide.
- **Mirror augmentation** (whole-study mirror, labels kept): 0.9175 -> 0.9133 (0.9145 with mirror TTA) — a
  consistent anatomical frame beats asking the network to infer medial/lateral.
- **AUC-margin fine-tuning** (Deep AUC Maximization, Yuan et al. ICCV 2021; BCE + AUC-M on y>0.5, 3 epochs from
  the trained student): no gain (weak 0.9171 vs 0.9175, gold12 0.956 vs 0.958; pre-registered decision rule).
- **Per-target or silence-aware mixing weights** fitted/defined on gold: no gain (0.8974 vs 0.8978; 0.9238 vs
  0.9221); forum reports of gold-tuned labels overfitting (gold 0.945 -> LB 0.910).
- DICOM PatientSex adds nothing beyond the image model (AUC change <= 0.0015).

### 6.7 Reporting heterogeneity
Report language (stopword heuristic): English 1,738, Spanish 681, Turkish 548, Croatian/Serbian 401, Greek 321,
German 263, Bulgarian 220, Dutch 151, French 79. Unaddressed findings: Spanish 26-65% per finding vs 0-15% elsewhere;
synovitis 62-92% everywhere; OA 47-59% of Turkish reports.

### 6.8 Reproducibility and efficiency
- Byte-identical training/inference volumes; fp16 == bf16 == fp32 (ConvNeXt and DINOv2); pure fp16 (no autocast)
  parity: AUC 0.9304 vs 0.9305.
- End to end on Kaggle (1,300 simulated test studies through the exact submission path): held-out predictions match
  local OOF (mean |dp| 0.0008, corr 1.0000, identical AUC); 0 preparation failures.
- Memory-bounded inference: preparing studies with an eager process-pool map lets CPU preparation run ahead of the
  GPUs; each prepared study is ~25 MB (uint8, 384 px), so with 15-25 models the backlog reached ~25-27 GB of the
  29 GB notebook RAM and two submissions exceeded the 9 h limit through thrashing (a 12-model rehearsal at ~22 GB
  had survived). A bounded prefetch window (2x workers), per-study error isolation and an absolute session deadline
  (unscored studies fall back to the baseline rank) fixed it: the same 25-model submission then completed (0.945).
- Runtime on 2xT4 per 1,300 studies: ~290 s per ConvNeXt-S model (pure fp16; 326 s with autocast), DICOM preparation
  ~1,150 s on 4 CPUs (CPU-bound). Ours (5 models) ~30 min for 0.944 vs the baseline ~6.4 h for 0.943.

## 7 Discussion and limitations (draft)
**Where the gains come from.** The largest single improvement came from the labels, not the network: calibrated
report labels beat public labels by ~0.075 weak AUC on the same recipe, and image-informed students added +0.038
gold58 AUC and +0.009 on the hidden test. Architecture changes within a fixed label set moved results by <= 0.005
(resolution, backbone size, DINOv2, EfficientNetV2), while recipe changes that target specific findings helped where
predicted (pos_weight -> rare findings such as MCL and fracture; denser axial sampling -> patellofemoral OA).

**Family saturation.** Five same-backbone families (resolution, windows, pos_weight, self-training round) reach
OOF weak AUC 0.918-0.921 each and 0.922-0.923 in any combination; 25 models score 0.945 on the hidden test vs 0.944
for 5. Further gains have to come from labels (multimodal) or genuinely different models, not more of the same.

**What the reference panel rewards.** The panel calls findings that reports under-call (small effusions, grade-2
sprains, synovitis without mention) and the gold sample is trauma-enriched. Calibrating to the panel and letting image
evidence fill report silence are therefore complementary: calibration fixes the operating point of what reports say;
students fix what reports do not say.

**When self-training stops helping.** One round of soft bootstrapping helps; a second round with the same
architecture does not (gold -0.004), and changing a label because it matches the panel better as a *report label*
(fracture <- contusion) can hurt the image model. Report-label AUC rose or stayed flat in both cases, so it is not a
safe yardstick; we decide label changes on the reference panel's direction only and never tune weights on gold58.

**Compute and efficiency.** On Kaggle 2xT4 our five-model ensemble scores 0.944 in ~30 min, the attributed public
ensemble 0.943 in ~6.4 h. Inference cost (~290 s per ConvNeXt-S model per 1,300 studies, measured) bounds how many of
our models fit beside the baseline in 9 h; we select families greedily by OOF objective per second.

**Limitations.** Gold58 is small (CIs +-0.03) and not prevalence-matched; the public leaderboard covers ~390 studies
(+-0.004); results on the private split are pending. Report languages were identified heuristically. Commercial LLM
labels depend on proprietary models (we also report open-weight models). We used no external MRI data.

## Pending
- Laterality-normalisation ablation (abl_nolat), aux-head and uniform-K ablations (queued on idle GPUs).
- 5-fold pos_weight / 24-window / 384 / DINOv2 families; ensemble-teacher round; full-data seeds; final LB numbers.
- Private LB after the deadline.
