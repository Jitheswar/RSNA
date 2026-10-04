# Paper outline (working draft) — RSNA 2026 Knee Abnormality Detection

**Working title:** Report-to-Reference Calibration: Learning Multi-Finding Knee MRI Classification from
Multilingual Radiology Reports with LLM Severity Grades and Anatomical Normalisation

## Problem
- 4,407 training knee MRI studies (19 sites, ~10 languages of reports), only 58 with reference labels
  (two MSK radiologists + adjudication, image-based, 12 findings); ~1.3k hidden test studies; macro AUC.
- Reports unavailable at test time -> weak supervision from report text.

## Contributions (each measured separately from the attributed public baseline)
1. **Canonical anatomical normalisation**: LPS reorientation, laterality normalisation (every knee mapped to a
   right knee), foreground-centred 160 mm crop; self-supervised laterality checker (99.0% held-out) that fixes
   ambiguous DICOM geometry (29/4,407 training studies corrected; validated against report-stated side).
2. **Report-to-reference calibration**: LLMs extract ordinal severity grades; the expert panel is more liberal
   than the written definitions (e.g. P(gold+|"small effusion") = 10/22; grade-2 MCL sprain 7/10; synovitis
   gold-positive in 11/35 reports that never mention it). Grade->P(gold) maps fit on gold58 (isotonic/Beta).
   Multi-source (frontier GPT-6.1-sol, GPT-6-astra, open Qwen3.8-27B / Gemma-4-31B, public labels) fusion.
   LOO log-loss vs gold58: 0.390 (fused) vs 0.406 (frontier alone) vs 0.422 (public, calibrated) vs 0.573 (public raw).
3. **Multi-series 2.5D transformer** with series-type/position embeddings and per-finding queries; auxiliary
   severity-grade head.
4. **Report x image label refinement** (teacher OOF soft bootstrapping).

## Experiments / tables to fill
- T1 label sources: gold58 AUC/log-loss (raw, calibrated, fused), LOO.
- T2 image models: OOF weak AUC, OOF gold58 AUC (+bootstrap CI), public LB, per target.
- T3 ablations: labels (public vs calibrated vs fused), laterality normalisation on/off, aux head, student round.
- T4 blend with attributed public baseline at pre-fixed weight: baseline alone / ours alone / blend.
- Negative results: Qwen3-8B relabelling (0.835 gold), fp16 sequence-head collapse, strict-definition prompts
  mis-calibrated (frontier p strict vs calibrated), Qwen thinking-mode cost.

## Reproducibility
- Code: scripts/v2/*; labels: artifacts/private/v2/labels/*; folds fixed (seed 42, gold spread 12/12/11/11/12).
- Compute: 2x A100-80GB (cluster), Kaggle 2xT4 for inference (9 h limit).

## Methods (draft, 2026-10-04)

### Data and splits
4,407 training studies (24,371 series, 819k DICOM slices; all uncompressed Explicit VR LE), 58 with reference labels.
Five-fold multilabel-stratified split (seed 42; stratified on public-label positives >0.6 plus the gold flag, so gold
studies spread 12/12/11/11/12). All OOF statistics use this split; gold studies are trained on (with their reference
labels) only when outside the validation fold.

### Canonical volume preparation (`scripts/v2/prep.py`)
Slices are ordered by projection of ImagePositionPatient on the series normal (InstanceNumber fallback); duplicate
positions removed. Each series is rotated/flipped (no resampling across planes) to a canonical LPS frame per plane
(sagittal: columns posterior, rows inferior, slices +x; coronal: columns patient-left, rows inferior; axial: columns
patient-left, rows posterior). Laterality: DICOM Laterality tag, else image-centre x (|x|>=20 mm), else weak geometry;
left knees are mirrored so every knee appears as a right knee (medial on image right for coronal/axial; sagittal slice
0 lateral). A self-supervised laterality checker (ResNet-18 on three posterior coronal slices, trained to detect
horizontally flipped normalised slices; 99.0% held-out accuracy) re-flips studies with p>0.99, or p>0.5 when the side
came from weak geometry (29/4,407 training studies; validated against report-stated sides). Square physical crop of
min(160 mm, FOV) centred on the foreground centroid; per-series 0.5/99.7 percentile window; 384 px uint8; depth cap 64.
Training/inference parity: byte-identical volumes from the packing (zip) and inference (file) paths.

### Report labels and reference calibration (`llm_label.py`, `frontier_label.py`, `build_labels.py`)
Prompt states the host definitions and asks, per finding, for an English evidence phrase, an ordinal severity grade
(finding-specific scale) and P(panel positive). Sources: GPT-6.1-Sol (high reasoning, via Codex; allowed by host
clarification topic 733965), GPT-6-Astra (medium, categorical states), open-weight Qwen3.8-27B and Gemma-4-31B (bf16,
vLLM), and the public llm_labels_v2 table. Raw strict-definition probabilities are mis-calibrated w.r.t. the reference
panel (e.g. "small effusion" gold-positive 10/22). Each source is mapped to P(gold+|grade) per finding on the gold
studies (isotonic regression over ordered grades with Beta(1,1)-style smoothing toward prevalence; categorical
states: smoothed per-state rates; probabilities: isotonic); calibrated sources are averaged with weights over the
sources available per study. Fold-specific tables exclude the validation fold's gold studies (leak-free gold OOF).

### Image model (`model.py`, `train.py`)
Up to six series per study chosen by type priority (fat-suppressed fluid-sensitive sagittal/coronal/axial first, then
non-FS). 2.5D windows (slice i-1,i,i+1 as RGB) sampled per plane (sagittal 20, coronal 12, axial 10; evenly spaced
with jitter in training), shared ImageNet-pretrained encoder (ConvNeXt-S in22k-ft-1k), GeM pooling, linear projection
to d=384, learned series-type (plane x FS x fluid) and slice-position (relative, mm) embeddings, 2-layer transformer
over all window tokens, 12 learned finding queries (cross-attention) -> logits. Auxiliary head regresses normalised
LLM severity grades (weight 0.3, masked when unaddressed). Training: BCE on soft calibrated labels, AdamW
(encoder 1e-4, head 5e-4, wd 0.05), 5% warmup + cosine, 10 epochs, EMA (0.998), bf16, GPU affine/intensity
augmentation without flips (laterality preserved), 2 studies x 4 accumulation.

### Results so far
| single fold-0 model | labels | OOF weak AUC (cal labels) | public LB |
|---|---|---|---|
| ConvNeXt-S 320 uniform K=12, transformer | public v2 | — | 0.896 |
| CoAtNet-2 384 bag-of-windows + MIL | calibrated v1 | 0.873 | 0.896 |
| ConvNeXt-S 320 per-plane K + aux | calibrated v1 | 0.904 | 0.911 |

### Results (2026-10-04 update)
**Label sources vs gold58 (raw probability AUC):** frontier GPT-6.1-Sol 0.894, public v2 0.887 (developed on gold58),
pilkwang 0.870, Qwen3.8-27B 0.852, Gemma-4-31B 0.842, GPT-5.6-Sol public 0.835. Calibrated multi-source fusion
LOO log-loss 0.387 (vs 0.573 raw public).

**Recipe comparison (fold 0, same calibrated labels, OOF weak AUC vs cal_v3):**
| model | weak AUC | public LB (1 fold) |
|---|---|---|
| CoAtNet-2 384, bag-of-12-windows + MIL (public recipe) | 0.873 (0.890 w/ 3x eval windows) | 0.896 |
| ResNet34 256, per-plane mean/max pooling ("simple baseline") | 0.843 | — |
| ConvNeXt-S 320, all-slice transformer, per-plane K, aux grades | 0.907 | 0.911 |
| ConvNeXt-S 384 (same) | 0.908 | — |
| ConvNeXt-B 320 (same) | 0.902 | — |
| **student** (labels 0.5 cal_v3 + 0.5 teacher OOF) | **0.918** | pending |
Public labels with the old uniform-K recipe: LB 0.896.

**Ensembling:** 2-fold teacher 0.935 LB; 5-fold 0.935 LB. Fold-0: 320+384 rank average +0.0035 weak AUC.
**Gold58 OOF (5-fold teacher):** 0.881 [0.841, 0.915]; ACL 0.988, effusion 0.984, medial meniscus 0.971, medial OA
0.952; weakest lateral OA 0.754, synovitis 0.772, lateral meniscus 0.799, PF OA 0.822.
**Student label quality on gold58:** report-only 0.915, image-only 0.881, 50/50 mix 0.927 (complementary).
**Runtime (2xT4, 1300 simulated test studies):** attributed baseline 6.4 h; our leg 20 min (CPU-bound DICOM prep),
~100 s per additional ConvNeXt-S model.
