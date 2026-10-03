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
