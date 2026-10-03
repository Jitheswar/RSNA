# RSNA v2 — fresh pipeline status (resume here)

Goal: 1st place on the private LB of RSNA Knee Abnormality Detection (deadline 2026-10-22 23:59 UTC).
Publication framing: public notebook = attributed baseline; our pipeline (labels + image models) is the
measured contribution (ours alone / baseline alone / blend at a pre-fixed weight, bootstrap CIs).

## Pipeline (all code in scripts/v2/)
1. `prep.py` + `pack.py`: canonical reorientation (LPS), laterality normalisation (every knee = right knee;
   coronal/axial medial on image right, sagittal slice 0 lateral), 160 mm foreground-centred crop, 384 px,
   per-series percentile window, uint8 [D<=64,384,384] -> `artifacts/private/v2/vol384/vol/<series>.npy`,
   metadata `vol384/meta-train-*.csv`. Verified visually (fibula lateral = image left).
   Pack on the LOGIN node (local disk 2.4 GB/s); cluster nodes are NFS-bound (~100 MB/s).
2. Labels: severity-aware prompt (host definitions, ordinal grade + P(panel positive)).
   - frontier: `frontier_label.py` via Codex CLI gpt-6.1-sol high (rules s.8.2 allow LLM subscriptions)
     -> `artifacts/private/v2/llm/sol61-high.jsonl` (log `.log`).
   - local open models (bf16, vLLM env `~/vllm-env`): `llm_label.py` Qwen3.8-27B (--think), Gemma-4-31B.
   - public baseline labels: `labels/pubv2.csv` (llm_labels_v2, 0.5->0.1).
3. Folds: `artifacts/private/v2/folds.csv` (5-fold multilabel stratified, seed 42, gold spread 12/12/11/11/12).
4. Model: `model.py` KneeNet = timm encoder on 2.5D slices (K=16/series, <=6 series by type priority)
   -> slice tokens + type/pos embeddings -> 2-layer transformer -> 12 target queries. `train.py` (bf16,
   EMA, GPU affine/intensity aug, no flips (laterality!)). Gold studies use gold labels in training folds.
5. GPU queue: `slurm/v2_gpu_worker.sbatch` workers run `artifacts/private/v2/queue/pending/*.task`
   (atomic claim). `scripts/v2/train_task.sh` copies volumes to node-local /tmp first.

## State log
- 2026-10-03 ~18:30 UTC: packing done on login node; workers 8249/8250 (A100, NODE3) running;
  task 005 = ConvNeXt-S 384 fold0 public labels (pipeline check); task 010 = Qwen3.8 labels (waits vLLM).
  Frontier labelling running (6 parallel Codex sessions).

## Next
- Evaluate label sources on gold58 (AUC per target) -> build ensemble labels `labels/ens_v1.csv`.
- Train 5 folds with best labels; backbones: convnext_small 384, then diverse (efficientnetv2/maxvit/coatnet).
- Build Kaggle inference notebook: v2 prep on test DICOM + our models (+ attributed baseline blend), 9 h limit.

## Key finding (2026-10-04 ~00:40 IST): reference panel is more liberal than written definitions
gold58 vs frontier (gpt-6.1-sol high) severity grades — P(gold positive | grade):
- Effusion: small 10/22, unspecified 13/16, moderate 7/8, large 5/5.  Baker's: small 3/7 (5 mm cyst positive!).
- MCL: grade-2 sprain/partial 7/10.  ACL: partial-unspecified 2/3.  Medial OA: grade 2 4/11; PF OA grade 1 3/7.
- Synovitis NOT MENTIONED in report: 11/35 gold positive (image decides).  Fracture not mentioned 4/27.
=> strict-definition LLM probabilities are mis-calibrated; use grade->P(gold) isotonic maps fit on gold58
   (`scripts/v2/build_labels.py`; LOO log-loss 0.40 vs public labels 0.57). Raw sol61 p AUC 0.894 vs pubv2 0.887.
Paper material: "report-to-reference calibration".
