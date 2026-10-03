# RSNA campaign handoff — October 3, 2026

Supersedes the October 2 handoff, which the owner flagged as outdated. The
ticketed CPU full-backbone and medical-head routes stay preserved but are no
longer the main path to rank.

## Situation (03:25 UTC)

- Public LB: leader 0.961; 0.955 ≈ rank 34; 0.946 ≈ rank 159; our best 0.936 ≈ rank 1,998 of 4,945.
- Reference labels: only 58 studies. The public `llm_labels_v2.csv` report labels
  reach only 0.887 macro AUC against them, so the reference reads are image-based.
  Strong image models trained on soft report labels are what the leaders have.
- Only 4 submissions were used in 10 days. Use the daily slots for verified-better candidates.

## Fast lane (new)

1. **Public frontier submission**: clean, unmodified private fork
   `jitheswar/rsna-knee-d4-public0946-fork` v1 of `yamadan96/rsna-knee-d4-public0946`
   (Apache notices inherited). Visible run COMPLETE; submitted as **56788419** (pending).
   The earlier D4 hidden failure (56739912) was our wrapped variant, not this recipe.
2. **Volume bank**: `scripts/fast/pack_volumes.py` packed all 23,101 Ticket 35
   full-slice series (4,178 studies) into 32×224×224 uint8 volumes under
   `artifacts/private/fast-v1/vol` (35 GB) plus `index-*.csv`.
3. **Inference parity**: `scripts/fast/parity_check.py` rebuilt 5 series from raw
   DICOM with `rsna_knee.preprocessing.prepare_series(PreprocessConfig(image_size=(224,224)))`;
   max abs error 0.0 versus the cache.
4. **Trainer**: `scripts/fast/train.py` — timm 2.5D (adjacent-slice RGB) encoder,
   slice transformer, series attention with plane/sequence-type embedding, soft
   report labels (0.5→0.1) plus gold overrides, 5 folds (seed 42; gold spread).
   GPU env `~/rsna-gpu2` (torch 2.14 cu130, timm 1.0.19, hf_hub<1.0).
5. **Cluster GPU**: `slurm/fast_train.sbatch` (cluster1-gpu-add, 1 GPU). Smoke job
   8148 queued behind other users; `vllm_gpu*` GRES is reserved to another account.
6. **Kaggle GPU route**: consolidated shards `kaggle-upload/vols-{0..7}.npy` uploading
   as private dataset `jitheswar/rsna-fast-vols`; runner `notebooks/fast-train/run.py`
   trains two folds concurrently on 2×T4 (`--fp16`).

## Next

- Read 56788419's score. If it is ≥0.945, it becomes the blend base.
- Train 5 folds (cluster A30 or Kaggle T4), then check OOF against gold58 and weak labels.
- Build the inference notebook: base public pipeline plus our model through the parity-checked
  preprocessing, with a rank blend weight chosen on the gold58/OOF evidence; submit.
- Larger levers: better report labels (local Qwen3-8B in `~/models`), higher-res/more slices,
  several backbones, pseudo-labelling the unlabeled test-like pool.

## Update (≈05:00 UTC)

- Dataset `jitheswar/rsna-fast-vols` ready. Kaggle `jitheswar/rsna-fast-train-f01` (folds 0,1; 10 epochs, 10.5 h guard) RUNNING.
  `k23` pushes automatically via a background retry loop (log `artifacts/private/fast-v1/logs/push-k23.log`) once a GPU session frees.
- Cluster: fold 4 full run job 8157 (`artifacts/private/fast-v1/run1`); smoke 8148 cancelled.
- Qwen3-8B yes/no log-prob labels (`scripts/fast/llm_label.py`) on all 58 gold: 0.835 alone, 0.887 blended
  vs public v2 0.887 — **rejected**; the Kaggle labelling kernel was superseded by a no-op v2.
- Inference: `scripts/fast/infer.py` + vendored `rsna_prep` package (`artifacts/private/fast-v1/model-pkg`),
  end-to-end CPU test passed; inference volumes byte-identical to training volumes.

## Update (≈08:45 UTC) — training moved to cluster CPU

- Owner stopped Kaggle training (GPU quota: 21,600 s/week, 20,186 used, refresh Oct 10; TPU 72,000 s unused).
  All Kaggle training kernels deleted; reservations 0. Submission 56788419 still scoring.
- Kaggle fp16 GPU runs collapsed (pred std ≈1e-4). Fixed: CNN-only autocast, LR 1e-4/3e-4, 10% warmup,
  collapse guard. CPU fp32 overfit check learns; TPU fixed-shape trainer `train_tpu.py` built (forward parity 2.4e-7), unused.
- CPU array **8189_0..4** (nodes 5/7/8/9/10, 64 cores, bf16 encoder autocast) → `artifacts/private/fast-v1/run3-cpu`.
  Epoch 0 (72 min each): weak AUC 0.708–0.734, pred std 0.07–0.12. GPU backup array 8186 queued → `run2`.
- `scripts/fast/eval_blend.py`: epoch-0 pooled Gold58 ours 0.623, anchor 0.9532; every rank-blend weight
  0.05–0.30 is negative with CI below zero. **Blend only if the final OOF gives a non-negative CI.**
- Fail-safe blended notebook prepared (`notebooks/blend/build.py`): public 0.946 + appended leg, verified
  locally (blend path; broken-model path keeps public CSV byte-identical). Needs dataset `jitheswar/rsna-fast-model`.

## Update (≈12:05 UTC)

- **56788419 COMPLETE: public 0.943** (author's label said 0.946). New best; rank 741/4,980; leader 0.961.
  Submission ran ~8.5 h end to end, so the 9 h hidden budget leaves little room for an appended leg.
- CPU folds after epoch 2: weak AUC 0.772–0.796; pooled Gold58 0.623 → 0.740 → **0.785**;
  w=0.05 rank blend with the 0.936 anchor now neutral (95% [-0.0016, +0.0015]).
- Owner approved all resources. Kaggle TPU smoke `jitheswar/rsna-fast-tpu-smoke2` queued >2 h (no TPU free yet).
