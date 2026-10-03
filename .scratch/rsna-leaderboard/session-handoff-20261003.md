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

## Direction (owner, ≈17:15 UTC) — publication framing

The work also feeds a university paper. Public notebooks are **attributed baselines only** (current: fork of
`yamadan96/rsna-knee-d4-public0946`, 0.943). Do not chase higher-scoring public notebooks. Every submission is
baseline + our model; report baseline alone, ours alone (OOF + Gold58) and the blend at a weight fixed in advance,
with bootstrap CIs. Keep negative results (Qwen3 relabelling, fp16 collapse) and parity evidence as paper material.
- TPU v2 (`jitheswar/rsna-fast-tpu-v2`: ConvNeXt-Tiny, 24 slices, 15 epochs, 5 folds, in-session smoke) queued 17:06 UTC.
  XLA logic verified in a torch_xla 2.8 container on the XLA CPU backend; mesh_reduce payload bug fixed first.
- Baseline switched (owner): best self-contained public notebook **`aastikrajan15/knee-s75-w50` (0.945)**; versia-5,
  jiweiliu fast-2xT4 and dinosaur-v5 all 0.943. Its one private dataset entry is unreferenced in code. Diff vs the
  0.943 baseline: Fracture added to `_RAD_EXCLUDE`, w50 CoAt weights, sequential CoAt readers.
  Blend notebook: `python3 notebooks/blend/build.py <w> rsna-knee-s75w50-fastblend s75w50`.
- Owner: if GPU/TPU are insufficient, use our cluster (CPU folds already there; GPU array 8186 queued).
- **Correction:** s75-w50/w60 are not reproducible baselines. Their final cell runs an embedded "S10 stage" that loads the
  author's private checkpoints `f{0..4}_convnext_tiny_320x24_pt_v5_cv9xxx_best.pt` (asserts on them) and rank-blends at
  OUR_WEIGHT 0.5/0.6; 0.943 public + that model = 0.945. Baseline reverted to the reproduced 0.943 notebook
  (`notebooks/blend/public0946.ipynb`, default stem). Their design (ConvNeXt-Tiny, 24 slices, 320 px, CV ~0.90–0.915)
  mirrors our TPU v2 and is external evidence that a ~0.90 weak-CV model adds ~+0.002 at a 0.5 blend.

## SESSION END SNAPSHOT (≈18:00 UTC) — read this first in the next session

**Leaderboard:** best 0.943 (56788419, unchanged fork of `yamadan96/rsna-knee-d4-public0946`), rank ~741/4,980; leader 0.961.
**Owner:** wants 1st place; all resources approved (Kaggle GPU/TPU, cluster). Work also feeds a university paper:
public notebooks = attributed baselines only (see memory `publication-baseline-framing`). s75-w50/w60 (0.945) are
NOT usable baselines (private checkpoints).

**Key discovery:** cluster GPUs are **A100-80GB** (`--partition=cluster1-gpu,cluster1-gpu-add --gres=gpu:1`).
GPU array 8186 (`slurm/fast_train_folds.sbatch`) completed all 5 folds at **~140 s/epoch** (CPU: 74 min/epoch).
Results `artifacts/private/fast-v1/run2`: weak AUC 0.828–0.849; **pooled Gold58 0.8437**; rank blend with anchor:
w=0.05 −0.0003 [−0.0023,+0.0010], w≥0.10 negative. Not yet submission-worthy. CPU duplicate 8189 cancelled.

**Running / queued:**
- 320 px volume rebuild, array 8231 on cluster2 nodes 11/13/14/15/16 → `artifacts/private/fast-v2-320/{vol,index-*.csv}`
  (~35 min from 17:53 UTC). Train with `RSNA_FAST_ROOT=artifacts/private/fast-v2-320`.
- Kaggle TPU `jitheswar/rsna-fast-tpu-v2` (ConvNeXt-Tiny, 24 slices, 15 ep, 5 folds, bs1) QUEUED since 17:06 UTC
  (TPU queue ~10 h). With A100s available it may be unnecessary; it costs nothing while queued.

**Next steps (fast on A100):**
1. Scale the model on cluster GPUs: ConvNeXt-Tiny/Small, 24 slices, 320 px (`--n-slices 24 --backbone ...`,
   RSNA_FAST_ROOT=fast-v2-320), more epochs. Target: external evidence says ~0.90 weak-CV models add ~+0.002 at w≈0.5.
2. Evaluate each run with `scripts/fast/eval_blend.py --run <dir>` (fixed weights, bootstrap CI).
3. When a run's blend CI is non-negative, build `notebooks/blend/build.py <w> rsna-knee-d4-fastblend` (baseline
   public0946), upload model package as dataset `jitheswar/rsna-fast-model` (infer.py, train.py, rsna_prep/,
   models/fold*.pt), push, submit. Kaggle GPU quota: ~23 min left until Oct 10 → one visible run.
- Correction 18:00 UTC: 320 px rebuild runs ~3.5 series/s → finishes ≈20:00 UTC (not 35 min).
- Old superseded coordinator `rsna-ticket73-configured-continuation-v4.service` still launches CPU jobs (e.g. 8236);
  stop it (`systemctl --user stop ...`) if the capacity is needed. Its outputs are not part of the fast lane.
