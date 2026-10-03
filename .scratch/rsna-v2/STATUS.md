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

## Plan (2026-10-04 01:20 IST)
Forum intel (topic 735304 etc.): single 5-fold models reach LB 0.95+ (ResNet50@224 0.954); resolution 224-320 fine
(FOV 140-160 mm); main lever = labels + pseudo-label "soft bootstrapping" (labels = 0.5 report + 0.5 teacher OOF).
Host (topic 733965): commercial LLM APIs for report labels explicitly allowed.
1. Labels L0 = gold-calibrated grades from sol61 (+qwen38, gemma4) [build_labels.py --fold-specific].
2. Teachers on L0 (fold-specific labels for leak-free gold OOF): T1 convnext_small 320 K12 10ep (+aux grades),
   T2 tf_efficientnetv2_s 320 K12. ~1.4-1.8 h/fold on A100.
3. Students: labels L1 = 0.5*L0 + 0.5*mean(T1,T2 OOF) -> retrain; evaluate gold OOF + LB.
4. Submissions: ours alone vs blend with attributed baseline (pre-fixed weight), bootstrap CIs.
Kaggle quota: GPU 24.4 h left this week (resets Oct 10) — NOT exhausted. Rehearsal kernel `jitheswar/rsna-v2-rehearsal`
(baseline + v2 leg with dummy ConvNeXt-S x5 on 1300 simulated studies) measures hidden-run time.
Laterality checker `lat_clf.pt` (99% held-out): flip if p>0.99 or (weak geometry & p>0.5); 29 train studies fixed.

## 2026-10-04 01:40 IST
- Codex usage limit hit (sol61 1,550/4,407 labelled); auto-resume script at 05:02 IST (6 workers).
- Astra (GPT-6-astra medium, user's earlier run) states for all 4,407 recovered -> labels/astra_grades.csv (categorical).
- LOO log-loss: sol61x2+astra+pubv2 calibrated 0.390 (best so far).
- Per-plane slice counts implemented (sag 20 / cor 12 / ax 10, slice masks) because ACL weak AUC lagged (0.79 @ep4).
- Rehearsal v1 failed in A5 stage (sim sample_submission mismatch) after DINOv2 stage = 2,600 s for 1,300 studies;
  v2 pushed with fixed sim. Kaggle GPU quota 23.7 h left.
- Auto: make_teachers.sh fires when Qwen labelling finishes (scratchpad/auto_teachers.sh).

## 2026-10-04 04:40 IST (after usage-limit pause)
- FIRST v2 LB: 1-fold ConvNeXt-S 320 public labels = **0.896** (sub 56806371); matches its fold gold AUC ~0.89 -> no
  test-time bug; recipe is weak vs public single-fold CoAtNet (~0.937-0.95).
- Public recipe (dreaddevelopment knee-mri-training-the-twelve-finding-model): 12 RANDOM 3-slice windows/study in
  training (24 eval), attention-MIL head, bb_lr 3e-5 / head 1e-3 OneCycle, BCE pos_weight, NO spatial aug,
  CoAtNet-2 384, 140 mm crop, top-3 epoch averaging.
- Implemented: --random-pos (bag sampling) + --ktr-* (train windows) / --k-* (eval), --head mil, --pos-weight,
  --aug {full,light,none}, --zoom (1.14 ~ 140 mm).
- Teachers (old recipe) paused -> queue/hold. Running fold-0 experiments: E1 coat384 new recipe + public labels,
  E2 same + cal_v1 labels.
- Gemma-4 labels done (raw AUC 0.842); adds nothing to calibration (LOO ll 0.389). sol61 1,650/4,407 (resume 05:02).
- Workers: 8250, 8254 running (8249 exited idle); 8258 queued.

## 2026-10-04 05:35 IST — label decision
- Same bag recipe (coat384) fold 0, epoch 4: E1 public labels weak 0.791 / gold12 0.823 vs E2 cal_v1 weak 0.866 /
  gold12 0.909 (E2 ahead at every epoch) -> commit to calibrated labels; E1 stopped (paper ablation can rerun later).
- Queue now (interleaved): family A = e2_coat384_cal1 (bag recipe, folds 0-4), family B = t1_cnxs320_cal1 (all-slice
  transformer recipe, folds 0-4). Fold 0 of each = recipe comparison under cal labels. ETA ~17:30 IST on 2 GPUs.
- Public label sets scored vs gold58: pubv2 0.887, pilkwang 0.870, GPT-5.6-Sol 0.835; dread soft labels exclude gold.
