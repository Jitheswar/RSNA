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

## 2026-10-04 08:15 IST — LB results
| model (single fold 0) | labels | CV weak (cal_v1_f0) | public LB |
|---|---|---|---|
| ConvNeXt-S 320, K=12 uniform, tf head | pubv2 | (0.860 vs pubv2) | 0.896 |
| CoAtNet-2 384 bag+MIL (6/4/4 eval) | cal_v1 | 0.873 | 0.896 |
| ConvNeXt-S 320, K 20/12/10, tf head, aux grades | cal_v1 | 0.904 | **0.911** |
- CV on cal-label weak AUC tracks LB. fp16 inference verified == bf16 (precision_check.py). Bag model gains +0.015
  weak with 3x eval windows (reeval.py) -> k-infer override for bag models.
- Simple ResNet34 pool head (forum advice) much weaker here (0.804 @ep4 vs 0.899).
- Queue: v1 cnxs384 f0, t1 f2/f3/f4, v2 convnext_base f0; e2 coat folds on hold. Then students from T1 OOF.
- Frontier labels 3,050/4,407; resume 10:08 IST.
- 09:25 IST: **2-fold ConvNeXt-S (f0+f1) ours-only = 0.935** (sub 56813074) vs 1-fold 0.911 -> fold averaging +0.024.
- Label fusion check on 24 gold (folds 0-1): report cal_v2 AUC 0.915, image OOF 0.912, 50/50 mix 0.944 -> students.
- Fold-0 blends: t1 + e2 (bag) adds nothing (0.9056 max at t1 weight 0.8+); simple ResNet34 pool final 0.843.
- 10:57 IST: frontier sol61 labels COMPLETE (4,407/4,407). cal_v3 (+_f0.._f4) = sol61x3+astra+pubv2x2+qwen38+gemma4;
  aux_grades_v2 = sol61 grades (full). Students will use mix(cal_v3, T1 OOF).
- V1 (ConvNeXt-S 384) tracks 320 (slightly below at ep5) -> resolution not a lever. Ablations queued at prio 9000.
- 12:50 IST: T1 folds 0-2 weak 0.904/0.911/0.909; OOF gold35 0.897 [0.846,0.936]. V1 384 = 0.903 (no gain).
  T1 f3/f4 running (done ~14:00). Auto: students (mix1 = 0.5 cal_v3 + 0.5 T1 OOF, prio 002) + 5-fold ours-only submit.
  Rehearsal2 (fixed sim dir, baseline + 1 v2 model, 1300 studies) running >4 h. Kaggle quota 13 h left.
- 14:00 IST: **T1 5-fold complete**: OOF weak 0.9046 (vs cal_v3), gold58 0.881 [0.841,0.915]; weak spots LatOA 0.75,
  Synovitis 0.77, LatMen 0.80, PF OA 0.82. Student labels stud_mix1 = 0.5 cal_v3 + 0.5 T1 OOF (gold58 AUC 0.927 vs
  report 0.915 / image 0.881). Students s_mix1 f0-f4 queued (prio 002x). 5-fold ours-only submission pushed (v5).
  Ablations moved to queue/hold (release when GPUs idle). V2 ConvNeXt-B f0 running.
- 14:25 IST: **5-fold T1 ours-only = 0.935** (same as 2-fold; LB noise ~0.005). Blend submission (baseline +
  T1 5-fold, rank w=0.5) pushed as jitheswar/rsna-v2-blend v1 (auto-submits after commit run). Rehearsal2 still
  running (>5.5 h for 1300 studies) -> runtime risk; the blend's hidden run answers it. Kaggle quota 10.75 h.
- 14:35 IST: blend submitted (baseline + T1 5-fold, w=0.5); commit log OK (baseline COMPLETE, v2 leg 5 models 34 s).
  Hidden scoring pending -> answers runtime. Queue: s_mix1 f0 running (8254), V2 cnxb f0 (8250), then s_mix1 f1-4,
  s_mix1_384 f0-4 (ETA all ~Oct 5 05:00). Per-target: 384 vs 320 complementary (avg >= best on most targets).
- 15:20 IST **Rehearsal2 (1300 studies, 2xT4) timings**: dinov2 2556s, a5 917s, radimagenet 1191s, public_raptor 5773s,
  residual_coat 3678s, global96_coat 3691s, d4 2330s, repairv1_coat 2870s -> baseline 23038s (6.4 h);
  v2 leg (1 model) 1174s (prep-bound, 1157s for 1300 studies on 4 CPUs) -> total 24222s (6.73 h).
  => ~2 h headroom: 10-20 ConvNeXt-S models feasible (~100 s inference per model on 2xT4) if hidden N ~1300.
- Student s_mix1 f0 @ep4 vs cal_v3: 0.9113 > teacher 0.9074 (fair yardstick).
- 15:45 IST fold-0 fair comparison (weak vs cal_v3 / gold12): teacher 0.9074/0.927, **student s_mix1 (ep6)
  0.9170/0.960**, ConvNeXt-B 0.9021/0.917 (no gain -> no 5-fold), 384 0.9078/0.920. Student+teacher 0.9148 (<student)
  -> final ensemble should be student-family based. Round-2 students (mix2 = 0.5 cal_v3 + 0.5 mean(T1,s_mix1) OOF,
  prio 004) auto-launch when s_mix1 5 folds finish.
- 16:20 IST: 3rd worker 8258 got KLEF-G-NODE2 = **A30 24 GB** (not A100): train.py auto-adapts (bs1, accum x2,
  grad ckpt, micro 16) -> ~6.6 h/fold. Round-2 automation re-armed (scratchpad/auto_r2.sh, prio 0029 = before
  384 students). Tip: never pkill -f with a pattern contained in the same command line (kills own shell).
- 17:05 IST active automations (scratchpad): auto_r2.sh -> round-2 students (s_mix2, prio 0029) when 5 s_mix1 folds
  done; auto_sub_v2.sh -> submit s_mix1 5-fold ours-only at >= 2026-10-05 05:33 IST. s_mix1 f0 final 0.9175 vs
  teacher 0.9074 (cal_v3). ConvNeXt-B f0 0.8955 (dropped). Blend submission (56819xxx) scoring, ETA ~21:30 IST.
- 18:40 IST: public LB top = 0.963 (rhoskeri), 0.958-0.962 for ~20 teams; ours-only 0.935, baseline 0.943 (yamadan96/rsna-knee-d4-public0946 scores 0.943 despite its name).
  Forum (topic 745214): DINOv2 single models 0.942 LB (gold 0.917); label versions with gold 0.945 scored LB 0.910
  -> do NOT tune labels on gold58 (our per-target LOO mix weights: 0.8974 vs fixed 0.5 = 0.8978 -> keep 0.5).
  External MRI data: host allows only data accessible without institutional sign-off (KneeCoT ruled out, OAI
  effectively out; MRNet no-derivatives) -> we use none (prize-eligibility + paper).
- Report heterogeneity (paper): Spanish reports leave 26-65% of findings unaddressed vs 0-15% for most languages;
  synovitis unaddressed in 62-92% everywhere -> image evidence matters for silent findings (students).
- s_mix1 gold per fold (12 gold each): f0 0.958 / f2 0.912 / f3 0.914 vs teacher 0.927 / 0.895 / 0.864.
  46-gold subset: student 0.907 vs teacher 0.876 (LatOA 0.839 vs 0.733, Synovitis 0.863 vs 0.802, Baker 0.973 vs 0.904).
- DINOv2 support (model.py: ViT img_size, [CLS, mean-patch] readout; train.py --layer-decay). Queued
  s_dinob_mix1 f0 (ViT-B/14 reg4, 280 px, lr 5e-5, layer decay 0.8; same stud_mix1 labels) at prio 00250 =
  diversity candidate before the 384 students.
- Mirror augmentation (`--mirror p`, data.py): mirror the whole study (sagittal slice order + window channels
  reversed, coronal/axial horizontal flip) so a normalised right knee becomes a left knee; labels kept (anatomical
  identity is mirror-invariant). reeval.py --mirror-tta scores plain / mirrored / average. Queued s_mix1_mir f0
  (prio 00251) to compare with s_mix1 f0 (0.9175 vs cal_v3).
- 19:16 IST inference window count (s_mix1 f0, weak vs cal_v3 / gold12): 14/8/7 0.9158/0.954, trained 20/12/10
  0.9175/0.958, 30/18/15 0.9181/0.958, 40/24/20 0.9185/0.958 -> keep 20/12/10; spend inference time on more models.
  PatientSex (DICOM, 90% present) adds nothing beyond the image model. Round-2 labels switched to
  0.5 cal_v3 + 0.5 s_mix1 OOF (auto_r2b.sh); 384 students will be switched to stud_mix2 (auto_384to_mix2.sh).
- 20:00 IST DINOv2-B student f0 per epoch (weak vs stud_mix1 / gold12): e3 0.936/0.929, e5 0.949/0.916 vs ConvNeXt-S
  student e3 0.955/0.933, e5 0.959/0.940 (slower convergence; judge by ensemble gain at e9). Queue: s_mix1 f4 (8254,
  ~21:00), DINO f0 (8250, ~20:50), s_mix1 f1 (A30, ~22:20); pending: mirror f0 (00251), EffNetV2-S 384 student f0
  (00252); 384 students held until stud_mix2 exists (auto_384to_mix2.sh releases them as s_mix2_384).
  family_eval.py: fold-0 student alone beats student+teacher / +384-teacher (weak and gold).
- 20:30 IST DINOv2-B fp16 inference check (300 fold-0 val studies): fp16 == bf16 == fp32 (weak 0.9620/0.9620/0.9621, no NaN, mean|dp| 1e-4) -> ViT safe on T4 fp16.
- 20:45 IST **DINOv2-B student f0 done**: weak vs cal_v3 0.9105 (ConvNeXt-S student 0.9175, teacher 0.9074), gold12
  0.925. Per target DINO better on LatMen 0.911/0.907, Synovitis 0.856/0.850, Contusion 0.953/0.948, Fracture
  0.935/0.926; much worse on ACL 0.923/0.958 (patch-14 at 280 px too coarse for the thin ligament?).
  Ensemble with ConvNeXt student: uniform rank-avg 0.9179 (+0.0004), per-target weights (ens_weights, shrink 0.5,
  in-sample) 0.9196 (+0.002). Modest but positive -> DINOv2 family queued on round-2 labels (s_dinob_mix2 prio 004x,
  auto_dino_mix2.sh once stud_mix2 exists).
- 21:00 IST FIX package_models.py: per-run `weight` and `family` were added to args AFTER torch.save, so packaged
  checkpoints lacked them and kaggle_infer silently used uniform weights / skipped ens_weights.json (no submission
  used non-uniform weights yet). Now saved. kaggle_infer: mirror TTA (orig + mirrored study) for models trained with
  --mirror (args.mirror kept in the package); parity verified vs data.py on 4 studies (plain and mirrored).
- 21:07 IST end-to-end kaggle_infer test (3 public test studies, packaged ConvNeXt s_mix1 w=2 + DINOv2 w=1 with forced mirror TTA, fp16): runs, weights applied correctly (weighted mean verified).
- 21:20 IST **BLEND (baseline + T1 5-fold teachers, rank w=0.5) = 0.946** (sub 56819531) vs baseline alone 0.943
  (CORRECTED by owner: the 'public0946' notebook scores 0.943) -> +0.003 from our teachers (0.935 alone) at the
  pre-fixed w=0.5 (public 30%, +-0.004 noise); completed within 9 h (runtime OK for baseline + 5 ConvNeXt-S). Next: students (fold gold +0.017..+0.047 over teachers) ours-only at 05:33 IST, then
  blend baseline + s_mix1 at the same pre-fixed w=0.5 (auto_blend2.sh, after the ours-only submission).
- 21:25 IST s_mix1 f4 done. Student vs teacher (weak vs cal_v3 | fold gold): f0 0.918/0.907 | 0.958/0.927;
  f2 0.914/0.913 | 0.912/0.895; f3 0.916/0.905 | 0.911/0.864; f4 0.924/0.913 | 0.913/0.881 -> gold +0.032 mean.
  Running: mirror f0 (8250), EffNetV2-S 384 student f0 (8254), s_mix1 f1 (A30, ~23:30); pending abl_nolat f0.
- 21:45 IST **Cross-target calibration (paper)**: panel Fracture is ranked better by the report's CONTUSION statement
  than by its fracture statement (gold58 LOO: fracture-only 0.788, contusion-only 0.885, logistic fusion 0.893; fold-
  specific leak-free 0.788 -> 0.899; report-label macro 0.855 -> 0.865). Coefficients stable across folds (fracture
  0.65-0.80, contusion 0.80-0.93). Pre-specified pairs Synovitis<-Effusion, Contusion<-Fracture, LatOA<-PF OA,
  PF OA<-MedOA, Baker<-Effusion, MCL<-MedMen do NOT help (all LOO <= own) -> only Fracture changed.
  `scripts/v2/xcal_labels.py` -> labels/cal_v4(+_f0.._f4) = cal_v3 with Fracture replaced. Round 2 now uses
  REPORT=cal_v4 (auto_r2c.sh; make_students_t1.sh takes $REPORT).
  Also: gold58 is trauma-enriched vs the training pool (calibrated means ACL 0.41 vs 0.16, MCL 0.16 vs 0.06,
  contusion 0.33 vs 0.19, fracture 0.31 vs 0.15) -> the panel sample (and maybe the test) is not prevalence-matched.
- 23:10 IST **Mirror augmentation = negative** (fold 0, weak vs cal_v3 / gold12): s_mix1 0.9175/0.958; s_mix1_mir
  (p=0.5 whole-study mirror) plain 0.9133/0.952, mirrored input 0.9138, mirror-TTA avg 0.9145/0.954. Ensemble
  s_mix1 + mir 0.9180 (+0.0005); + DINO 0.9187. -> keep laterality-normalised training without mirroring (supports
  contribution 1: a consistent anatomical frame beats asking the network to infer medial/lateral). No mirror family.
- 23:40 IST **s_mix1 5-fold complete: OOF weak 0.9179 (teacher 0.9046), gold58 0.9195 [0.892, 0.943] vs teacher
  0.881 [0.840, 0.915]** (+0.038; every target up: LatOA 0.873->0.903, MCL 0.869->0.892, Contusion 0.932->0.949).
  EffNetV2-S 384 student f0 0.9004 weak (< ConvNeXt 0.9175, DINO 0.9105) and hurts ensembles -> dropped.
  **Round 2 launched 23:28** (auto_r2c): stud_mix2 = 0.5 cal_v4 + 0.5 s_mix1 OOF (label gold58 AUC 0.946 vs mix1
  0.927; image part alone 0.9195). Queue: s_mix2 f0 (A30) + f1 running, f2-f4, then s_mix2_384 f0-4 (003x), then
  s_dinob_mix2 f0-4 (004x); abl_nolat f0 running (8250).
- 23:50 IST paired bootstrap gold58 student - teacher = +0.038 [+0.020, +0.061] (P<0.001).
- 23:55 IST queued recipe tests on stud_mix2 (fold 0): s_mix2_pw (--pos-weight, rare targets) prio 00295,
  s_mix2_k24 (24/16/14 windows) prio 00296 -> compare with s_mix2 f0. Automations for Oct 5 (UTC day starts
  05:30 IST): auto_sub_v2 (05:33 s_mix1 ours-only) -> auto_blend2 (baseline + s_mix1, w=0.5) -> auto_sub_mix2
  (s_mix2 5-fold ours-only once its 5 folds are done AND the blend is submitted, so dataset versions don't clash).
- 23:40 IST OWNER PRIORITY: 1st = winning, 2nd = paper. -> GPU order: round-2 students, s_mix2_384 + DINO
  families, recipe tests for round 3; paper ablations (abl_noaux, abl_uniK) only on otherwise-idle GPUs (abl_nolat
  already ~60% done, left to finish). External MRI data still excluded (prize-eligibility / DQ risk).
- 23:45 IST silence-aware mixing (report weight 0.5 if a frontier source addresses the finding, else 0.25): gold58
  label AUC 0.9238 vs uniform 0.5 0.9221 (fold-specific cal_v4 + s_mix1 OOF) -> within noise, keep uniform 0.5.
  (fuse_labels' 0.946 for stud_mix2 uses non-fold-specific cal_v4 -> optimistic; leak-free = 0.922.)
- 23:50 IST abl_nolat was only ~30% done (not 60%) -> stopped (winning first) and returned to hold
  (9000a0_abl_nolat_f0.task); 8250 now runs s_mix2 f2. All three paper ablations wait for idle GPUs.
- 02:00 IST round-2 s_mix2 f1 done: weak vs cal_v3 0.9179 (s_mix1 0.9174, teacher 0.9072); vs cal_v4 0.9208 (0.9209);
  gold12 0.936 (s_mix1 0.953; 12 studies, noisy). Rank-avg s_mix1 + s_mix2 0.9227 vs cal_v4 (+0.002 over either)
  -> round 2 ~ round 1 alone (diminishing returns) but the two rounds ensemble well. Fracture weak (cal_v4) 0.942
  vs 0.940.
- 02:05 IST slot-3 automation changed: auto_sub_mix12.sh submits round-1 + round-2 students (10 models) ours-only after the s_mix1 blend is submitted and s_mix2 has 5 folds (replaces auto_sub_mix2.sh).
- 02:25 IST s_mix2 f2: weak(cal_v4) 0.9197 vs s_mix1 0.9155; gold11 0.900 vs 0.912. Folds 1+2: round 2 >= round 1
  on weak labels but -0.012..-0.017 on fold gold (23 studies, noisy) -> watch for confirmation drift; decide with the
  5-fold gold58 paired bootstrap when f0/f3/f4 finish (~06:10). 384 + DINO families train on stud_mix2.
- 04:10 IST **Round 2 is WORSE on gold** (46 gold, folds 1-4; f3/f4 = best-so-far ckpts): macro 0.9042 vs round-1
  0.9147 (-0.010), lower in 4/4 folds. Fracture -0.032 (cal_v4 cross-calibration helps REPORT labels but hurts the
  image student's fracture ranking), MCL -0.029, MedMen -0.019, LatOA -0.024, Effusion -0.014 (labels unchanged ->
  confirmation drift from self-distilling the same architecture). Weak AUC (report labels) was flat/up -> report
  metric misleads here; trust panel (gold) direction.
  DECISION: revert to round-1 labels (stud_mix1 = 0.5 cal_v3 + 0.5 T1 OOF) for all new families. Switched pending
  tasks: s_mix1_384 f0-4 (003x), s_dinob_mix1 f1-4 (004x; f0 exists), recipe tests s_mix1_pw / s_mix1_k24 (0029x,
  compare with s_mix1 f0). Killed auto_384to_mix2 / auto_dino_mix2. s_mix2 models kept only as ensemble candidates.
  Next round (if any) should use a DIFFERENT teacher family/ensemble (e.g. ConvNeXt+DINO+384 OOF), not self-distil.
- 04:30 IST final folds 1-4 (46 gold): round-1 0.9147, round-2 0.9076, rank-avg 0.9135 -> round 2 confirmed worse; avg no better than round 1. Running: s_mix2 f0 (A30), s_mix1_pw f0 (8254), s_mix1_k24 f0 (8250); next s_mix1_384 f0-4, s_dinob_mix1 f1-4.
- 04:40 IST Oct 5 submission plan (UTC day starts 05:30 IST): slot 1 auto_sub_v2 (05:33, s_mix1 ours-only); slot 2
  auto_blend2 (baseline + s_mix1, w=0.5); slot 3 auto_blend3 = blend-weight PROBE (w=0.65 if ours-only >= 0.943
  baseline else 0.35; same dataset version) -- paired comparison on the same public studies is far less noisy than
  independent scores. (auto_sub_mix12 cancelled: round 2 adds nothing on gold.)
- 06:02 IST slot 1 submitted: **sub 56836780 = s_mix1 5-fold students ours-only** (00:06 UTC, PENDING). The new
  Kaggle CLI prints "N submissions remaining today" instead of "Successfully submitted" -> automations stalled;
  FIX submit_only.sh / submit_nb.sh now echo an explicit "Successfully submitted" marker (marker added manually to
  submit7.log). Slot 2 (blend baseline + s_mix1, w=0.5) pushed as rsna-v2-blend v2 at 06:01.
- 06:06 IST **STUDENTS ours-only = 0.944** (sub 56836780) vs teachers 0.935 (+0.009) and attributed baseline 0.943
  -> our own pipeline alone now beats the public baseline. Gold58 OOF gain (+0.038) transferred to the panel test.
  Probe (slot 3) will use w=0.65. Blend w=0.5 (slot 2) in commit run.
- 06:20 IST queued full-data students s_mix1_full_s1/s2 (seeds 43/44, all 4,407 studies, prio 006x, out runs/<run>/full/
  last.pt). Armed auto_r2e.sh: once s_mix1_384 (5 folds) and s_dinob_mix1 (f1-4) are done, labels stud_mix1e =
  0.5 cal_v3 + 0.5 mean(s_mix1, s_mix1_384, s_dinob_mix1 OOF) (ENSEMBLE teacher, not self-distillation) ->
  ConvNeXt-S 320 students s_mix1e f0-4 at prio 005x (before the full-data seeds).
- 06:25 IST Oct 5 slots used 3/5: 56836780 students ours-only 0.944; 56837853 blend baseline + s_mix1 w=0.5 (PENDING);
  56838122 blend w=0.65 probe (PENDING; scores ~13:00 IST). Blend commit runs take ~8-10 min of Kaggle GPU.
- 07:05 IST round 2 full 5-fold: weak vs cal_v3 0.9178 (round 1 0.9179); gold58 0.9155 vs 0.9195, diff -0.004 [-0.014, +0.006] (n.s.) -> no gain from self-distillation round; round-1 labels kept for new families. s_mix1_pw f0 done (own-label weak 0.9625 / gold12 0.966 vs s_mix1 0.9606 / 0.958); k24 at e7 0.9633 / 0.974.
