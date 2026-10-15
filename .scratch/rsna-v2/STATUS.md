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
- 07:25 IST **recipe tests (fold 0, weak vs cal_v3 / gold12)**: baseline s_mix1 0.9175/0.958; --pos-weight 0.9197/0.966
  (MCL 0.897->0.918, Fracture 0.926->0.940, LatMen, LatOA up); 24/16/14 windows 0.9191/0.967 (PF OA 0.886->0.904,
  Synovitis, LatMen up); pw + k24 rank-avg 0.9222/0.975 (best pair so far). -> queued s_mix1_pw f1-4 (0035x) and
  s_mix1_k24 f1-4 (0036x) BEFORE DINO f1-4 (004x); full-data seeds (006x) switched to the combined recipe
  (24/16/14 + pos_weight); make_students_t1.sh takes RECIPE/EXTRA; auto_r2e.sh now waits for 384+pw+k24+DINO and
  builds stud_mix1e from the 5-family ensemble OOF, students with the combined recipe (prio 005x).
- 08:07 IST timing probe pushed: jitheswar/rsna-v2-timing v1 = ours-only, 12 models (10 ConvNeXt-S 320, DINOv2-B 280, ConvNeXt-S K24) on 1300 simulated studies (never submitted) -> real 2xT4 per-model cost for sizing the final ensemble (the earlier ~100 s/model was a FLOP estimate, not a measurement).
- 09:30 IST **TIMING (measured on Kaggle 2xT4)**: 12 models (10 ConvNeXt-S 320 K20/12/10, DINOv2-B 280, ConvNeXt-S K24)
  on 1300 simulated studies = 3918 s, GPU-bound -> ~320 s per model per 1300 studies (NOT ~100 s). Budget: baseline
  ~23,000 s + ours N*320 s <= 32,400 s - ~45 min margin -> N <= ~20-22 models. cudnn.benchmark / channels_last /
  micro 96 give no speed-up (A100 bench 152-158 ms/study). -> pick the final ~20 models by greedy forward selection
  on OOF rank-averages; consider fewer windows (14/8/7 = 0.7x cost, -0.0017 weak) only if needed.
- 10:05 IST s_mix1_384 f1 done: weak 0.9147 (320: 0.9174), gold12 0.947; 320+384 rank-avg 0.9185 (+0.0011; teachers gave +0.0035); 320+round2 0.9196. A30 384 f0 ~2,600 s/epoch -> done ~14:00.
- 12:15 IST **Forum/code scan**: (1) EFFICIENCY PRIZE track ($7k/$6k/$5k): score = AUC/(0.5 - maxAUC) + runtime/32400
  (minimise) -> ~-2.1*AUC + hours/9: 1 h of runtime ~ 0.05 AUC; eligible only if the submission is one of our 2 final
  picks. Our ours-only (0.944 in ~30 min) is naturally efficient. Efficiency LB (Oct 4 snapshot) #1 Scott Willis 0.959.
  (2) speed tips (topic 743374): pure fp16 (no autocast) ~2.08x on T4; threaded DICOM reading 59 -> 372 MB/s.
  (3) single 5-fold models up to 0.954 (ResNet50@224, CoolinLai); Chris Deotte predicts private gold ~0.96, 1st ~0.975.
  (4) public stack considered overfit -> shake-up risk for stack forks (supports an ours-only final pick).
  IMPLEMENTED pure fp16: model.py dtype-aware forward (identical in training), kaggle_infer --pure-fp16, leg.py uses it.
  Parity on 150 fold-0 studies: ConvNeXt AUC 0.9304 vs autocast 0.9305 (max|dp| 0.0015), DINOv2 0.9247 = 0.9247;
  A100 speed 1.23x / 1.18x (T4 gain expected larger).
- 12:25 IST prep profile (3 test studies, warm cache): ~1.7 s/study CPU = dcmread 19%, pydicom pixel_array 19%, cv2.resize 12%, np.stack 11%, percentile 7% -> Kaggle prep is CPU-bound (4 vCPU). Added prep._raw_pixels (PixelData via np.frombuffer; bit-identical to pixel_array on 2,500 random train files; falls back to pydicom otherwise): volumes identical, prep ~15% faster. Main track unaffected (GPU-bound); helps the efficiency track.
- 12:45 IST **RACE FIX (submission safety)**: timing probe v2 crashed in 8 s because the notebook mounted the PREVIOUS
  rsna-v2-models version (old kaggle_infer without --pure-fp16): `datasets status` reported "ready" before the new
  version existed. A real submission could silently run old models (this morning's 0.944 = students, so it was OK).
  Now: release.sh writes a unique release_tag.txt, polls until Kaggle serves that tag; build.py bakes the tag into
  leg.py, which raises on a stale mount (outside the try -> kernel ERROR -> nothing submitted).
  torch.compile(enc.forward_features, max-autotune-no-cudagraphs): A100 1.41x (130 -> 92 ms/study) but 9 min
  compile/autotune -> not used now (maybe later for big ensembles if the cache amortises it).
- 12:57 IST **BLEND baseline + s_mix1 students (w=0.5) = 0.948** (sub 56837853) vs ours-only 0.944, baseline 0.943, teacher blend 0.946. w=0.65 probe pending.
- 13:10 IST **blend w=0.65 = 0.948** (sub 56838122) = w=0.5 (0.948): weight insensitive in 0.5-0.65 -> use ~0.55-0.6.
  Topic 744230 check (bf16 on T4 -> 50x slower emulation + silent 0.5 fallbacks): NOT our case. Inference is fp16 only
  (no bf16 in kaggle_infer/model/data/prep/leg), no per-study time guard, measured T4 timing; timing-run log: 0 study /
  prep failures in 1,300 studies, 11 laterality-checker flips (0.8%, train 0.66%). **End-to-end Kaggle parity at scale**:
  held-out (OOF) predictions of s_mix1 / s_mix2 fold models through the exact Kaggle path on 1,300 simulated studies vs
  local OOF: mean|dp| 0.0008, max 0.024, corr 1.0000, weak AUC 0.9205 = 0.9205 (s_mix2 0.9209 vs 0.9208).
- 13:51 IST **Pure-fp16 timing (Kaggle 2xT4, 12 models, 1300 studies)**: 3507 s vs 3918 s autocast -> 1.12x only
  (forum's 2.08x was CoAtNet; ConvNeXt depthwise 7x7 convs dominate on T4). ~292 s/model. Blend budget: baseline
  ~22,800 s + N*292 s + startup <= 32,400 - 2,700 margin -> N ~ 20 (family costs: ConvNeXt-S 320 x5 = 1,460 s,
  K24 x5 ~1,880 s, 384 x5 ~2,100 s). Release-tag guard worked (v3 ran on the right package).
- 13:58 IST armed auto_evening.sh: when 384+pw+k24 families complete (~21:00-21:30) -> slot 4 ours-only 25 models; then family_select (budget 6,400 s, 292 s/model) -> slot 5 blend baseline + selected families, w=0.6 (selection parsing dry-run OK). Skips if families are not ready by 03:00.
- 14:00 IST implemented AUC-margin fine-tuning (train.py --init-ckpt, --aucm-w, --aucm-margin; Yuan et al. ICCV 2021
  DAM, CheXpert 1st place): loss = BCE(soft) + w * AUC-M(y>0.5), alpha by SGD ascent (maximize=True), a/b in AdamW.
  Smoke test OK. Queued s_mix1_aucm f0 (init s_mix1 f0, 3 epochs, lr 2e-5/1e-4) at prio 00350 (next free GPU).
- 14:10 IST armed auto_aucm.sh (pre-registered: GO if s_mix1_aucm f0 weak >= s_mix1 f0 + 0.0015 and gold12 drop <= 0.01 -> queue AUC-M fine-tunes for s_mix1 f1-f4 at prio 0036a). Parser bug (matched the 'a + b' combo row) caught in dry run and fixed.
- 14:14 IST s_mix1_384 5-fold complete: weak 0.9179, gold58 0.9188 [0.890, 0.944] (= 320 family 0.9179 / 0.9195). Rank-avg 320+384: weak 0.9200, gold 0.9212 (best pair); 320+round2+384: 0.9207 / 0.9209. AUC-M test running on the A30.
- 15:40 IST public LB (5,196 teams): our team 'Elyra' (abhignan567, devaansh052, jitheswar, rithviksainagamally) rank 158, 0.948. Teams >= 0.963: 2, >= 0.960: 7, >= 0.958: 24, >= 0.955: 54, >= 0.950: 114, >= 0.943 (public stack): 1,397. NOTE: 5 daily submissions are shared by the 4-member team; tonight's automation uses slots 4-5.
- 16:17 IST **AUC-margin (DAM) fine-tuning = NEGATIVE** (pre-registered NOGO): s_mix1_aucm f0 weak 0.9171 vs 0.9175, gold12 0.9557 vs 0.9580; per target ~unchanged. Likely: binarising soft labels at 0.5 discards information; CheXpert gains came with 200k+ images. pw + k24 remain the effective recipe changes (pair 0.9222).
- 16:22 IST the freed A30 grabbed k24 f1 (~8.5 h on the A30 -> evening subs after midnight) -> moved z140 test to the queue head (00355), stopped k24 f1 on the A30 (it now runs z140), re-queued k24 f1 for the next A100 (~17:10). k24 family ETA ~22:35 -> evening submissions ~22:40.
- 16:21 IST queued s_mix1_pwk24 f0 (pos_weight + 24/16/14 windows combined; untested combination used by the ensemble-teacher round and full-data seeds) at prio 00365 (after k24 folds, before DINO) -> check vs pw / k24 alone before the round starts (~03:00).
- 18:16 IST **s_mix1_pw 5-fold** (BCE pos_weight): weak 0.9192 (+0.0013 vs s_mix1), gold58 0.9237 [0.897, 0.946] (+0.004) -> best single family; MCL weak 0.892 -> 0.912, PF OA 0.883 -> 0.876. Ensembles: 384+pw 0.9210/0.9244; 384+pw+round2 0.9214/0.9250 (best gold); all four 0.9215/0.9233. k24 family (PF OA) due ~22:35.
- 21:30 IST **MULTIMODAL LABELS (GPT-6.1-Sol via Codex, 3 FS montages x 6 slices + report; host 733965 allows
  Competition Data to LLM APIs)**: gold58 AUC (leak-free, no fitting) 0.930 vs report-only raw sol61 0.894, calibrated
  fusion 0.855, student OOF 0.919/0.924 (pw). Rank-avg student + multimodal 0.943. Per target multimodal: PF OA 0.953
  (student 0.842), MCL 0.992, LatMen 0.909, synovitis 0.833, Baker 0.944, contusion 0.901, fracture 0.914.
  Also: raw sol61 p ranks better than our calibrated fusion (0.894 vs 0.855 leak-free) -> calibration helps log-loss,
  not AUC. NEXT: effort-low check, then label all 4,407 studies (usage limits!) -> students on mm+student mix.
- 21:20 IST Codex usage unavailable (owner) -> stopped the Codex multimodal run (mm-sol61low.jsonl keeps the 56 gold
  low-effort rows: gold AUC 0.9245 vs medium 0.9300). Switched to Claude Opus 5.5 subagents: mm_prepare.py renders
  montages (artifacts/private/v2/mm/montages/<uid>/{sagittal,coronal,axial}.png) + batch JSON + INSTRUCTIONS.md;
  subagents view images (Read) and write {uid: {target: p}}; mm_collect.py -> JSONL. Pilot: 58 gold studies, 6 Opus
  subagents x ~10 studies. (.claude/agents/mm-labeler.md = Opus, effort medium, Read/Write only; registers next
  session — this session uses general-purpose + model opus.)
- 21:30 IST **Opus 5.5 multimodal pilot (58 gold, 6 subagents)**: gold58 AUC 0.926 (GPT-6.1-Sol mm 0.930; corr 0.94);
  student + Opus 0.941, pw + Opus + GPT 0.945. ~10k tokens / study, ~25 s / study per agent. SCALE-UP: 4,349 non-gold
  studies -> 290 batches x 15 (artifacts/private/v2/mm/opus-all, ordered by #unaddressed findings), ~14 Opus subagents
  in flight (general-purpose + model opus), next-batch counter in scratchpad/opus_next.txt. Collect with
  mm_collect.py --tag opus-all.
- 02:05 IST (Oct 6) evening submissions DID go in (k24 family finished 00:02): 56859633 ours-only 25 students (00:14 IST),
  56859907 blend baseline + [s_mix1_pw, s_mix1_384, s_mix2] w=0.6 (00:30 IST; family_select obj 0.9232, gold 0.9250).
  Claude session limit (reset 02:00) killed all 14 Opus labellers (0 batches done) -> relaunched batches 0-13.
  GPU: DINO f3/f4 + pwk24 f0 running; full-data seeds pending; auto_r2e waits for DINO.
  Slurm 3-day limit: 8250 ends ~00:00 Oct 7, 8254 ~04:00, 8258 ~15:40. worker.sh now refuses to claim a task with
  < 4 h (A100) / 10 h (A30) left; scripts/v2/queue_watchdog.sh (detached, every 10 min) requeues tasks of dead jobs
  and keeps 3 v2-gpu workers submitted while tasks are pending.
- 07:02 IST (Oct 6) Opus labelling: 14/290 batches done (session limit hit again ~02:40, reset 07:00); owner: max 2-4 concurrent subagents -> running 4 (batches 14-17), counter scratchpad/opus_next.txt=18. auto_r2e queued s_mix1e f0-4 at 03:58 (5-family ensemble teacher, recipe 24/16/14 + pos_weight); f0 running. Submissions 56859633 (ours-only 25) / 56859907 (blend) still PENDING.
- 07:05 IST overnight families: s_dinob_mix1 5-fold weak 0.9110 / gold58 0.9066 (weaker; not selected);
  s_mix1_k24 5-fold 0.9192 / 0.9164; s_mix1_z140 f0 0.9181 / gold12 0.939 (vs 0.9175 / 0.958: no gain).
  family_select (unlimited budget) stops at pw + 384 + s_mix2 (weak 0.9214, gold 0.9250) = the pending blend's set.
  Opus labels so far (210 non-gold, terse-report-first ordering): mean p much lower than student OOF on these atypical
  studies; Spearman with student 0.30-0.63 per finding (independent second reading).
- 07:08 IST **label mixes on gold58 (probability space, leak-free)**: student ensemble (pw+384+mix2) OOF 0.9252,
  Opus mm 0.9256, report cal 0.8553; 0.5*mm + 0.5*student = **0.9430** (0.4/0.6: 0.9433, flat 0.4-0.5); adding
  report lowers it (0.926-0.934); current-style 0.5 report + 0.5 student 0.9182. -> new labels stud_mm =
  0.5 Opus mm + 0.5 student-ensemble OOF where mm exists, fallback stud_mix1 (scripts/v2/build_mm_labels.py).
  Opus progress: 18/290 batches (running 4: 18-21).
- 12:06-12:41 IST (Oct 6) **9 h timeouts diagnosed + fixed** (commit 517cb0f): eager pool.map prepared all 1300 studies
  ahead of the GPUs (25 MB/study measured from vol384) -> 15-25-model runs parked ~25-27 GB in the 29 GB notebook RAM
  (12-model probe ~22 GB survived; 5-model runs no backlog) -> thrashing. Now bounded prefetch (2x workers), per-study
  try/except in consumer threads, absolute session deadline (start + 9 h - 30 min; unscored studies stay NaN -> blend
  keeps baseline rank), per-model s/study + RSS logged. Tested: 3-study e2e parity 3e-4, 16-study deadline run (4/16
  scored, NaN rows -> baseline rank exactly). Resubmitted both runs.
- 16:59 IST **ours-only 25 models = 0.945** (sub 56873662; completed in time -> fix confirmed) vs 5-model students 0.944:
  5x more same-family models add +0.001 -> family saturation; need diversity / better labels. Blend 15 models (w=0.6)
  PENDING. Claude session limit 12:50-17:00 paused everything; idle GPU workers exited after 90 min (IDLE_MAX) and
  other users took G-NODE3 -> only 8258 running; 8289/8290 pending (Priority). Lesson: keep a low-priority backlog.
- 17:05 IST s_mix1e (5-family ensemble-teacher labels) 5-fold: weak 0.9211 (best single family), gold58 0.9178 (mid);
  all family combos saturate at weak ~0.922-0.923 / gold ~0.920-0.923. Queued diversity: s_r50_224 (ResNet50-D @224)
  f0-4 (0070x), s_coat1_224 (CoAtNet-1 rmlp @224) f0 (00705) + f1-4 (008x), s_mix1e_full_s1 (0090); labels stud_mix1e.
  Submissions (slots 3-4): ours-only s_mix1e 5-fold, then ours-only 2 full-data seeds (auto_sub_e.sh).
- 17:10 IST **label-quality audit (gold58)**. Pitfall: cal_v3.csv maps are fit on all 58 gold -> in-sample for gold
  rows (0.9145); fold-specific maps (cal_v3_f*) pooled across folds mix 5 calibration scales -> artificially low
  (0.855). Artifact-free metric = AUC over WITHIN-fold pos/neg pairs only: cal_v3 (oos) 0.898 = raw sol61 0.898;
  ens5 OOF 0.931; Opus mm 0.928; GPT mm 0.933; 0.5 cal_v3 + 0.5 ens5 0.936 (= 0.5 sol61 + 0.5 ens5 0.936);
  **0.5 Opus mm + 0.5 ens5 0.949** (GPT 0.948); adding the report to mm+ens lowers it (0.940). -> multimodal labels are
  +0.013 over the report mix: Opus labelling resumed (4 concurrent; batches 38-41; counter 42). An earlier 17:00 check
  that said "no gain" used the in-sample cal_v3 - wrong, corrected here.
- 17:21 IST **GPU access**: only 8258 (A30 24 GB, G-NODE2, ends ~15:40 IST Oct 7; stops claiming at <10 h left).
  G-NODE3 (4 GPUs) is CPU-saturated by job 8261 (other user, gpu:2, 64/64 CPUs, 359 days left) -> 2 GPUs idle but
  unschedulable; our 8289/8290 pending at priority 1001 behind others (1131-1248). A30 runs ConvNeXt-S folds in ~8.7 h,
  ResNet50@224 ~5 h (0.5 s/study, GPU 40%; train.py now skips grad-ckpt for <=256 px on small GPUs).
  Queue order: r50 f0 (running) -> s_mm1_full_s1 (to queue when labels are rebuilt, ~22:00) -> s_mix1e_full_s1 ->
  coat1 f0 -> r50 f1-4 -> coat1 f1-4.
- 17:21 IST Laterality: an Opus labeller noticed a mirrored knee (fibula on image right) in a study the checker had
  flipped (tag R, p_flip 0.996). The checker (self-supervised, 99% held-out) overrides tag/geometry at p>0.99: 17 tag +
  1 geometry + 11 weak-geometry train flips; 11/1300 flips in the rehearsal. Base rates suggest tag overrides may be
  mostly false positives -> blinded Opus audit of 29 flipped + 12 control coronal montages running. fix_laterality.py
  now syncs node-local volumes to the flip list exactly (flips back removed entries; tested restore = identical).
- Oct 6 17:25 -> Oct 11 13:30 IST: **paused by the Claude weekly usage limit** (Opus labelling subagents used most of it).
  Results that landed meanwhile: ours-only s_mix1e 5-fold **0.946** (best ours-only; sub 56879705); blend 15 models
  w=0.6 0.945, w=0.75 0.948; ours-only 2 full-data seeds = ERROR (likely: hidden rerun mounted the NEWER dataset version
  released right after -> release-tag guard raised; local e2e of the full model is fine). FIX (d67d878): separate model
  datasets per notebook type (rsna-v2-models for blends, rsna-v2-models-only for ours-only); never release onto a dataset
  while a submission using it is pending. GPU queue finished: s_r50_224 5-fold weak 0.8983 / gold 0.8980; s_coat1_224
  0.9108 / 0.9087 -> both weaker and do not help ensembles (negative); s_mix1e_full_s1 done. All GPU nodes now allocated
  to other users; worker resubmitted (pending). Opus labels: 47/290 batches (~760 non-gold studies).
- Oct 11 14:15 IST submitted chains: blend baseline + s_mix1e w=0.65 (submit18); ours-only s_mix1e 5 + full_s1 (submit19).
- Oct 11 results: ours-only s_mix1e 5 + full_s1 **0.946** (= best ours-only); blend baseline + s_mix1e w=0.65 **0.948** (= best).
  Cluster GPU queue unusable (other users' long jobs) -> training moved to Kaggle T4 (kernel jitheswar/rsna-v2-finetune-mm).
- Oct 12 06:40 IST Kaggle fine-tune v2 done (seed 45, init s_mix1e_full_s1, stud_mm1 labels, 3 ep, ~2.2 h/epoch on T4,
  loss 0.933 -> 0.920) -> runs/s_mm1ft_full/full/last.pt. Submitted ours-only 7 models (s_mix1e 5 + full_s1 + mm1ft;
  submit20, kernel rsna-v2-only v12). Pushed fine-tune v3 (seed 46) on Kaggle. Hourly auto-resume cron (session-only).
- Oct 12 07:29 IST ours-only 7 models (+ mm1ft) = **0.946** (sub 57090843) = 6-model 0.946 -> one mm fine-tune adds
  nothing visible at ensemble level. Blend 7 models w=0.65 (sub 57091668) PENDING. 3 team slots left today.
- Oct 12 11:35 IST **cluster GPUs back**: workers 8375 + 8376 running on G-NODE3 (2x A100-80GB). 8375 took the queued
  00600 s_mm1_full_s1 (full data, stud_mm1 labels, 10 ep, ~8 h -> ~19:30 IST). Queued s_mm1 5-fold (00610-00614, same
  recipe as s_mix1e but stud_mm1 labels, seed 42): gives the clean gold58 within-fold test of mm labels; 8376 runs f0.
  Workers 8377/8395-8397 still pending (G-NODE2). Keep the queue non-empty (workers exit after 90 min idle).
- Oct 12 13:35 IST Kaggle fine-tune v3 (seed 46) done (loss 0.934 -> 0.921) -> runs/s_mm1ft_full_s46. LB A/B of mm
  labels (chained, ours-only dataset, sequential): A = s_mix1e_full_s1 alone (submit22), B = its two mm fine-tunes
  s45+s46 (submit23, after A scores). Blend 7 models (57091668) still PENDING. Queued s_mm1_384 5-fold (00620-00624)
  as GPU backlog; worker 8377 (G-NODE2) started -> 3 GPUs on s_mm1 work.
- Oct 12 15:00 IST **BLEND baseline + 7 (s_mix1e 5 + full_s1 + mm1ft) w=0.65 = 0.949 = NEW BEST** (sub 57091668; vs 0.948
  without mm1ft). LB A/B: control full_s1 alone 0.943 (57100708); treatment mm fine-tunes s45+s46 0.944 (57101576) ->
  +0.001, within noise (and partly 2-seed ensembling): mm fine-tune non-negative, not proven. Slot 5: blend 8 models
  (+ mm1ft s46) w=0.65 (submit24).
- Oct 12 15:40 IST s_mm1_full_s1 done (full data, mm labels, 10 ep, 3.3 h). **s_mm1 f0 vs s_mix1e f0**: weak (cal_v3
  report labels, independent of mm) 0.9227 vs 0.9192 (+0.0035), gold12 0.9647 vs 0.9608 -> first clean signal that
  from-scratch training on mm labels helps. f1-f4 running on 4 GPUs (~18:00 IST). Slot 5 used: blend 8 models (submit24).
  Plan Oct 13: ours-only s_mm1 5-fold + s_mm1_full_s1; blend baseline + s_mm1 (+ s_mix1e) w=0.65.
- Oct 12 17:30 IST s_mm1 folds 0,2,3,4 vs s_mix1e (pooled): weak 0.9222 vs 0.9209 (+0.0013), gold46 0.9147 = 0.9147;
  pair rank-avg weak 0.9231. -> mm labels at 17.7% coverage: small/neutral at family level. f1 (A30) pending.
- Oct 12 ~17:45 IST owner asked to give one GPU to a friend (jeevan): cancelled workers 8375 + 8396 (re-grabbed it),
  requeued their 384 tasks, HELD pending worker 8397 (`scontrol release 8397` to restore). jeevan's 8445 now running on
  G-NODE3. We keep 3 GPUs (8376, 8395 on NODE3; 8377 on NODE2).
- Oct 12 19:17 IST GPU handover done: jeevan's 8465 running on G-NODE3 (A100-80GB freed by cancelling our 8395; its task
  requeued). Our workers now 8376 + 8397 (NODE3) + 8377 (NODE2, s_mm1 f1). 8397 got released and started before the
  re-hold took effect, so no held worker remains.
- Oct 12 21:05 IST **s_mm1 5-fold complete**: weak 0.9222 vs s_mix1e 0.9211 (+0.0011); gold58 0.9178 = 0.9178;
  rank-avg pair weak 0.9234 (+0.0023, diversity). Verdict: mm labels at 17.7% coverage = neutral as a family, useful as
  an ensemble member. Plan Oct 13 (after 05:30 reset): ours-only s_mix1e 5 + s_mm1 5 + both full-data models (12);
  blend baseline + same 12 at w=0.65 once the pending 8-model blend (57103106) has scored.
- Oct 12 21:45 IST **local open-weight VLM multimodal labels** (scripts/v2/mm_local.py, vLLM 0.30, enforce_eager — torch
  compile crashes on the mm profile run; gemma needs max-model-len 8192). Gold58 within-fold (mm_eval.py): alone /
  0.5 src + 0.5 ens5 (prob): Opus 0.928 / 0.949; **Qwen3.8-27B 0.889 / 0.942**; Gemma-4-31B 0.894 / 0.937; gemma+qwen
  avg 0.903 / 0.944; current non-mm recipe (0.5 report + 0.5 ens5) 0.936. -> Qwen mm labels are +0.006 over the
  current labels on the 82% of studies without Opus labels, at zero Claude usage. Running: Qwen labelling of all
  non-Opus studies, 2 shards on the 2 A100s (0053_*; 384 tasks preempted + requeued). Held: Qwen --think gold test
  (queue/hold/0054). Next: stud_mm2 = build_mm_labels --mm qwen-s0 qwen-s1 opus-gold opus-all (Opus overrides)
  --students ens5 --fallback stud_mix1e -> s_mm2 5-fold + full. Ensemble for the mix stays ens5 (pre-fixed; 7-family
  variants within +-0.001 on gold).
- Oct 12 22:40 IST Qwen3.8 mm labelling complete: 3,629/3,629 parsed (~2.2 s/study/A100, ~65 min on 2 GPUs).
  **labels/stud_mm2.csv** = 0.5 mm + 0.5 ens5 OOF on 100% of studies (Opus on 778, Qwen elsewhere; Opus rows identical
  to stud_mm1). Queued at top: s_mm2 5-fold (00550-00554, seed 42) + s_mm2_full_s1 (00560); 384 tasks requeued behind.
  ETA (2 A100): f0/f1 ~01:20, f2/f3 ~04:10, f4 + full ~07:30 IST Oct 13 -> evaluate vs s_mix1e / s_mm1 on gold58, then
  submit (Oct 13 slots 3-5).
- Oct 13 00:29 IST blend 8 models (+ mm1ft s46) w=0.65 = **0.949** (sub 57103106) = 7-model 0.949 (best, tie).
- Oct 13 01:30 IST **s_mm2 folds 0-1** (gold24): s_mm2 0.9509 vs s_mm1 0.9446 vs s_mix1e 0.9404 (+0.011 over s_mix1e);
  weak (report cal_v3) flat 0.9205 vs 0.9206 — expected, the labels moved away from the reports toward images.
  f2/f3 running (~04:10), then f4 + full (~07:30).
- Oct 13 04:25 IST s_mm2 folds 0-3 (gold47). Pooled: s_mm2 0.9190 vs s_mm1 0.9159 vs s_mix1e 0.9153. Within-fold:
  **s_mm2 0.9350** vs s_mm1 0.9323 vs s_mix1e 0.9305 (+0.0045); pairs 0.933-0.934. Consistent ordering mm2 > mm1 > mix1e.
  Cancelled the 05:41 auto-submission of old models; today's 5 slots go to s_mm2 after f4 + full (~07:30 IST):
  ours-only s_mm2 5 + full; blend baseline + s_mm2 (+ s_mm1/s_mix1e) w=0.65.
- Oct 13 07:10 IST **s_mm2 5-fold complete** (gold58 within-fold): s_mm2 **0.9318** vs s_mix1e 0.9271 vs s_mm1 0.9265
  (+0.0047); pooled 0.9212 vs 0.9178; weak 0.9210 = 0.9211. Mixing with older families lowers gold (0.9285-0.9295).
  Submitted: ours-only s_mm2 5-fold (submit25) + blend baseline + s_mm2 w=0.65 (submit26). s_mm2_full_s1 ~11:00 IST.
- Oct 13 08:29 IST ours-only s_mm2 5-fold = **0.946** (sub 57126563) = s_mix1e 5-fold 0.946: gold +0.0047 not visible on public LB (+-0.004). Blend (57126661) pending.
- Oct 13 08:40 IST round-2 label test: 0.5 mm + 0.5 s_mm2 OOF = Qwen 0.9421 / Opus 0.9481 (vs ens5 0.9422 / 0.9488) -> no gain; label quality now bounded by the mm source. Queued Qwen --think gold test at the head (0050).
- Oct 13 09:40 IST s_mm2_full_s1 done (07:40). Submitted ours-only s_mm2 5 + full (submit27). Preempted stray s_mm1_384 f3 for the Qwen n5/think gold tests (n5 running).
- Oct 13 09:58 IST Qwen variants on gold58 (alone / prob-mix): greedy 0.889/0.942; n=5 self-consistency 0.890/0.941; --think 0.895/0.941 -> no gain; keep greedy labels.
- Oct 13 10:29 IST **ours-only s_mm2 5 + full = 0.947** (sub 57129815) = best ours-only (prev 0.946). Blend s_mm2 (57126661) pending.
- Oct 13 12:00 IST R7-only probe (heliosli R7, Raptor/OAI removed; slug rsna-v2-timing v4) submitted (57133494; commit run 7 min). 1 slot left today.
- Oct 13 14:32 IST **R7-only probe = 0.949** (57133494): heliosli R7 ConvNeXt stage alone (OAI-free) = our best blend
  level; the public 0.957 comes mostly from the OAI-trained Raptor half. Submitted (last slot today): BLEND R7 + s_mm2
  5-fold, rank w=0.5 (build.py --base=r7, slug rsna-v2-rehearsal, current rsna-v2-models version; submit29).
- Oct 13 14:40 IST R7 + s_mm2 blend submitted (commit run: R7 stage complete, v2 leg done, w=0.5). 0 slots left today.
- Oct 13 15:05 IST s_mm2_384 f0-1 within-fold gold24: 0.9434 vs s_mm2 0.9601 (pair 0.9548) -> 384 not better so far (noisy, 24 studies). Queued s_mm2_s2 5-fold (seed 52, 00590-00594); A30 moved from stale s_mm1_384 to s_mm2_384 f4.
- Oct 13 15:14 IST blend baseline + s_mm2 5-fold w=0.65 = **0.949** (57126661) = best (ties). R7 + s_mm2 blend (57137824) pending.
- Oct 13 17:39 IST **BLEND R7 (attributed, OAI-free) + s_mm2 5-fold, w=0.5 = 0.954 = NEW BEST** (57137824; +0.005 over
  0.949; R7 alone 0.949, s_mm2 alone 0.946). Two different, equally strong pipelines blend well. Oct 14 plan: R7 base
  + richer ours leg (s_mm2 5 + full; + s_mm2_s2 / 384 when ready), weight probes 0.4 / 0.6.
- Oct 13 18:15 IST s_mm2_384 f0-3 within-fold gold47: alone 0.9257 (< s_mm2 0.9350) but pair s_mm2+384 0.9367 (+0.0017) -> include in Oct 14 leg once f4 (A30, ~01:30) is done.
- Oct 14 07:05 IST overnight incidents: Slurm controller outage (sbatch "Transport endpoint is not connected" 05:38-06:48),
  workers 8376/8397 hit the 3-day limit, 8377 (A30) cancelled by NODE FAILURE at 03:51 (s_mm2_384 f4 lost mid-run ->
  requeued; s_mm2_s2 f1-f4 requeued). Session was blocked ~23:40-07:00 so the 05:41 one-shot never fired. Workers
  8377/8510/8511 start 07:02 (extra watchdog jobs 8513-8515 cancelled). s_mm2_full_s2 + s_mm2_s2 f0 finished.
  Submitted: R7 + 11 s_mm2 models (5-fold + full s1/s2 + 384 f0-3) w=0.5 (submit30); chained w=0.6 probe (submit31).
- Oct 14 07:28 IST both submitted (submit30 w=0.5, submit31 w=0.6). 3 slots left. Workers 8511 (NODE3) + 8377 (NODE2) running, 8510 pending.
- Oct 14 12:05 IST **R7 + 11 s_mm2 models = 0.955 at both w=0.5 (57162957) and w=0.6 (57163247) = NEW BEST** (+0.001
  over R7 + s_mm2 5-fold 0.954; within noise). Duplicate run found: the watchdog had requeued s_mm2_384 f4 while 8377
  (A30) was still training it, so 8510 (A100) ran it too, writing to the same dir -> killed the A30 copy (A100 ETA ~14:00);
  A30 now on s_mm2_s2 f2. Public0946 + R7 3-way rejected (public0946 alone >5.5 h -> 9 h risk). Submitted weight-curve
  probes w=0.4 (submit32, slug rsna-v2-rehearsal) and w=0.7 (submit33, slug rsna-v2-blend), same rsna-v2-models version.
  Last slot today: + s_mm2_384 f4 (12 models) AFTER submits 32/33 score (re-release of rsna-v2-models must wait).
  Research sweep missed at 09:23 (session busy) -> done 12:00; hourly cron now catches up sweeps older than 13 h.
- Oct 14 12:07 IST watchdog fix: squeue failures during controller hiccups were read as "job gone" -> live tasks
  requeued (s_mm2_384 f4 ran twice; s_mm2_s2 f1 finished at 11:39 and was re-run from scratch, overwriting the finished
  checkpoint; rerun ETA ~15:40). Watchdog now takes one job list per pass and skips the pass on squeue failure (restarted).
- Oct 14 12:07 IST R7 provenance: its OrthoFoundation-L init is an MIT-licensed public foundation model pretrained partly
  on OAI (see research.md). Treat as allowed pre-trained model (rules allow public pre-trained models); finals plan:
  #1 R7 + ours (best LB), #2 hedge with no OAI lineage at all (ours-only) -> flagged to user for the final call.
- Oct 14 12:11 IST probes submitted: w=0.4 (57170830) and w=0.7 (57170856). 1 slot left (12-model leg after these score).
- Oct 14 15:30 IST s_mm2_384 5-fold complete (gold58 within-fold): alone 0.9252 vs s_mm2 0.9318, pair 0.9335 (+0.0017) -> include all 5 folds in the 13-model leg. s_mm2_s2 f1 rerun done; f2 (A30), f3/f4 (A100) running.
- Oct 14 15:47 IST weight curve (R7 + 11 s_mm2): w=0.4 0.955 (57170830), 0.5 0.955, 0.6 0.955, 0.7 0.954 (57170856) ->
  flat plateau 0.4-0.6; keep w=0.5. Submitting last slot: R7 + 13 models (s_mm2 5-fold + full s1/s2 + 384 5-fold +
  s_mm2_s2 f0), w=0.5 (submit34, re-release of rsna-v2-models; nothing pending).
- Oct 14 16:28 IST R7 + 13 models w=0.5 submitted (submit34). 0 slots left today; do not re-release rsna-v2-models until it scores.
- Oct 14 17:30 IST queued s_mm2_dinob 5-fold (00600-00604, DINOv2-B 280, stud_mm2) behind s_mm2_s2 f2-f4 for leg diversity.
- Oct 14 19:58 IST R7 + 13 models w=0.5 = **0.955** (57179044) = ties 11-model 0.955; public LB saturated at 0.955 for this blend family (private decides). Oct 15 plan: add s_mm2_s2 5-fold + s_mm2_dinob when done; hedge final = ours-only (no OAI lineage).
- Oct 14 19:59 IST detached scripts/v2/oct15_submit.sh (pid 484817 after restart) fires 05:36 IST Oct 15: ours-only hedge (submit35) + R7 blend w=0.5 (submit36) with all finished s_mm2 / s_mm2_384 / s_mm2_s2 / s_mm2_dinob folds.
- Oct 14 20:00 IST s_mm2_dinob f0-1 gold24: alone 0.9348, mm2+dinob 0.9540 vs mm2 0.9601 / mm2+384 0.9548 -> weak so far; oct15_submit.sh now gates dinob via scripts/v2/gate_family.py (include only if gold58 within-fold base s_mm2+384 not lowered; currently 0.9366 < 0.9410 on 35 -> excluded).
- Oct 14 20:01 IST queued s_mm2_full_s3 (00610, seed 48) + s_mm2_384_full_s1 (00611, seed 49) behind dinob f4 -> ready ~Oct 15 morning/midday for later Oct 15 subs (not in the 05:36 script).
- Oct 14 21:28 IST s_mm2_s2 5-fold done. Gold58 within-fold: mm2 0.9318, s2 0.9282, mm2+s2 0.9350, mm2+384 0.9335, (mm2+s2)/2+384 0.9326 -> all within +-0.002 (noise); keep s2 in the leg (seed ensembling), dinob gated. dinob f2-f4 running.
- Oct 14 21:52 IST s_mm2_dinob folds 0-3 (gold46 within-fold): alone 0.9174; mm2+384 0.9367 -> +dinob 0.9318; mm2+384+s2 0.9369 -> +dinob 0.9324 => DINOv2-B lowers the leg by ~0.005 -> excluded (gate in oct15_submit.sh agrees). Full-data s3 / 384_full_s1 now training on the A100s; dinob f4 on A30.
- Oct 15 01:28 IST s_mm2_full_s3 done (10 ep). oct15_submit.sh restarted (pid 775423) to include finished full-data models (s3 now; 384_full_s1 if done by 05:36). Dry run: 18 models (s_mm2 5 + 384 5 + s2 5 + full s1/s2/s3), dinob gated out.
- Oct 15 02:28 IST 384_full_s1 + dinob f4 done -> 05:36 subs will carry 19 models. Queue was empty: queued s_mm2_384_s2 5-fold (00620-00624, seed 54). Worker 8511 idled out; watchdog refills to 3.
- Oct 15 05:57 IST morning subs in: R7 + 19 models w=0.5 (57199718) and OURS-ONLY hedge 19 models (57199886). 3 slots left. Do not re-release rsna-v2-models or rsna-v2-models-only until these score.
- Oct 15 05:58 IST heliosli S6 (compact-v1) as third arm: in rjings' public blend R7+S6 vs R7 alone shows no visible gain (0.956 both, with Raptor); same code/family as R7 -> skipped. Holding 3 slots until 57199718/57199886 score.
- Oct 15 08:28 IST **OURS-ONLY 19 models = 0.948** (57199886) = best ours-only (prev 0.947); hedge final candidate (no OAI lineage). Blend 57199718 pending.
- Oct 15 08:45 IST user: speed up; Opus relabel approved capped ~1,500 studies, then switched to HAIKU subagents. Pilot: Haiku on gold58 (mm/haiku-gold, 4 agents) to compare vs Qwen 0.942 / Opus 0.949 (prob-mix). Prepared mm/haiku-all: 1,500 Qwen-labelled studies with largest |Qwen - OOF| (60 batches x 25). Cluster full (other user 3/4 NODE3 GPUs + 5 pending) -> no extra GPUs.
- Oct 15 09:23 IST Haiku pilot gold58: alone 0.8874, prob-mix 0.9364 (< Qwen 0.9422); 0.5 Qwen + 0.5 Haiku prob-mix 0.9447. Started Haiku relabel (12/agent, 125 batches) then USER STOPPED ALL SUBAGENTS (4 stopped, no outputs). Haiku relabel on hold.
- Oct 15 09:30 IST free labels: gold58 prob-mix Qwen 0.9422, Gemma 0.9359, Qwen+Gemma avg 0.9443, Qwen+Haiku 0.9447, Q+G+H 0.9452 -> queued local Gemma-4-31B on the 3,629 non-Opus studies (0001_mm_gemma4_all_s0/s1, guarded to 80 GB GPUs; ~2 h on one A100 from ~10:30). Then labels stud_mm3 = 0.5*(avg Qwen,Gemma | Opus) + 0.5 ens5 -> retrain s_mm3.
- Oct 15 09:45 IST HF login OK; downloading google/medgemma-27b-it (best MedGemma multimodal) to ~/models/hf; gold pilot task 0000_mm_medgemma_gold queued automatically when download completes (80 GB guard), ahead of Gemma all-studies shards.
- Oct 15 10:25 IST R7 + 19 models w=0.5 = **0.955** (57199718) = tie with 11/13 models; public LB saturated for this family. 3 slots left today; holding (no new information to test until MedGemma/Gemma labels -> s_mm3).
- Oct 15 10:36 IST MedGemma-27B-it gold58: alone 0.8352, prob-mix 0.9300 (< Gemma-4 0.9368 < Qwen 0.9422); Q+MedGemma 0.9440, Q+G+MedGemma 0.9444 vs Q+G 0.9443 -> no gain, dropped. Gemma-4 all-studies labelling resumed (s0 512 done).
- Oct 15 10:37 IST detached scripts/v2/chain_mm3.sh (pid 1315817): when both Gemma shards are done -> mm-qg-all.jsonl (avg Qwen+Gemma) -> labels/stud_mm3.csv (Opus overrides; 0.5 mm + 0.5 ens5 OOF) -> queues s_mm3 5-fold (00020-00024) at the queue head. ETA labels ~12:30, s_mm3 folds by ~Oct 16 midday on 1-2 A100s.
- Oct 15 12:53 IST stud_mm3 built (Gemma averaged into 3,567/3,629 Qwen studies; Opus overrides; 100% mm mix). s_mm3 5-fold queued at head; preempted 384_s2 f4 on 8510 (had just started) -> s_mm3 f0 running on A100. 384_s2 f3 on A30. ETA s_mm3: ~4 h/fold on one A100 -> all 5 by ~Oct 16 09:00 (sooner if more GPUs free).
- Oct 15 15:29 IST user asked about submissions: launched 2 of 3 remaining slots: ours-only 22 models (+384_s2 f0-2; submit37) and R7 + 22 w=0.5 (submit38). 1 slot kept for tonight.
- Oct 15 15:48 IST submitted: ours-only 22 (57226091), R7 + 22 w=0.5 (57226300). 1 slot left; both model datasets locked until these score.
- Oct 15 15:49 IST s_mm3 f0 done: gold12 within-fold 0.9606 vs s_mm2 0.9686 (pair 0.9629) -- 12 studies, too noisy; weak AUC not comparable (different label targets). Decide after f0-f2. Last slot today not spent on s_mm3.
