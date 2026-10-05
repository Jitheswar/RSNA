# Research log — RSNA 2026 knee (v2)

Sweeps: forum, public notebooks, LB / efficiency LB, papers & docs. Newest at the bottom. Each entry: findings ->
actions (with status).

## 2026-10-05 12:15 IST — forum + notebooks scan
- Efficiency Prize track ($7k/$6k/$5k): Efficiency = AUC/(Benchmark - maxAUC) + RuntimeSeconds/32400, minimise
  (Benchmark = sample_submission ~0.5, maxAUC = best private AUC ~0.97) -> ~ -2.1*AUC + hours/9; 1 h ~ 0.05 AUC.
  Eligible only if the submission is one of the team's selected finals. Leaderboard (public, daily):
  kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb (#1 Scott Willis 0.959 on Oct 4 snapshot).
  -> ACTION: make a fast ours-only submission one of the two finals; optimise its runtime (below).
- Speed (topic 743374, Prateek Grover): fp16 weights + fp16 inputs without autocast 2.08x on T4; threaded DICOM reads
  (59 -> 372 MB/s with 16 threads); parse headers / read PixelData directly (5x faster than pydicom); percentile on a
  strided subsample. -> DONE pure fp16 (parity verified); TODO threaded I/O + direct DICOM parsing in prep.
- Single 5-fold models: ResNet50@224 0.954 (CoolinLai), DINOv2 0.942, EffNet-B4 0.940 (SpeedSci); crop 140 mm best
  (100 too small, 160 similar). Chris Deotte: private gold ~0.96, 1st ~0.975.
- Shake-up: public stack widely forked and called overfit -> keep an ours-only final.
- hengck23: show VLM the report + images to label ROIs (active labelling); others report 80-90% VLM accuracy.
  -> not pursued (cost/accuracy), revisit if labels plateau.
- goodpjw2008 notebook: stack 0.943 + own ConvNeXt-T 2.5D reader (0.929 alone) at 70/30 -> 0.944; notes synovitis
  labels are wrong when reports are silent (matches our analysis).

## 2026-10-05 12:35 IST — literature: self-training drift, report-derived labels
- Confirmation bias: students overfit teacher errors; gains saturate after 1-2 rounds; mitigations = re-initialise
  each round (we do), entropy/confidence-based pseudo-label selection, calibration, noised/larger students
  (Radhakrishnan et al., WACV 2024 "Design Choices for Enhancing Noisy Student Self-Training";
  arxiv.org/abs/2301.07294 "Enhancing Self-Training Methods"). Matches our round-2 result (same-architecture
  self-distillation: no gain, gold -0.004). -> ACTION (queued): round 2 from a 5-family ENSEMBLE teacher (s_mix1e);
  candidate next: stronger student noise (drop_path 0.2-0.3, intensity aug) or confidence-weighted mixing
  (more report weight where teacher entropy is high).
- LLM-extracted radiology labels: LLMs beat rule labellers (fracture sensitivity 95% vs CheXpert 51%); image models
  are fairly robust to label noise; hedging language should become soft/uncertain labels (GPT-4o MSK radiograph study,
  2026; MICCAI 2025 paper 0580; arxiv 2510.05664). Our grade->P(panel) calibration already encodes hedging.
Sources: openaccess.thecvf.com/content/WACV2024/html/Radhakrishnan_Design_Choices_for_Enhancing_Noisy_Student_Self-Training_WACV_2024_paper.html ;
arxiv.org/abs/2301.07294v1 ; papers.miccai.org/miccai-2025/0580-Paper3788.html ; arxiv.org/abs/2510.05664v1

## 2026-10-05 12:35 IST — docs: faster T4 inference (decision: how many models fit; efficiency track)
- TensorRT / Torch-TensorRT (ONNX -> trtexec FP16 engine; layer fusion, tensor cores) gives the largest T4 gains but
  needs TensorRT available offline on Kaggle (wheels in a dataset, ~GBs) -> heavy; keep as a later efficiency option.
  (docs.pytorch.org/TensorRT/_notebooks/Resnet50-example.html ; catalog.ngc.nvidia.com torch_tensorrt_notebooks)
- torch.compile: mode="max-autotune" / "reduce-overhead" use CUDA graphs (static shapes only); our encoder already
  sees fixed micro-chunks [48,3,320,320] -> compile the encoder only; Inductor/Triton supports T4 (sm75); compile time
  per model is the cost, partly amortised by the FX graph cache for same-architecture models.
  (huggingface.co/docs/transformers/v5.14.0/torch_compile ; docs.pytorch.org/docs/2.1/generated/torch.compile.html)
  -> ACTION: benchmark encoder compile (reduce-overhead vs max-autotune-no-cudagraphs) on the A100; if >=1.3x, test on
  Kaggle with the timing notebook.
- Pure fp16 (no autocast): implemented 2026-10-05 (forum claim 2.08x on T4); Kaggle timing v2 running.

## 2026-10-05 12:45 IST — papers: report supervision + knee MRI architectures
- "When are radiology reports useful for training medical image classifiers?" (arXiv 2510.24385v2, Oct 2025):
  image-text alignment pre-training helps diagnostic labels well represented in text, HURTS findings weakly
  represented in text; for noisy report-derived labels: avoid explicit alignment, combine SELF-SUPERVISED pre-training
  with auxiliary text objectives; for findings rarely mentioned in reports (our synovitis): prioritise SSL
  pre-training; distillation from an image+report teacher gives no advantage when text is highly predictive.
  -> IDEA (cost/benefit TBD): domain SSL (continue DINOv2 self-distillation on our ~820k knee slices, then fine-tune
  the DINO student) to help silent findings; our aux grade head already is an auxiliary text objective.
- CoPAS (Nature Communications 2024, doi 10.1038/s41467-024-51888-4; HKUST SmartX): 1,748 subjects, 12 knee
  abnormalities, Co-Plane Attention across MRI Sequences, decoupled per-sequence spatial features; beats junior
  radiologists. Our joint transformer over all series tokens (type + position embeddings, finding queries) already
  covers cross-plane/sequence attention; possible refinement: explicit plane-to-plane cross-attention blocks.
- ECM-Knee (HKUST + SMU, EClinicalMedicine 2025): multi-task system, 9 knee abnormalities.
Sources: arxiv.org/html/2510.24385v2 ; ideas.repec.org/a/nat/natcom/v15y2024i1d10.1038_s41467-024-51888-4.html ;
smartx.cse.ust.hk/2024/09/08/copas/ ; smartx.cse.ust.hk/2025/09/29/ecmknee/

## 2026-10-05 13:51 IST — measured: pure fp16 on Kaggle T4 = 1.12x (not 2x)
- Forum's 2.08x (CoAtNet) does not transfer to ConvNeXt-S: T4 time is dominated by 7x7 depthwise convolutions.
  torch.compile(max-autotune-no-cudagraphs) gave 1.41x on A100 (Triton depthwise conv kernels) but 9 min autotune.
  -> next speed options if needed: compile with autotune cache shipped in the package, fewer windows (14/8/7 = 0.7x
  cost, -0.0017 AUC), or ViT families (DINOv2 tensor-core friendly) for the efficiency track.

## 2026-10-05 13:58 IST — paper: Deep AUC Maximization (our metric is macro AUC; we train BCE)
- Yuan et al., ICCV 2021 "Large-scale Robust Deep AUC Maximization" (arXiv 2012.03173): AUC-margin (min-max) loss;
  two-stage = CE pre-training then AUC-M fine-tuning (PESG); 1st place Stanford CheXpert competition (report-derived
  noisy radiology labels — closest public analogue to our setting); beats CE "by a large margin". LibAUC library
  (docs.libauc.org; MultiLabelAUCM; margin bug fixed Feb 2024).
  -> ACTION: implement AUC-M fine-tuning stage in train.py (--init-ckpt from a trained student, --aucm-w, binary
  targets y>0.5 for the AUC term + BCE on soft labels), fold-0 test (~40 min).
Sources: openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html ;
github.com/Optimization-AI/LibAUC ; arxiv.org/pdf/2306.03065

## 2026-10-05 16:17 IST — result: Deep AUC Maximization does not transfer here
- s_mix1_aucm f0 (BCE + AUC-margin on y>0.5, 3 epochs from the trained student, lr 2e-5/1e-4): weak 0.9171 vs 0.9175,
  gold12 0.9557 vs 0.9580 -> NOGO (pre-registered rule). Hypotheses: soft-label noise (binarisation), small data
  (3.5k studies vs CheXpert 224k), BCE on calibrated soft labels already ranks well. Not pursued further.
