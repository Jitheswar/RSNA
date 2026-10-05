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
