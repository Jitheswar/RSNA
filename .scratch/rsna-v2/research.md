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
