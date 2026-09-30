# Establish image-learning options for scarce queued GPU time

Parent: ../map.md
Type: research
Labels: wayfinder:research
Status: resolved
Assignee: baseline_facts
Blocked by: none

Research branch: research/rsna-image-learning-options
Research worktree: /tmp/rsna-wayfinder-image-learning-options

## Question

Which freely available representations and MRI study architectures merit evaluation when the existing 100-step compact baseline shows no measurable image signal, GPU access is queued, and external spending is zero? Compare a small number of concrete 2D/2.5D/multiplanar/pretraining approaches using primary papers and official model documentation. Distinguish source benchmark results from expected competition results, identify exact weights and licensing unknowns, and recommend learning diagnostics that precede expensive architecture scaling. Do not download weights or launch experiments.

## Answer

Resolved 2026-09-25 by the research agent. The previous wider model only increased tiny CNN channel widths; it did not test transfer learning. Diagnose real-input integrity, training-only memorization, gradient flow, prevalence dependence, and learning curves first. The bounded research shortlist is pretrained ResNet-18 with plane-aware pooling, a subsequent adjacent-slice 2.5D comparison using the same backbone, and frozen DINOv2 ViT-S/14 features as a representation comparator. These are proposed experiments, not selected resources or demonstrated competition improvements.

Asset: [Image-learning options under scarce queued GPU access](../../../docs/research/leaderboard-image-learning-options.md).

Context pointer: branch `research/rsna-image-learning-options`, commit `a11dcef85527b37a924c074ef554e64a427579e0`, report `docs/research/leaderboard-image-learning-options.md`. Exact competition eligibility and actual VRAM, throughput, queue availability, and full inference runtime remain unverified. No weights were downloaded and no training ran.
