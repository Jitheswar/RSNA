# Image-learning options under scarce queued GPU access

Research date: 2026-09-25. Decision support for **Establish image-learning options for scarce queued GPU time**. This report proposes experiments; it does not authorize weights, downloads, jobs, or a candidate change. No weights or competition data were downloaded and no training or tests were run.

## Recommendation

Diagnose whether the current model can learn the training images before buying architectural complexity with queue time. Then compare one small pretrained convolutional backbone against one frozen self-supervised backbone using the same admissible supervision and fold exclusions. Preserve plane identity and retain a simple pooling comparator. Adjacent-slice 2.5D is a subsequent controlled input change, not another independently pretrained resource.

This is a shortlist for an initial decision, not a claim that these models can win. Zero spending does not establish how many GPU hours will become available. No local speed, memory, AUC gain, or eligibility claim for these resources has been measured.

## What the existing evidence actually tests

The current [model](../../src/rsna_knee/model.py) is three randomly initialized convolutional layers with widths 8/16/32, spatial average pooling, masked mean pooling over slices, and another mean over series. Thus series plane/sequence identity is not explicitly retained by the prediction head. [Training defaults](../../src/rsna_knee/training.py) use 128×128 grayscale, up to 16 series and eight slices per series, two studies per batch, 100 successful Adam updates at 0.001, a constant learning rate, no augmentation, and the final checkpoint. Nominally 200 study presentations across 38–39 training studies are roughly five passes, subject to batching/masking; this is an exposure calculation, not measured convergence.

[Corrected OOF evidence](../evidence/ticket-10-oof.md) reports 0.464313 macro AUC over 58 studies (95% interval 0.404–0.524), versus 0.474353 from fold training prevalence alone. Within-fold permutation p=0.918 and the near-constant predictions do not establish usable image signal. The [wider variant](../../src/rsna_knee/image_variant.py) only changes channels to 16/32/64; its [comparison](../../.scratch/rsna-2026-fresh/issues/19-evaluate-one-stronger-image-model-variant.md) is +0.0066 with interval −0.0295 to +0.0442. This does not test pretraining, meaningful convergence, higher resolution, anatomical context, or plane-aware aggregation. It also does not prove a coding defect.

## A small concrete shortlist

| Route | Exact resource and evidence | Proposed adaptation and queue tradeoff |
| --- | --- | --- |
| First trainable comparator: ResNet-18, 2D slices, plane-aware pooling | `torchvision.models.ResNet18_Weights.IMAGENET1K_V1`, official file `resnet18-f37072fd.pth`. Official metadata: 11,689,512 parameters, 1.81 GFLOPS per documented image, 44.7 MB file; ImageNet top-1 69.758%. [Model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), [weight definition](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/resnet.html). | Repeat a normalized grayscale slice into RGB; remove classification head; initially freeze encoder and pool features per series and plane, then train twelve outputs with explicit missing-plane masks. If a useful signal exists, test partial fine-tuning with a small encoder learning rate. Keep mean pooling as comparator before testing max/attention. Feature extraction is reusable when encoder/preprocessing are fixed; fine-tuning is not. |
| Controlled spatial-context variant: same ResNet-18, 2.5D triplets | Same exact ImageNet weights; no extra external resource. Adjacent-slice channels are a proposed experiment, not an established result from the cited 2D sources. | Supply preceding/central/following slices as three channels, all from one geometrically ordered series; freeze edge handling and physical-spacing policy. Never join across series or pretend three arbitrary sampled slices are adjacent. Maintains the encoder's channel count and approximately its per-triplet compute; decoding/cache reads and triplet membership change. Evaluate only after the simpler comparator learns. |
| Queue-efficient representation comparator: frozen DINOv2 ViT-S/14 without registers | Official `dinov2_vits14` backbone, approximately 21M parameters; official README reports ImageNet linear evaluation 81.1%, not knee MRI performance. Public alternative representation: `facebook/dinov2-small`, repository revision `169e7c34622d7a7c466d7d9aabcec09c35c9a91e`, `pytorch_model.bin` listed as 88.3 MB. [Official repository](https://github.com/facebookresearch/dinov2), [versioned file listing](https://huggingface.co/facebook/dinov2-small/tree/169e7c34622d7a7c466d7d9aabcec09c35c9a91e). | Freeze the backbone, extract 384-dimensional class-token features, train a small fold-specific pooling/head model on CPU. Use one chosen distribution/API, not a mixture of hub and converted artifacts. The [official model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md) documents 14-pixel patches and supports simple downstream classifiers. Test 224×224 first (a multiple of 14); higher resolution is a later cost/quality decision. |

The ResNet checkpoint source is [the exact public weights URL](https://download.pytorch.org/models/resnet18-f37072fd.pth). This link identifies the proposed artifact; it was not downloaded or hash-verified. The filename contains a hash prefix, not a complete recorded SHA-256. The DINOv2 revision identifies a particular public repository snapshot, but a complete local weight digest and runtime compatibility still require later authorized admission.

### Why preserve planes

MRNet is a useful architecture precedent: ImageNet-initialized AlexNet encodes slices, max pooling combines them, and separate sagittal/coronal/axial models feed a combination model. On its 120-exam internal validation set, the paper reports AUCs 0.937 for abnormality, 0.965 for ACL tear, and 0.847 for meniscal tear. It used 1,370 total exams, different targets and expert reference labeling; these values cannot predict twelve-target performance on 58 competition reference studies. The paper also reports an external ACL AUC of 0.824 before retraining, illustrating a domain shift. [Original MRNet paper](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699).

The practical hypothesis is to share a backbone and preserve plane/sequence descriptors in a small masked head, rather than immediately train nine separate models. Do not blindly apply RGB center cropping to knees: retain the full knee field with a documented resize/padding policy and treat any departure from pretrained transforms as part of the experiment. Public pretrained preprocessing is an initialization convention, not evidence that a specific knee crop is correct.

### Resource and eligibility boundaries

DINOv2's official repository and model card state Apache 2.0 for the ordinary DINOv2 code/weights. This does not automatically cover newer X-ray/other specialized releases in the same repository. Torchvision explicitly warns that pretrained models can carry terms derived from their training data; a code license alone is insufficient. [Torchvision weight terms and preprocessing guidance](https://docs.pytorch.org/vision/stable/models.html), [DINOv2 release](https://github.com/facebookresearch/dinov2/blob/main/README.md).

Neither resource has passed this project's exact competition eligibility, provenance, availability-cutoff, redistribution, or winner-delivery review. No clinical reviewer is available; visual checks can establish image plausibility, sampling and orientation, but cannot establish new diagnostic labels. MRI-specific checkpoints or MRNet datasets are not assumed freely transferable merely because a paper is public. An exact medical checkpoint with verified rights and relevant pretraining could change the shortlist; none was admitted in this bounded review.

## Diagnostics before expensive scaling

These are proposed execution gates, not work performed during research:

1. **Audit the signal path on real training studies.** Trace target names/order, study identity, normalized pixels, nonzero valid masks, accepted series, slice positions, and labels through loss computation. Inspect aggregate range/variance and image thumbnails locally. Check train/eval preprocessing alignment and actual sampling coverage; cached decoder success alone does not establish an informative input.
2. **Establish optimization competence on a tiny training-only subset.** Fix a few diverse gold training studies, disable stochastic augmentation, and check that a bounded run can markedly reduce supervised loss and separate their labels. Record per-layer gradients and parameter changes, logits, valid-target counts, and BCE versus the training-prevalence intercept. Failure should lead to diagnosis of data/loss/optimizer/masking; success only demonstrates memorization, not generalization. Do not consume held-out studies in this diagnostic.
3. **Check budget and pooling separately.** Compare bounded, predeclared exposure budgets using learning curves and matched scientific identity; 100 steps is not a convergence criterion. Keep the held-out fold for evaluation rather than repeatedly choosing the best checkpoint from it. Define any tuning split or fixed budget before the comparison, acknowledging the tiny support.
4. **Compare the two pretrained representations under one label policy.** Frozen features offer a relatively cheap first comparison. Fit any feature normalization, pooling, dimensionality reduction, head, and thresholds exclusively in each training fold. A deterministic, externally frozen encoder may process held-out images for prediction, but no adaptation may see them. Self-supervised fitting on competition images must also exclude the held-out fold under the existing project contract.
5. **Demand useful predictive evidence.** Report paired gold OOF changes, per-target support, fold variation, study bootstrap uncertainty, prevalence/constant controls, and input-shuffling/ablating diagnostics interpreted at the appropriate split. Predeclare the learning-success criterion with the owner; the present report does not silently replace existing gates. A training-only fit or tiny positive CV delta cannot warrant a leaderboard claim.

Repeated use of all 58 references remains adaptive development. Report supervision can expand training exposure only under the separate admission and leakage policy; it cannot make weak labels equivalent to reference truth. Image and supervision comparisons need matched gold exposure so the mechanism of a change is interpretable.

## Quantify the unknowns before requesting GPU time

For a proposed 224×224 RGB batch of two studies, three selected series per study and sixteen sampled slices per series, the raw float32 image tensor is `2×3×16×3×224×224×4 = 57,802,752` bytes (about 55.1 MiB). This excludes activations, gradients, optimizer state, decoder buffers, and workspace. Full current maxima of sixteen series would multiply that input volume by 16/3. Shape arithmetic is not a VRAM measurement.

For frozen features stored per slice as float32, 4,407 studies × three series × sixteen slices × 384 dimensions × four bytes is about **325 MB decimal**. At 512 dimensions it is about **433 MB**. Both are hypothetical selected-series budgets, not actual corpus counts; identifiers, manifests, alternate crops and additional feature sets add storage. Patch-token features are much larger and should not be cached by default. Preprocessing/encoder/version identity must be part of the cache key.

ResNet's documented 1.81 GFLOPS per 224-pixel image implies roughly 87 GFLOPS for 48 selected slices per study before heads; it does not yield wall-clock time or training cost on this cluster. Frozen features can reuse a single extraction pass across fold-specific heads only while the extractor is externally fixed. Encoder fine-tuning requires separate fold checkpoints and new features. Three separately fine-tuned folds approximately triple encoder inference versus one checkpoint; measure rather than assume the full notebook still fits.

A concrete resource proposal should measure one approved representative forward/backward step, maximum allocated/reserved VRAM, decoder and feature throughput, long-tail series counts, checkpoint size, and full CSV pipeline time. Record actual GPU model/share and queue wait separately from run time. Use these observations to estimate fold-hours and inference seconds for the selected number of slices and folds. No whole-corpus training duration or A40 memory fit is currently verified for either resource.

## Decisions still needed

- Whether to admit ResNet-18, DINOv2-small, or both after exact resource review.
- The bounded diagnostic budget, what constitutes a successful learning check, and the GPU allocation that is realistically available.
- Supervision coverage and reliability, independently of model capacity.
- Fixed slice/series/plane policy, whether pretrained features are sufficient, and when fine-tuning or 2.5D earns its cost.
- How to freeze candidate selection under repeated CV and sparse labels; this research does not select a final model.

Defer full 3D training, large transformers, domain self-supervised pretraining and ensembles until an informative single-model result and measured budget justify them. This is an opportunity-cost recommendation under the owner's constraints, not evidence those approaches cannot help.
