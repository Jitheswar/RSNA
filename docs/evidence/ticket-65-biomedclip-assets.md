# Ticket 65: pinned BiomedCLIP asset admission

The sole pinned checkpoint is locally verified and loads with PyTorch's
restricted tensor loader. Strict official model loading, original-MRI output
parity, throughput, downstream accuracy and final delivery remain untested.

## Resource and rules

Model `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, revision
`9f341de24bfb00180f1b847274256e9b65a3a32e`. Fresh anonymous API metadata
confirms public ungated access. The exact Microsoft [weight license](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/blob/9f341de24bfb00180f1b847274256e9b65a3a32e/LICENSE.md)
is MIT; retain its copyright/notice. The [model card](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/blob/9f341de24bfb00180f1b847274256e9b65a3a32e/README.md)
describes research use and places deployed uses outside its intended scope.
This private investigation establishes no final winner-delivery eligibility
and grants no third-party training-data rights.

October 2 browser rendering of the official [competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
retains external-model accessibility requirements and winner code/weights/
environment delivery obligations. Default-browser failure and current-browser
DOM-dump timeout remain preserved alongside the successful separate CDP
capture, 36,776 text characters. No rules acceptance occurred.

The campaign agent applied standing owner authority through
[the exact resource decision](../../.scratch/rsna-leaderboard/resource-approvals/ticket65-biomedclip-assets-20261002.json).
Plan SHA: `076e4f1a3c72e25852f9a129df4a91e30bba6390130e3bbada6f913488854a86`.
One attempt, 900-second watchdog, one CPU thread, 4 GiB observed peak RSS,
zero scheduler jobs, accelerators or spend. No new exact owner numeric approval
is claimed. Worker and reviewed evidence were frozen before acquisition.
Private root: `artifacts/private/ticket65-biomedclip-asset-admission-v1/`.

## Actual evidence

| Check | Observation |
| --- | --- |
| File | `open_clip_pytorch_model.bin`, 783,705,670 bytes |
| Full SHA-256 | `52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be` |
| Loader | `torch.load(map_location='cpu', weights_only=True)`, no unrestricted fallback |
| Structure | Flat dictionary, no `module.` prefix, 352 tensors |
| Vision namespace | 151 finite FP32 tensors, 344,767,488 bytes |
| Projection / positions | `[512,768]` / `[1,197,768]` |
| Elapsed / peak RSS | 28.470579 seconds / 1,169,944,576 bytes |
| Storage projection | 1,834,491,103,747 bytes including Ticket 64 reserves; limit 2,000,000,000,000 |

Independent `sha256sum`/`stat` and a separate verification receipt reconcile
complete bytes, digest, frozen evidence, resource bounds and observed vision
shapes. Receipt: `independent-asset-verification.json`, result SHA
`294c9daaee4a15e7984ff0bc2c6a4b39e7d1881990abcf0c64ee151ce5bae025`.
Six boundary tests cover digest/length mismatch and supported/rejected tensor
state mappings. No MRI/label read, dependency install, model construction,
fitting, publication or submission occurred; accelerator ledgers are unchanged.

Next is the separately admitted isolated runtime and fixed eight-study CPU
pilot from [the research recipe](../research/medical-image-encoder-options-2026-10-02.md),
respecting the two-job owner limit and Ticket 64's frozen science. Assets alone
establish neither predictive improvement nor first place.
