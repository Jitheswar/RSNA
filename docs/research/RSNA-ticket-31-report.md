# Ticket 31 report

**Prepared:** 2026-09-25. **Current disposition:** Ticket 31 is closed as a completed one-fold non-promotion. Ticket 32 completed the predeclared three-fold comparison and remains provisional/inconclusive; it was not promoted as a final candidate. The separate winner-release question remains unresolved and blocks final candidate freeze, winner publication, redistribution, and release. See TSK7 below; earlier sections preserve the preparation history.

## Synthetic lifecycle verification

I added `tests/test_ticket31_synthetic_lifecycle.py` because the existing suite did not exercise the Ticket 31 ResNet-18 fold path. It uses nine generated in-memory DICOM studies and the project's random fixture encoder, never the ImageNet checkpoint.

The test freezes three folds, trains fold 0 for one optimizer step (six synthetic training studies, three held-out studies), and verifies all of the following:

- Training writes a selected checkpoint and returns finite twelve-target held-out predictions for the three held-out studies.
- A newly constructed model reloads that checkpoint. Replaying raw held-out studies through the shared inference engine reproduces the cached-feature held-out scores to absolute tolerance `1e-6`.
- The run writes `offline-study-submission.csv` and `offline-study-evidence.json` for one held-out study. The CSV has the expected target header and one study row; its scores match that study's held-out prediction to absolute tolerance `1e-6`. The evidence marks the output as one study from one fold, not full OOF, and records `runtime_network_download: false`.
- The model identity is the random synthetic fixture encoder, its pretrained digest is null, and no external-resource approval or weight path is supplied.

Command and result:

```text
./.venv/bin/python -m pytest -q tests/test_ticket31_synthetic_lifecycle.py
1 passed in 5.46s
```

This establishes software behavior on synthetic data only. It says nothing about pretrained-weight rights, real-study behavior, model performance, or held-out generalization. Pytest retained its temporary outputs under `/tmp/pytest-of-ch.jitheswar/pytest-567/test_ticket31_synthetic_fold_r0`; these contain only generated studies, the random fixture checkpoint, and fixture features. No competition data, pretrained artifact, paid service, Slurm job, or GPU was used.

## Rights and eligibility review

**Reviewed:** 2026-09-25. **Scope:** the exact public TorchVision `ResNet18_Weights.IMAGENET1K_V1` pretrained artifact for the prize-bearing 2026 RSNA Knee Abnormality Detection competition. This review uses public official sources only. The checkpoint and ImageNet image data were not downloaded or used; no competition data was accessed for this review.

### Artifact identity and revision evidence

| Field | Evidence |
| --- | --- |
| Artifact ID | `torchvision.models.ResNet18_Weights.IMAGENET1K_V1` (not `DEFAULT`, even though current TorchVision documentation aliases `DEFAULT` to V1) |
| Artifact URL | [`https://download.pytorch.org/models/resnet18-f37072fd.pth`](https://download.pytorch.org/models/resnet18-f37072fd.pth) |
| Source revision | The project pins `pytorch/vision@v0.22.1`; the [tagged TorchVision source](https://github.com/pytorch/vision/blob/v0.22.1/torchvision/models/resnet.py) defines `ResNet18_Weights.IMAGENET1K_V1` and the artifact URL. Current TorchVision stable docs identify themselves as 0.29 and show the same enum and URL. The weight entry does not identify an immutable training-recipe commit; its recipe link points to the mutable `main` branch. |
| Official metadata | TorchVision documentation identifies 11,689,512 parameters, 1.81 GFLOPS, 44.7 MB, and ImageNet-1K top-1 accuracy 69.758%. The stable documentation identifies itself as TorchVision 0.29; inspected 2026-09-25. |
| SHA-256 | **The official weight entry exposes only the eight-character prefix** `f37072fd`. PyTorch's model-zoo documentation says a filename suffix used with `check_hash=True` may be the first eight or more digits of the file SHA-256. The repository's existing expected full digest is `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` (46,830,571 bytes), but it is not published by the official entry and was not independently verified here. The weight file was not fetched or hashed. |
| Weight-specific license | No separate license for this checkpoint is stated in its TorchVision model entry. TorchVision warns that pretrained models may have terms derived from their training datasets and makes permission for a use case the user's responsibility. The BSD license for TorchVision source code alone does not establish a license for this checkpoint. |

Sources: [TorchVision 0.29 ResNet-18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), [TorchVision 0.29 ResNet source and weight metadata](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/resnet.html), [TorchVision pretrained-weight guidance](https://docs.pytorch.org/vision/stable/models.html), and [PyTorch 2.14 model-zoo hash documentation](https://docs.pytorch.org/docs/2.14/model_zoo.html) (visible “Last Updated On: Jun 11, 2025”). The TorchVision docs show version 0.29 but no page revision date; the ImageNet-1K V1 identifier is the published weight version, not a source commit.

### What the current Kaggle material establishes

The [official competition overview/evaluation page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation), retrieved from Kaggle's official indexed rendering on 2026-09-25, says that freely and publicly available external data is allowed, including pretrained models. It also requires offline Notebook execution with internet disabled. The overview lists cash prizes and states that winners are asked to publish a link to their open-sourced code and weights on the competition forum and share the final model publicly for open distribution and validation. The page's indexed snapshot shows no precise last-revised timestamp; Kaggle says the organizers may update the timeline.

The [official competition Data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data) currently labels the competition dataset license “Subject to Competition Rules.” It does not independently grant or describe rights to ImageNet or this pretrained checkpoint.

The direct [official Competition Rules page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) was opened on 2026-09-25, but the page returned an empty client-rendered body in the available browser extraction. The latest full-rule review already recorded in this repository is dated 2026-09-11 in [the delivery-contract research record](../../.scratch/rsna-2026-fresh/delivery-contract.md#source-register-and-access-evidence). That dated record reports that external data must be public and equally available without cost, or meet host reasonableness criteria that consider cost, access restrictions and software/tool licensing. It also records a winner-license type of `CC-BY-NC 4.0`, while the associated winner-license language refers to OSI approval and not limiting commercial use. **The official-rule capture does contain a general exception for incompatible-license pretrained models:** it excuses a winner from granting the winner open-source license to such a pretrained model. This is a carve-out from the winner-licensing obligation; it does not itself grant rights beyond the model's own terms or resolve the remaining resource-access, cost, disclosure, or publication conditions. The capture does not name TorchVision's ImageNet ResNet-18 or say whether its ImageNet terms make this specific checkpoint an incompatible-license pretrained model. The record also flags an unresolved wording conflict and winner duties to provide code, weights and reproducibility material. **These are dated 2026-09-11 findings, not a fresh verification of today's full Rules page.** The current overview establishes a high-level allowance, but not every current competition-specific restriction or the exact current winner-license clause for this artifact.

The TorchVision URL and published 44.7 MB file size make this appear to be a public, non-fee-gated artifact that could satisfy the overview's surface public-availability criterion. A public URL does not by itself establish that every participant can lawfully use it on equal terms, that the host accepts its underlying restrictions, or that a released winner model may include it.

### ImageNet and TorchVision rights terms

ImageNet's [official download page](https://image-net.org/download-images), reviewed 2026-09-25, states that ImageNet does not own image copyrights and provides access to researchers and educators for non-commercial research or educational purposes under terms. Its [access agreement](https://image-net.org/accessagreement) states in clause 1 that the researcher will use the database only for non-commercial research and educational purposes. The agreement's clause 6 also says an employer that is a for-profit commercial entity is bound by the agreement when its researcher uses the database. The agreement has no visible version or revision date in the page retrieved. The agreement addresses the ImageNet database; it does not expressly explain the rights in a third-party provider's learned checkpoint or a downstream fine-tuned model.

TorchVision's guidance is directly relevant: it warns that library pretrained models “may have their own licenses or terms and conditions derived from the dataset used for training” and places responsibility on the user to determine permission for their use case. The exact ResNet-18 entry identifies the model as ImageNet-1K V1, but it does not state a checkpoint-specific license, whether ImageNet terms attach to the checkpoint, or whether public distribution of derived/fine-tuned weights is allowed. It also does not supply an immutable training-recipe revision or the full checkpoint hash.

### Eligibility conclusion and required resolution

**Disposition: potentially permitted for competition participation at the overview level, but not clearly established as eligible through the prize and winner-release path. Do not download or use this checkpoint yet.** This is an ambiguity finding, not a conclusion that the artifact is prohibited.

The evidence points in two directions that the available primary sources do not reconcile:

1. Kaggle's current overview permits freely public pretrained models and explicitly expects winning code and weights to be published.
2. ImageNet's access agreement limits database use to non-commercial research and education. TorchVision warns that terms from a model's training dataset may carry to pretrained weights, while this exact weight entry provides no separate license or distribution permission.

The sources do not say whether entering this cash-prize research-code competition is non-commercial research under ImageNet's terms; whether those terms govern this already-trained ResNet checkpoint or a model fine-tuned from it; or whether and on what license the required public winner model/weights may be distributed. **Finding on the competition exception: a general pretrained-model carve-out is recorded, but its application to this exact ImageNet checkpoint is not established.** The current Rules page was not readable in this retrieval, the earlier capture does not name this artifact, and neither TorchVision nor ImageNet states a checkpoint-specific license that would determine whether the carve-out applies. A legally meaningful conclusion cannot be inferred from the public download URL or from the TorchVision software license.

The official competition sources do not make host pre-clearance a universal requirement for every pretrained model. For this exact ImageNet-1K V1 artifact, a host answer is the narrowest way to clarify how the captured exception and public-release obligations apply if the project relies on that exception; Ticket 31's September 25 comment currently makes that clarification a local condition before the real fold. The answer should address competition training and the winner code/model/weight publication path, plus current external-resource equal-access/reasonableness and disclosure requirements. A host answer would not supply third-party rights or resolve the legal scope of ImageNet's agreement. Separately, the project requires an immutable artifact identity and a full SHA-256 before loading; the official weight entry exposes only the filename prefix, and the configured full digest has not been independently checked. No host inquiry was sent as part of this review.

### Approval-record status and proposed resource boundary

**No artifact-specific owner approval record was prepared.** The source/license questions remain unresolved, and the required full SHA-256 and immutable training/source revision are unavailable without an authorized later artifact check. Owner approval alone would not settle the host's interpretation of ImageNet and winner-publication terms.

For later scoping only, the candidate boundary would be **one public 44.7 MB checkpoint, $0 acquisition cost, no paid service, CPU-only execution within the project's one-fold scope, and no download/use of the underlying ImageNet images**. The proposed intended use is to initialize a single ResNet-18 knee-MRI experiment trained/evaluated only with the competition's permitted training data, and, only if separately cleared, to package the exact resulting weights for the competition's offline inference and required public winner release. This scope is descriptive, not authorization. No ImageNet checkpoint was transferred, loaded, used for training, or cached; no real competition study was processed; and no Kaggle upload or public release occurred. The synthetic run described above used only a temporary random fixture encoder and generated data.

## Alternative checkpoint review

**Reviewed:** 2026-09-25. **Disposition:** no replacement checkpoint found with both an explicitly compatible checkpoint license and resolved training-data rights. The closest general-purpose alternative is documented below as a lead for a later rights decision, not as an admitted or approved resource. No weights were downloaded or used.

### Closest license-layer fit: DINOv2 ViT-S/14

| Field | Finding |
| --- | --- |
| Model/artifact ID | Meta DINOv2 small vision transformer, `facebook/dinov2-small`, corresponding to the upstream `facebookresearch/dinov2` Torch Hub entry `dinov2_vits14` (ViT-S/14, without registers). The upstream checkpoint filename is `dinov2_vits14_pretrain.pth`. |
| Pinned model revision | Hugging Face model repo revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`; selected file `model.safetensors`. The checkpoint file was added in parent revision `150c8e7bb7cef2d30ec31b13a517af14840ee3f7`; the later pinned revision's model-card metadata says Apache-2.0. This pins the license statement and the exact file together. |
| Official URLs | [Meta's DINOv2 repo/model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md); [Meta's upstream ViT-S/14 checkpoint](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth); [Meta account's pinned Hugging Face model snapshot](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056). |
| Checkpoint SHA-256 | `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` (88,249,960 bytes), exposed in the publisher-hosted [file's Xet/LFS pointer metadata](https://huggingface.co/facebook/dinov2-small/blob/ed25f3a31f01632728cabb09d1542f84ab7b0056/model.safetensors). This is publisher-hosted metadata; the file was not fetched or hashed here. |
| Checkpoint license | Apache License 2.0. Meta's DINOv2 README and model card state that the standard DINOv2 code and model weights are Apache-2.0; the pinned Meta-account model snapshot also labels this file `apache-2.0`. Apache-2.0 permits commercial use, modification, and redistribution subject to its notice and attribution conditions, so the checkpoint-license layer is compatible in principle with an open public winner release. **This does not clear the training data.** The original Hugging Face file-addition snapshot had `cc-by-nc-4.0` metadata; the selected pinned revision is the later revision where the AI at Meta account changed that field to `apache-2.0`. |
| Availability/cost | The exact checkpoint is listed on a public Meta account model page and the upstream Meta repository links a direct public download. No fee or access request is indicated for this checkpoint. Kaggle's current overview allows freely and publicly available external data, including pretrained models, and asks winners to publish code and weights and make the final model public. Thus it meets the visible zero-cost/public-access criterion; this does not settle the training-data rights or every full-rule condition. |
| Training-data source and rights | Meta's model card identifies `LVD-142M`. The authors' [paper](https://arxiv.org/html/2304.07193) describes LVD-142M as curated retrieval from an uncurated repository of crawled web images, including URLs extracted from page image tags, together with images from named curated datasets including ImageNet-1k and ImageNet-22k. The official materials reviewed do not publish licenses, permissions, or a rights audit for the resulting LVD-142M images. **Training-data rights therefore remain unresolved; this checkpoint cannot be called cleared based on Apache-2.0 weight metadata alone.** |

### Preprocessing and technical fit

The pinned model processor specifies RGB conversion, rescaling byte pixels by `1/255`, bicubic resize with shortest edge 256, center crop to 224×224, and ImageNet channel normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) in its [preprocessor config](https://huggingface.co/facebook/dinov2-small/blob/ed25f3a31f01632728cabb09d1542f84ab7b0056/preprocessor_config.json). The model is a 21-million-parameter ViT-S/14 with a 384-dimensional embedding. At 224×224 it processes 256 image patches plus the class token; the architecture accepts larger image sizes when both dimensions are multiples of 14. For knee MRI, a proposed adaptation would convert each grayscale slice to three identical channels and apply the same rescaling/normalization, while resizing and padding the full knee field to a patch-compatible shape rather than center-cropping anatomy. That full-field adaptation changes the standard natural-image evaluation crop and must be frozen and evaluated as a new preprocessing identity.

This is technically plausible as a frozen slice-feature encoder followed by the existing plane-aware study head, and its self-supervised broad-image features could be informative with few reference labels. No cited benchmark establishes its performance on knee MRI or the competition's twelve findings. Compared with Ticket 31's 11.7-million-parameter ResNet-18, this ViT has about 1.8 times as many parameters, an 88.2 MB safetensors file versus ResNet's documented 44.7 MB, a different feature interface, and patch-token processing; expect higher encoder memory/compute and implementation/cache changes. Its 384-dimensional features are smaller than the ResNet's pooled 512-dimensional feature, but no end-to-end runtime or performance comparison has been measured. Any result would require a separate candidate identity and a paired, fold-matched experiment; no AUC gain is predicted here.

### Other near-miss: medical 3D ResNet-18

Tencent's official [MedicalNet repository](https://github.com/Tencent/MedicalNet) lists `resnet_18_23dataset.pth` and describes MedicalNet as MIT-licensed, but the repository does not identify a complete list or rights terms for the 23 pretraining datasets. The checkpoint is a 3D ResNet and would not be a drop-in replacement for Ticket 31's 2D slice encoder. Its code/project license and public weight link do not establish the training-data permissions. It therefore also fails the resolved-rights gate and is not recommended as a cleared substitute.

### Decision

**No alternative is presently cleared.** DINOv2-S is the closest public, zero-cost checkpoint found whose current publisher-stated checkpoint license supports open redistribution, but its LVD-142M source-image rights are not resolved in the primary materials. MedicalNet is closer to the ResNet family and medical domain, but its pretraining corpus rights are likewise not documented. Keep Ticket 31's artifact selection unchanged and do not download/use either resource until the data-rights and competition eligibility questions are resolved. No candidate change, real fold, or model release occurred during this review.

## TSK3: competition requirements and project approval boundary

**Reviewed:** 2026-09-25. This section separates the competition terms recorded in the official Kaggle materials from the stricter local approval gates for Ticket 31. It makes no legal determination.

### What the competition materials require

The current [Kaggle overview/evaluation page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation) explicitly permits freely and publicly available external data, including pretrained models, and requires offline Notebook execution. The current [Kaggle Rules page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) again returned an empty client-rendered body when checked on this date. The full-rule details below therefore rely on the dated September 11 capture in the [delivery-contract record](../../.scratch/rsna-2026-fresh/delivery-contract.md#notebook-and-runtime-constraints) and its [winner-license section](../../.scratch/rsna-2026-fresh/delivery-contract.md#code-open-source-and-winner-obligations), not a fresh extraction of today's Rules page.

That capture records conditions on external resources: they must be publicly and equally accessible at no cost, or meet the host's reasonableness criteria, which can consider cost, access restrictions, and software/tool licensing. It also records a winner-license exception for incompatible-license input data or pretrained models: the winner need not grant the winner's open-source license to that component. This is an exception to the winner-license grant; it is not itself a license to use or redistribute the component, and the capture separately records a requirement for winners to publish model weights as a public Kaggle dataset. The overview asks winners to publish code and weights and make the final model publicly available for open distribution and validation. The capture also records conflicting winner-license wording (`CC-BY-NC 4.0` alongside OSI/commercial-use language). These terms do not state that every pretrained model requires individual host pre-approval, but they also do not resolve whether TorchVision's exact ImageNet checkpoint qualifies for the exception or whether a restricted fine-tuned model can satisfy the public release request.

### What Ticket 31 adds as local gates

The campaign's [V13 verification register](../../.scratch/rsna-2026-fresh/spec.md#verification-register) is a project/owner requirement, not a quoted Kaggle pre-clearance rule. Before use, it requires an exact source/model/version/hash, license, competition eligibility, cutoff-relative public availability, intended purpose, leakage assessment, reproduction method, compute and execution/transfer policy, and approval for that model or tightly defined family/version. V13 explicitly disallows silently assuming a generic ImageNet exception. Ticket 31 adds its own acceptance gate to verify current rules, exact source/revision or artifact identity, full digest, rights, and specific resource approval before download or use; inaccessible eligibility remains a gate in the [issue criteria](../../.scratch/rsna-leaderboard/issues/31-run-one-resnet18-fold-through-complete-lifecycle.md#acceptance-criteria). Its September 25 comment additionally says written host/sponsor clarification is required before the real fold. That comment makes clarification a current local project condition, although the acceptance criteria and captured competition text do not make host clarification a universal prerequisite. Only the project owner can relax or replace that recorded condition.

Those gates exceed the competition overview's general allowance. They require owner approval for a specific resource; they do not, by themselves, require asking the host. Owner approval also cannot create third-party rights or decide the host's interpretation of its rules.

The earlier TSK2 alternative screen also used a stricter “resolved rights for the entire upstream training corpus” filter. Neither the captured Kaggle rules nor V13/Ticket 31 explicitly requires an independent commercial/public-release clearance for every pretraining image. That broad corpus audit is not a separate competition gate. The required review is exact-artifact and intended-use focused: identify the checkpoint's own stated terms and known training provenance, determine whether an express restriction applies to competition training or the planned model/weight release, meet Kaggle's public/equal-access/cost conditions, and disclose the resource. The upstream training source remains relevant when its terms expressly govern the checkpoint or derivative; lack of a full image-by-image provenance audit alone does not establish ineligibility.

### Narrow approval standard for TorchVision ResNet-18

For a **single local fold**, the narrowest supportable approval would be limited to the exact `torchvision.ResNet18_Weights.IMAGENET1K_V1` artifact and a frozen research scope: no ImageNet image downloads, no paid access, no held-out adaptation, and no use outside the approved competition experiment. Before that approval, the project still needs an authoritative rights basis for using these weights in that scope, a verified full digest and pinned artifact/source identity, and evidence that the current external-resource conditions are met. Disclose that the tagged TorchVision source does not pin an immutable training-recipe revision. The existing review records that TorchVision publishes no checkpoint-specific license, ImageNet's terms govern access to the database but do not address these learned weights, and TorchVision warns that model terms may derive from training data. The full digest in project code has not been independently checked against the artifact. On the captured evidence, the gate is not yet satisfied.

For a **prize-path candidate**, approval must additionally cover the actual release plan: whether the checkpoint or fine-tuned weights can be made available as required, and how the incompatible-pretrained-model exception interacts with those publication duties. The rule exception can excuse the winner-license grant for an incompatible model; it does not clearly waive the separately recorded public-weight delivery condition or grant permission to distribute weights. This is an inference from the two recorded clauses, not a legal guarantee.

### Is host clarification required?

No captured rule makes a host inquiry a categorical prerequisite for every public pretrained model. The explicit project prerequisite is the owner's specific resource approval. However, current public overview text alone cannot resolve the exact artifact's rights, and the dated Rules capture does not establish that ImageNet's terms make these weights an “incompatible-license” model or that the exception resolves public release. Therefore, if approval depends on relying on that exception, written host clarification is the narrowest way to settle the competition-rule interpretation for this exact checkpoint and proposed release path. A host answer would not replace a rights statement from the relevant weight/data source, since the host cannot supply third-party rights. If authoritative artifact terms independently establish the needed use and release permissions, a host inquiry may be unnecessary as a competition matter; the required project approval and exact-artifact checks would still remain, and the existing Ticket 31 comment would need an owner-authorized disposition before the fold proceeds.

### Draft host question

The existing Ticket 31 comment makes written clarification a local condition before the real fold. No inquiry has been sent. Suggested single question:

> Does RSNA/Kaggle consider the public, no-cost `torchvision.ResNet18_Weights.IMAGENET1K_V1` checkpoint eligible for competition training, and, if a team wins, may it publish the required fine-tuned model/weights under the incompatible-pretrained-model exception while identifying the checkpoint and preserving any applicable ImageNet/TorchVision terms?

This asks for the competition's interpretation. It does not ask the host to grant or guarantee rights in ImageNet material.

**Disposition:** keep the selected artifact blocked. The current evidence establishes neither the exact checkpoint's rights nor the full digest, and the current Rules page could not be extracted. Do not download/use the weights or start the real fold until the V13/Ticket 31 owner gates are met. No host inquiry was sent, no artifact or code was changed, and no real fold was run for this review.

## TSK4: official host interpretation and the ResNet-18 checkpoint

**Reviewed:** 2026-09-25. This section checks the exact TSK4 question against official competition pages, an official host response, the captured Rules text, and first-party ImageNet/TorchVision sources. The direct Rules page still returned an empty client-rendered body in browser extraction on this date; the full-rule clauses below remain attributed to the September 11 capture in the [delivery-contract record](../../.scratch/rsna-2026-fresh/delivery-contract.md#code-open-source-and-winner-obligations). This is not a legal opinion or guarantee.

### Competition-use eligibility

The current [Kaggle competition overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview) explicitly allows freely and publicly available external data, including pretrained models. The captured Rules text adds that external resources must be publicly and equally accessible at no cost or meet the host's reasonableness criteria, which can consider cost, access restrictions, and software/tool licensing ([captured resource terms](../../.scratch/rsna-2026-fresh/delivery-contract.md#notebook-and-runtime-constraints)).

There is a later, directly attributable host interpretation of how the competition treats non-commercial research terms. In the official Kaggle discussion [“Use of Commercially Hosted LLMs”](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965#3517696), host Po-Hao “Howard” Chen wrote on August 27, 2026 that, from the competition's perspective, using research-only datasets is non-commercial; prize money alone does not make it commercial; and a dataset is not excluded solely because its license is non-commercial. The same reply treats access as a separate condition: straightforward registration or click-through terms generally meet accessibility, while institution-specific approval, negotiated agreements, IRB approval, or lengthy credentialing may create a barrier. It also says teams remain responsible for whether their intended use—including prize-bearing participation and any required release of code or models—is allowed by the applicable data terms. The host's earlier [KneeCoT ruling](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109) applied that accessibility distinction to a dataset requiring a formal agreement and disallowed that dataset for the competition. The Kaggle discussion page returned an empty client-rendered body in browser extraction; the host reply was verified on September 25 through Kaggle's read-only public `GetForumTopicById` response for topic `733965`, which marked comment `3517696` as `authorType: HOST` and included the named author, timestamp, and reply text.

Applied narrowly, the host's response gives a **yes at the competition-policy level**: use for competition training is treated as non-commercial research, and neither prize money alone nor an NC-only restriction disqualifies a resource on that basis. The current overview permits free/public pretrained models; the public, no-cost TorchVision checkpoint also appears to meet the public-access and cost dimensions on its face. The answer does not amount to an artifact-specific host approval of `torchvision.ResNet18_Weights.IMAGENET1K_V1`, waive other accessibility/reasonableness rules, or grant rights in ImageNet/TorchVision materials. It does not settle whether the checkpoint's underlying terms permit this team's use or later release.

No directly applicable host ruling for this exact TorchVision checkpoint was found. A participant's official Kaggle topic [“Are CC-BY-NC pretrained weights compatible with the winners open-licence obligation?”](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735121), posted August 14 about RadImageNet weights, asks a closely related winner-license question but had no host response when checked on September 25. In another official topic, [“External data behind an institutional data use agreement - does the KneeCoT ruling generalize?”](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737950#3519419), a participant specifically asks whether a public checkpoint trained partly on inaccessible private clinical data is allowed; no host response to that checkpoint question was present in the public discussion record on September 25. Both are participant questions, not rules or rulings.

### ImageNet and TorchVision terms

The official [ImageNet download page](https://image-net.org/download-images) says ImageNet does not own the images' copyrights and offers image access to researchers and educators for non-commercial research and/or educational purposes under its terms. That page governs access to ImageNet images; it does not state that its image-access terms govern the TorchVision checkpoint or its fine-tuned derivatives.

The official [TorchVision ResNet-18 model entry](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) identifies `IMAGENET1K_V1`; the project's pinned [TorchVision v0.22.1 source](https://github.com/pytorch/vision/blob/v0.22.1/torchvision/models/resnet.py) gives the checkpoint URL `https://download.pytorch.org/models/resnet18-f37072fd.pth`. The reviewed weight metadata does not state a separate checkpoint license. TorchVision's [pretrained-weight guidance](https://docs.pytorch.org/vision/stable/models.html) warns that model terms may derive from training data and assigns the user responsibility to determine permission for their use case. The official materials therefore do not establish that this checkpoint is either unrestricted or expressly “incompatible-license” for purposes of Kaggle's exception.

### Winner publication and the incompatible-model exception

The September 11 captured Rules text records both (a) an exception from granting the winner license to incompatible-license input data or pretrained models and (b) a separate requirement to provide model weights as a public Kaggle dataset. The current overview also asks winners to publish their code and weights and share the final model publicly for open distribution and validation. The capture records conflicting winner-license wording (`CC-BY-NC 4.0` alongside OSI/commercial-use language), so it does not determine the exact scope of either obligation.

The narrow reading is that the exception addresses Kaggle's winner-license grant to the incompatible component; separately, Kaggle requires delivery/publication of the model weights and the overview requests a publicly available final model for open distribution and validation. The exception text does not say that it grants third-party rights, authorizes redistribution, or waives that release obligation. The host's August 27 reply addresses the team's responsibility under source terms; it does not grant permission or interpret ImageNet/TorchVision terms. **Inference, not an express host ruling:** even if the exception were applicable, it would not by itself establish permission to publish this exact fine-tuned checkpoint or satisfy Kaggle's separate public-release request.

No official source reviewed states that ImageNet's image terms apply to these learned weights, nor that they do not apply. TorchVision's general warning expressly leaves that use-case permission unresolved. Since the official TorchVision entry has no checkpoint-specific license statement, this record also cannot determine that the pretrained component qualifies as an “incompatible-license” model. Identifying the checkpoint and retaining applicable notices/terms would improve disclosure, but does not create use or redistribution rights.

### Finding and disposition

For **competition training**, the answer is **yes at the competition-policy level for non-commercial research use**: freely/publicly available pretrained models are allowed, and the host has said neither prize money alone nor non-commercial terms by themselves disqualify a resource. A free public checkpoint appears to avoid the institutional-access barrier that led to the KneeCoT ruling. There is no specific host ruling for this exact checkpoint, and the host's answer does not supply the source-side permission or the required project-specific resource approval; those remain open.

For **winner publication**, the answer remains **unresolved**. The incompatible-pretrained-model exception is not a source license and does not clearly waive the separate public-weight/model release requirements. No guarantee can be made that a fine-tuned `IMAGENET1K_V1` model may be published while preserving applicable ImageNet/TorchVision terms. The exact source terms, whether any terms govern derivative weights, and the hosts' interpretation of how the exception interacts with the public-release request all remain open.

The owner can resolve the competition-interpretation question by asking the hosts the draft TSK3 question; that answer cannot replace the missing checkpoint/data-source rights statement. Ticket 31 remains frozen pending its current local gates. No host was contacted, no checkpoint was downloaded or used, no implementation or issue was changed, and no real fold was run for this review.

## TSK5: separate experimental use from winner-release eligibility (historical proposal)

**Prepared:** 2026-09-25. This section converts the official competition overview and the host interpretation documented in TSK4 into a precise, owner-reviewable proposal for one Ticket 31 experiment. It does not make a legal determination or infer that the winner-release exception applies to this checkpoint.

### Proposed owner decision — pending

The exact record is [`.scratch/rsna-leaderboard/resource-approvals/ticket31-resnet18-imagenet1k-v1.json`](../../.scratch/rsna-leaderboard/resource-approvals/ticket31-resnet18-imagenet1k-v1.json). Its decision and rights status are deliberately pending; it is not accepted by the training loader and does not authorize acquisition. The requested decision is to admit this exact, zero-cost public checkpoint for the single CPU reference-fold experiment in Ticket 31, and for no other use. Approval would authorize one-time acquisition followed by local/offline processing only after the acquired bytes independently match the full expected SHA-256 and size below. It would not decide whether any source terms apply to derivative weights.

| Approval field | Exact proposed value |
| --- | --- |
| Artifact | `torchvision.ResNet18_Weights.IMAGENET1K_V1` (not `DEFAULT`) |
| Source/version | `pytorch/vision@v0.22.1`; tagged TorchVision ResNet source defines the enum and URL. No immutable training-recipe revision is provided by the weight entry. |
| Acquisition URL | `https://download.pytorch.org/models/resnet18-f37072fd.pth` |
| Expected size | 46,830,571 bytes |
| Expected full SHA-256 | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| Expected cost | USD 0; public URL; no paid service or external spend |
| Scope | `ticket31-one-reference-fold-cpu`; one pre-frozen reference fold only |
| Digest status | Expected digest is configured in the acquisition/loader checks; actual bytes have not been acquired or independently hashed. Record the observed full digest, byte count, and verification time after approved acquisition and before loading or training. |

The proposed use is to initialize the frozen ResNet-18 encoder for the one reference fold, training the Ticket 31 head only on that fold's training partition and using held-out studies only for prediction/evaluation. No gradient updates, fitting, calibration, selection, or other adaptation may use held-out inputs or outcomes. Do not download or use ImageNet images. After the single checkpoint transfer, feature extraction, training, evaluation, checkpoint reload, and offline inference stay local and offline. Do not cache, mirror, share, or reuse these weights outside this experiment without a separate approval.

The provenance and boundary recorded in the JSON include the ImageNet-1K V1 identity and known dataset provenance, the publisher/source revision, exact URL/size/hash, source links, and the known limits: TorchVision's entry states no checkpoint-specific license; TorchVision cautions that terms may derive from training data; ImageNet's access terms cover image access for non-commercial research/education but do not say whether they govern this checkpoint or resulting fine-tuned weights; and the training recipe is not pinned to an immutable revision. TSK4 documents the official Kaggle overview's allowance for free public pretrained models and host Po-Hao “Howard” Chen's August 27 interpretation that research-only use is non-commercial for competition purposes, prize money alone is not commercial use, and an NC restriction alone does not exclude a resource. That is competition-policy evidence, not a third-party rights grant or ruling on this exact artifact.

The approval text expressly recognizes the remaining source-term uncertainty and makes the owner decision only for the bounded experiment. It does **not** rely on the incompatible-pretrained-model exception to justify training, characterize that exception as a license, or treat release eligibility as cleared. This is a project authorization, not legal advice or a guarantee.

The September 25 Ticket 31 comment requested written host/sponsor clarification before the real fold. This proposal supersedes that local precondition for this bounded experimental use only: the already-documented written host reply #3517696 supplies the competition-policy interpretation used here, and no artifact-specific host ruling is claimed or required for this experiment if the owner approves the record. This does not relax V13 or any Ticket 31 acceptance criterion. A specific resolution of source rights and winner-release obligations remains necessary before final candidate freeze or publication.

### Separate final-candidate and release gate

Even if the owner approves this experimental-use record, `IMAGENET1K_V1` remains **ineligible for final candidate freeze, winner publication, model/weight redistribution, and any other release**. The exact rights in the checkpoint and derivatives remain unclear, and the interaction between Kaggle's incompatible-pretrained-model exception and the separate public-weight/winner-release obligations remains unresolved. Those questions require their own evidence and a separate owner decision before the artifact can enter a candidate or be published. A one-fold experiment cannot satisfy or waive that gate.

Ticket 31's original acceptance criterion requiring current rules, exact identity, full digest, rights, and specific resource approval remains unchanged and unchecked. V13 is unchanged. Two additional unchecked Ticket 31 criteria record the split: experimental-use approval is narrowly scoped and does not satisfy V13 or the existing criterion; final freeze/publication/release stays blocked until the distinct redistribution and winner-rule question is cleared. The issue remains `ready-for-agent`; that status authorizes preparation, not the pending resource decision.

### State at proposal time (superseded by TSK6)

The record is pending owner approval, `approved_by` and `approved_at` are empty, and acquisition verification is pending. No checkpoint or ImageNet image was downloaded; no real study was processed; and no real fold was run. Once the owner approves this exact record, the acquisition path must verify the observed full SHA-256 and byte count before model loading. Then run only the one real reference fold and record its outcome; keep the release gate closed. See the [updated Ticket 31 issue](../../.scratch/rsna-leaderboard/issues/31-run-one-resnet18-fold-through-complete-lifecycle.md).

## TSK6: approved one-fold CPU run and measured outcome

**Completed:** 2026-09-25. The owner approved the exact resource record for one Ticket 31 reference-fold CPU experiment only. The decision does not authorize reuse, expansion, final-candidate freeze, publication, redistribution, or release. The separate release gate remains blocked.

### Artifact, fold, and execution

The exact resource record is `.scratch/rsna-leaderboard/resource-approvals/ticket31-resnet18-imagenet1k-v1.json`; owner approval was recorded at `2026-09-25T11:12:47Z`. The exact `torchvision.ResNet18_Weights.IMAGENET1K_V1` artifact was acquired from the approved URL. Before loading, independent `sha256sum` and `stat` checks at `2026-09-25T11:17:03Z` recorded SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` and 46,830,571 bytes, matching the approved record. No ImageNet source images were downloaded, and no external spend occurred.

Fold 0 was frozen before held-out outcomes were inspected using the non-superseded v2 manifest (SHA-256 `8460234d0fcbfd2eb9ca3a3e4165929f50a86acac37d2cd03be1164c2f287fb8`). It contains 38 reference training studies and 20 held-out studies; report-derived labels were not used. The integration manifest contains 158 image studies and has SHA-256 `c0b11e22cd2575af7d964fbe3b8b12603ad0c6291a6a1705e0fb53fcfd957b94`; its source fingerprint is `25a70674e5af8808c4cf6d7e0da6c8b8f894e1856a11e064470ea0bcdab25648`. The run used the frozen ResNet-18 encoder, 224-pixel whole-field preprocessing, plane-stratified series-mean pooling, and the preregistered 100-step Adam head configuration (learning rate 0.001, batch size 2, seeds 20260911/20260912). Held-out inputs were used for prediction/evaluation only.

The first Slurm attempt, job `7167`, ran 9m13s and stopped before head training because the feature cache had been created inside the output directory that the runner requires to be empty. It materialized 58 identity-scoped entries, then exited `1:0`; it did not produce metrics or a trained checkpoint. The recorded peak RSS was 27,453,616,128 bytes (about 25.6 GiB of 64 GiB); the failure was the logged path-layout error, not memory exhaustion. The corrected same-fold invocation reused the matching cache and changed no scientific settings.

Slurm job `7168` completed with exit `0:0` in 7m22s on `cluster1-cpu` (`KLEF-C-NODE2`), using 16 CPUs and 64 GiB. No GPU was requested or used. Preflight passed. Training completed 100/100 optimizer steps with no skipped batches, 200 sample exposures and 2,400 supervised-entry exposures. Loss changed from `0.6966536045074463` to `0.5687010288238525`.

### Held-out result and offline output

Across the 20 held-out predictions, all 12 target AUCs were defined. Macro AUC was `0.4976806547210959`.

| Target | AUC | Target | AUC |
| --- | ---: | --- | ---: |
| ACL | 0.3956 | MCL | 0.4902 |
| Medial Meniscus | 0.5859 | Lateral Meniscus | 0.4141 |
| Medial OA | 0.6933 | Lateral OA | 0.3906 |
| PF OA | 0.5385 | Effusion | 0.3438 |
| Synovitis | 0.2727 | Baker's | 0.6094 |
| Contusion | 0.7143 | Fracture | 0.5238 |

The run ID is `8f0ed832459c05009bacee39b6f9b209d723e1b365c512ce452db93dff775460`. The final checkpoint is `artifacts/private/ticket31-resnet18/fold-0-complete/checkpoints/final.pt`, 45,615,081 bytes, SHA-256 `17220c47d15e786e870eb0d46aa088314973b7929094ede697585c148211e872`. The offline study output has one held-out row and 12 finite scores; its CSV SHA-256 is `8ea7ef398a445c0ab93da952d6b38cd197fbbe9e327dfb500291e80ebda2eac8`. Its evidence ties the output to the checkpoint digest and records five valid series, zero rejected series, no fallback, and `runtime_network_download: false`; its scope is explicitly one held-out study from one fold, not full OOF.

### Cache diagnostic correction and disposition

The cache identity matched the encoder digest, source fingerprint, integration-manifest checksum, fold-manifest checksum, preprocessing, packing, and sampling policy. There were 58 entries, 58 hits, and zero newly materialized entries in job `7168`: 38 training plus 20 held-out. The generated `frozen-feature-cache.json` incorrectly reports 58 hits in `partitions.training`; inspection of `src/rsna_knee/training.py` shows that this field is calculated after held-out lookups, so it includes the 20 held-out hits. The correct partition split is 38 training and 20 held-out. This is a diagnostic-counter bug; the total, features, fitting, predictions, and metrics are unchanged. The raw evidence is preserved and this correction is recorded in the final-freeze register and ticket.

This is one fold only. It does not establish full OOF performance, a competition gain, or candidate superiority. The run was not promoted. Final candidate freeze, winner publication, redistribution, and release remain blocked pending a separate source-rights and competition-rule decision. The private plan is `artifacts/private/ticket31-resnet18/fold-0-run-plan.json`; full run evidence is in `artifacts/private/ticket31-resnet18/fold-0-complete/`, with public-safe run summary at `fold-0-complete.public.json`.

## TSK7: Ticket 31 closure and Ticket 32 gate outcome

**Completed:** 2026-09-25. Ticket 31 is closed as the completed one-fold experiment it specified. Fold 0 produced macro AUC `0.4976806547` on 20 held-out studies. The same-fold corrected v2 mechanics baseline was `0.4922830053`, a one-fold delta of `+0.0053976494`. Ticket 31 is a **non-promotion** result; one fold is insufficient to promote or reject the ResNet-18 route.

### Predeclared gate

The [Ticket 05 validation policy](.scratch/rsna-leaderboard/issues/05-choose-validation-policy.md) says to complete planned folds before interpreting an improvement, that a tiny positive delta alone is insufficient, and that no universal numeric threshold should be invented. [Ticket 06](.scratch/rsna-leaderboard/issues/06-choose-first-learning-campaign.md) says a single-fold positive is insufficient for promotion and calls for comparable OOF evidence. Ticket 32 is the already-declared corrected three-fold comparison. The gate therefore explicitly required full-CV confirmation, so TSK7's condition for proceeding was met by the policy itself, not by Ticket 31's successful execution.

Ticket 32 used a separate approval at `.scratch/rsna-leaderboard/resource-approvals/ticket32-resnet18-imagenet1k-v1.json`; the Ticket 31 one-fold approval was not changed. The analysis plan was frozen before the three-fold outcomes in `artifacts/private/ticket32-resnet18/ticket32-run-plan.json`.

### Ticket 32 result

| Fold | Corrected v2 baseline | ResNet-18 | Delta | Held-out studies |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.492283 | 0.497681 | +0.005398 | 20 |
| 1 | 0.421119 | 0.601088 | +0.179968 | 19 |
| 2 | 0.488201 | 0.547983 | +0.059782 | 19 |
| **Pooled OOF** | **0.464313** | **0.518125** | **+0.053812** | **58** |

The paired 2,000-resample study bootstrap for candidate minus baseline was `-0.004524` to `0.115394` (1 undefined resample). The candidate's pooled OOF bootstrap interval was `0.471360` to `0.560698` (1 undefined); the baseline interval was `0.404344` to `0.523516` (1 undefined). The candidate's pair-weighted within-fold macro AUC was `0.546154` versus `0.467378` for the baseline. A 10,000-resample within-fold label-permutation diagnostic gave `p=0.039296`, which meets the predeclared measurable-signal rule. The fold-specific training-prevalence control was `0.474353`; the constant-score control was `0.500000`.

The image ablation zeroed frozen image features at inference while retaining the actual masks and descriptors and keeping each fitted head unchanged. Full ResNet-18 minus ablation was `+0.015748`, with a paired 95% interval of `-0.045430` to `0.077447` (1 undefined resample). Per-target effects were mixed: the largest gains were Medial OA (`+0.194`), Synovitis (`+0.140`) and Medial Meniscus (`+0.113`); Effusion (`-0.087`), Fracture (`-0.046`) and Lateral Meniscus (`-0.017`) declined. The full per-target supports and intervals are in `artifacts/private/ticket32-resnet18/public/comparison.md`.

**Decision:** ResNet-18 remains **provisional and inconclusive**. The pooled point estimate and within-fold signal are promising, but the paired candidate-minus-baseline interval includes zero, the candidate OOF interval includes `0.5`, and the image-ablation interval includes zero. The model is not promoted as a confirmed improvement or final candidate, and it is not rejected as unhelpful. No threshold was added after seeing the results. No additional supervision, representation or fine-tuning branch was started; Ticket 38 DINOv2 remains an unstarted option requiring its own scoped resource decision.

### Execution and limits

The three folds completed sequentially in Slurm job `7169` on `cluster1-cpu`, using 16 CPUs and 64 GiB, with no GPU and zero external spend. Each fold completed 100/100 optimizer steps with no skipped batches. The matching frozen-feature cache had 58 hits and zero new entries per fold; held-out cache hits were 20, 19 and 19. Checkpoint reload reproduced saved fold predictions within `5.96e-8`. The run's `partitions.training.cache_hits` field again says 58 because it is calculated after held-out cache lookup; the correct training/held-out counts are 38/20, 39/19 and 39/19. This reporting-counter bug did not affect fitting, features, predictions or metrics.

The first two analysis attempts (jobs `7170` and `7171`) stopped before image loading or metric computation due to an import path and the preserved baseline's older record schema. Neither changed model outputs or the baseline. Corrected analysis job `7173` validated the preserved baseline, completed checkpoint replay, OOF comparison and ablation, and wrote the detailed evidence under `artifacts/private/ticket32-resnet18/`.

This remains repeated development CV on 58 studies, not an untouched test or clinical-generalization estimate. Reliable patient grouping is unavailable in the admitted manifest, so uncertainty was resampled by study and repeated-patient leakage cannot be ruled out. Final-candidate freeze, publication, redistribution and release remain blocked by the separate checkpoint/derivative rights gate.
