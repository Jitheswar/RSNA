# Public inference frontier — 2026-09-30

## Decision

The most defensible next inference baseline is the fully public four-reader CoAtNet recipe in [yamadan96's corrected public notebook, script version 354057270](https://www.kaggle.com/code/yamadan96/rsna-knee-d4-public0946?scriptVersionId=354057270). Its publisher explicitly says they submitted this public pipeline on September 28 and obtained **0.943**, and corrects the `0946` URL: the parent's extra 0.003 required private ConvNeXt checkpoints. This is the publisher's reported submission result, not this account's result and not an independently retrieved submission record.

Our attributed 0.936-author baseline remains submission **56705536**; this research did not poll or change its scoring state. The earlier [baseline research](public-ensemble-route-2026-09-29.md) records its validated visible GPU execution and pending score. Neither route proves first place. The documented September 29 leader was 0.961.

Start with an exact named-asset CPU preflight for the corrected public notebook, including all expected manifest and checkpoint hashes. Only after the active scoring request is terminal and the campaign ledger is reconciled should a bounded two-T4 inference dry run be considered. No training, notebook push, GPU reservation or submission was performed for this research.

## Evidence and reproducibility

The authenticated Kaggle CLI query was:

```text
/home/ch.jitheswar/.venvs/kaggle-cli-2.2.4/bin/kaggle kernels list \
  --competition rsna-knee-abnormality-detection \
  --sort-by scoreDescending --page-size 30 --csv
```

The corresponding typed `kernels_list` result is saved at `artifacts/private/external-notebook-audit/frontier-2026-09-30/score-sorted-kernels.json`. This first-party API returns ordering, titles, owners and timestamps, **not numeric scores**. Thus position is evidence for which notebooks to investigate, not evidence of a reproducible score.

All downloaded source, authenticated source metadata, public HTML, dataset metadata and inventory, and small publisher manifests are retained under `artifacts/private/external-notebook-audit/frontier-2026-09-30/`. `artifact-manifest.json` fingerprints these artifacts. Downloaded notebooks were statically compiled cell by cell without execution; all code cells compiled. This verifies syntax only.

| Score-sorted position | Primary notebook | Source finding |
| --- | --- | --- |
| 1 | [pjmathematician D4 blend](https://www.kaggle.com/code/pjmathematician/rsna-knee-d4-blend?scriptVersionId=353440703) | Adds a private student fleet after the public recipe. Three dataset entries are empty in API metadata. |
| 2 | [pjmathematician D4 lite](https://www.kaggle.com/code/pjmathematician/rsna-knee-d4-lite?scriptVersionId=352859223) | Also depends on the author's student asset packages; its apparent public score cannot be carried over to the public prefix alone. |
| 3–4 | [Aastik S75 W50](https://www.kaggle.com/code/aastikrajan15/knee-s75-w50), [W60](https://www.kaggle.com/code/aastikrajan15/knee-s75-w60) | W50 embeds a second inference script using `aastikrajan15/knee-ckpt`, writes `ours_s10.csv`, and uses 50% of that extra arm. |
| 5 | [Evgendvorkin Versia 5](https://www.kaggle.com/code/evgendvorkin/rsna-versia-5) | Public DINO/A5/Rad/Raptor plus four CoAt readers; basis for Roman's variant. |
| 6 | [Jiwei Liu fast 2×T4, script version 352245350](https://www.kaggle.com/code/jiweiliu/rsna-knee-fast-2xt4-inference?scriptVersionId=352245350) | Downloaded v11 cites Mattia Angeli's 0.943 v34 recipe and includes explicit memory and runtime contracts. |
| 7 | [Mattia Angeli Speedy Raptors](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-speedy-raptors-the-original?scriptVersionId=351665011) | Scored v34 anchor cited by Jiwei; an authenticated attempt to download `/34` returned HTTP 403, so do not treat it as independently inspected source here. |
| 8 | [Roman DINOsaur V5, script version 353896859](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v5?scriptVersionId=353896859) | Public four-reader graph with a new outer CoAt/Raptor-family routing weight of 0.70. This is a bounded hypothesis, with no verified gain for this account. |
| 18 | [yamadan96 corrected public D4](https://www.kaggle.com/code/yamadan96/rsna-knee-d4-public0946?scriptVersionId=354057270) | Best explicit public-only score attribution found: publisher reports 0.943 and explains why 0.946 requires private inputs. |

The first eight positions were returned by the API on September 30. Their latest download metadata can name a different date from the score-sorted listing; the source hash, downloaded version number and HTML script-version identifier are the pinning evidence used here. Do not assume a latest-source download is necessarily the version responsible for a displayed best score.

## Inaccessible apparent leaders

The D4 blend source searches for `rsna-knee-eff6-assets` and `rsna-knee-ens14-assets`, expects 8 and 43 checkpoints respectively, and executes the asset-provided `src/infer_kernel.py`. Its selected fleet uses `S_dist`, `S_coat`, and `S_cnxl`; it blends that extra arm into the public output with default rank weight 0.45. The D4-lite source uses a smaller `S_dist` leg. Both contain restoration paths that can produce a public-only CSV while losing the score contribution of the extra arm. These are source observations, not a claim about the authors' completed runs. [D4 blend source](https://www.kaggle.com/code/pjmathematician/rsna-knee-d4-blend), [D4 lite source](https://www.kaggle.com/code/pjmathematician/rsna-knee-d4-lite).

Authenticated `dataset_metadata` calls for `pjmathematician/rsna-knee-eff6-assets`, `pjmathematician/rsna-knee-ens14-assets`, and `aastikrajan15/knee-ckpt` all returned HTTP 403. An owner-filtered public dataset search for pjmathematician's RSNA assets returned no entries. This proves these guessed exact asset handles are unavailable to this account now; it does not prove the authors cannot access them. Results are saved in `dataset-metadata/access-status.json`. The decoded Aastik inference script is retained in the earlier audit directory at `artifacts/private/external-notebook-audit/top-route-2/decoded-ours-stage.py`.

These sources can inform training architecture ideas but cannot support a directly reproducible higher-score fork without their missing assets.

## Accessible complementary readers

The public family retains the already familiar DINOv2, five A5 folds, RadImageNet heads and four Raptor views, and adds four structurally distinct CoAtNet readers. Their mean probabilities are ranked as a family, then mixed at **40% CoAt family / 60% public Raptor**. The outer transformer-versus-CoAt/Raptor weight defaults to 0.60 with overrides ACL 0.75, Medial Meniscus 0.80, Lateral Meniscus 1.00, Lateral OA 0.75 and Fracture 0.75. [Inspected public source](https://www.kaggle.com/code/yamadan96/rsna-knee-d4-public0946?scriptVersionId=354057270).

| Reader | Public checkpoint/data source | Publisher manifest evidence |
| --- | --- | --- |
| Residual gated top three | [resgated epochs 4/6/8](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coat-resgated-ep10-top3) | Existing public component; expected manifest SHA-256 `98511a8fdeb9da0e6e70c78d013ff636e1476f31c80b5dc134d294b18c3f284e`. |
| D4 depth zones | [D4 package](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2) | One parent SWA checkpoint, member epochs 15/16, plus a 3,255-element adapter averaged across epochs 11/9/5. Uses 96 slices and three depth zones within each anatomical slot. |
| Global96 top three | [Global96 package](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coatnet-global96-top3) | Epochs 16/23/18; full-width 1,024-dimensional gated window reader, 96-slice stack and at most 94 windows. |
| Repair-v1 top three | [D4 package, `repairv1/` directory](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2) | Epochs 12/7/11; direct finding-specific spatial queries and 76-window maximum; explicitly audited ARM-to-x86 renderer differences. |

The three newly inspected publisher manifests were downloaded without model execution and their hashes match the notebook constants:

| Manifest | SHA-256 |
| --- | --- |
| `coatnet_pairfilm_manifest.json` | `7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d` |
| `coatnet_global96_top3_manifest.json` | `015f09030e76f86274cadae40777b3c3cf5c19f8c835d5ad88e15db41903779e` |
| `coatnet_repairv1_top3_manifest.json` | `99288a9a42f36ba544d7b0ad2fe645e3b1fc9c507c23802c16ca0d2e5085f684` |

They are stored in `publisher-manifests/`. D4 parent checkpoint hash is `5de38e333e3184d7b52fb068183f0cc198d6cbabb6ffbe36b26338622e82d297`, and adapter hash is `b7c48b19997bc7ecb7e997f8dcf491ec5701ff917d1dac4b369eb85ac97ae89a`. Global96 and Repair-v1 list the hashes of runtime files and each selected checkpoint under `files`/`checkpoints`; CPU preflight should verify every applicable file, not only the manifest itself. The manifests' Gold58 AUCs were used to select checkpoints, so they are **selection-set statistics**, not untouched OOF evidence or expected Kaggle scores. [D4 publisher](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2), [Global96 publisher](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coatnet-global96-top3).

## Attachment and rights audit

All **14** datasets in the corrected notebook resolve to authenticated metadata and complete file inventories. Both public training-notebook output sources list `rsna_rad_e11/v52_e11_heads.pt`, and the existing baseline had already mounted DINOv2-small. This is evidence of API-visible files; the proposed CPU preflight is still needed to prove the mounted paths and exact selected bytes. Results are in `full-dataset-access-status.json`, per-dataset `files.json`, and the two `sofiaanjenje--*--output-files.json` files. [E11 output source](https://www.kaggle.com/code/sofiaanjenje/rsna-knee-e11-train), [E13 output source](https://www.kaggle.com/code/sofiaanjenje/rsna-knee-e13-train), [DINOv2-small](https://www.kaggle.com/models/metaresearch/dinov2/PyTorch/small/1).

The three new CoAt packages declare CC0-1.0. Across all 14 attachments, nine declare CC0-1.0, three declare CC-BY-NC-SA-4.0, one declares `other`, and the OpenCV-wheel dataset declares `unknown`. Preserve component notices and underlying model terms. The notebook's downloadable metadata and source do not include a code-license declaration, and the currently retrievable Kaggle HTML omits it; **this audit has not verified a source-code license** for these stronger notebooks. Do not copy the baseline's Apache declaration onto them without primary evidence. Prior E13 output redistribution provenance and the RadImageNet asset restrictions remain relevant. [RadImageNet asset](https://www.kaggle.com/datasets/marwanmath/resnet-50-radimagenet-marwan), [V52 heads](https://www.kaggle.com/datasets/prvsiyan/rsna-knee-v52-radimagenet-heads-20260812), [prior notices](../../artifacts/private/public-notebook-adoption/dataset-notices/SOURCE_AND_LICENSES.md).

## Runtime limitations and next experiment

The inspected graph requires **exactly two T4s**, OpenCV 4.12.0 and timm 1.0.22, and includes pinned offline wheels. It uses a 28,800-second code budget. Readers run in parallel pairs only for cohorts up to 48 studies; larger cohorts run sequentially to limit contention and host memory on the stated 29-GB Kaggle machine. Assertions, child process deadlines and receipt files exist, but reader-level fallback and neutral prediction filling still remain in the source. A `status=COMPLETE` CSV alone is insufficient proof that the intended full ensemble ran. [Jiwei source](https://www.kaggle.com/code/jiweiliu/rsna-knee-fast-2xt4-inference?scriptVersionId=352245350).

Claims of full hidden inference in under 30 minutes in copied notebook introductions were not supported by a hidden-run receipt here. The visible three-study run cannot establish that. The older web-search snapshot of Jiwei's comments page showed v5 and 0.941, whereas today's authenticated source is v11; this is why we do not assign the stale web score to the current source. [Older primary snapshot](https://www.kaggle.com/code/jiweiliu/rsna-knee-fast-2xt4-inference/comments).

Recommended sequence:

1. Complete a **CPU-only, 600-second** asset audit for corrected public D4 v2, extracting only exact named-asset lookup and preflight definitions. Mount its 14 datasets, two notebook outputs and DINOv2-small; verify all component manifest inventories, offline wheels, Rad hashes and checkpoint files. CPU asset lookup and hashes do not require executing model source.
2. Resolve the stronger notebook's code-license attribution from the primary page before adopting its source. Preserve original publisher notices and score attribution.
3. After submission 56705536 becomes terminal and ledger/quota evidence permits it, run one bounded visible two-T4 inference dry run. Require all four CoAt readers, intended probability reduction, zero failed/fallback studies, all DINO/A5/Rad/Raptor arms and an exact ordered 13-column output. Record time and memory. This experiment tests reproducibility, not a gain on three unlabeled visible studies.
4. If complete, prepare **one** unchanged-recipe scoring comparison against the baseline. A second routing experiment can then hold all checkpoints fixed and change only the outer default from 0.60 to 0.70, as in [Roman's public v23 source](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v5?scriptVersionId=353896859). Its claimed S75 inspiration is weaker evidence than its actual public source and must not be reported as our measured improvement.

Frozen candidate source: `d4-public0946/rsna-knee-d4-public0946.ipynb`, SHA-256 `35547f8a28ed09fc68acefa6d23f12fa0bfd6a959f54fdadfdfc3dc3c32768f8`, 23 code cells, API version 2. Alternative fast source: `fast-2xt4/rsna-knee-fast-2xt4-inference.ipynb`, SHA-256 `cd5b8b69f89bcfaa6b6ee065f7b18f7134dc71968be3827c87ca3e5672007cac`, 23 code cells, API version 11. Routing variant: `dinosaur-v5/rsna-knee-dinosaur-v5.ipynb`, SHA-256 `4de0f094ab50c32dd409da9ee1de7cbdecbfcbce6e5492c69827427b91eed717`, 39 code cells, API version 23.

## 11:37 UTC follow-up — exact D4 v2 code license verified

**This supersedes the earlier unverified notebook-code-license limitation and resolves step 2 above for D4 v2.** A modern cached Chrome browser rendered the exact [D4 v2 page, script version 354057270](https://www.kaggle.com/code/yamadan96/rsna-knee-d4-public0946?scriptVersionId=354057270). Its License panel explicitly states that the notebook is released under Apache 2.0 and links to the Apache license. The same page shows Version 2 of 2, a Copy & Edit action, and **Best Score 0.943 V1**. Thus the live page independently confirms the author's scored v1 result; v2's added attribution/correction text does not itself establish another scored result. Visible v2 runtime is 332.1 seconds on two T4s, which still does not establish hidden-test runtime.

The service also grants a separate permission relevant to private forks. The currently active [Kaggle Terms of Use](https://www.kaggle.com/terms), effective June 22, 2025, **Section 9, fourth paragraph, Public User Submission**, grants all other users access and use rights for public submissions, bounded by the service's functionality. Its decisive wording is: “to use and exercise all rights in it, as permitted by the functionality of the Services.” This supports private in-service Copy & Edit/use of this public notebook; that application is an inference from the grant and the observed notebook functionality. Section 5 retains copyright and content restrictions, so a download button alone would not establish unrestricted external redistribution. Section 9's service grant is distinct from the now-observed Apache code license.

The current [competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) were also retrieved through the browser. **General Competition Rules, Section 6(b), Public Code Sharing** allows public sharing through competition forums/notebooks subject to third-party intellectual-property rights and deems shared code licensed under an OSI-approved license permitting commercial use. **Section 6(c), Use of Open Source** specifies that same license requirement for code generating submissions. Section 6(a)'s prohibition concerns private code sharing between teams; using a publicly available notebook in this account's private working copy is a different action. The Apache Software Foundation identifies [Apache 2.0 as OSI approved](https://www.apache.org/licenses/LICENSE-2.0).

For an attributed private adaptation, retain the original credits and notices, include the Apache license, and mark modified files. The notebook's explicit Apache declaration supplies the code license; model checkpoints, input data, notebook-output heads and eventual winner handoff retain their component-specific terms. [Apache license Sections 2 and 4](https://www.apache.org/licenses/LICENSE-2.0).

Raw evidence is preserved in `artifacts/private/external-notebook-audit/frontier-2026-09-30/license-use-audit/`: `d4-public-v2-rendered.html`/`.txt`, `terms-rendered.html`/`.txt`, `competition-rules-rendered.html`/`.txt`, browser logs, and `APACHE-2.0.txt`. `license-use-evidence.json` records the observed URL/version, statements, terms section, timestamp and file hashes. The static HTTP/web-reader responses missed the client-rendered panels; browser rendering supplied the missing primary evidence. No login, rules acceptance, messages, notebook push, training or scoring action occurred.

## Scored v1 versus frozen v2 — exact code comparison

The primary Kaggle notebook browser APIs identify the scored [v1 as script version 353578144](https://www.kaggle.com/code/yamadan96/rsna-knee-d4-public0946?scriptVersionId=353578144), notebook-version record 214864354. `GetKernelViewModel` links the author's **submission 56696090** to that source script version and reports **0.943**; its best-score record explicitly names v1. This is stronger provenance than the publisher's written claim alone. The live v1 page also explicitly declares Apache 2.0. [V1 page](https://www.kaggle.com/code/yamadan96/rsna-knee-d4-public0946?scriptVersionId=353578144).

V1 source was retrieved read-only through the public Code view's first-party `GetKernelSessionSource` response. Its **23 code cells are exactly identical**, character for character and in order, to the frozen [v2 source](https://www.kaggle.com/code/yamadan96/rsna-knee-d4-public0946?scriptVersionId=354057270). Removing v2's first added Markdown cell makes every remaining cell's type and source identical to v1. The only cell-source change is the added 25-line attribution and score-correction introduction. All **18 attached source/version identifiers** are also unchanged, including the 14 datasets, two notebook output versions, competition data and DINOv2 model.

The v1 raw notebook SHA-256 is `5e218bfeb0ed727f52d2047d02c89870686c217b1c2e1ceffc74034c9d241ede`. The common code-cell bundle SHA-256 is `86e4ad72b19bfb731a4bb7ef6d887cd350eb90b2009fa151389aecac6f718599`, computed from compact UTF-8 JSON of the ordered joined code-cell source strings. Notebook outputs and execution metadata differ between runs, so raw notebook hashes differ; source-cell comparison is the evidence for unchanged inference logic. This establishes the frozen v2 recipe's lineage to the author's scored v1, while this account still requires its own complete runtime and scoring evidence.

Artifacts are in `artifacts/private/external-notebook-audit/frontier-2026-09-30/d4-public0946-v1/`: exact `rsna-knee-d4-public0946.ipynb`, `code-version-comparison.json` with per-cell hashes and exact attachment IDs, `markdown-diff.patch`, the first-party version-list and view-model responses, and `scored-v1-code-browser/` containing the source response. The CLI `/1` route returned 403 and typed `version_label` attempts returned 404; the public browser's version history and source endpoint supplied the exact old version without an account mutation, training or submission.

## Executed CPU input check — September 30

Private CPU-only `jitheswar/rsna-public-frontier-cpu-asset-preflight` version 1 completed in 68.26 seconds of code execution. It extracts only named-asset lookup, hashing and preflight definitions from the exact source; no model inference runs. Downloaded code matches the immutable local request, and remote metadata confirms private, internet-off CPU execution with the expected attachments. Four pinned CoAt manifests and all 107 files they list were verified, including eleven checkpoints and 2,973,913,077 bytes of manifest-listed assets. E13 and DINO mounts and the thirteen-column competition schema also passed. GPU count and additional spend were zero. Evidence: `artifacts/private/public-frontier-adoption/cpu-asset-preflight-v1/validation.json`, raw `output/preflight.json` and authenticated `terminal-status.json`; output SHA-256 `fd1857e0947b00c27733356b877857d6c2dce155d1535c5fe1711c7e5ad8917b`.

The September 30 authenticated top-five leaderboard snapshot still shows rhoskeri leading at 0.961 (`artifacts/private/leaderboard-snapshots/2026-09-30-frontier-top5.json`). Submission 56705536 remains pending at this checkpoint. The successful CPU check establishes the checked mounts and bytes; GPU dependencies, full-reader inference, source license and scored comparison remain separate gates. [Competition leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard).
