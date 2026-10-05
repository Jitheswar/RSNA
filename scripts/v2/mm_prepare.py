"""Prepare multimodal labelling batches for Claude (Opus) subagents: montages on disk + batch JSON + instructions.

Each batch file lists studies with report text and the three montage paths (sagittal / coronal / axial, best
fat-suppressed fluid-sensitive series, six slices each; every knee shown as a RIGHT knee). A subagent reads the
instructions, views the images with its Read tool and writes {uid: {target: p}} to the batch's output path.

python scripts/v2/mm_prepare.py --gold --batch 10 --tag opus-gold
python scripts/v2/mm_collect.py --tag opus-gold --out artifacts/private/v2/llm/mm-opus-gold.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_series_table  # noqa: E402
from llm_label import SYSTEM, TARGETS  # noqa: E402
from mm_label import ORIENT, study_montages  # noqa: E402

V2 = Path("artifacts/private/v2")
ROOT = Path("/home/ch.jitheswar/RSNA")


def instructions():
    head = SYSTEM.split("For each target give:")[0].strip()
    head = head.replace("using the strict criteria below. Borderline or equivocal findings were graded NEGATIVE. "
                        "Reports often describe lower-grade findings that the panel would call negative, and sometimes "
                        "omit findings the panel would call positive.",
                        "using the definitions below. Reports sometimes omit findings the panel called positive.")
    head = head.replace("You will read one knee MRI report (it may be written in any language)",
                        "You will read knee MRI reports (any language) AND view the MRI images of each study")
    return "\n".join([
        "# Multimodal knee MRI labelling task (research dataset; competition rules allow this use)", "",
        head, "",
        "## Images", ORIENT, "",
        "## How the panel reads",
        "The reference panel read the IMAGES (not the report) and is more liberal than strict definitions (e.g. small "
        "effusions, grade-2 sprains, mild chondral loss and image-only synovitis are often called positive). Combine "
        "what you see with what the report says; when the report is silent, decide from the images.", "",
        "## Procedure (follow exactly)",
        "1. Read the batch JSON given in your task (fields: uid, report, images = 3 PNG paths).",
        "2. For EACH study, in order: read the report, then view all three images with the Read tool.",
        "3. For each of the 12 targets give p = integer 0-100 = probability that the panel marked it POSITIVE.",
        "4. When all studies are done, write ONE JSON object to the output path given in your task, mapping uid -> "
        "{target: p} for every study, with exactly these 12 keys: " + ", ".join(TARGETS) + ".",
        "5. Do not skip studies, do not run shell commands, do not modify any other file. Reply with a one-line "
        "summary (number of studies written)."])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--studies", default="", help="optional file with one StudyInstanceUID per line")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    tr = pd.read_csv(V2 / "train.csv").set_index("StudyInstanceUID")
    if args.studies:
        studies = [l.strip() for l in open(args.studies) if l.strip()]
    elif args.gold:
        studies = list(tr.index[tr.ACL.notna()])
    else:
        studies = list(tr.index)
    root = V2 / "mm" / args.tag
    (root / "batches").mkdir(parents=True, exist_ok=True)
    (root / "out").mkdir(parents=True, exist_ok=True)
    mont = V2 / "mm" / "montages"
    meta = load_series_table(sorted(glob.glob(str(V2 / "vol384/meta-train-*.csv"))), str(V2 / "train_series.csv"))
    surv = pd.read_csv(V2 / "survey-train.csv", usecols=["SeriesInstanceUID", "SeriesDescription"])
    series_desc = dict(zip(surv.SeriesInstanceUID, surv.SeriesDescription.astype(str)))
    flips = set(pd.read_csv(V2 / "lat_flip_v1.csv").StudyInstanceUID)
    (root / "INSTRUCTIONS.md").write_text(instructions())
    items = []
    for s in studies:
        d = mont / s
        if not (d / "axial.png").exists() or not (d / "sagittal.png").exists():
            d.mkdir(parents=True, exist_ok=True)
            study_montages(s, meta, series_desc, flips, d)
        imgs = [str((ROOT / d / f"{p}.png")) for p in ("sagittal", "coronal", "axial") if (d / f"{p}.png").exists()]
        items.append({"uid": s, "report": str(tr.loc[s, "Report"]).strip(), "images": imgs})
    for b in range(0, len(items), args.batch):
        k = b // args.batch
        (root / "batches" / f"batch_{k:04d}.json").write_text(json.dumps(items[b:b + args.batch], ensure_ascii=False, indent=1))
    print(f"{len(items)} studies -> {(len(items) + args.batch - 1) // args.batch} batches in {root}")


if __name__ == "__main__":
    main()
