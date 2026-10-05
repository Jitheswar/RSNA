"""Multimodal (image + report) panel-style labels from a frontier LLM via the Codex CLI (pilot on gold studies).

Host clarification (topic 733965): submitting Competition Data to an external LLM/API for inference is not, by
itself, prohibited sharing. For each study we render three montages (sagittal / coronal / axial, best fat-suppressed
fluid-sensitive series, six slices each) from the canonical volumes (every knee shown as a RIGHT knee), attach them
with the report text, and ask for P(reference panel positive) for the 12 findings.

python scripts/v2/mm_label.py --gold --workers 4 --out artifacts/private/v2/llm/mm-sol61-gold.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import choose_series, load_series_table  # noqa: E402
from llm_label import SYSTEM, TARGETS  # noqa: E402

CODEX = str(Path.home() / ".local/bin/codex")
V2 = Path("artifacts/private/v2")
PLANES = ["Sagittal", "Coronal", "Axial"]

ORIENT = ("Image orientation (all montages): every knee is displayed as a RIGHT knee. Coronal and axial images: the "
          "MEDIAL side is on the image RIGHT, lateral on the image left; coronal rows run superior (top) to inferior; "
          "axial rows run anterior (top) to posterior. Sagittal images: anterior is on the image LEFT, superior at the "
          "top; the six sagittal panels run from LATERAL (first) to MEDIAL (last). Panels are numbered in reading order "
          "(left to right, top to bottom). Sequences are fat-suppressed fluid-sensitive (PD/T2 FS or STIR) when "
          "available; the series description is printed in each montage.")


def montage(vol, title, n=6):
    D = vol.shape[0]
    idx = np.unique(np.linspace(0.15 * (D - 1), 0.85 * (D - 1), n).round().astype(int))
    tiles = []
    for k, i in enumerate(idx):
        t = cv2.cvtColor(np.ascontiguousarray(vol[i]), cv2.COLOR_GRAY2BGR)
        cv2.putText(t, str(k + 1), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        tiles.append(t)
    while len(tiles) < n:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r * 3:(r + 1) * 3]) for r in range(n // 3)]
    img = np.vstack(rows)
    bar = np.zeros((36, img.shape[1], 3), np.uint8)
    cv2.putText(bar, title[:110], (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return np.vstack([bar, img])


def study_montages(study, meta, series_desc, flips, out_dir):
    g = meta[meta.StudyInstanceUID == study]
    paths = []
    for plane in PLANES:
        gp = g[g.plane == plane]
        if gp.empty:
            continue
        row = choose_series(gp, 1, None)[0]
        vol = np.load(V2 / "vol384/vol" / f"{row.SeriesInstanceUID}.npy")
        if study in flips:  # laterality checker re-flip (applied to node-local copies during training)
            vol = vol[::-1] if plane == "Sagittal" else vol[:, :, ::-1]
        desc = series_desc.get(row.SeriesInstanceUID, "")
        img = montage(vol, f"{plane} | {desc} | FS={int(row.Fat_Suppression)} fluid={int(row.Fluid_Sensitive)}")
        p = Path(out_dir) / f"{plane.lower()}.png"
        cv2.imwrite(str(p), img)
        paths.append(str(p))
    return paths


def schema():
    item = {"type": "object", "additionalProperties": False, "required": ["p", "why"],
            "properties": {"p": {"type": "integer"}, "why": {"type": "string"}}}
    return {"type": "object", "additionalProperties": False, "required": TARGETS,
            "properties": {t: item for t in TARGETS}}


def build_prompt(report):
    head = SYSTEM.split("For each target give:")[0].strip()
    head = head.replace("using the strict criteria below. Borderline or equivocal findings were graded NEGATIVE. "
                        "Reports often describe lower-grade findings that the panel would call negative, and sometimes "
                        "omit findings the panel would call positive.",
                        "using the definitions below. Reports sometimes omit findings the panel called positive.")
    return "\n".join([
        head, "",
        "You are now ALSO given MRI images of this knee (attached montages) in addition to the original report.",
        ORIENT, "",
        "The reference panel read the IMAGES (not the report) and is more liberal than strict definitions (e.g. small "
        "effusions, grade-2 sprains, mild chondral loss and image-only synovitis are often called positive). Combine "
        "what you see with what the report says; when the report is silent, decide from the images.",
        "For each of the 12 targets return p = probability (0-100) that the panel marked it POSITIVE, and a short "
        "reason (<= 15 words). Do not run any commands or tools; just answer.", "",
        "Report:", "<<<", report.strip(), ">>>"])


def run_one(study, report, meta, series_desc, flips, model, effort, schema_path, timeout):
    with tempfile.TemporaryDirectory(prefix="mm-") as td:
        imgs = study_montages(study, meta, series_desc, flips, td)
        out = Path(td) / "out.json"
        cmd = [CODEX, "exec", "-m", model, "-c", f'model_reasoning_effort="{effort}"', "-s", "read-only",
               "--ephemeral", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules", "-C", td,
               "--output-schema", schema_path, "-o", str(out), "--color", "never"]
        for p in imgs:
            cmd += ["-i", p]
        cmd += ["-"]
        t0 = time.time()
        r = subprocess.run(cmd, input=build_prompt(report), capture_output=True, text=True, timeout=timeout)
        if not out.exists():
            raise RuntimeError(f"no output (rc={r.returncode}): {r.stderr[-400:]}")
        parsed = json.loads(out.read_text())
        return {"StudyInstanceUID": study, "parsed": parsed, "model": model, "effort": effort,
                "n_images": len(imgs), "s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", action="store_true", help="only the reference-labelled studies (pilot)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="gpt-6.1-sol")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", required=True)
    ap.add_argument("--order", default="", help="'unaddressed': studies whose reports leave most findings unaddressed first")
    args = ap.parse_args()
    tr = pd.read_csv(V2 / "train.csv").set_index("StudyInstanceUID")
    studies = list(tr.index[tr.ACL.notna()]) if args.gold else list(tr.index)
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["StudyInstanceUID"] for l in out.read_text().splitlines() if l.strip()}
    studies = [s for s in studies if s not in done]
    if args.order == "unaddressed":
        gr = pd.read_csv(V2 / "labels/sol61_grades.csv").set_index("StudyInstanceUID")
        n_un = (gr.reindex(studies)[TARGETS] == -1).sum(1).fillna(12)
        studies = list(n_un.sort_values(ascending=False, kind="stable").index)
    studies = studies[: args.limit or None]
    meta = load_series_table(sorted(glob.glob(str(V2 / "vol384/meta-train-*.csv"))), str(V2 / "train_series.csv"))
    surv = pd.read_csv(V2 / "survey-train.csv", usecols=["SeriesInstanceUID", "SeriesDescription"])
    series_desc = dict(zip(surv.SeriesInstanceUID, surv.SeriesDescription.astype(str)))
    flips = set(pd.read_csv(V2 / "lat_flip_v1.csv").StudyInstanceUID)
    sp = out.with_suffix(".schema.json")
    sp.write_text(json.dumps(schema()))
    lock = threading.Lock()
    print(f"{len(studies)} studies to label ({len(done)} done)", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(run_one, s, str(tr.loc[s, "Report"]), meta, series_desc, flips, args.model, args.effort,
                            str(sp), args.timeout): s for s in studies}
        for n, f in enumerate(as_completed(futs)):
            s = futs[f]
            try:
                row = f.result()
                with lock, open(out, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"[{n + 1}/{len(studies)}] {s[-8:]} ok {row['s']}s", flush=True)
            except Exception as exc:
                print(f"[{n + 1}/{len(studies)}] {s[-8:]} FAIL {type(exc).__name__}: {str(exc)[:200]}", flush=True)


if __name__ == "__main__":
    main()
