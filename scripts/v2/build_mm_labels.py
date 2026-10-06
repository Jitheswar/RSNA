"""Training labels from multimodal (image + report) LLM labels mixed with the student-ensemble OOF.

Where a study has a multimodal label: y = 0.5 * p_mm + 0.5 * p_student (pre-fixed weight; gold58 label AUC 0.943 vs
0.918 for the current 0.5 report + 0.5 student mix; adding the report back lowers it). Elsewhere: fall back to an
existing label table (default stud_mix1). Gold studies keep their reference labels inside train.py anyway.

python scripts/v2/build_mm_labels.py --mm artifacts/private/v2/llm/mm-opus-gold.jsonl artifacts/private/v2/llm/mm-opus-all.jsonl \
    --students runs/s_mix1_pw runs/s_mix1_384 runs/s_mix2 --fallback labels/stud_mix1.csv --out labels/stud_mm1.csv
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
V2 = Path("artifacts/private/v2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm", nargs="+", required=True, help="multimodal label JSONLs (later files override earlier)")
    ap.add_argument("--students", nargs="+", required=True, help="run dirs whose fold OOF predictions are averaged")
    ap.add_argument("--fallback", default="labels/stud_mix1.csv")
    ap.add_argument("--w-mm", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    mm = {}
    for f in args.mm:
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                mm[r["StudyInstanceUID"]] = {t: r["parsed"][t]["p"] / 100.0 for t in T}
    mm = pd.DataFrame(mm).T[T]
    oofs = []
    for r in args.students:
        fs = sorted(glob.glob(str(V2 / r / "fold*/oof.csv")))
        oofs.append(pd.concat([pd.read_csv(f) for f in fs]).set_index("StudyInstanceUID")[T])
    st = sum(oofs) / len(oofs)
    fb = pd.read_csv(V2 / args.fallback).set_index("StudyInstanceUID")[T]
    out = fb.copy()
    cov = mm.index.intersection(st.index).intersection(out.index)
    out.loc[cov] = args.w_mm * mm.loc[cov].values + (1 - args.w_mm) * st.loc[cov].values
    out.index.name = "StudyInstanceUID"
    out.reset_index().to_csv(V2 / args.out, index=False)
    print(f"wrote {args.out}: {len(out)} studies, multimodal mix on {len(cov)} ({100 * len(cov) / len(out):.1f}%), "
          f"fallback ({args.fallback}) on {len(out) - len(cov)}")


if __name__ == "__main__":
    main()
