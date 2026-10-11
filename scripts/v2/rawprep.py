"""On-the-fly study preparation from raw DICOMs for training without packed volumes (Kaggle notebooks).

Uses the exact inference path (kaggle_infer.prepare_study -> prep.prepare_series + laterality checker), so training
inputs match what the submission sees.
"""

from __future__ import annotations

import pandas as pd


class RawPrep:
    def __init__(self, comp: str, split: str = "train"):
        ser = pd.read_csv(f"{comp}/{split}_series.csv", dtype=str)
        self.root = f"{comp}/{split}_series"
        self.series = {s: {r.SeriesInstanceUID: r.Anatomical_Plane for r in g.itertuples()}
                       for s, g in ser.groupby("StudyInstanceUID")}

    def __call__(self, study: str) -> dict:
        from kaggle_infer import prepare_study
        _, out, _ = prepare_study((f"{self.root}/{study}", study, self.series.get(study, {})))
        return {se: vol for se, (vol, info) in out.items()}
