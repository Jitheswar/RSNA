"""Study-level dataset over packed canonical volumes (scripts/v2/prep.py output)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PLANES = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
# slot priority: (plane, fat-sat, fluid) ; fluid-sensitive fat-suppressed first
PRIORITY = [("Sagittal", 1, 1), ("Coronal", 1, 1), ("Axial", 1, 1), ("Sagittal", 0, 0), ("Coronal", 0, 0),
            ("Sagittal", 0, 1), ("Axial", 0, 1), ("Coronal", 0, 1), ("Axial", 0, 0), ("Sagittal", 1, 0),
            ("Coronal", 1, 0), ("Axial", 1, 0)]


def series_type(plane: str, fs: int, fluid: int) -> int:
    return PLANES.get(plane, 0) * 4 + int(fs) * 2 + int(fluid)


def load_series_table(meta_csvs, series_csv) -> pd.DataFrame:
    frames = []
    for m in meta_csvs:
        try:
            frames.append(pd.read_csv(m))
        except pd.errors.EmptyDataError:
            continue
    meta = pd.concat(frames, ignore_index=True).drop_duplicates("SeriesInstanceUID", keep="last")
    meta = meta[meta["error"].isna() & meta["depth"].notna()].copy()
    ser = pd.read_csv(series_csv)
    meta = meta.merge(ser[["SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression"]], on="SeriesInstanceUID", how="left")
    meta["Fluid_Sensitive"] = meta["Fluid_Sensitive"].fillna(0).astype(int)
    meta["Fat_Suppression"] = meta["Fat_Suppression"].fillna(0).astype(int)
    meta["depth"] = meta["depth"].astype(int)
    meta["stype"] = [series_type(p, f, u) for p, f, u in zip(meta.plane, meta.Fat_Suppression, meta.Fluid_Sensitive)]
    rank = {t: i for i, t in enumerate(PRIORITY)}
    meta["prio"] = [rank.get((p, f, u), 99) for p, f, u in zip(meta.plane, meta.Fat_Suppression, meta.Fluid_Sensitive)]
    return meta


def choose_series(g: pd.DataFrame, n_series: int, rng: np.random.Generator | None):
    """Pick one series per type in priority order, then extras; random tie-break when training."""
    g = g[g.depth >= 5]
    picks = []
    used = set()
    for prio in sorted(g.prio.unique()):
        cand = g[g.prio == prio]
        if rng is not None and len(cand) > 1:
            row = cand.iloc[int(rng.integers(len(cand)))]
        else:
            row = cand.sort_values("depth", ascending=False).iloc[0]
        picks.append(row)
        used.add(row.SeriesInstanceUID)
        if len(picks) == n_series:
            return picks
    rest = g[~g.SeriesInstanceUID.isin(used)].sort_values("prio")
    for _, row in rest.iterrows():
        if len(picks) == n_series:
            break
        picks.append(row)
    return picks


def slice_indices(depth: int, k: int, rng: np.random.Generator | None, lo=0.04, hi=0.96):
    a, b = lo * (depth - 1), hi * (depth - 1)
    if rng is None:
        pos = np.linspace(a, b, k)
    else:
        step = (b - a) / max(k - 1, 1)
        pos = np.linspace(a, b, k) + rng.uniform(-0.5, 0.5, k) * step
        if rng.random() < 0.3:  # random sub-window zoom along depth
            s = rng.uniform(0.0, 0.08) * (depth - 1)
            pos = np.clip(pos * (1 - 0.08) + s, 0, depth - 1)
    return np.clip(np.rint(pos), 0, depth - 1).astype(int)


class StudyDataset(Dataset):
    def __init__(self, studies, meta: pd.DataFrame, vol_dir: str, labels: np.ndarray | None,
                 n_series=6, k=16, train=False, seed=0, aux: np.ndarray | None = None, series_drop=0.15,
                 unnorm_lat: dict | None = None, k_plane: dict | None = None, random_pos: bool = False):
        self.studies = list(studies)
        self.by_study = {s: g for s, g in meta.groupby("StudyInstanceUID")}
        self.vol_dir = vol_dir
        self.labels = labels
        self.aux = aux
        self.n_series, self.k, self.train = n_series, k, train
        self.seed = seed
        self.series_drop = series_drop
        self.unnorm_lat = unnorm_lat or {}  # study -> True: undo laterality normalisation (ablation)
        self.k_plane = k_plane or {}  # per-plane slice counts, e.g. {"Sagittal": 20, "Coronal": 12, "Axial": 10}
        self.random_pos = random_pos  # training bag sampling: random window centres instead of an even grid
        self.epoch = 0

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        st = self.studies[i]
        rng = np.random.default_rng((self.seed, self.epoch, i)) if self.train else None
        S = self.n_series
        K = max([self.k] + list(self.k_plane.values())) if self.k_plane else self.k
        g = self.by_study.get(st)
        imgs, types, pos, present = None, np.zeros(S, np.int64), np.zeros((S, K, 2), np.float32), np.zeros(S, np.float32)
        smask = np.zeros((S, K), np.float32)
        picks = choose_series(g, S, rng) if g is not None else []
        if self.train and len(picks) > 2:
            keep = [p for j, p in enumerate(picks) if not (j > 0 and rng.random() < self.series_drop)]
            picks = keep
        for j, row in enumerate(picks):
            vol = np.load(f"{self.vol_dir}/{row.SeriesInstanceUID}.npy", mmap_mode="r")
            D = vol.shape[0]
            Kp = int(self.k_plane.get(row.plane, self.k)) if self.k_plane else K
            if self.random_pos and rng is not None:
                lo_c, hi_c = max(1, int(0.03 * (D - 1))), min(D - 2, int(np.ceil(0.97 * (D - 1))))
                cand = np.arange(lo_c, max(lo_c, hi_c) + 1)
                idx = np.sort(rng.choice(cand, size=Kp, replace=len(cand) < Kp))
            else:
                idx = slice_indices(D, Kp, rng)
            trip = np.stack([np.clip(idx - 1, 0, D - 1), idx, np.clip(idx + 1, 0, D - 1)], 1)  # [Kp,3]
            uniq, inv = np.unique(trip, return_inverse=True)
            block = np.asarray(vol[uniq])  # [U,H,W]
            arr = block[inv.reshape(trip.shape)]  # [K,3,H,W]
            if self.unnorm_lat.get(st):  # ablation: show the knee in its native (un-normalised) laterality
                arr = arr[::-1, ::-1] if row.plane == "Sagittal" else arr[..., ::-1]
                arr = np.ascontiguousarray(arr)
            if imgs is None:
                imgs = np.zeros((S, K, 3) + arr.shape[-2:], np.uint8)
            if arr.shape[-2:] != imgs.shape[-2:]:
                continue
            imgs[j, :Kp] = arr
            types[j] = int(row.stype)
            present[j] = 1.0
            smask[j, :Kp] = 1.0
            rel = idx / max(D - 1, 1)
            mm = (idx - (D - 1) / 2.0) * float(row.slice_mm if row.slice_mm == row.slice_mm else 3.0) / 50.0
            pos[j, :Kp, 0] = rel
            pos[j, :Kp, 1] = mm
        if imgs is None:
            imgs = np.zeros((S, K, 3, 384, 384), np.uint8)
        out = {"x": torch.from_numpy(imgs), "types": torch.from_numpy(types), "pos": torch.from_numpy(pos),
               "present": torch.from_numpy(present), "smask": torch.from_numpy(smask), "idx": i}
        if self.labels is not None:
            out["y"] = torch.from_numpy(self.labels[i].astype(np.float32))
        if self.aux is not None:
            out["aux"] = torch.from_numpy(self.aux[i].astype(np.float32))
        return out
