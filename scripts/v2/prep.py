"""Canonical knee-MRI series preparation (v2).

Shared by training packing and Kaggle inference, so both see identical pixels:

1. sort slices along the acquisition normal (geometry, InstanceNumber fallback);
2. reorient every slice to a canonical patient frame (LPS):
   sagittal  cols -> posterior (+y), rows -> inferior (-z), slices -> +x
   coronal   cols -> patient left (+x), rows -> inferior (-z), slices -> +y
   axial     cols -> patient left (+x), rows -> posterior (+y), slices -> -z
3. laterality-normalise so every knee looks like a RIGHT knee:
   coronal/axial: medial side on image right; sagittal: slice 0 lateral, last medial;
4. square physical crop (<= CROP_MM, centred on the foreground centroid), resize to SIZE;
5. per-series percentile window -> uint8; depth capped at MAX_DEPTH (nearest resampling).
"""

from __future__ import annotations

import numpy as np

SIZE = 384
CROP_MM = 160.0
MAX_DEPTH = 64
CANON = {
    # plane: (target column direction, target row direction, slice direction)
    "Sagittal": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0])),
    "Coronal": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),
    "Axial": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
}


def _f(v, n):
    try:
        a = np.asarray([float(x) for x in v], dtype=np.float64)
        return a if a.shape == (n,) and np.isfinite(a).all() else None
    except Exception:
        return None


def slice_geometry(ds):
    """(ipp, iop, pixel_spacing) as float arrays, or Nones."""
    return (_f(getattr(ds, "ImagePositionPatient", None) or [], 3),
            _f(getattr(ds, "ImageOrientationPatient", None) or [], 6),
            _f(getattr(ds, "PixelSpacing", None) or [], 2))


def plane_from_iop(iop):
    n = np.cross(iop[:3], iop[3:])
    return ("Sagittal", "Coronal", "Axial")[int(np.argmax(np.abs(n)))]


def image_center(ds):
    ipp, iop, ps = slice_geometry(ds)
    if ipp is None or iop is None or ps is None:
        return None
    rows, cols = int(ds.Rows), int(ds.Columns)
    return ipp + iop[:3] * ps[1] * cols / 2.0 + iop[3:] * ps[0] * rows / 2.0


def study_laterality(headers):
    """'R' / 'L' / None from the Laterality tag, else image-centre x (LPS: right knee has x<0).

    headers: list of pydicom datasets (any slices of any series of one study).
    Returns (side, source) where source is 'tag', 'geometry' or 'unresolved'.
    """
    tags = []
    xs = []
    for ds in headers:
        v = str(getattr(ds, "Laterality", "") or "").strip().upper()
        if v[:1] in ("L", "R"):
            tags.append(v[:1])
        c = image_center(ds)
        if c is not None:
            xs.append(float(c[0]))
    if tags and len(set(tags)) == 1:
        return tags[0], "tag"
    if xs:
        m = float(np.median(xs))
        if abs(m) >= 20.0:
            return ("R" if m < 0 else "L"), "geometry"
        return ("R" if m < 0 else "L"), "weak-geometry"
    return None, "unresolved"


def _orient_ops(iop, plane):
    """Return (transpose, flip_rows, flip_cols, row_dir, col_dir) mapping to the canonical frame."""
    tc, tr, _ = CANON[plane]
    rdir, cdir = iop[:3].copy(), iop[3:].copy()  # rdir: along columns (moving right); cdir: moving down
    transpose = abs(rdir @ tc) < abs(rdir @ tr)
    if transpose:
        rdir, cdir = cdir, rdir
    flip_cols = (rdir @ tc) < 0
    flip_rows = (cdir @ tr) < 0
    return transpose, flip_rows, flip_cols


_UNCOMPRESSED = {"1.2.840.10008.1.2", "1.2.840.10008.1.2.1"}


def _raw_pixels(ds):
    """Uncompressed single-frame mono pixels straight from PixelData (bit-identical to ds.pixel_array on 2,500
    random training files, ~3x faster); None -> caller falls back to pydicom."""
    try:
        ts = str(ds.file_meta.TransferSyntaxUID)
        if ts not in _UNCOMPRESSED or int(getattr(ds, "SamplesPerPixel", 1)) != 1 \
                or int(getattr(ds, "NumberOfFrames", 1) or 1) != 1:
            return None
        ba, pr = int(ds.BitsAllocated), int(getattr(ds, "PixelRepresentation", 0))
        if ba not in (8, 16) or (pr and int(ds.BitsStored) != ba):  # signed with unused bits needs sign extension
            return None
        dt = {(16, 0): np.uint16, (16, 1): np.int16, (8, 0): np.uint8, (8, 1): np.int8}[(ba, pr)]
        n = int(ds.Rows) * int(ds.Columns)
        a = np.frombuffer(ds.PixelData, dtype=dt)
        return a[:n].reshape(int(ds.Rows), int(ds.Columns)) if a.size >= n else None
    except Exception:
        return None


def _pixels(ds):
    a = _raw_pixels(ds)
    a = (ds.pixel_array if a is None else a).astype(np.float32)
    if a.ndim == 3:  # unexpected multi-frame / colour: take first frame or mean channels
        a = a[..., 0] if a.shape[-1] in (3, 4) else a[0]
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    a = a * slope + inter
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        a = a.max() - a
    return a


def _resize(img, size):
    import cv2
    h, w = img.shape
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


def prepare_series(datasets, plane_hint, side, size=SIZE, crop_mm=CROP_MM, max_depth=MAX_DEPTH):
    """datasets: pydicom datasets (with pixels) of one series. Returns (uint8 [D,size,size], info dict)."""
    geo = [slice_geometry(d) for d in datasets]
    iops = [g[1] for g in geo if g[1] is not None]
    if iops:
        iop = np.median(np.stack(iops), axis=0)
        plane = plane_from_iop(iop)
    else:
        iop, plane = None, plane_hint if plane_hint in CANON else "Sagittal"
    _, _, sdir = CANON[plane]
    # --- slice order along the canonical slice direction
    if iop is not None and all(g[0] is not None for g in geo):
        normal = np.cross(iop[:3], iop[3:])
        normal = normal if normal @ sdir >= 0 else -normal
        key = [float(g[0] @ normal) for g in geo]
        order_src = "geometry"
    else:
        key = [float(getattr(d, "InstanceNumber", i) or i) for i, d in enumerate(datasets)]
        order_src = "instance"
    order = np.argsort(np.asarray(key), kind="stable")
    # drop exact-duplicate positions (repeated acquisitions); keep first
    kept, last = [], None
    for i in order:
        if last is not None and order_src == "geometry" and abs(key[i] - last) < 1e-3:
            continue
        kept.append(i)
        last = key[i]
    # --- pixels, all resampled to the first slice's shape
    imgs = []
    shape0 = None
    for i in kept:
        try:
            a = _pixels(datasets[i])
        except Exception:
            continue
        if shape0 is None:
            shape0 = a.shape
        elif a.shape != shape0:
            import cv2
            a = cv2.resize(a, (shape0[1], shape0[0]), interpolation=cv2.INTER_LINEAR)
        imgs.append(a)
    if not imgs:
        raise ValueError("no decodable slices")
    vol = np.stack(imgs)  # [D, rows, cols]
    ps = next((g[2] for g in geo if g[2] is not None), np.array([0.5, 0.5]))
    row_mm, col_mm = float(ps[0]), float(ps[1])  # spacing between rows, between columns
    if iop is not None:
        transpose, flip_rows, flip_cols = _orient_ops(iop, plane)
        if transpose:
            vol = vol.transpose(0, 2, 1)
            row_mm, col_mm = col_mm, row_mm
        if flip_rows:
            vol = vol[:, ::-1, :]
        if flip_cols:
            vol = vol[:, :, ::-1]
    else:
        transpose = flip_rows = flip_cols = None
    # --- laterality normalisation (make every knee a right knee)
    lat_flip = side == "L"
    if lat_flip:
        if plane == "Sagittal":
            vol = vol[::-1]  # slice order: lateral -> medial for a right knee (+x is medial)
        else:
            vol = vol[:, :, ::-1]  # coronal/axial: right knee medial is +x = image right already
    # --- physical square crop around the foreground centroid
    D, H, W = vol.shape
    mid = vol[D // 4: max(D // 4 + 1, 3 * D // 4)].mean(0)
    thr = np.percentile(mid, 99) * 0.15
    fg = mid > thr
    if fg.sum() > 100:
        ys, xs = np.nonzero(fg)
        cy, cx = float(ys.mean()), float(xs.mean())
    else:
        cy, cx = H / 2.0, W / 2.0
    side_mm = min(crop_mm, H * row_mm, W * col_mm)
    hh, hw = side_mm / row_mm / 2.0, side_mm / col_mm / 2.0
    cy = float(np.clip(cy, hh, H - hh))
    cx = float(np.clip(cx, hw, W - hw))
    y0, y1 = int(round(cy - hh)), int(round(cy + hh))
    x0, x1 = int(round(cx - hw)), int(round(cx + hw))
    vol = vol[:, max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    # --- per-series intensity window
    sub = vol[:, ::4, ::4]
    lo = float(np.percentile(sub, 0.5))
    hi = float(np.percentile(sub, 99.7))
    if hi <= lo:
        hi = lo + 1.0
    # --- depth cap
    if vol.shape[0] > max_depth:
        idx = np.rint(np.linspace(0, vol.shape[0] - 1, max_depth)).astype(int)
        vol = vol[idx]
    out = np.empty((vol.shape[0], size, size), np.uint8)
    for k in range(vol.shape[0]):
        x = (vol[k] - lo) * (255.0 / (hi - lo))
        out[k] = np.clip(_resize(np.ascontiguousarray(x, dtype=np.float32), size) + 0.5, 0, 255).astype(np.uint8)
    spacing = np.diff(sorted(key)) if len(key) > 1 else np.array([0.0])
    info = {
        "plane": plane, "plane_hint": plane_hint, "n_src": len(datasets), "n_kept": len(kept), "depth": int(out.shape[0]),
        "order": order_src, "transpose": transpose, "flip_rows": flip_rows, "flip_cols": flip_cols,
        "lat_flip": bool(lat_flip), "crop_mm": round(side_mm, 1), "mm_px": round(side_mm / size, 4),
        "slice_mm": round(float(np.median(np.abs(spacing))), 3) if len(spacing) else 0.0,
    }
    return out, info
