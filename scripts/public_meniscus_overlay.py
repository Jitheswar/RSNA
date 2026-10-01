"""Fixed specialist vote and cache capture for the unchanged D4 anchor."""

import numpy as np
import pandas as pd
import random
import torch


TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
ID = "StudyInstanceUID"
TARGET = "Medial Meniscus"


def run_preserving_random_state(inference, *args, **kwargs):
    """Prevent specialist model initialization from advancing anchor RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            return inference(*args, **kwargs)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _frame(frame, columns, studies, name):
    if frame.columns.tolist() != [ID] + columns:
        raise ValueError(f"{name}: target schema differs")
    if frame[ID].astype(str).tolist() != studies:
        raise ValueError(f"{name}: study order/coverage differs")
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"{name}: predictions must be finite in [0,1]")


def apply_meniscus_overlay(anchor, transformer, hybrid, specialist):
    """Replace exactly 10% of the Medial Meniscus Transformer vote.

    Transformer holds the anchor's raw calibrated predictions; hybrid holds
    its unchanged CoAt/Raptor scores. The specialist holds raw probabilities.
    Ranks and the final rank use the anchor's average-tie percentile policy.
    """
    studies = anchor[ID].astype(str).tolist()
    if not studies or len(set(studies)) != len(studies) or any(not uid for uid in studies):
        raise ValueError("anchor: study identities must be nonempty and unique")
    for name, frame in [("anchor", anchor), ("transformer", transformer), ("hybrid", hybrid)]:
        _frame(frame, TARGETS, studies, name)
    _frame(specialist, ["Medial Meniscus", "Lateral Meniscus"], studies, "specialist")
    tr = transformer[TARGET].rank(method="average", pct=True).to_numpy()
    cr = hybrid[TARGET].to_numpy()
    expected_anchor = pd.Series((1.0 - 0.80) * tr + 0.80 * cr).rank(method="average", pct=True)
    # The anchor's release cell reparses a CSV with pandas' default float
    # reader. Permit its rounding while rejecting any changed rank or route.
    if not np.allclose(anchor[TARGET].to_numpy(), expected_anchor.to_numpy(), rtol=0, atol=2e-15):
        raise ValueError("Medial Meniscus anchor does not use the frozen 20/80 route")
    bag = specialist[TARGET].rank(method="average", pct=True).to_numpy()
    result = anchor.copy(deep=True)
    result[TARGET] = pd.Series(0.10 * tr + 0.80 * cr + 0.10 * bag).rank(
        method="average", pct=True).to_numpy()
    unchanged = [ID] + [name for name in TARGETS if name != TARGET]
    if not result[unchanged].equals(anchor[unchanged]):
        raise ValueError("Specialist overlay changed another target or study identity")
    _frame(result, TARGETS, studies, "overlay")
    return result


class MeniscusCacheCapture:
    """Retain the completed 336px cache, including an unlinked open memmap.

    The delegate's inputs, returned objects and array bytes remain unchanged.
    Exactly one matching cache is required before specialist inference.
    """

    def __init__(self, build_cache):
        self.delegate = build_cache
        self.candidates = []

    def build_cache(self, *args, **kwargs):
        result = self.delegate(*args, **kwargs)
        studies, cache, mask = result
        if isinstance(cache, np.ndarray) and cache.shape[1:] == (6, 12, 336, 336):
            studies = [str(uid) for uid in studies]
            if (cache.dtype != np.uint8 or len(studies) != len(cache)
                    or not isinstance(mask, np.ndarray) or mask.dtype != np.float32
                    or mask.shape != (len(studies), 6)
                    or not studies or any(not uid for uid in studies)
                    or len(set(studies)) != len(studies)):
                raise ValueError("Specialist cache identity, shape or dtype differs")
            self.candidates.append((studies, cache, mask))
        return result

    def take(self, expected_studies):
        if len(self.candidates) != 1:
            raise ValueError(f"Expected one completed specialist cache, got {len(self.candidates)}")
        studies, cache, mask = self.candidates[0]
        if studies != list(expected_studies):
            raise ValueError("Specialist cache study order differs from the anchor")
        return cache, mask, studies

    def clear(self):
        self.candidates.clear()
