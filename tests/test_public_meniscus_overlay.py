"""Output and hidden-size cache regressions for the fixed specialist vote."""

import importlib.util
import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


path = Path(__file__).resolve().parents[1] / "scripts/public_meniscus_overlay.py"
spec = importlib.util.spec_from_file_location("public_meniscus_overlay", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def frames():
    studies = ["a", "b", "c", "d"]
    tr = pd.DataFrame(np.full((4, 12), 0.3), columns=module.TARGETS)
    cr = tr.copy()
    tr[module.TARGET] = [0.1, 0.2, 0.3, 0.4]
    cr[module.TARGET] = [0.50, 0.45, 0.40, 0.35]
    anchor = tr.copy()
    anchor[module.TARGET] = ((1.0 - 0.8) * tr[module.TARGET].rank(pct=True)
                             + 0.8 * cr[module.TARGET]).rank(pct=True)
    for frame in (tr, cr, anchor):
        frame.insert(0, module.ID, studies)
    bag = pd.DataFrame({module.ID: studies, "Medial Meniscus": [0.8, 0.1, 0.5, 0.2],
                        "Lateral Meniscus": [0.1, 0.1, 0.1, 0.1]})
    return anchor, tr, cr, bag


def test_fixed_vote_changes_only_medial_meniscus_and_keeps_inputs():
    anchor, tr, cr, bag = frames()
    before = [frame.copy(deep=True) for frame in (anchor, tr, cr, bag)]
    result = module.apply_meniscus_overlay(anchor, tr, cr, bag)
    assert result[module.TARGET].tolist() == [1.0, 0.5, 0.75, 0.25]
    unchanged = [column for column in anchor if column != module.TARGET]
    pd.testing.assert_frame_equal(result[unchanged], anchor[unchanged], check_exact=True)
    for actual, expected in zip((anchor, tr, cr, bag), before):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_overlay_uses_study_order_instead_of_dataframe_index_labels():
    inputs = frames()
    expected = module.apply_meniscus_overlay(*inputs)
    inputs[1].index = [10, 11, 12, 13]
    inputs[2].index = [4, 5, 6, 7]
    inputs[3].index = [8, 9, 10, 11]
    result = module.apply_meniscus_overlay(*inputs)
    pd.testing.assert_frame_equal(result, expected, check_exact=True)


def test_anchor_csv_rounding_is_accepted_and_other_columns_stay_exact():
    studies = [str(index) for index in range(58)]
    raw = np.linspace(0.01, 0.99, 58)
    tr = pd.DataFrame({column: raw for column in module.TARGETS})
    cr = tr.copy()
    tr.insert(0, module.ID, studies)
    cr.insert(0, module.ID, studies)
    anchor = tr.copy()
    anchor[module.TARGET] = np.arange(1, 59, dtype=np.float64) / 58
    # The frozen anchor's release cell parses its staging CSV with pandas'
    # default reader, which changes 21 of these ranks by at most one ULP.
    anchor = pd.read_csv(io.StringIO(anchor.to_csv(index=False)), dtype={module.ID: str})
    bag = pd.DataFrame({module.ID: studies, "Medial Meniscus": raw,
                        "Lateral Meniscus": raw[::-1]})
    result = module.apply_meniscus_overlay(anchor, tr, cr, bag)
    unchanged = [column for column in anchor if column != module.TARGET]
    pd.testing.assert_frame_equal(result[unchanged], anchor[unchanged], check_exact=True)


@pytest.mark.parametrize("defect", ["order", "coverage", "nonfinite", "anchor_route", "schema"])
def test_incomplete_or_changed_inputs_are_fatal(defect):
    anchor, tr, cr, bag = frames()
    if defect == "order":
        bag = bag.iloc[::-1]
    elif defect == "coverage":
        bag = bag.iloc[:-1]
    elif defect == "nonfinite":
        bag.loc[0, module.TARGET] = np.nan
    elif defect == "anchor_route":
        anchor[module.TARGET] = anchor[module.TARGET].iloc[::-1].to_numpy()
    else:
        bag = bag.drop(columns=["Lateral Meniscus"])
    with pytest.raises(ValueError):
        module.apply_meniscus_overlay(anchor, tr, cr, bag)


def test_capture_keeps_hidden_size_memmap_after_anchor_unlinks_it(tmp_path):
    # A sparse mapping crosses the anchor's 1 GiB disk-cache threshold without
    # allocating or decoding a gigabyte of fixture pixels.
    filename = tmp_path / "pixels.npy"
    cache = np.lib.format.open_memmap(filename, mode="w+", dtype=np.uint8,
                                      shape=(140, 6, 12, 336, 336))
    cache[0, 0, 0, 0, 0] = 11
    cache[-1, -1, -1, -1, -1] = 77
    mask = np.ones((140, 6), dtype=np.float32)
    studies = [str(index) for index in range(140)]
    original = (studies, cache, mask)
    capture = module.MeniscusCacheCapture(lambda *args, **kwargs: original)
    assert capture.build_cache(None, tag="unchanged") is original
    cache.flush()
    filename.unlink()
    pixels, captured_mask, captured_studies = capture.take(studies)
    assert pixels is cache and captured_mask is mask and captured_studies == studies
    assert int(pixels[0, 0, 0, 0, 0]) == 11
    assert int(pixels[-1, -1, -1, -1, -1]) == 77
    capture.clear()
    with pytest.raises(ValueError):
        capture.take(studies)


def test_capture_rejects_ambiguous_cache_and_changed_study_order():
    pixels = np.zeros((1, 6, 12, 336, 336), dtype=np.uint8)
    mask = np.ones((1, 6), dtype=np.float32)
    capture = module.MeniscusCacheCapture(lambda: (["a"], pixels, mask))
    capture.build_cache()
    with pytest.raises(ValueError):
        capture.take(["b"])
    capture.build_cache()
    with pytest.raises(ValueError):
        capture.take(["a"])


@pytest.mark.parametrize("fails", [False, True])
def test_specialist_keeps_anchor_random_streams_on_success_and_failure(fails):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    result = {"status": "passed"}

    def inference():
        random.random()
        np.random.random(5)
        torch.randn(5)
        if fails:
            raise RuntimeError("inference failed")
        return result

    if fails:
        with pytest.raises(RuntimeError, match="inference failed"):
            module.run_preserving_random_state(inference)
    else:
        assert module.run_preserving_random_state(inference) is result
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    assert np.array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
