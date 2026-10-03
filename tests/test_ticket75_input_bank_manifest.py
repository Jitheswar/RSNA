"""Preparation fail-closed guards; no fabricated successful CPU/MRI proof."""
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec = importlib.util.spec_from_file_location('prepare_ticket75_input_bank_manifest',SCRIPTS/'prepare_ticket75_input_bank_manifest.py')
preparation = importlib.util.module_from_spec(spec);spec.loader.exec_module(preparation)


def test_rejected_source_chain_creates_no_preparation_output(monkeypatch,tmp_path):
    root = tmp_path/'uncreated-sidecar'
    monkeypatch.setattr(preparation,'ROOT',root)
    def reject():
        raise ValueError('Actual completed CPU preparation required')
    monkeypatch.setattr(preparation,'verified_sources',reject)
    with pytest.raises(ValueError,match='Actual completed CPU preparation required'):
        preparation.prepare()
    assert not root.exists()


def test_partial_preparation_cannot_retry_or_consume_source(monkeypatch,tmp_path):
    root = tmp_path/'partial-sidecar';root.mkdir()
    marker = root/'retained-failure';marker.write_bytes(b'preserve')
    monkeypatch.setattr(preparation,'ROOT',root)
    def unexpected():
        raise AssertionError('Source must not be consumed on a partial retry')
    monkeypatch.setattr(preparation,'verified_sources',unexpected)
    with pytest.raises(ValueError,match='never retry partial preparation'):
        preparation.prepare()
    assert marker.read_bytes() == b'preserve'
