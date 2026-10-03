"""Lossless immutable-layout input bank for the disposable16-batch pilot.

Validates frozen file bytes and original packing; actual MRI replay, completed
CPU job/source identity and accelerator admission must be established separately.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .targets import TARGET_COLUMNS
from .tpu_pilot_inputs import validate_pilot_study_arrays

SCHEMA = 'rsna-ticket75-lossless-input-bank-v1'
FIELDS = ('images','slice_mask','series_mask','series_descriptors','targets','supervision_mask')


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _sha(value: object) -> bool:
    return isinstance(value,str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def _stamp(path: Path) -> tuple[int,int,int,int,int]:
    value = path.stat()
    return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns


class FrozenPilotInputBank:
    def __init__(self, root: str | Path, *, expected_manifest_sha256: str, manifest_path: str | Path | None = None) -> None:
        self.root = Path(root).resolve()
        _require(_sha(expected_manifest_sha256), 'Exact admitted bank-manifest SHA required')
        explicit_sidecar = manifest_path is not None
        manifest_path = Path(manifest_path).resolve() if manifest_path is not None else self.root/'input-bank-manifest.json'
        # An explicitly hash-bound preparation sidecar can live outside the
        # immutable original array directory; deployed banks use the default.
        _require(manifest_path.is_file(), 'Exact bank manifest file required')
        _require(explicit_sidecar or manifest_path.resolve().is_relative_to(self.root),
            'Default bank manifest must remain inside the admitted bank')
        with manifest_path.open('rb') as stream:
            _require(hashlib.file_digest(stream,'sha256').hexdigest() == expected_manifest_sha256, 'Frozen bank manifest bytes changed')
            stream.seek(0);manifest = json.load(stream)
        _require(set(manifest) == {'schema_version','selection','arrays'} and manifest['schema_version'] == SCHEMA,
            'Exact lossless input-bank manifest schema required')
        selection_binding = manifest['selection']
        _require(selection_binding['path'] == 'selection.json' and set(selection_binding) == {'path','sha256'}
            and _sha(selection_binding['sha256']), 'Original frozen selection byte binding required')
        selection_path = self.root/'selection.json'
        _require(selection_path.resolve().is_relative_to(self.root), 'Selection must remain inside the admitted bank')
        with selection_path.open('rb') as stream:
            _require(hashlib.file_digest(stream,'sha256').hexdigest() == selection_binding['sha256'], 'Frozen original selection bytes changed')
            stream.seek(0);selected = json.load(stream)
        _require(selected['schema_version'] == 'rsna-ticket75-fixed-pilot-selection-v1' and type(selected['fold']) is int and selected['fold'] == 0
            and selected['successful_updates_per_path'] == 16 and selected['target_columns'] == list(TARGET_COLUMNS)
            and selected['real_source_roles'] == ['reference','reference','weak','weak'] and selected['padding_ranks'] == [4,5,6,7]
            and _sha(selected['reference_pool_sha256']) and _sha(selected['report_pool_sha256']),
            'Exact original fold0 pools/roles/target order/16-update selection required')
        order = selected['study_order'];batches = selected['batch_study_indices']
        _require(type(order) is list and len(order) == 64 and all(type(uid) is str and bool(uid) for uid in order)
            and len(set(order)) == 64 and type(batches) is list and len(batches) == 16
            and all(type(batch) is list and len(batch) == 4 and all(type(i) is int and i in range(64) for i in batch) for batch in batches)
            and sorted(i for batch in batches for i in batch) == list(range(64)),
            'All64 distinct original draws and unchanged16 four-study batches required')
        self.study_order = tuple(order)
        self._batches = tuple(tuple(batch) for batch in batches)
        rows = manifest['arrays']
        _require(type(rows) is list and len(rows) == 64*len(FIELDS), 'Complete64-study six-field byte inventory required')
        self._paths: dict[tuple[int,str],Path] = {}
        self._stamps: dict[Path,tuple[int,int,int,int,int]] = {}
        self._hashes: dict[Path,str] = {}
        total_bytes = 0
        for row in rows:
            _require(type(row) is dict and set(row) == {'study_index','field','path','sha256','bytes'}
                and type(row['study_index']) is int and row['study_index'] in range(64) and row['field'] in FIELDS
                and row['path'] == f"studies/{row['study_index']:02d}/{row['field']}.npy"
                and _sha(row['sha256']) and type(row['bytes']) is int and row['bytes'] > 0,
                'Exact canonical lossless array byte binding required')
            key = row['study_index'],row['field']
            _require(key not in self._paths, 'Duplicate/missing original study field')
            path = self.root/row['path']
            _require(path.resolve().is_relative_to(self.root) and path.is_file(), 'Original study array must stay in the admitted bank')
            stamp = _stamp(path)
            _require(stamp[2] == row['bytes'] and _digest(path) == row['sha256'], 'Frozen original array bytes changed')
            self._paths[key] = path;self._stamps[path] = stamp;self._hashes[path] = row['sha256']
            total_bytes += row['bytes']
        _require(total_bytes < 2_000_000_000, 'Lossless MRI/target/mask bank alone exceeds complete attachment limit')
        self._metadata_stamps = {manifest_path:_stamp(manifest_path),selection_path:_stamp(selection_path)}
        self._metadata_hashes = {manifest_path:expected_manifest_sha256,selection_path:selection_binding['sha256']}
        for index in range(64):
            validate_pilot_study_arrays(self.study_arrays(index))
        self.manifest_sha256 = expected_manifest_sha256
        self.array_bytes = total_bytes

    def _unchanged(self) -> None:
        _require(all(_stamp(path) == stamp for path,stamp in self._metadata_stamps.items()), 'Admitted bank metadata changed after verification')

    def study_arrays(self, index: int) -> dict[str,np.ndarray]:
        _require(type(index) is int and index in range(64), 'One exact original selected study index required')
        self._unchanged()
        result = {}
        for name in FIELDS:
            path = self._paths[index,name]
            _require(_stamp(path) == self._stamps[path], 'Original array changed after byte verification')
            value = np.load(path,mmap_mode='r',allow_pickle=False)
            _require(not value.flags.writeable and _stamp(path) == self._stamps[path], 'Original arrays must remain unchanged read-only mappings')
            result[name] = value
        return result

    def global_batch(self, index: int) -> dict[str,torch.Tensor]:
        _require(type(index) is int and index in range(16), 'Exact original0..15 global batch index required; never resample a subset')
        studies = [self.study_arrays(i) for i in self._batches[index]]
        # Stack copies isolate mutable host/device preparation from the original
        # mmap bytes; preserve field dtypes, uncertainty masks and original order.
        return {name:torch.from_numpy(np.stack([study[name] for study in studies])) for name in FIELDS}

    def replay_inputs(self) -> dict[str,torch.Tensor]:
        studies = [self.study_arrays(i) for i in range(8)]
        return {name:torch.from_numpy(np.stack([study[name] for study in studies])) for name in FIELDS[:4]}

    def ordered_masks(self) -> list[torch.Tensor]:
        return [torch.from_numpy(np.stack([self.study_arrays(i)['supervision_mask']
            for i in self._batches[step]])) for step in range(16)]

    def verify_final_bytes(self) -> dict[str,Any]:
        """Final full hash check; timestamps alone are not final byte evidence."""
        self._unchanged()
        _require(all(_digest(path) == digest for path,digest in self._metadata_hashes.items()), 'Frozen bank metadata bytes changed during execution')
        _require(all(_stamp(path) == self._stamps[path] and _digest(path) == digest for path,digest in self._hashes.items()),
            'Original bank bytes changed during execution')
        return dict(manifest_sha256=self.manifest_sha256,original_array_files=384,original_study_count=64,
            actual_array_bytes=self.array_bytes,lossless_bytes_reverified=True,original_MRI_identity_verified=False,
            actual_TPU_execution_proven=False,competition_submission=False,candidate_promotion=False)
