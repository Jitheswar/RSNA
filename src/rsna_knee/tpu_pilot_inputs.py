"""Fixed original-pool selection and lossless input contracts for the TPU pilot.

These functions admit metadata/array structure only. Original MRI source replay,
model identity, resource approval and actual TPU execution remain separate gates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .targets import NUM_TARGETS, TARGET_COLUMNS
from .training import _batch_indices

PILOT_STEPS = 16
REFERENCE_SEED = 20260912
REPORT_SEED = REFERENCE_SEED + 34034034


@dataclass(frozen=True)
class PilotSelection:
    """Immutable four-study batches from the complete original ordered pools."""
    study_order: tuple[str, ...]
    batch_study_indices: tuple[tuple[int, ...], ...]
    pool_indices: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    reference_pool_sha256: str
    report_pool_sha256: str

    def to_private_dict(self) -> dict[str, Any]:
        return dict(schema_version='rsna-ticket75-fixed-pilot-selection-v1', fold=0,
            study_order=list(self.study_order), batch_study_indices=[list(x) for x in self.batch_study_indices],
            original_pool_indices=[dict(reference=list(r), report=list(w)) for r, w in self.pool_indices],
            reference_pool_sha256=self.reference_pool_sha256, report_pool_sha256=self.report_pool_sha256,
            target_columns=list(TARGET_COLUMNS), successful_updates_per_path=PILOT_STEPS,
            real_source_roles=['reference', 'reference', 'weak', 'weak'], padding_ranks=[4, 5, 6, 7],
            candidate_promotion=False, competition_submission=False, original_MRI_verified=False)


def _pool_digest(pool: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(pool), separators=(',', ':')).encode()).hexdigest()


def select_original_pilot_batches(reference: Sequence[str], report: Sequence[str],
                                  held_out: Sequence[str]) -> PilotSelection:
    """Never resample a selected subset or infer source eligibility from its size."""
    if len(reference) != 38 or len(report) != 2731:
        raise ValueError('Complete original fold0 pools of38 reference/2731 reports required')
    if any(not isinstance(uid, str) or not uid for uid in (*reference, *report)):
        raise ValueError('Nonempty original study identities required')
    if len(set(reference)) != len(reference) or len(set(report)) != len(report):
        raise ValueError('Original pools must contain unique study identities')
    if set(reference) & set(report) or set(held_out) & (set(reference) | set(report)):
        raise ValueError('Original sources must be disjoint and exclude every held-out study')
    batches = []
    indices = []
    for attempt in range(PILOT_STEPS):
        r = _batch_indices(len(reference), 2, attempt, REFERENCE_SEED)
        w = _batch_indices(len(report), 2, attempt, REPORT_SEED)
        indices.append((r, w))
        batches.append(tuple(reference[i] for i in r) + tuple(report[i] for i in w))
    order = tuple(dict.fromkeys(uid for batch in batches for uid in batch))
    if len(order) != 64:
        raise ValueError('Fixed original first16 batches must retain all64 distinct draws')
    position = {uid: index for index, uid in enumerate(order)}
    return PilotSelection(order, tuple(tuple(position[uid] for uid in batch) for batch in batches),
        tuple(indices), _pool_digest(reference), _pool_digest(report))


def validate_pilot_study_arrays(value: Mapping[str, np.ndarray]) -> None:
    """Reject conversion/resizing, lost uncertainty masks, or malformed packing."""
    specs = dict(images=((16, 8, 1, 224, 224), np.dtype('float32')),
        slice_mask=((16, 8), np.dtype('bool')), series_mask=((16,), np.dtype('bool')),
        series_descriptors=((16, 10), np.dtype('float32')), targets=((NUM_TARGETS,), np.dtype('float32')),
        supervision_mask=((NUM_TARGETS,), np.dtype('bool')))
    if set(value) != set(specs):
        raise ValueError('Complete original pixel/descriptor/target/mask fields required')
    for name, (shape, dtype) in specs.items():
        array = value[name]
        if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype != dtype:
            raise ValueError(f'Original fixed shape and dtype required for {name}')
    if not np.isfinite(value['images']).all() or not np.isfinite(value['series_descriptors']).all():
        raise ValueError('Original packed pixels and descriptors must be finite')
    if not (value['slice_mask'] & value['series_mask'][:, None]).any():
        raise ValueError('A real training study must have admitted MRI slices')
    mask = value['supervision_mask']
    targets = value['targets'][mask]
    if not targets.size or not np.isfinite(targets).all() or not ((targets >= 0) & (targets <= 1)).all():
        raise ValueError('Nonempty finite bounded original supervision required')


def attachment_upper_bound(*, initial_state_bytes: int, step3_state_bytes: int,
                           code_and_metadata_bytes: int) -> int:
    """Reserve raw FP32 geometry; compressed size cannot substitute for admission."""
    numbers = (initial_state_bytes, step3_state_bytes, code_and_metadata_bytes)
    if any(type(x) is not int or x < 0 for x in numbers) or not initial_state_bytes or not step3_state_bytes:
        raise ValueError('Actual positive state sizes and nonnegative metadata bytes required')
    total = 64 * 16 * 8 * 224 * 224 * 4 + sum(numbers)
    if total > 2_000_000_000:
        raise ValueError('Complete lossless attachment exceeds the2GB hard envelope')
    return total
