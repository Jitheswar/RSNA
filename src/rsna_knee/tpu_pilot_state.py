"""Lossless, restricted-load state packets for the disposable TPU pilot.

Original CPU checkpoints remain immutable. Conversion happens on the trusted
source host; no NumPy pickle globals need to be permitted on the remote host.
This module does not acquire devices, train, upload, reserve or score anything.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
import math
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

SCHEMA = 'rsna-ticket75-portable-cpu-reference-v1'
RNG_SCHEMA = 'rsna-ticket75-portable-host-rng-v1'
BOUNDARIES = (0, 3, 8, 16)


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def encode_host_rng(state: Mapping[str, Any]) -> dict[str, Any]:
    """Keep every Python/MT19937/Torch bit as plain values and CPU tensors."""
    _require(set(state) == {'python', 'numpy', 'torch'}, 'Original CPU-only complete host RNG required')
    python = state['python']
    _require(isinstance(python, tuple) and len(python) == 3, 'Complete Python RNG tuple required')
    # Validate using a private generator; admission must not consume global RNG.
    local = random.Random()
    try:
        local.setstate(python)
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid original Python RNG') from exc
    numpy = state['numpy']
    _require(isinstance(numpy, tuple) and len(numpy) == 5, 'Complete original NumPy MT19937 state required')
    algorithm, keys, position, gaussian_present, gaussian = numpy
    _require(algorithm == 'MT19937' and isinstance(keys, np.ndarray) and keys.shape == (624,)
        and keys.dtype == np.dtype('uint32'), 'Unchanged NumPy MT19937624-word state required')
    _require(type(position) is int and 0 <= position <= 624 and type(gaussian_present) is int
        and gaussian_present in (0, 1) and type(gaussian) is float and math.isfinite(gaussian), 'Complete NumPy RNG cursor/cache required')
    torch_state = state['torch']
    _require(isinstance(torch_state, torch.Tensor) and torch_state.device.type == 'cpu'
        and torch_state.dtype == torch.uint8 and torch_state.ndim == 1 and torch_state.numel() > 0,
        'Original CPU Torch RNG byte vector required')
    local_torch = torch.Generator(device='cpu')
    try:
        local_torch.set_state(torch_state)
    except RuntimeError as exc:
        raise ValueError('Invalid original Torch RNG') from exc
    return dict(schema_version=RNG_SCHEMA, python=python,
        numpy=dict(algorithm=algorithm, keys=torch.from_numpy(keys.astype(np.int64)),
            original_dtype='uint32', position=position, gaussian_present=gaussian_present, gaussian=gaussian),
        torch=torch_state.detach().clone())


def decode_host_rng(packet: Mapping[str, Any]) -> dict[str, Any]:
    _require(set(packet) == {'schema_version', 'python', 'numpy', 'torch'}
        and packet['schema_version'] == RNG_SCHEMA, 'Exact portable host RNG schema required')
    numpy = packet['numpy']
    _require(isinstance(numpy, dict) and set(numpy) == {'algorithm', 'keys', 'original_dtype', 'position', 'gaussian_present', 'gaussian'},
        'Complete portable NumPy RNG required')
    keys = numpy['keys']
    _require(numpy['original_dtype'] == 'uint32' and isinstance(keys, torch.Tensor)
        and keys.device.type == 'cpu' and keys.dtype == torch.int64 and keys.shape == (624,)
        and bool(((keys >= 0) & (keys <= 2**32 - 1)).all()), 'Exact lossless uint32 RNG words required')
    result = dict(python=packet['python'], numpy=(numpy['algorithm'], keys.numpy().astype(np.uint32),
        numpy['position'], numpy['gaussian_present'], numpy['gaussian']), torch=packet['torch'])
    encode_host_rng(result)  # Validate all cursor/cache/Python/Torch fields too.
    return result


def capture_host_rng() -> dict[str, Any]:
    return encode_host_rng(dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state()))


def restore_host_rng(packet: Mapping[str, Any]) -> None:
    state = decode_host_rng(packet)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])


def _plain_cpu_copy(value: Any) -> Any:
    """Reject executable objects and accelerator tensors instead of allowlisting."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        _require(math.isfinite(value), 'Finite checkpoint scalar required')
        return value
    if isinstance(value, torch.Tensor):
        _require(type(value) is torch.Tensor and value.device.type == 'cpu', 'Only ordinary CPU tensors may enter a portable packet')
        _require(not value.is_floating_point() or bool(torch.isfinite(value).all()), 'Finite CPU checkpoint tensor required')
        return value.detach().clone()
    if isinstance(value, Mapping):
        _require(type(value) in (dict, OrderedDict), 'Plain checkpoint mappings required')
        _require(all(type(k) in (str, int) for k in value), 'Plain string/integer checkpoint keys required')
        result = {k: _plain_cpu_copy(v) for k, v in value.items()}
        if type(value) is OrderedDict:
            _require(set(vars(value)) <= {'_metadata'}, 'Unknown ordered checkpoint metadata required separate admission')
            ordered = OrderedDict(result)
            if hasattr(value, '_metadata'):
                setattr(ordered, '_metadata', _plain_cpu_copy(getattr(value, '_metadata')))
            return ordered
        return result
    if type(value) in (tuple, list):
        return type(value)(_plain_cpu_copy(v) for v in value)
    raise ValueError(f'Unsupported portable checkpoint value: {type(value).__name__}')


def portable_cpu_reference(payload: Mapping[str, Any], *, parent_sha256: str, step: int) -> dict[str, Any]:
    """Derive a lossless reference packet, without relabeling it as TPU output."""
    _require(_sha(parent_sha256) and type(step) is int and step in BOUNDARIES, 'Exact source SHA and declared CPU reference boundary required')
    _require(isinstance(payload.get('model_state_dict'), Mapping) and bool(payload['model_state_dict'])
        and isinstance(payload.get('config'), Mapping) and isinstance(payload.get('rng_state'), Mapping), 'Complete original reference state required')
    if step == 0:
        _require(payload.get('disposable') is True and _sha(payload.get('initial_model_state_sha256')),
            'Exact disposable original initial-model reference required')
        _require('optimizer_state_dict' not in payload and 'progress' not in payload, 'Initial state cannot claim learned updates')
    else:
        progress = payload.get('progress')
        _require(isinstance(progress, Mapping) and all(type(progress.get(k)) is int and progress[k] == step for k in
            ('successful_optimizer_steps', 'attempted_batches', 'scheduler_steps', 'next_batch_index'))
            and progress.get('sample_exposures') == 4 * step and progress.get('skipped_all_masked_batches') == 0,
            'Exact original real-source CPU reference boundary required')
        _require(isinstance(payload.get('optimizer_state_dict'), Mapping)
            and isinstance(payload.get('scheduler_state_dict'), Mapping) and isinstance(payload.get('sampler_state'), Mapping),
            'Complete original Adam/scheduler/sampler reference required')
        _require(payload.get('identity', {}).get('disposable') is True
            and payload['identity'].get('original_fit_updates_credited') is False, 'Independent disposable CPU reference identity required')
    source = dict(payload)
    source['rng_state'] = encode_host_rng(source['rng_state'])
    plain = _plain_cpu_copy(source)
    return dict(schema_version=SCHEMA, original_checkpoint_sha256=parent_sha256, successful_steps=step,
        original_CPU_reference=True, actual_TPU_execution=False, original_fit_updates_credited=False,
        competition_submission=False, candidate_promotion=False, payload=plain)


def unpack_cpu_reference(packet: Mapping[str, Any], *, parent_sha256: str, step: int) -> dict[str, Any]:
    _require(set(packet) == {'schema_version', 'original_checkpoint_sha256', 'successful_steps', 'original_CPU_reference',
        'actual_TPU_execution', 'original_fit_updates_credited', 'competition_submission', 'candidate_promotion', 'payload'}
        and packet['schema_version'] == SCHEMA and packet['original_checkpoint_sha256'] == parent_sha256
        and type(packet['successful_steps']) is int and packet['successful_steps'] == step
        and packet['original_CPU_reference'] is True and all(packet[k] is False for k in
            ('actual_TPU_execution', 'original_fit_updates_credited', 'competition_submission', 'candidate_promotion')),
        'Exact original portable CPU reference identity required')
    payload = dict(packet['payload'])
    payload['rng_state'] = decode_host_rng(payload['rng_state'])
    portable_cpu_reference(payload, parent_sha256=parent_sha256, step=step)
    return payload


def load_cpu_reference(path: str | Path, *, packet_sha256: str, parent_sha256: str, step: int) -> dict[str, Any]:
    _require(_sha(packet_sha256) and _sha(parent_sha256), 'Exact packet/source byte identities required')
    with Path(path).open('rb') as stream:
        _require(hashlib.file_digest(stream, 'sha256').hexdigest() == packet_sha256, 'Portable reference file changed')
    packet = torch.load(path, map_location='cpu', weights_only=True)
    return unpack_cpu_reference(packet, parent_sha256=parent_sha256, step=step)
