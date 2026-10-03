"""Restricted-load complete state for the separately identified disposable pilot.

Accepts CPU-converted state only. Actual XLA conversion, rank collection and
backend evidence belong to the admitted worker; this module cannot launch it.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .tpu_pilot_batches import global_source_exposures
from .tpu_pilot_state import _plain_cpu_copy, decode_host_rng

SCHEMA = 'rsna-ticket75-replicated-pilot-checkpoint-v1'
IDENTITY_KEYS = ('pilot_request_sha256', 'input_sha256', 'initial_state_sha256', 'scientific_plan_sha256')


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _sha(value: object) -> bool:
    return isinstance(value,str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def build_pilot_checkpoint(*, identity: Mapping[str,str], path: str, step: int, worker_generation: int,
    model_state: Mapping[str,Any], optimizer_state: Mapping[str,Any], scheduler_state: Mapping[str,Any],
    parameter_names: Sequence[Sequence[str]], config: Mapping[str,Any], ordered_masks: Sequence[torch.Tensor],
    rank_rng_states: Mapping[int,Mapping[str,Any]]) -> dict[str,Any]:
    """Preserve all eight host/device RNG states and the original global prefix."""
    exposures = global_source_exposures(ordered_masks,step)
    packet = dict(schema_version=SCHEMA, identity=dict(identity), path=path, successful_steps=step,
        worker_generation=worker_generation, world_size=8, disposable=True,
        original_fit_updates_credited=False, competition_submission=False, candidate_promotion=False,
        actual_TPU_execution_proven=False,
        model_state_dict=model_state, optimizer_state_dict=optimizer_state, scheduler_state_dict=scheduler_state,
        parameter_names=[list(group) for group in parameter_names], config=config,
        progress=dict(successful_optimizer_steps=step,attempted_batches=step,scheduler_steps=step,
            next_batch_index=step,sample_exposures=4*step,skipped_all_masked_batches=0,
            supervised_entry_exposures=sum(row['supervised_entry_exposures'] for row in exposures.values()),
            exposures_by_source=exposures),
        sampler_state=dict(kind='fixed-original-global-batches',next_batch_index=step,input_sha256=identity.get('input_sha256')),
        rank_rng_states=rank_rng_states)
    result = _plain_cpu_copy(packet)
    validate_pilot_checkpoint(result,expected_identity=identity,ordered_masks=ordered_masks)
    return result


def validate_pilot_checkpoint(packet: Mapping[str,Any], *, expected_identity: Mapping[str,str],
    ordered_masks: Sequence[torch.Tensor]) -> None:
    require(set(expected_identity) == set(IDENTITY_KEYS) and all(_sha(v) for v in expected_identity.values()),
        'Complete exact pilot/input/initial/scientific SHA identities required')
    fields = {'schema_version','identity','path','successful_steps','worker_generation','world_size','disposable',
        'original_fit_updates_credited','competition_submission','candidate_promotion','actual_TPU_execution_proven',
        'model_state_dict','optimizer_state_dict','scheduler_state_dict','parameter_names','config','progress','sampler_state','rank_rng_states'}
    require(set(packet) == fields and packet['schema_version'] == SCHEMA and packet['identity'] == dict(expected_identity)
        and packet['world_size'] == 8 and type(packet['world_size']) is int and packet['disposable'] is True
        and all(packet[k] is False for k in ('original_fit_updates_credited','competition_submission','candidate_promotion','actual_TPU_execution_proven')),
        'Separate disposable mechanics identity; packet alone cannot prove TPU execution')
    step,path,generation = packet['successful_steps'],packet['path'],packet['worker_generation']
    require(type(step) is int and step in (0,3,8,16) and path in ('continuous','staged')
        and type(generation) is int and generation == int(path == 'staged' and step == 16),
        'Exact declared checkpoint boundary and fresh-worker generation required')
    # Reject unsupported objects/devices and nonfinite tensors before inspecting
    # state structure. Validation never consumes global RNG or mutates the packet.
    _plain_cpu_copy(packet)
    exposures = global_source_exposures(ordered_masks,step)
    require(all(type(packet['progress'].get(k)) is int for k in ('successful_optimizer_steps','attempted_batches','scheduler_steps',
        'next_batch_index','sample_exposures','skipped_all_masked_batches','supervised_entry_exposures')), 'Integer original progress required')
    require(packet['progress'] == dict(successful_optimizer_steps=step,attempted_batches=step,scheduler_steps=step,
        next_batch_index=step,sample_exposures=4*step,skipped_all_masked_batches=0,
        supervised_entry_exposures=sum(row['supervised_entry_exposures'] for row in exposures.values()),exposures_by_source=exposures),
        'Complete unchanged original real-source progress/exposures required')
    require(packet['sampler_state'] == dict(kind='fixed-original-global-batches',next_batch_index=step,
        input_sha256=expected_identity['input_sha256']), 'Exact fixed original batch cursor/input required')
    rngs = packet['rank_rng_states']
    require(type(rngs) is dict and len(rngs) == 8 and all(type(rank) is int for rank in rngs) and set(rngs) == set(range(8)),
        'All eight distinct original rank RNG states required')
    for rng in rngs.values():
        require(type(rng) is dict and set(rng) == {'host','xla_seed'} and type(rng['xla_seed']) is int,
            'Complete host RNG and pinned XLA integer RNG state required')
        decode_host_rng(rng['host'])
    model,optimizer,scheduler = packet['model_state_dict'],packet['optimizer_state_dict'],packet['scheduler_state_dict']
    names = packet['parameter_names']
    require(isinstance(model,Mapping) and bool(model) and isinstance(packet['config'],Mapping)
        and type(names) is list and len(names) == 2 and all(type(group) is list and bool(group) for group in names),
        'Complete model/config and encoder/head parameter ordering required')
    flat = [name for group in names for name in group]
    require(all(type(name) is str for name in flat) and len(flat) == len(set(flat))
        and all(name.startswith(prefix) for group,prefix in zip(names,('encoder.','head.'),strict=True) for name in group),
        'Distinct original encoder/head parameter names required')
    require(all(type(name) is str and isinstance(value,torch.Tensor)
        and value.dtype == (torch.int64 if name.endswith('.num_batches_tracked') else torch.float32)
        for name,value in model.items()), 'Exact original FP32 model/buffer and int64 normalization-counter dtypes required')
    nonparameters = {'imagenet_mean','imagenet_std'} | {name for name in model
        if name.endswith(('.running_mean','.running_var','.num_batches_tracked'))}
    require(flat == [name for name in model if name not in nonparameters], 'Every original model parameter must belong to ordinary Adam')
    require(all(name in model and isinstance(model[name],torch.Tensor) and model[name].dtype == torch.float32 for name in flat),
        'Complete original FP32 parameter tensors required')
    require(isinstance(optimizer,Mapping) and set(optimizer) == {'state','param_groups'} and len(optimizer['param_groups']) == 2,
        'Complete ordinary Adam state and original two groups required')
    identifiers = []
    for group,group_names,lr in zip(optimizer['param_groups'],names,(1e-5,1e-4),strict=True):
        require(len(group['params']) == len(group_names) and group['lr'] == lr and group['betas'] == (.9,.999)
            and group['eps'] == 1e-8 and group['weight_decay'] == 0 and group['amsgrad'] is False
            and group['maximize'] is False and group['capturable'] is False and group['differentiable'] is False
            and group['foreach'] is None and group['fused'] is None and group.get('decoupled_weight_decay',False) is False,
            'Exact original ordinary Adam scope/defaults/rates required')
        identifiers.extend(group['params'])
    require(all(type(i) is int for i in identifiers) and identifiers == list(range(len(flat))),
        'Distinct original optimizer parameter identities required')
    states = optimizer['state']
    require(set(states) == (set(identifiers) if step else set()), 'No missing/extra Adam parameter state allowed')
    for identifier,name in zip(identifiers,flat,strict=True):
        if not step:
            continue
        state = states[identifier]
        require(set(state) == {'step','exp_avg','exp_avg_sq'} and isinstance(state['step'],torch.Tensor)
            and state['step'].dtype == torch.float32 and state['step'].shape == () and float(state['step']) == step,
            'Exact ordinary Adam scalar step and complete moments required')
        for key in ('exp_avg','exp_avg_sq'):
            require(isinstance(state[key],torch.Tensor) and state[key].shape == model[name].shape and state[key].dtype == torch.float32,
                'Every original parameter requires complete FP32 Adam moment shapes')
        require(bool((state['exp_avg_sq'] >= 0).all()), 'Nonnegative Adam second moments required')
    require(isinstance(scheduler,Mapping) and scheduler.get('last_epoch') == step and scheduler.get('_step_count') == step + 1
        and scheduler.get('base_lrs') == [1e-5,1e-4] and scheduler.get('_last_lr') == [1e-5,1e-4]
        and scheduler.get('lr_lambdas') == [None,None], 'Exact constant original scheduler state required')


def publish_pilot_checkpoint(packet: Mapping[str,Any], destination: str | Path, *, expected_identity: Mapping[str,str],
    ordered_masks: Sequence[torch.Tensor], byte_limit: int) -> str:
    """One immutable atomic CPU publication; no overwrite or partial retry."""
    validate_pilot_checkpoint(packet,expected_identity=expected_identity,ordered_masks=ordered_masks)
    return _publish_validated_cpu_packet(packet,destination,byte_limit=byte_limit)


def _publish_validated_cpu_packet(packet: Mapping[str,Any], destination: str | Path, *, byte_limit: int) -> str:
    """Shared atomic writer; callers must complete their own schema admission."""
    require(type(byte_limit) is int and byte_limit > 0, 'Actual admitted checkpoint byte bound required')
    path = Path(destination)
    temporary = path.with_name(path.name + '.tmp')
    require(not path.exists() and not temporary.exists(), 'Consumed or partial checkpoint publication; preserve without retry')
    with temporary.open('xb') as stream:
        torch.save(packet,stream)
        stream.flush();os.fsync(stream.fileno())
    require(temporary.stat().st_size <= byte_limit, 'Checkpoint exceeded admitted byte bound; preserve partial failure')
    with temporary.open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    os.link(temporary,path)
    temporary.unlink()
    directory = os.open(path.parent,os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return digest


def load_pilot_checkpoint(path: str | Path, *, expected_sha256: str, expected_identity: Mapping[str,str],
    ordered_masks: Sequence[torch.Tensor]) -> dict[str,Any]:
    require(_sha(expected_sha256), 'Exact published checkpoint SHA required')
    # Hash and restricted load the same open file to avoid a path replacement gap.
    with Path(path).open('rb') as stream:
        require(hashlib.file_digest(stream,'sha256').hexdigest() == expected_sha256, 'Published checkpoint bytes changed')
        stream.seek(0)
        packet = torch.load(stream,map_location='cpu',weights_only=True)
    validate_pilot_checkpoint(packet,expected_identity=expected_identity,ordered_masks=ordered_masks)
    return packet


def _exact(a: Any, b: Any, label: str) -> None:
    require(type(a) is type(b), f'Exact replay type mismatch: {label}')
    if isinstance(a,torch.Tensor):
        require(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a,b), f'Tensor-exact replay failed: {label}')
    elif isinstance(a,Mapping):
        require(a.keys() == b.keys(), f'Exact replay mapping mismatch: {label}')
        for key in a:
            _exact(a[key],b[key],f'{label}.{key}')
        if hasattr(a,'_metadata') or hasattr(b,'_metadata'):
            _exact(getattr(a,'_metadata',None),getattr(b,'_metadata',None),f'{label}._metadata')
    elif isinstance(a,(tuple,list)):
        require(len(a) == len(b), f'Exact replay sequence mismatch: {label}')
        for index,(left,right) in enumerate(zip(a,b,strict=True)):
            _exact(left,right,f'{label}.{index}')
    else:
        require(a == b, f'Exact replay value mismatch: {label}')


def verify_exact_pilot_replay(continuous: Mapping[str,Any], staged: Mapping[str,Any], *, expected_identity: Mapping[str,str],
    ordered_masks: Sequence[torch.Tensor]) -> dict[str,Any]:
    for packet,path in ((continuous,'continuous'),(staged,'staged')):
        validate_pilot_checkpoint(packet,expected_identity=expected_identity,ordered_masks=ordered_masks)
        require(packet['path'] == path and packet['successful_steps'] == 16, 'Both exact final declared paths required')
    for key in ('model_state_dict','optimizer_state_dict','scheduler_state_dict','parameter_names','config','progress','sampler_state','rank_rng_states'):
        _exact(continuous[key],staged[key],key)
    return dict(tensor_exact_replay=True,all_eight_rank_rng_states_exact=True,actual_TPU_execution_proven=False,
        original_fit_updates_credited=False,competition_submission=False,candidate_promotion=False)
