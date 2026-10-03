"""Model-only CPU delivery derived from an exact complete pilot checkpoint.

No optimizer/RNG state, native decoding, upload or scoring is performed here.
Actual backend, source replay and license-file verification remain external.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from .full_backbone import ResNet18FullBackboneStudyModel
from .model import ModelConfig
from .resnet18 import RESNET18_FULL_FINETUNE_MODEL_ID
from .tpu_pilot_checkpoint import (IDENTITY_KEYS,_publish_validated_cpu_packet,load_pilot_checkpoint,require,_sha)
from .tpu_pilot_state import _plain_cpu_copy

SCHEMA = 'rsna-ticket75-model-only-export-v1'


def _inventory(value: Mapping[str,Any]) -> None:
    require(set(value) == {'source_bindings','license_bindings'}, 'Complete source/license inventory required')
    for key in ('source_bindings','license_bindings'):
        rows = value[key]
        require(type(rows) is list and bool(rows), 'Nonempty frozen source/license binding inventory required')
        require(all(type(row) is dict and set(row) == {'name','sha256'} and type(row['name']) is str and bool(row['name'])
            and _sha(row['sha256']) for row in rows), 'Exact named SHA-bound inventory required')
        require(len({row['name'] for row in rows}) == len(rows), 'Unique source/license inventory names required')


def derive_model_export(checkpoint_path: str | Path, *, checkpoint_sha256: str, expected_identity: Mapping[str,str],
    ordered_masks: list[torch.Tensor], expected_config: Mapping[str,Any], provenance: Mapping[str,Any]) -> dict[str,Any]:
    checkpoint = load_pilot_checkpoint(checkpoint_path,expected_sha256=checkpoint_sha256,
        expected_identity=expected_identity,ordered_masks=ordered_masks)
    require(checkpoint['successful_steps'] == 16 and checkpoint['config'] == expected_config,
        'Exact final step16 checkpoint and frozen config required for model-only delivery')
    packet = _plain_cpu_copy(dict(schema_version=SCHEMA,identity=dict(expected_identity),parent_checkpoint_sha256=checkpoint_sha256,
        path=checkpoint['path'],successful_steps=16,config=checkpoint['config'],model_state_dict=checkpoint['model_state_dict'],
        parameter_names=checkpoint['parameter_names'],provenance=dict(provenance),disposable=True,
        original_fit_updates_credited=False,actual_TPU_execution_proven=False,
        original_MRI_replay_verified=False,competition_submission=False,candidate_promotion=False))
    validate_model_export(packet,expected_identity=expected_identity,expected_checkpoint_sha256=checkpoint_sha256,
        expected_config=expected_config,expected_provenance=provenance)
    return packet


def validate_model_export(packet: Mapping[str,Any], *, expected_identity: Mapping[str,str],expected_checkpoint_sha256: str,
    expected_config: Mapping[str,Any],expected_provenance: Mapping[str,Any]) -> None:
    require(set(expected_identity) == set(IDENTITY_KEYS) and all(_sha(v) for v in expected_identity.values())
        and _sha(expected_checkpoint_sha256), 'Exact source and parent-checkpoint identities required')
    fields = {'schema_version','identity','parent_checkpoint_sha256','path','successful_steps','config','model_state_dict',
        'parameter_names','provenance','disposable','original_fit_updates_credited','actual_TPU_execution_proven',
        'original_MRI_replay_verified','competition_submission','candidate_promotion'}
    require(set(packet) == fields and packet['schema_version'] == SCHEMA and packet['identity'] == dict(expected_identity)
        and packet['parent_checkpoint_sha256'] == expected_checkpoint_sha256 and packet['path'] in ('continuous','staged')
        and type(packet['successful_steps']) is int and packet['successful_steps'] == 16 and packet['disposable'] is True
        and all(packet[key] is False for key in ('original_fit_updates_credited','actual_TPU_execution_proven',
            'original_MRI_replay_verified','competition_submission','candidate_promotion')),
        'Separate unpromoted final pilot export required; packet alone proves no backend/native replay')
    _plain_cpu_copy(packet)
    require(packet['config'] == expected_config and packet['provenance'] == expected_provenance,
        'Exact frozen model/config/source/license inventory required')
    _inventory(expected_provenance)
    require(isinstance(expected_config,Mapping) and 'model' in expected_config, 'Complete frozen model configuration required')
    config = ModelConfig.from_dict(expected_config['model'])
    require(config.model_id == RESNET18_FULL_FINETUNE_MODEL_ID, 'Original full-backbone delivery architecture required')
    state,names = packet['model_state_dict'],packet['parameter_names']
    require(isinstance(state,Mapping) and bool(state) and type(names) is list and len(names) == 2
        and all(type(group) is list and bool(group) for group in names), 'Complete original model and parameter scope required')
    require(all(type(name) is str and isinstance(value,torch.Tensor)
        and value.dtype == (torch.int64 if name.endswith('.num_batches_tracked') else torch.float32)
        for name,value in state.items()), 'Original complete FP32 CPU state and normalization counters required')
    flat = [name for group in names for name in group]
    buffers = {'imagenet_mean','imagenet_std'} | {name for name in state
        if name.endswith(('.running_mean','.running_var','.num_batches_tracked'))}
    require(flat == [name for name in state if name not in buffers] and len(flat) == len(set(flat))
        and all(name.startswith(prefix) for group,prefix in zip(names,('encoder.','head.'),strict=True) for name in group),
        'Exact entire original parameter ordering required')


def publish_model_export(packet: Mapping[str,Any], destination: str | Path, *, byte_limit: int,
    expected_identity: Mapping[str,str],expected_checkpoint_sha256: str,expected_config: Mapping[str,Any],
    expected_provenance: Mapping[str,Any]) -> str:
    validate_model_export(packet,expected_identity=expected_identity,expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_config=expected_config,expected_provenance=expected_provenance)
    return _publish_validated_cpu_packet(packet,destination,byte_limit=byte_limit)


def load_model_export(path: str | Path, *, expected_sha256: str,expected_identity: Mapping[str,str],
    expected_checkpoint_sha256: str,expected_config: Mapping[str,Any],expected_provenance: Mapping[str,Any]) -> dict[str,Any]:
    require(_sha(expected_sha256), 'Exact model-export byte identity required')
    with Path(path).open('rb') as stream:
        require(hashlib.file_digest(stream,'sha256').hexdigest() == expected_sha256, 'Model export bytes changed')
        stream.seek(0)
        packet = torch.load(stream,map_location='cpu',weights_only=True)
    validate_model_export(packet,expected_identity=expected_identity,expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_config=expected_config,expected_provenance=expected_provenance)
    return packet


def restore_cpu_export(model: ResNet18FullBackboneStudyModel, packet: Mapping[str,Any], *, expected_identity: Mapping[str,str],
    expected_checkpoint_sha256: str,expected_config: Mapping[str,Any],expected_provenance: Mapping[str,Any]) -> None:
    validate_model_export(packet,expected_identity=expected_identity,expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_config=expected_config,expected_provenance=expected_provenance)
    require(type(model) is ResNet18FullBackboneStudyModel and model.config.to_dict() == expected_config['model']
        and all(p.device.type == 'cpu' for p in model.parameters()), 'Exact original CPU full model required')
    current,restored = model.state_dict(),packet['model_state_dict']
    names = [[f'encoder.{name}' for name,_ in model.encoder.named_parameters()],
        [f'head.{name}' for name,_ in model.head.named_parameters()]]
    require(names == packet['parameter_names'] and all(value.device.type == 'cpu' for value in current.values())
        and current.keys() == restored.keys() and getattr(current,'_metadata',None) == getattr(restored,'_metadata',None)
        and all(current[name].shape == restored[name].shape and current[name].dtype == restored[name].dtype for name in current),
        'Complete matching model architecture/buffers/metadata required before restore')
    model.load_state_dict(restored,strict=True)
    model.eval()


def replay_export_logits(model: ResNet18FullBackboneStudyModel, inputs: Mapping[str,torch.Tensor]) -> torch.Tensor:
    """Fixed eight training studies, original packing, batch-two native model path.

    The caller must independently decode and hash-check their original MRI. Input
    geometry alone cannot establish that original-source replay requirement.
    """
    require(type(model) is ResNet18FullBackboneStudyModel and all(p.device.type == 'cpu' for p in model.parameters()),
        'Original CPU full-backbone model required')
    h,w = model.config.image_size
    shapes = dict(images=(8,16,8,1,h,w),slice_mask=(8,16,8),series_mask=(8,16),series_descriptors=(8,16,10))
    require(set(inputs) == set(shapes), 'Complete fixed original packed study inputs required')
    for name,shape in shapes.items():
        value = inputs[name]
        require(type(value) is torch.Tensor and value.device.type == 'cpu' and tuple(value.shape) == shape
            and value.dtype == (torch.bool if name in ('slice_mask','series_mask') else torch.float32),
            'Exact original CPU input geometry/dtype required')
    require(bool(torch.isfinite(inputs['images']).all()) and bool(torch.isfinite(inputs['series_descriptors']).all()),
        'Finite original pixels/descriptors required')
    valid = inputs['slice_mask'] & inputs['series_mask'].unsqueeze(-1)
    require(bool(valid.flatten(1).any(1).all()), 'All eight studies require valid original MRI; no fallback')
    model.eval()
    with torch.inference_mode():
        result = torch.cat([model(*(inputs[name][start:start+2] for name in shapes)) for start in range(0,8,2)])
    require(result.shape == (8,12) and result.dtype == torch.float32 and bool(torch.isfinite(result).all()),
        'Complete finite fixed study logits required')
    return result
