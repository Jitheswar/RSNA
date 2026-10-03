"""Lossless CPU conversion and complete eight-rank checkpoint agreement."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

import torch

from .tpu_pilot_state import _plain_cpu_copy, decode_host_rng


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def convert_checkpoint_tree(value: Any, convert_tensors: Callable[[list[torch.Tensor]], Sequence[torch.Tensor]]) -> Any:
    """Flatten device tensors for collective conversion; retain ordered metadata."""
    tensors: list[torch.Tensor] = []
    index = 0
    def walk(node: Any, converted: Sequence[torch.Tensor] | None = None) -> Any:
        nonlocal index
        if isinstance(node,torch.Tensor):
            require(type(node) is torch.Tensor, 'Ordinary checkpoint tensors required')
            if converted is None:
                tensors.append(node);return None
            tensor = converted[index];index += 1
            require(type(tensor) is torch.Tensor and tensor.device.type == 'cpu'
                and tensor.shape == node.shape and tensor.dtype == node.dtype, 'Lossless complete CPU tensor conversion required')
            return tensor
        if type(node) in (dict,OrderedDict):
            result = type(node)((key,walk(child,converted)) for key,child in node.items())
            if hasattr(node,'_metadata'):
                metadata = walk(node._metadata,converted)
                if converted is not None: result._metadata = metadata
            return result
        if type(node) in (tuple,list):
            return type(node)(walk(child,converted) for child in node)
        require(node is None or type(node) in (bool,int,float,str), 'Plain checkpoint tree required before device conversion')
        return node
    walk(value)
    converted = convert_tensors(tensors)
    require(type(converted) in (tuple,list) and len(converted) == len(tensors), 'Every checkpoint tensor must convert exactly once')
    result = walk(value,converted)
    # Clone CPU values to avoid source/device aliases and validate finite values.
    return _plain_cpu_copy(result)


def checkpoint_tree_sha256(value: Any) -> str:
    """Content fingerprint includes types, order, tensor bytes and metadata."""
    digest = hashlib.sha256()
    def emit(data: bytes) -> None:
        digest.update(len(data).to_bytes(8,'big'));digest.update(data)
    def walk(node: Any) -> None:
        emit(type(node).__name__.encode())
        if isinstance(node,torch.Tensor):
            require(type(node) is torch.Tensor and node.device.type == 'cpu' and node.layout == torch.strided,
                'Ordinary strided CPU tensor fingerprint required')
            require(not node.is_floating_point() or bool(torch.isfinite(node).all()), 'Finite checkpoint fingerprint required')
            emit(str(node.dtype).encode());emit(json.dumps(list(node.shape)).encode())
            emit(node.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif type(node) in (dict,OrderedDict):
            emit(str(len(node)).encode())
            for key,child in node.items(): walk(key);walk(child)
            emit(b'metadata' if hasattr(node,'_metadata') else b'no-metadata')
            if hasattr(node,'_metadata'): walk(node._metadata)
        elif type(node) in (tuple,list):
            emit(str(len(node)).encode())
            for child in node: walk(child)
        else:
            require(node is None or type(node) in (bool,int,float,str), 'Plain checkpoint fingerprint required')
            emit(json.dumps(node,allow_nan=False).encode())
    walk(value)
    return digest.hexdigest()


def collect_rank_captures(rows: Sequence[Mapping[str,Any]], *, expected_context: Mapping[str,Any], expected_state_sha256: str) -> dict[str,Any]:
    require(type(rows) in (list,tuple) and len(rows) == 8, 'All eight checkpoint ranks required')
    require(all(isinstance(row,Mapping) and 'rank' in row for row in rows), 'Every rank record required')
    ranks = [row['rank'] for row in rows]
    require(all(type(rank) is int for rank in ranks) and set(ranks) == set(range(8)), 'Eight distinct exact checkpoint ranks required')
    require(type(expected_state_sha256) is str and len(expected_state_sha256) == 64
        and all(c in '0123456789abcdef' for c in expected_state_sha256), 'Exact complete-state SHA required')
    rngs = {}
    for row in rows:
        require(set(row) == {'rank','context','state_sha256','rng'} and row['context'] == dict(expected_context)
            and row['state_sha256'] == expected_state_sha256, 'All ranks must agree on complete checkpoint state/context')
        rng = row['rng']
        require(type(rng) is dict and set(rng) == {'host','xla_seed'} and type(rng['xla_seed']) is int, 'Complete rank host/device RNG required')
        decode_host_rng(rng['host'])
        rngs[row['rank']] = _plain_cpu_copy(rng)
    return dict(state_sha256=expected_state_sha256,rank_rng_states=rngs,replicas_checked=8)


def collect_master_publication(rows: Sequence[Mapping[str,Any]], *, expected_context: Mapping[str,Any]) -> str:
    require(type(rows) in (list,tuple) and len(rows) == 8, 'All eight publication acknowledgements required')
    require(all(isinstance(row,Mapping) and 'rank' in row for row in rows), 'Every rank record required')
    ranks = [row['rank'] for row in rows]
    require(all(type(rank) is int for rank in ranks) and set(ranks) == set(range(8)), 'Eight distinct publication ranks required')
    for row in rows:
        require(set(row) == {'rank','context','sha256'} and row['context'] == dict(expected_context), 'Exact publication context required')
        sha = row['sha256']
        require((row['rank'] != 0 and sha is None) or (row['rank'] == 0 and type(sha) is str
            and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha)), 'Only global rank0 may publish an exact checkpoint SHA')
    return next(row['sha256'] for row in rows if row['rank'] == 0)
