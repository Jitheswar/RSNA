"""Retain all original worker logic except the actual JSON-to-Path boundary."""
import ast
from pathlib import Path

import pytest

PROJECT = Path(__file__).parents[1]
OLD = PROJECT/'scripts/ticket75_cpu_reference.py'
NEW = PROJECT/'scripts/ticket77_cpu_reference.py'


def cache_call(path):
    calls = [n for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id == 'load_pixel_entry']
    assert len(calls) == 1
    return calls[0]


def test_original_json_string_reproduces_actual_path_protocol_failure():
    def loader(path,**kwargs):
        return path.is_file()
    call = cache_call(OLD)
    context = dict(binding=dict(pixel_path='unused.npz',pixel_file_sha256='a'*64),uid='selected',
        pixel_plan=dict(pixel_identity_sha256='b'*64),source='c'*64,pixels_root=Path('/unused'),
        digest=lambda _: 'd'*64,load_pixel_entry=loader,Path=Path)
    with pytest.raises(AttributeError,match="'str' object has no attribute 'is_file'"):
        eval(compile(ast.Expression(call),str(OLD),'eval'),context)
    seen = []
    def accepting_loader(path,**kwargs):
        assert type(path) is type(Path('unused.npz'))
        seen.append((path,kwargs))
    context['load_pixel_entry'] = accepting_loader
    eval(compile(ast.Expression(cache_call(NEW)),str(NEW),'eval'),context)
    assert seen == [(Path('unused.npz'),dict(expected_sha256='a'*64,expected_study_uid='selected',
        expected_identity_sha256='b'*64,expected_source_identity_sha256='c'*64,expected_plan_sha256='d'*64))]


def test_corrected_worker_diff_is_only_path_conversion():
    old = OLD.read_text();new = NEW.read_text()
    assert new == old.replace("load_pixel_entry(binding['pixel_path'],", "load_pixel_entry(Path(binding['pixel_path']),")
    original_ast = ast.parse(old);corrected_ast = ast.parse(new)
    call = next(n for n in ast.walk(corrected_ast) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id == 'load_pixel_entry')
    assert isinstance(call.args[0],ast.Call) and call.args[0].func.id == 'Path'
    call.args[0] = call.args[0].args[0]
    assert ast.dump(original_ast,include_attributes=False) == ast.dump(corrected_ast,include_attributes=False)
