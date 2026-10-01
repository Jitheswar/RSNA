"""Public child launchers must recover helper source from compiled cells."""

import ast
import inspect
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/prepare_public_frontier_gpu_preflight.py"


def generated_execution_loop():
    """Exercise the generator's actual execution loop without loading models."""
    tree = ast.parse(SCRIPT.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    # The generator stores the model-execution and validation block as a literal.
    block = next(
        node.value.value
        for node in main.body
        if isinstance(node, ast.AugAssign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and "for index,source in UPSTREAM_CELLS:" in node.value.value
    )
    return next(node for node in ast.parse(block).body[0].body if isinstance(node, ast.For))


@pytest.mark.parametrize("index", [3, 28])
def test_compiled_helpers_are_available_to_child_source_export(index):
    source = "def reader_input_helper(value):\n    return value + 1\n"
    namespace = {"UPSTREAM_CELLS": [(index, source)]}
    loop = generated_execution_loop()
    exec(compile(ast.Module(body=[loop], type_ignores=[]), "generator-loop-test", "exec"), namespace)
    # This is the same inspect call used to assemble the CoAt child workers.
    assert inspect.getsource(namespace["reader_input_helper"]) == source
    assert namespace["reader_input_helper"](4) == 5
