"""Inspect and relocate the scored notebook without executing its model code."""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path
import tokenize


COMPETITION_ROOTS = (
    "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    "/kaggle/input/rsna-knee-abnormality-detection",
)
DEVELOPMENT_ROOT = "/tmp/rsna_exact_scored_gold58"


def notebook_code(notebook: dict) -> str:
    """Normalize notebook string/list source serialization, retaining code bytes."""
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = cell["source"]
            if isinstance(source, list) and all(isinstance(part, str) for part in source):
                source = "".join(source)
            if not isinstance(source, str):
                raise ValueError("Notebook code source must be text")
            code.append(source)
    return "\n".join(code)


def read_notebook_code(path: Path) -> str:
    return notebook_code(json.loads(path.read_text()))


def extract_scored_recipe(code: str) -> tuple[str, ...]:
    """Require the unique literal five-cell recipe used by submission 56705536."""
    candidates = []
    for node in ast.walk(ast.parse(code)):
        if (isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "enumerate"
                and len(node.iter.args) == 1 and not node.iter.keywords):
            try:
                value = ast.literal_eval(node.iter.args[0])
            except (ValueError, TypeError):
                continue
            if (isinstance(value, (tuple, list)) and len(value) == 5
                    and all(isinstance(source, str) for source in value)):
                candidates.append(tuple(value))
    if len(candidates) != 1:
        raise ValueError("Scored five-cell recipe is missing or ambiguous")
    for source in candidates[0]:
        compile(source, "scored-anchor-source", "exec")
    return candidates[0]


class _RelocateRoots(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if isinstance(node.value, str) and node.value in COMPETITION_ROOTS:
            return ast.copy_location(ast.Constant(value=DEVELOPMENT_ROOT), node)
        return node


def relocate_scored_recipe(sources: tuple[str, ...]) -> tuple[str, ...]:
    """Change only the four root literals; retain every other source byte."""
    if len(sources) != 5:
        raise ValueError("Exactly five scored recipe cells required")
    relocated = []
    counts = []
    for source in sources:
        # Apply token spans in reverse so subsequent positions remain exact.
        lines = source.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        edits = []
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type != tokenize.STRING:
                continue
            try:
                value = ast.literal_eval(token.string)
            except (ValueError, SyntaxError):
                continue  # Interpolated strings are never path-literal edits.
            if isinstance(value, str) and value in COMPETITION_ROOTS:
                begin = offsets[token.start[0] - 1] + token.start[1]
                end = offsets[token.end[0] - 1] + token.end[1]
                edits.append((begin, end))
        transformed = source
        for begin, end in reversed(edits):
            transformed = transformed[:begin] + repr(DEVELOPMENT_ROOT) + transformed[end:]
        expected = _RelocateRoots().visit(ast.parse(source))
        if ast.dump(expected, include_attributes=False) != ast.dump(ast.parse(transformed), include_attributes=False):
            raise ValueError("Relocation changed scientific code")
        counts.append(len(edits))
        relocated.append(transformed)
    if counts != [2, 0, 0, 2, 0]:
        raise ValueError("Scored competition root layout differs")
    return tuple(relocated)
