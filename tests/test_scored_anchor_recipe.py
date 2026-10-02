"""Source relocation must never execute or rewrite scientific notebook logic."""

import json

import pytest

from rsna_knee.scored_anchor_recipe import (
    COMPETITION_ROOTS, DEVELOPMENT_ROOT, extract_scored_recipe,
    notebook_code, read_notebook_code, relocate_scored_recipe,
)


def sources():
    paths = f"paths = {COMPETITION_ROOTS!r}\n"
    return (paths + "resolution = 336\n", "raise RuntimeError('never execute')\n",
            "alpha = 0.4\n", paths + "k_eval = 62\n", "audit = True\n")


def test_literal_extraction_never_executes_the_notebook_or_embedded_code():
    code = "raise RuntimeError('never execute wrapper')\nfor i, source in enumerate(" + repr(sources()) + "):\n    exec(source)\n"
    assert extract_scored_recipe(code) == sources()


@pytest.mark.parametrize("code", ["for i, s in enumerate(make_sources()):\n    pass\n",
    "for i, s in enumerate(['a', 'b']):\n    pass\n", ""])
def test_missing_or_nonliteral_recipes_are_rejected(code):
    with pytest.raises(ValueError, match="missing or ambiguous"):
        extract_scored_recipe(code)


def test_ambiguous_recipe_is_rejected():
    loop = "for i, s in enumerate(" + repr(sources()) + "):\n    pass\n"
    with pytest.raises(ValueError, match="ambiguous"):
        extract_scored_recipe(loop * 2)


def test_relocation_preserves_every_byte_outside_four_literal_tokens():
    original = sources()
    actual = relocate_scored_recipe(original)
    assert actual[1:3] == original[1:3] and actual[4] == original[4]
    expected = original[0].replace(repr(COMPETITION_ROOTS[0]), repr(DEVELOPMENT_ROOT)).replace(repr(COMPETITION_ROOTS[1]), repr(DEVELOPMENT_ROOT))
    assert actual[0] == expected
    assert "resolution = 336" in actual[0] and "k_eval = 62" in actual[3]


def test_unexpected_root_layout_is_rejected():
    with pytest.raises(ValueError, match="layout differs"):
        relocate_scored_recipe(tuple(source.replace(COMPETITION_ROOTS[0], "/different") for source in sources()))


def test_notebook_source_serialization_is_normalized(tmp_path):
    notebook = {"cells": [{"cell_type": "markdown", "source": "ignored"},
                           {"cell_type": "code", "source": ["x = 1\n", "y = 2\n"]}]}
    path = tmp_path / "source.ipynb"
    path.write_text(json.dumps(notebook))
    assert read_notebook_code(path) == notebook_code(notebook) == "x = 1\ny = 2\n"
