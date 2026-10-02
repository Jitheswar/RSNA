import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("metadata_recovery", Path(__file__).parents[1] / "scripts/prepare_ticket70_kernel_metadata_recovery.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def kernel_name(nb):
    # The remote Papermill trace fails at this precise metadata lookup before
    # executing any cell. Use the known successful notebook's kernel contract.
    name = nb["metadata"].get("kernelspec", {}).get("name")
    if not name:
        raise ValueError("No kernel name found in notebook and no override provided.")
    return name


def test_repair_restores_successful_kernel_without_changing_any_code():
    failed = {"metadata": {}, "cells": [{"cell_type": "code", "metadata": {}, "source": ["immutable()\n"]}]}
    with pytest.raises(ValueError, match="No kernel name"):
        kernel_name(failed)
    fixed = module.repair(failed, {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}})
    assert kernel_name(fixed) == "python3"
    assert fixed["cells"][0]["source"] == failed["cells"][0]["source"]
    assert failed["metadata"] == {}
    assert fixed["cells"][0]["id"] == "exact-anchor-runtime"


@pytest.mark.parametrize("spec", [{}, {"name": "other", "language": "python"}, {"name": "python3", "language": "other"}])
def test_unknown_runtime_kernel_is_not_substituted(spec):
    with pytest.raises(ValueError):
        module.repair({"metadata": {}, "cells": [{"cell_type": "code", "source": "fixed"}]}, {"kernelspec": spec})
