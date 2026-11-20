"""Tests for native external function support in the CasADi backend."""

import ctypes
import os
import shutil

import numpy as np

from pymoca import ast, parser, tree
from pymoca.backends.casadi.external import (
    ExternalBuildError,
    ExternalFunctionError,
    ExternalLibrary,
    load_external_functions,
)

import pytest

MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

needs_compiler = pytest.mark.skipif(
    os.name == "nt" or shutil.which(os.environ.get("CC") or "cc") is None,
    reason="needs a C compiler (Windows compilers are not detected yet)",
)

_HALVE_SOURCE = """
#include "ModelicaUtilities.h"
double halve(double x) {
    if (x < 0) {
        ModelicaFormatError("negative input %g", x);
    }
    ModelicaFormatMessage("halving %g", x);
    return x / 2;
}
"""

_HALVE_GLUE = """
#include "pymoca_external.h"
double halve(double x);
PYMOCA_EXPORT int call_halve(void **a) {
    PYMOCA_TRY
    *(double *)a[0] = halve(*(double *)a[1]);
    PYMOCA_END
}
"""


@needs_compiler
def test_external_library(tmp_path, caplog):
    """A Library annotation compiles <IncludeDirectory>/<name>.c with the ModelicaUtilities shim."""
    sources = tmp_path / "C-Sources"
    sources.mkdir()
    (sources / "halve.c").write_text(_HALVE_SOURCE)
    resources = ast.ExternalResources(libraries=["halve"], include_dirs=[str(sources)])
    cache = tmp_path / "cache"

    library = ExternalLibrary(_HALVE_GLUE, [resources], cache_dir=cache)
    out, x = ctypes.c_double(), ctypes.c_double(3.0)
    args = (ctypes.c_void_p * 2)(ctypes.addressof(out), ctypes.addressof(x))
    with caplog.at_level("INFO", logger="pymoca"):
        library.call("call_halve", args)
    assert out.value == 1.5
    assert "halving 3" in caplog.text

    # ModelicaError returns to the wrapper, which reports it, and the library stays usable
    x.value = -1.0
    with pytest.raises(ExternalFunctionError, match="negative input -1"):
        library.call("call_halve", args)
    x.value = 5.0
    library.call("call_halve", args)
    assert out.value == 2.5

    assert ExternalLibrary(_HALVE_GLUE, [resources], cache_dir=cache).path == library.path


def test_external_library_without_compiler(tmp_path, monkeypatch):
    monkeypatch.setenv("CC", "pymoca-no-such-compiler")
    with pytest.raises(ExternalBuildError, match="C compiler"):
        ExternalLibrary(_HALVE_GLUE, [], cache_dir=tmp_path)


def _flat_external_functions(file_name: str, model_name: str) -> list:
    flat_tree = tree.flatten(
        parser.parse_file(os.path.join(MODEL_DIR, file_name)),
        ast.ComponentRef.from_string(model_name),
    )
    return [c for c in flat_tree.classes.values() if c.external is not None]


@needs_compiler
def test_external_argument_types(tmp_path):
    """Modelica arguments map to C types, arrays row-major with size_t sizes (MLS 12.9.1)."""
    functions = load_external_functions(
        _flat_external_functions("ExternalCasadi.mo", "ExternalCasadi.Calls"), cache_dir=tmp_path
    )
    weighted_sum = functions["ExternalCasadi.weightedSum"]
    (y,) = weighted_sum(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), np.array([10.0, 1.0]))
    np.testing.assert_array_equal(y, [12.0, 34.0, 56.0])

    # Booleans and enumerations are ints, and several outputs are pointer arguments
    assert functions["ExternalCasadi.scaleShift"](1.5, 2, True, 3) == [-3.0, 5]
    assert functions["ExternalCasadi.describe"]("x", 1.5) == ["x=1.5"]
    # Protected variables are buffers set by their bindings, and constants pass by value
    assert functions["ExternalCasadi.sumWithWork"](np.array([1.0, 2.0])) == [14.0]

    checked_sqrt = functions["ExternalCasadi.checkedSqrt"]
    assert checked_sqrt(4.0) == [2.0]
    with pytest.raises(ExternalFunctionError, match="negative input -1"):
        checked_sqrt(-1.0)
