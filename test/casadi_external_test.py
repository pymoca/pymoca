"""Tests for native external function support in the CasADi backend."""

import ctypes
import gc
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


@needs_compiler
def test_external_library_message_handler(tmp_path, caplog):
    """Messages reach the log after another load of the same cached library is freed."""
    sources = tmp_path / "C-Sources"
    sources.mkdir()
    (sources / "halve.c").write_text(_HALVE_SOURCE)
    resources = ast.ExternalResources(libraries=["halve"], include_dirs=[str(sources)])
    library = ExternalLibrary(_HALVE_GLUE, [resources], cache_dir=tmp_path / "cache")
    ExternalLibrary(_HALVE_GLUE, [resources], cache_dir=tmp_path / "cache")
    gc.collect()

    out, x = ctypes.c_double(), ctypes.c_double(3.0)
    args = (ctypes.c_void_p * 2)(ctypes.addressof(out), ctypes.addressof(x))
    with caplog.at_level("INFO", logger="pymoca"):
        library.call("call_halve", args)
    assert "halving 3" in caplog.text


@needs_compiler
def test_external_library_cache_includes(tmp_path):
    """A changed header in an include directory rebuilds the library."""
    sources = tmp_path / "C-Sources"
    sources.mkdir()
    (sources / "halve.c").write_text(
        '#include "factor.h"\ndouble halve(double x) { return x / FACTOR; }\n'
    )
    header = sources / "factor.h"
    resources = ast.ExternalResources(libraries=["halve"], include_dirs=[str(sources)])

    def halve(x: float) -> float:
        library = ExternalLibrary(_HALVE_GLUE, [resources], cache_dir=tmp_path / "cache")
        out, arg = ctypes.c_double(), ctypes.c_double(x)
        library.call(
            "call_halve", (ctypes.c_void_p * 2)(ctypes.addressof(out), ctypes.addressof(arg))
        )
        return out.value

    header.write_text("#define FACTOR 2\n")
    assert halve(3.0) == 1.5
    header.write_text("#define FACTOR 4\n")
    mtime = header.stat().st_mtime_ns + 10**9
    os.utime(header, ns=(mtime, mtime))
    assert halve(3.0) == 0.75


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


@needs_compiler
def test_external_bindings_and_constants(tmp_path):
    """Output bindings come before the call and constants are inputs (MLS 12.9, 12.9.1)."""
    functions = load_external_functions(
        _flat_external_functions("ExternalCasadi.mo", "ExternalCasadi.Calls"), cache_dir=tmp_path
    )
    y, w = functions["ExternalCasadi.addInPlace"](1.0)
    assert y == 3.5
    np.testing.assert_array_equal(w, [2.0, 3.0])

    # A default input is evaluated in the function, with the other inputs (MLS 12.4.1)
    add_two = functions["ExternalCasadi.addTwo"]
    assert add_two(1.0) == [3.0]
    assert add_two(1.0, 5.0) == [6.0]
    assert functions["ExternalCasadi.addNegative"](1.0) == [-0.5]
    assert "double addTwo(double, double);" in functions["ExternalCasadi.addNegative"].glue(True)

    # A constant array is passed as an array, and a constant scalar by value
    dot_table = functions["ExternalCasadi.dotTable"]
    assert "double dotTable(const double *, const double *, int);" in dot_table.glue(True)
    assert dot_table(np.array([1.0, 2.0, 3.0])) == [14.0]


@needs_compiler
def test_external_array_layout(tmp_path):
    """The arrayLayout annotation passes arrays in column-major order (MLS 12.9.1.2)."""
    functions = load_external_functions(
        _flat_external_functions("ExternalCasadi.mo", "ExternalCasadi.Calls"), cache_dir=tmp_path
    )
    (b,) = functions["ExternalCasadi.addIndices"](np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    np.testing.assert_array_equal(b, [[1, 3, 5], [14, 16, 18]])
