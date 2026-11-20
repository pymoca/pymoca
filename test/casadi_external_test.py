"""Tests for native external function support in the CasADi backend."""

import ctypes
import os
import shutil

from pymoca import ast
from pymoca.backends.casadi.external import (
    ExternalBuildError,
    ExternalFunctionError,
    ExternalLibrary,
)

import pytest

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
