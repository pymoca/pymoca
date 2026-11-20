"""Native code for Modelica external functions (MLS 12.9)."""

import ctypes
import hashlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from pymoca import ast

from ._evaluate import EvaluationError, Evaluator

logger = logging.getLogger("pymoca")


class ExternalBuildError(RuntimeError):
    """The native library of external functions could not be built."""


class ExternalFunctionError(RuntimeError):
    """An external function reported an error through ModelicaError (MLS 12.9.6)."""


# Host side of ModelicaUtilities.h (MLS 12.9.6). An error longjmps back to the innermost
# PYMOCA_TRY of a generated wrapper, so C errors never unwind through Python frames.
_SHIM_SOURCE = r"""
#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pymoca_external.h"

static jmp_buf *pymoca_jmp = NULL;
static char pymoca_error[4096];
static pymoca_message_handler pymoca_handler = NULL;

typedef struct pymoca_string { struct pymoca_string *next; } pymoca_string;
static pymoca_string *pymoca_strings = NULL;

PYMOCA_EXPORT void pymoca_set_message_handler(pymoca_message_handler handler) {
    pymoca_handler = handler;
}

PYMOCA_EXPORT const char *pymoca_error_message(void) { return pymoca_error; }

PYMOCA_EXPORT void pymoca_free_strings(void) {
    while (pymoca_strings != NULL) {
        pymoca_string *next = pymoca_strings->next;
        free(pymoca_strings);
        pymoca_strings = next;
    }
}

jmp_buf *pymoca_jmp_swap(jmp_buf *env) {
    jmp_buf *previous = pymoca_jmp;
    pymoca_jmp = env;
    return previous;
}

static void pymoca_emit(int level, const char *format, va_list args) {
    char buffer[4096];
    vsnprintf(buffer, sizeof(buffer), format, args);
    if (pymoca_handler != NULL) {
        pymoca_handler(level, buffer);
    } else {
        fputs(buffer, stderr);
    }
}

void ModelicaVFormatMessage(const char *format, va_list args) { pymoca_emit(0, format, args); }
void ModelicaVFormatWarning(const char *format, va_list args) { pymoca_emit(1, format, args); }

void ModelicaVFormatError(const char *format, va_list args) {
    vsnprintf(pymoca_error, sizeof(pymoca_error), format, args);
    if (pymoca_jmp != NULL) {
        longjmp(*pymoca_jmp, 1);
    }
    fprintf(stderr, "ModelicaError outside of an external call: %s\n", pymoca_error);
    abort();
}

void ModelicaFormatMessage(const char *format, ...) {
    va_list args; va_start(args, format); ModelicaVFormatMessage(format, args); va_end(args);
}
void ModelicaFormatWarning(const char *format, ...) {
    va_list args; va_start(args, format); ModelicaVFormatWarning(format, args); va_end(args);
}
void ModelicaFormatError(const char *format, ...) {
    va_list args; va_start(args, format); ModelicaVFormatError(format, args); va_end(args);
}
void ModelicaMessage(const char *string) { ModelicaFormatMessage("%s", string); }
void ModelicaWarning(const char *string) { ModelicaFormatWarning("%s", string); }
void ModelicaError(const char *string) { ModelicaFormatError("%s", string); }

char *ModelicaAllocateStringWithErrorReturn(size_t len) {
    pymoca_string *block = malloc(sizeof(pymoca_string) + len + 1);
    if (block == NULL) {
        return NULL;
    }
    block->next = pymoca_strings;
    pymoca_strings = block;
    char *string = (char *)(block + 1);
    string[len] = '\0';
    return string;
}

char *ModelicaAllocateString(size_t len) {
    char *string = ModelicaAllocateStringWithErrorReturn(len);
    if (string == NULL) {
        ModelicaFormatError("Failed to allocate a string of length %lu", (unsigned long)len);
    }
    return string;
}

char *ModelicaDuplicateStringWithErrorReturn(const char *str) {
    char *string = ModelicaAllocateStringWithErrorReturn(strlen(str));
    return string == NULL ? NULL : strcpy(string, str);
}

char *ModelicaDuplicateString(const char *str) {
    return strcpy(ModelicaAllocateString(strlen(str)), str);
}
"""

_HEADER = r"""
#ifndef PYMOCA_EXTERNAL_H
#define PYMOCA_EXTERNAL_H
#include <setjmp.h>
#include <stddef.h>
#if defined(_WIN32)
#define PYMOCA_EXPORT __declspec(dllexport)
#else
#define PYMOCA_EXPORT __attribute__((visibility("default")))
#endif
typedef void (*pymoca_message_handler)(int level, const char *message);
jmp_buf *pymoca_jmp_swap(jmp_buf *env);
/* Return 1 from the enclosing wrapper when the external code calls ModelicaError */
#define PYMOCA_TRY \
    jmp_buf pymoca_env_; \
    jmp_buf *pymoca_prev_ = pymoca_jmp_swap(&pymoca_env_); \
    if (setjmp(pymoca_env_)) { pymoca_jmp_swap(pymoca_prev_); return 1; }
#define PYMOCA_END pymoca_jmp_swap(pymoca_prev_); return 0;
#endif
"""

# Fallback for external code that includes ModelicaUtilities.h without shipping one
_MODELICA_UTILITIES_H = r"""
#ifndef MODELICA_UTILITIES_H
#define MODELICA_UTILITIES_H
#include <stdarg.h>
#include <stddef.h>
void ModelicaMessage(const char *string);
void ModelicaFormatMessage(const char *format, ...);
void ModelicaVFormatMessage(const char *format, va_list args);
void ModelicaWarning(const char *string);
void ModelicaFormatWarning(const char *format, ...);
void ModelicaVFormatWarning(const char *format, va_list args);
void ModelicaError(const char *string);
void ModelicaFormatError(const char *format, ...);
void ModelicaVFormatError(const char *format, va_list args);
char *ModelicaAllocateString(size_t len);
char *ModelicaAllocateStringWithErrorReturn(size_t len);
char *ModelicaDuplicateString(const char *str);
char *ModelicaDuplicateStringWithErrorReturn(const char *str);
#endif
"""

# Libraries built from several C-Sources files, with their defines (MSL BuildProjects)
_SOURCE_GROUPS = {
    "ModelicaExternalC": (
        ["ModelicaFFT.c", "ModelicaInternal.c", "ModelicaRandom.c", "ModelicaStrings.c"],
        [],
    ),
    "ModelicaMatIO": (["ModelicaMatIO.c", "snprintf.c"], []),
    "ModelicaStandardTables": (
        ["ModelicaStandardTables.c", "ModelicaStandardTablesUsertab.c"],
        ["DUMMY_FUNCTION_USERTAB=1"],
    ),
    "zlib": (["zlib/*.c"], ["HAVE_ZLIB=1"]),
}

_LIBRARY_EXTENSIONS = (".so", ".dylib", ".a", ".lib")
_INCLUDED_SUFFIXES = {".h", ".hpp", ".c", ".inc"}


def _platform_dirs() -> List[str]:
    """Return the platform subdirectories of a library directory (MLS 12.9.4)."""
    bits = "64" if sys.maxsize > 2**32 else "32"
    system = platform.system().lower()
    name = {"windows": "win", "darwin": "darwin"}.get(system, system)
    machine = platform.machine().lower()
    return [f"{name}{bits}", f"{machine}-{system}", ""]


def find_compiler() -> str:
    """Return the C compiler from $CC or the first of cc, gcc and clang on PATH.

    Raises:
        ExternalBuildError: if no compiler is found
    """
    candidates = [os.environ["CC"]] if os.environ.get("CC") else ["cc", "gcc", "clang"]
    for candidate in candidates:
        path = shutil.which(candidate)
        if path is not None:
            return path
    raise ExternalBuildError(
        "External functions need a C compiler, but none of {} was found. Set $CC to"
        " the compiler to use.".format(", ".join(candidates))
    )


def _default_cache_dir() -> Path:
    from pymoca.parser import _get_default_cache_path

    return _get_default_cache_path() / "external"


class _BuildInputs:
    """Sources, flags and libraries that make up one external library."""

    def __init__(self, glue: str):
        self.glue = glue
        self.sources: List[Path] = []
        self.include_dirs: List[Path] = []
        self.defines: List[str] = []
        self.link: List[str] = []

    def add_source(self, path: Path) -> None:
        if path not in self.sources:
            self.sources.append(path)

    def add_unique(self, items: list, item) -> None:
        if item not in items:
            items.append(item)

    def digest(self, compiler: str) -> str:
        hasher = hashlib.sha256()
        parts = [self.glue, _SHIM_SOURCE, _HEADER, _MODELICA_UTILITIES_H, compiler]
        for part in parts + [platform.platform()]:
            hasher.update(part.encode("utf-8"))
        for path in self.sources:
            hasher.update(str(path).encode("utf-8"))
            hasher.update(path.read_bytes())
        for item in [*self.include_dirs, *self.defines, *self.link]:
            hasher.update(str(item).encode("utf-8"))
        # Files the sources and Include annotations may include, and prebuilt libraries
        included = [
            p
            for d in self.include_dirs
            if d.is_dir()
            for p in sorted(d.rglob("*"))
            if p.suffix in _INCLUDED_SUFFIXES
        ]
        for path in included + [Path(item) for item in self.link]:
            if path.is_file():
                stat = path.stat()
                hasher.update(f"{path}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8"))
        return hasher.hexdigest()[:24]


def _find_library(name: str, dirs: Iterable[Path]) -> Optional[Path]:
    """Return a prebuilt library *name* from a platform subdirectory of *dirs*."""
    for directory in dirs:
        for sub in _platform_dirs():
            for prefix in ("lib", ""):
                for ext in _LIBRARY_EXTENSIONS:
                    path = directory / sub / f"{prefix}{name}{ext}"
                    if path.is_file():
                        return path
    return None


def _find_sources(name: str, dirs: Iterable[Path]) -> tuple:
    """Return the C sources and defines of library *name* from *dirs*, if found."""
    patterns, defines = _SOURCE_GROUPS.get(name, ([f"{name}.c", f"{name}/*.c"], []))
    for directory in dirs:
        sources = [p for pattern in patterns for p in sorted(directory.glob(pattern))]
        if sources:
            return sources, defines
    return [], []


def _on_message(level: int, message: bytes) -> None:
    text = message.decode("utf-8", "replace").rstrip()
    if level == 0:
        logger.info(text)
    else:
        logger.warning(text)


# Never freed, as a library loaded again from the cache keeps the handler set by any load
_message_handler = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)(_on_message)


class ExternalLibrary:
    """A shared library of external C code, its wrappers and the ModelicaUtilities shim."""

    def __init__(
        self,
        glue: str,
        resources: Sequence[ast.ExternalResources] = (),
        external_library_paths: Sequence[str] = (),
        cache_dir: Optional[Path] = None,
    ):
        self.path = self._build(glue, resources, external_library_paths, cache_dir)
        self.cdll = ctypes.CDLL(str(self.path))
        self.cdll.pymoca_error_message.restype = ctypes.c_char_p
        self.cdll.pymoca_set_message_handler(_message_handler)

    def call(self, wrapper: str, args: "ctypes.Array[ctypes.c_void_p]") -> None:
        """Call the generated *wrapper* with an array of argument pointers.

        Raises:
            ExternalFunctionError: if the external code calls ModelicaError
        """
        if getattr(self.cdll, wrapper)(args) != 0:
            message = self.cdll.pymoca_error_message().decode("utf-8", "replace")
            raise ExternalFunctionError(message)

    def free_strings(self) -> None:
        """Free the strings allocated through ModelicaAllocateString since the last call."""
        self.cdll.pymoca_free_strings()

    @staticmethod
    def _collect(
        glue: str,
        resources: Sequence[ast.ExternalResources],
        external_library_paths: Sequence[str],
        prefer_sources: bool = False,
    ) -> _BuildInputs:
        inputs = _BuildInputs(glue)
        extra_dirs = []
        for item in map(Path, external_library_paths):
            if item.is_dir():
                extra_dirs.append(item)
            elif item.suffix == ".c":
                inputs.add_source(item)
            else:
                inputs.add_unique(inputs.link, str(item))
        for directory in extra_dirs:
            inputs.add_unique(inputs.include_dirs, directory)
        for res in resources:
            for directory in map(Path, res.include_dirs):
                inputs.add_unique(inputs.include_dirs, directory)

        library_dirs = extra_dirs + [Path(d) for r in resources for d in r.library_dirs]
        source_dirs = extra_dirs + [Path(d) for r in resources for d in r.include_dirs]
        names = [name for r in resources for name in r.libraries]
        for name in dict.fromkeys(names):
            prebuilt = _find_library(name, library_dirs)
            sources, defines = _find_sources(name, source_dirs)
            if prebuilt is not None and not (prefer_sources and sources):
                inputs.add_unique(inputs.link, str(prebuilt))
            elif sources:
                for source in sources:
                    inputs.add_source(source)
                for define in defines:
                    inputs.add_unique(inputs.defines, define)
            else:
                # A system library, such as m or lapack
                inputs.add_unique(inputs.link, f"-l{name}")
        return inputs

    @classmethod
    def _build(
        cls,
        glue: str,
        resources: Sequence[ast.ExternalResources],
        external_library_paths: Sequence[str],
        cache_dir: Optional[Path],
    ) -> Path:
        compiler = find_compiler()
        cache_dir = cache_dir if cache_dir is not None else _default_cache_dir()
        inputs = cls._collect(glue, resources, external_library_paths)
        try:
            return cls._compile(compiler, inputs, cache_dir)
        except ExternalBuildError:
            # A prebuilt static library may not be position independent
            from_sources = cls._collect(glue, resources, external_library_paths, True)
            if from_sources.link == inputs.link:
                raise
            logger.info("Linking prebuilt libraries failed, compiling their sources instead")
            return cls._compile(compiler, from_sources, cache_dir)

    @staticmethod
    def _compile(compiler: str, inputs: _BuildInputs, cache_dir: Path) -> Path:
        ext = ".dll" if os.name == "nt" else ".so"
        target = cache_dir / f"pymoca_external_{inputs.digest(compiler)}{ext}"
        if target.is_file():
            return target

        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
            build_dir = Path(tmp)
            (build_dir / "pymoca_external.h").write_text(_HEADER)
            (build_dir / "pymoca_shim.c").write_text(_SHIM_SOURCE)
            (build_dir / "pymoca_glue.c").write_text(inputs.glue)
            fallback_dir = build_dir / "fallback"
            fallback_dir.mkdir()
            (fallback_dir / "ModelicaUtilities.h").write_text(_MODELICA_UTILITIES_H)

            output = build_dir / target.name
            command = [compiler, "-shared", "-fPIC", "-O2", "-w", "-o", str(output)]
            command += [f"-I{build_dir}"] + [f"-I{d}" for d in inputs.include_dirs]
            command += [f"-I{fallback_dir}"] + [f"-D{d}" for d in inputs.defines]
            command += [str(build_dir / "pymoca_shim.c"), str(build_dir / "pymoca_glue.c")]
            command += [str(s) for s in inputs.sources] + inputs.link + ["-lm"]
            logger.info("Compiling external functions: %s", " ".join(command))
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise ExternalBuildError(
                    "Compiling external functions failed:\n{}\n{}".format(
                        " ".join(command), result.stderr
                    )
                )
            # Atomic, so concurrent builds of the same library never load a partial file
            os.replace(output, target)
        return target


class _CType:
    """C representation of a Modelica scalar type (MLS 12.9.1.1)."""

    def __init__(self, name: str, ctype, dtype=None):
        self.name = name
        self.ctype = ctype
        self.dtype = dtype


_REAL = _CType("double", ctypes.c_double, np.float64)
_INTEGER = _CType("int", ctypes.c_int, np.intc)
_STRING = _CType("const char *", ctypes.c_char_p)
_OBJECT = _CType("void *", ctypes.c_void_p)
_SIZE = _CType("size_t", ctypes.c_size_t)
_BUILTIN_TYPES = {"Real": _REAL, "Integer": _INTEGER, "Boolean": _INTEGER, "String": _STRING}


def _is_scalar_dims(dimensions) -> bool:
    return all(isinstance(d, ast.Primary) and d.value is None for dims in dimensions for d in dims)


class _Argument:
    """One argument of an external call, or its returned output.

    An argument without *symbol* is an input whose value is its *expression*.
    """

    def __init__(
        self,
        ctype: _CType,
        symbol: Optional[ast.Symbol] = None,
        expression=None,
        is_array: bool = False,
    ):
        self.ctype = ctype
        self.symbol = symbol
        self.expression = expression
        self.is_array = is_array or symbol is not None and not _is_scalar_dims(symbol.dimensions)
        self.is_input = symbol is None or "input" in symbol.prefixes

    def c_parameter(self) -> str:
        if self.is_input and not self.is_array:
            return self.ctype.name
        if self.is_input and self.ctype is not _STRING:
            return f"const {self.ctype.name} *"
        # An output string is set to a string that the callee allocated
        return f"{self.ctype.name} *"

    def c_expression(self, index: int) -> str:
        parameter = self.c_parameter()
        if self.is_input and not self.is_array:
            return f"*({parameter} *)a[{index}]"
        return f"({parameter})a[{index}]"


def _array_layout(annotation: Optional[ast.ClassModification]) -> str:
    """Return the arrayLayout annotation of an external clause (MLS 12.9.1.2)."""
    for arg in annotation.arguments if annotation is not None else []:
        mod = arg.value
        if not isinstance(mod, ast.ElementModification) or str(mod.component) != "arrayLayout":
            continue
        value = mod.modifications[0] if mod.modifications else None
        if isinstance(value, ast.Primary) and value.value in ("rowMajor", "columnMajor"):
            return value.value
        raise ValueError(f'arrayLayout must be "rowMajor" or "columnMajor", not {value}')
    return "rowMajor"


class ExternalFunction:
    """A flat external function, called numerically through a generated C wrapper (MLS 12.9)."""

    def __init__(self, tree: ast.Class, wrapper: str):
        self.tree = tree
        self.name = tree.name
        self.wrapper = wrapper
        self.external = tree.external
        self.library: Optional[ExternalLibrary] = None
        self.inputs = [s for s in tree.symbols.values() if "input" in s.prefixes]
        self.outputs = [s for s in tree.symbols.values() if "output" in s.prefixes]
        language = self.external.language
        # "C" or a specific C standard such as "C99"
        if not (language == "builtin" or language.startswith("C")):
            raise NotImplementedError(
                f"External function {self.name}: language {language!r} is not supported"
            )
        # Flattening turns the bindings of protected variables and constants into statements
        self._bindings = {}
        for statement in tree.statements:
            (target,) = statement.left
            self._bindings[target.name] = statement.right
        self.column_major = _array_layout(self.external.annotation) == "columnMajor"
        output = self.external.output
        self.returned = None if output is None else self._argument(output)
        self.arguments = [self._argument(arg) for arg in self.external.arguments]

    def ctype_of(self, symbol: ast.Symbol) -> _CType:
        type_name = str(symbol.type)
        if type_name in _BUILTIN_TYPES:
            return _BUILTIN_TYPES[type_name]
        if type_name in self.tree.external_objects:
            return _OBJECT
        # Enumerations are passed as their ordinal (MLS 12.9.1.1)
        return _INTEGER

    def is_external_object(self, symbol: ast.Symbol) -> bool:
        return self.ctype_of(symbol) is _OBJECT

    def _binding(self, symbol: ast.Symbol):
        """Return the binding expression of *symbol*, or None if it has none."""
        binding = self._bindings.get(symbol.name, symbol.value)
        return None if isinstance(binding, ast.Primary) and binding.value is None else binding

    def _argument(self, expression) -> _Argument:
        if isinstance(expression, ast.ComponentRef) and expression.name in self.tree.symbols:
            if any(i is not None for dims in expression.indices for i in dims):
                raise NotImplementedError(
                    f"External function {self.name}: subscripted argument {expression}"
                )
            symbol = self.tree.symbols[expression.name]
            ctype = self.ctype_of(symbol)
            if "constant" in symbol.prefixes or "parameter" in symbol.prefixes:
                # Passed as an input (MLS 12.9.1)
                return _Argument(ctype, None, expression, not _is_scalar_dims(symbol.dimensions))
            return _Argument(ctype, symbol)
        if isinstance(expression, ast.Expression) and str(expression.operator) == "size":
            return _Argument(_SIZE, None, expression)
        # A scalar constant, typed by its value
        value = self._evaluator({})(expression)
        if isinstance(value, str):
            return _Argument(_STRING, None, expression)
        return _Argument(_REAL if isinstance(value, float) else _INTEGER, None, expression)

    def glue(self, declare: bool) -> str:
        """Return the C wrapper, with a prototype of the external function if *declare*."""
        ret = self.returned.ctype.name if self.returned is not None else "void"
        name = self.external.external_name
        lines = []
        if declare:
            parameters = ", ".join(a.c_parameter() for a in self.arguments) or "void"
            lines.append(f"{ret} {name}({parameters});")
        call = "{}({})".format(
            name, ", ".join(a.c_expression(i + 1) for i, a in enumerate(self.arguments))
        )
        statement = f"*({ret} *)a[0] = {call};" if self.returned is not None else f"{call};"
        lines += [
            f"PYMOCA_EXPORT int {self.wrapper}(void **a) {{",
            "    PYMOCA_TRY",
            f"    {statement}",
            "    PYMOCA_END",
            "}",
        ]
        return "\n".join(lines)

    def _evaluator(self, values: dict) -> Evaluator:
        def lookup(name):
            if name in values:
                return values[name]
            symbol = self.tree.symbols.get(name)
            if symbol is None:
                raise EvaluationError(f"Unknown name {name} in external function {self.name}")
            binding = self._binding(symbol)
            if binding is None:
                raise EvaluationError(f"{name} has no value in external function {self.name}")
            values[name] = evaluator(binding)
            return values[name]

        evaluator = Evaluator(lookup)
        return evaluator

    def _shape(self, symbol: ast.Symbol, evaluate) -> tuple:
        shape = []
        for dims in symbol.dimensions:
            for dim in dims:
                if isinstance(dim, ast.Primary) and dim.value is None:
                    continue
                if isinstance(dim, ast.Slice):
                    raise NotImplementedError(
                        f"External function {self.name}: the size of {symbol.name} is unknown"
                    )
                shape.append(int(evaluate(dim)))
        return tuple(shape)

    def _storage(self, ctype: _CType, value, is_array: bool, copy: bool = False):
        """Return the ctypes storage of a value and a pointer to it.

        An array is a *copy* of the value if the external code may write it.
        """
        if ctype is _STRING:
            if is_array:
                strings = [None if s is None else s.encode("utf-8") for s in np.ravel(value)]
                storage = (ctypes.c_char_p * max(len(strings), 1))(*strings)
            else:
                storage = ctypes.c_char_p(None if value is None else value.encode("utf-8"))
            return storage, ctypes.addressof(storage)
        if is_array:
            order = "F" if self.column_major else "C"
            convert = np.array if copy else np.asarray
            array = convert(value, dtype=ctype.dtype, order=order)
            return array, array.ctypes.data
        storage = ctype.ctype(value)
        return storage, ctypes.addressof(storage)

    @staticmethod
    def _read(ctype: _CType, symbol: ast.Symbol, storage, is_array: bool):
        if ctype is _STRING:
            if is_array:
                return [(s or b"").decode("utf-8") for s in storage]
            return (storage.value or b"").decode("utf-8")
        if is_array:
            return np.ascontiguousarray(storage)
        value = storage.value
        type_name = str(symbol.type)
        if type_name == "Boolean":
            return bool(value)
        return value

    def __call__(self, *inputs) -> list:
        """Call the function with its inputs in declaration order and return its outputs.

        Arrays are numpy arrays, strings are str, and external objects are pointers. Missing
        trailing inputs take their default values.

        Raises:
            ExternalFunctionError: if the external code calls ModelicaError
        """
        assert self.library is not None, "bind() the function to its library first"
        if len(inputs) > len(self.inputs):
            raise TypeError(
                f"{self.name} takes {len(self.inputs)} inputs, {len(inputs)} were given"
            )
        values = {s.name: v for s, v in zip(self.inputs, inputs)}
        evaluate = self._evaluator(values)
        for symbol in self.inputs[len(inputs) :]:
            # Defaults are evaluated in the function, with the other inputs (MLS 12.4.1)
            try:
                evaluate.lookup(symbol.name)
            except EvaluationError as e:
                raise TypeError(f"{self.name}: no argument for input {symbol.name}") from e

        storage = {}
        for symbol in self.tree.symbols.values():
            if "constant" in symbol.prefixes or "parameter" in symbol.prefixes:
                continue
            ctype = self.ctype_of(symbol)
            is_array = not _is_scalar_dims(symbol.dimensions)
            if "input" in symbol.prefixes or self._binding(symbol) is not None:
                # Bindings of outputs and protected variables come before the call (MLS 12.9)
                value = evaluate.lookup(symbol.name)
            elif ctype is _STRING:
                value = np.full(self._shape(symbol, evaluate), None) if is_array else None
            else:
                shape = self._shape(symbol, evaluate)
                value = np.zeros(shape, dtype=ctype.dtype) if is_array else 0
            copy = "input" not in symbol.prefixes
            storage[symbol.name] = self._storage(ctype, value, is_array, copy)

        keep = []
        pointers = (ctypes.c_void_p * (len(self.arguments) + 1))()
        returned = self.returned
        if returned is not None:
            pointers[0] = storage[returned.symbol.name][1]
        for i, argument in enumerate(self.arguments, 1):
            if argument.symbol is not None:
                pointers[i] = storage[argument.symbol.name][1]
                continue
            value = evaluate(argument.expression)
            item, pointer = self._storage(argument.ctype, value, argument.is_array)
            keep.append(item)
            pointers[i] = pointer

        try:
            self.library.call(self.wrapper, pointers)
            return [
                self._read(
                    self.ctype_of(s),
                    s,
                    storage[s.name][0],
                    not _is_scalar_dims(s.dimensions),
                )
                for s in self.outputs
            ]
        finally:
            self.library.free_strings()


def load_external_functions(
    functions: Sequence[ast.Class],
    external_library_paths: Sequence[str] = (),
    cache_dir: Optional[Path] = None,
) -> dict:
    """Build one library for the flat external *functions* and return them by name."""
    loaded = {f.name: ExternalFunction(f, f"pymoca_ext_{i}") for i, f in enumerate(functions)}
    includes = [inc for f in functions for inc in f.external.resources.includes]
    glue = ['#include "pymoca_external.h"', *dict.fromkeys(includes)]
    for function in loaded.values():
        glue.append(function.glue(declare=not function.external.resources.includes))
    library = ExternalLibrary(
        "\n".join(glue) + "\n",
        [f.external.resources for f in functions],
        external_library_paths,
        cache_dir,
    )
    for function in loaded.values():
        function.library = library
    return loaded
