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

from pymoca import ast

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
        for part in [self.glue, _SHIM_SOURCE, _HEADER, compiler, platform.platform()]:
            hasher.update(part.encode("utf-8"))
        for path in self.sources:
            hasher.update(str(path).encode("utf-8"))
            hasher.update(path.read_bytes())
        for item in [*self.include_dirs, *self.defines, *self.link]:
            hasher.update(str(item).encode("utf-8"))
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


class ExternalLibrary:
    """A shared library of external C code, its wrappers and the ModelicaUtilities shim."""

    _MESSAGE_HANDLER = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)

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
        self._handler = self._MESSAGE_HANDLER(self._on_message)
        self.cdll.pymoca_set_message_handler(self._handler)

    @staticmethod
    def _on_message(level: int, message: bytes) -> None:
        text = message.decode("utf-8", "replace").rstrip()
        if level == 0:
            logger.info(text)
        else:
            logger.warning(text)

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
