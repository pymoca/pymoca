"""Parallel sweep of every example model in a registered library suite.

By default each model is run through the flatten pipeline (tree.flatten_class).
Pass -t/--translator casadi to instead translate each model with the CasADi
backend. Model names come from the suite's discovery manifest::

    python test/library_sweep.py msl -j 8 -f Blocks -t casadi
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from multiprocessing import Pool

try:
    import resource  # Unix only.
except ImportError:
    resource = None  # type: ignore[assignment]

from library_suite import LibrarySuite, load_suite, manifest_models, suite_names

from pymoca import parser
from pymoca import tree

# ---------------------------------------------------------------------------
# Worker state (set once by _worker_init, reused for all tasks)
# _worker_tree is None in forked mode: each task parses its own tree.
# ---------------------------------------------------------------------------

_worker_tree = None
_modelicapath: list[str] = []
_translator = None
_options = None
# CasADi API module, imported lazily by _worker_init only when translating.
_casadi_api = None


def _worker_init(
    path: list[str],
    reuse_tree: bool = False,
    translator: str | None = None,
    options: dict | None = None,
) -> None:
    global _worker_tree, _modelicapath, _translator, _options
    global _casadi_api
    _modelicapath = path
    _translator = translator
    _options = options or {}
    if translator == "casadi":
        # Importing CasADi (and the generator) is slow; only do it when needed.
        from pymoca.backends.casadi import api as _api

        _casadi_api = _api
    if reuse_tree:
        _worker_tree = parser.modelicapath_to_tree(path)


def _peak_rss_mb() -> float:
    """Process peak resident set size in MB.

    ru_maxrss is a process-lifetime high-water mark, so this is per-model only
    when the process is short-lived: the parallel --forked path recycles
    each worker after one model (maxtasksperchild=1). In serial mode or with
    the default tree reuse the process is long-lived and the peak spans many models.
    """
    if resource is not None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports ru_maxrss in kB; macOS/BSD reports bytes.
        return peak / 1024.0 if sys.platform == "linux" else peak / 1024.0 / 1024.0
    # Windows: resource is unavailable. psutil's memory_info().peak_wset is the
    # Windows-specific peak working set, unlike its cross-platform .rss field
    # (current, not peak).
    import psutil

    return psutil.Process().memory_info().peak_wset / 1024.0 / 1024.0


def _process_one(model_name: str) -> tuple:
    """Process one model; return (status, message, elapsed_s, peak_rss_mb).

    peak_rss_mb is the whole process's peak RSS. With --forked
    (maxtasksperchild=1) each worker handles one model, so the peak reflects the
    tree parse plus that model; heavy models read above the ~constant tree baseline.
    """
    verb = "translating" if _translator == "casadi" else "flattening"
    worker_tree = _worker_tree
    if worker_tree is None:
        worker_tree = parser.modelicapath_to_tree(_modelicapath)
    t0 = time.perf_counter()
    try:
        if _translator == "casadi":
            # generate_model() operates on an already-parsed tree so that
            # --reuse-tree is honored (transfer_model reparses a folder on
            # every call and cannot reuse a tree).
            assert _casadi_api is not None
            _casadi_api.generate_model(worker_tree, model_name, _options)
        else:
            tree.flatten_class(worker_tree, model_name)
        elapsed = time.perf_counter() - t0
        peak_rss = _peak_rss_mb()
        # Report the fully qualified model path (model_name) rather than the
        # flattened instance's short name (e.g. "PID_Controller").
        return ("success", f"Success {verb} {model_name}", elapsed, peak_rss)
    except parser.ModelicaSyntaxError as exc:
        elapsed = time.perf_counter() - t0
        peak_rss = _peak_rss_mb()
        import io

        buf = io.StringIO()
        parser.print_syntax_error(exc, buf)
        return (
            "error",
            f"Error parsing (syntax error) {model_name}:\n{buf.getvalue()}",
            elapsed,
            peak_rss,
        )
    except NotImplementedError as exc:
        elapsed = time.perf_counter() - t0
        peak_rss = _peak_rss_mb()
        return (
            "error",
            f"Error parsing (parser error) {model_name}: {exc}",
            elapsed,
            peak_rss,
        )
    except tree.ModelicaError as exc:
        elapsed = time.perf_counter() - t0
        peak_rss = _peak_rss_mb()
        return ("error", f"Error {verb} {model_name}: {exc}", elapsed, peak_rss)
    except Exception:
        elapsed = time.perf_counter() - t0
        peak_rss = _peak_rss_mb()
        tb = traceback.format_exc()
        return ("error", f"Error {verb} {model_name}:\n{tb}", elapsed, peak_rss)


def _default_jobs() -> int:
    # Parallel by default; peak memory scales with the worker count (one parsed
    # tree per worker). Leave a quarter of the CPUs free
    # for other work, matching the xdist policy in conftest. Pass -j 1 for a serial
    # in-process run when reproducing ordering-dependent bugs.
    return max(1, (os.cpu_count() or 1) * 3 // 4)


def sweep(
    suite: LibrarySuite,
    jobs: int = 1,
    filters: list | None = None,
    omits: list | None = None,
    reuse_tree: bool = True,
    translator: str | None = None,
    options: dict | None = None,
) -> int:
    """Run every selected model of a suite and return the number of failures."""
    assert suite.discovery is not None
    path = [str(suite.discovery.root)]
    model_names = manifest_models(suite.discovery)
    if filters:
        model_names = [n for n in model_names if any(f in n for f in filters)]
    if omits:
        model_names = [n for n in model_names if not any(o in n for o in omits)]

    num_success = 0
    num_error = 0
    wall_t0 = time.perf_counter()

    pool = None
    if jobs == 1:
        # Serial: initialize worker state in-process so ordering bugs remain reproducible.
        _worker_init(path, reuse_tree, translator, options)
        results = map(_process_one, model_names)
    else:
        # maxtasksperchild=1 recycles each worker after one model, returning its
        # memory to the OS - the cross-platform (spawn or fork) equivalent of the
        # pytest --forked path. Tree reuse (the default) opts out of recycling to
        # keep the parsed tree warm across models, trading memory for speed.
        pool = Pool(
            processes=jobs,
            initializer=_worker_init,
            initargs=[path, reuse_tree, translator, options],
            maxtasksperchild=None if reuse_tree else 1,
        )
        results = pool.imap_unordered(_process_one, model_names)

    total_models = len(model_names)
    done = 0
    try:
        for status, message, elapsed, peak_rss in results:
            done += 1
            suffix = f"  [{elapsed:.2f}s {peak_rss:.0f}MB]"
            print(message + suffix, flush=True)
            print(f"[{done}/{total_models}]", end="\r", file=sys.stderr, flush=True)
            if status == "success":
                num_success += 1
            else:
                num_error += 1
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    wall_elapsed = time.perf_counter() - wall_t0
    total = num_success + num_error
    pct = num_success / total * 100 if total else 0.0
    verb = "translating" if translator == "casadi" else "flattening"
    print("==================================================================================")
    print(f"Success {verb} {num_success} of {total} ({pct:.2f}%)  wall time: {wall_elapsed:.1f}s")
    return num_error


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("suite", choices=suite_names(sweepable=True), help="library suite to sweep")
    ap.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=_default_jobs(),
        metavar="N",
        help=f"worker processes (default: {_default_jobs()}; 1 = serial)",
    )
    ap.add_argument(
        "-f",
        "--filter",
        dest="filters",
        action="append",
        metavar="PATTERN",
        help="only run models whose name contains PATTERN (repeatable, OR logic)",
    )
    ap.add_argument(
        "-o",
        "--omit",
        dest="omits",
        action="append",
        metavar="PATTERN",
        help="skip models whose name contains PATTERN (repeatable, OR logic; applied after --filter)",
    )
    ap.add_argument(
        "--forked",
        action="store_true",
        default=False,
        help="recycle each worker after one model so all memory returns to the OS like pytest --forked",
    )
    ap.add_argument(
        "-t",
        "--translator",
        dest="translator",
        choices=("casadi",),
        default=None,
        help="translate each model with this backend instead of only flattening",
    )
    ap.add_argument(
        "-D",
        "--define",
        dest="define",
        action="append",
        metavar="NAME=VALUE",
        help="translator option in the form NAME=VALUE (repeatable; only with -t)",
    )
    args = ap.parse_args(argv)

    # Reuse the CLI's option parser so -D semantics match `pymoca -D ...`.
    from pymoca.compiler import build_define_options

    cli_options, opt_errors = build_define_options(args)
    if opt_errors:
        return 2
    if args.define and not args.translator:
        ap.error("-D/--define requires -t/--translator")

    num_error = sweep(
        load_suite(args.suite),
        jobs=args.jobs,
        filters=args.filters,
        omits=args.omits,
        reuse_tree=not args.forked,
        translator=args.translator,
        options=cli_options,
    )
    return 1 if num_error else 0


if __name__ == "__main__":
    sys.exit(main())
