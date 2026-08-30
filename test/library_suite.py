"""Reusable framework for library regression suites (e.g. RTC-Tools).

A plain helper module imported by name, like conftest_parse.py - not a
conftest. Compiles named cases through the CasADi backend, compares their
flattened structure against golden fingerprints, and compares exported
timeseries CSVs against reference data. Discovers a library's cases into a
checked-in manifest and reports when that manifest goes stale. Registers the
suites the regeneration and sweep CLIs load by name.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from pymoca import ast, parser

import pytest


@dataclass
class LibraryCase:
    case_id: str
    model_name: str
    # Empty for a suite that flattens from a shared tree instead of a folder.
    model_folder: str = ""
    compiler_options: dict[str, Any] = field(default_factory=dict)
    library_folders: list[str] = field(default_factory=list)
    modelicapath: str = ""


# Packages whose classes are candidate cases, and the class kinds a case can be.
_EXAMPLE_PACKAGES = ("Examples",)
_CASE_CLASS_TYPES = ("model", "block")


@dataclass
class LibraryDiscovery:
    """Rule for generating a suite's model list from a Modelica checkout."""

    library: str
    root: Path
    manifest_path: Path
    # Replaces the example-package walk, for a library whose cases do not come
    # from a package tree at all. The tree is never parsed if set.
    discover: Optional[Callable[[], list]] = None

    # Configuration the discovery reads, recorded in the manifest so editing it
    # invalidates the manifest like a submodule bump does.
    rule: dict[str, Any] = field(default_factory=dict)


@dataclass
class LibrarySuite:
    expected_dir: Path
    cases: list[LibraryCase]
    discovery: Optional[LibraryDiscovery] = None
    # False for a suite whose cases are not flattened from one shared tree, and
    # so cannot be run by library_sweep.
    sweepable: bool = True


def build_params(
    cases: list[LibraryCase],
    xfail: dict[str, str],
    smoke_cases: frozenset[str] = frozenset(),
    smoke_mark: Any = None,
) -> list:
    """Build a pytest.param list carrying each case's xfail and smoke marks."""
    params = []
    for case in cases:
        marks = []
        if case.case_id in xfail:
            marks.append(pytest.mark.xfail(reason=xfail[case.case_id]))
        if smoke_mark is not None and case.case_id in smoke_cases:
            marks.append(smoke_mark)
        params.append(pytest.param(case, id=case.case_id, marks=marks))
    return params


# Model attributes fingerprinted by sorted name, and by count only.
_NAME_ATTRS = (
    "states",
    "der_states",
    "alg_states",
    "inputs",
    "outputs",
    "constants",
    "parameters",
)
_COUNT_ONLY_ATTRS = ("equations", "initial_equations")


def flatten_fingerprint(model) -> dict:
    """Structural fingerprint of a flattened CasADi model."""
    fp = {}
    for attr in _NAME_ATTRS:
        # Variable stringifies to its symbol name; model.outputs holds bare names.
        fp[attr] = sorted(str(elem) for elem in getattr(model, attr))
    for attr in _COUNT_ONLY_ATTRS:
        fp[attr + "_count"] = len(getattr(model, attr))
    return fp


def compile_case(case: LibraryCase):
    """Compile a LibraryCase through the CasADi backend with cache disabled."""
    options = dict(case.compiler_options)
    options["library_folders"] = list(case.library_folders)
    options["modelicapath"] = case.modelicapath
    options["cache"] = False

    # Imported here so a flatten-only suite does not need the CasADi extra installed.
    from pymoca.backends.casadi.api import transfer_model

    return transfer_model(case.model_folder, case.model_name, options)


def assert_fingerprint_matches(actual: dict, expected_path: Path):
    """Compare a fingerprint against a golden JSON file."""
    expected = json.loads(expected_path.read_text())
    mismatches = []
    for attr in _COUNT_ONLY_ATTRS:
        key = attr + "_count"
        if actual.get(key) != expected.get(key):
            mismatches.append(f"{key}: expected {expected.get(key)}, got {actual.get(key)}")
    for attr in _NAME_ATTRS:
        actual_names = Counter(actual.get(attr, []))
        expected_names = Counter(expected.get(attr, []))
        if actual_names != expected_names:
            missing = sorted((expected_names - actual_names).elements())
            extra = sorted((actual_names - expected_names).elements())
            mismatches.append(f"{attr}: -{missing} +{extra}")
    assert not mismatches, f"fingerprint mismatch vs {expected_path}:\n" + "\n".join(mismatches)


def read_timeseries_csv(path: Path):
    """Return (fieldnames, {column: [values]}) for a timeseries_export.csv.

    Numeric cells become floats; any other cell (a timestamp) stays a string.
    """
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return [], {}
    header = rows[0]
    cols = {name: [] for name in header}
    for row in rows[1:]:
        for name, value in zip(header, row):
            try:
                cols[name].append(float(value))
            except ValueError:
                cols[name].append(value)
    return header, cols


def compare_csv(
    actual_csv: Path, reference_csv: Path, abs_tol: float, rel_tol: float, skip_rows: int = 0
) -> Optional[str]:
    """Describe how two timeseries CSVs differ beyond tolerance, or None when they agree.

    A sample agrees when it is within `abs_tol` or `rel_tol` (math.isclose).
    `skip_rows` drops that many leading data rows from both sides, for a
    reference whose opening rows are known to be stale.
    """
    ah, acols = read_timeseries_csv(actual_csv)
    rh, rcols = read_timeseries_csv(reference_csv)
    if skip_rows:
        acols = {name: values[skip_rows:] for name, values in acols.items()}
        rcols = {name: values[skip_rows:] for name, values in rcols.items()}
    shared = [c for c in rh if c in acols]
    only_ref = [c for c in rh if c not in acols]
    only_actual = [c for c in ah if c not in rcols]
    total = 0
    out_of_tol = 0
    max_abs = 0.0
    max_rel = 0.0
    worst_col = None
    for col in shared:
        r = rcols[col]
        a = acols[col]
        if len(r) != len(a):
            return f"row count differs in {col} ({len(r)} vs {len(a)})"
        for xr, xa in zip(r, a):
            total += 1
            if isinstance(xr, str) or isinstance(xa, str):
                if xr != xa:
                    return f"non-numeric mismatch in {col}"
                continue
            if xr == xa or (math.isnan(xr) and math.isnan(xa)):
                continue
            if not (math.isfinite(xr) and math.isfinite(xa)):
                return f"non-finite mismatch in {col}"
            if not math.isclose(xr, xa, rel_tol=rel_tol, abs_tol=abs_tol):
                out_of_tol += 1
            diff = abs(xr - xa)
            rel = diff / max(abs(xr), abs(xa), 1e-12)
            if diff > max_abs:
                max_abs = diff
                worst_col = col
            max_rel = max(max_rel, rel)
    if not out_of_tol:
        return None
    note_parts = [f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} worst: {worst_col}"]
    if only_ref:
        note_parts.append(f"cols only in reference: {', '.join(only_ref)}")
    if only_actual:
        note_parts.append(f"cols only in actual: {', '.join(only_actual)}")
    return f"{out_of_tol} of {total} samples outside tolerance ({'; '.join(note_parts)})"


def assert_timeseries_close(
    actual_csv: Path, reference_csv: Path, abs_tol: float, rel_tol: float, skip_rows: int = 0
):
    mismatch = compare_csv(actual_csv, reference_csv, abs_tol, rel_tol, skip_rows)
    assert mismatch is None, f"timeseries mismatch: {mismatch}"


def assert_objective_close(
    actual_csv: Path, reference_csv: Path, column: str, rel_tol: float, cause: Exception
):
    """Fallback for optimization cases with a degenerate optimum: compare the sum
    of `column`, proportional to the objective, instead of the trajectory.

    Raised from `cause` (the trajectory-comparison failure) so both mismatches
    are visible.
    """
    _, acols = read_timeseries_csv(actual_csv)
    _, rcols = read_timeseries_csv(reference_csv)
    assert all(
        isinstance(v, float) for v in acols[column] + rcols[column]
    ), f"objective column {column!r} has a non-numeric cell"
    actual_sum = math.fsum(acols[column])
    reference_sum = math.fsum(rcols[column])
    diff = abs(actual_sum - reference_sum)
    rel = diff / max(abs(actual_sum), abs(reference_sum), 1e-12)
    if not rel <= rel_tol:
        raise AssertionError(
            f"objective mismatch on {column!r}: actual_sum={actual_sum!r} "
            f"reference_sum={reference_sum!r} rel={rel:.3e} (tol={rel_tol:.3e})"
        ) from cause


# ---------------------------------------------------------------------------
# Discovery and manifests
# ---------------------------------------------------------------------------


def entry_name(entry) -> str:
    """Model name of a manifest entry, which is a bare name or a payload dict."""
    return entry if isinstance(entry, str) else entry["name"]


def walk_classes(root: ast.Class) -> Iterator[tuple[str, ast.Class]]:
    """Yield (qualified name, class) for every class under an example package.

    Membership latches: once a package named in `_EXAMPLE_PACKAGES` is entered,
    everything below it is a candidate. Packages are traversed but never
    yielded, since a package cannot be flattened.
    """

    def walk(cls: ast.Class, path: list[str], in_example: bool) -> Iterator[tuple[str, ast.Class]]:
        for name, child in cls.classes.items():
            child_path = path + [name]
            child_in_example = in_example or name in _EXAMPLE_PACKAGES
            if child_in_example and child.type != "package":
                # Class prefixes like `partial` read as their defaults until the class parses.
                _ = child.extends
                yield ".".join(child_path), child
            if child.type == "package":
                yield from walk(child, child_path, child_in_example)

    yield from walk(root, [], False)


def discover_models(discovery: LibraryDiscovery) -> list:
    """Run a discovery rule and return its sorted manifest entries."""
    if discovery.discover is not None:
        return discovery.discover()
    tree = parser.modelicapath_to_tree([str(discovery.root)])
    return sorted(
        name
        for name, cls in walk_classes(tree)
        # A partial class cannot be instantiated, so it can never be simulated.
        if cls.type in _CASE_CLASS_TYPES and not cls.partial
    )


def _git(root: Path, *args: str) -> Optional[str]:
    """Run git in `root`, returning None when git or the checkout is unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def library_sha(root: Path) -> Optional[str]:
    return _git(root, "rev-parse", "HEAD")


def library_describe(root: Path) -> Optional[str]:
    return _git(root, "describe", "--tags", "--always")


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def manifest_entries(discovery: LibraryDiscovery) -> list:
    """Entries from a discovery's manifest, empty when none is written yet.

    A missing manifest is not an error: the regeneration CLI imports a suite
    module to reach its discovery, so discovery has to work before a manifest
    exists. A suite's own staleness test is what reports the gap.
    """
    if not discovery.manifest_path.is_file():
        return []
    return read_manifest(discovery.manifest_path)["models"]


def manifest_models(discovery: LibraryDiscovery) -> list[str]:
    """Model names from a discovery's manifest."""
    return [entry_name(entry) for entry in manifest_entries(discovery)]


def write_manifest(discovery: LibraryDiscovery) -> Path:
    """Discover models and write the manifest, returning its path."""
    manifest = {
        "library": discovery.library,
        "sha": library_sha(discovery.root),
        "describe": library_describe(discovery.root),
        "rule": discovery.rule,
        "models": discover_models(discovery),
    }
    discovery.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    discovery.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return discovery.manifest_path


def manifest_staleness(discovery: LibraryDiscovery, suite_name: str) -> Optional[str]:
    """Describe why a manifest is out of date, or None when it is current."""
    manifest = read_manifest(discovery.manifest_path)
    rerun = f"rerun: python test/library_suite.py --regenerate {suite_name}"
    # Compared after a JSON round trip, so tuples read back as lists
    rule = json.loads(json.dumps(discovery.rule))
    if manifest.get("rule") != rule:
        return f"{discovery.library} discovery rule changed to {rule!r}; {rerun}"
    actual = library_sha(discovery.root)
    recorded = manifest.get("sha")
    if actual is None or recorded is None:
        return None
    if actual != recorded:
        return f"{discovery.library} moved {recorded[:7]} -> {actual[:7]}; {rerun}"
    return None


# ---------------------------------------------------------------------------
# Suite registry and golden regeneration CLI
# ---------------------------------------------------------------------------

# Suite name (as passed to the command-line tools) -> module under test/ exposing a
# module-level SUITE: LibrarySuite.
_SUITE_MODULES = {
    "compliance": "compliance_test",
    "msl": "msl_examples_test",
    "rtc_tools": "rtc_tools_test",
}


def suite_names(sweepable: bool = False) -> list[str]:
    """Registered suite names, optionally only those library_sweep can run."""
    names = sorted(_SUITE_MODULES)
    if sweepable:
        suites = {name: load_suite(name) for name in names}
        names = [n for n in names if suites[n].sweepable and suites[n].discovery is not None]
    return names


def load_suite(name: str) -> LibrarySuite:
    """Import a registered suite module and return its LibrarySuite."""
    import importlib
    import sys

    module_name = _SUITE_MODULES.get(name)
    if module_name is None:
        raise SystemExit(f"unknown suite {name!r}; known: {suite_names()}")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return importlib.import_module(module_name).SUITE


def _iter_selected(cases: list[LibraryCase], only: str):
    only_set = {s for s in only.split(",") if s}
    for case in cases:
        example_name = case.case_id.split("__", 1)[0]
        if only_set and example_name not in only_set and case.case_id not in only_set:
            continue
        yield case


def regenerate(cases: list[LibraryCase], expected_dir: Path, only: str = ""):
    expected_dir.mkdir(parents=True, exist_ok=True)
    for case in _iter_selected(cases, only):
        print(f"regenerating {case.case_id} ...")
        model = compile_case(case)
        fingerprint = flatten_fingerprint(model)
        out_path = expected_dir / f"{case.case_id}.json"
        out_path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True) + "\n")
        print(f"  wrote {out_path}")


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", required=True, help="suite name, e.g. rtc_tools")
    ap.add_argument("--only", default="", help="comma-separated example names to include")
    args = ap.parse_args(argv)

    suite = load_suite(args.regenerate)
    # A suite regenerates its discovery manifest, its golden fingerprints, or both.
    if suite.discovery is not None:
        print(f"discovering {suite.discovery.library} ...")
        path = write_manifest(suite.discovery)
        count = len(read_manifest(path)["models"])
        print(f"  wrote {path} ({count} models)")
    if suite.cases:
        regenerate(suite.cases, suite.expected_dir, only=args.only)


if __name__ == "__main__":
    main()
