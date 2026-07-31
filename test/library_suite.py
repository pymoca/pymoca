"""Reusable framework for library regression suites (e.g. RTC-Tools).

A plain helper module imported by name, like conftest_parse.py - not a
conftest. Compiles named cases through the CasADi backend, compares their
flattened structure against golden fingerprints, and compares exported
timeseries CSVs against reference data.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pymoca.backends.casadi.api import transfer_model


@dataclass
class LibraryCase:
    case_id: str
    model_folder: str
    model_name: str
    compiler_options: dict[str, Any] = field(default_factory=dict)
    library_folders: list[str] = field(default_factory=list)
    modelicapath: str = ""
    run_script: Optional[Path] = None
    reference_csv: Optional[Path] = None


@dataclass
class LibrarySuite:
    name: str
    root: Path
    expected_dir: Path
    cases: list[LibraryCase]
    available: bool


def _symbol_names(elements):
    names = []
    for elem in elements:
        name = None
        sym = getattr(elem, "symbol", None)
        if sym is not None:
            try:
                name = sym.name()
            except Exception:
                name = None
        if name is None:
            name = getattr(elem, "name", None)
            if callable(name):
                try:
                    name = name()
                except Exception:
                    name = None
        if name is None:
            name = str(elem)
        names.append(str(name))
    return sorted(names)


def flatten_fingerprint(model) -> dict:
    """Structural fingerprint of a flattened CasADi model.

    Keeps both the *_count keys and the sorted name lists; the name lists
    catch a dropped input/output prefix or extra parameters that a count
    alone would miss.
    """
    fp = {}
    for attr in (
        "states",
        "der_states",
        "alg_states",
        "inputs",
        "outputs",
        "constants",
        "parameters",
    ):
        elements = list(getattr(model, attr))
        fp[attr + "_count"] = len(elements)
        fp[attr] = _symbol_names(elements)
    for attr in ("equations", "initial_equations"):
        fp[attr + "_count"] = len(getattr(model, attr))
    return fp


def compile_case(case: LibraryCase):
    """Compile a LibraryCase through the CasADi backend with cache disabled."""
    options = dict(case.compiler_options)
    options["library_folders"] = list(case.library_folders)
    options["modelicapath"] = case.modelicapath
    options["cache"] = False

    return transfer_model(case.model_folder, case.model_name, options)


def assert_fingerprint_matches(actual: dict, expected_path: Path):
    """Compare a fingerprint against a golden JSON file.

    Reports per-group -missing +extra name-set diffs rather than a raw dict
    inequality, so a failure points straight at what changed.
    """
    expected = json.loads(expected_path.read_text())
    count_attrs = [
        "states_count",
        "der_states_count",
        "alg_states_count",
        "inputs_count",
        "outputs_count",
        "constants_count",
        "parameters_count",
        "equations_count",
        "initial_equations_count",
    ]
    name_attrs = [
        "states",
        "der_states",
        "alg_states",
        "inputs",
        "outputs",
        "constants",
        "parameters",
    ]
    mismatches = []
    for key in count_attrs:
        if actual.get(key) != expected.get(key):
            mismatches.append(f"{key}: expected {expected.get(key)}, got {actual.get(key)}")
    for attr in name_attrs:
        actual_set = set(actual.get(attr, []) or [])
        expected_set = set(expected.get(attr, []) or [])
        if actual_set != expected_set:
            missing = sorted(expected_set - actual_set)
            extra = sorted(actual_set - expected_set)
            mismatches.append(f"{attr}: -{missing} +{extra}")
    assert not mismatches, f"fingerprint mismatch vs {expected_path}:\n" + "\n".join(mismatches)


def read_timeseries_csv(path: Path):
    """Return (fieldnames, {column: [floats]}) for a timeseries_export.csv."""
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
            except (ValueError, TypeError):
                cols[name].append(math.nan)
    return header, cols


def compare_csv(actual_csv: Path, reference_csv: Path):
    """Return (max_abs, max_rel, note) comparing two timeseries CSVs."""
    ah, acols = read_timeseries_csv(actual_csv)
    rh, rcols = read_timeseries_csv(reference_csv)
    shared = [c for c in rh if c in acols]
    only_ref = [c for c in rh if c not in acols]
    only_actual = [c for c in ah if c not in rcols]
    max_abs = 0.0
    max_rel = 0.0
    worst_col = None
    for col in shared:
        r = rcols[col]
        a = acols[col]
        if len(r) != len(a):
            return None, None, f"row count differs in {col} ({len(r)} vs {len(a)})"
        for xr, xa in zip(r, a):
            if math.isnan(xr) and math.isnan(xa):
                continue
            if math.isnan(xr) or math.isnan(xa):
                return None, None, f"NaN mismatch in {col}"
            diff = abs(xr - xa)
            denom = max(abs(xr), abs(xa), 1e-12)
            rel = diff / denom
            if diff > max_abs:
                max_abs = diff
                worst_col = col
            max_rel = max(max_rel, rel)
    note_parts = []
    if only_ref:
        note_parts.append(f"cols only in reference: {', '.join(only_ref)}")
    if only_actual:
        note_parts.append(f"cols only in actual: {', '.join(only_actual)}")
    if worst_col:
        note_parts.append(f"worst: {worst_col}")
    return max_abs, max_rel, "; ".join(note_parts)


def assert_timeseries_close(actual_csv: Path, reference_csv: Path, abs_tol: float, rel_tol: float):
    max_abs, max_rel, note = compare_csv(actual_csv, reference_csv)
    assert max_abs is not None and max_rel is not None, note
    assert (
        max_abs <= abs_tol or max_rel <= rel_tol
    ), f"timeseries mismatch: max_abs={max_abs:.3e} max_rel={max_rel:.3e} ({note})"


# ---------------------------------------------------------------------------
# Golden regeneration CLI
# ---------------------------------------------------------------------------

# Suite name (as passed to --regenerate) -> module under test/ exposing a
# module-level SUITE: LibrarySuite.
_SUITE_MODULES = {"rtc_tools": "rtc_tools_test"}


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
    import importlib
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", required=True, help="suite name, e.g. rtc_tools")
    ap.add_argument("--only", default="", help="comma-separated example names to include")
    args = ap.parse_args(argv)

    module_name = _SUITE_MODULES.get(args.regenerate)
    if module_name is None:
        raise SystemExit(f"unknown suite {args.regenerate!r}; known: {sorted(_SUITE_MODULES)}")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    module = importlib.import_module(module_name)
    regenerate(module.SUITE.cases, module.SUITE.expected_dir, only=args.only)


if __name__ == "__main__":
    main()
