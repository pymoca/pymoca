"""Tests for the library suite framework's comparison helpers."""

import json
import math

from library_suite import (
    assert_fingerprint_matches,
    assert_objective_close,
    compare_csv,
    suite_names,
)

import pytest


def _write_csv(path, rows):
    path.write_text("time,Q\n" + "".join(f"{t},{q}\n" for t, q in rows))


def test_compare_csv_applies_tolerance_per_sample(tmp_path):
    """A sample agrees within either tolerance: a tiny offset from zero passes on
    abs_tol and a tiny error on a large value passes on rel_tol."""
    reference = tmp_path / "reference.csv"
    actual = tmp_path / "actual.csv"
    _write_csv(reference, [(0, 0.0), (1, 1000000.0)])
    _write_csv(actual, [(0, 5e-7), (1, 1000000.5)])
    assert compare_csv(actual, reference, abs_tol=1e-6, rel_tol=1e-6) is None
    _write_csv(actual, [(0, 5e-7), (1, 1000002.0)])
    mismatch = compare_csv(actual, reference, abs_tol=1e-6, rel_tol=1e-6)
    assert mismatch.startswith("1 of 4 samples outside tolerance")


def test_compare_csv_rejects_non_finite_mismatch(tmp_path):
    reference = tmp_path / "reference.csv"
    actual = tmp_path / "actual.csv"
    _write_csv(reference, [(0, 1.0), (1, math.inf)])
    _write_csv(actual, [(0, math.inf), (1, math.inf)])
    assert compare_csv(actual, reference, abs_tol=1e-6, rel_tol=1e-6) == "non-finite mismatch in Q"


def test_compare_csv_rejects_timestamp_mismatch(tmp_path):
    reference = tmp_path / "reference.csv"
    actual = tmp_path / "actual.csv"
    _write_csv(reference, [("2020-01-01 00:00:00", 1.0), ("2020-01-01 01:00:00", 2.0)])
    _write_csv(actual, [("2020-01-01 00:00:00", 1.0), ("2020-01-01 02:00:00", 2.0)])
    assert (
        compare_csv(actual, reference, abs_tol=1e-6, rel_tol=1e-6) == "non-numeric mismatch in time"
    )


def test_fingerprint_comparison_keeps_multiplicity(tmp_path):
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps({"outputs": ["x"], "equations_count": 1}))
    with pytest.raises(AssertionError, match=r"outputs: -\[\] \+\['x'\]"):
        assert_fingerprint_matches({"outputs": ["x", "x"], "equations_count": 1}, expected)


def test_objective_fallback_rejects_non_numeric_cell(tmp_path):
    reference = tmp_path / "reference.csv"
    actual = tmp_path / "actual.csv"
    _write_csv(reference, [(0, 1.0), (1, 2.0)])
    actual.write_text("time,Q\n0,1.0\n1,n/a\n")
    with pytest.raises(AssertionError, match="non-numeric cell"):
        assert_objective_close(actual, reference, "Q", rel_tol=1e-4, cause=AssertionError())


def test_sweep_cli_offers_only_sweepable_suites():
    assert "rtc_tools" not in suite_names(sweepable=True)
