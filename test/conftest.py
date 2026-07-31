"""Pytest configuration for ModelicaCompliance and MSL tests."""

import os
import sys

import pytest  # type: ignore[import-untyped]

# Exclude the MSL test file from normal collection: it parses the entire MSL
# library at import time to build parametrize params, which adds several seconds
# to every pytest run even when MSL tests are deselected.  Run them explicitly:
#   pytest test/msl_examples_test.py
# Also exclude test/libraries: it holds submodule checkouts, not pymoca's own
# tests, and rtc-tools vendors its own pytest suite that needs rtctools
# installed to even collect.
collect_ignore = [
    os.path.join(os.path.dirname(__file__), "msl_examples_test.py"),
    os.path.join(os.path.dirname(__file__), "libraries"),
]

# Ratio of the total os.cpu_count() to use for testing
XDIST_CPU_RATIO = 3 / 4

# Add test directory to sys.path so test helper modules (e.g. conftest_parse) can be imported
sys.path.insert(0, os.path.dirname(__file__))

# Must match pyproject.toml's addopts.
DEFAULT_MARKEXPR = "not library"

# Test files that are only ever run when named explicitly on the command line
# (msl_examples_test.py is collect_ignore'd above; rtc_tools_test.py has a
# static case table so it doesn't need to be, but pyproject.toml's addopts
# still deselects its "library"-marked tests by default).
OPT_IN_TEST_FILES = ("msl_examples_test.py", "rtc_tools_test.py")


def pytest_configure(config):
    config.addinivalue_line("markers", "compliance: ModelicaCompliance test")
    config.addinivalue_line("markers", "flattening: Flattening level compliance test")
    config.addinivalue_line("markers", "library: any test in the library regression family")
    config.addinivalue_line("markers", "msl: MSL examples pipeline test")
    config.addinivalue_line(
        "markers", "msl_smoke: fast MSL example subset run in CI as a smoke check"
    )
    config.addinivalue_line("markers", "rtc_tools: RTC-Tools example suite regression test")
    config.addinivalue_line(
        "markers", "rtc_tools_smoke: fast RTC-Tools example subset run in CI as a smoke check"
    )
    # pytest-forked provides this marker; register it too so it isn't an unknown
    # mark (warning, or error under --strict-markers) when forked isn't installed.
    config.addinivalue_line("markers", "forked: run each test in a forked subprocess")

    # pyproject.toml's addopts deselects library-marked tests by default (they're
    # slow and require submodules). The opt-in files above are excluded from
    # normal collection (or cheap enough not to need to be), so they are only
    # ever collected when named explicitly; in that case the default markexpr is
    # redundant and would silently deselect every test they collect. Drop it so
    # `pytest test/rtc_tools_test.py` runs those tests without also requiring
    # `-m library` (or similar) on the command line.
    if config.option.markexpr == DEFAULT_MARKEXPR and any(
        any(name in arg for name in OPT_IN_TEST_FILES) for arg in config.args
    ):
        config.option.markexpr = ""


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config):
    """Leave some cpu power open for other work during tests using `-n auto`

    Declared optional so disabling xdist (e.g. ``-p no:xdist``) doesn't raise
    PluginValidationError for this otherwise-unknown hook.
    """
    if config.option.numprocesses != "auto":
        return None
    cpu_count = os.cpu_count()
    return int(cpu_count * XDIST_CPU_RATIO) if cpu_count else None
