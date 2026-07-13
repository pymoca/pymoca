"""Pytest configuration for ModelicaCompliance and MSL tests."""

import os
import sys

import pytest  # type: ignore[import-untyped]

# Exclude the MSL test file from normal collection: it parses the entire MSL
# library at import time to build parametrize params, which adds several seconds
# to every pytest run even when MSL tests are deselected.  Run them explicitly:
#   pytest test/msl_examples_test.py
collect_ignore = [os.path.join(os.path.dirname(__file__), "msl_examples_test.py")]

# Ratio of the total os.cpu_count() to use for testing
XDIST_CPU_RATIO = 3 / 4

# Add test directory to sys.path so test helper modules (e.g. conftest_parse) can be imported
sys.path.insert(0, os.path.dirname(__file__))


def pytest_configure(config):
    config.addinivalue_line("markers", "compliance: ModelicaCompliance test")
    config.addinivalue_line("markers", "flattening: Flattening level compliance test")
    config.addinivalue_line("markers", "msl: MSL examples pipeline test")
    # pytest-forked provides this marker; register it too so it isn't an unknown
    # mark (warning, or error under --strict-markers) when forked isn't installed.
    config.addinivalue_line("markers", "forked: run each test in a forked subprocess")

    # pyproject.toml's addopts deselects msl-marked tests by default (they're slow
    # and require the MSL submodule). msl_examples_test.py is excluded from normal
    # collection above, so it is only ever collected when named explicitly; in that
    # case the default "-m 'not msl'" is redundant and would silently deselect
    # every test it collects. Drop it so `pytest test/msl_examples_test.py` runs
    # the msl tests without also requiring `-m msl` on the command line.
    if config.option.markexpr == "not msl" and any(
        "msl_examples_test.py" in arg for arg in config.args
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
