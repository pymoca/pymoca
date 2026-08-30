"""Pytest configuration for ModelicaCompliance and MSL tests."""

import os
import sys

import pytest  # type: ignore[import-untyped]

# Exclude test/libraries: it holds submodule checkouts, not pymoca's own tests,
# and rtc-tools vendors its own pytest suite that needs rtctools installed to
# even collect.
collect_ignore = [
    os.path.join(os.path.dirname(__file__), "libraries"),
]

# Ratio of the total os.cpu_count() to use for testing
XDIST_CPU_RATIO = 3 / 4

# Add test directory to sys.path so test helper modules (e.g. conftest_parse) can be imported
sys.path.insert(0, os.path.dirname(__file__))


def pytest_configure(config):
    config.addinivalue_line("markers", "compliance: ModelicaCompliance test")
    config.addinivalue_line("markers", "flattening: Flattening level compliance test")
    config.addinivalue_line("markers", "library: slow library suite, deselected by default")
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
