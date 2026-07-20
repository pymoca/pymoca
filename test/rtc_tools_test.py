"""RTC-Tools example suite regression tests.

Stage 1 (``test_flatten_structure``) compiles every case through the CasADi
backend and checks the flattened model's structure against a golden
fingerprint. It needs only the ``rtc-tools`` and ``rtc-tools-channel-flow``
submodules, no RTC-Tools installation.

Stage 2 (``test_timeseries_export``) runs the real RTC-Tools example script
and compares its exported timeseries against the reference CSV checked into
the submodule. RTC-Tools pins ``pymoca == 0.9.*``, which conflicts with this
checkout, so it cannot be installed into the same venv as pymoca's own dev
dependencies. Build a separate venv instead::

    python -m venv /tmp/rtc-tools-venv
    /tmp/rtc-tools-venv/bin/pip install rtc-tools
    /tmp/rtc-tools-venv/bin/pip install -e . --no-deps

pip will report the ``pymoca`` version conflict; it is expected and harmless,
since ``--no-deps`` keeps rtc-tools' own pymoca pin from being reinstalled.
Run the stage-2 tests with that venv's interpreter. Under any other
interpreter ``rtctools`` fails to import and stage 2 is skipped.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

from library_suite import (
    LibraryCase,
    LibrarySuite,
    assert_fingerprint_matches,
    assert_timeseries_close,
    compile_case,
    flatten_fingerprint,
)

import pytest

MY_DIR = os.path.dirname(os.path.realpath(__file__))
RTC_TOOLS_DIR = os.path.join(MY_DIR, "libraries", "rtc-tools")
EXAMPLES_DIR = os.path.join(RTC_TOOLS_DIR, "examples")
CHANNEL_FLOW_MODELICA_DIR = os.path.join(
    MY_DIR, "libraries", "rtc-tools-channel-flow", "src", "rtctools_channel_flow", "modelica"
)
MSL4_DIR = os.path.join(MY_DIR, "libraries", "MSL-4.0.x")
EXPECTED_DIR = Path(MY_DIR) / "expected" / "rtc_tools"

RTC_TOOLS_AVAILABLE = os.path.isfile(os.path.join(EXAMPLES_DIR, "basic", "model", "Example.mo"))
RTC_TOOLS_INSTALLED = importlib.util.find_spec("rtctools") is not None

# Common to both rtctools.optimization.modelica_mixin.ModelicaMixin and
# rtctools.simulation.simulation_problem.SimulationProblem's compiler_options()
# methods, minus library_folders/cache which compile_case fills in per case.
COMPILER_OPTIONS = {
    "expand_vectors": True,
    "eliminate_constant_assignments": True,
    "replace_constant_values": True,
    "replace_constant_expressions": True,
    "replace_parameter_expressions": True,
    "eliminable_variable_expression": r"(.*[.]|^)_\w+(\[[\d,]+\])?\Z",
    "expand_mx": True,
    "detect_aliases": True,
}


def _case(
    case_id,
    example,
    model_name,
    *,
    model_example=None,
    script=None,
    reference=None,
    simulation=False,
):
    model_folder = os.path.join(EXAMPLES_DIR, model_example or example, "model")
    run_script = (
        Path(os.path.join(EXAMPLES_DIR, example, "src", script)) if script is not None else None
    )
    reference_csv = Path(os.path.join(EXAMPLES_DIR, *reference)) if reference is not None else None
    compiler_options = dict(COMPILER_OPTIONS)
    if not simulation:
        # Only ModelicaMixin (optimization) sets this; SimulationProblem leaves
        # it at pymoca's own default (True).
        compiler_options["allow_derivative_aliases"] = False
    return LibraryCase(
        case_id=case_id,
        model_folder=model_folder,
        model_name=model_name,
        compiler_options=compiler_options,
        library_folders=[CHANNEL_FLOW_MODELICA_DIR],
        modelicapath=MSL4_DIR,
        run_script=run_script,
        reference_csv=reference_csv,
    )


# 16 cases. RTC-Tools ships 18 example scripts, but two channel_wave_damping
# mixins have no run trigger and compile nothing. Of the 16, 11 have a
# reference timeseries_export.csv to compare stage 2 against; the other 5
# don't produce an independent, unambiguous export (see notes below).
CASES = [
    _case(
        "basic__example",
        "basic",
        "Example",
        script="example.py",
        reference=("basic", "reference_output", "timeseries_export.csv"),
    ),
    _case(
        "cascading_channels__example",
        "cascading_channels",
        "Example",
        script="example.py",
        reference=("cascading_channels", "reference_output", "timeseries_export.csv"),
    ),
    _case(
        # 4 subclasses (ExampleInertialWave, ...SemiImplicit, ExampleSaintVenant,
        # ...Upwind) share this model and folder; no single export is unambiguous.
        "channel_pulse__example",
        "channel_pulse",
        "Example",
        script="example.py",
    ),
    _case(
        # Not an independent data point for stage 2: example_optimization.py
        # runs this class too, and its own export is the one that lands.
        "channel_wave_damping__example_local_control",
        "channel_wave_damping",
        "ExampleLocalControl",
        script="example_local_control.py",
    ),
    _case(
        # example_optimization.py's module-level code runs ExampleOptimization
        # and then ExampleLocalControl, so anything that records only the last
        # transfer_model call describes the wrong class. This golden is
        # ExampleOptimization's own structure, checked against
        # ExampleOptimization.mo: Example's 2 inputs and 3 outputs plus its own
        # Q_dam_middle/Q_dam_upstream inputs, and no new outputs.
        "channel_wave_damping__example_optimization",
        "channel_wave_damping",
        "ExampleOptimization",
        script="example_optimization.py",
        reference=("channel_wave_damping", "reference_output", "timeseries_export.csv"),
    ),
    _case(
        # Ensemble output is split across forecast1/forecast2 subfolders, not a
        # single flat timeseries_export.csv.
        "ensemble__example",
        "ensemble",
        "Example",
        script="example.py",
    ),
    _case(
        # Uses the basic example's model via base_folder=BASIC_EXAMPLE_FOLDER; no
        # reference_output committed for fallback_option itself.
        "fallback_option__example",
        "fallback_option",
        "Example",
        model_example="basic",
        script="example.py",
    ),
    _case(
        "fallback_option__example_with_gp",
        "fallback_option",
        "Example",
        model_example="basic",
        script="example_with_gp.py",
    ),
    _case(
        "goal_programming__example",
        "goal_programming",
        "Example",
        script="example.py",
        reference=("goal_programming", "reference_output", "timeseries_export.csv"),
    ),
    _case(
        # No reference_output dir; the tracked baseline lives directly in
        # output/. The script runs ExampleOpt (optimization) and then
        # ExampleSim, so the simulation compiler options are the ones that
        # describe its export.
        "integrator_delay__example",
        "integrator_delay",
        "Example",
        script="example.py",
        reference=("integrator_delay", "output", "timeseries_export.csv"),
        simulation=True,
    ),
    _case(
        "lookup_table__example",
        "lookup_table",
        "Example",
        script="example.py",
        reference=("lookup_table", "reference_output", "timeseries_export.csv"),
    ),
    _case(
        "mixed_integer__example",
        "mixed_integer",
        "Example",
        script="example.py",
        reference=("mixed_integer", "reference_output", "timeseries_export.csv"),
    ),
    _case(
        "pumped_hydropower_system__example",
        "pumped_hydropower_system",
        "PumpedStoragePlant",
        script="example.py",
        reference=("pumped_hydropower_system", "reference_output", "timeseries_export.csv"),
    ),
    _case(
        "simulation_with_custom_equations__simple_model",
        "simulation_with_custom_equations",
        "SimpleModel",
        script="simple_model.py",
        reference=(
            "simulation_with_custom_equations",
            "reference_output",
            "timeseries_export.csv",
        ),
        simulation=True,
    ),
    _case(
        "simulation__example",
        "simulation",
        "Example",
        script="example.py",
        reference=("simulation", "reference_output", "timeseries_export.csv"),
        simulation=True,
    ),
    _case(
        "single_reservoir__single_reservoir",
        "single_reservoir",
        "SingleReservoir",
        script="single_reservoir.py",
        reference=("single_reservoir", "reference_output", "timeseries_export.csv"),
    ),
]

SUITE = LibrarySuite(
    name="rtc_tools",
    root=Path(RTC_TOOLS_DIR),
    expected_dir=EXPECTED_DIR,
    cases=CASES,
    available=RTC_TOOLS_AVAILABLE,
)

# Cheapest 2-case CI smoke set: touches no Deltares/MSL library code, plus the
# canonical basic example. Flatten stage only (see tox.ini's rtc-tools-smoke).
RTC_TOOLS_SMOKE_CASES = frozenset(
    {"simulation_with_custom_equations__simple_model", "basic__example"}
)

# Empty: every RTC-Tools example now flattens to its golden fingerprint.
FLATTEN_XFAIL: dict[str, str] = {}

# Provisional: seeded from an out-of-tree benchmark of these examples against
# pymoca 0.9.2, not yet from a stage-2 run in a rtc-tools venv. Recheck each
# reason the first time stage 2 runs for real.
NUMERIC_XFAIL = {
    "goal_programming__example": "solver reports INFEASIBLE under this pymoca version",
    "mixed_integer__example": "solver reports INFEASIBLE under this pymoca version",
    "channel_wave_damping__example_optimization": "upstream Deltares.ChannelFlow "
    "Distance-vs-Length defect makes the solver fail (TOO_FEW_DOF)",
    "cascading_channels__example": "benign extra min_abs_Q/min_divisor parameters "
    "compared with the 0.9.2 baseline",
}


def build_params(cases: list[LibraryCase], xfail: dict[str, str]):
    """Build pytest.param list with conditional xfail/smoke marks."""
    params = []
    for case in cases:
        marks = []
        if case.case_id in xfail:
            marks.append(pytest.mark.xfail(reason=xfail[case.case_id]))
        if case.case_id in RTC_TOOLS_SMOKE_CASES:
            marks.append(pytest.mark.rtc_tools_smoke)
        params.append(pytest.param(case, id=case.case_id, marks=marks))
    return params


pytestmark = [
    pytest.mark.library,
    pytest.mark.rtc_tools,
    pytest.mark.skipif(not RTC_TOOLS_AVAILABLE, reason="rtc-tools submodule not initialized"),
]


@pytest.mark.parametrize("case", build_params(CASES, FLATTEN_XFAIL))
def test_flatten_structure(case: LibraryCase):
    model = compile_case(case)
    fingerprint = flatten_fingerprint(model)
    assert_fingerprint_matches(fingerprint, EXPECTED_DIR / f"{case.case_id}.json")


def _run_example_script(case: LibraryCase, tmp_path: Path) -> Path:
    """Run case.run_script in an isolated tmp_path copy; return its output CSV."""
    assert case.run_script is not None
    example_dir = case.run_script.parents[1]
    dest_dir = tmp_path / example_dir.name
    shutil.copytree(example_dir, dest_dir)
    script_path = dest_dir / "src" / case.run_script.name

    env = dict(os.environ)
    if case.modelicapath:
        env["MODELICAPATH"] = case.modelicapath
    env["MPLBACKEND"] = "Agg"  # never pop up plot windows
    subprocess.run([sys.executable, str(script_path)], cwd=dest_dir, env=env, check=True)

    return dest_dir / "output" / "timeseries_export.csv"


@pytest.mark.skipif(not RTC_TOOLS_INSTALLED, reason="rtctools not importable in this venv")
@pytest.mark.parametrize("case", build_params(CASES, NUMERIC_XFAIL))
def test_timeseries_export(case: LibraryCase, tmp_path):
    if case.reference_csv is None:
        pytest.skip(f"{case.case_id} has no independent, unambiguous reference CSV")
    actual_csv = _run_example_script(case, tmp_path)
    assert_timeseries_close(actual_csv, case.reference_csv, abs_tol=1e-6, rel_tol=1e-6)
