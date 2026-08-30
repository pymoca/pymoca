"""Pytest parametrized tests over the MSL-4.0.x Example models.

Each model is flattened through tree.flatten_class. For a standalone sweep,
including CasADi translation, see library_sweep.py.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

from library_suite import (
    LibraryCase,
    LibraryDiscovery,
    LibrarySuite,
    build_params,
    library_sha,
    manifest_models,
    manifest_staleness,
)

from pymoca import parser, tree

import pytest  # type: ignore[import-untyped]

MY_DIR = os.path.dirname(os.path.realpath(__file__))
MSL4_BASE_DIR = os.path.join(MY_DIR, "libraries", "MSL-4.0.x")
MSL4_AVAILABLE = os.path.isfile(os.path.join(MSL4_BASE_DIR, "Modelica", "package.mo"))
EXPECTED_DIR = Path(MY_DIR) / "expected" / "msl"

# Known-missing feature to error signature map to xfail
KNOWN_MISSING_FEATURES = {
    "ExternalObject": "Extends name ExternalObject not found",
    "stream connectors": "Unsupported connector variable prefixes ['stream']",
}

# Fast CI smoke subset (msl_smoke marker): one known-green model per Modelica
# sub-package, chosen from per-model timings, plus two canonical Blocks examples.
MSL_SMOKE_MODELS = frozenset(
    {
        "Modelica.Blocks.Examples.BusUsage",
        "Modelica.Blocks.Examples.PID_Controller",
        "Modelica.Clocked.Examples.Systems.Utilities.ComponentsMixingUnit."
        "MixingUnitWithContinuousControl",
        "Modelica.ComplexBlocks.Examples.TestConversionBlock",
        "Modelica.Electrical.Digital.Examples.DFFREGSRH",
        "Modelica.Fluid.Examples.ControlledTankSystem.Utilities.NormalOperation",
        "Modelica.Magnetic.FluxTubes.Examples.Utilities.TranslatoryArmatureAndStopper",
        "Modelica.Math.Random.Examples.GenerateRandomNumbers",
        "Modelica.Mechanics.Translational.Examples.Utilities.SpringDamperNoRelativeStates",
        "Modelica.Media.Examples.SolveOneNonlinearEquation.InverseIncompressible_sh_T",
        "Modelica.StateGraph.Examples.Utilities.Source",
        "Modelica.Thermal.HeatTransfer.Examples.Utilities.Conduction",
        "Modelica.Utilities.Examples.WriteRealMatrixToFile",
    }
)

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

# Model names come from a checked-in manifest rather than a walk of the parsed
# MSL, so importing this module costs a JSON read. Regenerate it whenever the
# MSL-4.0.x submodule pointer moves (test_msl_manifest_current fails until you
# do):  python test/library_suite.py --regenerate msl
DISCOVERY = LibraryDiscovery(
    library="MSL-4.0.x",
    root=Path(MSL4_BASE_DIR),
    manifest_path=EXPECTED_DIR / "manifest.json",
)

# MSL tree of lazy-parse stubs, shared by every test in the process (--forked
# children inherit it copy-on-write). Built on first use, not at import, so a
# run that collects but never flattens (the default fast suite) does not pay it.
_msl_tree = None


def _get_msl_tree():
    global _msl_tree
    if _msl_tree is None:
        _msl_tree = parser.modelicapath_to_tree([MSL4_BASE_DIR])
    return _msl_tree


# No cases: this suite regenerates a discovery manifest, not golden fingerprints.
SUITE = LibrarySuite(expected_dir=EXPECTED_DIR, cases=[], discovery=DISCOVERY)


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

# Tests run in-process by default; --forked (Unix) boxes each model in a subprocess
# but is opt-in, since fork() from a multi-threaded xdist worker is deadlock-prone.
pytestmark = [
    pytest.mark.skipif(not MSL4_AVAILABLE, reason="MSL-4.0.x submodule not initialized"),
]


# Yield the shared tree: flattening never mutates the parsed AST (guarded by the
# pickle checks in conftest_parse), so tests can reuse one tree.
@pytest.fixture(scope="function")
def msl_tree():
    yield _get_msl_tree()
    # Flattening builds large cyclic InstanceClass graphs that reference counting
    # alone can't reclaim. Force a collection between models so in-process runs
    # don't accumulate cyclic garbage.
    gc.collect()


def _msl_params() -> list:
    """All manifest cases, with MSL_SMOKE_MODELS carrying the msl_smoke mark."""
    cases = [LibraryCase(case_id=name, model_name=name) for name in manifest_models(DISCOVERY)]
    if not cases:
        return []
    missing = MSL_SMOKE_MODELS - {case.case_id for case in cases}
    assert not missing, f"MSL_SMOKE_MODELS not in the manifest: {sorted(missing)}"
    # No xfail dict: MSL xfails are matched on the exception, not the case name.
    return build_params(cases, {}, MSL_SMOKE_MODELS, pytest.mark.msl_smoke)


@pytest.mark.msl
def test_msl_manifest_current():
    """Fail with the regeneration command when the manifest no longer matches MSL."""
    if library_sha(DISCOVERY.root) is None:
        pytest.skip("MSL-4.0.x is not a git checkout")
    assert DISCOVERY.manifest_path.is_file(), (
        f"{DISCOVERY.manifest_path} is missing; "
        "rerun: python test/library_suite.py --regenerate msl"
    )
    staleness = manifest_staleness(DISCOVERY, "msl")
    assert staleness is None, staleness


@pytest.mark.library
@pytest.mark.msl
@pytest.mark.parametrize("case", _msl_params() if MSL4_AVAILABLE else [])
def test_msl_example(case: LibraryCase, msl_tree):
    try:
        flat_instance = tree.flatten_class(msl_tree, case.model_name)
    except Exception as exc:
        for feature, signature in KNOWN_MISSING_FEATURES.items():
            if signature in str(exc):
                pytest.xfail(f"{feature} not supported yet")
        raise
    assert flat_instance is not None
