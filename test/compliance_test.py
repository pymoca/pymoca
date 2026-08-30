"""Automated ModelicaCompliance tests for flattening and value checking."""

import collections
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from library_suite import (
    LibraryCase,
    LibraryDiscovery,
    LibrarySuite,
    build_params,
    library_sha,
    manifest_entries,
    manifest_staleness,
)

import pymoca.ast as ast
import pymoca.parser
from pymoca.parser import ModelicaSyntaxError
from pymoca.tree import (
    InstanceSymbol,
    ModelicaError,
    flatten_instance,
    instantiate,
)

import pytest

# ---------------------------------------------------------------------------
# Discovery infrastructure
# ---------------------------------------------------------------------------

MY_DIR = os.path.dirname(os.path.realpath(__file__))
# The submodule checkout, which is what git commands run in; COMPLIANCE_DIR is
# the Modelica package inside it.
COMPLIANCE_ROOT = Path(MY_DIR) / "libraries" / "Modelica-Compliance"
COMPLIANCE_DIR = os.path.join(COMPLIANCE_ROOT, "ModelicaCompliance")
COMPLIANCE_AVAILABLE = os.path.isfile(os.path.join(COMPLIANCE_DIR, "Icons.mo"))
EXPECTED_DIR = Path(MY_DIR) / "expected" / "compliance"

_SHOULD_PASS_RE = re.compile(r"shouldPass\s*=\s*(true|false)", re.IGNORECASE)

# Matches: assert(expr == value, "msg") or assert(expr == value, "msg");
_ASSERT_DIRECT_RE = re.compile(
    r"assert\s*\(\s*"
    r"([\w.]+)\s*==\s*"  # lhs == rhs
    r"([\d.eE+-]+)"  # numeric literal
    r'\s*,\s*"[^"]*"\s*\)',
)

# Matches: assert(Util.compareReal(expr, value), "msg")
_ASSERT_COMPARE_RE = re.compile(
    r"assert\s*\(\s*Util\.compareReal\s*\(\s*"
    r"([\w.]+)\s*,\s*"  # variable
    r"([\d.eE+-]+)"  # numeric literal
    r'\s*\)\s*,\s*"[^"]*"\s*\)',
)

AssertionInfo = collections.namedtuple("AssertionInfo", ["variable", "expected", "approx"])


def parse_assertions(mo_file_path):
    """Extract assert(...) conditions from a .mo file.

    Returns list of AssertionInfo(variable, expected, approx) where approx=True
    for Util.compareReal checks.
    """
    with open(mo_file_path, "r", encoding="utf-8") as f:
        text = f.read()
    results = []
    for m in _ASSERT_DIRECT_RE.finditer(text):
        results.append(AssertionInfo(m.group(1), float(m.group(2)), False))
    for m in _ASSERT_COMPARE_RE.finditer(text):
        results.append(AssertionInfo(m.group(1), float(m.group(2)), True))
    return results


def parse_should_pass(mo_file_path):
    """Extract shouldPass value from __ModelicaAssociation(TestCase(...)) annotation."""
    with open(mo_file_path, "r", encoding="utf-8") as f:
        text = f.read()
    m = _SHOULD_PASS_RE.search(text)
    if m is None:
        raise ValueError(f"No shouldPass annotation found in {mo_file_path}")
    return m.group(1).lower() == "true"


def mo_path_to_model_name(mo_file_path):
    """Convert .mo file path to fully-qualified Modelica class name."""
    rel = os.path.relpath(mo_file_path, os.path.dirname(COMPLIANCE_DIR))
    return os.path.splitext(rel)[0].replace(os.sep, ".")


def load_compliance_model(mo_file_path):
    """Parse Icons.mo + target .mo file and return merged ast.Tree."""
    icon_ast = pymoca.parser.parse_file(os.path.join(COMPLIANCE_DIR, "Icons.mo"))
    target_ast = pymoca.parser.parse_file(mo_file_path)
    if None in (icon_ast, target_ast):
        return None
    icon_ast.extend(target_ast)
    return icon_ast


NAME_LOOKUP_CATEGORIES = [
    "Scoping/NameLookup/Simple",
    "Scoping/NameLookup/Composite",
    "Scoping/NameLookup/Global",
    "Scoping/NameLookup/Imports",
    "Scoping/MemberAccess",
    "Scoping/Visibility",
]

FLATTENING_CATEGORIES = [
    "Inheritance/Flattening",
    "Modification/Flattening",
    "Redeclare/Flattening",
]

ALL_CATEGORIES = NAME_LOOKUP_CATEGORIES + FLATTENING_CATEGORIES


def discover_compliance_files(subdirectory):
    """Walk one category under COMPLIANCE_DIR, returning its manifest entries.

    Each entry carries the model name and the shouldPass expectation read out
    of its annotation, so a test run needs neither the walk nor the sources.
    """
    base = os.path.join(COMPLIANCE_DIR, subdirectory)
    if not os.path.isdir(base):
        return []
    entries = []
    for dirpath, _dirs, files in os.walk(base):
        for fname in sorted(files):
            if fname in ("package.mo", "package.order") or not fname.endswith(".mo"):
                continue
            abs_path = os.path.join(dirpath, fname)
            entries.append(
                {
                    "name": mo_path_to_model_name(abs_path),
                    "should_pass": parse_should_pass(abs_path),
                }
            )
    return entries


@dataclass
class ComplianceCase(LibraryCase):
    """A case that is one .mo file with a shouldPass expectation.

    compile_case does not apply: a compliance model is parsed on its own and
    merged with Icons.mo, so there is no model folder.
    """

    mo_path: Optional[Path] = None
    should_pass: bool = True


def _make_case(entry) -> ComplianceCase:
    return ComplianceCase(
        case_id=entry["name"],
        model_name=entry["name"],
        mo_path=COMPLIANCE_ROOT / (entry["name"].replace(".", "/") + ".mo"),
        should_pass=entry["should_pass"],
    )


def _discover():
    """Discovery hook: the cases are files on disk, not classes in a package tree."""
    entries = [e for category in ALL_CATEGORIES for e in discover_compliance_files(category)]
    return sorted(entries, key=lambda entry: entry["name"])


# Cases come from a checked-in manifest rather than a walk of the checkout.
# Regenerate it whenever the Modelica-Compliance submodule pointer moves or
# ALL_CATEGORIES changes (test_compliance_manifest_current fails until you do):
#   python test/library_suite.py --regenerate compliance
DISCOVERY = LibraryDiscovery(
    library="Modelica-Compliance",
    root=COMPLIANCE_ROOT,
    manifest_path=EXPECTED_DIR / "manifest.json",
    discover=_discover,
    rule={"categories": ALL_CATEGORIES},
)

# No cases: this suite regenerates a discovery manifest, not golden fingerprints.
# Not sweepable: library_sweep flattens from one shared tree, which would drop
# the per-file Icons.mo merge and the shouldPass expectations.
SUITE = LibrarySuite(
    expected_dir=EXPECTED_DIR,
    cases=[],
    discovery=DISCOVERY,
    sweepable=False,
)


@pytest.mark.compliance
def test_compliance_manifest_current():
    """Fail with the regeneration command when the manifest no longer matches the library."""
    if library_sha(DISCOVERY.root) is None:
        pytest.skip("Modelica-Compliance is not a git checkout")
    assert DISCOVERY.manifest_path.is_file(), (
        f"{DISCOVERY.manifest_path} is missing; "
        "rerun: python test/library_suite.py --regenerate compliance"
    )
    staleness = manifest_staleness(DISCOVERY, "compliance")
    assert staleness is None, staleness


# Known failures: model_name -> xfail reason
#
# Global name lookup (MLS 5.3.3) is implemented; this shouldPass=false model needs
# the not-yet-implemented "class used during lookup may not be partial" rejection
_GLOBAL_MODELS = [
    "ModelicaCompliance.Scoping.NameLookup.Global." + n
    for n in [
        "GlobalPartialClass",
    ]
]
_GLOBAL_REASON = "Global name lookup: partial-class rejection not implemented"

KNOWN_FAILURES = {}

for _model in _GLOBAL_MODELS:
    KNOWN_FAILURES[_model] = _GLOBAL_REASON

# Parser-level failures — assert() in algorithm section not supported
_PARSER_FAILURES = {
    "ModelicaCompliance.Redeclare.Flattening.BasicBindingRedeclare": (
        "Parser error: assert() in algorithm section"
    ),
    "ModelicaCompliance.Redeclare.Flattening.InheritancePublicClass": (
        "Parser error: empty function_arguments"
    ),
    "ModelicaCompliance.Scoping.MemberAccess.AccessAlgorithm": (
        "Parser error: algorithm RHS member access (a.x) not handled in statement_component_reference"
    ),
    "ModelicaCompliance.Scoping.MemberAccess.AccessNestedAlgorithm": (
        "Parser error: algorithm RHS nested member access (a.b.c.x) not handled in statement_component_reference"
    ),
}
for _model, _reason in _PARSER_FAILURES.items():
    KNOWN_FAILURES[_model] = _reason

# shouldPass=false models that pymoca does not reject (missing error detection)
_SHOULD_FAIL_UNDETECTED = {
    "ModelicaCompliance.Scoping.NameLookup.Composite.FunctionInOperatorLookupViaComp": (
        "shouldPass=false: pymoca does not enforce function-only lookup via operator component"
    ),
    "ModelicaCompliance.Scoping.NameLookup.Composite.FunctionLookupViaArrayComp": (
        "shouldPass=false: pymoca does not reject non-function composite lookup via array component"
    ),
    "ModelicaCompliance.Scoping.NameLookup.Composite.FunctionLookupViaArrayElement": (
        "shouldPass=false: pymoca does not reject non-function composite lookup via array element"
    ),
    "ModelicaCompliance.Scoping.NameLookup.Composite.FunctionLookupViaNonClassComp": (
        "shouldPass=false: pymoca does not reject function lookup via non-class component"
    ),
    "ModelicaCompliance.Scoping.NameLookup.Composite.OperatorFunctionLookupViaComp": (
        "shouldPass=false: pymoca does not enforce operator-function-only lookup via component"
    ),
    "ModelicaCompliance.Scoping.NameLookup.Composite.PartialClassLookup": (
        "shouldPass=false: pymoca does not reject lookup inside partial class"
    ),
    "ModelicaCompliance.Scoping.NameLookup.Imports.RedeclareImport": (
        "shouldPass=false: pymoca does not reject redeclare in import"
    ),
    "ModelicaCompliance.Scoping.NameLookup.Imports.UnqualifiedImportConflict": (
        "shouldPass=false: pymoca does not reject name found in multiple unqualified imports"
    ),
    "ModelicaCompliance.Inheritance.Flattening.DuplicateInheritedNeqClasses": (
        "shouldPass=false: pymoca does not detect duplicate inherited unequal classes"
    ),
    "ModelicaCompliance.Inheritance.Flattening.DuplicateInheritedNeqComps": (
        "shouldPass=false: pymoca does not detect duplicate inherited unequal components"
    ),
    "ModelicaCompliance.Inheritance.Flattening.ProtectedInheritance": (
        "shouldPass=false: pymoca does not enforce protected inheritance restrictions"
    ),
    "ModelicaCompliance.Redeclare.Flattening.InheritanceDimensionClass": (
        "shouldPass=false: pymoca does not reject dimension mismatch in redeclared class inheritance"
    ),
    "ModelicaCompliance.Redeclare.Flattening.InheritanceProtectedClass": (
        "shouldPass=false: pymoca does not enforce protected class redeclare restrictions"
    ),
    "ModelicaCompliance.Redeclare.Flattening.InheritanceProtectedComp": (
        "shouldPass=false: pymoca does not enforce protected component redeclare restrictions"
    ),
    "ModelicaCompliance.Redeclare.Flattening.InheritanceVisibilityComp": (
        "shouldPass=false: pymoca does not enforce visibility restrictions in redeclared component"
    ),
    "ModelicaCompliance.Scoping.MemberAccess.AccessMissingAlgorithm": (
        "shouldPass=false: pymoca does not detect access to non-existing member in algorithm"
    ),
    "ModelicaCompliance.Scoping.MemberAccess.AccessMissingEquation": (
        "shouldPass=false: pymoca does not detect access to non-existing member in equation"
    ),
    "ModelicaCompliance.Scoping.Visibility.AccessInheritedProtectedClassInvalid": (
        "shouldPass=false: pymoca does not enforce protected class access restrictions"
    ),
    "ModelicaCompliance.Scoping.Visibility.AccessProtectedClass": (
        "shouldPass=false: pymoca does not enforce protected class access restrictions"
    ),
    "ModelicaCompliance.Scoping.Visibility.AccessProtectedClassClass": (
        "shouldPass=false: pymoca does not enforce protected nested class access restrictions"
    ),
    "ModelicaCompliance.Scoping.Visibility.AccessProtectedClassComp": (
        "shouldPass=false: pymoca does not enforce protected class access via component restrictions"
    ),
    "ModelicaCompliance.Scoping.Visibility.AccessInheritedProtectedCompInvalid": (
        "shouldPass=false: pymoca does not enforce protected inherited component dot-access restrictions"
    ),
    "ModelicaCompliance.Scoping.Visibility.AccessProtectedComp": (
        "shouldPass=false: pymoca does not enforce protected component access restrictions"
    ),
    "ModelicaCompliance.Scoping.Visibility.AccessProtectedCompClass": (
        "shouldPass=false: pymoca does not enforce protected component access via class restrictions"
    ),
    "ModelicaCompliance.Scoping.Visibility.AccessProtectedCompComp": (
        "shouldPass=false: pymoca does not enforce protected component access via component restrictions"
    ),
    "ModelicaCompliance.Scoping.Visibility.ModifyProtectedClass": (
        "shouldPass=false: pymoca does not enforce protected class modification restrictions"
    ),
    "ModelicaCompliance.Scoping.Visibility.ModifyProtectedComp": (
        "shouldPass=false: pymoca does not enforce protected component modification restrictions"
    ),
    "ModelicaCompliance.Scoping.Visibility.ProtectedMultiClass": (
        "shouldPass=false: pymoca does not enforce protected class multi-section restrictions"
    ),
    "ModelicaCompliance.Scoping.Visibility.ProtectedMultiComp": (
        "shouldPass=false: pymoca does not enforce protected component multi-section restrictions"
    ),
    "ModelicaCompliance.Scoping.Visibility.RedeclareProtectedClass": (
        "shouldPass=false: pymoca does not enforce protected class redeclaration restrictions"
    ),
    "ModelicaCompliance.Scoping.Visibility.RedeclareProtectedComp": (
        "shouldPass=false: pymoca does not enforce protected component redeclaration restrictions"
    ),
}
for _model, _reason in _SHOULD_FAIL_UNDETECTED.items():
    KNOWN_FAILURES[_model] = _reason

# Value check failures: assertion variable not resolvable to a literal
_VALUE_CHECK_WIP = {
    "ModelicaCompliance.Scoping.NameLookup.Simple.Encapsulation": (
        "Value check: bare assertion name 'x' inside nested class; "
        "flat symbol is 'a.x' (assertion-scope mismatch)"
    ),
    "ModelicaCompliance.Scoping.NameLookup.Simple.EnclosingClassLookupConstant": (
        "Value check: bare assertion name 'y' inside nested class; "
        "flat symbol is 'a.y' (assertion-scope mismatch)"
    ),
    "ModelicaCompliance.Scoping.NameLookup.Simple.ImplicitShadowingFor": (
        "Value check: for-equation expansion not implemented "
        "(y set by 'for x in {3} loop y = x;')"
    ),
    "ModelicaCompliance.Scoping.NameLookup.Simple.ImplicitShadowingReduction": (
        "Value check: reduction expression 'sum(x for x in 1:5)' not evaluated"
    ),
    "ModelicaCompliance.Scoping.NameLookup.Imports.ImportScopeType": (
        "Value check: imported-constant redirect chain (a=P2.y=P.x) "
        "and function call (b=f(4.0)) not evaluated"
    ),
}
for _model, _reason in _VALUE_CHECK_WIP.items():
    KNOWN_FAILURES[_model] = _reason

# Flattening or value check failures
_FLATTEN_WIP = {
    "ModelicaCompliance.Redeclare.Flattening.InheritanceStream": (
        "Flattening WIP: inStream not in transform_operator"
    ),
    "ModelicaCompliance.Inheritance.Flattening.ReplacedBaseClass": (
        "Value check WIP: extends through redeclared base class"
    ),
    "ModelicaCompliance.Modification.Flattening.Complicated": (
        "Value check WIP: nested modification scope resolution (d1.c.b.x)"
    ),
}
for _model, _reason in _FLATTEN_WIP.items():
    KNOWN_FAILURES[_model] = _reason


def _compliance_params() -> list:
    """All manifest cases, xfailed per KNOWN_FAILURES."""
    return build_params(
        [_make_case(entry) for entry in manifest_entries(DISCOVERY)], KNOWN_FAILURES
    )


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not COMPLIANCE_AVAILABLE, reason="Modelica-Compliance submodule not initialized"
)

# ---------------------------------------------------------------------------
# Value resolution helpers
# ---------------------------------------------------------------------------


def _extract_value_from_class_mod(sym):
    """Extract numeric value from a Symbol's class_modification (value=X pattern).

    For symbols like `constant Real x = 5.1`, the value 5.1 is stored in
    class_modification as ElementModification(component=value, modifications=[Primary(value=5.1)]).
    """
    cm = sym.class_modification
    if not cm:
        return None
    for arg in cm.arguments:
        em = arg.value
        if hasattr(em, "component") and str(em.component) == "value":
            if hasattr(em, "modifications") and em.modifications:
                m = em.modifications[0]
                if isinstance(m, ast.Primary) and isinstance(m.value, (int, float)):
                    return m.value
    return None


def _find_composite_binding(sym):
    """Check if a parent InstanceSymbol has a value binding to another component.

    For composite types like ``parameter C1 x5(a=5)`` with modification ``x5=x3``,
    the binding ``x3`` is stored as an ElementModification(component=value,
    modifications=['x3']) in the parent InstanceSymbol's modification_environment.

    Returns (ref_name, parent_name) if found, else None.
    """
    # Walk up: sym → InstanceClass (type) → InstanceSymbol (parent component)
    current = getattr(sym, "parent_instance", None)
    while current is not None:
        if isinstance(current, InstanceSymbol):
            for arg in current.modification_environment.arguments:
                em = arg.value
                if str(getattr(em, "component", "")) != "value":
                    continue
                modifications = getattr(em, "modifications", None)
                if modifications:
                    ref = modifications[0]
                    if isinstance(ref, (str, ast.ComponentRef)):
                        return str(ref), current.name
            break  # Only check the immediate parent symbol
        current = getattr(current, "parent_instance", None)
    return None


def _resolve_symbol_value(flat, var_name, _visited=None):
    """Try to resolve a symbol's value to a numeric literal.

    Follows InstanceSymbol reference chains through flat.symbols to find resolved
    numeric values. For package constants not in flat.symbols, falls back to
    ast_ref.class_modification.

    Returns (value, reason) where value is the numeric literal or None, and
    reason is an empty string on success or a short description of why
    resolution failed (for use in assertion messages and xfail documentation).
    """
    if _visited is None:
        _visited = set()
    if var_name in _visited:
        return None, "cycle detected"
    _visited.add(var_name)

    sym = flat.symbols.get(var_name)
    if sym is None:
        return None, "symbol not in flat.symbols"

    # Check for composite binding first: a parent component may bind to another
    # component (e.g., x5=x3), redirecting sub-component lookups.  This takes
    # precedence over the sub-component's own default value.
    binding = _find_composite_binding(sym)
    if binding is not None:
        ref_name, parent_name = binding
        parts = var_name.split(".")
        for i, part in enumerate(parts):
            if part == parent_name:
                parts[i] = ref_name
                redirected = ".".join(parts)
                result, _ = _resolve_symbol_value(flat, redirected, _visited)
                if result is not None:
                    return result, ""
                break

    v = sym.value
    if isinstance(v, (int, float)):
        return v, ""
    if not isinstance(v, InstanceSymbol):
        # Value may have been moved to an equation by _generate_value_equations.
        # Search flat.equations for  ComponentRef(var_name) = rhs.
        if hasattr(v, "value") and v.value is None:
            for eq in getattr(flat, "equations", []):
                if (
                    isinstance(eq, ast.Equation)
                    and isinstance(eq.left, ast.ComponentRef)
                    and eq.left.name == var_name
                ):
                    rhs = eq.right
                    if isinstance(rhs, ast.Primary) and isinstance(rhs.value, (int, float)):
                        return rhs.value, ""
                    if isinstance(rhs, ast.ComponentRef):
                        result, _ = _resolve_symbol_value(flat, rhs.name, set(_visited))
                        if result is not None:
                            return result, ""
            reason = "non-literal value: Primary(None)"
        elif hasattr(v, "operator"):
            reason = f"non-literal value: Expression({v.operator})"
        else:
            reason = f"non-literal value: {type(v).__name__}"
        return None, reason

    ref_name = v.name

    # Strategy 1: Walk up prefix hierarchy in flat.symbols
    # e.g., var_name="d1.c.b.x", ref_name="x" -> try "d1.c.b.x", "d1.c.x", "d1.x", "x"
    prefix = var_name.rsplit(".", 1)[0] if "." in var_name else ""
    while prefix:
        candidate = prefix + "." + ref_name
        if candidate != var_name:  # Avoid self-reference cycle
            result, _ = _resolve_symbol_value(flat, candidate, set(_visited))
            if result is not None:
                return result, ""
        prefix = prefix.rsplit(".", 1)[0] if "." in prefix else ""
    # Try without prefix (root level)
    if ref_name != var_name:
        result, _ = _resolve_symbol_value(flat, ref_name, set(_visited))
        if result is not None:
            return result, ""

    # Strategy 2: ast_ref.class_modification for package constants
    # e.g., ref_name="P.P.x" where P.x has `constant Real x = 5.1`
    result = _extract_value_from_class_mod(v.ast_ref)
    if result is not None:
        return result, ""

    return None, f"unresolved reference chain from {ref_name}"


# ---------------------------------------------------------------------------
# Flattening tests
# ---------------------------------------------------------------------------


@pytest.mark.compliance
@pytest.mark.flattening
@pytest.mark.parametrize("case", _compliance_params() if COMPLIANCE_AVAILABLE else [])
def test_flatten(case: ComplianceCase):
    """Test that flattening succeeds or fails as expected, with value checking."""
    mo_path = case.mo_path
    model_name = case.model_name
    if case.should_pass:
        ast_tree = load_compliance_model(mo_path)
        assert ast_tree is not None, f"Failed to parse {mo_path}"
        instance = instantiate(ast_tree, model_name)
        assert instance is not None
        flat = flatten_instance(instance)
        assert flat is not None

        # Value checking: verify assert() expectations from .mo file
        assertions = parse_assertions(mo_path)
        for assertion in assertions:
            value, reason = _resolve_symbol_value(flat, assertion.variable)
            assert value is not None, f"{assertion.variable}: unresolved ({reason})"
            if assertion.approx:
                assert (
                    abs(value - assertion.expected) < 1e-10
                ), f"{assertion.variable}: expected ~{assertion.expected}, got {value}"
            else:
                assert (
                    value == assertion.expected
                ), f"{assertion.variable}: expected {assertion.expected}, got {value}"
    else:
        # NotImplementedError is deliberately not accepted: it means the
        # construct was not attempted, not that pymoca correctly rejected it.
        with pytest.raises((ModelicaError, ModelicaSyntaxError)):
            ast_tree = load_compliance_model(mo_path)
            assert ast_tree is not None, f"Failed to parse {mo_path}"
            instance = instantiate(ast_tree, model_name)
            flatten_instance(instance)
