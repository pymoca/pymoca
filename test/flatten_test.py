#!/usr/bin/env python
"""
Tests for tree.flatten_instance() / tree.flatten().
"""

import os
import pickle

from conftest_parse import (
    MSL4_DIR,
    _flush,
    parse_and_flatten_model,
    parse_and_instantiate_model,
    parse_model_files,
)

from pymoca import ast
from pymoca import parser
from pymoca import tree

import pytest


def test_aircraft():
    parse_and_flatten_model("Aircraft.mo", "Aircraft")
    _flush()


def test_bouncing_ball():
    parse_and_flatten_model("BouncingBall.mo", "BouncingBall")
    _flush()


def test_estimator():
    parse_and_flatten_model("Estimator.mo", "Estimator")
    _flush()


def test_spring():
    parse_and_flatten_model("Spring.mo", "Spring")
    _flush()


def test_duplicate_state():
    parse_and_flatten_model("DuplicateState.mo", "DuplicateState")
    _flush()


def test_flatten_tree_contains_only_model_and_functions():
    ast_tree = parse_model_files("Spring.mo")
    flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name="Spring"))
    assert list(flat_tree.classes) == ["Spring"]


def test_function_pull():
    ast_tree = parse_model_files("FunctionPull.mo")

    class_name = "Level1.Level2.Level3.Function5"
    comp_ref = ast.ComponentRef.from_string(class_name)

    flat_tree = tree.flatten(ast_tree, comp_ref)

    # Flat output holds only the model and its pulled functions, no builtin stubs
    assert all(c.startswith("Level1.") for c in flat_tree.classes)

    # Check if all referenced functions are pulled in
    assert "Level1.Level2.Level3.f" in flat_tree.classes
    assert "Level1.Level2.Level3.TestPackage.times2" in flat_tree.classes
    assert "Level1.Level2.Level3.TestPackage.square" in flat_tree.classes
    assert "Level1.Level2.Level3.TestPackage.not_called" not in flat_tree.classes

    # Check if the classes in the flattened tree have the right type
    assert flat_tree.classes["Level1.Level2.Level3.Function5"].type == "model"

    assert flat_tree.classes["Level1.Level2.Level3.f"].type == "function"
    assert flat_tree.classes["Level1.Level2.Level3.TestPackage.times2"].type == "function"
    assert flat_tree.classes["Level1.Level2.Level3.TestPackage.square"].type == "function"

    # Check whether input/output information of functions comes along properly
    func_t2 = flat_tree.classes["Level1.Level2.Level3.TestPackage.times2"]
    assert "input" in func_t2.symbols["x"].prefixes
    assert "output" in func_t2.symbols["y"].prefixes

    # Check if built-in function call statement comes along properly
    func_f = flat_tree.classes["Level1.Level2.Level3.f"]
    assert func_f.statements[0].right.operator == "*"
    # Check if user-specified function call statement comes along properly
    assert (
        func_f.statements[0].right.operands[0].operator == "Level1.Level2.Level3.TestPackage.times2"
    )


def test_flattening_inheritance_tree():
    """Test flattening multi-level class hierarchy with modifications but no equations"""
    flat_tree = parse_and_flatten_model("InstantiationTree.mo", "TreeModel.Tree")

    # Non-value attribute modifications stay on the symbol
    expect_attrs = (
        ("w", "nominal", 2.0),
        ("b.t", "nominal", 1.0),
        ("t", "nominal", 2.0),
    )
    for name, attr, value in expect_attrs:
        assert name in flat_tree.symbols
        symbol = flat_tree.symbols[name]
        assert getattr(symbol, attr) == value

    # Value modifications on plain variables (Integer l.c, l2.c) become equations;
    # the symbol .value is cleared to sentinel.
    for name in ("l.c", "l2.c"):
        assert name in flat_tree.symbols
        sym = flat_tree.symbols[name]
        assert isinstance(sym.value, ast.Primary) and sym.value.value is None

    # Equations contain l.c = 1 and l2.c = 2
    eq_map = {
        eq.left.name: eq.right
        for eq in flat_tree.equations
        if isinstance(eq, ast.Equation) and isinstance(eq.left, ast.ComponentRef)
    }
    assert isinstance(eq_map["l.c"], ast.Primary) and eq_map["l.c"].value == 1
    assert isinstance(eq_map["l2.c"], ast.Primary) and eq_map["l2.c"].value == 2


def test_flattening_spring_system():
    """Test flattening of a simple class hierarchy with equations"""
    flat_tree = parse_and_flatten_model("SpringSystemExample.mo", "Example.SpringSystem")
    for name in ("spring.x", "spring.f", "damper.v", "damper.f", "damper.c"):
        assert name in flat_tree.symbols, f"Name not flattened: {name}"

    # Verify equations were collected and refs resolved
    assert len(flat_tree.equations) > 0, "No equations collected"


def test_equation_ref_resolution_bouncing_ball():
    """Verify equation ComponentRefs are resolved to flat names"""
    flat_tree = parse_and_flatten_model("BouncingBall.mo", "BouncingBall")

    # BouncingBall has: der(height) = velocity; der(velocity) = -g;
    assert len(flat_tree.equations) >= 2

    # Collect all ComponentRef names from equations
    ref_names = set()
    _collect_component_ref_names(flat_tree.equations, ref_names)

    # All refs should be flat (no children) and match known symbols or builtins
    for name in ref_names:
        assert "." not in name or name in flat_tree.symbols, f"Unresolved hierarchical ref: {name}"
    # Known symbols should appear
    assert "height" in ref_names
    assert "velocity" in ref_names

    # All refs should have no children (fully flattened)
    _assert_no_child_refs(flat_tree.equations)


def test_equation_ref_resolution_spring_system():
    """Verify cross-component equation refs are resolved with prefixes"""
    flat_tree = parse_and_flatten_model("SpringSystemExample.mo", "Example.SpringSystem")

    ref_names = set()
    _collect_component_ref_names(flat_tree.equations, ref_names)

    # SpringSystem equations reference nested symbols: spring.x, damper.v, etc.
    assert "spring.x" in ref_names
    assert "damper.v" in ref_names
    assert "spring.f" in ref_names
    assert "damper.f" in ref_names
    # Spring's own equation f = -k*x should be resolved to spring.f, spring.k, spring.x
    assert "spring.k" in ref_names

    # All resolved refs should exist in flat_class.symbols
    for name in ref_names:
        if name not in ("time",):  # builtins are OK to not be in symbols
            assert name in flat_tree.symbols, f"Ref {name!r} not in flat symbols"

    # All refs should have no children (fully flattened)
    _assert_no_child_refs(flat_tree.equations)


def test_equation_ref_resolution_nested_component():
    """Verify equations from nested components get correct prefix"""
    flat_tree = parse_and_flatten_model("SpringSystemExample.mo", "Example.SpringSystem")

    # Spring's equation is f = -k*x. After flattening under prefix "spring",
    # this should become spring.f = -spring.k * spring.x
    # Find the equation that has spring.k (the spring constant equation)
    spring_eq_found = False
    for eq in flat_tree.equations:
        ref_names = set()
        _collect_component_ref_names([eq], ref_names)
        if "spring.k" in ref_names:
            spring_eq_found = True
            assert "spring.f" in ref_names or "spring.x" in ref_names
    assert spring_eq_found, "Spring's internal equation not found"


def _collect_component_ref_names(nodes, names):
    """Recursively collect all ComponentRef names from a list of AST nodes"""
    for node in nodes:
        if isinstance(node, ast.ComponentRef):
            names.add(node.name)
        if isinstance(node, ast.Node):
            for attr in node.__dict__:
                val = getattr(node, attr)
                if isinstance(val, ast.Node):
                    _collect_component_ref_names([val], names)
                elif isinstance(val, list):
                    _collect_component_ref_names(val, names)
                elif isinstance(val, dict):
                    _collect_component_ref_names(list(val.values()), names)


def _assert_no_child_refs(nodes):
    """Assert all ComponentRefs in the tree have no children (fully flattened)"""
    for node in nodes:
        if isinstance(node, ast.ComponentRef):
            assert (
                len(node.child) == 0
            ), f"ComponentRef {node.name!r} still has children: {node.child}"
        if isinstance(node, ast.Node):
            for attr in node.__dict__:
                val = getattr(node, attr)
                if isinstance(val, ast.Node):
                    _assert_no_child_refs([val])
                elif isinstance(val, list):
                    _assert_no_child_refs(val)
                elif isinstance(val, dict):
                    _assert_no_child_refs(list(val.values()))


def test_flattening_modification_rhs_evaluation():
    """Test for correct scope and evaluation of right-hand-side values"""
    flat_tree = parse_and_flatten_model("ModificationScopeFlatten.mo", "B")
    # Check that the symbol value set by modifications have the right value and scope
    expect = (  # symbol_name, value_type, value, value_parent_name
        ("R", "literal", 3.0, None),  # Literal has no value parent
        ("a.R", "literal", 4.0, "B"),
        ("b.R", "symbol", "R", "B"),  # Reparented to flat class
        ("c.R", "symbol", "R", "B"),
        ("d.R", "symbol", "d.R", "B"),  # Self-ref in LoadError: d.R=d.R; reparented to flat class
        ("d.c", "literal", 84.0, "B"),
        ("e.R", "symbol", "R", "B"),
        ("f.R", "symbol", "R", "B"),
        ("g.R", "symbol", "R", "B"),
        ("h.R", "literal", 42.0, "B"),
        ("i.R", "symbol", "R", "B"),
        ("j.R", "literal", 2.0, "B"),
    )
    for symbol_name, value_type, value, value_parent_name in expect:
        symbol_value = flat_tree.symbols[symbol_name].value
        if value_type == "literal":
            assert symbol_value == value, "Wrong value for symbol: " + symbol_name
        elif value_type == "symbol":
            # Check that we have the correct reference for symbolic values
            assert isinstance(symbol_value, ast.Symbol)
            assert symbol_value.name == value, "Wrong value for symbol: " + symbol_name
            # Check that we have the correct scope for symbolic values
            assert symbol_value.parent_instance.name == value_parent_name, (
                "Wrong value scope for symbol: " + symbol_name
            )
        else:
            raise AssertionError(f"Unexpected value type in test: {value_type}")


def test_flattening_modification_rhs_omc_simulation():
    """Verify folded parameter values match OMC 1.22.0 simulation results."""
    flat = parse_and_flatten_model("ModificationScopeFlatten.mo", "B", evaluate_parameters=True)
    expected = {
        "R": 3.0,
        "a.R": 4.0,
        "b.R": 3.0,
        "c.R": 3.0,
        "d.c": 84.0,
        "e.R": 3.0,
        "f.R": 3.0,
        "g.R": 3.0,
        "h.R": 42.0,
        "i.R": 3.0,
        "j.R": 2.0,
    }
    for name, want in expected.items():
        v = flat.symbols[name].value
        assert isinstance(v, ast.Primary), f"{name} not folded: {v!r}"
        assert v.value == want, f"{name}: {v.value} != {want}"
    # d.R remains a symbolic self-reference even with evaluate_parameters=True
    d_R = flat.symbols["d.R"].value
    assert isinstance(d_R, ast.Symbol) and d_R.name == "d.R"


def test_evaluate_parameters_folds_expressions():
    """evaluate_parameters=True folds Expression values, not just direct references,
    regardless of forward/backward declaration order."""
    txt = """
    model M
      parameter Real q = 3*p;
      parameter Real p = 2;
    end M;
    """
    ast_tree = parser.parse(txt)
    flat = tree.flatten_class(ast_tree, "M", evaluate_parameters=True)
    q = flat.symbols["q"].value
    assert isinstance(q, ast.Primary) and q.value == 6


def test_extends_order():
    flat_tree = parse_and_flatten_model("ExtendsOrder.mo", "P.M")

    assert flat_tree.symbols["at.m"].value == 0.0


def test_constant_references():
    instance = parse_and_instantiate_model("ConstantReferences.mo", "b")

    # Since b extends a and a has no symbols, new instantiation stops there
    # Instead we will check for the redeclare being applied correctly
    b_m_mod = instance.extends[0].classes["m"].modification_environment.arguments[0]
    assert b_m_mod.value.name == "m"
    assert b_m_mod.value.component.name == "m2"

    flat_tree = tree.flatten_instance(instance)

    # Constant package references are now folded to their literal values
    assert flat_tree.symbols["y"].value == 2
    assert flat_tree.symbols["z"].value.name == "P0.p"  # Array constant — not folded (subscript)
    # TODO: Uncomment after equation references and constant references are implemented
    # self.assertIn("M2.m.f", flat_tree.symbols)
    # self.assertEqual(flat_tree.symbols["M2.m.f"].value.name, "m.f")
    # self.assertEqual(flat_tree.symbols["M2.m.f"].value.parent.name, "M1")
    # self.assertEqual(flat_tree.symbols["M2.m.f"].value.value, 3.0)


def test_parameter_modification_scope():
    instance = parse_and_instantiate_model("ParameterScope.mo", "ScopeTest")

    p_sym = instance.symbols["p"].type.symbols["Real"]
    p_sym_mod = p_sym.modification_environment.arguments[0]
    assert p_sym_mod.value.component.name == "value"
    assert p_sym_mod.value.modifications[0].value == 1.0
    nc_p_sym = instance.symbols["nc"].type.symbols["p"].type.symbols["Real"]
    nc_p_mod = nc_p_sym.modification_environment.arguments[0]
    assert nc_p_mod.value.modifications[0].name == "p"

    flat_tree = tree.flatten_instance(instance)

    assert flat_tree.symbols["nc.p"].value.name == "p"


def test_extends_modification():
    instance = parse_and_instantiate_model("ExtendsModification.mo", "MainModel")

    e_HQ_H = instance.symbols["e"].type.extends[0].extends[0].symbols["HQ"].type.symbols["H"]
    H_mod = e_HQ_H.type.symbols["Real"].modification_environment.arguments[0].value
    assert H_mod.component.name == "min"
    min_mod = H_mod.modifications[0]
    assert min_mod.name == "H_b"

    flat_tree = tree.flatten_instance(instance)

    assert flat_tree.symbols["e.HQ.H"].min.name == "e.H_b"


def test_extends_modification_expression():
    """Cross-scope modification whose value is an Expression, not a bare ComponentRef.

    LinearExpr has `extends PartialStorage(HQ.H(min = 2 * H_b))`.  At flatten
    time the modification scope is PartialStorage (base class) but the flat-
    naming root is MainModelExpr.  The expression `2 * H_b` refers to H_b from
    LinearExpr; the flattened result on e.HQ.H.min should either be the
    evaluated scalar (2 * -2.0 = -4.0) or an ast.Expression whose inner
    ComponentRef names "e.H_b" relative to MainModelExpr.
    """
    flat_tree = parse_and_flatten_model("ExtendsModification.mo", "MainModelExpr")

    min_val = flat_tree.symbols["e.HQ.H"].min
    if isinstance(min_val, (int, float)):
        # Constant-folding path: -2.0 * 2 = -4.0
        assert min_val == pytest.approx(-4.0)
    else:
        # Symbolic path: the ComponentRef inside the Expression must be flat-named
        assert isinstance(min_val, ast.Expression)
        operand = min_val.operands[1]
        assert isinstance(operand, ast.ComponentRef)
        assert operand.name == "e.H_b"


def test_custom_units():
    instance = parse_and_instantiate_model("CustomUnits.mo", "A")

    dummy = instance.extends[0].symbols["dummy_parameter"].type.extends[0].symbols["Real"]
    dummy_value_mod = dummy.modification_environment.arguments[0]
    assert dummy_value_mod.value.component.name == "unit"
    assert dummy_value_mod.value.modifications[0].value == "m/s"
    dummy_value_mod = dummy.modification_environment.arguments[-1]
    assert dummy_value_mod.value.component.name == "value"
    assert dummy_value_mod.value.modifications[0].value == 10.0

    flat_tree = tree.flatten_instance(instance)

    assert flat_tree.symbols["dummy_parameter"].unit == "m/s"
    assert flat_tree.symbols["dummy_parameter"].value == 10.0


def test_modification_sub_attr_and_value():
    """Combined k(unit="V/A")=1 form sets both the sub-attribute and the value."""
    flat_tree = parse_and_flatten_model("ModificationSubAttrAndValue.mo", "M")
    assert flat_tree.symbols["k"].unit == "V/A"
    assert flat_tree.symbols["k"].value == 1.0

    flat_tree2 = parse_and_flatten_model("ModificationSubAttrAndValue.mo", "Outer")
    assert flat_tree2.symbols["m.k"].unit == "m/s"
    assert flat_tree2.symbols["m.k"].value == 2.0


def test_nested_modification_arg_scope_resolves_param():
    """Nested ClassModification args (k(unit=p_unit)=2) must resolve the inner
    param ref from the instance scope, not the raw parse-time AST class scope.
    """
    flat = parse_and_flatten_model("NestedModificationArgScope.mo", "NestedModificationArgScope")
    assert flat.symbols["m.k"].value == 2.0
    assert isinstance(
        flat.symbols["m.k"].unit, tree.InstanceSymbol
    ), "unit sub-attr must resolve to InstanceSymbol for p_unit, not stay as raw ComponentRef"


def test_flatten_to_tree_bouncing_ball():
    """flatten_to_tree returns ast.Tree with ast.Class/Symbol, not Instance types."""
    ast_tree = parse_model_files("BouncingBall.mo")
    comp_ref = ast.ComponentRef.from_string("BouncingBall")
    result = tree.flatten_to_tree(ast_tree, comp_ref)

    # Result is ast.Tree
    assert isinstance(result, ast.Tree)

    # Model class is ast.Class, not InstanceClass
    model = result.classes["BouncingBall"]
    assert isinstance(model, ast.Class)
    assert not isinstance(model, tree.InstanceClass)

    # All symbols are ast.Symbol, not InstanceSymbol
    for name, sym in model.symbols.items():
        assert isinstance(sym, ast.Symbol), f"{name} is {type(sym)}"
        assert not isinstance(sym, tree.InstanceSymbol), f"{name} is InstanceSymbol"
        # All .type fields are ComponentRef, not InstanceClass
        assert isinstance(sym.type, ast.ComponentRef), f"{name}.type is {type(sym.type)}"

    # Equations are non-empty
    assert len(model.equations) > 0

    # Expected symbols present
    assert "height" in model.symbols
    assert "velocity" in model.symbols


def test_flatten_to_tree_spring_system():
    """flatten_to_tree handles nested components and state annotation."""
    ast_tree = parse_model_files("SpringSystemExample.mo")
    comp_ref = ast.ComponentRef.from_string("Example.SpringSystem")
    result = tree.flatten_to_tree(ast_tree, comp_ref)

    model = result.classes["Example.SpringSystem"]
    assert isinstance(model, ast.Class)
    assert not isinstance(model, tree.InstanceClass)

    # Nested component symbols present
    for name in ("spring.x", "spring.f", "damper.v", "damper.f", "damper.c"):
        assert name in model.symbols, f"Missing nested symbol: {name}"
        sym = model.symbols[name]
        assert isinstance(sym, ast.Symbol)
        assert not isinstance(sym, tree.InstanceSymbol)

    # Equations collected
    assert len(model.equations) > 0


def test_flatten_to_tree_function_pull():
    ast_tree = parse_model_files("FunctionPull.mo")
    class_name = "Level1.Level2.Level3.Function5"
    comp_ref = ast.ComponentRef.from_string(class_name)

    flat_tree = tree.flatten_to_tree(ast_tree, comp_ref)

    # Functions pulled to top-level classes
    assert "Level1.Level2.Level3.f" in flat_tree.classes
    assert "Level1.Level2.Level3.TestPackage.times2" in flat_tree.classes
    assert "Level1.Level2.Level3.TestPackage.square" in flat_tree.classes
    assert "Level1.Level2.Level3.TestPackage.not_called" not in flat_tree.classes

    # Correct types
    assert flat_tree.classes[class_name].type == "model"
    assert flat_tree.classes["Level1.Level2.Level3.f"].type == "function"

    # Input/output prefixes preserved
    func_t2 = flat_tree.classes["Level1.Level2.Level3.TestPackage.times2"]
    assert "input" in func_t2.symbols["x"].prefixes
    assert "output" in func_t2.symbols["y"].prefixes

    # Function call operators are fully-scoped strings
    func_f = flat_tree.classes["Level1.Level2.Level3.f"]
    assert func_f.statements[0].right.operator == "*"
    assert (
        func_f.statements[0].right.operands[0].operator == "Level1.Level2.Level3.TestPackage.times2"
    )


def test_extend_from_self():
    txt = """
    model A
      extends A;
    end A;"""

    ast_tree = parser.parse(txt)

    with pytest.raises(tree.ModelicaSemanticError, match="Cannot extend class 'A' with itself"):
        instance = tree.instantiate(ast_tree, "A")  # noqa: F841

    class_name = "A"
    comp_ref = ast.ComponentRef.from_string(class_name)

    with pytest.raises(tree.ModelicaSemanticError, match="Cannot extend class 'A' with itself"):
        flat_tree = tree.flatten(ast_tree, comp_ref)  # noqa: F841


def test_connect_equations_hq():
    """Test connect equation generation with flow/non-flow connectors."""
    flat = parse_and_flatten_model("ConnectorHQ.mo", "System")

    # All connector sub-variables present
    for name in (
        "a.up.H",
        "a.up.Q",
        "a.down.H",
        "a.down.Q",
        "b.up.H",
        "b.up.Q",
        "b.down.H",
        "b.down.Q",
        "c.up.H",
        "c.up.Q",
        "c.down.H",
        "c.down.Q",
        "qa.down.H",
        "qa.down.Q",
        "p.H",
        "p.Q",
        "hb.up.H",
        "hb.up.Q",
        "zerotest.H",
        "zerotest.Q",
    ):
        assert name in flat.symbols, f"Missing symbol: {name}"

    # No ConnectClauses remain (all expanded)
    for eq in flat.equations:
        assert not isinstance(eq, ast.ConnectClause), f"Unexpanded ConnectClause: {eq!r}"

    # Non-flow equality equations: internal connects (up.H = down.H per Channel)
    eq_strs = _equation_strings(flat.equations)
    assert ("a.up.H", "=", "a.down.H") in eq_strs
    assert ("b.up.H", "=", "b.down.H") in eq_strs
    assert ("c.up.H", "=", "c.down.H") in eq_strs

    # Cross-connect equality equations
    assert ("qa.down.H", "=", "a.up.H") in eq_strs
    assert ("p.H", "=", "c.up.H") in eq_strs
    assert ("a.down.H", "=", "b.up.H") in eq_strs
    assert ("b.down.H", "=", "hb.up.H") in eq_strs

    # Flow sum-to-zero: 3-way junction at b.up (a.down + c.down + b.up)
    has_3way_flow = any(
        _is_flow_sum_equation(eq, {"a.down.Q", "c.down.Q", "b.up.Q"}) for eq in flat.equations
    )
    assert has_3way_flow, "Expected 3-way flow sum-to-zero for a.down.Q + c.down.Q + b.up.Q"

    # Flow sum-to-zero: qa.down connected to a.up
    has_qa_flow = any(_is_flow_sum_equation(eq, {"qa.down.Q", "a.up.Q"}) for eq in flat.equations)
    assert has_qa_flow, "Expected flow sum-to-zero for qa.down.Q + a.up.Q"

    # Disconnected flow variable (zerotest.Q) defaults to 0
    has_zerotest_q = any(_is_zero_equation(eq, "zerotest.Q") for eq in flat.equations)
    assert has_zerotest_q, "Disconnected flow variable zerotest.Q not set to 0"


def test_connect_equations_simple_circuit():
    """Test connect equation generation with nested connectors via extends."""
    flat = parse_and_flatten_model("SimpleCircuit.mo", "SimpleCircuit")

    # Connector sub-variables present (Pin has v and flow i)
    for comp in ("R1", "C", "R2", "L", "AC"):
        for port in ("p", "n"):
            assert f"{comp}.{port}.v" in flat.symbols
            assert f"{comp}.{port}.i" in flat.symbols
    assert "G.p.v" in flat.symbols
    assert "G.p.i" in flat.symbols

    # No ConnectClauses remain
    for eq in flat.equations:
        assert not isinstance(eq, ast.ConnectClause), f"Unexpanded ConnectClause: {eq!r}"

    # Equality equations for non-flow (v) connections
    eq_strs = _equation_strings(flat.equations)
    assert ("AC.p.v", "=", "R1.p.v") in eq_strs
    assert ("R1.n.v", "=", "C.p.v") in eq_strs


def test_connect_equations_channel():
    """Test connect in a simple two-connector model."""
    flat = parse_and_flatten_model("ConnectorHQ.mo", "Channel")

    # Symbols
    assert "up.H" in flat.symbols
    assert "up.Q" in flat.symbols
    assert "down.H" in flat.symbols
    assert "down.Q" in flat.symbols

    # Equality for non-flow
    eq_strs = _equation_strings(flat.equations)
    assert ("up.H", "=", "down.H") in eq_strs

    # Flow sum-to-zero
    has_flow_eq = any(_is_flow_sum_equation(eq, {"up.Q", "down.Q"}) for eq in flat.equations)
    assert has_flow_eq, "Expected flow sum-to-zero for up.Q + down.Q"


def test_connect_equations_param_index():
    """Test that connect with a parameter (non-literal) array index does not crash.

    Modelica allows connect(a[n], b) where n is a parameter.  The connector-equation
    generator must handle ComponentRef indices, not just Primary (literal) ones.
    """
    flat = parse_and_flatten_model("ArrayConnectParam.mo", "ArrayConnectParam")
    # Connector sub-variables exist
    assert "a.v" in flat.symbols
    assert "a.i" in flat.symbols
    assert "b.v" in flat.symbols
    assert "b.i" in flat.symbols
    # No unexpanded ConnectClauses remain
    for eq in flat.equations:
        assert not isinstance(eq, ast.ConnectClause), f"Unexpanded ConnectClause: {eq!r}"


def _equation_strings(equations):
    """Extract (left_name, '=', right_name) tuples from simple equality equations."""
    result = set()
    for eq in equations:
        if isinstance(eq, ast.Equation):
            left = _ref_name(eq.left)
            right = _ref_name(eq.right)
            if left and right:
                result.add((left, "=", right))
    return result


def _ref_name(node):
    """Get the name from a ComponentRef or similar."""
    if isinstance(node, ast.ComponentRef):
        return node.name
    return None


def _is_zero_equation(eq, var_name):
    """Check if equation is var_name = 0."""
    if not isinstance(eq, ast.Equation):
        return False
    if not (isinstance(eq.right, ast.Primary) and eq.right.value == 0):
        return False
    return isinstance(eq.left, ast.ComponentRef) and eq.left.name == var_name


def _is_flow_sum_equation(eq, var_names):
    """Check if equation is a sum-to-zero of the given variable names."""
    if not isinstance(eq, ast.Equation):
        return False
    if not (isinstance(eq.right, ast.Primary) and eq.right.value == 0):
        return False
    refs = set()
    _collect_component_ref_names([eq.left], refs)
    return refs == var_names


def _flatten_inline(txt, model_name):
    """Parse inline Modelica, instantiate, and flatten. Returns the flat InstanceClass."""
    ast_tree = parser.parse(txt)
    pickled_before = pickle.dumps(ast_tree)
    instance = tree.instantiate(ast_tree, model_name)
    flat = tree.flatten_instance(instance)
    pickled_after = pickle.dumps(ast_tree)
    assert pickled_before == pickled_after, "AST was modified during instantiation/flattening"
    return flat


def test_expression_evaluator_via_modifications_binary():
    """_resolve_modifications evaluates a binary literal expression and sets symbol value."""
    flat = _flatten_inline(
        """
    model M
        constant Real x = 3.0 * 4.0;
    end M;""",
        "M",
    )
    assert flat.symbols["x"].value == 12.0


def test_expression_evaluator_via_modifications_unary():
    """_resolve_modifications evaluates a unary literal expression and sets symbol value."""
    flat = _flatten_inline(
        """
    model M
        constant Real x = -5.0;
    end M;""",
        "M",
    )
    assert flat.symbols["x"].value == -5.0


def test_expression_evaluator_via_modifications_componentref():
    """Constant expressions referencing other constants are now fully folded.

    'n' is a constant whose value is resolved via its type's modification_environment
    even at modification time, so n * 2.0 folds to 6.0.
    """
    flat = _flatten_inline(
        """
    model M
        constant Real n = 3.0;
        constant Real x = n * 2.0;
    end M;""",
        "M",
    )
    assert flat.symbols["x"].value == 6.0


def test_expression_evaluator_via_modifications_error_kept():
    """_resolve_modifications swallows evaluation errors and keeps the expression as-is.

    Division by zero raises ZeroDivisionError → ModelicaSemanticError inside
    _resolve_expression, which _resolve_modifications catches and ignores.
    The symbol value should remain as ast.Expression.
    """
    flat = _flatten_inline(
        """
    model M
        constant Real x = 1.0 / 0.0;
    end M;""",
        "M",
    )
    assert isinstance(flat.symbols["x"].value, ast.Expression)


def test_expression_evaluator_via_dimensions_literal():
    """_resolve_dimensions evaluates a pure-literal expression dimension."""
    flat = _flatten_inline(
        """
    model M
        Real x[2 + 1];
    end M;""",
        "M",
    )
    dims = flat.symbols["x"].dimensions
    assert len(dims) == 1
    assert isinstance(dims[0][0], ast.Primary)
    assert dims[0][0].value == 3


def test_expression_evaluator_via_dimensions_componentref():
    """_resolve_dimensions resolves a ComponentRef operand then evaluates.

    Unlike _resolve_modifications, _resolve_dimensions processes dimensions after
    symbols are instantiated, so parameter values are available.
    """
    flat = _flatten_inline(
        """
    model M
        parameter Integer n = 3;
        Real x[n + 1];
    end M;""",
        "M",
    )
    dims = flat.symbols["x"].dimensions
    assert len(dims) == 1
    assert isinstance(dims[0][0], ast.Primary)
    assert dims[0][0].value == 4


def test_generate_value_equations_variable():
    """Value modification on a plain variable becomes an equation (MLS 5.6.2 step 1.4)."""
    flat = _flatten_inline(
        """
    model M
        Real x = 3.0;
    end M;""",
        "M",
    )
    # Symbol value cleared to sentinel
    assert isinstance(flat.symbols["x"].value, ast.Primary)
    assert flat.symbols["x"].value.value is None
    # Equation x = 3.0 emitted
    eq_map = {
        eq.left.name: eq.right
        for eq in flat.equations
        if isinstance(eq, ast.Equation) and isinstance(eq.left, ast.ComponentRef)
    }
    assert "x" in eq_map, "Expected value equation for x"
    assert isinstance(eq_map["x"], ast.Primary) and eq_map["x"].value == 3.0


def test_generate_value_equations_parameter_retained():
    """Parameter value stays on the symbol — no equation emitted (MLS 5.6.2 step 1.4)."""
    flat = _flatten_inline(
        """
    model M
        parameter Real p = 1.0;
    end M;""",
        "M",
    )
    assert flat.symbols["p"].value == 1.0
    eq_names = {
        eq.left.name
        for eq in flat.equations
        if isinstance(eq, ast.Equation) and isinstance(eq.left, ast.ComponentRef)
    }
    assert "p" not in eq_names, "Parameter should not get a value equation"


def test_generate_value_equations_constant_retained():
    """Constant value stays on the symbol — no equation emitted."""
    flat = _flatten_inline(
        """
    model M
        constant Real c = 2.0;
    end M;""",
        "M",
    )
    assert flat.symbols["c"].value == 2.0
    eq_names = {
        eq.left.name
        for eq in flat.equations
        if isinstance(eq, ast.Equation) and isinstance(eq.left, ast.ComponentRef)
    }
    assert "c" not in eq_names, "Constant should not get a value equation"


def test_generate_value_equations_nested_component():
    """Value equation for a nested component uses the flat symbol name as LHS."""
    flat = _flatten_inline(
        """
    model Inner
        Real y = 2.0;
    end Inner;
    model Outer
        Inner sub;
    end Outer;""",
        "Outer",
    )
    eq_map = {
        eq.left.name: eq.right
        for eq in flat.equations
        if isinstance(eq, ast.Equation) and isinstance(eq.left, ast.ComponentRef)
    }
    assert "sub.y" in eq_map, "Expected value equation for sub.y"
    assert isinstance(eq_map["sub.y"], ast.Primary) and eq_map["sub.y"].value == 2.0


def test_generate_value_equations_ref_rhs_flattened():
    """Value equation RHS ComponentRef is resolved to flat name."""
    flat = _flatten_inline(
        """
    model Inner
        Real a = 1.0;
        Real b = a;
    end Inner;
    model Outer
        Inner sub;
    end Outer;""",
        "Outer",
    )
    eq_map = {
        eq.left.name: eq.right
        for eq in flat.equations
        if isinstance(eq, ast.Equation) and isinstance(eq.left, ast.ComponentRef)
    }
    assert "sub.b" in eq_map, "Expected value equation for sub.b"
    rhs = eq_map["sub.b"]
    # RHS must be a ComponentRef with the flat name (sub.a), not the source-scope name (a)
    assert isinstance(rhs, ast.ComponentRef), f"Expected ComponentRef, got {rhs!r}"
    assert rhs.name == "sub.a", f"Expected flat ref 'sub.a', got {rhs.name!r}"


def test_generate_value_equations_not_skipped_when_explicit_equation_exists():
    """An inherited modification value equation coexists with an explicit equation for it.

    `extends SubTree(t = 2)` overrides SubTree's declaration equation (MLS "Declaration
    Equations"), giving `t = 2`. Modelica equations are simultaneous (MLS 8.1/4.4): an
    equation-section equation does not override a declaration/modification equation, it
    is just another equation of the flat system. So both `t = 2` and Tree's own explicit
    equation must be emitted, even though the resulting system is over-determined.
    """
    flat = _flatten_inline(
        """
    model SubTree
        Integer t = 1;
    end SubTree;
    model Tree
        extends SubTree(t = 2);
    equation
        t = 3;
    end Tree;""",
        "Tree",
    )
    t_eqs = [
        eq
        for eq in flat.equations
        if isinstance(eq, ast.Equation)
        and (
            (isinstance(eq.left, ast.ComponentRef) and eq.left.name == "t")
            or (isinstance(eq.right, ast.ComponentRef) and eq.right.name == "t")
        )
    ]
    assert len(t_eqs) == 2, f"Expected both equations for t, got {t_eqs!r}"


def test_builtin_redeclare_replaceable_propagated():
    """replaceable on a builtin-type symbol survives the redeclare → flatten path."""
    flat = _flatten_inline(
        """
    model Base
        replaceable Real x = 0.0, y = 1.0;
    end Base;
    model Derived
        extends Base(replaceable Integer x = 3, redeclare Integer y = 4);
    end Derived;""",
        "Derived",
    )
    assert flat.symbols["x"].type.name == "Integer"
    assert flat.symbols["x"].replaceable is True
    assert flat.symbols["y"].type.name == "Integer"
    assert flat.symbols["y"].replaceable is False


def test_builtin_redeclare_replaceable_nested_component():
    """replaceable propagates for a builtin-type component inside a nested instance."""
    flat = _flatten_inline(
        """
    model Base
        replaceable Real x = 0.0;
    end Base;
    model Derived
        extends Base(replaceable Integer x = 3);
    end Derived;
    model Outer
        Derived d;
    end Outer;""",
        "Outer",
    )
    assert flat.symbols["d.x"].type.name == "Integer"
    assert flat.symbols["d.x"].replaceable is True


def test_builtin_alias_final_inner_outer_propagated():
    """prefixes declared on a type-alias component reach the flattened builtin symbol."""
    flat = _flatten_inline(
        """
    model M
        type MyReal = Real;
        MyReal a;
        final MyReal x;
        inner MyReal y;
        outer MyReal z;
    end M;""",
        "M",
    )
    assert flat.symbols["a"].type.name == "Real"
    assert flat.symbols["a"].final is False
    assert flat.symbols["a"].inner is False
    assert flat.symbols["a"].outer is False
    assert flat.symbols["x"].final is True
    assert flat.symbols["y"].inner is True
    assert flat.symbols["z"].outer is True


def test_constant_in_modification_scope():
    """Constant used as modification value in its declaring class resolves across extends.

    PartialFriction declares 'constant Integer Backward = -1' and uses it in
    'mode(final min = Backward)'.  BearingFriction extends PartialFriction.
    Flattening GearType1 (which has a BearingFriction component) must resolve
    Backward from the PartialFriction extends scope.
    """
    flat = parse_and_flatten_model("ConstantInModificationScope.mo", "GearType1")
    assert "bearingFriction.Backward" in flat.symbols
    assert flat.symbols["bearingFriction.Backward"].value == -1
    # mode.min must resolve to the Backward symbol (not stay as a raw ComponentRef)
    assert flat.symbols["bearingFriction.mode"].min is not None


def test_composite_lookup_via_extends():
    """Composite reference (component.attr) resolves attr through extends-of-extends.

    clutch.tau: 'tau' is not directly in Clutch — it lives in PartialCompliant,
    which is extended by PartialCompliantWithRelativeStates, which Clutch extends.
    The lookup must traverse the full extends chain to find tau.
    """
    flat = parse_and_flatten_model("CompositeLookupViaExtends.mo", "CompositeLookupViaExtends")
    assert "clutch.tau" in flat.symbols
    assert "tau_copy" in flat.symbols


def test_inherited_connector_stub_via_extends():
    """Connectors inherited via extends get connector stubs so connect() resolves (MLS 9.1.3)."""
    ast_tree = parse_model_files("InheritedConnectorConnect.mo")
    flat = tree.flatten(ast_tree, ast.ComponentRef.from_string("P.System"))
    symbols = flat.classes["P.System"].symbols
    assert "m.flange_a.s" in symbols
    assert "m.flange_b.s" in symbols


def test_expandable_connector_stub():
    """Expandable connectors get connector stubs so whole-bus connect() resolves (MLS 9.1.3).

    Without a stub for the expandable-connector-typed symbol, expand_connectors
    raised KeyError looking up the bus name in node.symbols.
    """
    ast_tree = parse_model_files("ExpandableConnectorBus.mo")
    flat = tree.flatten(ast_tree, ast.ComponentRef.from_string("P.System"))
    ref_names = set()
    _collect_component_ref_names(flat.classes["P.System"].equations, ref_names)
    assert "s.bus.speed" in ref_names
    assert "bus.speed" in ref_names


def test_multiple_extends_array_dim():
    """Array dimension from a second unnamed extends resolves in the correct scope.

    When a class has multiple extends clauses, each is an unnamed instance (name='').
    The dimension parameter of the second extends must not be confused with the
    first extends instance due to the shared empty key in parent_instance.classes.
    """
    flat = parse_and_flatten_model("MultipleExtendsArrayDim.mo", "P.Both")

    assert "arr" in flat.symbols
    (dim,) = flat.symbols["arr"].dimensions[0]
    assert isinstance(dim, ast.Primary) and dim.value == 2


def test_redeclare_inherits_modification():
    """Redeclared replaceable whose subtype extends the original must accept
    the original modifier (e.g. 'final m=m').

    Bug: when _instantiate_class's early-return path synced instantiation_state
    onto a stub InstanceClass (no symbols, no extends), that stub was later used
    as 'from_class' for a new instantiation with modifications.  Because the stub
    had empty symbols, _check_modification_targets raised 'Trying to modify symbol
    m' even though m is a valid direct symbol of PartialPort.
    """
    flat = parse_and_flatten_model("RedeclareInheritsModification.mo", "P.Example")

    assert "mac.port.m" in flat.symbols
    assert "mac.port.v" in flat.symbols
    assert "mac.port.flag" in flat.symbols


def test_clock_builtin_type():
    """Clock is a predefined opaque type (MLS 16.2) and must be resolvable as a base type.

    connector ClockInput = input Clock triggers 'extends Clock' during instantiation.
    Before the fix, this raised ModelicaSemanticError('Extends name Clock not found').
    """
    ast_tree = ast.Tree()
    assert "Clock" in ast_tree.classes
    assert ast_tree.classes["Clock"].type == "type"

    assert parse_and_flatten_model("ClockBuiltin.mo", "ClockBuiltin.UsesClock") is not None


def test_inherited_equation_refs_resolved():
    """Equations referencing inherited (via extends) symbols flatten before resolution (MLS 5.6.2)."""
    flat = parse_and_flatten_model("InheritedEquationConnector.mo", "P.B")
    ref_names = set()
    _collect_component_ref_names(flat.equations, ref_names)
    assert "flange_a.phi" in ref_names
    _assert_no_child_refs(flat.equations)


def test_encapsulated_import_enclosing_package():
    """Encapsulated function that imports its enclosing package to type its inputs resolves.

    Before the fix, _get_common_parent returned (pkg, "") when the import target was an
    ancestor of the scope; _find_composite_name("", pkg) returned None, so the import
    silently failed and the input type raised NameLookupError.
    """
    flat = parse_and_flatten_model("EncapsulatedImportEnclosingPackage.mo", "UseOrientation")
    assert "R.T[1,1]" in flat.symbols or any(k.startswith("R") for k in flat.symbols)


def test_class_extends_redeclare_record():
    """Flatten a redeclared `class extends` record without infinite recursion (MLS 7.3.2)."""
    flat = parse_and_flatten_model("ClassExtendsRedeclare.mo", "P.M")
    assert "s.d" in flat.symbols
    assert "s.T" in flat.symbols


def test_inherited_type_in_enclosing_scope():
    """Resolve a record field type inherited via extends into an enclosing package (MLS 5.3)."""
    flat = parse_and_flatten_model("InheritedTypeEnclosingScope.mo", "P.M")
    assert "s.d" in flat.symbols


def test_user_defined_enum_flatten():
    """User-defined enumeration types parse and flatten like builtin StateSelect.

    ``type E = enumeration(one, two, three)`` should produce an enum type whose
    typed variables (``E e``) are treated as simple types in flattening — i.e. the
    literal symbols are *not* flattened into the model as child variables.
    """
    src = """
    model M
      type E = enumeration(one, two, three);
      E e(start = E.one);
    end M;
    """
    ast_tree = parser.parse(src)
    # Enum class parses correctly
    enum_cls = ast_tree.classes["M"].classes["E"]
    assert enum_cls.enumeration is True
    assert list(enum_cls.symbols.keys()) == ["one", "two", "three"]
    assert isinstance(enum_cls.symbols["two"], ast.EnumerationLiteral)
    assert enum_cls.symbols["two"].ordinal == 2

    # Flattening succeeds and treats 'e' as a simple type (no spurious child symbols)
    flat = tree.flatten_class(ast_tree, "M")
    assert "e" in flat.symbols
    # Enum literals must not leak into the flat model as child symbols
    assert not any(name.startswith("e.") for name in flat.symbols)


def test_builtin_enum_literals_are_enumeration_literals():
    """Builtin enum types (StateSelect) use EnumerationLiteral with ordinals."""
    ast_tree = ast.Tree()
    ss = ast_tree.classes["StateSelect"]
    assert ast.is_enumeration(ss)
    assert list(ss.symbols.keys()) == ["never", "avoid", "default", "prefer", "always"]
    for expected_ordinal, (name, sym) in enumerate(ss.symbols.items(), start=1):
        assert isinstance(sym, ast.EnumerationLiteral), f"{name!r} should be EnumerationLiteral"
        assert sym.ordinal == expected_ordinal


def test_replaceable_record_modification():
    """Modifications targeting symbols that only exist in the concrete redeclaration
    of a replaceable record must not raise a false 'symbol does not exist' error.

    Before the fix, _check_modification_targets ran against the abstract (empty) base
    class and rejected modifications that are valid for the concrete redeclared class.
    """
    flat = parse_and_flatten_model("ReplaceableModification.mo", "P.Example")
    assert "state.p" in flat.symbols
    assert "state.T" in flat.symbols


def test_time_in_modification():
    """'time' used as a modification value must not raise NameLookupError.

    'time' is a Modelica built-in variable (MLS §2.7) and must be left as-is
    during modification resolution rather than looked up in the instance tree.
    """
    flat = parse_and_flatten_model("TimeInModification.mo", "TimeInModification")
    assert "wz" in flat.symbols
    assert "x" in flat.symbols


def test_encapsulated_fully_qualified_type():
    """Inside an encapsulated class, a fully-qualified name whose first component is a
    root-level package must still resolve (MLS §13.2.3).

    Before the fix, step 7c in _find_simple_name blocked 'package' classes at the root
    scope, so Modelica.Units.SI.Position failed with 'Type ... not found'.
    """
    instance = parse_and_instantiate_model(
        "EncapsulatedFullyQualifiedType.mo", "P.EncapsulatedModel"
    )
    flat = tree.flatten_instance(instance)
    assert "x" in flat.symbols


def test_div_builtin_in_dimension():
    """div() built-in used in a parameter dimension expression must evaluate correctly.

    Before the fix, div was absent from ExpressionEvaluator.binary_operator, so
    'n = div(shift, resolution)' stayed as an unevaluated Expression.  The downstream
    dimension 'buf[n+1]' then tried Python arithmetic on the Expression value and raised
    ModelicaSemanticError('Unable to evaluate expression: ...')
    """
    instance = parse_and_instantiate_model("DivInDimension.mo", "P.Example")
    flat = tree.flatten_instance(instance)
    # buf is a 2-element array: div(2,2)+1 = 2; flattened as b.buf[1] and b.buf[2]
    assert "b.buf[1]" in flat.symbols or "b.buf" in flat.symbols


@pytest.mark.msl
@pytest.mark.skipif(
    not os.path.isfile(os.path.join(MSL4_DIR, "Modelica", "package.mo")),
    reason="MSL-4.0.x submodule not present",
)
def test_idealgash2o_stale_stub_import():
    """Flatten Modelica.Media.Examples.IdealGasH2O through a stale root stub (MLS 5.3).

    Guards this commit's fix: Media.Common is fully instantiated while the root
    Modelica package is still a partial stub, so the enclosing
    ``import Modelica.Units.SI`` must resolve via the authoritative root instance
    (else ``Type SI.Area of symbol AMIN not found``).  Not minimally reproducible
    — it needs the real MSL Media hierarchy — so this is an ``msl``-marked test,
    deselected from the default fast run.  Run it with ``pytest -m msl``.
    """
    root = parser.modelicapath_to_tree([MSL4_DIR])
    flat = tree.flatten_class(root, "Modelica.Media.Examples.IdealGasH2O")
    assert "state.p" in flat.symbols
    assert "smoothState.T" in flat.symbols


if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__])
