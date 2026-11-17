#!/usr/bin/env python
"""
Tests for tree.instantiate() behavior.
"""

import os

from conftest_parse import (
    MODEL_DIR,
    check_instance_tree_is_all_instances,
    check_redeclare_expects,
    get_modifiers_by_name,
    parse_and_instantiate_model,
    parse_model_files,
    redeclare_expect,
)

from pymoca import ast
from pymoca import parser
from pymoca import tree
from pymoca.tree._name_lookup import _find_name, _flatten_first_and_find_rest

import pytest


def test_instantiation_same_names():
    """Test instantiating a class with the same name as the enclosing class"""
    instance = parse_and_instantiate_model("TreeInTree.mo", "Tree.Tree")

    # Check symbol is present and has the correct modification
    assert "t" in instance.extends[0].symbols, "t not found in instance"
    t = instance.extends[0].symbols["t"]
    t_value_mods = get_modifiers_by_name(t, "value")
    t_value_mod = t_value_mods[-1]
    assert t_value_mod.value.modifications[-1].value == 2


def test_instantiation_modification_scope_instance_class():
    """Test that modification scopes are updated to InstanceClass"""
    ast_tree = parse_model_files("TreeInTree.mo")
    instance = tree.instantiate(ast_tree, "Tree.Tree")
    assert instance is not None

    assert "t" in instance.extends[0].symbols, "t not found in instance"
    t = instance.extends[0].symbols["t"]
    for mod in t.type.symbols["Integer"].modification_environment.arguments:
        assert isinstance(mod.scope, tree.InstanceClass), "scope not InstanceClass"


def test_instantiation_modification_scope_spec_example():
    """Test scopes of modification references and values with example from spec"""

    instance = parse_and_instantiate_model("ModificationScope.mo", "B")
    flat = tree.flatten_instance(instance)

    # Literal values
    assert flat.symbols["R"].value == 3
    assert flat.symbols["a.R"].value == 4
    assert flat.symbols["j.R"].value == 2

    # c(R=5) is an explicit instance override; the rest inherit Load's original R=R binding
    assert flat.symbols["c.R"].value == 5

    # b, e, g have no instance override; Load's original R=R is preserved through the redeclare
    # so they resolve to the outer R from B (per OMC 1.25.0 reference output)
    for name in ("b.R", "e.R", "g.R"):
        assert isinstance(flat.symbols[name].value, tree.InstanceSymbol), f"{name} should be ref"

    # d.R: LoadError has extends Resistor(R=R) — self-referencing in Resistor scope
    assert isinstance(flat.symbols["d.R"].value, tree.InstanceSymbol)

    # f.R, h.R, i.R: InstanceSymbol references (parameter refs)
    for name in ("f.R", "h.R", "i.R"):
        assert isinstance(flat.symbols[name].value, tree.InstanceSymbol), f"{name} should be ref"


def test_instantiation_tree():
    instance = parse_and_instantiate_model("InstantiationTree.mo", "TreeModel.Tree")

    # Check that we have a fully connected InstanceTree with only Instances
    non_instances = check_instance_tree_is_all_instances(instance.root)

    assert len(non_instances) == 0, f"\nFound non-instances in InstanceTree:\n{non_instances}"

    # Check that instantiation copies only what's needed (not full package hierarchy)
    assert "TreeModel" in instance.root.classes
    # According to the spec, the instantiated class has root as parent
    assert "Tree" in instance.root.classes
    tree_model = instance.root.classes["TreeModel"]
    assert sorted(tree_model.classes) == ["Tree"]

    # Check that all symbols are present
    assert sorted(instance.symbols) == ["b", "l", "l2", "w"]
    assert "t" in instance.extends[0].symbols
    # Check redeclare of w: Wood extends Maple extends Real(nominal=2.0)
    w = instance.symbols["w"]
    assert w.type.ast_ref.name == "Wood"
    w_real = w.type.extends[0].extends[0].symbols["Real"]
    w_mod = w_real.modification_environment.arguments[-1]
    assert w_mod.value.component.name == "nominal"
    assert w_mod.value.modifications[0].value == pytest.approx(2.0)
    # Check redeclare of b.t: Wood extends Oak extends Real(nominal=1.0)
    b_t = instance.symbols["b"].type.extends[0].symbols["t"]
    assert b_t.type.ast_ref.name == "Wood"
    b_t_real = b_t.type.extends[0].extends[0].symbols["Real"]
    b_t_mod = b_t_real.modification_environment.arguments[-1]
    assert b_t_mod.value.component.name == "nominal"
    assert b_t_mod.value.modifications[0].value == pytest.approx(1.0)
    # Check modification of l.c: Cell extends Integer, last value mod (=1) wins
    l_c = instance.symbols["l"].type.symbols["c"]
    l_c_int = l_c.type.extends[0].symbols["Integer"]
    l_c_mod = l_c_int.modification_environment.arguments[-1]
    assert l_c_mod.value.component.name == "value"
    assert l_c_mod.value.modifications[0].value == 1
    # Check inherited symbol t: Wood extends Maple extends Real(nominal=2.0)
    t = instance.extends[0].symbols["t"]
    assert t.type.ast_ref.name == "Wood"
    t_real = t.type.extends[0].extends[0].symbols["Real"]
    t_mod = t_real.modification_environment.arguments[-1]
    assert t_mod.value.component.name == "nominal"
    assert t_mod.value.modifications[0].value == pytest.approx(2.0)


@pytest.mark.skip(reason="Only keeping first of same names not implemented yet in instantiation")
def test_instantiation_function_input_order():
    """Check that only first definitions of inputs are used in function instantiation."""
    txt = """
        package P
          function A
            input Real a;
            input Real b;
          end A;

          function B
            extends A;
            input Real b; // Should be discarded as 2nd instance of b
            input Real a; // Should be discarded as 2nd instance of a
          end B;

          function C
            input Real b;
            input Real a;
            extends A;    // Should be discarded as 2nd instances of a and b
          end C;

          function D
            input Real a; // Should be kept
            extends A;    // a should be discarded and b kept from this
            input Real b; // Should be discarded as 2nd instance of b
          end D;
        end P;
    """

    ast_tree = parser.parse(txt)

    # TODO: Update this test when we implement maintaining declaration order
    # Our current implementation does not maintain perfect declaration order because
    # during parsing we store the declarations from the Modelica code in separate
    # symbols and extends dictionaries. To maintain order we need to store the
    # declarations in one list or ordered dictionary containing both symbols and
    # extends. The checks below are intended to fail until we have implemented this.

    instance = tree.instantiate(ast_tree, "P.B")
    assert ("a", "b") == tuple(instance.extends[0].symbols)
    assert "b" not in instance.symbols
    assert "a" not in instance.symbols

    instance = tree.instantiate(ast_tree, "P.C")
    assert ("b", "a") == tuple(instance.symbols)
    assert "a" not in instance.extends[0].symbols
    assert "b" not in instance.extends[0].symbols

    instance = tree.instantiate(ast_tree, "P.D")
    assert "a" in instance.symbols
    assert "b" not in instance.symbols
    assert "a" not in instance.extends[0].symbols
    assert "b" in instance.extends[0].symbols


# TODO: Add additional tests for child name culling
# See in Modelica Language Spec v3.5:
# * Section 5.6.1.4 Steps of Instantiation, under "The inherited contents of the element"
# * Section 4.3 Declaration Order and Usage before Declaration
# * Chapter 12
# Tests to add:
# - function output order
# - intermixed funtion input and output declarations
# - default on one function argument but not the other with the same name - error?
# - records used as arguments to external functions
# - instances with same name but different type
# - instances with same name, same type, but different prefixes
# - instances with same name, same type, but with modifications of differing components
# - instances with same name, same type, but with modifications of same comps, different values
# - instances with same name, same type, but different redeclarations
# - ...?


def test_inheritance():
    instance = parse_and_instantiate_model("InheritanceInstantiation.mo", "C2")

    bcomp1_b = instance.symbols["bcomp1"].type.extends[0].symbols["b"]
    bcomp1_b_type = bcomp1_b.type.symbols["Integer"]
    bcomp1_b_mod = bcomp1_b_type.modification_environment.arguments[-1].value
    assert bcomp1_b_mod.component.name == "value"
    bcomp1_b_value = bcomp1_b_mod.modifications[-1].value
    assert bcomp1_b_value == 3

    bcomp3_a = instance.symbols["bcomp3"].type.extends[0].extends[0].symbols["a"]
    bcomp3_a_type = bcomp3_a.type.symbols["Real"]
    bcomp3_a_mod = bcomp3_a_type.modification_environment.arguments[-1].value
    assert bcomp3_a_mod.component.name == "value"
    bcomp3_a_value = bcomp3_a_mod.modifications[-1].value
    assert bcomp3_a_value == 1

    bcomp3_b = instance.symbols["bcomp3"].type.extends[0].extends[0].symbols["b"]
    bcomp3_b_type = bcomp3_b.type.symbols["Integer"]
    bcomp3_b_mod = bcomp3_b_type.modification_environment.arguments[-1].value
    assert bcomp3_b_mod.component.name == "value"
    bcomp3_b_value = bcomp3_b_mod.modifications[-1].value
    assert bcomp3_b_value == 2

    flat_tree = tree.flatten_instance(instance)

    assert flat_tree.symbols["bcomp1.b"].value == 3
    assert flat_tree.symbols["bcomp3.a"].value == 1
    assert flat_tree.symbols["bcomp3.b"].value == 2


def test_inheritance_resistor():
    instance = parse_and_instantiate_model("InheritanceInstantiationResistor.mo", "P.M")

    # Tests that only depend on instantiation, not flattening
    p = instance.extends[0].extends[0].symbols["p"]
    p_v = p.type.symbols["v"]
    p_v_modification = p_v.type.extends[0].symbols["Real"].modification_environment
    for arg in p_v_modification.arguments:
        if arg.value.component.name == "nominal":
            p_v_nominal = arg.value.modifications[0].value
        if arg.value.component.name == "max":
            p_v_max = arg.value.modifications[0].value
    assert p_v_nominal == 1
    assert p_v_max == 3.0

    n = instance.extends[0].extends[0].symbols["n"]
    n_v = n.type.symbols["v"]
    n_v_modification = n_v.type.extends[0].symbols["Real"].modification_environment
    for arg in n_v_modification.arguments:
        if arg.value.component.name == "nominal":
            n_v_nominal = arg.value.modifications[0].value
        if arg.value.component.name == "max":
            n_v_max = arg.value.modifications[0].value
    assert n_v_nominal == 2
    assert n_v_max == 3.0

    flat_tree = tree.flatten_instance(instance)
    assert flat_tree.symbols["p.v"].nominal == 1.0
    assert flat_tree.symbols["p.v"].max == 3.0
    assert flat_tree.symbols["n.v"].nominal == 2.0
    assert flat_tree.symbols["n.v"].max == 3.0


def test_inheritance_instantiation():
    instance = parse_and_instantiate_model("RecursiveInstantiation.mo", "M")
    ast_tree = instance.ast_ref.root

    # Test parse that b modification to redeclare A contains the p=1 modification
    b_mod = ast_tree.classes["M"].symbols["b"].class_modification
    assert b_mod is not None
    redeclare_mod = b_mod.arguments[0].value.class_modification
    assert len(redeclare_mod.arguments) == 1
    assert redeclare_mod.arguments[0].value.component.name == "p"
    assert redeclare_mod.arguments[0].value.modifications[0].value == 1

    # Test instantiation for inheritance, modifications, redeclares including scope

    assert "b" in instance.symbols
    b_type = instance.symbols["b"].type
    assert "a" in b_type.symbols
    a_extends_names = [extends.ast_ref.name for extends in b_type.symbols["a"].type.extends]
    assert "D" in a_extends_names, "model A not properly redeclared in b"
    a_extends_D_index = a_extends_names.index("D")
    a_extends_D = b_type.symbols["a"].type.extends[a_extends_D_index]
    assert "p" in a_extends_D.symbols
    p = a_extends_D.symbols["p"]
    assert "parameter" in p.prefixes
    assert isinstance(p.type, tree.InstanceClass), "b.a.p not properly instantiated"
    assert p.type.name == "E"
    assert len(p.type.extends) == 1, "b.a.p extends not properly instantiated"
    assert "Integer" in p.type.extends[0].symbols, "b.a.p incorrect type E scope"
    p_int_symbol = p.type.extends[0].symbols["Integer"]
    p_int_symbol_modification = p_int_symbol.modification_environment.arguments[0]
    p_int_symbol_value = p_int_symbol_modification.value.modifications[0].value
    assert p_int_symbol_value == 1, "b.a.p=1 modification not applied"
    assert len(a_extends_D.extends) == 1, "b.a extends not properly instantiated"
    a_extends_D_extends_C = a_extends_D.extends[0]
    assert "e" in a_extends_D_extends_C.symbols
    b_a_e = a_extends_D_extends_C.symbols["e"]
    e_type = b_a_e.type
    assert len(e_type.extends) == 1, "b.a.e extends not properly instantiated"
    assert "Real" in e_type.extends[0].symbols, "b.a.e incorrect type E scope"

    flat_tree = tree.flatten_instance(instance)

    assert flat_tree.symbols["b.a.e"].type.name == "Real"
    assert flat_tree.symbols["b.a.p"].type.name == "Integer"
    assert flat_tree.symbols["b.a.p"].value == 1


def test_visibility_in_instance():
    """Test visibility is set correctly in instance elements"""

    ast_tree = parse_model_files("Visibility.mo")

    instance = tree.instantiate(ast_tree, "A")
    # Test visibility of the classes
    B = instance.classes["B"]
    C = instance.classes["C"]
    assert ast.Visibility.PUBLIC == instance.visibility
    assert ast.Visibility.PROTECTED == B.visibility
    assert ast.Visibility.PUBLIC == C.visibility

    # Test visibility of the symbols
    assert ast.Visibility.PROTECTED == instance.symbols["b"].visibility
    assert ast.Visibility.PUBLIC == instance.symbols["c"].visibility
    assert ast.Visibility.PUBLIC == instance.symbols["d"].visibility
    assert ast.Visibility.PROTECTED == instance.symbols["e"].visibility
    assert ast.Visibility.PROTECTED == instance.symbols["f"].visibility

    # Test that we propagate visibility to symbols in a symbol type
    for symbol in instance.symbols["b"].type.symbols.values():
        assert ast.Visibility.PROTECTED == symbol.visibility

    # Test extends visibility
    c_type_extends_B = instance.symbols["c"].type.extends[0]
    assert ast.Visibility.PROTECTED == c_type_extends_B.visibility

    # Test for protected visibility in inherited symbols in a public class
    for symbol in c_type_extends_B.symbols.values():
        assert ast.Visibility.PROTECTED == symbol.visibility


def test_extends_lookup_not_in_extended():
    """
    Check that we do not find 'model B' in the unnamed node 'A' in the
    following model. We also check that the order of inheritance does not
    matter for the error message.
    """
    txt = """
    model A
        model B
            parameter Real x = 1.0;
        end B;
        parameter Real y = 2.0;
    end A;

    model C
        extends {};
        extends {};
    end C;
    """

    for extends_1, extends_2 in [("A", "B"), ("B", "A")]:
        ast_tree = parser.parse(txt.format(extends_1, extends_2))

        with pytest.raises(tree.ModelicaSemanticError, match="Extends name B not found in scope C"):
            instance = tree.instantiate(ast_tree, "C")  # noqa: F841


def test_error_extends_class_also_extended_name_simple():
    """
    Check that we are not allowed to inherit from `model B`, because a
    symbol with the same name is inherited from `model A`. We also check
    that the order of inheritance does not matter for the error message.
    """
    txt = """
    model A
        parameter Real B = 1.0;
        parameter Real y = 2.0;
    end A;

    model B
        parameter Real x = 1.0;
    end B;

    model C
        extends {};
        extends {};
    end C;
    """

    for extends_1, extends_2 in [("A", "B")]:
        ast_tree = parser.parse(txt.format(extends_1, extends_2))

        with pytest.raises(
            tree.ModelicaSemanticError,
            match="Cannot extend 'C' with 'B'; 'B' also exists in names inherited from 'A'",
        ):
            instance = tree.instantiate(ast_tree, "C")  # noqa: F841


def test_extends_class_name_matches_own_member():
    """
    Extending `model A` where A has a member named `A` is valid: the
    extends-name lookup uses search_inherited=False, so the inherited symbol
    cannot shadow the class name during resolution.  Only cross-contamination
    between *different* extends clauses is prohibited (MLS §5.6.1).
    """
    txt = """
    model A
        parameter Real A = 1.0;
        parameter Real y = 2.0;
    end A;

    model C
        extends A;
    end C;
    """

    ast_tree = parser.parse(txt)
    instance = tree.instantiate(ast_tree, "C")
    assert instance is not None


def test_error_extends_class_also_extended_name_nested():
    """
    Check that we are not allowed to inherit from `model B.C`, because a
    symbol with the the name 'B' is inherited from `model A`. We also check
    that the order of inheritance does not matter for the error message.
    """
    txt = """
    model A
        parameter Real B = 1.0;
        parameter Real y = 2.0;
    end A;

    package B
        model C
            parameter Real x = 1.0;
        end C;
    end B;

    model D
        extends {};
        extends {};
    end D;
    """

    for extends_1, extends_2 in [("A", "B.C"), ("B.C", "A")]:
        ast_tree = parser.parse(txt.format(extends_1, extends_2))

        with pytest.raises(
            tree.ModelicaSemanticError,
            match="Cannot extend 'D' with 'B.C'; 'B' also exists in names inherited from 'A'",
        ):
            instance = tree.instantiate(ast_tree, "D")  # noqa: F841


def test_extends_path_middle_component_not_checked():
    """A name that appears in the MIDDLE of a qualified extends path and is also
    inherited from another base class must not trigger a false ambiguity error.

    Only the FIRST path component can be ambiguous (it is the one resolved from
    the merged scope); subsequent components are resolved relative to the preceding
    component and are not affected by inherited names (MLS §5.3.2).
    """
    instance = parse_and_instantiate_model("ExtendsPathMiddleConflict.mo", "ExtendsOuter")
    assert instance is not None


def test_nested_classes():
    instance = parse_and_instantiate_model("NestedClasses.mo", "C2")

    c2_v1 = instance.extends[0].symbols["v1"].type.extends[0].symbols["Real"]
    c2_v1_mod = c2_v1.modification_environment.arguments[-1].value
    assert c2_v1_mod.component.name == "nominal"
    assert c2_v1_mod.modifications[0].value == 1000

    c2_v2 = instance.extends[0].symbols["v1"].type.extends[0].symbols["Real"]
    c2_v2_mod = c2_v2.modification_environment.arguments[-1].value
    assert c2_v2_mod.component.name == "nominal"
    assert c2_v2_mod.modifications[0].value == 1000

    flat_tree = tree.flatten_instance(instance)

    assert flat_tree.symbols["v1"].nominal == 1000.0
    assert flat_tree.symbols["v2"].nominal == 1000.0


def test_nested_symbol_modifier():
    instance = parse_and_instantiate_model("NestedClasses.mo", "C3")

    c3_v1_type = instance.symbols["c"].type.extends[0].symbols["v1"].type
    c3_v1 = c3_v1_type.extends[0].symbols["Real"]
    c3_v1_mod = c3_v1.modification_environment.arguments[-1].value
    assert c3_v1_mod.component.name == "nominal"
    assert c3_v1_mod.modifications[0].value == 10

    flat_tree = tree.flatten_instance(instance)

    assert flat_tree.symbols["c.v1"].nominal == 10.0
    assert flat_tree.symbols["c.v2"].nominal == 1000.0


def test_inheritance_symbol_modifiers():
    instance = parse_and_instantiate_model("Inheritance.mo", "Sub")

    x_type = instance.extends[0].symbols["x"].type.symbols["Real"]
    x_mod = x_type.modification_environment.arguments[0]
    assert x_mod.value.component.name == "max"
    assert x_mod.value.modifications[0].value == 30.0

    flat_tree = tree.flatten_instance(instance)

    assert flat_tree.symbols["x"].max == 30.0


def test_modification_typo():
    with open(os.path.join(MODEL_DIR, "ModificationTypo.mo"), "r") as f:
        txt = f.read()

    for c in ["Wrong1", "Wrong2"]:
        with pytest.raises(tree.ModelicaSemanticError):
            ast_tree = parser.parse(txt)
            flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name=c))

    for c in ["Good1", "Good2"]:
        ast_tree = parser.parse(txt)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name=c))  # noqa: F841


def test_tree_lookup():
    instance = parse_and_instantiate_model("TreeLookup.mo", "Level1.Level2.Level3.Test")

    flat_tree = tree.flatten_instance(instance)

    expect = sorted(["elem.tc.i", "elem.tc.a", "b"])
    assert expect == sorted(flat_tree.symbols.keys())


def test_nested_symbol_modification():
    instance = parse_and_instantiate_model("NestedSymbolModification.mo", "E")

    E_c_x = instance.extends[0].symbols["c"].type.extends[0].symbols["x"]
    mod = E_c_x.type.symbols["Real"].modification_environment.arguments[0].value
    assert mod.component.name == "nominal"
    assert mod.modifications[0].value == 2

    flat_tree = tree.flatten_instance(instance)

    assert flat_tree.symbols["c.x"].nominal == 2.0


def test_nonreplaceable_component_contains_replaceable():
    """Test that nonreplaceable components can contain replaceable components"""

    parse_and_instantiate_model("NonReplaceableContainsReplaceable.mo", "P.Test")


def test_extends_transitively_nonreplaceable_error():
    """Test that extends should fail with a replaceable in the hierarchy"""

    with pytest.raises(
        tree.ModelicaSemanticError,
        match="In P.Outer extends InnerModel, InnerModel and parents cannot be replaceable",
    ):
        parse_and_instantiate_model("NonReplaceableContainsReplaceable.mo", "P.TestFail")


def test_lookup_needs_instantiation():
    """Test that demonstrates the need for instantiation in name lookup

    Example taken from Modelica MCP 0019 discussion leading to
    Modelica Spec 3.4 Chapter 5 changes.
    """

    # Instantiation should fail without instantiation in composite name lookup
    instance = parse_and_instantiate_model("InstantiationInLookup.mo", "MyModel")
    check_redeclare_expects(instance, [redeclare_expect("x", "Real", 2.0, False)])


def test_instantiation_lookup_scope():
    """Test lookup during instantiation

    Example taken from ModelicaSpecification issues discussion on
    Modelica Spec 3.4 Chapter 5 changes.
    """

    instance = parse_and_instantiate_model("InstantiationLookupScope.mo", "M")
    x_type = instance.symbols["m"].type.symbols["x"].type
    assert x_type.name == "R"
    # Check that m.x type R is from M.B.R, not M.R
    assert len(x_type.extends), "Incorrect R scope"
    x_type_extends = x_type.extends[0]
    assert "Real" in x_type_extends.symbols
    # TODO: Fix missing dimensions for extends like in this model: `type R = Real[3]`
    # The dimensions are currently lost from the `ExtendsClause.component`.
    # See. spec 3.5 section 10.1:
    # "[Example: "The number of dimensions and the dimensions sizes are part of the
    # type, and shall be checked for example at redeclarations."
    # This should be a separate test case for the fix instead of here.
    # Section 10.1 has some test examples for this.
    # self.assertEqual(x_type_extends.symbols["Real"].dimensions[0], 3)


def test_instantiation_lookup_scope_with_modifications():
    """Test lookup during instantiation with modifications

    Example taken from ModelicaSpecification issues discussion on
    Modelica Spec 3.4 Chapter 5 changes.
    """

    instance = parse_and_instantiate_model("LexicalVsInstanceScope.mo", "P.C")
    check_redeclare_expects(instance, [redeclare_expect("n", "Integer", 3, False)])


def test_value_modification_ordering():
    """Test that value-only modifications at 3+ nesting levels pick the outermost value.

    ValF defines Real x = 1.0. ValD extends ValF(x=2.0). ValA extends ValC(ValD(x=3.0)).
    ValM extends ValA.ValD(x=4.0). The outermost value (4.0) should win.

    This is the same root cause as the redeclare ordering bug in
    test_redeclare_component_complicated but for non-redeclare value modifications.
    """

    instance = parse_and_instantiate_model(
        "NestedExtendsModification.mo", "NestedExtendsModification.ValM"
    )
    from pymoca.tree import LookupOptions, RecursionGuard

    x = _find_name(instance, "x", RecursionGuard(), LookupOptions(check_encapsulated=False))
    assert x is not None, "x not found in ValM instance"
    value_args = get_modifiers_by_name(x, "value")
    assert len(value_args) > 0, "x missing value modification"
    # "Last wins" — the last value modification should be 4.0 (outermost)
    value_mod = value_args[-1].value.modifications[0]
    assert isinstance(value_mod, ast.Primary)
    assert value_mod.value == 4.0, "Outermost value modification (4.0) should win"


def test_derived_type_value_modification():
    """Test modifying the value of a derived type"""
    txt = """
        package A
            type X = Integer;
            type Y = X; /* Derived type */
            model B
                Y y = 1; /* Modification 1 */
            end B;
            model C
                B c(y = 2); /* Modification 2 */
            end C;
            model D = C(c.y = 3); /* Modification 3 */
            model E
                D d(c.y = 4); /* Modification 4 */
            end E;
        end A;
    """
    ast_tree = parser.parse(txt)
    instance = tree.instantiate(ast_tree, "A.D")
    assert instance is not None
    c_y = (
        instance.extends[0]
        .symbols["c"]
        .type.symbols["y"]
        .type.extends[0]
        .extends[0]
        .symbols["Integer"]
    )
    c_y_mod = c_y.modification_environment.arguments[-1]
    assert c_y_mod.value.component.name == "value"
    assert c_y_mod.value.modifications[0].value == 3

    instance = tree.instantiate(ast_tree, "A.E")
    d_c_y = (
        instance.symbols["d"]
        .type.extends[0]
        .symbols["c"]
        .type.symbols["y"]
        .type.extends[0]
        .extends[0]
        .symbols["Integer"]
    )
    d_c_y_mod = d_c_y.modification_environment.arguments[-1]
    assert d_c_y_mod.value.component.name == "value"
    assert d_c_y_mod.value.modifications[0].value == 4

    flat_tree = tree.flatten_instance(instance)
    # Value modification on a plain variable becomes an equation (MLS 5.6.2 step 1.4)
    assert "d.c.y" in flat_tree.symbols
    assert isinstance(flat_tree.symbols["d.c.y"].value, ast.Primary)
    assert flat_tree.symbols["d.c.y"].value.value is None  # cleared to sentinel
    eq_map = {
        eq.left.name: eq.right
        for eq in flat_tree.equations
        if isinstance(eq, ast.Equation) and isinstance(eq.left, ast.ComponentRef)
    }
    assert "d.c.y" in eq_map, f"No value equation for d.c.y; equations: {flat_tree.equations}"
    assert isinstance(eq_map["d.c.y"], ast.Primary) and eq_map["d.c.y"].value == 4


def test_flatten_first_and_find_rest_quoted_dotted_symbol():
    """Looking up a quoted name with a dot in a temporarily flattened class keeps it whole."""
    from pymoca.tree import LookupOptions, RecursionGuard

    ast_tree = parser.parse("model A constant Real 'k.v' = 2; end A;")
    found = _flatten_first_and_find_rest(
        ast_tree.classes["A"],
        "'k.v'",
        RecursionGuard(),
        LookupOptions(check_encapsulated=False, instantiate_in_place=True),
    )
    assert found is not None
    assert found.name == "'k.v'"


if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__])
