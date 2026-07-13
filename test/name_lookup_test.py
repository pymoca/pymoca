#!/usr/bin/env python
"""
Test against Modelica name lookup rules
"""

import os
import pickle

import pymoca.ast
import pymoca.parser
from pymoca.tree import NameLookupError, find_name, flatten, flatten_instance, instantiate

import pytest  # type: ignore[import-untyped]

MY_DIR = os.path.dirname(os.path.realpath(__file__))
COMPLIANCE_DIR = os.path.join(MY_DIR, "libraries", "Modelica-Compliance", "ModelicaCompliance")
NAME_LOOKUP_DIR = os.path.join(COMPLIANCE_DIR, "Scoping", "NameLookup")
SIMPLE_LOOKUP_DIR = os.path.join(NAME_LOOKUP_DIR, "Simple")
COMPOSITE_LOOKUP_DIR = os.path.join(NAME_LOOKUP_DIR, "Composite")
GLOBAL_LOOKUP_DIR = os.path.join(NAME_LOOKUP_DIR, "Global")
IMPORTED_LOOKUP_DIR = os.path.join(NAME_LOOKUP_DIR, "Imports")


def parse_file(pathname):
    "Parse given full path name and return parsed ast.Tree"
    with open(pathname, "r", encoding="utf-8") as mo_file:
        txt = mo_file.read()
    return pymoca.parser.parse(txt)


def parse_lookup_file(pathname, relative_to_dir):
    "Parse given path relative to relative_to_dir and return parsed ast.Tree"
    arg_ast = parse_file(os.path.join(relative_to_dir, pathname))
    icon_ast = parse_file(os.path.join(COMPLIANCE_DIR, "Icons.mo"))
    if None in (arg_ast, icon_ast):
        return None
    icon_ast.extend(arg_ast)
    return icon_ast


def lookup_composite_using_simple_only(composite_name, start_scope):
    """Lookup given composite name using only simple name lookup"""
    simple_names = composite_name.split(".")
    scope = start_scope
    while True:
        if scope is None:
            return None
        name = simple_names.pop(0)
        found = find_name(scope, name)
        if len(simple_names) == 0 or found is None:
            return found
        if isinstance(found, pymoca.ast.Symbol):
            scope = lookup_composite_using_simple_only(found.type.name, found.parent)
        else:
            scope = found


def parse_simple_lookup_file(pathname):
    "Parse given path relative to SIMPLE_LOOKUP_DIR and return parsed ast.Tree"
    return parse_lookup_file(pathname, SIMPLE_LOOKUP_DIR)


def parse_composite_lookup_file(pathname):
    "Parse given path relative to COMPOSITE_LOOKUP_DIR and return parsed ast.Tree"
    return parse_lookup_file(pathname, COMPOSITE_LOOKUP_DIR)


def parse_global_lookup_file(pathname):
    "Parse given path relative to GLOBAL_LOOKUP_DIR and return parsed ast.Tree"
    return parse_lookup_file(pathname, GLOBAL_LOOKUP_DIR)


def parse_imported_lookup_file(pathname):
    "Parse given path relative to GLOBAL_LOOKUP_DIR and return parsed ast.Tree"
    return parse_lookup_file(pathname, IMPORTED_LOOKUP_DIR)


def flatten_compliance_model(ast_tree, full_model_name):
    """Instantiate and flatten a compliance model by full dotted name."""
    instance = instantiate(ast_tree, full_model_name)
    return flatten_instance(instance)


def get_flat_symbol_value(flat, sym_name):
    """Return the effective numeric value of a flat symbol.

    Constants/parameters keep their value on the symbol; plain variables have
    their value moved to an equation by _generate_value_equations (MLS 5.6.2
    step 1.4).  This helper checks both places.
    """
    sym = flat.symbols[sym_name]
    v = sym.value
    if isinstance(v, (int, float)):
        return v
    # Check equations for  ComponentRef(sym_name) = Primary(literal)
    for eq in flat.equations:
        if (
            isinstance(eq, pymoca.ast.Equation)
            and isinstance(eq.left, pymoca.ast.ComponentRef)
            and eq.left.name == sym_name
            and isinstance(eq.right, pymoca.ast.Primary)
        ):
            return eq.right.value
    return None


# Simple name lookup tests from ModelicaCompliance


def test_encapsulation():
    """Tests that names can be found or not if the scope is encapsulated"""
    ast = parse_simple_lookup_file("Encapsulation.mo")
    scope = lookup_composite_using_simple_only(
        "Scoping.NameLookup.Simple.Encapsulation.A", ast.classes["ModelicaCompliance"]
    )
    found = find_name(scope, "x")
    assert found is not None
    assert isinstance(found, pymoca.ast.Symbol)

    flat = flatten_compliance_model(
        ast, "ModelicaCompliance.Scoping.NameLookup.Simple.Encapsulation"
    )
    assert "a.x" in flat.symbols

    # Now go in the reverse direction, bumping into encapsulation
    found = find_name(found.parent, "Encapsulation")
    assert found is None

    # Check that builtin abs function is looked up correctly in encapsulated scope
    abs_scope = lookup_composite_using_simple_only(
        "Scoping.NameLookup.Simple.Encapsulation.A", ast.classes["ModelicaCompliance"]
    )
    assert abs_scope is not None
    found = find_name(abs_scope, "abs")
    # TODO: Uncomment after implementing abs built-in function lookup
    # assert found is not None


def test_enclosing_class_lookup_class():
    """Tests that classes can be looked up in an enclosing scope"""
    ast = parse_simple_lookup_file("EnclosingClassLookupClass.mo")
    scope = lookup_composite_using_simple_only(
        "Scoping.NameLookup.Simple.EnclosingClassLookupClass",
        ast.classes["ModelicaCompliance"],
    )
    found = lookup_composite_using_simple_only("b.a.x", scope)
    assert found is not None
    assert isinstance(found, pymoca.ast.Symbol)
    # Now go reverse direction, looking for a compound name but not fully qualified
    found = lookup_composite_using_simple_only("Scoping.NameLookup", found.parent)
    assert found is not None
    assert isinstance(found, pymoca.ast.Class)
    flat = flatten_compliance_model(
        ast, "ModelicaCompliance.Scoping.NameLookup.Simple.EnclosingClassLookupClass"
    )
    assert get_flat_symbol_value(flat, "b.a.x") == 2


def test_enclosing_class_lookup_constant():
    """Tests that constants can be looked up in an enclosing scope"""
    ast = parse_simple_lookup_file("EnclosingClassLookupConstant.mo")
    scope = lookup_composite_using_simple_only(
        "Scoping.NameLookup.Simple.EnclosingClassLookupConstant",
        ast.classes["ModelicaCompliance"],
    )
    assert scope is not None
    assert isinstance(scope, pymoca.ast.Class)
    scope = find_name(scope, "A")
    assert scope is not None
    assert isinstance(scope, pymoca.ast.Class)
    found = find_name(scope, "x")
    assert found is not None
    assert isinstance(found, pymoca.ast.Symbol)
    flat = flatten_compliance_model(
        ast, "ModelicaCompliance.Scoping.NameLookup.Simple.EnclosingClassLookupConstant"
    )
    assert get_flat_symbol_value(flat, "x") == 4


def test_enclosing_class_lookup_nonconstant():
    """Tests that variables found in an enclosing scope must be declared constant"""
    ast = parse_simple_lookup_file("EnclosingClassLookupNonConstant.mo")
    scope = lookup_composite_using_simple_only(
        "Scoping.NameLookup.Simple.EnclosingClassLookupNonConstant",
        ast.classes["ModelicaCompliance"],
    )
    scope = find_name(scope, "A")
    assert scope is not None
    assert isinstance(scope, pymoca.ast.Class)
    with pytest.raises(NameLookupError, match=r"Non-constant Symbol found in enclosing class"):
        find_name(scope, "x")


def test_enclosing_class_lookup_shadowed_constant():
    """Tests that variables found in an enclosing scope must be declared constant"""
    ast = parse_simple_lookup_file("EnclosingClassLookupShadowedConstant.mo")
    scope = lookup_composite_using_simple_only(
        "Scoping.NameLookup.Simple.EnclosingClassLookupShadowedConstant",
        ast.classes["ModelicaCompliance"],
    )
    assert scope is not None
    assert isinstance(scope, pymoca.ast.Class)
    scope = find_name(scope, "A")
    assert scope is not None
    assert isinstance(scope, pymoca.ast.Class)
    scope = find_name(scope, "B")
    assert scope is not None
    assert isinstance(scope, pymoca.ast.Class)
    with pytest.raises(NameLookupError, match=r"Non-constant Symbol found in enclosing class"):
        find_name(scope, "x")


def test_local_class_name_lookup():
    """Tests that variables found in an enclosing scope must be declared constant"""
    ast = parse_simple_lookup_file("LocalClassNameLookup.mo")
    scope = lookup_composite_using_simple_only(
        "Scoping.NameLookup.Simple.LocalClassNameLookup",
        ast.classes["ModelicaCompliance"],
    )
    assert scope is not None
    assert isinstance(scope, pymoca.ast.Class)
    found = find_name(scope, "A")
    assert found is not None
    assert isinstance(found, pymoca.ast.Class)


def test_local_comp_name_lookup():
    """Tests that a component name in the local scope can be found"""
    ast = parse_simple_lookup_file("LocalCompNameLookup.mo")
    scope = lookup_composite_using_simple_only(
        "Scoping.NameLookup.Simple.LocalCompNameLookup",
        ast.classes["ModelicaCompliance"],
    )
    assert scope is not None
    assert isinstance(scope, pymoca.ast.Class)
    found = find_name(scope, "x")
    assert found is not None
    assert isinstance(found, pymoca.ast.Symbol)
    found = find_name(scope, "y")
    assert found is not None
    assert isinstance(found, pymoca.ast.Symbol)


def test_outside_encapsulation():
    """Tests that elements defined outside an encapsulated scope
    can't be found in the encapsulated scope"""
    ast = parse_simple_lookup_file("OutsideEncapsulation.mo")
    scope = lookup_composite_using_simple_only(
        "Scoping.NameLookup.Simple.OutsideEncapsulation.A",
        ast.classes["ModelicaCompliance"],
    )
    assert scope is not None
    assert isinstance(scope, pymoca.ast.Class)
    found = find_name(scope, "x")
    assert found is None


def test_outside_encapsulation_multi():
    """Tests that elements defined outside an encapsulated scope
    can't be found in the encapsulated scope"""
    ast = parse_simple_lookup_file("OutsideEncapsulationMulti.mo")
    scope = lookup_composite_using_simple_only(
        "Scoping.NameLookup.Simple.OutsideEncapsulationMulti",
        ast.classes["ModelicaCompliance"],
    )
    assert scope is not None
    assert isinstance(scope, pymoca.ast.Class)
    scope = find_name(scope, "A")
    assert scope is not None
    assert isinstance(scope, pymoca.ast.Class)
    scope = find_name(scope, "B")
    assert scope is not None
    assert isinstance(scope, pymoca.ast.Class)
    found = find_name(scope, "x")
    assert found is None


# Composite name lookup tests from ModelicaCompliance or us


def test_package_lookup_class():
    """Checks that it's possible to look up a class in a package"""
    ast = parse_composite_lookup_file("PackageLookupClass.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Composite.PackageLookupClass",
    )
    found = find_name(scope, "a.x")
    assert found is not None
    assert isinstance(found, pymoca.ast.Symbol)
    flat = flatten_compliance_model(
        ast, "ModelicaCompliance.Scoping.NameLookup.Composite.PackageLookupClass"
    )
    assert get_flat_symbol_value(flat, "a.x") == 531.0


def test_package_lookup_constant():
    """Checks that it's possible to look up a constant in a package"""
    ast = parse_composite_lookup_file("PackageLookupConstant.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Composite.PackageLookupConstant",
    )
    found = find_name(scope, "P.x")
    assert found is not None
    assert isinstance(found, pymoca.ast.Symbol)
    flat = flatten_compliance_model(
        ast, "ModelicaCompliance.Scoping.NameLookup.Composite.PackageLookupConstant"
    )
    # y = P.x folds the constant reference to its value in the value equation
    assert get_flat_symbol_value(flat, "y") == 5.1


def test_nested_comp_lookup():
    """Checks that composite names where each identifier is a component can be looked up"""
    ast = parse_composite_lookup_file("NestedCompLookup.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Composite.NestedCompLookup",
    )
    found = find_name(scope, "c.b.a.x")
    assert found is not None
    assert isinstance(found, pymoca.ast.Symbol)
    flat = flatten_compliance_model(
        ast, "ModelicaCompliance.Scoping.NameLookup.Composite.NestedCompLookup"
    )
    assert get_flat_symbol_value(flat, "c.b.a.x") == 17


def test_partial_class_lookup():
    """Checks that it's not allowed to look up a name in a partial class

    Above is what PartialClassLookup.mo says, but according to the spec,
    it's only forbidden in a simulation model, so the check for partial
    is left to the caller and find_name returns the found class."""
    ast = parse_composite_lookup_file("PartialClassLookup.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Composite.PartialClassLookup",
    )
    found = find_name(scope, "P.x")
    assert found is not None
    assert found.parent.partial


def test_non_function_lookup_via_comp():
    """Checks that it's not allowed to look up a non-function class via a component."""
    ast = parse_composite_lookup_file("NonFunctionLookupViaComp.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Composite.NonFunctionLookupViaComp",
    )
    found = find_name(scope, "a.B")
    assert found is None


def test_non_package_lookup_comp():
    """Checks that looking up an non-encapsulated element, in this
    case a component, inside a class which does not satisfy the requirements for
    a package is forbidden"""
    ast = parse_composite_lookup_file("NonPackageLookupComp.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Composite.NonPackageLookupComp",
    )
    with pytest.raises(
        pymoca.tree.NameLookupError, match=r"A is not a package so x must be encapsulated"
    ):
        _ = find_name(scope, "A.x")


def test_non_package_lookup_encapsulated():
    """Checks that looking up an encapsulated element inside a class
    which does not satisfy the requirements for a package is allowed."""
    ast = parse_composite_lookup_file("NonPackageLookupEncapsulated.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Composite.NonPackageLookupEncapsulated",
    )
    found = find_name(scope, "A.B")
    assert found is not None


def test_non_package_lookup_non_encapsulated():
    """Checks that looking up an non-encapsulated element inside a class
    which does not satisfy the requirements for a package is forbidden"""
    ast = parse_composite_lookup_file("NonPackageLookupNonEncapsulated.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Composite.NonPackageLookupNonEncapsulated",
    )
    with pytest.raises(
        pymoca.tree.NameLookupError, match=r"A is not a package so B must be encapsulated"
    ):
        _ = find_name(scope, "A.B")


def test_function_lookup_via_comp():
    """Checks that it's allowed to look up a function via a component"""
    ast = parse_composite_lookup_file("FunctionLookupViaComp.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Composite.FunctionLookupViaComp",
    )
    found = find_name(scope, "a.f")
    assert found is not None


@pytest.mark.skip("Needs function-call-context tracking in name lookup")
def test_function_lookup_via_comp_non_call():
    """Checks that it's only legal to look up a function name via a
    component if the name is used as a function call"""
    # TODO: How to check that name is used as a function call?
    ast = parse_composite_lookup_file("FunctionLookupViaComp.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Composite.FunctionLookupViaComp",
    )
    with pytest.raises(pymoca.tree.NameLookupError, match=r"ADD REGEX WHEN UNSKIPPED"):
        _ = find_name(scope, "a.f")


def test_function_lookup_via_class_comp():
    """Checks that it's allowed to look up a function via a component
    if the rest of the composite name consists of class references"""
    ast = parse_composite_lookup_file("FunctionLookupViaClassComp.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Composite.FunctionLookupViaClassComp",
    )
    found = find_name(scope, "a.B.C.f")
    assert found is not None


def test_function_lookup_via_non_class_comp():
    """Checks that looking up a function via a component is only
    allowed if the rest of the composite name consists of class references"""
    ast = parse_composite_lookup_file("FunctionLookupViaNonClassComp.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Composite.FunctionLookupViaNonClassComp",
    )
    found = find_name(scope, "a.B.c.f")
    assert found is None


@pytest.mark.skip("TODO: Do this test when operator functions are implemented")
def test_function_in_operator_lookup_via_comp():
    """Checks that it's not allowed to look up a function in an operator
    via a component"""
    ast = parse_composite_lookup_file("FunctionInOperatorLookupViaComp.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Composite.FunctionInOperatorLookupViaComp.FunctionInOperatorLookupViaComp",
    )
    with pytest.raises(pymoca.tree.NameLookupError, match=r"ADD REGEX WHEN UNSKIPPED"):
        _ = find_name(scope, "or1.'+'.add")


@pytest.mark.skip("TODO: Do this test when operator functions are implemented")
def test_operator_function_lookup_via_comp():
    """Checks that it's not allowed to look up an operator function
    via a component"""
    ast = parse_composite_lookup_file("OperatorFunctionLookupViaComp.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Composite.OperatorFunctionLookupViaComp.OperatorFunctionLookupViaComp",
    )
    with pytest.raises(pymoca.tree.NameLookupError, match=r"ADD REGEX WHEN UNSKIPPED"):
        _ = find_name(scope, "or1.'+'")


def test_function_lookup_via_array_comp():
    """Checks that it's not allowed to look up a function via an
    array component"""
    ast = parse_composite_lookup_file("FunctionLookupViaArrayComp.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Composite.FunctionLookupViaArrayComp",
    )
    with pytest.raises(
        pymoca.tree.NameLookupError, match=r"Array a must have subscripts to lookup function f"
    ):
        _ = find_name(scope, "a.f")


@pytest.mark.skip("Needs array subscript handling in find_name")
def test_function_lookup_via_array_element():
    """Checks that it's allowed to look up a function via an
    array element if the element is a scalar component"""
    ast = parse_composite_lookup_file("FunctionLookupViaArrayElement.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Composite.FunctionLookupViaArrayElement",
    )
    # with pytest.raises(pymoca.tree.NameLookupError):
    found = find_name(scope, "a[2].f")
    assert found is not None


def test_need_for_temporary_flattening():
    """Test name lookup through 2 levels of inheritance with symbol value modifications

    This is a case where name lookup through 2 levels of inheritance with
    symbol value modifications works, even if it is not "temporarily
    flattened" as mentioned in the Modelica 3.5 spec section 5.3.2.
    """
    txt = """
    class A
        // Doesn't have a class B itself, but gets one via C
        extends C(B.bla=2);
    end A;
    class C
        encapsulated class B
            constant Integer bla = 0;
        end B;
    end C;
    class M
        extends A.B(bla=1);
    end M;
    """
    ast_tree = pymoca.parser.parse(txt)
    class_name = "M"
    comp_ref = pymoca.ast.ComponentRef.from_string(class_name)
    flat_tree = flatten(ast_tree, comp_ref)
    assert flat_tree.classes[class_name].symbols["bla"].value.value == 1


# Global name lookup tests from ModelicaCompliance


def test_encapsulated_global_lookup():
    """Checks that it's possible to look up a global name, even if
    the current scope is encapsulated"""
    ast = parse_global_lookup_file("EncapsulatedGlobalLookup.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Global.EncapsulatedGlobalLookup",
    )
    found = find_name(scope, "a.y")
    assert found is not None
    assert isinstance(found, pymoca.ast.Symbol)


def test_global_name_shadowed_by_local():
    """Checks that global name lookup starts from the global scope even when a
    local class shadows the first identifier (MLS 5.3.3)"""
    ast = pymoca.parser.parse(
        """
        package Top
          constant Real c = 111;
        end Top;
        package Q
          package Top
            constant Real c = 222;
          end Top;
          model M
            Real x = .Top.c;
            Real y = Top.c;
          end M;
        end Q;
        """
    )
    flat = flatten_compliance_model(ast, "Q.M")
    assert get_flat_symbol_value(flat, "x") == 111
    assert get_flat_symbol_value(flat, "y") == 222


def test_global_type_name():
    """Checks that a leading-dot type specifier resolves from the global scope
    even when a local class shadows the first identifier (MLS 5.3.3)"""
    ast = pymoca.parser.parse(
        """
        package Top
          type T = Real;
        end Top;
        package Q
          package Top
            type T = Integer;
          end Top;
          model M
            .Top.T x;
            Top.T y;
          end M;
        end Q;
        """
    )
    flat = flatten_compliance_model(ast, "Q.M")
    assert flat.symbols["x"].type.name == "Real"
    assert flat.symbols["y"].type.name == "Integer"


def test_global_name_find_name():
    """Checks that find_name resolves a leading-dot name from the root scope"""
    ast = pymoca.parser.parse(
        """
        package Top
          constant Real c = 111;
        end Top;
        package Q
          package Top
            constant Real c = 222;
          end Top;
        end Q;
        """
    )
    found = find_name(ast.classes["Q"], ".Top.c")
    assert isinstance(found, pymoca.ast.Symbol)
    assert found.parent is ast.classes["Top"]


# Imported name lookup tests from ModelicaCompliance


def test_encapsulated():
    """Checks that it's possible to import from inside an
    encapsulated model"""
    ast = parse_imported_lookup_file("EncapsulatedImport.mo")
    found = find_name(
        ast,
        "ModelicaCompliance.Scoping.NameLookup.Imports.EncapsulatedImport.a.m.x",
    )
    assert found is not None
    assert isinstance(found, pymoca.ast.Symbol)
    flat = flatten_compliance_model(
        ast, "ModelicaCompliance.Scoping.NameLookup.Imports.EncapsulatedImport"
    )
    assert get_flat_symbol_value(flat, "a.m.x") == 2.0


def test_extend_import():
    """Checks that imports are not inherited"""
    ast = parse_imported_lookup_file("ExtendImport.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Imports.ExtendImport",
    )
    # We can't access C.A directly because C is not a package or encapsulated
    # Make C a package to work around the check when accessing A
    found = find_name(scope, "C")
    found.type = "package"
    found = find_name(found, "A")
    assert found is None


def test_local_scope():
    """Checks that the lookup of an imported name is not started
    in the local scope"""
    ast = parse_imported_lookup_file("ImportLookupLocalScope.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Imports.ImportLookupLocalScope",
    )
    found = find_name(scope, "B")
    assert found is None


def test_scope_type():
    """Checks that it's allowed to import into any kind of class"""
    ast = parse_imported_lookup_file("ImportScopeType.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Imports.ImportScopeType",
    )
    found = find_name(scope, "a")
    assert found is not None
    found = find_name(scope, "b")
    assert found is not None
    found = find_name(scope, "m.y")
    assert found is not None

    flat = flatten_compliance_model(
        ast, "ModelicaCompliance.Scoping.NameLookup.Imports.ImportScopeType"
    )
    assert "a" in flat.symbols  # a = P2.y; value not literal-traceable (ref chain)
    assert get_flat_symbol_value(flat, "m.y") == 2.0


@pytest.mark.skip("Needs import modification validation")
def test_modify_import():
    """Checks that it's not allowed to modify an import"""
    ast = parse_imported_lookup_file("ModifyImport.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Imports.ModifyImport",
    )
    found = find_name(scope, "b")
    assert found is None


def test_qualified_import():
    """Tests that a qualified import works"""
    ast = parse_imported_lookup_file("QualifiedImport.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Imports.QualifiedImport",
    )
    found = find_name(scope, "b.a.x")
    assert found is not None

    flat = flatten_compliance_model(
        ast, "ModelicaCompliance.Scoping.NameLookup.Imports.QualifiedImport"
    )
    assert get_flat_symbol_value(flat, "b.a.x") == 1.0


def test_qualified_import_conflict():
    """Checks that it's not allowed to have multiple qualified
    import-clauses with the same import name"""
    # This is caught at the parse stage, not in name lookup
    with pytest.raises(pymoca.parser.ModelicaSyntaxError):
        _ = parse_imported_lookup_file("QualifiedImportConflict.mo")


def test_qualified_import_non_package():
    """Checks that it's not allowed to import a definition which is
    not a package or package element via a qualified import"""
    ast = parse_imported_lookup_file("QualifiedImportNonPackage.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Imports.QualifiedImportNonPackage",
    )
    with pytest.raises(
        pymoca.tree.NameLookupError,
        match=r"QualifiedImportNonPackage must be a package in import ModelicaCompliance.Scoping.NameLookup.Imports.QualifiedImportNonPackage.A in scope ModelicaCompliance.Scoping.NameLookup.Imports.QualifiedImportNonPackage",
    ):
        _ = find_name(scope, "A2")


def test_qualified_import_protected():
    """Checks that it's an error to import a protected element"""
    ast = parse_imported_lookup_file("QualifiedImportProtected.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Imports.QualifiedImportProtected",
    )
    with pytest.raises(
        pymoca.tree.NameLookupError,
        match=r"Import y must not be protected in scope ModelicaCompliance.Scoping.NameLookup.Imports.QualifiedImportProtected.A",
    ):
        _ = find_name(scope, "A.y")


def test_recursive():
    """Tests that a named recursive import does not work"""
    ast = parse_imported_lookup_file("Recursive.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Imports.Recursive",
    )
    with pytest.raises(
        pymoca.tree.NameLookupError,
        match=r"Import ModelicaCompliance.Scoping.NameLookup.Imports.Recursive in scope ModelicaCompliance.Scoping.NameLookup.Imports.Recursive is recursive",
    ):
        _ = find_name(scope, "A")


@pytest.mark.skip("Needs import redeclaration validation")
def test_redeclare_import():
    """Checks that it's not allowed to redeclare an import"""
    ast = parse_imported_lookup_file("RedeclareImport.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Imports.RedeclareImport",
    )
    found = find_name(scope, "b")
    assert found is None


def test_renaming_import():
    """Tests that a renaming import works"""
    ast = parse_imported_lookup_file("RenamingImport.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Imports.RenamingImport",
    )
    found = find_name(scope, "b.a.x")
    assert found is not None

    flat = flatten_compliance_model(
        ast, "ModelicaCompliance.Scoping.NameLookup.Imports.RenamingImport"
    )
    assert get_flat_symbol_value(flat, "b.a.x") == 1.0


def test_renaming_import_non_package():
    """Checks that it's not allowed to import a definition which is
    not a package or package element via a renaming import"""
    ast = parse_imported_lookup_file("RenamingImportNonPackage.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Imports.RenamingImportNonPackage",
    )
    with pytest.raises(
        pymoca.tree.NameLookupError,
        match=r"RenamingImportNonPackage must be a package in import ModelicaCompliance.Scoping.NameLookup.Imports.RenamingImportNonPackage.A in scope ModelicaCompliance.Scoping.NameLookup.Imports.RenamingImportNonPackage",
    ):
        _ = find_name(scope, "A2")


def test_renaming_single_definition_import():
    """Tests that a renaming import works"""
    ast = parse_imported_lookup_file("RenamingSingleDefinitionImport.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Imports.RenamingSingleDefinitionImport",
    )
    found = find_name(scope, "b.a.x")
    assert found is not None

    flat = flatten_compliance_model(
        ast, "ModelicaCompliance.Scoping.NameLookup.Imports.RenamingSingleDefinitionImport"
    )
    assert get_flat_symbol_value(flat, "b.a.x") == 1.0


def test_single_definition_import():
    """Tests that a single definition import works"""
    ast = parse_imported_lookup_file("SingleDefinitionImport.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Imports.SingleDefinitionImport",
    )
    found = find_name(scope, "b.a.x")
    assert found is not None

    flat = flatten_compliance_model(
        ast, "ModelicaCompliance.Scoping.NameLookup.Imports.SingleDefinitionImport"
    )
    assert get_flat_symbol_value(flat, "b.a.x") == 1.0


def test_unqualified_import():
    """Tests that an unqualified import works"""
    ast = parse_imported_lookup_file("UnqualifiedImport.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Imports.UnqualifiedImport",
    )
    found = find_name(scope, "b.a.x")
    assert found is not None

    flat = flatten_compliance_model(
        ast, "ModelicaCompliance.Scoping.NameLookup.Imports.UnqualifiedImport"
    )
    assert get_flat_symbol_value(flat, "b.a.x") == 1.0


def test_unqualified_import_does_not_mutate_parsed_ast():
    """Verify star-import lookup must not mutate the shared parsed AST.

    The cache write only fires when the imported package is reached via the scope.root
    fallback encountered when importing across two unrelated top-level packages.
    """
    txt = """
    package Lib
      package P
        package Q
          model A
            Real x = 1.0;
          end A;
        end Q;
      end P;
    end Lib;

    package Client
      model B
        import Lib.P.Q.*;
        A a;
      end B;
    end Client;
    """
    ast_tree = pymoca.parser.parse(txt)
    pickled_before = pickle.dumps(ast_tree)
    found = lookup_composite_using_simple_only("Client.B.A", ast_tree)
    assert found is not None
    assert found.full_name == "Lib.P.Q.A"
    assert (
        pickle.dumps(ast_tree) == pickled_before
    ), "star-import lookup mutated the shared parsed AST"


@pytest.mark.skip("Unqualified import name lookup conflicts not implemented")
def test_unqualified_import_conflict():
    """Checks that it's an error if the same name is found in
    multiple unqualified imports"""
    # This will slow down already slow unqualified name lookup. Unqualified
    # imports are rare and this test case is even more rare, so it seems like maybe
    # not a good idea at this stage for Pymoca.
    ast = parse_imported_lookup_file("UnqualifiedImportConflict.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Imports.UnqualifiedImportConflict",
    )
    with pytest.raises(pymoca.tree.NameLookupError, match=r"ADD REGEX WHEN UNSKIPPED"):
        _ = find_name(scope, "A.x")


def test_unqualified_import_non_conflict():
    """Checks that it's not an error to be able to find a name in
    multiple unqualified imports, it's only an error if such a name is
    used during name lookup. I.e. both P and P2 contains x in this test, but
    that's ok since x is not used by the importer A."""
    ast = parse_imported_lookup_file("UnqualifiedImportNonConflict.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Imports.UnqualifiedImportNonConflict",
    )
    found = find_name(scope, "a")
    assert found is not None


def test_unqualified_import_non_package():
    """Checks that an unqualified import is not allowed to import
    from a non-package"""
    ast = parse_imported_lookup_file("UnqualifiedImportNonPackage.mo")
    with pytest.raises(
        pymoca.tree.NameLookupError,
        match=r"A must be a package in import ModelicaCompliance.Scoping.NameLookup.Imports.UnqualifiedImportNonPackage.A.B in scope ModelicaCompliance.Scoping.NameLookup.Imports.UnqualifiedImportNonPackage",
    ):
        _ = find_name(
            ast,
            "ModelicaCompliance.Scoping.NameLookup.Imports.UnqualifiedImportNonPackage.B",
        )


def test_import_package_name_prefix_overlap():
    """Test import lookup when package names share a character prefix"""
    txt = """
    package Outer
        package ABCPackage
            model M
                import Outer.ABC; // Shares same prefix as ABCPackage
            end M;
        end ABCPackage;
        package ABC
            constant Integer x = 42;
        end ABC;
    end Outer;
    """
    ast_tree = pymoca.parser.parse(txt)
    scope = find_name(ast_tree, "Outer.ABCPackage.M")
    assert scope is not None
    # Resolving ABC via the import must not infinite-loop or crash
    found = find_name(scope, "ABC")
    assert found is not None
    assert isinstance(found, pymoca.ast.Class)


def test_unqualified_import_protected():
    """Checks that the name lookup only considers public members of
    packages imported via unqualified imports"""
    ast = parse_imported_lookup_file("UnqualifiedImportProtected.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Imports.UnqualifiedImportProtected",
    )
    with pytest.raises(
        pymoca.tree.NameLookupError,
        match=r"Import y must not be protected in scope ModelicaCompliance.Scoping.NameLookup.Imports.UnqualifiedImportProtected.A",
    ):
        _ = find_name(scope, "A.y")


# test_qualified_import_priority and test_unqualified_import_priority use models
# from the NameLookup.Simple package, but need imports and composite name lookup so
# we put them here.
def test_qualified_import_priority():
    """Tests that qualified imports have lower priority than local
    and inherited names during name lookup"""
    ast = parse_simple_lookup_file("QualifiedImportPriority.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Simple.QualifiedImportPriority",
    )
    assert scope is not None
    assert isinstance(scope, pymoca.ast.Class)
    expect = (("d.x", 2.0), ("b.x", 3.0), ("c.x", 1.0))
    for name, _value in expect:
        found = find_name(scope, name)
        assert found is not None, f" for {name}"

    flat = flatten_compliance_model(
        ast, "ModelicaCompliance.Scoping.NameLookup.Simple.QualifiedImportPriority"
    )
    assert get_flat_symbol_value(flat, "d.y") == 2.0
    assert get_flat_symbol_value(flat, "b.x") == 3.0
    assert get_flat_symbol_value(flat, "c.x") == 1.0


def test_unqualified_import_priority():
    """Tests that unqualified imports have lowest priority"""
    ast = parse_simple_lookup_file("UnqualifiedImportPriority.mo")
    scope = find_name(
        ast.classes["ModelicaCompliance"],
        "Scoping.NameLookup.Simple.UnqualifiedImportPriority",
    )
    assert scope is not None
    assert isinstance(scope, pymoca.ast.Class)
    expect = (("e.x", 2.0), ("b.x", 3.0), ("c.x", 1.0), ("d.x", 4.0))
    for name, _value in expect:
        found = find_name(scope, name)
        assert found is not None, f" for {name}"

    flat = flatten_compliance_model(
        ast, "ModelicaCompliance.Scoping.NameLookup.Simple.UnqualifiedImportPriority"
    )
    assert get_flat_symbol_value(flat, "e.y") == 2.0
    assert get_flat_symbol_value(flat, "b.x") == 3.0
    assert get_flat_symbol_value(flat, "c.x") == 1.0
    assert get_flat_symbol_value(flat, "d.y") == 4.0


if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__])
