#!/usr/bin/env python
"""
Test against Modelica name lookup rules
"""

import os
import unittest

import pymoca.ast
import pymoca.parser
from pymoca.tree import NameLookupError, find_name, flatten

MY_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(MY_DIR, "models")
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


def parse_model_files(*pathnames):
    "Parse given files from MODEL_DIR and return parsed ast.Tree"
    tree = None
    for path in pathnames:
        file_tree = parse_file(os.path.join(MODEL_DIR, path))
        if tree:
            tree.extend(file_tree)
        else:
            tree = file_tree
    return tree


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
        found = find_name(name, scope)
        if len(simple_names) == 0 or found is None:
            return found
        if isinstance(found, pymoca.ast.Symbol):
            scope = find_name(found.type.name, found.parent)
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


class SimpleNameLookupTest(unittest.TestCase):
    """Simple name lookup tests from ModelicaCompliance"""

    # TODO: Update when new name lookup is connected to flattening (see todos hints below)

    def test_encapsulation(self):
        """Tests that names can be found or not if the scope is encapsulated"""
        ast = parse_simple_lookup_file("Encapsulation.mo")
        scope = lookup_composite_using_simple_only(
            "Scoping.NameLookup.Simple.Encapsulation.A", ast.classes["ModelicaCompliance"]
        )
        found = find_name("x", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Symbol)
        # TODO: flatten and check x.value

        # Now go in the reverse direction, bumping into encapsulation
        found = find_name("Encapsulation", found.parent)
        self.assertIsNone(found)

        # Check that builtin abs function is looked up correctly in encapsulated scope
        abs_scope = lookup_composite_using_simple_only(
            "Scoping.NameLookup.Simple.Encapsulation.A", ast.classes["ModelicaCompliance"]
        )
        self.assertIsNotNone(abs_scope)
        found = find_name("abs", abs_scope)
        # TODO: Uncomment after implementing abs built-in function lookup
        # self.assertIsNotNone(found)

    def test_enclosing_class_lookup_class(self):
        """Tests that classes can be looked up in an enclosing scope"""
        ast = parse_simple_lookup_file("EnclosingClassLookupClass.mo")
        scope = lookup_composite_using_simple_only(
            "Scoping.NameLookup.Simple.EnclosingClassLookupClass",
            ast.classes["ModelicaCompliance"],
        )
        found = lookup_composite_using_simple_only("b.a.x", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Symbol)
        # Now go reverse direction, looking for a compound name but not fully qualified
        found = lookup_composite_using_simple_only("Scoping.NameLookup", found.parent)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Class)
        # TODO: flatten and check x.value == 2

    def test_enclosing_class_lookup_constant(self):
        """Tests that constants can be looked up in an enclosing scope"""
        ast = parse_simple_lookup_file("EnclosingClassLookupConstant.mo")
        scope = lookup_composite_using_simple_only(
            "Scoping.NameLookup.Simple.EnclosingClassLookupConstant",
            ast.classes["ModelicaCompliance"],
        )
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        scope = find_name("A", scope)
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        found = find_name("x", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Symbol)
        # TODO: flatten and check y.value == 4

    def test_enclosing_class_lookup_nonconstant(self):
        """Tests that variables found in an enclosing scope must be declared constant"""
        ast = parse_simple_lookup_file("EnclosingClassLookupNonConstant.mo")
        scope = lookup_composite_using_simple_only(
            "Scoping.NameLookup.Simple.EnclosingClassLookupNonConstant",
            ast.classes["ModelicaCompliance"],
        )
        scope = find_name("A", scope)
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        with self.assertRaises(NameLookupError):
            find_name("x", scope)

    def test_enclosing_class_lookup_shadowed_constant(self):
        """Tests that variables found in an enclosing scope must be declared constant"""
        ast = parse_simple_lookup_file("EnclosingClassLookupShadowedConstant.mo")
        scope = lookup_composite_using_simple_only(
            "Scoping.NameLookup.Simple.EnclosingClassLookupShadowedConstant",
            ast.classes["ModelicaCompliance"],
        )
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        scope = find_name("A", scope)
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        scope = find_name("B", scope)
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        with self.assertRaises(NameLookupError):
            find_name("x", scope)

    def test_local_class_name_lookup(self):
        """Tests that variables found in an enclosing scope must be declared constant"""
        ast = parse_simple_lookup_file("LocalClassNameLookup.mo")
        scope = lookup_composite_using_simple_only(
            "Scoping.NameLookup.Simple.LocalClassNameLookup",
            ast.classes["ModelicaCompliance"],
        )
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        found = find_name("A", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Class)

    def test_local_comp_name_lookup(self):
        """Tests that a component name in the local scope can be found"""
        ast = parse_simple_lookup_file("LocalCompNameLookup.mo")
        scope = lookup_composite_using_simple_only(
            "Scoping.NameLookup.Simple.LocalCompNameLookup",
            ast.classes["ModelicaCompliance"],
        )
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        found = find_name("x", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Symbol)
        found = find_name("y", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Symbol)

    def test_outside_encapsulation(self):
        """Tests that elements defined outside an encapsulated scope
        can't be found in the encapsulated scope"""
        ast = parse_simple_lookup_file("OutsideEncapsulation.mo")
        scope = lookup_composite_using_simple_only(
            "Scoping.NameLookup.Simple.OutsideEncapsulation.A",
            ast.classes["ModelicaCompliance"],
        )
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        found = find_name("x", scope)
        self.assertIsNone(found)

    def test_outside_encapsulation_multi(self):
        """Tests that elements defined outside an encapsulated scope
        can't be found in the encapsulated scope"""
        ast = parse_simple_lookup_file("OutsideEncapsulationMulti.mo")
        scope = lookup_composite_using_simple_only(
            "Scoping.NameLookup.Simple.OutsideEncapsulationMulti",
            ast.classes["ModelicaCompliance"],
        )
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        scope = find_name("A", scope)
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        scope = find_name("B", scope)
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        found = find_name("x", scope)
        self.assertIsNone(found)

    def test_qualified_import_priority(self):
        """Tests that qualified imports have lower priority than local
        and inherited names during name lookup"""
        ast = parse_simple_lookup_file("QualifiedImportPriority.mo")
        scope = lookup_composite_using_simple_only(
            "Scoping.NameLookup.Simple.QualifiedImportPriority",
            ast.classes["ModelicaCompliance"],
        )
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        found = lookup_composite_using_simple_only("d.x", scope)
        self.assertIsNotNone(found)
        x_value = found.class_modification.arguments[0].value.modifications[0].value
        self.assertAlmostEqual(x_value, 2.0)
        found = lookup_composite_using_simple_only("b.x", scope)
        self.assertIsNotNone(found)
        x_value = found.class_modification.arguments[0].value.modifications[0].value
        self.assertAlmostEqual(x_value, 3.0)
        found = lookup_composite_using_simple_only("c.x", scope)
        self.assertIsNotNone(found)
        x_value = found.class_modification.arguments[0].value.modifications[0].value
        self.assertAlmostEqual(x_value, 1.0)
        # # This is how we would do the "d.x" case above in current Pymoca 0.10:
        # # Can't look up symbols directly, so look up containing class
        # cref = pymoca.ast.ComponentRef.from_string(
        #     "ModelicaCompliance.Scoping.NameLookup.Simple.QualifiedImportPriority.D"
        # )
        # d_class = ast.find_class(cref)
        # self.assertIsNotNone(found)
        # # Can't look up symbols, so need to flatten to access the imported symbol
        # d_flat_class = pymoca.tree.flatten(
        #     d_class, cref
        # )
        # # Below fails because the import is flattened into fully qualified name
        # x_value = d_flat_class.symbols["x"].value.value
        # self.assertAlmostEqual(x_value, 2.0)
        # # TODO: What happens if we try to generate a casadi model for D in Pymoca 0.10?

    def test_unqualified_import_priority(self):
        """Tests that unqualified imports have lowest priority"""
        ast = parse_simple_lookup_file("UnqualifiedImportPriority.mo")
        scope = lookup_composite_using_simple_only(
            "Scoping.NameLookup.Simple.UnqualifiedImportPriority",
            ast.classes["ModelicaCompliance"],
        )
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        found = lookup_composite_using_simple_only("e.x", scope)
        self.assertIsNotNone(found)
        x_value = found.class_modification.arguments[0].value.modifications[0].value
        self.assertAlmostEqual(x_value, 2.0)
        found = lookup_composite_using_simple_only("b.x", scope)
        self.assertIsNotNone(found)
        x_value = found.class_modification.arguments[0].value.modifications[0].value
        self.assertAlmostEqual(x_value, 3.0)
        found = lookup_composite_using_simple_only("c.x", scope)
        self.assertIsNotNone(found)
        x_value = found.class_modification.arguments[0].value.modifications[0].value
        self.assertAlmostEqual(x_value, 1.0)
        found = lookup_composite_using_simple_only("d.x", scope)
        self.assertIsNotNone(found)
        x_value = found.class_modification.arguments[0].value.modifications[0].value
        self.assertAlmostEqual(x_value, 4.0)


class CompositeNameLookupTest(unittest.TestCase):
    """Composite name lookup tests from ModelicaCompliance or us"""

    # TODO: Update when new name lookup is connected to flattening (see todos hints below)

    def test_package_lookup_class(self):
        """Checks that it's possible to look up a class in a package"""
        ast = parse_composite_lookup_file("PackageLookupClass.mo")
        scope = find_name(
            "Scoping.NameLookup.Composite.PackageLookupClass",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("a.x", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Symbol)
        # TODO: flatten and check a.x.value = 531.0

    def test_package_lookup_constant(self):
        """Checks that it's possible to look up a constant in a package"""
        ast = parse_composite_lookup_file("PackageLookupConstant.mo")
        scope = find_name(
            "Scoping.NameLookup.Composite.PackageLookupConstant",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("P.x", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Symbol)
        # TODO: flatten and check y.value = 5.1

    def test_nested_comp_lookup(self):
        """Checks that composite names where each identifier is a component can be looked up"""
        ast = parse_composite_lookup_file("NestedCompLookup.mo")
        scope = find_name(
            "Scoping.NameLookup.Composite.NestedCompLookup",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("c.b.a.x", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Symbol)
        # TODO: flatten and check y.value = 17 (integer)

    def test_partial_class_lookup(self):
        """Checks that it's not allowed to look up a name in a partial class

        Above is what PartialClassLookup.mo says, but according to the spec,
        it's only forbidden in a simulation model, so the check for partial
        is left to the caller and find_name returns the found class."""
        ast = parse_composite_lookup_file("PartialClassLookup.mo")
        scope = find_name(
            "Scoping.NameLookup.Composite.PartialClassLookup",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("P.x", scope)
        self.assertIsNotNone(found)
        self.assertTrue(found.parent.partial)

    def test_non_function_lookup_via_comp(self):
        """Checks that it's not allowed to look up a non-function class via a component."""
        ast = parse_composite_lookup_file("NonFunctionLookupViaComp.mo")
        scope = find_name(
            "Scoping.NameLookup.Composite.NonFunctionLookupViaComp",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("a.B", scope)
        self.assertIsNone(found)

    def test_non_package_lookup_comp(self):
        """Checks that looking up an non-encapsulated element, in this
        case a component, inside a class which does not satisfy the requirements for
        a package is forbidden"""
        ast = parse_composite_lookup_file("NonPackageLookupComp.mo")
        scope = find_name(
            "Scoping.NameLookup.Composite.NonPackageLookupComp",
            ast.classes["ModelicaCompliance"],
        )
        # TODO: Implement assertRaisesRegex for this and other NameLookupError cases
        with self.assertRaises(pymoca.tree.NameLookupError):
            _ = find_name("A.x", scope)

    def test_non_package_lookup_encapsulated(self):
        """Checks that looking up an encapsulated element inside a class
        which does not satisfy the requirements for a package is allowed."""
        ast = parse_composite_lookup_file("NonPackageLookupEncapsulated.mo")
        scope = find_name(
            "Scoping.NameLookup.Composite.NonPackageLookupEncapsulated",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("A.B", scope)
        self.assertIsNotNone(found)

    def test_non_package_lookup_non_encapsulated(self):
        """Checks that looking up an non-encapsulated element inside a class
        which does not satisfy the requirements for a package is forbidden"""
        ast = parse_composite_lookup_file("NonPackageLookupNonEncapsulated.mo")
        scope = find_name(
            "Scoping.NameLookup.Composite.NonPackageLookupNonEncapsulated",
            ast.classes["ModelicaCompliance"],
        )
        with self.assertRaises(pymoca.tree.NameLookupError):
            _ = find_name("A.B", scope)

    def test_function_lookup_via_comp(self):
        """Checks that it's allowed to look up a function via a component"""
        ast = parse_composite_lookup_file("FunctionLookupViaComp.mo")
        scope = find_name(
            "Scoping.NameLookup.Composite.FunctionLookupViaComp",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("a.f", scope)
        self.assertIsNotNone(found)

    @unittest.skip("TODO: Do this test when new instantiation/flattening is implemented")
    def test_function_lookup_via_comp_non_call(self):
        """Checks that it's only legal to look up a function name via a
        component if the name is used as a function call"""
        # TODO: How to check that name is used as a function call?
        ast = parse_composite_lookup_file("FunctionLookupViaComp.mo")
        scope = find_name(
            "Scoping.NameLookup.Composite.FunctionLookupViaComp",
            ast.classes["ModelicaCompliance"],
        )
        with self.assertRaises(pymoca.tree.NameLookupError):
            _ = find_name("a.f", scope)

    def test_function_lookup_via_class_comp(self):
        """Checks that it's allowed to look up a function via a component
        if the rest of the composite name consists of class references"""
        ast = parse_composite_lookup_file("FunctionLookupViaClassComp.mo")
        scope = find_name(
            "Scoping.NameLookup.Composite.FunctionLookupViaClassComp",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("a.B.C.f", scope)
        self.assertIsNotNone(found)

    def test_function_lookup_via_non_class_comp(self):
        """Checks that looking up a function via a component is only
        allowed if the rest of the composite name consists of class references"""
        ast = parse_composite_lookup_file("FunctionLookupViaNonClassComp.mo")
        scope = find_name(
            "Scoping.NameLookup.Composite.FunctionLookupViaNonClassComp",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("a.B.c.f", scope)
        self.assertIsNone(found)

    @unittest.skip("TODO: Do this test when operator functions are implemented")
    def test_function_in_operator_lookup_via_comp(self):
        """Checks that it's not allowed to look up a function in an operator
        via a component"""
        ast = parse_composite_lookup_file("FunctionInOperatorLookupViaComp.mo")
        scope = find_name(
            "Scoping.NameLookup.Composite.FunctionInOperatorLookupViaComp.FunctionInOperatorLookupViaComp",
            ast.classes["ModelicaCompliance"],
        )
        with self.assertRaises(pymoca.tree.NameLookupError):
            _ = find_name("or1.'+'.add", scope)

    @unittest.skip("TODO: Do this test when operator functions are implemented")
    def test_operator_function_lookup_via_comp(self):
        """Checks that it's not allowed to look up an operator function
        via a component"""
        ast = parse_composite_lookup_file("OperatorFunctionLookupViaComp.mo")
        scope = find_name(
            "Scoping.NameLookup.Composite.OperatorFunctionLookupViaComp.OperatorFunctionLookupViaComp",
            ast.classes["ModelicaCompliance"],
        )
        with self.assertRaises(pymoca.tree.NameLookupError):
            _ = find_name("or1.'+'", scope)

    def test_function_lookup_via_array_comp(self):
        """Checks that it's not allowed to look up a function via an
        array component"""
        ast = parse_composite_lookup_file("FunctionLookupViaArrayComp.mo")
        scope = find_name(
            "Scoping.NameLookup.Composite.FunctionLookupViaArrayComp",
            ast.classes["ModelicaCompliance"],
        )
        # with self.assertRaises(pymoca.tree.NameLookupError):
        with self.assertRaises(pymoca.tree.NameLookupError):
            _ = find_name("a.f", scope)

    @unittest.skip("TODO: Do this test when new instantiation/flattening is implemented")
    def test_function_lookup_via_array_element(self):
        """Checks that it's allowed to look up a function via an
        array element if the element is a scalar component"""
        ast = parse_composite_lookup_file("FunctionLookupViaArrayElement.mo")
        scope = find_name(
            "Scoping.NameLookup.Composite.FunctionLookupViaArrayElement",
            ast.classes["ModelicaCompliance"],
        )
        # with self.assertRaises(pymoca.tree.NameLookupError):
        found = find_name("a[2].f", scope)
        self.assertIsNotNone(found)

    # TODO: Remove xFail decoration when new flattening is implemented
    @unittest.expectedFailure  # New name lookup with instantiation clashes with old flattening
    def test_need_for_temporary_flattening(self):
        """Test name lookup through 2 levels of inheritance with symbol value modifications

        This is a case where ast.Class.find_class fails and tree.find_name
        works, even if it is not "temporarily flattened" as mentioned in the
        Modelica 3.5 spec section 5.3.2.
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
        # TODO: Remove this `use_find_name` call if `find_name` becomes default
        pymoca.ast.Class.use_find_name(True)
        flat_tree = flatten(ast_tree, comp_ref)
        self.assertEqual(flat_tree.classes[class_name].symbols["bla"].value.value, 1)
        pymoca.ast.Class.use_find_name(False)


@unittest.skip("TODO: Do these after global name syntax is implemented")
class GlobalNameLookupTest(unittest.TestCase):
    """Global name lookup tests from ModelicaCompliance"""

    # TODO: Update when new name lookup is connected to flattening (see todos hints below)

    def test_encapsulated_global_lookup(self):
        """Checks that it's possible to look up a global name, even if
        the current scope is encapsulated"""
        ast = parse_global_lookup_file("EncapsulatedGlobalLookup.mo")
        scope = find_name(
            "Scoping.NameLookup.Global.EncapsulatedGlobalLookup",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("a.y", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Symbol)
        # TODO: flatten and check a.y.value = 1.4


class ImportedNameLookupTest(unittest.TestCase):
    """Imported name lookup tests from ModelicaCompliance"""

    # TODO: Update when new name lookup is connected to flattening (see todos hints below)
    def test_encapsulated(self):
        """Checks that it's possible to import from inside an
        encapsulated model"""
        ast = parse_imported_lookup_file("EncapsulatedImport.mo")
        scope = find_name(
            "Scoping.NameLookup.Imports.EncapsulatedImport",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("a.m.x", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Symbol)
        # TODO: flatten and check a.m.x.value = 2.0

    def test_extend_import(self):
        """Checks that imports are not inherited"""
        ast = parse_imported_lookup_file("ExtendImport.mo")
        scope = find_name(
            "Scoping.NameLookup.Imports.ExtendImport",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("C.A", scope)
        self.assertIsNone(found)

    def test_local_scope(self):
        """Checks that the lookup of an imported name is not started
        in the local scope"""
        ast = parse_imported_lookup_file("ImportLookupLocalScope.mo")
        scope = find_name(
            "Scoping.NameLookup.Imports.ImportLookupLocalScope",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("B", scope)
        self.assertIsNone(found)

    def test_scope_type(self):
        """Checks that it's allowed to import into any kind of class"""
        ast = parse_imported_lookup_file("ImportScopeType.mo")
        scope = find_name(
            "Scoping.NameLookup.Imports.ImportScopeType",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("a", scope)
        self.assertIsNotNone(found)
        found = find_name("b", scope)
        self.assertIsNotNone(found)
        found = find_name("m.y", scope)
        self.assertIsNotNone(found)
        # TODO: flatten and check a.value = 2.0, b.value = 8.0, and m.y.value = 2.0

    @unittest.skip("TODO: Do this test when new instantiation/flattening is implemented")
    def test_modify_import(self):
        """Checks that it's not allowed to modify an import"""
        ast = parse_imported_lookup_file("ModifyImport.mo")
        scope = find_name(
            "Scoping.NameLookup.Imports.ModifyImport",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("b", scope)
        self.assertIsNone(found)

    def test_qualified_import(self):
        """Tests that a qualified import works"""
        ast = parse_imported_lookup_file("QualifiedImport.mo")
        scope = find_name(
            "Scoping.NameLookup.Imports.QualifiedImport",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("b.a.x", scope)
        self.assertIsNotNone(found)
        # TODO: flatten and check a.value = 2.0, b.value = 8.0, and m.y.value = 2.0

    def test_qualified_import_conflict(self):
        """Checks that it's not allowed to have multiple qualified
        import-clauses with the same import name"""
        # This is caught at the parse stage, not in name lookup
        with self.assertRaises(pymoca.parser.ModelicaSyntaxError):
            _ = parse_imported_lookup_file("QualifiedImportConflict.mo")

    def test_qualified_import_non_package(self):
        """Checks that it's not allowed to import a definition which is
        not a package or package element via a qualified import"""
        ast = parse_imported_lookup_file("QualifiedImportNonPackage.mo")
        scope = find_name(
            "Scoping.NameLookup.Imports.QualifiedImportNonPackage",
            ast.classes["ModelicaCompliance"],
        )
        with self.assertRaises(pymoca.tree.NameLookupError):
            _ = find_name("A2", scope)

    def test_qualified_import_protected(self):
        """Checks that it's an error to import a protected element"""
        ast = parse_imported_lookup_file("QualifiedImportProtected.mo")
        scope = find_name(
            "Scoping.NameLookup.Imports.QualifiedImportProtected",
            ast.classes["ModelicaCompliance"],
        )
        with self.assertRaises(pymoca.tree.NameLookupError):
            _ = find_name("A.y", scope)

    def test_recursive(self):
        """Tests that a named recursive import does not work"""
        ast = parse_imported_lookup_file("Recursive.mo")
        scope = find_name(
            "Scoping.NameLookup.Imports.Recursive",
            ast.classes["ModelicaCompliance"],
        )
        with self.assertRaises(pymoca.tree.NameLookupError):
            _ = find_name("A", scope)

    @unittest.skip("TODO: Do this test when new instantiation/flattening is implemented")
    def test_redeclare_import(self):
        """Checks that it's not allowed to redeclare an import"""
        ast = parse_imported_lookup_file("RedeclareImport.mo")
        scope = find_name(
            "Scoping.NameLookup.Imports.RedeclareImport",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("b", scope)
        self.assertIsNone(found)

    def test_renaming_import(self):
        """Tests that a renaming import works"""
        ast = parse_imported_lookup_file("RenamingImport.mo")
        scope = find_name(
            "Scoping.NameLookup.Imports.RenamingImport",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("b.a.x", scope)
        self.assertIsNotNone(found)
        # TODO: flatten and check b.a.x.value = 1.0

    def test_renaming_import_non_package(self):
        """Checks that it's not allowed to import a definition which is
        not a package or package element via a renaming import"""
        ast = parse_imported_lookup_file("RenamingImportNonPackage.mo")
        scope = find_name(
            "Scoping.NameLookup.Imports.RenamingImportNonPackage",
            ast.classes["ModelicaCompliance"],
        )
        with self.assertRaises(pymoca.tree.NameLookupError):
            _ = find_name("A2", scope)

    def test_renaming_single_definition_import(self):
        """Tests that a renaming import works"""
        ast = parse_imported_lookup_file("RenamingSingleDefinitionImport.mo")
        scope = find_name(
            "Scoping.NameLookup.Imports.RenamingSingleDefinitionImport",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("b.a.x", scope)
        self.assertIsNotNone(found)
        # TODO: flatten and check b.a.x.value = 1.0

    def test_single_definition_import(self):
        """Tests that a single definition import works"""
        ast = parse_imported_lookup_file("SingleDefinitionImport.mo")
        scope = find_name(
            "Scoping.NameLookup.Imports.SingleDefinitionImport",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("b.a.x", scope)
        self.assertIsNotNone(found)
        # TODO: flatten and check b.a.x.value = 1.0

    def test_unqualified_import(self):
        """Tests that an unqualified import works"""
        ast = parse_imported_lookup_file("UnqualifiedImport.mo")
        scope = find_name(
            "Scoping.NameLookup.Imports.UnqualifiedImport",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("b.a.x", scope)
        self.assertIsNotNone(found)
        # TODO: flatten and check b.a.x.value = 1.0

    @unittest.skip("ModelicaCompliance test case has a bug")
    def test_unqualified_import_conflict(self):
        """Checks that it's an error if the same name is found in
        multiple unqualified imports"""
        # This ModelicaCompliance test case has a bug.
        # See PR https://github.com/modelica/Modelica-Compliance/pull/75
        # UnqualifiedImportConflict.mo imports from QualifiedImportConflict which is OK
        # if ModelicaCompliance is in MODELICAPATH but not OK as a standalone test that
        # we prefer here. If we do implement this after the ModelicaCompliance bug is
        # fixed it will slow down already slow unqualified name lookup. Unqualified
        # imports are rare and this test case is even more rare, so it seems like maybe
        # not a good idea at this stage for Pymoca.
        ast = parse_imported_lookup_file("UnqualifiedImportConflict.mo")
        scope = find_name(
            "Scoping.NameLookup.Imports.UnqualifiedImportConflict",
            ast.classes["ModelicaCompliance"],
        )
        with self.assertRaises(pymoca.tree.NameLookupError):
            _ = find_name("A.x", scope)

    @unittest.skip("TODO: Do this test when new instantiation/flattening is implemented")
    def test_unqualified_import_non_conflict(self):
        """Checks that it's not an error to be able to find a name in
        multiple unqualified imports, it's only an error if such a name is
        used during name lookup. I.e. both P and P2 contains x in this test, but
        that's ok since x is not used by the importer A."""
        ast = parse_imported_lookup_file("UnqualifiedImportNonConflict.mo")
        scope = find_name(
            "Scoping.NameLookup.Imports.UnqualifiedImportNonConflict",
            ast.classes["ModelicaCompliance"],
        )
        found = find_name("a", scope)
        self.assertIsNotNone(found)

    def test_unqualified_import_non_package(self):
        """Checks that an unqualified import is not allowed to import
        from a non-package"""
        ast = parse_imported_lookup_file("UnqualifiedImportNonPackage.mo")
        scope = find_name(
            "Scoping.NameLookup.Imports.UnqualifiedImportNonPackage",
            ast.classes["ModelicaCompliance"],
        )
        with self.assertRaises(pymoca.tree.NameLookupError):
            _ = find_name("B", scope)

    def test_unqualified_import_protected(self):
        """Checks that the name lookup only considers public members of
        packages imported via unqualified imports"""
        ast = parse_imported_lookup_file("UnqualifiedImportProtected.mo")
        scope = find_name(
            "Scoping.NameLookup.Imports.UnqualifiedImportProtected",
            ast.classes["ModelicaCompliance"],
        )
        with self.assertRaises(pymoca.tree.NameLookupError):
            _ = find_name("A.y", scope)


if __name__ == "__main__":
    unittest.main()
