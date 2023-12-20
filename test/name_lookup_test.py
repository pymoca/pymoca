#!/usr/bin/env python
"""
Test against Modelica name lookup rules
"""

import os
import unittest

import pymoca.ast
import pymoca.parser
import pymoca.tree

MY_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(MY_DIR, "models")
COMPLIANCE_DIR = os.path.join(MY_DIR, "libraries", "Modelica-Compliance", "ModelicaCompliance")
SIMPLE_LOOKUP_DIR = os.path.join(COMPLIANCE_DIR, "Scoping", "NameLookup", "Simple")
COMPOSITE_LOOKUP_DIR = os.path.join(COMPLIANCE_DIR, "Scoping", "NameLookup", "Composite")


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


def parse_simple_lookup_file(pathname):
    "Parse given path relative to SIMPLE_LOOKUP_DIR and return parsed ast.Tree"
    arg_ast = parse_file(os.path.join(SIMPLE_LOOKUP_DIR, pathname))
    icon_ast = parse_file(os.path.join(COMPLIANCE_DIR, "Icons.mo"))
    if None in (arg_ast, icon_ast):
        return None
    icon_ast.extend(arg_ast)
    return icon_ast


def parse_composite_lookup_file(pathname):
    "Parse given path relative to SIMPLE_LOOKUP_DIR and return parsed ast.Tree"
    arg_ast = parse_file(os.path.join(COMPOSITE_LOOKUP_DIR, pathname))
    icon_ast = parse_file(os.path.join(COMPLIANCE_DIR, "Icons.mo"))
    if None in (arg_ast, icon_ast):
        return None
    icon_ast.extend(arg_ast)
    return icon_ast


finder = pymoca.tree.NameFinder()


class SimpleNameLookupTest(unittest.TestCase):
    """Simple name lookup tests from ModelicaCompliance"""

    # TODO: Update when new name lookup is connected to flattening (see todos hints below)

    def test_encapsulation(self):
        """Tests that names can be found or not if the scope is encapsulated"""
        ast = parse_simple_lookup_file("Encapsulation.mo")
        scope = finder.find_name(
            "Scoping.NameLookup.Simple.Encapsulation.A", ast.classes["ModelicaCompliance"]
        )
        found = finder.find_name("x", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Symbol)
        # TODO: flatten and check x.value

        # Now go in the reverse direction, bumping into encapsulation
        found = finder.find_name("Encapsulation", found.parent)
        self.assertIsNone(found)

        # Check that builtin abs function is looked up correctly in encapsulated scope
        abs_scope = finder.find_name(
            "Scoping.NameLookup.Simple.Encapsulation.A", ast.classes["ModelicaCompliance"]
        )
        self.assertIsNotNone(abs_scope)
        found = finder.find_name("abs", abs_scope)
        # TODO: Uncomment after implementing abs built-in function lookup
        # self.assertIsNotNone(found)

    def test_enclosing_class_lookup_class(self):
        """Tests that classes can be looked up in an enclosing scope"""
        ast = parse_simple_lookup_file("EnclosingClassLookupClass.mo")
        scope = finder.find_name(
            "Scoping.NameLookup.Simple.EnclosingClassLookupClass",
            ast.classes["ModelicaCompliance"],
        )
        found = finder.find_name("b.a.x", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Symbol)
        # Now go reverse direction, looking for a compound name but not fully qualified
        found = finder.find_name("Scoping.NameLookup", found.parent)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Class)
        # TODO: flatten and check x.value == 2

    def test_enclosing_class_lookup_constant(self):
        """Tests that constants can be looked up in an enclosing scope"""
        ast = parse_simple_lookup_file("EnclosingClassLookupConstant.mo")
        scope = finder.find_name(
            "Scoping.NameLookup.Simple.EnclosingClassLookupConstant",
            ast.classes["ModelicaCompliance"],
        )
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        scope = finder.find_name("A", scope)
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        found = finder.find_name("x", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Symbol)
        # TODO: flatten and check y.value == 4

    def test_enclosing_class_lookup_nonconstant(self):
        """Tests that variables found in an enclosing scope must be declared constant"""
        ast = parse_simple_lookup_file("EnclosingClassLookupNonConstant.mo")
        scope = finder.find_name(
            "Scoping.NameLookup.Simple.EnclosingClassLookupNonConstant",
            ast.classes["ModelicaCompliance"],
        )
        scope = finder.find_name("A", scope)
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        with self.assertRaises(pymoca.tree.NameLookupError):
            finder.find_name("x", scope)

    def test_enclosing_class_lookup_shadowed_constant(self):
        """Tests that variables found in an enclosing scope must be declared constant"""
        ast = parse_simple_lookup_file("EnclosingClassLookupShadowedConstant.mo")
        scope = finder.find_name(
            "Scoping.NameLookup.Simple.EnclosingClassLookupShadowedConstant",
            ast.classes["ModelicaCompliance"],
        )
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        scope = finder.find_name("A", scope)
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        scope = finder.find_name("B", scope)
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        with self.assertRaises(pymoca.tree.NameLookupError):
            finder.find_name("x", scope)

    def test_local_class_name_lookup(self):
        """Tests that variables found in an enclosing scope must be declared constant"""
        ast = parse_simple_lookup_file("LocalClassNameLookup.mo")
        scope = finder.find_name(
            "Scoping.NameLookup.Simple.LocalClassNameLookup",
            ast.classes["ModelicaCompliance"],
        )
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        found = finder.find_name("A", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Class)

    def test_local_comp_name_lookup(self):
        """Tests that a component name in the local scope can be found"""
        ast = parse_simple_lookup_file("LocalCompNameLookup.mo")
        scope = finder.find_name(
            "Scoping.NameLookup.Simple.LocalCompNameLookup",
            ast.classes["ModelicaCompliance"],
        )
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        found = finder.find_name("x", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Symbol)
        found = finder.find_name("y", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Symbol)

    def test_outside_encapsulation(self):
        """Tests that elements defined outside an encapsulated scope
        can't be found in the encapsulated scope"""
        ast = parse_simple_lookup_file("OutsideEncapsulation.mo")
        scope = finder.find_name(
            "Scoping.NameLookup.Simple.OutsideEncapsulation.A",
            ast.classes["ModelicaCompliance"],
        )
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        found = finder.find_name("x", scope)
        self.assertIsNone(found)

    def test_outside_encapsulation_multi(self):
        """Tests that elements defined outside an encapsulated scope
        can't be found in the encapsulated scope"""
        ast = parse_simple_lookup_file("OutsideEncapsulationMulti.mo")
        scope = finder.find_name(
            "Scoping.NameLookup.Simple.OutsideEncapsulationMulti",
            ast.classes["ModelicaCompliance"],
        )
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        scope = finder.find_name("A", scope)
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        scope = finder.find_name("B", scope)
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        found = finder.find_name("x", scope)
        self.assertIsNone(found)

    def test_qualified_import_priority(self):
        """Tests that qualified imports have lower priority than local
        and inherited names during name lookup"""
        ast = parse_simple_lookup_file("QualifiedImportPriority.mo")
        scope = finder.find_name(
            "Scoping.NameLookup.Simple.QualifiedImportPriority",
            ast.classes["ModelicaCompliance"],
        )
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        found = finder.find_name("d.x", scope)
        self.assertIsNotNone(found)
        x_value = found.class_modification.arguments[0].value.modifications[0].value
        self.assertAlmostEqual(x_value, 2.0)
        found = finder.find_name("b.x", scope)
        self.assertIsNotNone(found)
        x_value = found.class_modification.arguments[0].value.modifications[0].value
        self.assertAlmostEqual(x_value, 3.0)
        found = finder.find_name("c.x", scope)
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
        scope = finder.find_name(
            "Scoping.NameLookup.Simple.UnqualifiedImportPriority",
            ast.classes["ModelicaCompliance"],
        )
        self.assertIsNotNone(scope)
        self.assertIsInstance(scope, pymoca.ast.Class)
        found = finder.find_name("e.x", scope)
        self.assertIsNotNone(found)
        x_value = found.class_modification.arguments[0].value.modifications[0].value
        self.assertAlmostEqual(x_value, 2.0)
        found = finder.find_name("b.x", scope)
        self.assertIsNotNone(found)
        x_value = found.class_modification.arguments[0].value.modifications[0].value
        self.assertAlmostEqual(x_value, 3.0)
        found = finder.find_name("c.x", scope)
        self.assertIsNotNone(found)
        x_value = found.class_modification.arguments[0].value.modifications[0].value
        self.assertAlmostEqual(x_value, 1.0)
        found = finder.find_name("d.x", scope)
        self.assertIsNotNone(found)
        x_value = found.class_modification.arguments[0].value.modifications[0].value
        self.assertAlmostEqual(x_value, 4.0)


class CompositeNameLookupTest(unittest.TestCase):
    """Composite name lookup tests from ModelicaCompliance"""

    # TODO: Update when new name lookup is connected to flattening (see todos hints below)

    def test_package_lookup_class(self):
        """Checks that it's possible to look up a class in a package"""
        ast = parse_composite_lookup_file("PackageLookupClass.mo")
        scope = finder.find_name(
            "Scoping.NameLookup.Composite.PackageLookupClass",
            ast.classes["ModelicaCompliance"],
        )
        found = finder.find_name("a.x", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Symbol)
        # TODO: flatten and check a.x.value = 531.0

    def test_package_lookup_constant(self):
        """Checks that it's possible to look up a constant in a package"""
        ast = parse_composite_lookup_file("PackageLookupConstant.mo")
        scope = finder.find_name(
            "Scoping.NameLookup.Composite.PackageLookupConstant",
            ast.classes["ModelicaCompliance"],
        )
        found = finder.find_name("P.x", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Symbol)
        # TODO: flatten and check y.value = 5.1

    def test_nested_comp_lookup(self):
        """Checks that composite names where each identifier is a component can be looked up"""
        ast = parse_composite_lookup_file("NestedCompLookup.mo")
        scope = finder.find_name(
            "Scoping.NameLookup.Composite.NestedCompLookup",
            ast.classes["ModelicaCompliance"],
        )
        found = finder.find_name("c.b.a.x", scope)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, pymoca.ast.Symbol)
        # TODO: flatten and check y.value = 17 (integer)

    def test_partial_class_lookup(self):
        """Checks that it's not allowed to look up a name in a partial class

        Above is what PartialClassLookup.mo says, but according to the spec,
        it's only forbidden in a simulation model, so the check for partial
        is left to the caller and find_name returns the found class."""
        ast = parse_composite_lookup_file("PartialClassLookup.mo")
        scope = finder.find_name(
            "Scoping.NameLookup.Composite.PartialClassLookup",
            ast.classes["ModelicaCompliance"],
        )
        found = finder.find_name("P.x", scope)
        self.assertIsNotNone(found)
        self.assertTrue(found.parent.partial)

    def test_non_function_lookup_via_comp(self):
        """Checks that it's not allowed to look up a non-function class via a component."""
        ast = parse_composite_lookup_file("NonFunctionLookupViaComp.mo")
        scope = finder.find_name(
            "Scoping.NameLookup.Composite.NonFunctionLookupViaComp",
            ast.classes["ModelicaCompliance"],
        )
        found = finder.find_name("a.B", scope)
        self.assertIsNone(found)

    def test_non_package_lookup_comp(self):
        """Checks that looking up an non-encapsulated element, in this
        case a component, inside a class which does not satisfy the requirements for
        a package is forbidden"""
        ast = parse_composite_lookup_file("NonPackageLookupComp.mo")
        scope = finder.find_name(
            "Scoping.NameLookup.Composite.NonPackageLookupComp",
            ast.classes["ModelicaCompliance"],
        )
        with self.assertRaises(pymoca.tree.NameLookupError):
            _ = finder.find_name("A.x", scope)


if __name__ == "__main__":
    unittest.main()
