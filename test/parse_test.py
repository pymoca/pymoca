#!/usr/bin/env python
"""
Modelica parse Tree to AST tree.
"""
from __future__ import absolute_import, division, print_function, print_function, unicode_literals

import contextlib
import enum
import logging
import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import pymoca
from pymoca import ast
from pymoca import parser
from pymoca import tree
from pymoca.parser import DEFAULT_MODEL_CACHE_DB


MY_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(MY_DIR, "models")
COMPLIANCE_DIR = os.path.join(MY_DIR, "libraries", "Modelica-Compliance", "ModelicaCompliance")
IMPORTS_DIR = os.path.join(COMPLIANCE_DIR, "Scoping", "NameLookup", "Imports")
MSL3_DIR = os.path.join(MY_DIR, "libraries", "MSL-3.2.3")
MSL4_DIR = os.path.join(MY_DIR, "libraries", "MSL-4.0.x")


class WorkDirState(enum.Enum):
    CLEAN = "clean"
    DIRTY = "dirty"


@contextlib.contextmanager
def modify_version(version_type: WorkDirState):
    pymoca_version = pymoca.__version__
    if pymoca_version.endswith(".dirty"):
        clean_version = pymoca_version[:-6]
    else:
        clean_version = pymoca_version

    dirty_version = clean_version + ".dirty"

    if version_type == WorkDirState.CLEAN:
        pymoca.__version__ = clean_version
    elif version_type == WorkDirState.DIRTY:
        pymoca.__version__ = dirty_version
    else:
        raise ValueError("Unknown version type")

    try:
        yield
    finally:
        pymoca.__version__ = pymoca_version


class ParseTest(unittest.TestCase):
    """
    Parse test
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @staticmethod
    def flush():
        sys.stdout.flush()
        sys.stdout.flush()
        time.sleep(0.1)

    def test_aircraft(self):
        with open(os.path.join(MODEL_DIR, "Aircraft.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        print("AST TREE\n", ast_tree)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name="Aircraft"))
        print("AST TREE FLAT\n", flat_tree)
        self.flush()

    def test_bouncing_ball(self):
        with open(os.path.join(MODEL_DIR, "BouncingBall.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        print("AST TREE\n", ast_tree)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name="BouncingBall"))
        print(flat_tree)
        print("AST TREE FLAT\n", flat_tree)
        self.flush()

    def test_deep_copy_timeout(self):
        with open(os.path.join(MODEL_DIR, "DeepCopyTimeout.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        # Start a background thread which will run the flattening, such that
        # we can kill it if takes to long.
        thread = threading.Thread(
            target=tree.flatten,
            args=(
                ast_tree,
                ast.ComponentRef(name="Test"),
            ),
        )

        # Daemon threads automatically stop when the program stops (and do not
        # prevent the program from exiting)
        thread.setDaemon(True)
        thread.start()

        # Use a timeout of 5 seconds. We check every 100 ms sec, such that the
        # test is fast to succeed when everything works as expected.
        for _ in range(50):
            time.sleep(0.1)
            if not thread.is_alive():
                return
        self.assertFalse(thread.is_alive(), msg="Timeout occurred")

    def test_estimator(self):
        with open(os.path.join(MODEL_DIR, "./Estimator.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        print("AST TREE\n", ast_tree)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name="Estimator"))
        print("AST TREE FLAT\n", flat_tree)
        self.flush()

    def test_spring(self):
        with open(os.path.join(MODEL_DIR, "Spring.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        print("AST TREE\n", ast_tree)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name="Spring"))
        print("AST TREE FLAT\n", flat_tree)
        self.flush()

    def test_spring_system(self):
        with open(os.path.join(MODEL_DIR, "SpringSystem.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        print("AST TREE\n", ast_tree)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name="SpringSystem"))
        print("AST TREE FLAT\n", flat_tree)
        self.flush()

    def test_duplicate_state(self):
        with open(os.path.join(MODEL_DIR, "DuplicateState.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        print("AST TREE\n", ast_tree)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name="DuplicateState"))
        print("AST TREE FLAT\n", flat_tree)
        self.flush()

    def test_connector(self):
        with open(os.path.join(MODEL_DIR, "Connector.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)  # noqa: F841
        # states = ast_tree.classes['Aircraft'].states
        # names = sorted([state.name for state in states])
        # names_set = sorted(list(set(names)))
        # if names != names_set:
        #     raise IOError('{:s} != {:s}'.format(str(names), str(names_set)))
        self.flush()

    def test_inheritance(self):
        with open(os.path.join(MODEL_DIR, "InheritanceInstantiation.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name="C2"))

        self.assertEqual(flat_tree.classes["C2"].symbols["bcomp1.b"].value.value, 3.0)
        self.assertEqual(flat_tree.classes["C2"].symbols["bcomp3.a"].value.value, 1.0)
        self.assertEqual(flat_tree.classes["C2"].symbols["bcomp3.b"].value.value, 2.0)

    def test_nested_classes(self):
        with open(os.path.join(MODEL_DIR, "NestedClasses.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name="C2"))

        self.assertEqual(flat_tree.classes["C2"].symbols["v1"].nominal.value, 1000.0)
        self.assertEqual(flat_tree.classes["C2"].symbols["v2"].nominal.value, 1000.0)

    def test_inheritance_symbol_modifiers(self):
        with open(os.path.join(MODEL_DIR, "Inheritance.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name="Sub"))

        self.assertEqual(flat_tree.classes["Sub"].symbols["x"].max.value, 30.0)

    def test_extends_modification(self):
        with open(os.path.join(MODEL_DIR, "ExtendsModification.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name="MainModel"))

        self.assertEqual(flat_tree.classes["MainModel"].symbols["e.HQ.H"].min.name, "e.H_b")

    def test_modification_typo(self):
        with open(os.path.join(MODEL_DIR, "ModificationTypo.mo"), "r") as f:
            txt = f.read()

        for c in ["Wrong1", "Wrong2"]:
            with self.assertRaises(tree.ModificationTargetNotFound):
                ast_tree = parser.parse(txt)
                flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name=c))

        for c in ["Good1", "Good2"]:
            ast_tree = parser.parse(txt)
            flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name=c))  # noqa: F841

    def test_tree_lookup(self):
        with open(os.path.join(MODEL_DIR, "TreeLookup.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        # The class we want to flatten. We first have to turn it into a
        # full-fledged ComponentRef.
        class_name = "Level1.Level2.Level3.Test"
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        self.assertIn("elem.tc.i", flat_tree.classes["Level1.Level2.Level3.Test"].symbols.keys())
        self.assertIn("elem.tc.a", flat_tree.classes["Level1.Level2.Level3.Test"].symbols.keys())
        self.assertIn("b", flat_tree.classes["Level1.Level2.Level3.Test"].symbols.keys())

    def test_function_pull(self):
        with open(os.path.join(MODEL_DIR, "FunctionPull.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        class_name = "Level1.Level2.Level3.Function5"
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        # Check if all referenced functions are pulled in
        self.assertIn("Level1.Level2.Level3.f", flat_tree.classes)
        self.assertIn("Level1.Level2.Level3.TestPackage.times2", flat_tree.classes)
        self.assertIn("Level1.Level2.Level3.TestPackage.square", flat_tree.classes)
        self.assertNotIn("Level1.Level2.Level3.TestPackage.not_called", flat_tree.classes)

        # Check if the classes in the flattened tree have the right type
        self.assertEqual(flat_tree.classes["Level1.Level2.Level3.Function5"].type, "model")

        self.assertEqual(flat_tree.classes["Level1.Level2.Level3.f"].type, "function")
        self.assertEqual(
            flat_tree.classes["Level1.Level2.Level3.TestPackage.times2"].type, "function"
        )
        self.assertEqual(
            flat_tree.classes["Level1.Level2.Level3.TestPackage.square"].type, "function"
        )

        # Check whether input/output information of functions comes along properly
        func_t2 = flat_tree.classes["Level1.Level2.Level3.TestPackage.times2"]
        self.assertIn("input", func_t2.symbols["x"].prefixes)
        self.assertIn("output", func_t2.symbols["y"].prefixes)

        # Check if built-in function call statement comes along properly
        func_f = flat_tree.classes["Level1.Level2.Level3.f"]
        self.assertEqual(func_f.statements[0].right.operator, "*")
        # Check if user-specified function call statement comes along properly
        self.assertEqual(
            func_f.statements[0].right.operands[0].operator,
            "Level1.Level2.Level3.TestPackage.times2",
        )

    def test_nested_symbol_modification(self):
        with open(os.path.join(MODEL_DIR, "NestedSymbolModification.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        class_name = "E"
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        self.assertEqual(flat_tree.classes["E"].symbols["c.x"].nominal.value, 2.0)

    def test_redeclare_in_extends(self):
        with open(os.path.join(MODEL_DIR, "RedeclareInExtends.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        class_name = "ChannelZ"
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        self.assertIn("down.Z", flat_tree.classes["ChannelZ"].symbols)

    def test_redeclaration_scope(self):
        with open(os.path.join(MODEL_DIR, "RedeclarationScope.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        class_name = "ChannelZ"
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        self.assertIn("c.up.Z", flat_tree.classes["ChannelZ"].symbols)
        self.assertIn("c.down.A", flat_tree.classes["ChannelZ"].symbols)

    def test_redeclaration_scope_alternative(self):
        with open(os.path.join(MODEL_DIR, "RedeclarationScopeAlternative.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        class_name = "ChannelZ"
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        self.assertIn("c.up.Z", flat_tree.classes["ChannelZ"].symbols)
        self.assertIn("c.down.A", flat_tree.classes["ChannelZ"].symbols)

    def test_extends_redeclareable(self):
        with open(os.path.join(MODEL_DIR, "ExtendsRedeclareable.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        class_name = "E"
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        self.assertIn("z.y", flat_tree.classes["E"].symbols)
        self.assertEqual(flat_tree.classes["E"].symbols["z.y"].nominal.value, 2.0)

    def test_redeclare_nested(self):
        with open(os.path.join(MODEL_DIR, "RedeclareNestedClass.mo.fail_parse"), "r") as f:
            txt = f.read()

        ast_tree = parser.parse(txt)
        self.assertIsNone(ast_tree)

    def test_extends_order(self):
        with open(os.path.join(MODEL_DIR, "ExtendsOrder.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        class_name = "P.M"
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        self.assertEqual(flat_tree.classes["P.M"].symbols["at.m"].value.value, 0.0)

    def test_constant_references(self):
        with open(os.path.join(MODEL_DIR, "ConstantReferences.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        class_name = "b"
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        self.assertEqual(flat_tree.classes["b"].symbols["m.p"].value.value, 2.0)
        self.assertEqual(flat_tree.classes["b"].symbols["M2.m.f"].value.value, 3.0)

    def test_parameter_modification_scope(self):
        with open(os.path.join(MODEL_DIR, "ParameterScope.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        class_name = "ScopeTest"
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        self.assertEqual(flat_tree.classes["ScopeTest"].symbols["nc.p"].value.name, "p")

    def test_custom_units(self):
        with open(os.path.join(MODEL_DIR, "CustomUnits.mo"), "r") as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        class_name = "A"
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        self.assertEqual(flat_tree.classes["A"].symbols["dummy_parameter"].unit.value, "m/s")
        self.assertEqual(flat_tree.classes["A"].symbols["dummy_parameter"].value.value, 10.0)

    def test_extend_from_self(self):
        txt = """
        model A
          extends A;
        end A;"""

        ast_tree = parser.parse(txt)

        class_name = "A"
        comp_ref = ast.ComponentRef.from_string(class_name)

        with self.assertRaisesRegex(Exception, "Cannot extend class 'A' with itself"):
            flat_tree = tree.flatten(ast_tree, comp_ref)  # noqa: F841

    def test_unit_type(self):
        txt = """
            model A
              parameter Integer x = 1;
              parameter Real y = 1.0;
              parameter Real z = 1;  // Mismatch
              parameter Integer w = 1.0;  // Mismatch
            end A;
        """

        ast_tree = parser.parse(txt)

        class_name = "A"
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        # For the moment, we do not raise errors/warnings or do
        # auto-conversions in the parser/flattener.
        self.assertIsInstance(flat_tree.classes["A"].symbols["x"].value.value, int)
        self.assertIsInstance(flat_tree.classes["A"].symbols["y"].value.value, float)
        # self.assertIsInstance(flat_tree.classes['A'].symbols['z'].value.value, int)
        # self.assertIsInstance(flat_tree.classes['A'].symbols['w'].value.value, float)

    def test_unit_type_array(self):
        txt = """
            model A
              parameter Integer x[2, 2] = {{1, 2}, {3, 4}};
              parameter Real y[2, 2] = {{1.0, 2.0}, {3.0, 4.0}};
            end A;
        """

        ast_tree = parser.parse(txt)

        class_name = "A"
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        # For the moment, we leave type conversions to the backends. We only want to
        # be sure that we read in the correct type in the parser.
        for i in range(2):
            for j in range(2):
                self.assertIsInstance(
                    flat_tree.classes["A"].symbols["x"].value.values[i].values[j].value, int
                )
                self.assertIsInstance(
                    flat_tree.classes["A"].symbols["y"].value.values[i].values[j].value, float
                )

    def test_signed_expression(self):
        """Test that both + and - prefix operators work in expressions"""
        txt = """
            model A
              parameter Integer iplus = +1;
              parameter Integer ineg = -iplus;
              parameter Real rplus = +1.0;
              parameter Real rneg = -1.0;
              parameter Real rboth = -1.0 - +1.0;
              parameter Boolean option = true;
              parameter Integer itest = if option then +2 else -2;
            end A;
        """

        ast_tree = parser.parse(txt)

        class_name = "A"
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        # Test that parses into correct expressions.
        symbols = flat_tree.classes["A"].symbols
        for sym in "iplus", "rplus":
            self.assertEqual(symbols[sym].value.operator, "+")
            self.assertEqual(len(symbols[sym].value.operands), 1)
        for sym in "ineg", "rneg":
            self.assertEqual(symbols[sym].value.operator, "-")
            self.assertEqual(len(symbols[sym].value.operands), 1)
        self.assertEqual(symbols["rboth"].value.operands[1].operator, "+")
        self.assertEqual(len(symbols["rboth"].value.operands[1].operands), 1)
        self.assertEqual(symbols["itest"].value.expressions[0].operator, "+")
        self.assertEqual(symbols["itest"].value.expressions[1].operator, "-")
        self.assertEqual(len(symbols["itest"].value.expressions[0].operands), 1)
        self.assertEqual(len(symbols["itest"].value.expressions[1].operands), 1)

    def parse_file(self, pathname):
        "Parse given full path name and return parsed ast.Tree"
        with open(pathname, "r") as mo_file:
            txt = mo_file.read()
        return parser.parse(txt)

    def parse_model_files(self, *pathnames):
        "Parse given files from MODEL_DIR and return parsed ast.Tree"
        tree = None
        for path in pathnames:
            file_tree = self.parse_file(os.path.join(MODEL_DIR, path))
            if tree:
                tree.extend(file_tree)
            else:
                tree = file_tree
        return tree

    def parse_dir_files(self, directory, *pathnames):
        """Parse given file paths relative to dir and return parsed ast.Tree

        Dir is os-specific and paths are unix-style but are transformed to os specific.
        """
        tree = None
        for pathname in pathnames:
            split_path = pathname.split("/")
            full_path = os.path.join(directory, *split_path)
            file_tree = self.parse_file(full_path)
            self.assertIsNotNone(file_tree, "Parse failed: " + full_path)
            if tree:
                tree.extend(file_tree)
            else:
                tree = file_tree
        return tree

    def test_import(self):
        library_tree = self.parse_model_files("TreeLookup.mo", "Import.mo")

        comp_ref = ast.ComponentRef.from_string("A")
        flat_tree = tree.flatten(library_tree, comp_ref)
        expected_symbols = [
            "b.pcb.tc.a",
            "b.pcb.tc.i",
            "b.tb.b",
            "b.tb.elem.tc.a",
            "b.tb.elem.tc.i",
            "pca.tc.a",
            "pca.tc.i",
            "ta.b",
            "ta.elem.tc.a",
            "ta.elem.tc.i",
            "tce_mod.a",
            "tce_mod.i",
            "tce_mod.tcet.b",
            "tce_mod.tcet.elem.tc.a",
            "tce_mod.tcet.elem.tc.i",
        ]
        expected_symbols.sort()
        actual_symbols = sorted(flat_tree.classes["A"].symbols.keys())
        self.assertListEqual(expected_symbols, actual_symbols)
        for eqn in flat_tree.classes["A"].equations:
            if eqn.left == "tce_mod.tect.b":
                self.assertEqual(eqn.right.value, 4)
            elif eqn.left == "b.tb.b":
                self.assertEqual(eqn.right.value, 3)

    # Import tests from the Modelica Compliance library (mostly the shouldPass=true cases)
    def parse_imports_file(self, pathname):
        "Parse given path relative to IMPORTS_DIR and return parsed ast.Tree"
        arg_ast = self.parse_file(os.path.join(IMPORTS_DIR, pathname))
        icon_ast = self.parse_file(os.path.join(COMPLIANCE_DIR, "Icons.mo"))
        icon_ast.extend(arg_ast)
        return icon_ast

    def test_import_encapsulated(self):
        library_ast = self.parse_imports_file("EncapsulatedImport.mo")
        model_name = "ModelicaCompliance.Scoping.NameLookup.Imports.EncapsulatedImport"
        flat_class = ast.ComponentRef.from_string(model_name)
        flat_ast = tree.flatten(library_ast, flat_class)
        self.assertIn("a.m.x", flat_ast.classes[model_name].symbols)

    def test_import_scope_type(self):
        library_ast = self.parse_imports_file("ImportScopeType.mo")
        model_name = "ModelicaCompliance.Scoping.NameLookup.Imports.ImportScopeType"
        flat_class = ast.ComponentRef.from_string(model_name)
        flat_ast = tree.flatten(library_ast, flat_class)
        self.assertIn("a", flat_ast.classes[model_name].symbols)
        self.assertIn("b", flat_ast.classes[model_name].symbols)
        self.assertIn("m.y", flat_ast.classes[model_name].symbols)

    def test_import_qualified(self):
        library_ast = self.parse_imports_file("QualifiedImport.mo")
        model_name = "ModelicaCompliance.Scoping.NameLookup.Imports.QualifiedImport"
        flat_class = ast.ComponentRef.from_string(model_name)
        flat_ast = tree.flatten(library_ast, flat_class)
        self.assertIn("b.a.x", flat_ast.classes[model_name].symbols)

    def test_import_renaming(self):
        library_ast = self.parse_imports_file("RenamingImport.mo")
        model_name = "ModelicaCompliance.Scoping.NameLookup.Imports.RenamingImport"
        flat_class = ast.ComponentRef.from_string(model_name)
        flat_ast = tree.flatten(library_ast, flat_class)
        self.assertIn("b.a.x", flat_ast.classes[model_name].symbols)

    def test_import_renaming_single_definition(self):
        library_ast = self.parse_imports_file("RenamingSingleDefinitionImport.mo")
        model_name = "ModelicaCompliance.Scoping.NameLookup.Imports.RenamingSingleDefinitionImport"
        flat_class = ast.ComponentRef.from_string(model_name)
        flat_ast = tree.flatten(library_ast, flat_class)
        self.assertIn("b.a.x", flat_ast.classes[model_name].symbols)

    def test_import_single_definition(self):
        library_ast = self.parse_imports_file("SingleDefinitionImport.mo")
        model_name = "ModelicaCompliance.Scoping.NameLookup.Imports.SingleDefinitionImport"
        flat_class = ast.ComponentRef.from_string(model_name)
        flat_ast = tree.flatten(library_ast, flat_class)
        self.assertIn("b.a.x", flat_ast.classes[model_name].symbols)

    def test_import_unqualified(self):
        library_ast = self.parse_imports_file("UnqualifiedImport.mo")
        model_name = "ModelicaCompliance.Scoping.NameLookup.Imports.UnqualifiedImport"
        flat_class = ast.ComponentRef.from_string(model_name)
        flat_ast = tree.flatten(library_ast, flat_class)
        self.assertIn("b.a.x", flat_ast.classes[model_name].symbols)

    def test_import_unqualified_nonconflict(self):
        library_ast = self.parse_imports_file("UnqualifiedImportNonConflict.mo")
        model_name = "ModelicaCompliance.Scoping.NameLookup.Imports.UnqualifiedImportNonConflict"
        flat_class = ast.ComponentRef.from_string(model_name)
        flat_ast = tree.flatten(library_ast, flat_class)
        self.assertIn("a.y", flat_ast.classes[model_name].symbols)

    def test_import_not_inherited(self):
        library_ast = self.parse_imports_file("ExtendImport.mo")
        model_name = "ModelicaCompliance.Scoping.NameLookup.Imports.ExtendImport"
        flat_class = ast.ComponentRef.from_string(model_name)
        with self.assertRaises(ast.ClassNotFoundError):
            flat_ast = tree.flatten(library_ast, flat_class)  # noqa: F841

    # Tests using the Modelica Standard Library
    def test_msl_opamp_units(self):
        """Test import from Modelica Standard Library 4.0.0 using SI.Units

        This is the simplest case found that works around current pymoca issues
        flattening MSL examples.
        """
        library_tree = self.parse_dir_files(
            MSL4_DIR,
            "Modelica/Icons.mo",
            "Modelica/Units.mo",
            "Modelica/Electrical/package.mo",  # to pick up SI import
            "Modelica/Electrical/Analog/Interfaces/PositivePin.mo",
            "Modelica/Electrical/Analog/Interfaces/NegativePin.mo",
            "Modelica/Electrical/Analog/Basic/OpAmp.mo",
        )
        model_name = "Modelica.Electrical.Analog.Basic.OpAmp"
        flat_class = ast.ComponentRef.from_string(model_name)
        flat_tree = tree.flatten(library_tree, flat_class)
        symbols = flat_tree.classes[model_name].symbols
        self.assertIn("in_p.i", symbols)
        self.assertEqual(symbols["in_p.i"].unit.value, "A")
        self.assertEqual(symbols["in_p.i"].quantity.value, "ElectricCurrent")
        self.assertIn("vin", symbols)
        self.assertEqual(symbols["vin"].unit.value, "V")
        self.assertEqual(symbols["vin"].quantity.value, "ElectricPotential")

    def test_msl3_twopin_units(self):
        """Test import from Modelica Standard Library 3.2.3 using SIunits

        This is a simple case that works around current pymoca issues
        flattening MSL examples.
        """
        library_tree = self.parse_dir_files(
            MSL3_DIR,
            "Modelica/Icons.mo",
            "Modelica/SIunits.mo",
            "Modelica/Electrical/Analog/package.mo",  # to pick up SI import
            "Modelica/Electrical/Analog/Interfaces.mo",
        )
        model_name = "Modelica.Electrical.Analog.Interfaces.TwoPort"
        flat_class = ast.ComponentRef.from_string(model_name)
        flat_tree = tree.flatten(library_tree, flat_class)
        symbols = flat_tree.classes[model_name].symbols
        self.assertIn("p1.i", symbols)
        self.assertEqual(symbols["p1.i"].unit.value, "A")
        self.assertEqual(symbols["p1.i"].quantity.value, "ElectricCurrent")
        self.assertIn("v1", symbols)
        self.assertEqual(symbols["v1"].unit.value, "V")
        self.assertEqual(symbols["v1"].quantity.value, "ElectricPotential")

    def test_msl_flange_units(self):
        """Test displayUnit attribute imported from MSL 4.0.0 SI.Units"""
        library_tree = self.parse_dir_files(
            MSL4_DIR,
            "Modelica/Icons.mo",
            "Modelica/Units.mo",
            "Modelica/Mechanics/package.mo",  # to pick up SI import
            "Modelica/Mechanics/Rotational/Interfaces/Flange.mo",
            "Modelica/Mechanics/Rotational/Interfaces/Flange_a.mo",
            "Modelica/Mechanics/Rotational/Interfaces/PartialAbsoluteSensor.mo",
        )
        model_name = "Modelica.Mechanics.Rotational.Interfaces.PartialAbsoluteSensor"
        flat_class = ast.ComponentRef.from_string(model_name)
        flat_tree = tree.flatten(library_tree, flat_class)
        symbols = flat_tree.classes[model_name].symbols
        self.assertIn("flange.phi", symbols)
        self.assertEqual(symbols["flange.phi"].unit.value, "rad")
        self.assertEqual(symbols["flange.phi"].displayUnit.value, "deg")
        self.assertEqual(symbols["flange.phi"].quantity.value, "Angle")

    def test_class_comment(self):
        """Test that class comment/description is retained after flattening"""
        library_tree = self.parse_model_files("Aircraft.mo")
        comp_ref = ast.ComponentRef.from_string("Aircraft")
        flat_tree = tree.flatten(library_tree, comp_ref)
        aircraft = flat_tree.classes["Aircraft"]
        self.assertEqual(aircraft.comment, "the aircraft")
        self.assertEqual(aircraft.symbols["accel.a_x"].comment, "true acceleration")
        self.assertEqual(aircraft.symbols["accel.ma_x"].comment, "measured acceleration")
        self.assertEqual(aircraft.symbols["body.g"].comment, "")

    def test_derived_type_value_modification(self):
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
        class_name = "A.D"
        comp_ref = ast.ComponentRef.from_string(class_name)
        flat_tree = tree.flatten(ast_tree, comp_ref)
        self.assertIsNone(flat_tree.classes[class_name].symbols["c.y"].value.value)
        self.assertEqual(flat_tree.classes[class_name].equations[0].left.name, "c.y")
        self.assertEqual(flat_tree.classes[class_name].equations[0].right.value, 3)

        # Parsing AST again, otherwise there is an error in A.E flattening
        ast_tree = parser.parse(txt)
        class_name = "A.E"
        comp_ref = ast.ComponentRef.from_string(class_name)
        flat_tree = tree.flatten(ast_tree, comp_ref)
        self.assertIsNone(flat_tree.classes[class_name].symbols["d.c.y"].value.value)
        self.assertEqual(flat_tree.classes[class_name].equations[0].left.name, "d.c.y")
        self.assertEqual(flat_tree.classes[class_name].equations[0].right.value, 4)

    def test_parse_cache_hit(self):
        """Test caching of models"""

        txt = """
            model A
              parameter Real x, y;
            equation
              der(y) = x;
            end A;
        """

        with modify_version(WorkDirState.CLEAN), tempfile.TemporaryDirectory() as tmpdirname:
            full_db_path = Path(tmpdirname) / DEFAULT_MODEL_CACHE_DB

            # The cache database should not exist yet
            self.assertFalse(full_db_path.exists())

            _ = parser.parse(txt, model_cache_folder=Path(tmpdirname))

            # And now the database _should exist_, and we check its contents
            # where we expect to find a single cached entry
            self.assertTrue(full_db_path.exists())

            conn = sqlite3.connect(full_db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT value FROM metadata WHERE key='created_at'")
            first_created_at = int(cursor.fetchone()[0])

            cursor.execute("SELECT last_hit FROM models")
            first_hit_time = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM models")
            self.assertEqual(cursor.fetchone()[0], 1)

            # Check that we get log messages saying the cache entry was found
            # We also force an update to the cache hit time
            logger = logging.getLogger("pymoca")
            with self.assertLogs(logger, level="DEBUG") as cm:
                _ = parser.parse(
                    txt, model_cache_folder=Path(tmpdirname), always_update_last_hit=True
                )
                self.assertIn(") found in cache", cm.output[0])

            cursor.execute("SELECT value FROM metadata WHERE key='created_at'")
            second_created_at = int(cursor.fetchone()[0])

            cursor.execute("SELECT last_hit FROM models")
            second_hit_time = cursor.fetchone()[0]

            # Check that the created_at time was _not_ updated, i.e. the
            # database was not recreated for some reason.
            self.assertEqual(first_created_at, second_created_at)

            # Check that, if we parse it _again_, the `last_hit` updates
            self.assertGreater(second_hit_time, first_hit_time)

            # Check that there's still only one model in the cache
            cursor.execute("SELECT COUNT(*) FROM models")
            self.assertEqual(cursor.fetchone()[0], 1)

            cursor.close()
            conn.close()

    def test_parse_cache_purge(self):
        """Test that models that have not been hit in N days are purged"""

        model_a = """
            model A
              parameter Real x, y;
            equation
              der(y) = x;
            end A;
        """

        model_b = """
            model B
              parameter Real x, y;
            equation
              der(y) = x;
            end B;
        """

        with modify_version(WorkDirState.CLEAN), tempfile.TemporaryDirectory() as tmpdirname:
            full_db_path = Path(tmpdirname) / DEFAULT_MODEL_CACHE_DB

            # Parse the models to add them to the cache
            for txt in [model_a, model_b]:
                _ = parser.parse(txt, model_cache_folder=Path(tmpdirname))

            conn = sqlite3.connect(full_db_path)
            cursor = conn.cursor()

            # Check that the models are in the cache
            cursor.execute("SELECT COUNT(*) FROM models")
            self.assertEqual(cursor.fetchone()[0], 2)

            cursor.execute("SELECT value FROM metadata WHERE key='last_prune'")
            first_prune_time = int(cursor.fetchone()[0])

            cursor.execute("SELECT value FROM metadata WHERE key='created_at'")
            first_created_at = int(cursor.fetchone()[0])

            # Reimport the module to force a cache purge check, but with an
            # expiration time such that the models should not be purged
            import importlib

            importlib.reload(parser)

            _ = parser.parse(
                model_b,
                model_cache_folder=Path(tmpdirname),
                cache_expiration_days=1,
            )

            cursor.execute("SELECT COUNT(*) FROM models")
            self.assertEqual(cursor.fetchone()[0], 2)

            # Reimport the module again, but now we force a purge by setting
            # expiration to zero
            importlib.reload(parser)

            _ = parser.parse(
                model_b,
                model_cache_folder=Path(tmpdirname),
                cache_expiration_days=0,
            )

            cursor.execute("SELECT value FROM metadata WHERE key='last_prune'")
            second_prune_time = int(cursor.fetchone()[0])

            cursor.execute("SELECT value FROM metadata WHERE key='created_at'")
            second_created_at = int(cursor.fetchone()[0])

            # Check that the other model has been purged from the cache.
            cursor.execute("SELECT COUNT(*) FROM models")
            self.assertEqual(cursor.fetchone()[0], 1)

            # Check that the last prune time was updated
            self.assertGreater(second_prune_time, first_prune_time)

            # And that the creation time was not
            self.assertEqual(first_created_at, second_created_at)

            cursor.close()
            conn.close()

    def test_dirty_no_caching(self):
        """Test cache and cache creation bypass if working directory is dirty"""

        txt = """
            model A
              parameter Real x, y;
            equation
              der(y) = x;
            end A;
        """

        with modify_version(WorkDirState.DIRTY), tempfile.TemporaryDirectory() as tmpdirname:
            full_db_path = Path(tmpdirname) / DEFAULT_MODEL_CACHE_DB

            # The cache database should not exist yet
            self.assertFalse(full_db_path.exists())

            _ = parser.parse(txt, model_cache_folder=Path(tmpdirname))

            # And now the database should not exist
            self.assertFalse(full_db_path.exists())

    def test_unpickling_error(self):
        """Test that we can handle unpickling errors, and then recreate the cache entry"""

        txt = """
            model A
              parameter Real x, y;
            equation
              der(y) = x;
            end A;
        """

        with modify_version(WorkDirState.CLEAN), tempfile.TemporaryDirectory() as tmpdirname:
            full_db_path = Path(tmpdirname) / DEFAULT_MODEL_CACHE_DB

            # The cache database should not exist yet
            self.assertFalse(full_db_path.exists())

            _ = parser.parse(txt, model_cache_folder=Path(tmpdirname))

            self.assertTrue(full_db_path.exists())

            # Modify the single entry in the cache to make it unpickleable
            conn = sqlite3.connect(full_db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT txt_hash FROM models")
            txt_hash = cursor.fetchone()[0]
            cursor.execute(
                "UPDATE models SET data = ? WHERE txt_hash = ?", (b"not a pickle", txt_hash)
            )
            conn.commit()

            cursor.close()
            conn.close()

            # Check that we get log messages saying the cache entry is corrupt
            logger = logging.getLogger("pymoca")
            with self.assertLogs(logger, level="WARNING") as cm:
                _ = parser.parse(txt, model_cache_folder=Path(tmpdirname))

                self.assertIn("failed to unpickle", cm.output[0])
                n_warnings = len(cm.output)

                # Check that we get no additional warnings when unpickling again
                _ = parser.parse(txt, model_cache_folder=Path(tmpdirname))
                self.assertEqual(len(cm.output), n_warnings)

    def test_incorrect_table_layout(self):
        """Test that a corrupt cache file is ignored"""

        txt = """
            model A
              parameter Real x, y;
            equation
              der(y) = x;
            end A;
        """

        with modify_version(WorkDirState.CLEAN), tempfile.TemporaryDirectory() as tmpdirname:
            full_db_path = Path(tmpdirname) / DEFAULT_MODEL_CACHE_DB

            # The cache database should not exist yet
            self.assertFalse(full_db_path.exists())

            # Create a database with incorrectly structured tables
            conn = sqlite3.connect(full_db_path)
            cursor = conn.cursor()

            dummy_table_str = """
                CREATE TABLE {} (
                    wrong_key TEXT,
                    wrong_value TEXT,
                    PRIMARY KEY (wrong_key)
                )
            """

            cursor.execute(dummy_table_str.format("models"))
            cursor.execute(dummy_table_str.format("metadata"))

            conn.close()

            # And now the database should exist
            self.assertTrue(full_db_path.exists())

            # Check that we get log messages saying the layout is incorrect
            logger = logging.getLogger("pymoca")
            with self.assertLogs(logger, level="WARNING") as cm:
                _ = parser.parse(txt, model_cache_folder=Path(tmpdirname))

                self.assertIn("Model text cache table layout didn't match", cm.output[0])
                self.assertIn("Metadata table layout didn't match", cm.output[1])

    def test_corrupt_cache_file(self):
        """Test that a corrupt cache file is ignored"""

        txt = """
            model A
              parameter Real x, y;
            equation
              der(y) = x;
            end A;
        """

        with modify_version(WorkDirState.CLEAN), tempfile.TemporaryDirectory() as tmpdirname:
            full_db_path = Path(tmpdirname) / DEFAULT_MODEL_CACHE_DB

            # The cache database should not exist yet
            self.assertFalse(full_db_path.exists())

            # Create a corrupt cache file
            with open(full_db_path, "w") as f:
                f.write("This is not a valid SQLite database file")

            logger = logging.getLogger("pymoca")
            with self.assertLogs(logger, level="WARNING") as cm:
                _ = parser.parse(txt, model_cache_folder=Path(tmpdirname))

                self.assertIn("Model cache database is corrupt", cm.output[0])


if __name__ == "__main__":
    unittest.main()
