#!/usr/bin/env python
"""
Modelica parse Tree to AST tree.
"""
from __future__ import print_function, absolute_import, division, print_function, unicode_literals

import os
import sys
import time
import unittest
import threading

from pymoca import parser
from pymoca import tree
from pymoca import ast

MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')


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
        with open(os.path.join(MODEL_DIR, 'Aircraft.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        print('AST TREE\n', ast_tree)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name='Aircraft'))
        print('AST TREE FLAT\n', flat_tree)
        self.flush()

    def test_bouncing_ball(self):
        with open(os.path.join(MODEL_DIR, 'BouncingBall.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        print('AST TREE\n', ast_tree)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name='BouncingBall'))
        print(flat_tree)
        print('AST TREE FLAT\n', flat_tree)
        self.flush()

    def test_deep_copy_timeout(self):
        with open(os.path.join(MODEL_DIR, 'DeepCopyTimeout.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        # Start a background thread which will run the flattening, such that
        # we can kill it if takes to long.
        # noinspection PyTypeChecker
        thread = threading.Thread(target=tree.flatten, args=(ast_tree, ast.ComponentRef(name='Test'),))

        # Daemon threads automatically stop when the program stops (and do not
        # prevent the program from exiting)
        thread.setDaemon(True)
        thread.start()

        # Use a timeout of 5 seconds. We check every 100 ms sec, such that the
        # test is fast to succeed when everything works as expected.
        for i in range(50):
            time.sleep(0.1)
            if not thread.isAlive():
                return
        self.assertFalse(thread.isAlive(), msg='Timeout occurred')

    def test_estimator(self):
        with open(os.path.join(MODEL_DIR, './Estimator.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        print('AST TREE\n', ast_tree)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name='Estimator'))
        print('AST TREE FLAT\n', flat_tree)
        self.flush()

    def test_spring(self):
        with open(os.path.join(MODEL_DIR, 'Spring.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        print('AST TREE\n', ast_tree)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name='Spring'))
        print('AST TREE FLAT\n', flat_tree)
        self.flush()

    def test_spring_system(self):
        with open(os.path.join(MODEL_DIR, 'SpringSystem.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        print('AST TREE\n', ast_tree)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name='SpringSystem'))
        print('AST TREE FLAT\n', flat_tree)
        self.flush()

    def test_duplicate_state(self):
        with open(os.path.join(MODEL_DIR, 'DuplicateState.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        print('AST TREE\n', ast_tree)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name='DuplicateState'))
        print('AST TREE FLAT\n', flat_tree)
        self.flush()

    def test_connector(self):
        with open(os.path.join(MODEL_DIR, 'Connector.mo'), 'r') as f:
            txt = f.read()
        # noinspection PyUnusedLocal
        ast_tree = parser.parse(txt)
        # states = ast_tree.classes['Aircraft'].states
        # names = sorted([state.name for state in states])
        # names_set = sorted(list(set(names)))
        # if names != names_set:
        #     raise IOError('{:s} != {:s}'.format(str(names), str(names_set)))
        self.flush()

    def test_inheritance(self):
        with open(os.path.join(MODEL_DIR, 'InheritanceInstantiation.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name='C2'))

        self.assertEqual(flat_tree.classes['C2'].symbols['bcomp1.b'].value.value, 3.0)
        self.assertEqual(flat_tree.classes['C2'].symbols['bcomp3.a'].value.value, 1.0)
        self.assertEqual(flat_tree.classes['C2'].symbols['bcomp3.b'].value.value, 2.0)

    def test_nested_classes(self):
        with open(os.path.join(MODEL_DIR, 'NestedClasses.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name='C2'))

        self.assertEqual(flat_tree.classes['C2'].symbols['v1'].nominal.value, 1000.0)
        self.assertEqual(flat_tree.classes['C2'].symbols['v2'].nominal.value, 1000.0)

    def test_inheritance_symbol_modifiers(self):
        with open(os.path.join(MODEL_DIR, 'Inheritance.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name='Sub'))

        self.assertEqual(flat_tree.classes['Sub'].symbols['x'].max.value, 30.0)

    def test_extends_modification(self):
        with open(os.path.join(MODEL_DIR, 'ExtendsModification.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name='MainModel'))

        self.assertEqual(flat_tree.classes['MainModel'].symbols['e.HQ.H'].min.name, "e.H_b")

    def test_modification_typo(self):
        with open(os.path.join(MODEL_DIR, 'ModificationTypo.mo'), 'r') as f:
            txt = f.read()

        for c in ["Wrong1", "Wrong2"]:
            with self.assertRaises(tree.ModificationTargetNotFound):
                ast_tree = parser.parse(txt)
                flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name=c))

        for c in ["Good1", "Good2"]:
            ast_tree = parser.parse(txt)
            flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name=c))

    def test_tree_lookup(self):
        with open(os.path.join(MODEL_DIR, 'TreeLookup.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        # The class we want to flatten. We first have to turn it into a
        # full-fledged ComponentRef.
        class_name = 'Level1.Level2.Level3.Test'
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        # NOTE: We currently do not flatten the component ref in the final
        # tree's keys, so we use it once again to lookup the flattened class.
        self.assertIn('elem.tc.i', flat_tree.classes['Test'].symbols.keys())
        self.assertIn('elem.tc.a', flat_tree.classes['Test'].symbols.keys())
        self.assertIn('b',         flat_tree.classes['Test'].symbols.keys())

    def test_function_pull(self):
        with open(os.path.join(MODEL_DIR, 'FunctionPull.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        class_name = 'Level1.Level2.Level3.Function5'
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        # Check if all referenced functions are pulled in
        self.assertIn('Level1.Level2.Level3.f', flat_tree.classes)
        self.assertIn('Level1.Level2.Level3.TestPackage.times2', flat_tree.classes)
        self.assertIn('Level1.Level2.Level3.TestPackage.square', flat_tree.classes)
        self.assertNotIn('Level1.Level2.Level3.TestPackage.not_called', flat_tree.classes)

        # Check if the classes in the flattened tree have the right type
        self.assertEqual(flat_tree.classes['Function5'].type, 'model')

        self.assertEqual(flat_tree.classes['Level1.Level2.Level3.f'].type, 'function')
        self.assertEqual(flat_tree.classes['Level1.Level2.Level3.TestPackage.times2'].type, 'function')
        self.assertEqual(flat_tree.classes['Level1.Level2.Level3.TestPackage.square'].type, 'function')

        # Check whether input/output information of functions comes along properly
        func_t2 = flat_tree.classes['Level1.Level2.Level3.TestPackage.times2']
        self.assertIn("input", func_t2.symbols['x'].prefixes)
        self.assertIn("output", func_t2.symbols['y'].prefixes)

        # Check if built-in function call statement comes along properly
        func_f = flat_tree.classes['Level1.Level2.Level3.f']
        self.assertEqual(func_f.statements[0].right.operator, '*')
        # Check if user-specified function call statement comes along properly
        self.assertEqual(func_f.statements[0].right.operands[0].operator,
                         'Level1.Level2.Level3.TestPackage.times2')

    def test_nested_symbol_modification(self):
        with open(os.path.join(MODEL_DIR, 'NestedSymbolModification.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        class_name = 'E'
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        self.assertEqual(flat_tree.classes['E'].symbols['c.x'].nominal.value, 2.0)

    def test_redeclare_in_extends(self):
        with open(os.path.join(MODEL_DIR, 'RedeclareInExtends.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        class_name = 'ChannelZ'
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        self.assertIn('down.Z', flat_tree.classes['ChannelZ'].symbols)

    def test_redeclaration_scope(self):
        with open(os.path.join(MODEL_DIR, 'RedeclarationScope.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        class_name = 'ChannelZ'
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        self.assertIn('c.up.Z', flat_tree.classes['ChannelZ'].symbols)
        self.assertIn('c.down.A', flat_tree.classes['ChannelZ'].symbols)

    def test_extends_redeclareable(self):
        with open(os.path.join(MODEL_DIR, 'ExtendsRedeclareable.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        class_name = 'E'
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        self.assertIn('z.y', flat_tree.classes['E'].symbols)
        self.assertEqual(flat_tree.classes['E'].symbols['z.y'].nominal.value, 2.0)

    def test_redeclare_nested(self):
        with open(os.path.join(MODEL_DIR, 'RedeclareNestedClass.mo.fail_parse'), 'r') as f:
            txt = f.read()

        with self.assertRaises(Exception):
            ast_tree = parser.parse(txt)

    def test_extends_order(self):
        with open(os.path.join(MODEL_DIR, 'ExtendsOrder.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        class_name = 'P.M'
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        self.assertEqual(flat_tree.classes['M'].symbols['at.m'].value.value, 0.0)

    def test_constant_references(self):
        with open(os.path.join(MODEL_DIR, 'ConstantReferences.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        class_name = 'b'
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        self.assertEqual(flat_tree.classes['b'].symbols['m.p'].value.value, 2.0)
        self.assertEqual(flat_tree.classes['b'].symbols['M2.m.f'].value.value, 3.0)


    def test_parameter_modification_scope(self):
        with open(os.path.join(MODEL_DIR, 'ParameterScope.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        class_name = 'ScopeTest'
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        self.assertEqual(flat_tree.classes['ScopeTest'].symbols['nc.p'].value.name, 'p')

    def test_custom_units(self):
        with open(os.path.join(MODEL_DIR, 'CustomUnits.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        class_name = 'A'
        comp_ref = ast.ComponentRef.from_string(class_name)

        flat_tree = tree.flatten(ast_tree, comp_ref)

        self.assertEqual(flat_tree.classes['A'].symbols['dummy_parameter'].unit.value, "m/s")
        self.assertEqual(flat_tree.classes['A'].symbols['dummy_parameter'].value.value, 10.0)


if __name__ == "__main__":
    unittest.main()
