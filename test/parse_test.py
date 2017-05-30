#!/usr/bin/env python
"""
Modelica parse Tree to AST tree.
"""
from __future__ import print_function, absolute_import, division, print_function, unicode_literals

import os
import sys
import time
import unittest

from pymola import parser
from pymola import tree

TEST_DIR = os.path.dirname(os.path.realpath(__file__))


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
        with open(os.path.join(TEST_DIR, 'Aircraft.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        print('AST TREE\n', ast_tree)
        flat_tree = tree.flatten(ast_tree, 'Aircraft')
        print('AST TREE FLAT\n', flat_tree)
        self.flush()

    @unittest.skip
    def test_bouncing_ball(self):
        with open(os.path.join(TEST_DIR, 'BouncingBall.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        print('AST TREE\n', ast_tree)
        flat_tree = tree.flatten(ast_tree, 'BouncingBall')
        print(flat_tree)
        print('AST TREE FLAT\n', flat_tree)
        self.flush()

    def test_estimator(self):
        with open(os.path.join(TEST_DIR, './Estimator.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        print('AST TREE\n', ast_tree)
        flat_tree = tree.flatten(ast_tree, 'Estimator')
        print('AST TREE FLAT\n', flat_tree)
        self.flush()

    def test_spring(self):
        with open(os.path.join(TEST_DIR, 'Spring.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        print('AST TREE\n', ast_tree)
        flat_tree = tree.flatten(ast_tree, 'Spring')
        print('AST TREE FLAT\n', flat_tree)
        self.flush()

    def test_duplicate_state(self):
        with open(os.path.join(TEST_DIR, 'DuplicateState.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        print('AST TREE\n', ast_tree)
        flat_tree = tree.flatten(ast_tree, 'DuplicateState')
        print('AST TREE FLAT\n', flat_tree)
        self.flush()

    def test_connector(self):
        with open(os.path.join(TEST_DIR, 'Connector.mo'), 'r') as f:
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
        with open(os.path.join(TEST_DIR, 'InheritanceInstantiation.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        flat_tree = tree.flatten(ast_tree, 'C2')

        self.assertEqual(flat_tree.classes['C2'].symbols['bcomp.b'].value.value, 3.0)

    # TODO: Currently nested (local) classes are not parsed correctly. Not
    # entirely sure if their scope is local, or other (non-extending) classes
    # can make instantiations as well. That will likely influence the way they
    # need to be stored.
    @unittest.skip
    def test_inheritance(self):
        with open(os.path.join(TEST_DIR, 'InheritanceInstantiation.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        flat_tree = tree.flatten(ast_tree, 'C2')

        self.assertEqual(flat_tree.classes['C2'].symbols['bcomp.b'].value.value, 3.0)

    def test_inheritance_symbol_modifiers(self):
        with open(os.path.join(TEST_DIR, 'Inheritance.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        flat_tree = tree.flatten(ast_tree, 'Sub')

        self.assertEqual(flat_tree.classes['Sub'].symbols['x'].max.value, 30.0)

if __name__ == "__main__":
    unittest.main()
