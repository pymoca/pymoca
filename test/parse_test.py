#!/usr/bin/env python
"""
Modelica parse Tree to AST tree.
"""
from __future__ import print_function, absolute_import, division, print_function, unicode_literals

import os
import sys
import unittest
import time

from pymola import parser
from pymola import tree

TEST_DIR = os.path.dirname(os.path.realpath(__file__))

class ParseTest(unittest.TestCase):
    "Testing"

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def flush(self):
        sys.stdout.flush()
        sys.stdout.flush()
        time.sleep(0.01)

    def test_aircraft(self):
        with open(os.path.join(TEST_DIR, 'Aircraft.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        flat_tree = tree.flatten(ast_tree, 'Aircraft')
        self.flush()

    def test_bouncing_ball(self):
        with open(os.path.join(TEST_DIR, 'BouncingBall.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        flat_tree = tree.flatten(ast_tree, 'BouncingBall')
        self.flush()

    def test_estimator(self):
        with open(os.path.join(TEST_DIR, './Estimator.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        flat_tree = tree.flatten(ast_tree, 'Estimator')
        self.flush()

    def test_spring(self):
        with open(os.path.join(TEST_DIR, 'Spring.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        flat_tree = tree.flatten(ast_tree, 'Spring')
        self.flush()

    def test_duplicate_state(self):
        with open(os.path.join(TEST_DIR, 'DuplicateState.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        states = ast_tree.classes['DuplicateState'].states
        names = sorted([state.name for state in states])
        names_set = sorted(list(set(names)))
        if names != names_set:
            raise IOError('{:s} != {:s}'.format(str(names), str(names_set)))
        self.flush()

    def test_connector(self):
        with open(os.path.join(TEST_DIR, 'Connector.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        states = ast_tree.classes['Aircraft'].states
        names = sorted([state.name for state in states])
        names_set = sorted(list(set(names)))
        if names != names_set:
            raise IOError('{:s} != {:s}'.format(str(names), str(names_set)))
        self.flush()

if __name__ == "__main__":
    unittest.main()
