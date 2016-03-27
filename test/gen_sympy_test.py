#!/usr/bin/env python
"""
Modelica parse Tree to AST tree.
"""
from __future__ import print_function, absolute_import, division, print_function, unicode_literals

import os
import sys
import unittest

from pymola import parser
from pymola import tree
from pymola import gen_sympy

import jinja2

TEST_DIR = os.path.dirname(os.path.realpath(__file__))

class GenSympyTest(unittest.TestCase):
    "Testing"

    def setUp(self):
        sys.stdout.flush()
        sys.stderr.flush()

    def tearDown(self):
        sys.stdout.flush()
        sys.stderr.flush()

    def test_estimator(self):
        sys.stdout.flush()
        sys.stderr.flush()
        ast_tree = parser.parse(open(os.path.join(TEST_DIR, './Estimator.mo'), 'r').read())
        text = gen_sympy.generate(ast_tree, 'Estimator')
        with open(os.path.join(TEST_DIR, 'generated/Estimator.py'), 'w') as f:
           f.write(text)
        from generated.Estimator import Estimator as Estimator
        e = Estimator()
        res = e.simulate(x0=[1])
        sys.stdout.flush()
        sys.stderr.flush()

    def test_spring(self):
        sys.stdout.flush()
        sys.stderr.flush()
        ast_tree = parser.parse(open(os.path.join(TEST_DIR, './Spring.mo'), 'r').read())
        text = gen_sympy.generate(ast_tree, 'Spring')
        with open(os.path.join(TEST_DIR, 'generated/Spring.py'), 'w') as f:
           f.write(text)
        from generated.Spring import Spring as Spring
        e = Spring()
        res = e.simulate(x0=[1, 1])
        sys.stdout.flush()
        sys.stderr.flush()

    def test_aircraft(self):
        sys.stdout.flush()
        sys.stderr.flush()
        ast_tree = parser.parse(open(os.path.join(TEST_DIR, './Aircraft.mo'), 'r').read())
        text = gen_sympy.generate(ast_tree, 'Aircraft')
        #with open(os.path.join(TEST_DIR, 'generated/Aircraft.py'), 'w') as f:
        #   f.write(text)
        #from generated.Aircraft import Aircraft as Aircraft
        #e = Aircraft()
        #res = e.simulate()
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    unittest.main()