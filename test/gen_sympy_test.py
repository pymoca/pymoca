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

TEST_DIR = os.path.dirname(os.path.realpath(__file__))

class GenSympyTest(unittest.TestCase):
    "Testing"

    def test_estimator(self):
        ast_tree = parser.parse(os.path.join(TEST_DIR, './Estimator.mo'))
        ast_walker = tree.TreeWalker()
        #flat_tree = tree.flatten(ast_tree, 'Estimator')
        flat_tree = ast_tree
        sympy_gen = gen_sympy.SympyGenerator()
        ast_walker.walk(sympy_gen, flat_tree)
        #print(flat_tree)
        print(sympy_gen.src[flat_tree])
        with open(os.path.join(TEST_DIR, 'generated/Estimator.py'), 'w') as f:
            f.write(sympy_gen.src[flat_tree])
        sys.stdout.flush()

    def test_aircraft(self):
        ast_tree = parser.parse(os.path.join(TEST_DIR, './Aircraft.mo'))
        ast_walker = tree.TreeWalker()
        flat_tree = tree.flatten(ast_tree, 'Aircraft')
        #sympy_gen = gen_sympy.SympyGenerator()
        #ast_walker.walk(sympy_gen, flat_tree)
        #print(flat_tree)
        #print(sympy_gen.src[flat_tree])
        #with open(os.path.join(TEST_DIR, 'generated/Aircraft.py'), 'w') as f:
        #    f.write(sympy_gen.src[flat_tree])
        #sys.stdout.flush()

if __name__ == "__main__":
    unittest.main()