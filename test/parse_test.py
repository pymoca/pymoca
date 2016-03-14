#!/usr/bin/env python
"""
Modelica parse Tree to AST tree.
"""
from __future__ import print_function, absolute_import, division, print_function, unicode_literals
import unittest
from pymola import parser
from pymola import tree
import os
import time
import json
import subprocess

TEST_DIR = os.path.dirname(os.path.realpath(__file__))

class Test(unittest.TestCase):
    "Testing"

    def test_aircraft(self):
        ast_tree = parser.parse(os.path.join(TEST_DIR, './Aircraft.mo'))

        ast_walker = tree.TreeWalker()
        ast_walker.walk(tree.ComponentRenameListener("blah"), ast_tree)

        for cls_name, cls in ast_tree.classes.items():
            print('class', cls_name)
            for sym_name, sym in cls.symbols.items():
                print('\tsymbol', sym.name)

        print(ast_tree)
        flat_tree = tree.flatten(ast_tree, 'Aircraft')
        print(flat_tree)

        # we are printing a lot, need to let it finishing outputting before test finishes
        time.sleep(1)

    def test_bouncing_ball(self):
        ast_tree = parser.parse(os.path.join(TEST_DIR, './BouncingBall.mo'))

        ast_walker = tree.TreeWalker()
        ast_walker.walk(tree.ComponentRenameListener("blah"), ast_tree)

        for cls_name, cls in ast_tree.classes.items():
            print('class', cls_name)
            for sym_name, sym in cls.symbols.items():
                print('\tsymbol', sym.name)

        print(ast_tree)
        flat_tree = tree.flatten(ast_tree, 'BouncingBall')
        print(flat_tree)

        # we are printing a lot, need to let it finishing outputting before test finishes
        time.sleep(1)

    def test_estimator(self):
        ast_tree = parser.parse(os.path.join(TEST_DIR, './Estimator.mo'))

        ast_walker = tree.TreeWalker()
        ast_walker.walk(tree.ComponentRenameListener("blah"), ast_tree)

        for cls_name, cls in ast_tree.classes.items():
            print('class', cls_name)
            for sym_name, sym in cls.symbols.items():
                print('\tsymbol', sym.name)

        print(ast_tree)
        flat_tree = tree.flatten(ast_tree, 'Estimator')
        print(flat_tree)

        # we are printing a lot, need to let it finishing outputting before test finishes
        time.sleep(1)


if __name__ == "__main__":
    unittest.main()