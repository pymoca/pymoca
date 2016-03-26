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
        sys.stdout.flush()

    def test_aircraft(self):
        ast_tree = parser.parse(open(os.path.join(TEST_DIR, './Aircraft.mo'), 'r').read())

        #ast_walker = tree.TreeWalker()
        #ast_walker.walk(tree.ComponentRenameListener("blah"), ast_tree)

        #for cls_name, cls in ast_tree.classes.items():
        #    print('class', cls_name)
        #    for sym_name, sym in cls.symbols.items():
        #        print('\tsymbol', sym.name)

        print(ast_tree)
        flat_tree = tree.flatten(ast_tree, 'Aircraft')
        print(flat_tree)
        time.sleep(1)
        sys.stdout.flush()

    def test_bouncing_ball(self):
        ast_tree = parser.parse(open(os.path.join(TEST_DIR, './BouncingBall.mo'), 'r').read())

        #ast_walker = tree.TreeWalker()
        #ast_walker.walk(tree.ComponentRenameListener("blah"), ast_tree)

        #for cls_name, cls in ast_tree.classes.items():
        #    print('class', cls_name)
        #    for sym_name, sym in cls.symbols.items():
        #        print('\tsymbol', sym.name)#

        print(ast_tree)
        flat_tree = tree.flatten(ast_tree, 'BouncingBall')
        print(flat_tree)
        sys.stdout.flush()


    def test_estimator(self):
        ast_tree = parser.parse(open(os.path.join(TEST_DIR, './Estimator.mo'), 'r').read())

        ast_walker = tree.TreeWalker()
        ast_walker.walk(tree.ComponentRenameListener("blah"), ast_tree)

        for cls_name, cls in ast_tree.classes.items():
            print('class', cls_name)
            for sym_name, sym in cls.symbols.items():
                print('\tsymbol', sym.name)

        print(ast_tree)
        flat_tree = tree.flatten(ast_tree, 'Estimator')
        print(flat_tree)
        sys.stdout.flush()



if __name__ == "__main__":
    unittest.main()