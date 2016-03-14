#!/usr/bin/env python
"""
Modelica parse Tree to AST tree.
"""
from __future__ import print_function
import unittest
from pymola import parser
from pymola import tree
import os
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

        #walker.walk(ComponentRenameListener(prefix='blah'), ast_tree)
        #ast_tree.walk(ComponentRenameListener(prefix='blah'))
        print(ast_tree)
        #print(tree)
        #flat_tree = flatten(tree, 'Aircraft')
        #print(flat_tree)
        #compRenameVisitor = ComponentRenameVisitor()
        #visitor = TreeVisitor()
        #visitor.visit(compRenameVisitor, flat_tree)
        #listener = Listener()
        #tree.walk(listener)
        #print(flatten(tree, 'Aircraft', ''))
