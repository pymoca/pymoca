#!/usr/bin/env python
"""
Modelica parse Tree to AST tree.
"""
from __future__ import print_function, absolute_import, division, print_function, unicode_literals
import unittest
from pymola import parser, tree, ast
import os
import json
import subprocess

TEST_DIR = os.path.dirname(os.path.realpath(__file__))


ast_tree = parser.parse(os.path.join(TEST_DIR, './Aircraft.mo'))

ast_walker = tree.TreeWalker()
ast_walker.walk(tree.ComponentRenameListener("blah"), ast_tree)

for cls_name, cls in ast_tree.classes.items():
    print('class', cls_name)
    for sym_name, sym in cls.symbols.items():
        print('\tsymbol', sym.name)

#print(json.dumps(ast_tree, indent=2, sort_keys=True))
#flat_tree = tree.flatten(ast_tree, 'Aircraft')
#print(flat_tree)