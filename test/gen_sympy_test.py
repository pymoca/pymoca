#!/usr/bin/env python
"""
Modelica parse Tree to AST tree.
"""
from __future__ import print_function, absolute_import, division, print_function, unicode_literals

import os
import sys
import time
import unittest

import pymoca.backends.sympy.generator as gen_sympy
from pymoca import parser
from pymoca import tree
from pymoca import ast

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(TEST_DIR, 'models')
GENERATED_DIR = os.path.join(TEST_DIR, 'generated')


class GenSympyTest(unittest.TestCase):
    """
    Sympy generation test
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @staticmethod
    def flush():
        sys.stdout.flush()
        sys.stdout.flush()
        time.sleep(0.01)

    def test_estimator(self):
        with open(os.path.join(MODEL_DIR, 'Estimator.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        text = gen_sympy.generate(ast_tree, 'Estimator')
        with open(os.path.join(GENERATED_DIR, 'Estimator.py'), 'w') as f:
            f.write(text)
        from test.generated.Estimator import Estimator as Estimator
        e = Estimator()
        e.linearize_symbolic()
        e.linearize()
        # noinspection PyUnusedLocal
        res = e.simulate(x0=[1.0])
        self.flush()

    def test_spring(self):
        with open(os.path.join(MODEL_DIR, 'SpringSystem.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name='SpringSystem'))
        print(flat_tree)
        text = gen_sympy.generate(ast_tree, 'SpringSystem')
        with open(os.path.join(GENERATED_DIR, 'Spring.py'), 'w') as f:
            f.write(text)
        from test.generated.Spring import SpringSystem as SpringSystem
        e = SpringSystem()
        e.linearize_symbolic()
        e.linearize()
        # noinspection PyUnusedLocal
        res = e.simulate(x0=[1.0, 1.0])
        self.flush()

    def test_aircraft(self):
        with open(os.path.join(MODEL_DIR, 'Aircraft.mo'), 'r') as f:
            txt = f.read()
        # noinspection PyUnusedLocal
        ast_tree = parser.parse(txt)
        text = gen_sympy.generate(ast_tree, 'Aircraft')
        with open(os.path.join(GENERATED_DIR, 'Aircraft.py'), 'w') as f:
            f.write(text)
        from test.generated.Aircraft import Aircraft as Aircraft
        e = Aircraft()
        e.linearize_symbolic()
        e.linearize()
        # noinspection PyUnusedLocal
        res = e.simulate()
        self.flush()

    def test_quad(self):
        with open(os.path.join(MODEL_DIR, 'Quad.mo'), 'r') as f:
            txt = f.read()
        # noinspection PyUnusedLocal
        ast_tree = parser.parse(txt)
        text = gen_sympy.generate(ast_tree, 'Quad')
        with open(os.path.join(GENERATED_DIR, 'Quad.py'), 'w') as f:
            f.write(text)
        from test.generated.Quad import Quad as Quad
        e = Quad()
        e.linearize_symbolic()
        e.linearize()
        # noinspection PyUnusedLocal
        res = e.simulate()
        self.flush()

    @unittest.skip
    def test_connector(self):
        with open(os.path.join(MODEL_DIR, 'Connector.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        # print(ast_tree)

        # noinspection PyUnusedLocal
        flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name='Aircraft'))
        # print(flat_tree)

        # noinspection PyUnusedLocal
        walker = tree.TreeWalker()
        # noinspection PyUnusedLocal
        classes = ast_tree.classes
        # noinspection PyUnusedLocal
        root = ast_tree.classes['Aircraft']

        # instantiator = tree.Instantiator(classes=classes)
        # walker.walk(instantiator, root)
        # print(instantiator.res[root].symbols.keys())
        # print(instantiator.res[root])

        # print('INSTANTIATOR\n-----------\n\n')
        # print(instantiator.res[root])

        # connectExpander = tree.ConnectExpander(classes=classes)
        # walker.walk(connectExpander, instantiator.res[root])

        # print('CONNECT EXPANDER\n-----------\n\n')
        # print(connectExpander.new_class)

        # text = gen_sympy.generate(ast_tree, 'Aircraft')
        # print(text)
        # with open(os.path.join(MODEL_DIR, 'generated/Connect.py'), 'w') as f:
        #    f.write(text)

        # from generated.Connect import Aircraft as Aircraft
        # e = Aircraft()
        # res = e.simulate()
        self.flush()


if __name__ == "__main__":
    unittest.main()
