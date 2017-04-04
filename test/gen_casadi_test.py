#!/usr/bin/env python
"""
Modelica parse Tree to AST tree.
"""
from __future__ import print_function, absolute_import, division, print_function, unicode_literals

import os
import sys
import unittest
import time
import pylab as pl
from pymola import parser
from pymola import tree
from pymola import gen_casadi
import casadi as ca
import numpy as np
import jinja2


CasadiSysModel = gen_casadi.CasadiSysModel
TEST_DIR = os.path.dirname(os.path.realpath(__file__))

class GenCasadiTest(unittest.TestCase):
    "Testing"


    def assert_model_equivalent(self, A, B, tol=1e-9):
        self.assertEqual(len(A.states),len(B.states))
        self.assertEqual(len(A.der_states),len(B.der_states))
        self.assertEqual(len(A.inputs),len(B.inputs))
        self.assertEqual(len(A.outputs),len(B.outputs))
        self.assertEqual(len(A.constants),len(B.constants))
        self.assertEqual(len(A.parameters),len(B.parameters))
        self.assertEqual(len(A.equations),len(B.equations))

        this = A.get_function()
        that = B.get_function()

        args_in = []
        for i in range(this.n_in()):
            sp = this.sparsity_in(0)
            r = ca.DM(sp,np.random.random(sp.nnz()))
            args_in.append(r)

        this_out = this.call(args_in)
        that_out = that.call(args_in)

        for i, (a,b) in enumerate(zip(this_out,that_out)):
            test = float(ca.norm_2(a-b))<=tol
            if not test:
                print("Expr mismatch")
                print("A: ", A.equations[i], a)
                print("B: ", B.equations[i], b)
            self.assertTrue(test)

        return True

    def test_spring(self):
        with open(os.path.join(TEST_DIR, 'Spring.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'Spring')
        ref_model = CasadiSysModel()
        print(casadi_model)
        x = ca.MX.sym("x")
        v_x = ca.MX.sym("v_x")
        der_x = ca.MX.sym("der_x")
        der_v_x = ca.MX.sym("der_v_x")
        k = ca.MX.sym("k")
        c = ca.MX.sym("c")
        ref_model.states = [x,v_x]
        ref_model.der_states = [der_x,der_v_x]
        ref_model.parameters = [c,k]
        ref_model.equations =  [ der_x - v_x, der_v_x - (-k*x - c*v_x)]

        self.assert_model_equivalent(ref_model, casadi_model)

if __name__ == "__main__":
    unittest.main()
