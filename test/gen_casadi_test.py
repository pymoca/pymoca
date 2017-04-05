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

        this_mx = this.mx_in()
        that_mx = that.mx_in()
        this_in = [e.name() for e in this_mx]
        that_in = [e.name() for e in that_mx]

        that_from_this = []
        this_mx_dict = dict(zip(this_in,this_mx))
        for e in that_in:
            self.assertTrue(e in this_in)
            that_from_this.append(this_mx_dict[e])
        that = ca.Function('f',this_mx,that.call(that_from_this))

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

    def test_estimator(self):
        with open(os.path.join(TEST_DIR, 'Estimator.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'Estimator')
        ref_model = CasadiSysModel()
        print(casadi_model)

        x = ca.MX.sym("x")
        der_x = ca.MX.sym("der_x")
        y = ca.MX.sym("y")

        ref_model.states = [x]
        ref_model.der_states = [der_x]
        ref_model.outputs = [y]
        ref_model.equations =  [ der_x + x, y-x]

        self.assert_model_equivalent(ref_model, casadi_model)

    def test_aircraft(self):
        with open(os.path.join(TEST_DIR, 'Aircraft.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'Aircraft')
        ref_model = CasadiSysModel()

    def test_connector(self):
        with open(os.path.join(TEST_DIR, 'ConnectorHQ.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'System')
        ref_model = CasadiSysModel()
        print(casadi_model)

        a__up__H = ca.MX.sym("a__up__H")
        a__up__Q = ca.MX.sym("a__up__Q")
        a__down__H = ca.MX.sym("a__down__H")
        a__down__Q = ca.MX.sym("a__down__Q")

        b__up__H = ca.MX.sym("b__up__H")
        b__up__Q = ca.MX.sym("b__up__Q")
        b__down__H = ca.MX.sym("b__down__H")
        b__down__Q = ca.MX.sym("b__down__Q")

        c__up__H = ca.MX.sym("c__up__H")
        c__up__Q = ca.MX.sym("c__up__Q")
        c__down__H = ca.MX.sym("c__down__H")
        c__down__Q = ca.MX.sym("c__down__Q")

        qa__down__H = ca.MX.sym("qa__down__H")
        qa__down__Q = ca.MX.sym("qa__down__Q")
        qc__down__H = ca.MX.sym("qc__down__H")
        qc__down__Q = ca.MX.sym("qc__down__Q")

        hb__up__H = ca.MX.sym("hb__up__H")
        hb__up__Q = ca.MX.sym("hb__up__Q")

        ref_model.alg_states = [qc__down__H, a__down__H, b__down__H, c__down__H, c__up__H, hb__up__H, a__up__H, b__up__H, qa__down__H, a__up__Q, qa__down__Q, c__down__Q, hb__up__Q, c__up__Q, b__up__Q, b__down__Q, qc__down__Q, a__down__Q]

        ref_model.equations = [ a__up__H-a__down__H,
              a__up__Q+a__down__Q,
	            c__up__H-c__down__H,
              c__up__Q+c__down__Q,

              b__up__H-b__down__H,
              b__up__Q+b__down__Q,


              qa__down__Q,
              qc__down__Q,

              hb__up__H,

              qa__down__H-a__up__H,
              qc__down__H-c__up__H,
              a__down__H-b__up__H,
              c__down__H-b__up__H,
              b__down__H-hb__up__H,

              c__down__Q+a__down__Q+b__up__Q,
              qc__down__Q+c__up__Q,
              b__down__Q+hb__up__Q,
              a__up__Q+qa__down__Q]



        print(ref_model)
        self.assert_model_equivalent(ref_model, casadi_model)

    def test_duplicate(self):
        with open(os.path.join(TEST_DIR, 'DuplicateState.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'DuplicateState')
        print(casadi_model)
        ref_model = CasadiSysModel()

        x = ca.MX.sym("x")
        der_x = ca.MX.sym("der_x")
        y = ca.MX.sym("y")
        der_y = ca.MX.sym("der_y")

        ref_model.states = [x,y]
        ref_model.der_states = [der_x, der_y]
        ref_model.equations =  [ der_x+der_y-1, der_x-2]

        self.assert_model_equivalent(ref_model, casadi_model)

    def test_inheritance(self):
        with open(os.path.join(TEST_DIR, 'Inheritance.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'Sub')
        print("inheritance",casadi_model)
        ref_model = CasadiSysModel()

        x = ca.MX.sym("x")
        der_x = ca.MX.sym("der_x")
        y = ca.MX.sym("y")
        der_y = ca.MX.sym("y")
        k = ca.MX.sym("k")

        ref_model.states = [x]
        ref_model.der_states = [der_x]
        ref_model.alg_states = [y]
        ref_model.parameters = [k]
        ref_model.equations =  [ der_x-k*x, x+y-3]

        self.assert_model_equivalent(ref_model, casadi_model)

    def test_builtin(self):
        with open(os.path.join(TEST_DIR, 'BuiltinFunctions.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'BuiltinFunctions')
        print("BuiltinFunctions",casadi_model)
        ref_model = CasadiSysModel()

        x = ca.MX.sym("x")
        y = ca.MX.sym("y")
        z = ca.MX.sym("z")
        w = ca.MX.sym("w")
        u = ca.MX.sym("u")
        time = ca.MX.sym("time")

        ref_model.inputs = [x]
        ref_model.time = time
        ref_model.outputs = [y, z, w, u]
        ref_model.equations =  [ y-ca.sin(time), z-ca.cos(x),w-ca.fmin(y,z), u-ca.fabs(w)]

        self.assert_model_equivalent(ref_model, casadi_model)

    def test_forloop(self):
        with open(os.path.join(TEST_DIR, 'ForLoop.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'ForLoop')
        print(casadi_model)
        ref_model = CasadiSysModel()

        x = ca.MX.sym("x",10)

        ref_model.alg_states = [x]
        ref_model.equations =  [ x-range(1,11)]

        self.assert_model_equivalent(ref_model, casadi_model)

    def test_forloop(self):
        with open(os.path.join(TEST_DIR, 'ArrayExpressions.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'ArrayExpressions')
        print(casadi_model)
        ref_model = CasadiSysModel()

        a = ca.MX.sym("a", 3)
        b = ca.MX.sym("b", 4)
        c = ca.MX.sym("c", 3)
        d = ca.MX.sym("d", 3)
        e = ca.MX.sym("e", 3)

        scalar_f = ca.MX.sym("scalar_f")

        ref_model.alg_states = [a,b,c,d,e,scalar_f]
        ref_model.equations =  [ c-(a+b[0:3]), d-(ca.sin(a/b[1:4])), e - (d+scalar_f)]

        self.assert_model_equivalent(ref_model, casadi_model)

if __name__ == "__main__":
    unittest.main()
