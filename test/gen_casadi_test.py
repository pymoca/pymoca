#!/usr/bin/env python
"""
Modelica parse Tree to AST tree.
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import os
import glob
import unittest
import itertools

import casadi as ca
import numpy as np

import pymoca.backends.casadi.generator as gen_casadi
from pymoca.backends.casadi.alias_relation import AliasRelation
from pymoca.backends.casadi.model import CASADI_ATTRIBUTES, Model, Variable, DelayArgument
from pymoca.backends.casadi.api import transfer_model, CachedModel
from pymoca import parser, ast


MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')


# noinspection PyPep8Naming,PyUnresolvedReferences
class GenCasadiTest(unittest.TestCase):
    def assert_model_equivalent(self, A, B):
        def sstr(a): return set([str(e) for e in a])

        for l in ["alg_states", "states", "der_states", "inputs", "outputs", "constants",
                  "parameters"]:
            self.assertEqual(sstr(getattr(A, l)), sstr(getattr(B, l)))

    def assert_model_equivalent_numeric(self, A, B, tol=1e-9):
        self.assertEqual(len(A.states), len(B.states))
        self.assertEqual(len(A.der_states), len(B.der_states))
        self.assertEqual(len(A.alg_states), len(B.alg_states))
        self.assertEqual(len(A.inputs), len(B.inputs))
        self.assertEqual(len(A.outputs), len(B.outputs))
        self.assertEqual(len(A.constants), len(B.constants))
        self.assertEqual(len(A.parameters), len(B.parameters))
        self.assertEqual(len(A.delay_states), len(B.delay_states))

        if not isinstance(A, CachedModel) and not isinstance(B, CachedModel):
            self.assertEqual(len(A.equations), len(B.equations))
            self.assertEqual(len(A.initial_equations), len(B.initial_equations))
            self.assertEqual(len(A.delay_arguments), len(B.delay_arguments))

        for f_name in ['dae_residual', 'initial_residual', 'variable_metadata', 'delay_arguments']:
            this = getattr(A, f_name + '_function')
            that = getattr(B, f_name + '_function')

            np.random.seed(0)

            args_in = []
            for i in range(this.n_in()):
                sp = this.sparsity_in(i)
                r = ca.DM(sp, np.random.random(sp.nnz()))
                args_in.append(r)

            this_out = this.call(args_in)
            that_out = that.call(args_in)

            # N.B. Here we require that the order of the equations in the two models is identical.
            for i, (a, b) in enumerate(zip(this_out, that_out)):
                for j in range(a.size1()):
                    for k in range(a.size2()):
                        if a[j, k].is_regular() or b[j, k].is_regular():
                            test = float(ca.norm_2(ca.vec(a[j, k] - b[j, k]))) <= tol
                            if not test:
                                print(j)
                                print(k)
                                print(a[j,k])
                                print(b[j,k])
                                print(f_name)
                            self.assertTrue(test)

        return True

    def test_spring(self):
        with open(os.path.join(MODEL_DIR, 'Spring.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'Spring')
        ref_model = Model()
        print(casadi_model)
        x = ca.MX.sym("x")
        v_x = ca.MX.sym("v_x")
        der_x = ca.MX.sym("der(x)")
        der_v_x = ca.MX.sym("der(v_x)")
        k = ca.MX.sym("k")
        c = ca.MX.sym("c")
        ref_model.states = list(map(Variable, [x, v_x]))
        ref_model.der_states = list(map(Variable, [der_x, der_v_x]))
        ref_model.parameters = list(map(Variable, [c, k]))
        ref_model.parameters[0].value = 0.1
        ref_model.parameters[1].value = 2
        ref_model.equations = [der_x - v_x, der_v_x - (-k * x - c * v_x)]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_estimator(self):
        with open(os.path.join(MODEL_DIR, 'Estimator.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'Estimator')
        ref_model = Model()
        print(casadi_model)

        x = ca.MX.sym("x")
        der_x = ca.MX.sym("der(x)")
        y = ca.MX.sym("y")

        ref_model.states = list(map(Variable, [x]))
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [y]))
        ref_model.outputs = list(map(Variable, [y]))
        ref_model.equations = [der_x + x, y - x]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_aircraft(self):
        with open(os.path.join(MODEL_DIR, 'Aircraft.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        # noinspection PyUnusedLocal
        casadi_model = gen_casadi.generate(ast_tree, 'Aircraft')
        # noinspection PyUnusedLocal
        ref_model = Model()
        self.assertTrue(True)

    def test_connector_hq(self):
        with open(os.path.join(MODEL_DIR, 'ConnectorHQ.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'System')
        ref_model = Model()
        print(casadi_model)

        a__up__H = ca.MX.sym("a.up.H")
        a__up__Q = ca.MX.sym("a.up.Q")
        a__down__H = ca.MX.sym("a.down.H")
        a__down__Q = ca.MX.sym("a.down.Q")

        b__up__H = ca.MX.sym("b.up.H")
        b__up__Q = ca.MX.sym("b.up.Q")
        b__down__H = ca.MX.sym("b.down.H")
        b__down__Q = ca.MX.sym("b.down.Q")

        c__up__H = ca.MX.sym("c.up.H")
        c__up__Q = ca.MX.sym("c.up.Q")
        c__down__H = ca.MX.sym("c.down.H")
        c__down__Q = ca.MX.sym("c.down.Q")

        qa__down__H = ca.MX.sym("qa.down.H")
        qa__down__Q = ca.MX.sym("qa.down.Q")

        p__H = ca.MX.sym("p.H")
        p__Q = ca.MX.sym("p.Q")

        hb__up__H = ca.MX.sym("hb.up.H")
        hb__up__Q = ca.MX.sym("hb.up.Q")

        zerotest__H = ca.MX.sym("zerotest.H")
        zerotest__Q = ca.MX.sym("zerotest.Q")

        ref_model.alg_states = list(map(Variable, [
            a__up__H, a__down__H, b__up__H, b__down__H, c__up__H, c__down__H,
            qa__down__H, p__H, hb__up__H, zerotest__H,
            a__up__Q, a__down__Q, b__up__Q, b__down__Q, c__up__Q, c__down__Q,
            qa__down__Q, p__Q, hb__up__Q, zerotest__Q]))
        ref_model.equations = [a__up__H - a__down__H,
                               b__up__H - b__down__H,
                               c__up__H - c__down__H,
                               qa__down__Q,
                               hb__up__H,
                               p__Q,
                               qa__down__H - a__up__H,
                               p__H - c__up__H,
                               a__down__H - b__up__H,
                               c__down__H - b__up__H,
                               b__down__H - hb__up__H,
                               a__up__Q + a__down__Q,
                               b__up__Q + b__down__Q,
                               c__up__Q + c__down__Q,
                               qa__down__Q + a__up__Q,
                               -p__Q + c__up__Q,
                               a__down__Q + (b__up__Q + c__down__Q),
                               b__down__Q + hb__up__Q,
                               zerotest__Q]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_connector_hqz(self):
        with open(os.path.join(MODEL_DIR, 'ConnectorHQZ.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'SystemZ')
        ref_model = Model()
        print(casadi_model)

        a__up__H = ca.MX.sym("a.up.H")
        a__up__Q = ca.MX.sym("a.up.Q")
        a__up__Z = ca.MX.sym("a.up.Z")
        a__down__H = ca.MX.sym("a.down.H")
        a__down__Q = ca.MX.sym("a.down.Q")
        a__down__Z = ca.MX.sym("a.down.Z")

        b__up__H = ca.MX.sym("b.up.H")
        b__up__Q = ca.MX.sym("b.up.Q")
        b__up__Z = ca.MX.sym("b.up.Z")
        b__down__H = ca.MX.sym("b.down.H")
        b__down__Q = ca.MX.sym("b.down.Q")
        b__down__Z = ca.MX.sym("b.down.Z")

        c__up__H = ca.MX.sym("c.up.H")
        c__up__Q = ca.MX.sym("c.up.Q")
        c__up__Z = ca.MX.sym("c.up.Z")
        c__down__H = ca.MX.sym("c.down.H")
        c__down__Q = ca.MX.sym("c.down.Q")
        c__down__Z = ca.MX.sym("c.down.Z")

        d__up__H = ca.MX.sym("d.up.H")
        d__up__Q = ca.MX.sym("d.up.Q")
        d__down__H = ca.MX.sym("d.down.H")
        d__down__Q = ca.MX.sym("d.down.Q")
        d__down__Z = ca.MX.sym("d.down.Z")

        qa__down__H = ca.MX.sym("qa.down.H")
        qa__down__Q = ca.MX.sym("qa.down.Q")
        qa__down__Z = ca.MX.sym("qa.down.Z")
        qc__down__H = ca.MX.sym("qc.down.H")
        qc__down__Q = ca.MX.sym("qc.down.Q")
        qc__down__Z = ca.MX.sym("qc.down.Z")

        hb__up__H = ca.MX.sym("hb.up.H")
        hb__up__Q = ca.MX.sym("hb.up.Q")
        hb__up__Z = ca.MX.sym("hb.up.Z")

        ref_model.alg_states = list(map(Variable, [
            a__up__H, a__down__H, b__up__H, b__down__H,
            c__up__H, c__down__H, d__up__H, d__down__H,
            qa__down__H, qc__down__H, hb__up__H,
            a__up__Q, a__down__Q, b__up__Q, b__down__Q,
            c__up__Q, c__down__Q, d__up__Q, d__down__Q,
            qa__down__Q, qc__down__Q, hb__up__Q,
            a__up__Z, a__down__Z, b__up__Z, b__down__Z,
            c__up__Z, c__down__Z, d__down__Z,
            qa__down__Z, qc__down__Z, hb__up__Z]))

        ref_model.equations = [a__up__H - a__down__H,
                               a__up__Q + a__down__Q,
                               b__up__H - b__down__H,
                               b__up__Q + b__down__Q,
                               c__up__H - c__down__H,
                               c__up__Q + c__down__Q,
                               d__up__H - d__down__H,
                               d__up__Q + d__down__Q,
                               qa__down__Q,
                               qc__down__Q,
                               hb__up__H,
                               qa__down__H - a__up__H,
                               qa__down__Z - a__up__Z,
                               qc__down__H - c__up__H,
                               qc__down__Z - c__up__Z,
                               a__down__H - b__up__H,
                               a__down__Z - b__up__Z,
                               c__down__H - b__up__H,
                               c__down__Z - b__up__Z,
                               b__down__H - hb__up__H,
                               b__down__Z - hb__up__Z,
                               qa__down__Q + a__up__Q,
                               qc__down__Q + c__up__Q,
                               a__down__Q + (b__up__Q + c__down__Q),
                               b__down__Q + hb__up__Q,
                               d__up__Q,
                               d__down__Q]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_tree_lookup(self):
        with open(os.path.join(MODEL_DIR, 'TreeLookup.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        casadi_model = gen_casadi.generate(ast_tree, 'Level1.Level2.Level3.Test')
        ref_model = Model()
        print(casadi_model)

        elem__tc__i = ca.MX.sym("elem.tc.i")
        elem__tc__a = ca.MX.sym("elem.tc.a")
        b = ca.MX.sym("b")

        ref_model.alg_states = list(map(Variable, [elem__tc__i, elem__tc__a, b]))

        ref_model.equations = [elem__tc__i - 1,
                               elem__tc__a - b]

        print(ref_model)
        self.assert_model_equivalent(ref_model, casadi_model)

    def test_duplicate(self):
        with open(os.path.join(MODEL_DIR, 'DuplicateState.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'DuplicateState')
        print(casadi_model)
        ref_model = Model()

        x = ca.MX.sym("x")
        der_x = ca.MX.sym("der(x)")
        y = ca.MX.sym("y")
        der_y = ca.MX.sym("der(y)")

        ref_model.states = list(map(Variable, [x, y]))
        ref_model.der_states = list(map(Variable, [der_x, der_y]))
        ref_model.equations = [der_x + der_y - 1, der_x - 2]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_if_else(self):
        with open(os.path.join(MODEL_DIR, 'IfElse.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'IfElse')
        ref_model = Model()

        x = ca.MX.sym("x")
        y1 = ca.MX.sym("y1")
        y2 = ca.MX.sym("y2")
        y3 = ca.MX.sym("y3")
        y_max = ca.MX.sym("y_max")

        ref_model.inputs = list(map(Variable, [x]))
        ref_model.outputs = list(map(Variable, [y1, y2, y3]))
        ref_model.alg_states = list(map(Variable, [y1, y2, y3]))
        ref_model.parameters = list(map(Variable, [y_max]))
        ref_model.parameters[0].value = 10
        ref_model.equations = [
            y1 - ca.if_else(x > 0, 1, 0) * y_max,
            ca.if_else(x > 1, ca.vertcat(y2 - y_max, y3 - 100),
                       ca.if_else(x > 2, ca.vertcat(y2 - y_max - 1, y3 - 1000),
                                  ca.vertcat(y2, y3 - 10000)))]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_inheritance(self):
        with open(os.path.join(MODEL_DIR, 'Inheritance.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'Sub')
        ref_model = Model()

        x = ca.MX.sym("x")
        der_x = ca.MX.sym("der(x)")
        y = ca.MX.sym("y")
        # noinspection PyUnusedLocal
        der_y = ca.MX.sym("y")
        k = ca.MX.sym("k")

        ref_model.states = list(map(Variable, [x]))
        ref_model.states[0].max = 30.0
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [y]))
        ref_model.parameters = list(map(Variable, [k]))
        ref_model.parameters[0].value = -1.0
        ref_model.equations = [der_x - k * x, x + y - 3]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_inheritance_instantiation(self):
        with open(os.path.join(MODEL_DIR, 'InheritanceInstantiation.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'C2')
        ref_model = Model()
        print(casadi_model)

        bcomp1_a = ca.MX.sym('bcomp1.a')
        bcomp1_b = ca.MX.sym('bcomp1.b')
        bcomp2_a = ca.MX.sym('bcomp2.a')
        bcomp2_b = ca.MX.sym('bcomp2.b')
        bcomp3_a = ca.MX.sym('bcomp3.a')
        bcomp3_b = ca.MX.sym('bcomp3.b')

        bcomp1_v = ca.MX.sym('bcomp1.v', 3)
        bcomp2_v = ca.MX.sym('bcomp2.v', 4)
        bcomp3_v = ca.MX.sym('bcomp3.v', 2)

        ref_model.states = []
        ref_model.der_states = []
        ref_model.alg_states = list(map(Variable, [bcomp1_v, bcomp2_v, bcomp3_v]))
        ref_model.parameters = list(map(Variable, [bcomp1_a, bcomp2_a, bcomp3_a, bcomp1_b, bcomp2_b, bcomp3_b]))
        ref_model.parameters[0].value = 0
        ref_model.parameters[1].value = 0
        ref_model.parameters[2].value = 1
        ref_model.parameters[3].value = 3
        ref_model.parameters[4].value = 4
        ref_model.parameters[5].value = 2
        ref_model.equations = []

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_nested_classes(self):
        with open(os.path.join(MODEL_DIR, 'NestedClasses.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'C2')
        ref_model = Model()
        print(casadi_model)

        v1 = ca.MX.sym('v1')
        v2 = ca.MX.sym('v2')

        ref_model.states = []
        ref_model.der_states = []
        ref_model.alg_states = list(map(Variable, [v1, v2]))
        ref_model.equations = []
        ref_model.alg_states[0].nominal = 1000.0
        ref_model.alg_states[1].nominal = 1000.0

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_builtin(self):
        with open(os.path.join(MODEL_DIR, 'BuiltinFunctions.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'BuiltinFunctions')
        print("BuiltinFunctions", casadi_model)
        ref_model = Model()

        x = ca.MX.sym("x")
        y = ca.MX.sym("y")
        z = ca.MX.sym("z")
        w = ca.MX.sym("w")
        u = ca.MX.sym("u")

        ref_model.inputs = list(map(Variable, [x]))
        ref_model.outputs = list(map(Variable, [y, z, w, u]))
        ref_model.alg_states = list(map(Variable, [y, z, w, u]))
        ref_model.equations = [y - ca.sin(ref_model.time), z - ca.cos(x), w - ca.fmin(y, z), u - ca.fabs(w)]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_function_call(self):
        with open(os.path.join(MODEL_DIR, 'FunctionCall.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'FunctionCall')
        print("FunctionCall", casadi_model)
        ref_model = Model()

        radius = ca.MX.sym('radius')
        diameter = radius * 2
        circle_properties = ca.Function('circle_properties', [radius], [3.14159*diameter, 3.14159*radius**2, ca.if_else(3.14159*radius**2 > 10, 1, 2), ca.if_else(3.14159*radius**2 > 10, 10, 3.14159*radius**2), 8, 3, 12])

        c = ca.MX.sym("c")
        a = ca.MX.sym("a")
        d = ca.MX.sym("d")
        e = ca.MX.sym("e")
        S1 = ca.MX.sym("S1")
        S2 = ca.MX.sym("S2")
        r = ca.MX.sym("r")
        ref_model.alg_states = list(map(Variable, [r, c, a, d, e, S1, S2]))
        ref_model.outputs = list(map(Variable, [c, a, d, e, S1, S2]))
        ref_model.equations = [ca.vertcat(c, a, d, e, S1, S2) - ca.vertcat(*circle_properties.call([r])[0:-1])]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    @unittest.skip
    def test_double_function_call(self):
        with open(os.path.join(MODEL_DIR, 'DoubleFunctionCall.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'FunctionCall')
        print("FunctionCall", casadi_model)
        ref_model = Model()

        # Check that both instances of the function call refer to the same node in the tree
        func_a = casadi_model.equations[0].dep(1).dep(0).dep(0).dep(0).getFunction().__hash__()
        func_b = casadi_model.equations[1].dep(1).dep(0).dep(0).dep(0).getFunction().__hash__()

        self.assertEqual(func_a, func_b)

    def test_forloop(self):
        with open(os.path.join(MODEL_DIR, 'ForLoop.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'ForLoop')
        print(casadi_model)
        ref_model = Model()

        x = ca.MX.sym("x", 10)
        y = ca.MX.sym("y", 10)
        z = ca.MX.sym("z", 10)
        u = ca.MX.sym('u', 10, 2)
        v = ca.MX.sym('v', 2, 10)
        w = ca.MX.sym('w', 2, 10)
        b = ca.MX.sym("b")
        n = ca.MX.sym("n")
        s = ca.MX.sym('s', 10)
        Arr = ca.MX.sym('Arr', 2, 2)
        der_s = ca.MX.sym('der(s)', 10)

        ref_model.states = list(map(Variable, [s]))
        ref_model.der_states = list(map(Variable, [der_s]))
        ref_model.alg_states = list(map(Variable, [x, y, z, u, v, w, b, Arr]))
        ref_model.parameters = list(map(Variable, [n]))
        ref_model.parameters[0].value = 10
        ref_model.equations = [
            ca.horzcat(x - (np.arange(1, 11) + b), w[0, :].T - np.arange(1, 11), w[1, :].T - np.arange(2, 21, 2), u - np.ones((10, 2)), v.T - np.ones((10, 2))),
            y[0:5] - np.zeros(5), y[5:] - np.ones(5),
            ca.horzcat(z[0:5] - np.array([2, 2, 2, 2, 2]), z[5:10] - np.array([1, 1, 1, 1, 1])), der_s - np.ones(10),
            ca.horzcat(Arr[:, 1], Arr[:, 0]) - np.array([[2, 1], [2, 1]])]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_arrayexpressions(self):
        with open(os.path.join(MODEL_DIR, 'ArrayExpressions.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'ArrayExpressions')
        print(casadi_model)
        ref_model = Model()

        a = ca.MX.sym("a", 3)
        b = ca.MX.sym("b", 4)
        c = ca.MX.sym("c", 3)
        d = ca.MX.sym("d", 3)
        e = ca.MX.sym("e", 3)
        g = ca.MX.sym("g", 1)
        h = ca.MX.sym("h", 1)
        i = ca.MX.sym('i', 2, 3)
        B = ca.MX.sym("B", 3)
        C = ca.MX.sym("C", 2)
        D = ca.MX.sym("D", 3)
        E = ca.MX.sym("E", 2)
        arx = ca.MX.sym("ar.x", 3)
        arcy = ca.MX.sym("arc.y", 2)
        arcw = ca.MX.sym("arc.w", 2)
        nested1z = ca.MX.sym('nested1.z', 3)
        nested2z = ca.MX.sym('nested2.z', 2, 3)
        nested1n = ca.MX.sym('nested1.n', 1)
        nested2n = ca.MX.sym('nested2.n', 2)

        scalar_f = ca.MX.sym("scalar_f")
        c_dim = ca.MX.sym("c_dim")
        d_dim = ca.MX.sym("d_dim")

        ref_model.alg_states = list(map(Variable, [arx, arcy, arcw, nested1z, nested2z, a, c, d, e, scalar_f, g, h, i]))
        ref_model.alg_states[6].min = [0, 0, 0]
        ref_model.parameters = list(map(Variable, [nested2n, nested1n, d_dim]))
        parameter_values = [np.array([3, 3]), 3, 3]
        for const, val in zip(ref_model.parameters, parameter_values):
            const.value = val
        ref_model.outputs = list(map(Variable, [h]))
        ref_model.constants = list(map(Variable, [b, c_dim, B, C, D, E]))
        constant_values = [np.array([2.7, 3.7, 4.7, 5.7]), 2, ca.linspace(1., 2., 3), 1.7 * ca.DM.ones(2),
                                     ca.DM.zeros(3), ca.DM.ones(2)]
        for const, val in zip(ref_model.constants, constant_values):
            const.value = val
        ref_model.equations = [c - (a + b[0:3] * e), d - (ca.sin(a / b[1:4])), e - (d + scalar_f), g - ca.sum1(c),
                               h - B[1], arx[1] - scalar_f, nested1z - ca.DM.ones(3), nested2z[0, :].T - np.array([4, 5, 6]),
                               nested2z[1, 0] - 3, nested2z[1, 1] - 2, nested2z[1, 2] - 1, i[0, :] - ca.transpose(ca.DM.ones(3)), i[1, :] - ca.transpose(ca.DM.ones(3)), arcy[0] - arcy[1],
                               arcw[0] + arcw[1], a - np.array([1, 2, 3]), scalar_f - 1.3]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_matrixexpressions(self):
        with open(os.path.join(MODEL_DIR, 'MatrixExpressions.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'MatrixExpressions')
        print(casadi_model)
        ref_model = Model()

        A = ca.MX.sym("A", 3, 3)
        b = ca.MX.sym("b", 3)
        c = ca.MX.sym("c", 3)
        d = ca.MX.sym("d", 3)
        C = ca.MX.sym("C", 2, 3)
        D = ca.MX.sym("D", 3, 2)
        E = ca.MX.sym("E", 2, 3)
        I = ca.MX.sym("I", 5, 5)
        F = ca.MX.sym("F", 3, 3)

        ref_model.alg_states = list(map(Variable, [A, b, c, d]))
        ref_model.constants = list(map(Variable, [C, D, E, I, F]))
        constant_values = [1.7 * ca.DM.ones(2, 3), ca.DM.zeros(3, 2), ca.DM.ones(2, 3), ca.DM.eye(5),
                                     ca.DM.triplet([0, 1, 2], [0, 1, 2], [1, 2, 3], 3, 3)]
        for const, val in zip(ref_model.constants, constant_values):
            const.value = val
        ref_model.equations = [ca.mtimes(A, b) - c, ca.mtimes(A.T, b) - d, F[1, 2]]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_delay(self):
        with open(os.path.join(MODEL_DIR, 'Delay.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        casadi_model = gen_casadi.generate(ast_tree, 'Delay')
        # Important to also test expansion of delay expressions of scalar size
        ast_tree = parser.parse(txt)
        casadi_model_expanded = gen_casadi.generate(ast_tree, 'Delay')
        casadi_model_expanded.simplify({'expand_vectors': True})

        ref_model = Model()

        x = ca.MX.sym("x")
        x_delayed = ca.MX.sym("_pymoca_delay_0")
        y = ca.MX.sym("y")
        hour = ca.MX.sym("hour")

        ref_model.alg_states = list(map(Variable, [x, y]))
        ref_model.parameters = list(map(Variable, [hour]))
        ref_model.inputs = list(map(Variable, [x_delayed]))
        ref_model.parameters[0].value = 3600
        ref_model.equations = [y - x_delayed]
        ref_model.delay_states = [x_delayed]
        ref_model.delay_arguments = [DelayArgument(x, 6 * hour)]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)
        self.assert_model_equivalent_numeric(ref_model, casadi_model_expanded)

    def test_delay_for_loop(self):
        with open(os.path.join(MODEL_DIR, 'DelayForLoop.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'DelayForLoop')
        casadi_model.simplify({'detect_aliases': True})
        print(casadi_model)

        ref_model = Model()

        x = ca.MX.sym("x", 3)
        y = ca.MX.sym("y", 3)
        z = ca.MX.sym("z", 3)
        at3_delayed = ca.MX.sym("_pymoca_delay_0", 2)
        delay_time = ca.MX.sym("delay_time")
        eps = ca.MX.sym("eps")

        ref_model.alg_states = list(map(Variable, [x, y]))
        ref_model.parameters = list(map(Variable, [eps]))
        ref_model.inputs = list(map(Variable, [at3_delayed, z, delay_time]))
        ref_model.inputs[-1].fixed = True
        ref_model.equations = [ca.horzcat(x[1:3] - 5 * z[1:3] * eps, y[1:3] - at3_delayed)]
        ref_model.delay_states = [at3_delayed]
        ref_model.delay_arguments = [DelayArgument(3 * x[1:3] * eps, delay_time)]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_delay_for_loop_with_expand_vectors(self):
        with open(os.path.join(MODEL_DIR, 'DelayForLoop.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'DelayForLoop')
        casadi_model.simplify({'expand_vectors': True, 'detect_aliases': True})
        print(casadi_model)

        ref_model = Model()

        def _array_mx(name, n):
            return np.array([ca.MX.sym("{}[{}]".format(name, i+1)) for i in range(n)])

        x = _array_mx("x", 3)
        y = _array_mx("y", 3)
        a = _array_mx("a", 3)
        z = _array_mx("z", 3)
        at3_delayed = _array_mx("_pymoca_delay_0", 2)
        delay_time = ca.MX.sym("delay_time")
        eps = _array_mx("eps", 1)

        ref_model.alg_states = list(map(Variable, [*x, *y, *a]))
        ref_model.parameters = list(map(Variable, [*eps]))
        ref_model.inputs = list(map(Variable, [*at3_delayed, *z, delay_time]))
        ref_model.inputs[-1].fixed = True
        # TODO: "a" is not yet detected as an alias of "x" when expanding.
        ref_model.equations = [*(x[1:3] - 5 * z[1:3] * eps), *(y[1:3] - at3_delayed), *(a - x)]
        ref_model.delay_states = [*at3_delayed]
        ref_model.delay_arguments = [DelayArgument(3 * a_i * eps[0], delay_time) for a_i in a[1:3]]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_array_3d(self):
        casadi_model = transfer_model(MODEL_DIR, 'Array3D', {'expand_vectors': True})

        target_param_values = np.array([[[ -5.326999,  54.050758,  0.000000],
                                    [ -1.0,        0.0,       0.0]],
                                   [[  0.000426,  -0.001241,  2.564056],
                                    [ -1.0,        0.0,       0.0]],
                                   [[  2.577975,  -5.203480,  0.000000],
                                    [ -1.0,        0.0,       0.0]],
                                   [[ 13.219650,  -3.097600, -7.551339],
                                    [ -1.0,        0.0,       0.0]]])

        param_symbols = [x.symbol for x in casadi_model.parameters]

        for ind in np.ndindex(target_param_values.shape):
            flat_index = np.ravel_multi_index(ind, target_param_values.shape)
            self.assertEqual(param_symbols[flat_index].name(), "x[{}]".format(",".join((str(x + 1) for x in ind))))
            self.assertEqual(casadi_model.parameters[flat_index].value, target_param_values[ind])

    def test_attributes(self):
        with open(os.path.join(MODEL_DIR, 'Attributes.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'Attributes')
        print(casadi_model)
        ref_model = Model()

        nested_p1 = ca.MX.sym('nested.p1')
        nested_p = ca.MX.sym('nested.p')
        nested_s = ca.MX.sym('nested.s')
        i = ca.MX.sym("int")
        b = ca.MX.sym("bool")
        r = ca.MX.sym("real")
        der_r = ca.MX.sym("der(real)")
        test_state = ca.MX.sym("test_state")
        i1 = ca.MX.sym("i1")
        i2 = ca.MX.sym("i2")
        i3 = ca.MX.sym("i3")
        i4 = ca.MX.sym("i4")
        cst = ca.MX.sym("cst")
        prm = ca.MX.sym("prm")
        protected_variable = ca.MX.sym("protected_variable")

        ref_model.states = list(map(Variable, [r]))
        ref_model.states[0].start = 20
        ref_model.der_states = list(map(Variable, [der_r]))
        ref_model.alg_states = list(map(Variable, [nested_s, i, b, i4, test_state, protected_variable]))
        ref_model.alg_states[1].min = -5
        ref_model.alg_states[1].max = 10
        ref_model.inputs = list(map(Variable, [i1, i2, i3]))
        ref_model.inputs[0].fixed = True
        ref_model.outputs = list(map(Variable, [i4, protected_variable]))
        ref_model.constants = list(map(Variable, [cst]))
        constant_values = [1]
        for c, v in zip(ref_model.constants, constant_values):
            c.value = v
        ref_model.parameters = list(map(Variable, [nested_p1, nested_p, prm]))
        parameter_values = [1, 2 * nested_p1, 2]
        for c, v in zip(ref_model.parameters, parameter_values):
            c.value = v
        ref_model.equations = [i4 - ((i1 + i2) + i3), der_r - (i1 + ca.if_else(b, 1, 0, True) * i),
                               protected_variable - (i1 + i2), nested_s - 3 * nested_p, test_state - r]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_type(self):
        with open(os.path.join(MODEL_DIR, 'Type.mo'), 'r') as f:
            txt = f.read()
        # noinspection PyUnusedLocal
        ast_tree = parser.parse(txt)
        self.assertTrue(True)

    def test_variable_metadata_function(self):
        # Clear cache
        db_file = os.path.join(MODEL_DIR, 'ParameterAttributes')
        try:
            os.remove(db_file)
        except FileNotFoundError:
            pass

        # Alias detection results in fmin/fmax function calls in the attributes
        compiler_options = \
            {'cache': True,
             'detect_aliases': True,
             'expand_vectors': True}

        ref_model = transfer_model(MODEL_DIR, 'ParameterAttributes', compiler_options)
        self.assertIsInstance(ref_model, Model)
        self.assertNotIsInstance(ref_model, CachedModel)

        cached_model = transfer_model(MODEL_DIR, 'ParameterAttributes', compiler_options)
        self.assertIsInstance(cached_model, Model)
        self.assertIsInstance(cached_model, CachedModel)

        in_var = ca.veccat(*ref_model._symbols(ref_model.parameters))
        function_metadata = ref_model.variable_metadata_function(in_var)

        cached_in_var = ca.veccat(*cached_model._symbols(cached_model.parameters))

        orig_metadata = []
        cached_metadata = []

        variables_with_metadata = ['states', 'alg_states', 'inputs', 'parameters', 'constants']

        for model, metadata in [(ref_model, orig_metadata), (cached_model, cached_metadata)]:
            for variable_list in variables_with_metadata:
                attribute_lists = [[] for i in range(len(CASADI_ATTRIBUTES))]
                for variable in getattr(model, variable_list):
                    for attribute_list_index, attribute in enumerate(CASADI_ATTRIBUTES):
                        value = ca.MX(getattr(variable, attribute))
                        value = value if value.numel() != 1 else ca.repmat(
                            value, *variable.symbol.size())
                        attribute_lists[attribute_list_index].append(value)
                expr = ca.horzcat(*[
                    ca.veccat(*attribute_list) for attribute_list in attribute_lists])
                metadata.append(expr)

        # Check that the output of the variable metadata function matches the
        # original metadata and cached metadata
        for a, b, c in zip(orig_metadata, function_metadata, cached_metadata):
            f_a = ca.Function('tmp', [in_var], [a])
            f_b = ca.Function('tmp', [in_var], [b])
            f_c = ca.Function('tmp', [cached_in_var], [c])

            args_in = ca.DM([2, 11])

            a_out = np.array(f_a(args_in))
            b_out = np.array(f_b(args_in))
            c_out = np.array(f_c(args_in))

            self.assertTrue(np.allclose(a_out, b_out, 0, 0, True))
            self.assertTrue(np.allclose(a_out, c_out, 0, 0, True))

    def test_cache(self):
        # Clear cache
        db_file = os.path.join(MODEL_DIR, 'Aircraft')
        try:
            os.remove(db_file)
        except FileNotFoundError:
            pass

        # Create model, cache it, and load the cache
        compiler_options = \
            {'cache': True}

        ref_model = transfer_model(MODEL_DIR, 'Aircraft', compiler_options)
        self.assertIsInstance(ref_model, Model)
        self.assertNotIsInstance(ref_model, CachedModel)

        cached_model = transfer_model(MODEL_DIR, 'Aircraft', compiler_options)
        self.assertIsInstance(cached_model, Model)
        self.assertIsInstance(cached_model, CachedModel)

        # Compare
        self.assert_model_equivalent_numeric(ref_model, cached_model)

    def test_cache_delay_arguments(self):
        # Clear cache
        db_file = os.path.join(MODEL_DIR, 'Delay')
        try:
            os.remove(db_file)
        except FileNotFoundError:
            pass

        compiler_options = {'cache': True}

        ref_model = transfer_model(MODEL_DIR, 'Delay', compiler_options)
        self.assertIsInstance(ref_model, Model)
        self.assertNotIsInstance(ref_model, CachedModel)

        cached_model = transfer_model(MODEL_DIR, 'Delay', compiler_options)
        self.assertIsInstance(cached_model, Model)
        self.assertIsInstance(cached_model, CachedModel)

        self.assert_model_equivalent_numeric(ref_model, cached_model)

    def test_codegen(self):
        # Clear cache
        db_file = os.path.join(MODEL_DIR, 'Aircraft')
        try:
            os.remove(db_file)
        except FileNotFoundError:
            pass

        for f in glob.glob(os.path.join(MODEL_DIR, "Aircraft*.so")):
            os.remove(f)
        for f in glob.glob(os.path.join(MODEL_DIR, "Aircraft*.dll")):
            os.remove(f)
        for f in glob.glob(os.path.join(MODEL_DIR, "Aircraft*.dylib")):
            os.remove(f)

        # Create model, cache it, and load the cache
        compiler_options = \
            {'codegen': True}

        ref_model = transfer_model(MODEL_DIR, 'Aircraft', compiler_options)
        self.assertIsInstance(ref_model, Model)
        self.assertNotIsInstance(ref_model, CachedModel)

        cached_model = transfer_model(MODEL_DIR, 'Aircraft', compiler_options)
        self.assertIsInstance(cached_model, Model)
        self.assertIsInstance(cached_model, CachedModel)

        # Compare
        self.assert_model_equivalent_numeric(ref_model, cached_model)

    def test_simplify_replace_constant_values(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'replace_constant_values': True}

        casadi_model = transfer_model(MODEL_DIR, 'Simplify', compiler_options)

        ref_model = Model()

        p1 = ca.MX.sym('p1')
        p2 = ca.MX.sym('p2')
        p3 = ca.MX.sym('p3')
        p4 = ca.MX.sym('p4')
        x = ca.MX.sym('x')
        der_x = ca.MX.sym('der(x)')
        alias = ca.MX.sym('alias')
        y = ca.MX.sym('y')
        _tmp = ca.MX.sym('_tmp')
        cst = ca.MX.sym('cst')

        ref_model.states = list(map(Variable, [x]))
        ref_model.states[0].min = 0
        ref_model.states[0].max = 3
        ref_model.states[0].nominal = 10
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [alias, y, _tmp, cst]))
        ref_model.alg_states[0].min = 1
        ref_model.alg_states[0].max = 2
        ref_model.alg_states[0].nominal = 1
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = []
        ref_model.constants = list(map(Variable, []))
        constant_values = []
        for _cst, v in zip(ref_model.constants, constant_values):
            _cst.value = v
        ref_model.parameters = list(map(Variable, [p1, p2, p3, p4]))
        parameter_values = [2.0, 2 * p1, np.nan, 2 * p3]
        for par, v in zip(ref_model.parameters, parameter_values):
            par.value = v
        ref_model.equations = [der_x - x - p1 - p2 - p3 - p4, alias - x, y - x - 3 - _tmp - cst, _tmp - 0.1 * x, cst - 4]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)

    def test_simplify_replace_parameter_expressions(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'replace_parameter_expressions': True}

        casadi_model = transfer_model(MODEL_DIR, 'Simplify', compiler_options)

        ref_model = Model()

        c = ca.MX.sym('c')
        p1 = ca.MX.sym('p1')
        p3 = ca.MX.sym('p3')
        x = ca.MX.sym('x')
        der_x = ca.MX.sym('der(x)')
        alias = ca.MX.sym('alias')
        y = ca.MX.sym('y')
        _tmp = ca.MX.sym('_tmp')
        cst = ca.MX.sym('cst')

        ref_model.states = list(map(Variable, [x]))
        ref_model.states[0].min = 0
        ref_model.states[0].max = 3
        ref_model.states[0].nominal = 10
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [alias, y, _tmp, cst]))
        ref_model.alg_states[0].min = 1
        ref_model.alg_states[0].max = 2
        ref_model.alg_states[0].nominal = 1
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = []
        ref_model.constants = list(map(Variable, [c]))
        constant_values = [3]
        for _cst, v in zip(ref_model.constants, constant_values):
            _cst.value = v
        ref_model.parameters = list(map(Variable, [p1, p3]))
        parameter_values = [2.0, np.nan]
        for par, v in zip(ref_model.parameters, parameter_values):
            par.value = v
        ref_model.equations = [der_x - x - 3 * p1 - 3 * p3, alias - x, y - x - c - _tmp - cst, _tmp - 0.1 * x, cst - 4]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)

    def test_simplify_replace_parameter_values(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'replace_parameter_values': True}

        casadi_model = transfer_model(MODEL_DIR, 'Simplify', compiler_options)

        ref_model = Model()

        c = ca.MX.sym('c')
        p2 = ca.MX.sym('p2')
        p3 = ca.MX.sym('p3')
        p4 = ca.MX.sym('p4')
        x = ca.MX.sym('x')
        der_x = ca.MX.sym('der(x)')
        alias = ca.MX.sym('alias')
        y = ca.MX.sym('y')
        _tmp = ca.MX.sym('_tmp')
        cst = ca.MX.sym('cst')

        ref_model.states = list(map(Variable, [x]))
        ref_model.states[0].min = 0
        ref_model.states[0].max = 3
        ref_model.states[0].nominal = 10
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [alias, y, _tmp, cst]))
        ref_model.alg_states[0].min = 1
        ref_model.alg_states[0].max = 2
        ref_model.alg_states[0].nominal = 1
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = []
        ref_model.constants = list(map(Variable, [c]))
        constant_values = [3]
        for _cst, v in zip(ref_model.constants, constant_values):
            _cst.value = v
        ref_model.parameters = list(map(Variable, [p2, p3, p4]))
        parameter_values = [4, np.nan, 2 * p3]
        for par, v in zip(ref_model.parameters, parameter_values):
            par.value = v
        ref_model.equations = [der_x - x - 2 - p2 - p3 - p4, alias - x, y - x - c - _tmp - cst, _tmp - 0.1 * x, cst - 4]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)

    def test_simplify_replace_parameter_values_and_expressions(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'replace_parameter_values': True,
             'replace_parameter_expressions': True}

        casadi_model = transfer_model(MODEL_DIR, 'Simplify', compiler_options)

        ref_model = Model()

        c = ca.MX.sym('c')
        p3 = ca.MX.sym('p3')
        x = ca.MX.sym('x')
        der_x = ca.MX.sym('der(x)')
        alias = ca.MX.sym('alias')
        y = ca.MX.sym('y')
        _tmp = ca.MX.sym('_tmp')
        cst = ca.MX.sym('cst')

        ref_model.states = list(map(Variable, [x]))
        ref_model.states[0].min = 0
        ref_model.states[0].max = 3
        ref_model.states[0].nominal = 10
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [alias, y, _tmp, cst]))
        ref_model.alg_states[0].min = 1
        ref_model.alg_states[0].max = 2
        ref_model.alg_states[0].nominal = 1
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = []
        ref_model.constants = list(map(Variable, [c]))
        constant_values = [3]
        for _cst, v in zip(ref_model.constants, constant_values):
            _cst.value = v
        ref_model.parameters = list(map(Variable, [p3]))
        parameter_values = [np.nan]
        for par, v in zip(ref_model.parameters, parameter_values):
            par.value = v
        ref_model.equations = [der_x - x - 6 - 3 * p3, alias - x, y - x - c - _tmp - cst, _tmp - 0.1 * x, cst - 4]

        print(casadi_model)
        print(ref_model)
        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)

    def test_simplify_eliminate_constant_assignments(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'eliminate_constant_assignments': True}

        casadi_model = transfer_model(MODEL_DIR, 'Simplify', compiler_options)

        ref_model = Model()

        c = ca.MX.sym('c')
        p1 = ca.MX.sym('p1')
        p2 = ca.MX.sym('p2')
        p3 = ca.MX.sym('p3')
        p4 = ca.MX.sym('p4')
        x = ca.MX.sym('x')
        der_x = ca.MX.sym('der(x)')
        alias = ca.MX.sym('alias')
        y = ca.MX.sym('y')
        _tmp = ca.MX.sym('_tmp')
        cst = ca.MX.sym('cst')

        ref_model.states = list(map(Variable, [x]))
        ref_model.states[0].min = 0
        ref_model.states[0].max = 3
        ref_model.states[0].nominal = 10
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [alias, y, _tmp]))
        ref_model.alg_states[0].min = 1
        ref_model.alg_states[0].max = 2
        ref_model.alg_states[0].nominal = 1
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = []
        ref_model.constants = list(map(Variable, [c, cst]))
        constant_values = [3, 4]
        for _cst, v in zip(ref_model.constants, constant_values):
            _cst.value = v
        ref_model.parameters = list(map(Variable, [p1, p2, p3, p4]))
        parameter_values = [2.0, 2 * p1, np.nan, 2 * p3]
        for par, v in zip(ref_model.parameters, parameter_values):
            par.value = v
        ref_model.equations = [der_x - x - p1 - p2 - p3 - p4, alias - x, y - x - c - _tmp - cst, _tmp - 0.1 * x]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)

    def test_simplify_eliminable_variable_expression(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'eliminable_variable_expression': r'_\w+'}

        casadi_model = transfer_model(MODEL_DIR, 'Simplify', compiler_options)

        ref_model = Model()

        c = ca.MX.sym('c')
        p1 = ca.MX.sym('p1')
        p2 = ca.MX.sym('p2')
        p3 = ca.MX.sym('p3')
        p4 = ca.MX.sym('p4')
        x = ca.MX.sym('x')
        der_x = ca.MX.sym('der(x)')
        alias = ca.MX.sym('alias')
        y = ca.MX.sym('y')
        cst = ca.MX.sym('cst')

        ref_model.states = list(map(Variable, [x]))
        ref_model.states[0].min = 0
        ref_model.states[0].max = 3
        ref_model.states[0].nominal = 10
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [alias, y, cst]))
        ref_model.alg_states[0].min = 1
        ref_model.alg_states[0].max = 2
        ref_model.alg_states[0].nominal = 1
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = []
        ref_model.constants = list(map(Variable, [c]))
        constant_values = [3]
        for _cst, v in zip(ref_model.constants, constant_values):
            _cst.value = v
        ref_model.parameters = list(map(Variable, [p1, p2, p3, p4]))
        parameter_values = [2.0, 2 * p1, np.nan, 2 * p3]
        for par, v in zip(ref_model.parameters, parameter_values):
            par.value = v
        ref_model.equations = [der_x - x - p1 - p2 - p3 - p4, alias - x, y - x - c - 0.1 * x - cst, cst - 4]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)

    def test_simplify_detect_aliases(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'detect_aliases': True}

        casadi_model = transfer_model(MODEL_DIR, 'Simplify', compiler_options)

        ref_model = Model()

        c = ca.MX.sym('c')
        p1 = ca.MX.sym('p1')
        p2 = ca.MX.sym('p2')
        p3 = ca.MX.sym('p3')
        p4 = ca.MX.sym('p4')
        x = ca.MX.sym('x')
        der_x = ca.MX.sym('der(x)')
        y = ca.MX.sym('y')
        _tmp = ca.MX.sym('_tmp')
        cst = ca.MX.sym('cst')

        ref_model.states = list(map(Variable, [x]))
        ref_model.states[0].min = 1
        ref_model.states[0].max = 2
        ref_model.states[0].nominal = 10
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [y, _tmp, cst]))
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = []
        ref_model.constants = list(map(Variable, [c]))
        constant_values = [3]
        for _cst, v in zip(ref_model.constants, constant_values):
            _cst.value = v
        ref_model.parameters = list(map(Variable, [p1, p2, p3, p4]))
        parameter_values = [2.0, 2 * p1, np.nan, 2 * p3]
        for par, v in zip(ref_model.parameters, parameter_values):
            par.value = v
        ref_model.equations = [der_x - x - p1 - p2 - p3 - p4, y - x - c - _tmp - cst, _tmp - 0.1 * x, cst - 4]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)
        self.assertEquals(casadi_model.states[0].aliases, ['alias'])

    def test_simplify_detect_negative_alias(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'detect_aliases': True}

        casadi_model = transfer_model(MODEL_DIR, 'NegativeAlias', compiler_options)

        ref_model = Model()

        x = ca.MX.sym('x')
        der_x = ca.MX.sym('der(x)')

        ref_model.states = list(map(Variable, [x]))
        ref_model.states[0].min = 1
        ref_model.states[0].max = 2
        ref_model.states[0].nominal = 10
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, []))
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = []
        ref_model.equations = [der_x - x]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)
        self.assertEquals(casadi_model.states[0].aliases, ['-alias'])

    def test_simplify_expand_vectors(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'expand_vectors': True}

        casadi_model = transfer_model(MODEL_DIR, 'SimplifyVector', compiler_options)

        ref_model = Model()

        x1 = ca.MX.sym('x[1]')
        x2 = ca.MX.sym('x[2]')
        y1 = ca.MX.sym('y[1]')
        y2 = ca.MX.sym('y[2]')
        der_x1 = ca.MX.sym('der(x)[1]')
        der_x2 = ca.MX.sym('der(x)[2]')

        ref_model.states = list(map(Variable, [x1, x2]))
        ref_model.der_states = list(map(Variable, [der_x1, der_x2]))
        ref_model.alg_states = list(map(Variable, [y1, y2]))
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = ['y[1]', 'y[2]']
        ref_model.equations = [der_x1 - x1, der_x2 - x2, y1 - x1, y2 - x2]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)

    def test_simplify_reduce_affine_expression_loop(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'expand_vectors': True,
             'detect_aliases': True,
             'reduce_affine_expression': True,
             'replace_constant_expressions': True,
             'replace_constant_values': True,
             'replace_parameter_expressions': True,
             'replace_parameter_values': True}

        casadi_model = transfer_model(MODEL_DIR, 'SimplifyLoop', compiler_options)

        ref_model = Model()

        x = ca.MX.sym('x')
        y1 = ca.MX.sym('y[1]')
        y2 = ca.MX.sym('y[2]')

        A = ca.MX(2, 3)
        A[0, 0] = -1
        A[0, 1] = 1
        A[0, 2] = 0
        A[1, 0] = -2
        A[1, 1] = 0
        A[1, 2] = 1
        b = ca.MX(2, 1)
        b[0, 0] = 0
        b[1, 0] = 0

        ref_model.states = list(map(Variable, []))
        ref_model.der_states = list(map(Variable, []))
        ref_model.alg_states = list(map(Variable, [x, y1, y2]))
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = []
        x = ca.vertcat(x, y1, y2)
        ref_model.equations = [ca.mtimes(A, x) + b]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)

    def test_simplify_reduce_affine_expression(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'reduce_affine_expression': True}

        casadi_model = transfer_model(MODEL_DIR, 'Simplify', compiler_options)

        ref_model = Model()

        c = ca.MX.sym('c')
        p1 = ca.MX.sym('p1')
        p2 = ca.MX.sym('p2')
        p3 = ca.MX.sym('p3')
        p4 = ca.MX.sym('p4')
        x = ca.MX.sym('x')
        der_x = ca.MX.sym('der(x)')
        alias = ca.MX.sym('alias')
        y = ca.MX.sym('y')
        _tmp = ca.MX.sym('_tmp')
        cst = ca.MX.sym('cst')

        ref_model.states = list(map(Variable, [x]))
        ref_model.states[0].min = 0
        ref_model.states[0].max = 3
        ref_model.states[0].nominal = 10
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [alias, y, _tmp, cst]))
        ref_model.alg_states[0].min = 1
        ref_model.alg_states[0].max = 2
        ref_model.alg_states[0].nominal = 1
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = []
        ref_model.constants = list(map(Variable, [c]))
        constant_values = [3]
        for _cst, v in zip(ref_model.constants, constant_values):
            _cst.value = v
        ref_model.parameters = list(map(Variable, [p1, p2, p3, p4]))
        parameter_values = [2.0, 2 * p1, np.nan, 2 * p3]
        for par, v in zip(ref_model.parameters, parameter_values):
            par.value = v

        A = ca.MX(5, 6)
        A[0, 0] = -1.0
        A[0, 1] = 1.0
        A[1, 2] = 1.0
        A[1, 0] = -1.0
        A[2, 0] = -1.0
        A[2, 3] = 1.0
        A[2, 4] = -1.0
        A[2, 5] = -1.0
        A[3, 0] = -0.1
        A[3, 4] = 1.0
        A[4, 5] = 1.0
        b = ca.MX(5, 1)
        b[0] = -p1 - p2 - p3 - p4
        b[2] = -c
        b[4] = -4
        x = ca.vertcat(x, der_x, alias, y, _tmp, cst)
        ref_model.equations = [ca.mtimes(A, x) + b]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)

    def test_simplify_all(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'expand_vectors': True,
             'replace_constant_values': True,
             'replace_constant_expressions': True,
             'replace_parameter_values': True,
             'replace_parameter_expressions': True,
             'eliminate_constant_assignments': True,
             'detect_aliases': True,
             'eliminable_variable_expression': r'_\w+',
             'reduce_affine_expression': True}

        casadi_model = transfer_model(MODEL_DIR, 'Simplify', compiler_options)

        ref_model = Model()

        p3 = ca.MX.sym('p3')
        x = ca.MX.sym('x')
        der_x = ca.MX.sym('der(x)')
        y = ca.MX.sym('y')

        ref_model.states = list(map(Variable, [x]))
        ref_model.states[0].min = 1
        ref_model.states[0].max = 2
        ref_model.states[0].nominal = 10
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [y]))
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = []
        ref_model.constants = list(map(Variable, []))
        constant_values = []
        for _cst, v in zip(ref_model.constants, constant_values):
            _cst.value = v
        ref_model.parameters = list(map(Variable, [p3]))
        parameter_values = [np.nan]
        for par, v in zip(ref_model.parameters, parameter_values):
            par.value = v

        A = ca.MX(2, 3)
        A[0, 0] = -1.0
        A[0, 1] = 1.0
        A[1, 0] = -1.1
        A[1, 2] = 1.0
        b = ca.MX(2, 1)
        b[0] = -6 - 3 * p3
        b[1] = -7
        x = ca.vertcat(x, der_x, y)
        ref_model.equations = [ca.mtimes(A, x) + b]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)

    def test_simplify_expand_mx(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'expand_mx': True}

        casadi_model = transfer_model(MODEL_DIR, 'Simplify', compiler_options)

        ref_model = Model()

        c = ca.MX.sym('c')
        p1 = ca.MX.sym('p1')
        p2 = ca.MX.sym('p2')
        p3 = ca.MX.sym('p3')
        p4 = ca.MX.sym('p4')
        x = ca.MX.sym('x')
        der_x = ca.MX.sym('der(x)')
        alias = ca.MX.sym('alias')
        y = ca.MX.sym('y')
        _tmp = ca.MX.sym('_tmp')
        cst = ca.MX.sym('cst')

        ref_model.states = list(map(Variable, [x]))
        ref_model.states[0].min = 0
        ref_model.states[0].max = 3
        ref_model.states[0].nominal = 10
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [alias, y, _tmp, cst]))
        ref_model.alg_states[0].min = 1
        ref_model.alg_states[0].max = 2
        ref_model.alg_states[0].nominal = 1
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = []
        ref_model.constants = list(map(Variable, [c]))
        constant_values = [3]
        for _cst, v in zip(ref_model.constants, constant_values):
            _cst.value = v
        ref_model.parameters = list(map(Variable, [p1, p2, p3, p4]))
        parameter_values = [2.0, 2 * p1, np.nan, 2 * p3]
        for par, v in zip(ref_model.parameters, parameter_values):
            par.value = v
        ref_model.equations = [der_x - x - p1 - p2 - p3 - p4, alias - x, y - x - c - _tmp - cst, _tmp - 0.1 * x, cst - 4]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)

    def test_state_annotator(self):
        with open(os.path.join(MODEL_DIR, 'StateAnnotator.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'StateAnnotator')
        print(casadi_model)
        ref_model = Model()

        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        z = ca.MX.sym('z')
        u = ca.MX.sym('u', 3)
        v = ca.MX.sym('v', 3)
        w = ca.MX.sym('w', 3)
        der_x = ca.MX.sym('der(x)')
        der_y = ca.MX.sym('der(y)')
        der_z = ca.MX.sym('der(z)')
        der_u = ca.MX.sym('der(u)', 3)
        der_v = ca.MX.sym('der(v)', 3)
        der_w = ca.MX.sym('der(w)', 3)

        ref_model.states = list(map(Variable, [x, y, z, u, v, w]))
        ref_model.der_states = list(map(Variable, [der_x, der_y, der_z, der_u, der_v, der_w]))
        ref_model.equations = [der_x + der_y - 1,
                               der_x * y + x * der_y - 2,
                               (der_x * y - x * der_y) / (y**2) - 3,
                               2 * x * der_x - 4,
                               der_z - 5,
                               der_x * z + x * der_z + der_y * z + y * der_z - 4,
                               0,
                               der_u - np.array([1, 2, 3]),
                               der_v[0] - 4,
                               der_v[1] - 5,
                               der_v[2] - 6,
                               der_w - np.array([7, 8, 9])]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_alias_relation(self):
        a = AliasRelation()
        self.assertEqual(a.canonical_signed('-a'), ('a', -1))
        a.add('a', '-b')
        a.add('b', 'c')
        a.add('d', '-b')
        self.assertEqual(list(a), [('d', ['a', '-b', '-c'])])

    def test_cat_params(self):
        casadi_model = transfer_model(MODEL_DIR, 'Concat', {'replace_constant_values': True})
        c = [0, 1, 2, 2, 2, 0, 1]
        for i, e in enumerate(c):
            self.assertEqual(casadi_model.parameters[1].value[i], e)

    def test_inline_input_assignment(self):
        casadi_model = transfer_model(MODEL_DIR, 'InlineAssignment')
        self.assertTrue(casadi_model.inputs[0].fixed)
        self.assertFalse(casadi_model.alg_states[0].fixed)
        casadi_model = transfer_model(MODEL_DIR, 'InlineAssignment', {'detect_aliases': True})
        self.assertTrue(casadi_model.inputs[0].fixed)

    def test_unspecified_dimensions(self):
        with open(os.path.join(MODEL_DIR, 'UnspecifiedDimensions.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'B')

        a_x = casadi_model.parameters[0]
        a_y = casadi_model.parameters[1]

        self.assertTrue(a_x.symbol.name() == "a.x")
        self.assertEqual(a_x.value, [[ 88.3224,  281.642,  143.011],
                                     [ 58.8183, -24.9845,      0.0],
                                     [-1.45483,      0.0,      0.0]])

        self.assertTrue(a_y.symbol.name() == "a.y")
        self.assertEqual(a_y.value, [[1, 2],
                                     [3, 4],
                                     [5, 6]])

    def test_wrong_unspecified_dimensions(self):
        txt = """
            model A
              parameter Real x[:, :];
              parameter Real y[:, 4];
            end A;

            model B
              A a(x = {{ 88.3224,   281.642,  143.011},
                       { 58.8183,  -24.9845,      0.0},
                       {-1.45483,       0.0,      0.0}},
                  y = {{ 88.3224,   281.642,  143.011},
                       { 58.8183,  -24.9845,      0.0},
                       {-1.45483,       0.0,      0.0}});
            end B;"""

        ast_tree = parser.parse(txt)
        try:
            casadi_model = gen_casadi.generate(ast_tree, 'B')
        except Exception as e:
            assert e.args[0] == 'Dimension 2 of definition and ' \
                                'value for symbol a.y differs: 4 != 3'
            return

        assert False, "No exception raised on wrong dimensionality."

    def test_skip_annotations(self):
        with open(os.path.join(MODEL_DIR, 'Annotations.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        casadi_model = gen_casadi.generate(ast_tree, 'A')

    def test_size_one_array(self):
        txt = """
            model Test
                parameter Integer m = 2;
                parameter Integer n = 1;
                Real v[m];
                Real w[n];
                Real x[1];
                Real y;
                Real z[1,1];
            equation
                x[1] = y
                w[1] = y
                z[1,1] = v[2]
            end Test;
            """
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'Test')
        casadi_model.simplify({'expand_vectors': True})
        print(casadi_model)

        ref_model = Model()
        m = ca.MX.sym('m')
        n = ca.MX.sym('n')
        v1 = ca.MX.sym('v[1]')
        v2 = ca.MX.sym('v[2]')
        w1 = ca.MX.sym('w[1]')
        x1 = ca.MX.sym('x[1]')
        y = ca.MX.sym('y')
        z11 = ca.MX.sym('z[1,1]')

        ref_model.alg_states = list(map(Variable, [v1, v2, w1, x1, y, z11]))
        ref_model.parameters = list(map(Variable, [m, n]))
        ref_model.equations = [x1 - y,
                               w1 - y,
                               z11 - v2]

        self.assert_model_equivalent(ref_model, casadi_model)

    def test_nested_indices_simple(self):
        txt = """
            model A
                Real x[3];
            end A;

            model B
                A a;
            end B;

            model C
                B b[2];
            end C;

            model Test
                C c;
            equation
                c.b[1].a.x[1] = 1
                c.b[2].a.x[3] = 2
            end Test;
            """
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'Test', {'expand_vectors': True})
        casadi_model.simplify({'expand_vectors': True})
        print(casadi_model)

        ref_model = Model()
        c11 = ca.MX.sym('c.b[1].a.x[1]')
        c12 = ca.MX.sym('c.b[1].a.x[2]')
        c13 = ca.MX.sym('c.b[1].a.x[3]')
        c21 = ca.MX.sym('c.b[2].a.x[1]')
        c22 = ca.MX.sym('c.b[2].a.x[2]')
        c23 = ca.MX.sym('c.b[2].a.x[3]')

        ref_model.alg_states = list(map(Variable, [c11, c12, c13, c21, c22, c23]))
        ref_model.equations = [c11 - 1,
                               c23 - 2]

        self.assert_model_equivalent(ref_model, casadi_model)
        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_nested_indices_hard(self):
        txt = """
            model A
                Real x[1,2];
            end A;

            model B
                A a[3,1];
            end B;

            model C
                B b;
            end C;

            model D
                C c[2];
            end D;

            model Test
                D d;
            end Test;
            """
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'Test', {'expand_vectors': True})
        casadi_model.simplify({'expand_vectors': True})
        print(casadi_model)

        ref_model = Model()
        c11111 = ca.MX.sym('d.c[1].b.a[1,1].x[1,1]')
        c11112 = ca.MX.sym('d.c[1].b.a[1,1].x[1,2]')
        c12111 = ca.MX.sym('d.c[1].b.a[2,1].x[1,1]')
        c12112 = ca.MX.sym('d.c[1].b.a[2,1].x[1,2]')
        c13111 = ca.MX.sym('d.c[1].b.a[3,1].x[1,1]')
        c13112 = ca.MX.sym('d.c[1].b.a[3,1].x[1,2]')
        c21111 = ca.MX.sym('d.c[2].b.a[1,1].x[1,1]')
        c21112 = ca.MX.sym('d.c[2].b.a[1,1].x[1,2]')
        c22111 = ca.MX.sym('d.c[2].b.a[2,1].x[1,1]')
        c22112 = ca.MX.sym('d.c[2].b.a[2,1].x[1,2]')
        c23111 = ca.MX.sym('d.c[2].b.a[3,1].x[1,1]')
        c23112 = ca.MX.sym('d.c[2].b.a[3,1].x[1,2]')

        ref_model.alg_states = list(map(Variable,
                    [c11111, c11112, c12111, c12112, c13111, c13112,
                     c21111, c21112, c22111, c22112, c23111, c23112]))

        self.assert_model_equivalent(ref_model, casadi_model)

    def test_assigning_without_index(self):
        txt = """
            model Test
                Real x[3];
                Real y[3]
            equation
                x = {4, 5, 6}; //this should be equivalent to x[:] = {1, 2, 3}
                y[:] = {3, 2, 1};
            end Test;"""
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'Test', {'expand_vectors': True})
        casadi_model.simplify({'expand_vectors': True})
        print(casadi_model)

        ref_model = Model()
        x1 = ca.MX.sym('x[1]')
        x2 = ca.MX.sym('x[2]')
        x3 = ca.MX.sym('x[3]')
        y1 = ca.MX.sym('y[1]')
        y2 = ca.MX.sym('y[2]')
        y3 = ca.MX.sym('y[3]')

        ref_model.alg_states = list(map(Variable, [x1, x2, x3, y1, y2, y3]))
        ref_model.equations = [x1 - 4, x2 - 5, x3 - 6, y1 - 3, y2 - 2, y3 - 1]

        self.assert_model_equivalent(ref_model, casadi_model)
        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_assigning_without_index_nested(self):
        txt = """
            model A
                Real x[3];
            end A;

            model Test
                A a[2];
            equation
                a[1].x[:] = {4, 5, 6};
                a[2].x = {3, 2, 1}; //this should be equivalent to a[2].x[:] = {3, 2, 1}
            end Test;"""
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'Test')
        casadi_model.simplify({'expand_vectors': True})
        print(casadi_model)

        ref_model = Model()
        v11 = ca.MX.sym('a[1].x[1]')
        v12 = ca.MX.sym('a[1].x[2]')
        v13 = ca.MX.sym('a[1].x[3]')
        v21 = ca.MX.sym('a[2].x[1]')
        v22 = ca.MX.sym('a[2].x[2]')
        v23 = ca.MX.sym('a[2].x[3]')

        ref_model.alg_states = list(map(Variable, [v11, v12, v13, v21, v22, v23]))
        ref_model.equations = [v11 - 4, v12 - 5, v13 - 6, v21 - 3, v22 - 2, v23 - 1]

        self.assert_model_equivalent(ref_model, casadi_model)
        self.assert_model_equivalent_numeric(ref_model, casadi_model)


if __name__ == "__main__":
    unittest.main()
