#!/usr/bin/env python
"""
Modelica parse Tree to AST tree.
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import os
import unittest

import casadi as ca
import numpy as np

from pymola import gen_casadi
from pymola import parser

CasadiSysModel = gen_casadi.CasadiSysModel
TEST_DIR = os.path.dirname(os.path.realpath(__file__))


# noinspection PyPep8Naming,PyUnresolvedReferences
class GenCasadiTest(unittest.TestCase):
    def assert_model_equivalent(self, A, B):
        def sstr(a): return set([str(e) for e in a])

        for l in ["states", "der_states", "inputs", "outputs", "constants", "parameters", "equations"]:
            self.assertEqual(sstr(getattr(A, l)), sstr(getattr(B, l)))

    def assert_model_equivalent_numeric(self, A, B, tol=1e-9):
        self.assertEqual(len(A.states), len(B.states))
        self.assertEqual(len(A.der_states), len(B.der_states))
        self.assertEqual(len(A.inputs), len(B.inputs))
        self.assertEqual(len(A.outputs), len(B.outputs))
        self.assertEqual(len(A.constants), len(B.constants))
        self.assertEqual(len(A.constant_values), len(B.constant_values))
        self.assertEqual(len(A.parameters), len(B.parameters))
        self.assertEqual(len(A.equations), len(B.equations))

        for a, b in zip(A.constant_values, B.constant_values):
            delta = ca.vec(a - b)
            for i in range(delta.size1()):
                test = float(delta[i]) <= tol
                self.assertTrue(test)

        this = A.dae_residual_function(group_arguments=False)
        that = B.dae_residual_function(group_arguments=False)

        this_mx = this.mx_in()
        that_mx = that.mx_in()
        this_in = [e.name() for e in this_mx if e.is_symbolic()]
        that_in = [e.name() for e in that_mx if e.is_symbolic()]

        that_from_this = []
        this_mx_dict = dict(zip(this_in, this_mx))
        that_mx_dict = dict(zip(that_in, that_mx))
        for e in that_in:
            self.assertTrue(e in this_in)
            self.assertEqual(this_mx_dict[e].size1(), that_mx_dict[e].size1())
            self.assertEqual(this_mx_dict[e].size2(), that_mx_dict[e].size2())
            that_from_this.append(this_mx_dict[e])

        that = ca.Function('f', this_mx, that.call(that_from_this))

        np.random.seed(0)

        args_in = []
        for i in range(this.n_in()):
            sp = this.sparsity_in(0)
            r = ca.DM(sp, np.random.random(sp.nnz()))
            args_in.append(r)

        this_out = this.call(args_in)
        that_out = that.call(args_in)

        for i, (a, b) in enumerate(zip(this_out, that_out)):
            test = float(ca.norm_2(ca.vec(a - b))) <= tol
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
        der_x = ca.MX.sym("der(x)")
        der_v_x = ca.MX.sym("der(v_x)")
        k = ca.MX.sym("k")
        c = ca.MX.sym("c")
        ref_model.states = [x, v_x]
        ref_model.der_states = [der_x, der_v_x]
        ref_model.parameters = [c, k]
        ref_model.equations = [der_x - v_x, der_v_x - (-k * x - c * v_x)]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_estimator(self):
        with open(os.path.join(TEST_DIR, 'Estimator.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'Estimator')
        ref_model = CasadiSysModel()
        print(casadi_model)

        x = ca.MX.sym("x")
        der_x = ca.MX.sym("der(x)")
        y = ca.MX.sym("y")

        ref_model.states = [x]
        ref_model.der_states = [der_x]
        ref_model.alg_states = [y]
        ref_model.outputs = [y]
        ref_model.equations = [der_x + x, y - x]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_aircraft(self):
        with open(os.path.join(TEST_DIR, 'Aircraft.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        # noinspection PyUnusedLocal
        casadi_model = gen_casadi.generate(ast_tree, 'Aircraft')
        # noinspection PyUnusedLocal
        ref_model = CasadiSysModel()
        self.assertTrue(True)

    def test_connector(self):
        with open(os.path.join(TEST_DIR, 'ConnectorHQ.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'System')
        ref_model = CasadiSysModel()
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
        qc__down__H = ca.MX.sym("qc.down.H")
        qc__down__Q = ca.MX.sym("qc.down.Q")

        hb__up__H = ca.MX.sym("hb.up.H")
        hb__up__Q = ca.MX.sym("hb.up.Q")

        ref_model.alg_states = [qc__down__H, a__down__H, b__down__H, c__down__H, c__up__H, hb__up__H, a__up__H,
                                b__up__H, qa__down__H, a__up__Q, qa__down__Q, c__down__Q, hb__up__Q, c__up__Q, b__up__Q,
                                b__down__Q, qc__down__Q, a__down__Q]

        ref_model.equations = [a__up__H - a__down__H,
                               a__up__Q + a__down__Q,
                               c__up__H - c__down__H,
                               c__up__Q + c__down__Q,

                               b__up__H - b__down__H,
                               b__up__Q + b__down__Q,

                               qa__down__Q,
                               qc__down__Q,

                               hb__up__H,

                               qa__down__H - a__up__H,
                               qc__down__H - c__up__H,
                               a__down__H - b__up__H,
                               c__down__H - b__up__H,
                               b__down__H - hb__up__H,

                               a__down__Q + (b__up__Q + c__down__Q),
                               qc__down__Q + c__up__Q,
                               b__down__Q + hb__up__Q,
                               qa__down__Q + a__up__Q]

        print(ref_model)
        self.assert_model_equivalent(ref_model, casadi_model)

    def test_connector_hqz(self):
        with open(os.path.join(TEST_DIR, 'ConnectorHQZ.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'SystemZ')
        ref_model = CasadiSysModel()
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

        qa__down__H = ca.MX.sym("qa.down.H")
        qa__down__Q = ca.MX.sym("qa.down.Q")
        qa__down__Z = ca.MX.sym("qa.down.Z")
        qc__down__H = ca.MX.sym("qc.down.H")
        qc__down__Q = ca.MX.sym("qc.down.Q")
        qc__down__Z = ca.MX.sym("qc.down.Z")

        hb__up__H = ca.MX.sym("hb.up.H")
        hb__up__Q = ca.MX.sym("hb.up.Q")
        hb__up__Z = ca.MX.sym("hb.up.Z")

        ref_model.alg_states = [qc__down__H, a__down__H, b__down__H, c__down__H, c__up__H, hb__up__H, a__up__H,
                                b__up__H, qa__down__H, a__up__Q, qa__down__Q, c__down__Q, hb__up__Q, c__up__Q, b__up__Q,
                                b__down__Q, qc__down__Q, a__down__Q, a__up__Z, a__down__Z, b__up__Z, b__down__Z,
                                c__up__Z, c__down__Z, qa__down__Z, qc__down__Z, hb__up__Z]

        ref_model.equations = [a__up__H - a__down__H,
                               a__up__Q + a__down__Q,
                               c__up__H - c__down__H,
                               c__up__Q + c__down__Q,

                               b__up__H - b__down__H,
                               b__up__Q + b__down__Q,

                               qa__down__Q,
                               qc__down__Q,

                               hb__up__H,

                               qa__down__H - a__up__H,
                               qc__down__H - c__up__H,
                               a__down__H - b__up__H,
                               c__down__H - b__up__H,
                               b__down__H - hb__up__H,

                               a__down__Q + (b__up__Q + c__down__Q),
                               qc__down__Q + c__up__Q,
                               b__down__Q + hb__up__Q,
                               qa__down__Q + a__up__Q,

                               qa__down__Z - a__up__Z,
                               qc__down__Z - c__up__Z,
                               a__down__Z - b__up__Z,
                               c__down__Z - b__up__Z,
                               b__down__Z - hb__up__Z]

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
        der_x = ca.MX.sym("der(x)")
        y = ca.MX.sym("y")
        der_y = ca.MX.sym("der(y)")

        ref_model.states = [x, y]
        ref_model.der_states = [der_x, der_y]
        ref_model.equations = [der_x + der_y - 1, der_x - 2]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_if_else(self):
        with open(os.path.join(TEST_DIR, 'IfElse.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'IfElse')
        ref_model = CasadiSysModel()

        x = ca.MX.sym("x")
        y1 = ca.MX.sym("y1")
        y2 = ca.MX.sym("y2")
        y3 = ca.MX.sym("y3")
        y_max = ca.MX.sym("y_max")

        ref_model.inputs = [x]
        ref_model.outputs = [y1, y2, y3]
        ref_model.alg_states = [x, y1, y2, y3]
        ref_model.parameters = [y_max]
        ref_model.equations = [
            y1 - ca.if_else(x > 0, 1, 0) * y_max,
            ca.if_else(x > 1, ca.vertcat(y3 - 100, y2 - y_max),
                       ca.if_else(x > 2, ca.vertcat(y3 - 1000, y2 - y_max - 1),
                                  ca.vertcat(y3 - 10000, y2)))]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_inheritance(self):
        with open(os.path.join(TEST_DIR, 'Inheritance.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'Sub')
        ref_model = CasadiSysModel()

        x = ca.MX.sym("x")
        der_x = ca.MX.sym("der(x)")
        y = ca.MX.sym("y")
        # noinspection PyUnusedLocal
        der_y = ca.MX.sym("y")
        k = ca.MX.sym("k")

        ref_model.states = [x]
        ref_model.der_states = [der_x]
        ref_model.alg_states = [y]
        ref_model.parameters = [k]
        ref_model.equations = [der_x - k * x, x + y - 3]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_inheritance_instantiation(self):
        with open(os.path.join(TEST_DIR, 'InheritanceInstantiation.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'C2')
        ref_model = CasadiSysModel()
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
        ref_model.alg_states = [bcomp1_v, bcomp2_v, bcomp3_v]
        ref_model.parameters = [bcomp1_a, bcomp1_b, bcomp2_a, bcomp2_b, bcomp3_a, bcomp3_b]
        ref_model.equations = []

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_builtin(self):
        with open(os.path.join(TEST_DIR, 'BuiltinFunctions.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'BuiltinFunctions')
        print("BuiltinFunctions", casadi_model)
        ref_model = CasadiSysModel()

        x = ca.MX.sym("x")
        y = ca.MX.sym("y")
        z = ca.MX.sym("z")
        w = ca.MX.sym("w")
        u = ca.MX.sym("u")

        ref_model.inputs = [x]
        ref_model.outputs = [y, z, w, u]
        ref_model.alg_states = [x, y, z, w, u]
        ref_model.equations = [y - ca.sin(ref_model.time), z - ca.cos(x), w - ca.fmin(y, z), u - ca.fabs(w)]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_forloop(self):
        with open(os.path.join(TEST_DIR, 'ForLoop.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'ForLoop')
        print(casadi_model)
        ref_model = CasadiSysModel()

        x = ca.MX.sym("x", 10)
        y = ca.MX.sym("y", 10)
        z = ca.MX.sym("z", 10)
        w = ca.MX.sym('w', 2, 10)
        b = ca.MX.sym("b")
        n = ca.MX.sym("n")

        ref_model.alg_states = [x, y, z, w, b]
        ref_model.parameters = [n]
        ref_model.equations = [
            ca.horzcat(x - (np.arange(1, 11) + b), w[0, :].T - np.arange(1, 11), w[1, :].T - np.arange(2, 21, 2)),
            y[0:5] - np.zeros(5), y[5:] - np.ones(5),
            ca.horzcat(z[0:5] - np.array([2, 2, 2, 2, 2]), z[5:10] - np.array([1, 1, 1, 1, 1]))]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_arrayexpressions(self):
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
        g = ca.MX.sym("g", 1)
        h = ca.MX.sym("h", 1)
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

        ref_model.alg_states = [a, c, d, e, scalar_f, g, arx, arcy, arcw, nested1z, nested2z, h]
        ref_model.parameters = [d_dim, nested1n, nested2n]
        ref_model.outputs = [h]
        ref_model.constants = [b, c_dim, B, C, D, E]
        ref_model.constant_values = [np.array([2.7, 3.7, 4.7, 5.7]), 2, ca.linspace(1, 2, 3), 1.7 * ca.DM.ones(2),
                                     ca.DM.zeros(3), ca.DM.ones(2)]
        ref_model.equations = [c - (a + b[0:3] * e), d - (ca.sin(a / b[1:4])), e - (d + scalar_f), g - ca.sum1(c),
                               h - B[1], arx[1] - scalar_f, nested1z - ca.DM.ones(3), nested2z[0, :].T - ca.DM.zeros(3),
                               nested2z[1, 0] - 3, nested2z[1, 1] - 2, nested2z[1, 2] - 1, arcy[0] - arcy[1],
                               arcw[0] + arcw[1]]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_matrixexpressions(self):
        with open(os.path.join(TEST_DIR, 'MatrixExpressions.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'MatrixExpressions')
        print(casadi_model)
        ref_model = CasadiSysModel()

        A = ca.MX.sym("A", 3, 3)
        b = ca.MX.sym("b", 3)
        c = ca.MX.sym("c", 3)
        d = ca.MX.sym("d", 3)
        C = ca.MX.sym("C", 2, 3)
        D = ca.MX.sym("D", 3, 2)
        E = ca.MX.sym("E", 2, 3)
        I = ca.MX.sym("I", 5, 5)
        F = ca.MX.sym("F", 3, 3)

        ref_model.alg_states = [A, b, c, d]
        ref_model.constants = [C, D, E, I, F]
        ref_model.constant_values = [1.7 * ca.DM.ones(2, 3), ca.DM.zeros(3, 2), ca.DM.ones(2, 3), ca.DM.eye(5),
                                     ca.DM.triplet([0, 1, 2], [0, 1, 2], [1, 2, 3], 3, 3)]
        ref_model.equations = [ca.mtimes(A, b) - c, ca.mtimes(A.T, b) - d, F[1, 2]]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_attributes(self):
        with open(os.path.join(TEST_DIR, 'Attributes.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'Attributes')
        print(casadi_model)
        ref_model = CasadiSysModel()

        i = ca.MX.sym("int")
        b = ca.MX.sym("bool")
        r = ca.MX.sym("real")
        der_r = ca.MX.sym("der(real)")
        i1 = ca.MX.sym("i1")
        i2 = ca.MX.sym("i2")
        i3 = ca.MX.sym("i3")
        i4 = ca.MX.sym("i4")
        cst = ca.MX.sym("cst")
        prm = ca.MX.sym("prm")
        protected_variable = ca.MX.sym("protected_variable")

        ref_model.states = [r]
        ref_model.der_states = [der_r]
        ref_model.alg_states = [i, b, i1, i2, i3, i4, protected_variable]
        ref_model.inputs = [i1, i2, i3]
        ref_model.outputs = [i4, protected_variable]
        ref_model.constants = [cst]
        ref_model.constant_values = [1]
        ref_model.parameters = [prm]
        ref_model.equations = [i4 - ((i1 + i2) + i3), der_r - (i1 + ca.if_else(b, 1, 0) * i),
                               protected_variable - (i1 + i2)]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_type(self):
        with open(os.path.join(TEST_DIR, 'Type.mo'), 'r') as f:
            txt = f.read()
        # noinspection PyUnusedLocal
        ast_tree = parser.parse(txt)
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
