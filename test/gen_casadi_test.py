#!/usr/bin/env python
"""
Modelica parse Tree to AST tree.
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import os
import unittest
import itertools

import casadi as ca
import numpy as np

import pymola.backends.casadi.generator as gen_casadi
from pymola.backends.casadi.model import Model, Variable
from pymola.backends.casadi.api import transfer_model, CachedModel
from pymola import parser

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
        self.assertEqual(len(A.parameters), len(B.parameters))

        if not isinstance(A, CachedModel) and not isinstance(B, CachedModel):
            self.assertEqual(len(A.equations), len(B.equations))
            self.assertEqual(len(A.initial_equations), len(B.initial_equations))

        # TODO test this using metadata function.
        for a, b in zip(A.constants, B.constants):
            delta = ca.vec(a.value - b.value)
            for i in range(delta.size1()):
                test = float(delta[i]) <= tol
                self.assertTrue(test)

        for (f_name, m_name) in [('dae_residual', 'equations'), ('initial_residual', 'initial_equations')]:
            this = getattr(A, f_name + '_function')
            that = getattr(B, f_name + '_function')

            if not isinstance(A, CachedModel) and not isinstance(B, CachedModel):
                # Since arguments are grouped, we first split them into their constituent parts.
                # This is to render the test insensitive to the order in which arguments are declared.
                this_mx = [[e.dep(i) for i in range(e.n_dep())] if e.is_op(ca.OP_VERTCAT) else [e] for e in this.mx_in()]
                that_mx = [[e.dep(i) for i in range(e.n_dep())] if e.is_op(ca.OP_VERTCAT) else [e] for e in that.mx_in()]
                this_in = [[repr(e) for e in l] for l in this_mx]
                that_in = [[repr(e) for e in l] for l in that_mx]

                that_from_this = []
                this_mx_dict = dict(zip(itertools.chain(*this_in), itertools.chain(*this_mx)))
                that_mx_dict = dict(zip(itertools.chain(*that_in), itertools.chain(*that_mx)))
                for this_l, that_l in zip(this_in, that_in):
                    self.assertEqual(set(this_l), set(that_l))
                    for e in this_l:
                        self.assertEqual(this_mx_dict[e].size1(), that_mx_dict[e].size1())
                        self.assertEqual(this_mx_dict[e].size2(), that_mx_dict[e].size2())
                    that_from_this.append(ca.vertcat(*[this_mx_dict[e] for e in that_l]))
                that = ca.Function('f', [ca.vertcat(*l) for l in this_mx], that.call(that_from_this))

            np.random.seed(0)

            args_in = []
            for i in range(this.n_in()):
                sp = this.sparsity_in(0)
                r = ca.DM(sp, np.random.random(sp.nnz()))
                args_in.append(r)

            this_out = this.call(args_in)
            that_out = that.call(args_in)

            # TODO order
            for i, (a, b) in enumerate(zip(this_out, that_out)):
                test = float(ca.norm_2(ca.vec(a - b))) <= tol
                if not test:
                    print("Expr mismatch")
                    print("A: ", getattr(A, m_name)[i], a)
                    print("B: ", getattr(B, m_name)[i], b)
                self.assertTrue(test)

        return True

    def test_spring(self):
        with open(os.path.join(TEST_DIR, 'Spring.mo'), 'r') as f:
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
        ref_model.equations = [der_x - v_x, der_v_x - (-k * x - c * v_x)]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_estimator(self):
        with open(os.path.join(TEST_DIR, 'Estimator.mo'), 'r') as f:
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
        with open(os.path.join(TEST_DIR, 'Aircraft.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        # noinspection PyUnusedLocal
        casadi_model = gen_casadi.generate(ast_tree, 'Aircraft')
        # noinspection PyUnusedLocal
        ref_model = Model()
        self.assertTrue(True)

    def test_connector(self):
        with open(os.path.join(TEST_DIR, 'ConnectorHQ.mo'), 'r') as f:
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
        qc__down__H = ca.MX.sym("qc.down.H")
        qc__down__Q = ca.MX.sym("qc.down.Q")

        hb__up__H = ca.MX.sym("hb.up.H")
        hb__up__Q = ca.MX.sym("hb.up.Q")

        ref_model.alg_states = map(Variable, [qc__down__H, a__down__H, b__down__H, c__down__H, c__up__H, hb__up__H, a__up__H,
                                b__up__H, qa__down__H, a__up__Q, qa__down__Q, c__down__Q, hb__up__Q, c__up__Q, b__up__Q,
                                b__down__Q, qc__down__Q, a__down__Q])

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

        qa__down__H = ca.MX.sym("qa.down.H")
        qa__down__Q = ca.MX.sym("qa.down.Q")
        qa__down__Z = ca.MX.sym("qa.down.Z")
        qc__down__H = ca.MX.sym("qc.down.H")
        qc__down__Q = ca.MX.sym("qc.down.Q")
        qc__down__Z = ca.MX.sym("qc.down.Z")

        hb__up__H = ca.MX.sym("hb.up.H")
        hb__up__Q = ca.MX.sym("hb.up.Q")
        hb__up__Z = ca.MX.sym("hb.up.Z")

        ref_model.alg_states = list(map(Variable, [qc__down__H, a__down__H, b__down__H, c__down__H, c__up__H, hb__up__H, a__up__H,
                                b__up__H, qa__down__H, a__up__Q, qa__down__Q, c__down__Q, hb__up__Q, c__up__Q, b__up__Q,
                                b__down__Q, qc__down__Q, a__down__Q, a__up__Z, a__down__Z, b__up__Z, b__down__Z,
                                c__up__Z, c__down__Z, qa__down__Z, qc__down__Z, hb__up__Z]))

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
        with open(os.path.join(TEST_DIR, 'IfElse.mo'), 'r') as f:
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
        ref_model = Model()

        x = ca.MX.sym("x")
        der_x = ca.MX.sym("der(x)")
        y = ca.MX.sym("y")
        # noinspection PyUnusedLocal
        der_y = ca.MX.sym("y")
        k = ca.MX.sym("k")

        ref_model.states = list(map(Variable, [x]))
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [y]))
        ref_model.parameters = list(map(Variable, [k]))
        ref_model.equations = [der_x - k * x, x + y - 3]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_inheritance_instantiation(self):
        with open(os.path.join(TEST_DIR, 'InheritanceInstantiation.mo'), 'r') as f:
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
        ref_model.equations = []

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_builtin(self):
        with open(os.path.join(TEST_DIR, 'BuiltinFunctions.mo'), 'r') as f:
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

    def test_forloop(self):
        with open(os.path.join(TEST_DIR, 'ForLoop.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'ForLoop')
        print(casadi_model)
        ref_model = Model()

        x = ca.MX.sym("x", 10)
        y = ca.MX.sym("y", 10)
        z = ca.MX.sym("z", 10)
        w = ca.MX.sym('w', 2, 10)
        b = ca.MX.sym("b")
        n = ca.MX.sym("n")

        ref_model.alg_states = list(map(Variable, [x, y, z, w, b]))
        ref_model.parameters = list(map(Variable, [n]))
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
        ref_model = Model()

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

        ref_model.alg_states = list(map(Variable, [arx, arcy, arcw, nested1z, nested2z, a, c, d, e, scalar_f, g, h]))
        ref_model.parameters = list(map(Variable, [nested2n, nested1n, d_dim]))
        ref_model.outputs = list(map(Variable, [h]))
        ref_model.constants = list(map(Variable, [b, c_dim, B, C, D, E]))
        constant_values = [np.array([2.7, 3.7, 4.7, 5.7]), 2, ca.linspace(1, 2, 3), 1.7 * ca.DM.ones(2),
                                     ca.DM.zeros(3), ca.DM.ones(2)]
        for const, val in zip(ref_model.constants, constant_values):
            const.value = val
        ref_model.equations = [c - (a + b[0:3] * e), d - (ca.sin(a / b[1:4])), e - (d + scalar_f), g - ca.sum1(c),
                               h - B[1], arx[1] - scalar_f, nested1z - ca.DM.ones(3), nested2z[0, :].T - ca.DM.zeros(3),
                               nested2z[1, 0] - 3, nested2z[1, 1] - 2, nested2z[1, 2] - 1, arcy[0] - arcy[1],
                               arcw[0] + arcw[1], a - np.array([1, 2, 3]), scalar_f - 1.3]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_matrixexpressions(self):
        with open(os.path.join(TEST_DIR, 'MatrixExpressions.mo'), 'r') as f:
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

    def test_attributes(self):
        with open(os.path.join(TEST_DIR, 'Attributes.mo'), 'r') as f:
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
        ref_model.der_states = list(map(Variable, [der_r]))
        ref_model.alg_states = list(map(Variable, [nested_s, i, b, i4, test_state, protected_variable]))
        ref_model.inputs = list(map(Variable, [i1, i2, i3]))
        ref_model.outputs = list(map(Variable, [i4, protected_variable]))
        ref_model.constants = list(map(Variable, [cst]))
        constant_values = [1]
        for c, v in zip(ref_model.constants, constant_values):
            c.value = v
        ref_model.parameters = list(map(Variable, [nested_p1, nested_p, prm]))
        parameter_values = [2, 2 * nested_p1, 1]
        for c, v in zip(ref_model.parameters, parameter_values):
            c.value = v
        ref_model.equations = [i4 - ((i1 + i2) + i3), der_r - (i1 + ca.if_else(b, 1, 0) * i),
                               protected_variable - (i1 + i2), nested_s - 3 * nested_p, test_state - r]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_type(self):
        with open(os.path.join(TEST_DIR, 'Type.mo'), 'r') as f:
            txt = f.read()
        # noinspection PyUnusedLocal
        ast_tree = parser.parse(txt)
        self.assertTrue(True)

    def test_caching(self):
        # Clear cache
        shelve_file = os.path.join(TEST_DIR, 'Aircraft')
        try:
            os.remove(shelve_file)
        except:
            pass

        # Create model, cache it, and load the cache
        compiler_options = \
            {'cache': True}

        ref_model = transfer_model(TEST_DIR, 'Aircraft', compiler_options)
        cached_model = transfer_model(TEST_DIR, 'Aircraft', compiler_options)

        # Compare
        self.assert_model_equivalent_numeric(ref_model, cached_model)

    def test_simplify_replace_constant_values(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'replace_constant_values': True}

        casadi_model = transfer_model(TEST_DIR, 'Simplify', compiler_options)

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

        ref_model.states = list(map(Variable, [x]))
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [alias, y, _tmp]))
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = list(map(Variable, []))
        ref_model.constants = list(map(Variable, []))
        constant_values = []
        for cst, v in zip(ref_model.constants, constant_values):
            cst.value = v
        ref_model.parameters = list(map(Variable, [p1, p2, p3, p4]))
        parameter_values = [2.0, 2 * p1, np.nan, 2 * p3]
        for par, v in zip(ref_model.parameters, parameter_values):
            par.value = v
        ref_model.equations = [der_x - x - p1 - p2 - p3 - p4, alias - x, y - x - 3 - _tmp, _tmp - 0.1 * x]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)

    def test_simplify_replace_parameter_expressions(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'replace_parameter_expressions': True}

        casadi_model = transfer_model(TEST_DIR, 'Simplify', compiler_options)

        ref_model = Model()

        c = ca.MX.sym('c')
        p1 = ca.MX.sym('p1')
        p3 = ca.MX.sym('p3')
        x = ca.MX.sym('x')
        der_x = ca.MX.sym('der(x)')
        alias = ca.MX.sym('alias')
        y = ca.MX.sym('y')
        _tmp = ca.MX.sym('_tmp')

        ref_model.states = list(map(Variable, [x]))
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [alias, y, _tmp]))
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = list(map(Variable, []))
        ref_model.constants = list(map(Variable, [c]))
        constant_values = [3]
        for cst, v in zip(ref_model.constants, constant_values):
            cst.value = v
        ref_model.parameters = list(map(Variable, [p1, p3]))
        parameter_values = [2.0, np.nan]
        for par, v in zip(ref_model.parameters, parameter_values):
            par.value = v
        ref_model.equations = [der_x - x - 3 * p1 - 3 * p3, alias - x, y - x - c - _tmp, _tmp - 0.1 * x]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)

    def test_simplify_replace_parameter_values(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'replace_parameter_values': True}

        casadi_model = transfer_model(TEST_DIR, 'Simplify', compiler_options)

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

        ref_model.states = list(map(Variable, [x]))
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [alias, y, _tmp]))
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = list(map(Variable, []))
        ref_model.constants = list(map(Variable, [c]))
        constant_values = [3]
        for cst, v in zip(ref_model.constants, constant_values):
            cst.value = v
        ref_model.parameters = list(map(Variable, [p2, p3, p4]))
        parameter_values = [4, np.nan, 2 * p3]
        for par, v in zip(ref_model.parameters, parameter_values):
            par.value = v
        ref_model.equations = [der_x - x - 2 - p2 - p3 - p4, alias - x, y - x - c - _tmp, _tmp - 0.1 * x]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)

    def test_simplify_replace_parameter_values_and_expressions(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'replace_parameter_values': True,
             'replace_parameter_expressions': True}

        casadi_model = transfer_model(TEST_DIR, 'Simplify', compiler_options)

        ref_model = Model()

        c = ca.MX.sym('c')
        p3 = ca.MX.sym('p3')
        x = ca.MX.sym('x')
        der_x = ca.MX.sym('der(x)')
        alias = ca.MX.sym('alias')
        y = ca.MX.sym('y')
        _tmp = ca.MX.sym('_tmp')

        ref_model.states = list(map(Variable, [x]))
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [alias, y, _tmp]))
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = list(map(Variable, []))
        ref_model.constants = list(map(Variable, [c]))
        constant_values = [3]
        for cst, v in zip(ref_model.constants, constant_values):
            cst.value = v
        ref_model.parameters = list(map(Variable, [p3]))
        parameter_values = [np.nan]
        for par, v in zip(ref_model.parameters, parameter_values):
            par.value = v
        ref_model.equations = [der_x - x - 6 - 3 * p3, alias - x, y - x - c - _tmp, _tmp - 0.1 * x]

        print(casadi_model)
        print(ref_model)
        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)

    def test_simplify_eliminable_variable_expression(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'eliminable_variable_expression': r'_\w+'}

        casadi_model = transfer_model(TEST_DIR, 'Simplify', compiler_options)

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

        ref_model.states = list(map(Variable, [x]))
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [alias, y]))
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = list(map(Variable, []))
        ref_model.constants = list(map(Variable, [c]))
        constant_values = [3]
        for cst, v in zip(ref_model.constants, constant_values):
            cst.value = v
        ref_model.parameters = list(map(Variable, [p1, p2, p3, p4]))
        parameter_values = [2.0, 2 * p1, np.nan, 2 * p3]
        for par, v in zip(ref_model.parameters, parameter_values):
            par.value = v
        ref_model.equations = [der_x - x - p1 - p2 - p3 - p4, alias - x, y - x - c - 0.1 * x]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)

    def test_simplify_detect_aliases(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'detect_aliases': True}

        casadi_model = transfer_model(TEST_DIR, 'Simplify', compiler_options)

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

        ref_model.states = list(map(Variable, [x]))
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [y, _tmp]))
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = list(map(Variable, []))
        ref_model.constants = list(map(Variable, [c]))
        constant_values = [3]
        for cst, v in zip(ref_model.constants, constant_values):
            cst.value = v
        ref_model.parameters = list(map(Variable, [p1, p2, p3, p4]))
        parameter_values = [2.0, 2 * p1, np.nan, 2 * p3]
        for par, v in zip(ref_model.parameters, parameter_values):
            par.value = v
        ref_model.equations = [der_x - x - p1 - p2 - p3 - p4, y - x - c - _tmp, _tmp - 0.1 * x]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)
        self.assertEquals(casadi_model.states[0].aliases, ['alias'])

    def test_simplify_reduce_affine_expression(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'reduce_affine_expression': True}

        casadi_model = transfer_model(TEST_DIR, 'Simplify', compiler_options)

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

        ref_model.states = list(map(Variable, [x]))
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [alias, y, _tmp]))
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = list(map(Variable, []))
        ref_model.constants = list(map(Variable, [c]))
        constant_values = [3]
        for cst, v in zip(ref_model.constants, constant_values):
            cst.value = v
        ref_model.parameters = list(map(Variable, [p1, p2, p3, p4]))
        parameter_values = [2.0, 2 * p1, np.nan, 2 * p3]
        for par, v in zip(ref_model.parameters, parameter_values):
            par.value = v

        A = ca.MX(4, 5)
        A[0, 0] = -1.0
        A[0, 1] = 1.0
        A[1, 2] = 1.0
        A[1, 0] = -1.0
        A[2, 0] = -1.0
        A[2, 3] = 1.0
        A[2, 4] = -1.0
        A[3, 0] = -0.1
        A[3, 4] = 1.0
        b = ca.MX(4, 1)
        b[0] = -p1 - p2 - p3 - p4
        b[2] = -c
        x = ca.vertcat(x, der_x, alias, y, _tmp)
        ref_model.equations = [ca.mtimes(A, x) + b]
            
        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)

    def test_simplify_all(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'replace_constant_values': True,
             'replace_parameter_values': True,
             'replace_parameter_expressions': True,
             'detect_aliases': True,
             'eliminable_variable_expression': r'_\w+',
             'reduce_affine_expression': True}

        casadi_model = transfer_model(TEST_DIR, 'Simplify', compiler_options)

        ref_model = Model()

        p3 = ca.MX.sym('p3')
        x = ca.MX.sym('x')
        der_x = ca.MX.sym('der(x)')
        y = ca.MX.sym('y')

        ref_model.states = list(map(Variable, [x]))
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [y]))
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = list(map(Variable, []))
        ref_model.constants = list(map(Variable, []))
        constant_values = []
        for cst, v in zip(ref_model.constants, constant_values):
            cst.value = v
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
        b[1] = -3
        x = ca.vertcat(x, der_x, y)
        ref_model.equations = [ca.mtimes(A, x) + b]
            
        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)

    def test_simplify_expand(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'expand': True}

        casadi_model = transfer_model(TEST_DIR, 'Simplify', compiler_options)

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

        ref_model.states = list(map(Variable, [x]))
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [alias, y, _tmp]))
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = list(map(Variable, []))
        ref_model.constants = list(map(Variable, [c]))
        constant_values = [3]
        for cst, v in zip(ref_model.constants, constant_values):
            cst.value = v
        ref_model.parameters = list(map(Variable, [p1, p2, p3, p4]))
        parameter_values = [2.0, 2 * p1, np.nan, 2 * p3]
        for par, v in zip(ref_model.parameters, parameter_values):
            par.value = v
        ref_model.equations = [der_x - x - p1 - p2 - p3 - p4, alias - x, y - x - c - _tmp, _tmp - 0.1 * x]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)


if __name__ == "__main__":
    unittest.main()
