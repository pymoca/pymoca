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

import pymola.backends.casadi.generator as gen_casadi
from pymola.backends.casadi.alias_relation import AliasRelation
from pymola.backends.casadi.model import Model, Variable
from pymola.backends.casadi.api import transfer_model, CachedModel
from pymola import parser, ast


TEST_DIR = os.path.dirname(os.path.realpath(__file__))


def MXArray(name, *dimensions):
    """
    Function with the same signature as `ca.MX.sym`, but instead returns
    a NumPy array of scalar MX symbols
    """
    if not dimensions:
        return np.array([ca.MX.sym(name)])

    arr = np.empty(dimensions, dtype=object)

    for ind, _ in np.ndenumerate(arr):
        arr[ind] = ca.MX.sym("{}[{}]".format(name, ", ".join((str(x + 1) for x in ind))))

    return arr


# noinspection PyPep8Naming,PyUnresolvedReferences
class GenCasadiTest(unittest.TestCase):
    def assert_model_equivalent(self, A, B):
        def sstr(a): return set([str(e) for e in a])

        for l in ["states", "der_states", "inputs", "outputs", "constants", "parameters"]:
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

        for f_name in ['dae_residual', 'initial_residual', 'variable_metadata']:
            this = getattr(A, f_name + '_function')
            that = getattr(B, f_name + '_function')

            np.random.seed(0)

            args_in = []
            for i in range(this.n_in()):
                sp = this.sparsity_in(0)
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
        ref_model.parameters[0].value = 0.1
        ref_model.parameters[1].value = 2
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

    def test_connector_hq(self):
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

        p__H = ca.MX.sym("p.H")
        p__Q = ca.MX.sym("p.Q")

        hb__up__H = ca.MX.sym("hb.up.H")
        hb__up__Q = ca.MX.sym("hb.up.Q")

        zerotest__H = ca.MX.sym("zerotest.H")
        zerotest__Q = ca.MX.sym("zerotest.Q")

        ref_model.alg_states = list(map(Variable, [p__H, a__down__H, b__down__H, c__down__H, c__up__H, hb__up__H, a__up__H,
                                b__up__H, qa__down__H, a__up__Q, qa__down__Q, c__down__Q, hb__up__Q, c__up__Q, b__up__Q,
                                b__down__Q, p__Q, a__down__Q, zerotest__H, zerotest__Q]))

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

        ref_model.alg_states = list(map(Variable, [qc__down__H, a__down__H, b__down__H, c__down__H, c__up__H, hb__up__H, a__up__H,
                                b__up__H, qa__down__H, a__up__Q, qa__down__Q, c__down__Q, hb__up__Q, c__up__Q, b__up__Q,
                                b__down__Q, qc__down__Q, a__down__Q, a__up__Z, a__down__Z, b__up__Z, b__down__Z,
                                c__up__Z, c__down__Z, d__up__H, d__up__Q, d__down__Q, d__down__H, d__down__Z, qa__down__Z, qc__down__Z, hb__up__Z]))

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
        with open(os.path.join(TEST_DIR, 'TreeLookup.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)

        casadi_model = gen_casadi.generate(ast_tree, 'Level1.Level2.Level3.Test')
        ref_model = Model()
        print(casadi_model)

        elem__tc__i = ca.MX.sym("elem.tc.i")
        elem__tc__a = ca.MX.sym("elem.tc.a")
        b = ca.MX.sym("b")

        ref_model.alg_states = map(Variable, [elem__tc__i, elem__tc__a, b])

        ref_model.equations = [elem__tc__i - 1,
                               elem__tc__a - b]

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
        ref_model.parameters[0].value = 10
        ref_model.equations = [
            y1 - ca.if_else(x > 0, 1, 0) * y_max,
            ca.if_else(x > 1,
                           y2 - y_max,
                           ca.if_else(x > 2,
                               y2 - y_max - 1,
                               y2)),
            ca.if_else(x > 1,
                           y3 - 100,
                           ca.if_else(x > 2,
                               y3 - 1000,
                               y3 - 10000))
            ]

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
        ref_model.states[0].max = 30.0
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [y]))
        ref_model.parameters = list(map(Variable, [k]))
        ref_model.parameters[0].value = -1.0
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
        ref_model.parameters[0].value = 0
        ref_model.parameters[1].value = 0
        ref_model.parameters[2].value = 1
        ref_model.parameters[3].value = 3
        ref_model.parameters[4].value = 4
        ref_model.parameters[5].value = 2
        ref_model.equations = []

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_nested_classes(self):
        with open(os.path.join(TEST_DIR, 'NestedClasses.mo'), 'r') as f:
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

    def test_function_call(self):
        with open(os.path.join(TEST_DIR, 'FunctionCall.mo'), 'r') as f:
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
        ref_model.alg_states = list(map(Variable, [c, a, d, e, S1, S2, r]))
        ref_model.outputs = list(map(Variable, [c, a, d, e, S1, S2]))
        ref_model.equations = [ca.vertcat(c, a, d, e, S1, S2) - ca.vertcat(*circle_properties.call([r])[0:-1])]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    @unittest.skip
    def test_double_function_call(self):
        with open(os.path.join(TEST_DIR, 'DoubleFunctionCall.mo'), 'r') as f:
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
        with open(os.path.join(TEST_DIR, 'ForLoop.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'ForLoop')
        print(casadi_model)
        ref_model = Model()

        x = MXArray("x", 10)
        y = MXArray("y", 10)
        z = MXArray("z", 10)
        u = MXArray('u', 10, 2)
        v = MXArray('v', 2, 10)
        w = MXArray('w', 2, 10)
        b = MXArray("b")
        n = MXArray("n")
        s = MXArray('s', 10)
        Arr = MXArray('Arr', 2, 2)
        der_s = MXArray('der(s)', 10)

        ref_model.states = list(map(Variable, [*s]))
        ref_model.der_states = list(map(Variable, [*der_s]))
        ref_model.alg_states = list(map(Variable, [*x, *y, *z, *u.flatten(), *v.flatten(), *w.flatten(), *b, *Arr.flatten()]))
        ref_model.parameters = list(map(Variable, [*n]))
        ref_model.parameters[0].value = 10

        i = np.arange(1, 11)
        ref_model.equations = [
            *(x - (i + b)),
            *(w[0, :] - i),
            *(w[1, :] - 2 * i),
            *(u - 1).flatten(),
            *(v - 1).T.flatten(),
            *y[0:5],
            *(y[5:10] - 1),
            *(z[0:5] - 2),
            *(z[5:10] - 1),
            *(der_s - 1),
            *(Arr[0:2, 1] - 2),
            *(Arr[0:2, 0] - 1)]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_arrayexpressions(self):
        with open(os.path.join(TEST_DIR, 'ArrayExpressions.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'ArrayExpressions')
        print(casadi_model)
        ref_model = Model()

        a_1 = ca.MX.sym("a[1]")
        a_2 = ca.MX.sym("a[2]")
        a_3 = ca.MX.sym("a[3]")
        b_1 = ca.MX.sym("b[1]")
        b_2 = ca.MX.sym("b[2]")
        b_3 = ca.MX.sym("b[3]")
        b_4 = ca.MX.sym("b[4]")
        c_1 = ca.MX.sym("c[1]")
        c_2 = ca.MX.sym("c[2]")
        c_3 = ca.MX.sym("c[3]")
        d_1 = ca.MX.sym("d[1]")
        d_2 = ca.MX.sym("d[2]")
        d_3 = ca.MX.sym("d[3]")
        e_1 = ca.MX.sym("e[1]")
        e_2 = ca.MX.sym("e[2]")
        e_3 = ca.MX.sym("e[3]")
        g = ca.MX.sym("g")
        h = ca.MX.sym("h")
        i_1_1 = ca.MX.sym('i[1,1]')
        i_1_2 = ca.MX.sym('i[1,2]')
        i_1_3 = ca.MX.sym('i[1,3]')
        i_2_1 = ca.MX.sym('i[2,1]')
        i_2_2 = ca.MX.sym('i[2,2]')
        i_2_3 = ca.MX.sym('i[2,3]')

        B_1 = ca.MX.sym("B[1]")
        B_2 = ca.MX.sym("B[2]")
        B_3 = ca.MX.sym("B[3]")
        C_1 = ca.MX.sym("C[1]")
        C_2 = ca.MX.sym("C[2]")
        D_1 = ca.MX.sym("D[1]")
        D_2 = ca.MX.sym("D[2]")
        D_3 = ca.MX.sym("D[3]")
        E_1 = ca.MX.sym("E[1]")
        E_2 = ca.MX.sym("E[2]")

        ar_x_1 = ca.MX.sym("ar.x[1]")
        ar_x_2 = ca.MX.sym("ar.x[2]")
        ar_x_3 = ca.MX.sym("ar.x[3]")
        arcy_1 = ca.MX.sym("arc.y[1]")
        arcy_2 = ca.MX.sym("arc.y[2]")
        arcw_1 = ca.MX.sym("arc.w[1]")
        arcw_2 = ca.MX.sym("arc.w[2]")
        nested1z_1 = ca.MX.sym('nested1.z[1]')
        nested1z_2 = ca.MX.sym('nested1.z[2]')
        nested1z_3 = ca.MX.sym('nested1.z[3]')
        nested2z_1_1 = ca.MX.sym('nested2.z[1,1]')
        nested2z_1_2 = ca.MX.sym('nested2.z[1,2]')
        nested2z_1_3 = ca.MX.sym('nested2.z[1,3]')
        nested2z_2_1 = ca.MX.sym('nested2.z[2,1]')
        nested2z_2_2 = ca.MX.sym('nested2.z[2,2]')
        nested2z_2_3 = ca.MX.sym('nested2.z[2,3]')
        nested1n = ca.MX.sym('nested1.n')
        nested2n_1 = ca.MX.sym('nested2.n[1]')
        nested2n_2 = ca.MX.sym('nested2.n[2]')

        scalar_f = ca.MX.sym("scalar_f")
        c_dim = ca.MX.sym("c_dim")
        d_dim = ca.MX.sym("d_dim")

        ref_model.alg_states = list(map(Variable, [
            ar_x_1, ar_x_2, ar_x_3,
            arcy_1, arcy_2,
            arcw_1, arcw_2,
            nested1z_1, nested1z_2,
            nested1z_3, nested2z_1_1, nested2z_1_2, nested2z_1_3, nested2z_2_1, nested2z_2_2, nested2z_2_3,
            a_1, a_2, a_3,
            c_1, c_2, c_3,
            d_1, d_2, d_3,
            e_1, e_2, e_3,
            scalar_f,
            g,
            h,
            i_1_1, i_1_2, i_1_3, i_2_1, i_2_2, i_2_3]))

        for i in range(19, 22):
            ref_model.alg_states[i].min = 0.0

        ref_model.parameters = list(map(Variable, [nested2n_1, nested2n_2, nested1n, d_dim]))
        parameter_values = [3, 3, 3, 3]
        for const, val in zip(ref_model.parameters, parameter_values):
            const.value = val
        ref_model.outputs = list(map(Variable, [h]))
        ref_model.constants = list(map(Variable, [b_1, b_2, b_3, b_4, c_dim, B_1, B_2, B_3, C_1, C_2, D_1, D_2, D_3, E_1, E_2]))
        constant_values = [2.7, 3.7, 4.7, 5.7, 2, 1.0, 1.5, 2.0, 1.7, 1.7, 0.0, 0.0, 0.0, 1.0, 1.0]
        for const, val in zip(ref_model.constants, constant_values):
            const.value = val

        ref_model.equations = [c_1 - (a_1 + b_1 * e_1),
                               c_2 - (a_2 + b_2 * e_2),
                               c_3 - (a_3 + b_3 * e_3),

                               d_1 - (ca.sin(a_1 / b_2)),
                               d_2 - (ca.sin(a_2 / b_3)),
                               d_3 - (ca.sin(a_3 / b_4)),

                               e_1 - (d_1 + scalar_f),
                               e_2 - (d_2 + scalar_f),
                               e_3 - (d_3 + scalar_f),

                               g - (c_1 + c_2 + c_3),

                               h - B_2,

                               ar_x_2 - scalar_f,

                               nested1z_1 - 1,
                               nested1z_2 - 1,
                               nested1z_3 - 1,
                               nested2z_1_1 - 4,
                               nested2z_1_2 - 5,
                               nested2z_1_3 - 6,
                               nested2z_2_1 - 3,
                               nested2z_2_2 - 2,
                               nested2z_2_3 - 1,

                               i_1_1 - 1,
                               i_1_2 - 1,
                               i_1_3 - 1,
                               i_2_1 - 1,
                               i_2_2 - 1,
                               i_2_3 - 1,

                               arcy_1 - arcy_2,
                               arcw_1 + arcw_2,

                               a_1 - 1,
                               a_2 - 2,
                               a_3 - 3,

                               scalar_f - 1.3]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_matrixexpressions(self):
        with open(os.path.join(TEST_DIR, 'MatrixExpressions.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'MatrixExpressions')
        print(casadi_model)
        ref_model = Model()

        A_1_1 = ca.MX.sym("A[1,1]")
        A_1_2 = ca.MX.sym("A[1,2]")
        A_1_3 = ca.MX.sym("A[1,3]")
        A_2_1 = ca.MX.sym("A[2,1]")
        A_2_2 = ca.MX.sym("A[2,2]")
        A_2_3 = ca.MX.sym("A[2,3]")
        A_3_1 = ca.MX.sym("A[3,1]")
        A_3_2 = ca.MX.sym("A[3,2]")
        A_3_3 = ca.MX.sym("A[3,3]")
        b_1 = ca.MX.sym("b[1]")
        b_2 = ca.MX.sym("b[2]")
        b_3 = ca.MX.sym("b[3]")
        c_1 = ca.MX.sym("c[1]")
        c_2 = ca.MX.sym("c[2]")
        c_3 = ca.MX.sym("c[3]")
        d_1 = ca.MX.sym("d[1]")
        d_2 = ca.MX.sym("d[2]")
        d_3 = ca.MX.sym("d[3]")
        C_1_1 = ca.MX.sym("C[1,1]")
        C_1_2 = ca.MX.sym("C[1,2]")
        C_1_3 = ca.MX.sym("C[1,3]")
        C_2_1 = ca.MX.sym("C[2,1]")
        C_2_2 = ca.MX.sym("C[2,2]")
        C_2_3 = ca.MX.sym("C[2,3]")
        D_1_1 = ca.MX.sym("D[1,1]")
        D_1_2 = ca.MX.sym("D[1,2]")
        D_2_1 = ca.MX.sym("D[2,1]")
        D_2_2 = ca.MX.sym("D[2,2]")
        D_3_1 = ca.MX.sym("D[3,1]")
        D_3_2 = ca.MX.sym("D[3,2]")
        E_1_1 = ca.MX.sym("E[1,1]")
        E_1_2 = ca.MX.sym("E[1,2]")
        E_1_3 = ca.MX.sym("E[1,3]")
        E_2_1 = ca.MX.sym("E[2,1]")
        E_2_2 = ca.MX.sym("E[2,2]")
        E_2_3 = ca.MX.sym("E[2,3]")
        I_1_1 = ca.MX.sym("I[1,1]")
        I_1_2 = ca.MX.sym("I[1,2]")
        I_1_3 = ca.MX.sym("I[1,3]")
        I_1_4 = ca.MX.sym("I[1,4]")
        I_1_5 = ca.MX.sym("I[1,5]")
        I_2_1 = ca.MX.sym("I[2,1]")
        I_2_2 = ca.MX.sym("I[2,2]")
        I_2_3 = ca.MX.sym("I[2,3]")
        I_2_4 = ca.MX.sym("I[2,4]")
        I_2_5 = ca.MX.sym("I[2,5]")
        I_3_1 = ca.MX.sym("I[3,1]")
        I_3_2 = ca.MX.sym("I[3,2]")
        I_3_3 = ca.MX.sym("I[3,3]")
        I_3_4 = ca.MX.sym("I[3,4]")
        I_3_5 = ca.MX.sym("I[3,5]")
        I_4_1 = ca.MX.sym("I[4,1]")
        I_4_2 = ca.MX.sym("I[4,2]")
        I_4_3 = ca.MX.sym("I[4,3]")
        I_4_4 = ca.MX.sym("I[4,4]")
        I_4_5 = ca.MX.sym("I[4,5]")
        I_5_1 = ca.MX.sym("I[5,1]")
        I_5_2 = ca.MX.sym("I[5,2]")
        I_5_3 = ca.MX.sym("I[5,3]")
        I_5_4 = ca.MX.sym("I[5,4]")
        I_5_5 = ca.MX.sym("I[5,5]")
        F_1_1 = ca.MX.sym("F[1,1]")
        F_1_2 = ca.MX.sym("F[1,2]")
        F_1_3 = ca.MX.sym("F[1,3]")
        F_2_1 = ca.MX.sym("F[2,1]")
        F_2_2 = ca.MX.sym("F[2,2]")
        F_2_3 = ca.MX.sym("F[2,3]")
        F_3_1 = ca.MX.sym("F[3,1]")
        F_3_2 = ca.MX.sym("F[3,2]")
        F_3_3 = ca.MX.sym("F[3,3]")

        ref_model.alg_states = list(map(Variable, [
            A_1_1, A_1_2, A_1_3, A_2_1, A_2_2, A_2_3, A_3_1, A_3_2, A_3_3,
            b_1, b_2, b_3,
            c_1, c_2, c_3,
            d_1, d_2, d_3]))

        ref_model.constants = list(map(Variable, [
            C_1_1, C_1_2, C_1_3, C_2_1, C_2_2, C_2_3,
            D_1_1, D_1_2, D_2_1, D_2_2, D_3_1, D_3_2,
            E_1_1, E_1_2, E_1_3, E_2_1, E_2_2, E_2_3,
            I_1_1, I_1_2, I_1_3, I_1_4, I_1_5,
            I_2_1, I_2_2, I_2_3, I_2_4, I_2_5,
            I_3_1, I_3_2, I_3_3, I_3_4, I_3_5,
            I_4_1, I_4_2, I_4_3, I_4_4, I_4_5,
            I_5_1, I_5_2, I_5_3, I_5_4, I_5_5,
            F_1_1, F_1_2, F_1_3, F_2_1, F_2_2, F_2_3, F_3_1, F_3_2, F_3_3]))

        constant_values = [1.7, 1.7, 1.7, 1.7, 1.7, 1.7,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0,

                           1.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 1.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 1.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 1.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 1.0,

                           1.0, 0.0, 0.0,
                           0.0, 2.0, 0.0,
                           0.0, 0.0, 3.0]

        for const, val in zip(ref_model.constants, constant_values):
            const.value = val
        ref_model.equations = [
            # A * b - c
            A_1_1 * b_1 + A_1_2 * b_2 + A_1_3 * b_3 - c_1,
            A_2_1 * b_1 + A_2_2 * b_2 + A_2_3 * b_3 - c_2,
            A_3_1 * b_1 + A_3_2 * b_2 + A_3_3 * b_3 - c_3,

            # A.T * b - d
            A_1_1 * b_1 + A_2_1 * b_2 + A_3_1 * b_3 - d_1,
            A_1_2 * b_1 + A_2_2 * b_2 + A_3_2 * b_3 - d_2,
            A_1_3 * b_1 + A_2_3 * b_2 + A_3_3 * b_3 - d_3,

            F_2_3]

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
        with open(os.path.join(TEST_DIR, 'Type.mo'), 'r') as f:
            txt = f.read()
        # noinspection PyUnusedLocal
        ast_tree = parser.parse(txt)
        self.assertTrue(True)

    def test_caching(self):
        # Clear cache
        db_file = os.path.join(TEST_DIR, 'Aircraft')
        try:
            os.remove(db_file)
        except:
            pass

        for f in glob.glob(os.path.join(TEST_DIR, "Aircraft*.so")):
            os.remove(f)
        for f in glob.glob(os.path.join(TEST_DIR, "Aircraft*.dll")):
            os.remove(f)
        for f in glob.glob(os.path.join(TEST_DIR, "Aircraft*.dylib")):
            os.remove(f)

        # Create model, cache it, and load the cache
        compiler_options = \
            {'cache': True}

        ref_model = transfer_model(TEST_DIR, 'Aircraft', compiler_options)
        self.assertIsInstance(ref_model, Model)
        self.assertNotIsInstance(ref_model, CachedModel)

        cached_model = transfer_model(TEST_DIR, 'Aircraft', compiler_options)
        self.assertIsInstance(cached_model, Model)
        self.assertIsInstance(cached_model, CachedModel)

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
        ref_model.outputs = list(map(Variable, []))
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
        ref_model.outputs = list(map(Variable, []))
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
        ref_model.outputs = list(map(Variable, []))
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

        casadi_model = transfer_model(TEST_DIR, 'Simplify', compiler_options)

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
        ref_model.outputs = list(map(Variable, []))
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
        ref_model.outputs = list(map(Variable, []))
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
        ref_model.outputs = list(map(Variable, []))
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
        cst = ca.MX.sym('cst')

        ref_model.states = list(map(Variable, [x]))
        ref_model.states[0].min = 1
        ref_model.states[0].max = 2
        ref_model.states[0].nominal = 10
        ref_model.der_states = list(map(Variable, [der_x]))
        ref_model.alg_states = list(map(Variable, [y, _tmp, cst]))
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = list(map(Variable, []))
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

        casadi_model = transfer_model(TEST_DIR, 'NegativeAlias', compiler_options)

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
        ref_model.outputs = list(map(Variable, []))
        ref_model.equations = [der_x - x]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)
        self.assertEquals(casadi_model.states[0].aliases, ['-alias'])

    def test_simplify_expand_vectors(self):
        # Create model, cache it, and load the cache
        compiler_options = \
            {'expand_vectors': True}

        casadi_model = transfer_model(TEST_DIR, 'SimplifyVector', compiler_options)

        ref_model = Model()

        x0 = ca.MX.sym('x[0]')
        x1 = ca.MX.sym('x[1]')
        der_x0 = ca.MX.sym('der(x)[0]')
        der_x1 = ca.MX.sym('der(x)[1]')

        ref_model.states = list(map(Variable, [x0, x1]))
        ref_model.der_states = list(map(Variable, [der_x0, der_x1]))
        ref_model.alg_states = list(map(Variable, []))
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = list(map(Variable, []))
        ref_model.equations = [der_x0 - x0, der_x1 - x1]

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

        casadi_model = transfer_model(TEST_DIR, 'SimplifyLoop', compiler_options)

        ref_model = Model()

        x = ca.MX.sym('x')
        y0 = ca.MX.sym('y[0]')
        y1 = ca.MX.sym('y[1]')

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
        ref_model.alg_states = list(map(Variable, [x, y0, y1]))
        ref_model.inputs = list(map(Variable, []))
        ref_model.outputs = list(map(Variable, []))
        x = ca.vertcat(x, y0, y1)
        ref_model.equations = [ca.mtimes(A, x) + b]

        # Compare
        self.assert_model_equivalent_numeric(casadi_model, ref_model)

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
        ref_model.outputs = list(map(Variable, []))
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

        casadi_model = transfer_model(TEST_DIR, 'Simplify', compiler_options)

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
        ref_model.outputs = list(map(Variable, []))
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
        ref_model.outputs = list(map(Variable, []))
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
        with open(os.path.join(TEST_DIR, 'StateAnnotator.mo'), 'r') as f:
            txt = f.read()
        ast_tree = parser.parse(txt)
        casadi_model = gen_casadi.generate(ast_tree, 'StateAnnotator')
        print(casadi_model)
        ref_model = Model()

        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        z = ca.MX.sym('z')
        der_x = ca.MX.sym('der(x)')
        der_y = ca.MX.sym('der(y)')
        der_z = ca.MX.sym('der(z)')

        ref_model.states = list(map(Variable, [x, y, z]))
        ref_model.der_states = list(map(Variable, [der_x, der_y, der_z]))
        ref_model.equations = [der_x + der_y - 1, der_x * y + x * der_y - 2, (der_x * y - x * der_y) / (y**2) - 3, 2 * x * der_x - 4, der_z - 5, der_x * z + x * der_z + der_y * z + y * der_z - 4, 0]

        self.assert_model_equivalent_numeric(ref_model, casadi_model)

    def test_alias_relation(self):
        a = AliasRelation()
        self.assertEqual(a.canonical_signed('-a'), ('a', -1))
        a.add('a', '-b')
        a.add('b', 'c')
        a.add('d', '-b')
        self.assertEqual(list(a), [('d', ['a', '-b', '-c'])])

    def test_cat_params(self):
        casadi_model = transfer_model(TEST_DIR, 'Concat', {'replace_constant_values': True})
        c = [0, 1, 2, 2, 2, 0, 1]
        self.assertEqual(len(c), len(casadi_model.parameters) - 1)

        for v, e in zip(c, casadi_model.parameters[1:]):
            self.assertEqual(v, e.value)

    def test_inline_input_assignment(self):
        casadi_model = transfer_model(TEST_DIR, 'InlineAssignment')
        self.assertTrue(casadi_model.inputs[0].fixed)
        self.assertFalse(casadi_model.alg_states[0].fixed)
        casadi_model = transfer_model(TEST_DIR, 'InlineAssignment', {'detect_aliases': True})
        self.assertTrue(casadi_model.inputs[0].fixed)

if __name__ == "__main__":
    unittest.main()
