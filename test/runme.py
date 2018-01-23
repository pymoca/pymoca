import casadi as ca
import numpy as np

import os

import unittest
import pymola.backends.casadi.generator as gen_casadi
from pymola.backends.casadi.alias_relation import AliasRelation
from pymola.backends.casadi.model import Model, Variable
from pymola.backends.casadi.api import transfer_model, CachedModel
from pymola import parser, ast

TEST_DIR = os.path.dirname(os.path.realpath(__file__))

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

        if not self.assert_model_equivalent_numeric(ref_model, casadi_model):
            raise Exception("Failed test")


c = GenCasadiTest()
c.test_attributes()
