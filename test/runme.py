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

    def test_cat_params(self):
        casadi_model = transfer_model(TEST_DIR, 'Concat', {'replace_constant_values': True})

        c = [0, 1, 2, 2, 2, 0, 1]

        self.assertEqual(len(c), len(casadi_model.parameters) - 1)

        for v, e in zip(c, casadi_model.parameters[1:]):
            self.assertEqual(v, e.value)

c = GenCasadiTest()
c.test_cat_params()
