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

        if not self.assert_model_equivalent_numeric(ref_model, casadi_model):
            raise Exception("Failed test")

c = GenCasadiTest()
c.test_matrixexpressions()
