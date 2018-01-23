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

        if not self.assert_model_equivalent_numeric(ref_model, casadi_model):
            raise Exception("Failed test")

c = GenCasadiTest()
c.test_matrixexpressions()
