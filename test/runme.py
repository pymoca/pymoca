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

def MXArray(name, *dimensions):
    if not dimensions:
        return np.array([ca.MX.sym(name)])

    arr = np.empty(dimensions, dtype=object)

    for ind, _ in np.ndenumerate(arr):
        arr[ind] = ca.MX.sym("{}[{}]".format(name, ", ".join((str(x + 1) for x in ind))))

    return arr

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

c = GenCasadiTest()
c.test_state_annotator()
