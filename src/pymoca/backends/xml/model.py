"""
This module contiains various mathematical models for a modelica model.
"""
from typing import Any, Dict, List, Union

import casadi as ca

SYM = Union[ca.SX, ca.MX]


class CasadiModel:

    def __init__(self, sym, *args, **kwargs):
        self.sym = sym

        # handle user args
        for k, v in kwargs.items():
            if k in self.__dict__.keys():
                setattr(self, k, v)
            else:
                raise ValueError('unknown argument', k)

        # Use just-in-time compilation to speed up the evaluation
        if ca.Importer.has_plugin('clang'):
            with_jit = True
            compiler = 'clang'
        elif ca.Importer.has_plugin('shell'):
            with_jit = True
            compiler = 'shell'
        else:
            print("WARNING; running without jit. "
                  "This may result in very slow evaluation times")
            with_jit = False
            compiler = ''
        self.func_opt = {'jit': with_jit, 'compiler': compiler}

    def __repr__(self):
        s = "\n"
        for k, v in sorted(self.__dict__.items()):
            if isinstance(v, self.sym):
                s += "{:8s}({:3d}):\t{:s}\n".format(k, v.shape[0], str(v))
        return s


# noinspection PyPep8Naming
class HybridOde(CasadiModel):
    """Hybrid Oridinary Differential Equation Model"""

    def __init__(self, sym: type = ca.SX, **kwargs):
        self.c = sym(0, 1)  # conditions
        self.dx = sym(0, 1)  # states derivatives
        self.f_c = sym(0, 1)  # condition relations
        self.f_i = sym(0, 1)  # reinit equations
        self.f_m = sym(0, 1)  # discrete update
        self.f_x_rhs = sym(0, 1)  # continuous integration
        self.y_rhs = sym(0, 1)  # algebraic states as a function of state
        self.m = sym(0, 1)  # discrete states
        self.ng = sym(0, 1)  # gaussian noise
        self.nu = sym(0, 1)  # uniform noise
        self.p = sym(0, 1)  # parameters and constants
        self.pre_m = sym(0, 1)  # discrete pre states
        self.pre_c = sym(0, 1)  # pre conditions
        self.prop = {}  # properties
        self.sym = sym  # symbol type
        self.t = sym()  # time
        self.x = sym(0, 1)  # states (have derivatives)
        self.y = sym(0, 1)  # algebraic states
        super().__init__(sym, **kwargs)

    def create_function_f_c(self):
        """condition function"""
        return ca.Function(
            'f_c',
            [self.t, self.x, self.y, self.m, self.p, self.ng, self.nu],
            [self.f_c],
            ['t', 'x', 'y', 'm', 'p', 'ng', 'nu'], ['c'], self.func_opt)

    def create_function_f_i(self):
        """state reinitialization (reset) function"""
        return ca.Function(
            'f_i',
            [self.t, self.x, self.y, self.m, self.p, self.c, self.pre_c, self.ng, self.nu],
            [self.f_i],
            ['t', 'x', 'y', 'm', 'p', 'c', 'pre_c', 'ng', 'nu'], ['x_n'], self.func_opt)

    def create_function_f_x_rhs(self):
        """ODE rhs for continuous state integration"""
        return ca.Function(
            'f_x_rhs',
            [self.t, self.x, self.y, self.m, self.p, self.c, self.ng, self.nu],
            [self.f_x_rhs],
            ['t', 'x', 'y', 'm', 'p', 'c', 'ng', 'nu'], ['x_rhs'], self.func_opt)

    def create_function_f_m(self):
        """Discrete state dynamics"""
        return ca.Function(
            'f_m',
            [self.t, self.x, self.y, self.m, self.p, self.c, self.pre_c, self.ng, self.nu],
            [self.f_m],
            ['t', 'x', 'y', 'm', 'p', 'c', 'pre_c', 'ng', 'nu'], ['m'], self.func_opt)

    def create_function_f_J(self):
        """Jacobian for state integration"""
        return ca.Function(
            'J',
            [self.t, self.x, self.y, self.m, self.p, self.c, self.ng, self.nu],
            [ca.jacobian(self.f_x_rhs, self.x)],
            ['t', 'x', 'y', 'm', 'p', 'c', 'ng', 'nu'], ['J'], self.func_opt)

    def create_function_f_y(self):
        """output function"""
        return ca.Function(
            'y',
            [self.t, self.x, self.m, self.p, self.c, self.ng, self.nu],
            [self.y_rhs],
            ['t', 'x', 'm', 'p', 'c', 'ng', 'nu'], ['y'], self.func_opt)


class HybridDae(CasadiModel):
    """Hybrid Differential Algebraic Equation Model"""

    def __init__(self, sym: type = ca.SX, **kwargs):
        self.sym = sym
        self.c = sym(0, 1)  # conditions
        self.dx = sym(0, 1)  # states derivatives
        self.f_c = sym(0, 1)  # condition relations
        self.f_i = sym(0, 1)  # reinit equations
        self.f_m = sym(0, 1)  # discrete update
        self.f_x = sym(0, 1)  # continuous integration
        self.m = sym(0, 1)  # discrete states
        self.ng = sym(0, 1)  # gaussian noise
        self.nu = sym(0, 1)  # uniform noise
        self.p = sym(0, 1)   # parameters and constants
        self.pre_m = sym(0, 1)  # discrete pre states
        self.pre_c = sym(0, 1)  # pre conditions
        self.prop = {}  # properties
        self.t = sym()  # time
        self.x = sym(0, 1)  # states (have derivatives)
        self.y = sym(0, 1)  # algebraic states

        super().__init__(sym, **kwargs)

    def to_ode(self) -> HybridOde:
        """Convert to a HybridOde"""
        res_split = split_dae_alg(self.f_x, self.dx)
        alg = res_split['alg']
        dae = res_split['dae']

        x_rhs = tangent_approx(dae, self.dx, assert_linear=True)
        y_rhs = tangent_approx(alg, self.y, assert_linear=True)

        return HybridOde(
            c=self.c,
            dx=self.dx,
            f_c=self.f_c,
            f_i=self.f_i,
            f_m=self.f_m,
            f_x_rhs=x_rhs,
            y_rhs=y_rhs,
            m=self.m,
            ng=self.ng,
            nu=self.nu,
            p=self.p,
            pre_m=self.pre_m,
            pre_c=self.pre_c,
            prop=self.prop,
            sym=self.sym,
            t=self.t,
            x=self.x,
            y=self.y,
        )


def split_dae_alg(eqs: SYM, dx: SYM) -> Dict[str, SYM]:
    """Split equations into differential algebraic and algebraic only"""
    dae = []
    alg = []
    for eq in ca.vertsplit(eqs):
        if ca.depends_on(eq, dx):
            dae.append(eq)
        else:
            alg.append(eq)
    return {
        'dae': ca.vertcat(*dae),
        'alg': ca.vertcat(*alg)
    }


def permute(x: SYM, perm: List[int]) -> SYM:
    """Perumute a vector"""
    x_s = []
    for i in perm:
        x_s.append(x[i])
    return ca.vertcat(*x_s)


# noinspection PyPep8Naming,SpellCheckingInspection
def blt(f: List[SYM], x: List[SYM]) -> Dict[str, Any]:
    """
    Sort equations by dependence
    """
    J = ca.jacobian(f, x)
    nblock, rowperm, colperm, rowblock, colblock, coarserow, coarsecol = J.sparsity().btf()
    return {
        'J': J,
        'nblock': nblock,
        'rowperm': rowperm,
        'colperm': colperm,
        'rowblock': rowblock,
        'colblock': colblock,
        'coarserow': coarserow,
        'coarsecol': coarsecol
    }


# noinspection PyPep8Naming
def tangent_approx(f: SYM, x: SYM, a: SYM = None, assert_linear: bool = False) -> Dict[str, SYM]:
    """
    Create a tangent approximation of a non-linear function f(x) about point a
    using a block lower triangular solver

    0 = f(x) = f(a) + J*x   # taylor series about a (if f(x) linear in x, then globally valid)
    J*x = -f(a)             # solve for x
    x = -J^{-1}f(a)         # but inverse is slow, so we use solve
    where J = df/dx
    """
    # find f(a)
    if a is None:
        a = ca.DM.zeros(x.numel(), 1)
    f_a = ca.substitute(f, x, a)  # f(a)
    J = ca.jacobian(f, x)
    if assert_linear and ca.depends_on(J, x):
        raise AssertionError('not linear')
    # solve is smart enough to to convert to blt if necessary
    return ca.solve(J, -f_a)
