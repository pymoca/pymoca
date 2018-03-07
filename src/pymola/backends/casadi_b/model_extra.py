import casadi as ca
import numpy as np


def sort(eqs, v):
    perm = ca.jacobian(eqs, v).sparsity().btf()
    print(perm)
    row_perm = perm[1]
    col_perm = perm[2]

    # sort equations
    new_eqs = []
    for i in row_perm:
        new_eqs.append(eqs[i])
    new_eqs = ca.vertcat(*new_eqs)

    # sort variables
    new_v = []
    for i in col_perm:
        new_v.append(v[i])
    new_v = ca.vertcat(*new_v)
    return new_eqs, new_v

def split_dae(self):
    x_symbols = set([x.symbol for x in self.states])
    dx_symbols = set([x.symbol for x in self.der_states])
    dae_eqs = []
    alg_eqs = []
    for eq in model.equations:
        if not set(ca.symvar(eq)).isdisjoint(dx_symbols):
            dae_eqs.append(eq)
        else:
            alg_eqs.append(eq)
    dae_eqs = ca.vertcat(*dae_eqs)
    #model_vars = ca.vertcat(*[x.symbol for x in (model.states + model.inputs + model.outputs + model.alg_states + model.der_states)])
    #model_eqs = ca.vertcat(*model.equations)
    #[eq_sort, v_sort] = sort(model_eqs, model_vars)
    return dae_eqs, alg_eqs

def blt_solve(x_symbols, dx_symbols, dae_eqs):
    x = ca.vertcat(*x_symbols)
    dx = ca.vertcat(*dx_symbols)
    perm = ca.jacobian(dae_eqs, dx).sparsity().btf()
    n_blocks = perm[0]
    row_perm = perm[1]
    col_perm = perm[2]
    row_block = perm[3]
    col_block = perm[4]

    # sort equations
    new_dae_eqs = []
    for i in row_perm:
        new_dae_eqs.append(dae_eqs[i])
    dae_eqs = ca.vertcat(*new_dae_eqs)

    # sort variables
    new_x = []
    new_dx = []
    for i in col_perm:
        new_x.append(x[i])
        new_dx.append(dx[i])
    x = ca.vertcat(*new_x)
    dx = ca.vertcat(*new_dx)

    J = ca.jacobian(dae_eqs, dx)

    new_ode = []

    # loop over blocks
    for i_b in range(n_blocks):
        # get variables in the block
        x_b = x[col_block[i_b]:col_block[i_b + 1]]
        dx_b = dx[col_block[i_b]:col_block[i_b + 1]]

        # get equations in the block
        eqs_b = dae_eqs[row_block[i_b]:row_block[i_b + 1]]

        # get the local Jacobian
        J_b = J[row_block[i_b]:row_block[i_b + 1], col_block[i_b]:col_block[i_b+1]]

        J_b.sparsity().spy()

        # if Jb depends on xb, then the state derivative does not enter linearly
        # in the ODE and we cannot solve for the state derivative
        if ca.depends_on(J_b, dx_b):
            raise RuntimeError("Cannot find an explicit epxression for variables {:s}".format(x_b))

        # divide fb into a part which depends on vb and a part which doesn't according to
        # "eqs_b == mul(Jb, vb) + eqs_b_res"
        eqs_b_res = ca.substitute(eqs_b, dx_b, ca.MX(dx_b.shape[0]))

        # solve for vb
        eqs_b_exp = ca.vertsplit(ca.solve(J_b, -eqs_b_res))
        new_ode.append(eqs_b_exp)

    #eliminate inter-dependencies
    #ca.substitute_inplace(dx, ca.vertcat(*new_ode), ca.MX(), False)