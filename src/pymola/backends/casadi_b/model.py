"""
This module contains a differential algebraic equation model and
some helper functions.
"""
# pylint: disable=invalid-name, too-many-instance-attributes, import-error
from typing import Dict
import numpy as np
import scipy.integrate
import casadi as ca


class Dae:
    """
    A differential algebraic equation model.

    variables
    ------------------------
    x: state
    dx: der_state
    z: alg_state
    t: time
    p: parameter
    c: constant

    equations
    ------------------------
    dae_eq: 0 = f(dx, x, z, t)
    alg_eq: 0 = g(x, z, t)
    ode_rhs: dx = f(x, z, t)
    output_rhs: y = f(x, z, t)
    """

    def __init__(self):
        self.eq = ca.vertcat()
        self.state = ca.vertcat()
        self.der_state = ca.vertcat()
        self.input = ca.vertcat()
        self.output = ca.vertcat()
        self.alg_state = ca.vertcat()
        self.constant = ca.vertcat()
        self.parameter = ca.vertcat()
        self.property = {}
        self.time = None
        self.dae_eq = ca.vertcat()
        self.alg_eq = ca.vertcat()
        self.ode_rhs = ca.vertcat()
        self.output_rhs = ca.vertcat()
        self.sym_class = None

    @property
    def unknown(self):
        """unknown variables"""
        return ca.vertcat(self.state, self.alg_state, self.output)

    @property
    def equation(self):
        """equations to solve"""
        return ca.vertcat(self.dae_eq, self.alg_eq, self.ode_rhs)

    @property
    def balanced(self):
        """check if system is balanced"""
        return self.unknown.shape[0] == self.equation.shape[0]

    def __str__(self):
        line_width = 78
        s = '='*line_width + '\n'
        for name, group in [
                ('constants', ['input', 'constant', 'parameter']),
                ('variables', ['output', 'state', 'alg_state']),
                ('equations', ['dae_eq', 'alg_eq', 'ode_rhs', 'output_rhs'])]:
            header = '-'*30 + '{:9s} ({:4d})'.format(
                name, np.sum([getattr(self, f).shape[0] for f in group]))
            extra = line_width - len(header)
            if extra < 0:
                extra = 0
            header += '-'*extra + '\n'
            s += header
            for f in group:
                s += '{:10s} {:4d}: {:s}\n'.format(
                    f, getattr(self, f).shape[0], str(getattr(self, f)))
        s += '='*line_width + '\n'
        return s


    __repr__ = __str__


    def evaluate(self, type_name: str):
        """evaluate a set of constants"""
        v = getattr(self, type_name)
        if v.shape[0] > 0:
            vals = [self.property[c]['value'] for c in ca.vertsplit(v)]
            self.dae_eq = ca.substitute(self.dae_eq, v, vals)
            self.alg_eq = ca.substitute(self.alg_eq, v, vals)
            self.ode_rhs = ca.substitute(self.ode_rhs, v, vals)
            setattr(self, type_name, ca.vertcat())

    #pylint: disable=too-many-locals
    def make_explicit(self):
        """
        This method attempts to turn the model into an explicit ODE, assuming
        all the relationships are linear.

        Assume we start with the equations:

        0 = f(dx, x, z, t) = M dx + A x + B z + f'(t)
        0 = g(x, z, t) = M dx + A x + B z + f'(t)

        If M is invertible can be transformed to semi-explicit form via:

        dx = -M^{-1}(A x + B z + f't)
        0 = g(x, z, t)

        If we assume g is linear and dg/dz=D is invertible, then:

        g(x, z, t) = Cx + Dz + f''(t) = 0
        z = -D^{-1}(Cx + f''(t))

        Substituting into the above:

        dx = -M^{-1}(A - BD^{-1}C)x - M^{-1}(f'(t) - BD^{-1}f''(t))$
        dx = A'x + f'''(t)

        where:

        A' = -M^{-1}(A - BD^{-1}C)
        f'''(t) = - M^{-1}(f'(t) - BD^{-1}f''(t))
        """
        if self.dae_eq.shape[0] < 1:
            print('no dae_eq')
            return

        self.evaluate('constant')

        # create vector for ease of use
        dx = self.der_state
        x = self.state
        z = ca.vertcat(self.alg_state, self.output)
        p = self.parameter
        c = self.constant
        f = self.dae_eq
        g = self.alg_eq
        t = self.time

        # find forcing terms
        f_c = ca.substitute(f, ca.vertcat(x, dx, z), 0)
        g_c = ca.substitute(g, ca.vertcat(x, z), 0)

        # linearize
        M = ca.jacobian(f, dx)
        M_rank = ca.sprank(M)
        if ca.sprank(M) < x.shape[0]:
            print('cannot make explicit, mass matrix rank {:d} '
                  'less than n_states {:d}'.format(M_rank, x.shape[0]))
            return False
        A = ca.jacobian(f, x)
        B = ca.jacobian(f, z)
        C = ca.jacobian(g, x)
        D = ca.jacobian(g, z)
        if ca.sprank(D) < z.shape[0]:
            print('cannot make explicit, dg/dz rank {:d} '
                  'less than nz {:d}'.format(M_rank, z.shape[0]))
            return False

        # assert everything is linearizable
        if self.sym_class == ca.SX:
            assert ca.norm_2(ca.simplify(
                ca.mtimes(M, dx) + ca.mtimes(A, x) + ca.mtimes(B, z) + f_c - f)) == 0
            assert ca.norm_2(ca.simplify(
                ca.mtimes(C, x) + ca.mtimes(D, z) + g_c - g)) == 0

        # find new model
        f_Ap = ca.Function(
            "Ap",
            [p, c],
            [-ca.mtimes([ca.inv(M), A - ca.mtimes([B, ca.inv(D), C])])])
        f_t = ca.Function(
            "f_t",
            [p, c, t],
            [-ca.mtimes([ca.inv(M), f_c - ca.mtimes([B, ca.inv(D), g_c])])])
        f_z = ca.Function('z', [x, p, c, t], [-ca.mtimes(ca.inv(D), (ca.mtimes(C, x) + g_c))])
        ode = ca.mtimes(f_Ap(p, c), x) + f_t(p, c, t)

        # adjust model
        self.ode_rhs = ode
        self.dae_eq = ca.vertcat()
        self.alg_eq = ca.vertcat()
        self.output = self.alg_state
        self.alg_state = ca.vertcat()
        self.output_rhs = f_z(x, p, c, t)
        return True

    # pylint: disable=too-many-branches, too-many-statements
    def sim_ode(self, options: Dict=None):
        """
        Simulates a Dae model.
        """
        if self.ode_rhs.shape[0] < 1:
            raise ValueError("there are no ODE equations to simulate, "
                             "check that the model is explicit")

        # set options
        opt = {
            'x0': [self.property[x]['start'] for x in ca.vertsplit(self.state)],
            'p': [self.property[x]['value'] for x in ca.vertsplit(self.parameter)],
            't0': 0,
            'tf': 1,
            'dt': 0.1,
            'integrator': 'vode',
            'use_scipy': True
        }
        if options is not None:
            for k in options.keys():
                if k in opt.keys():
                    opt[k] = options[k]
                else:
                    raise ValueError("unknown option {:s}".format(k))

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
        func_opt = {'jit': with_jit, 'compiler': compiler}

        # create output function
        output_func = ca.Function(
            'y',
            [self.state, self.parameter, self.time],
            [self.output_rhs], func_opt)

        # initialize sim loop
        t0 = opt['t0']
        tf = opt['tf']
        x = opt['x0']
        p = opt['p']
        y = ca.vertsplit(output_func(x, p, t0))
        dt = opt['dt']
        n = int(tf/dt)
        data = {
            't': np.arange(t0, tf, dt),
            'x': np.zeros((n, len(x))),
            'y': np.zeros((n, len(y)))
        }

        # create integrator
        if opt['use_scipy']:
            f_ode = ca.Function(
                'f',
                [self.time, self.state, self.parameter],
                [self.ode_rhs], func_opt)
            f_J = ca.Function(
                'J',
                [self.time, self.state, self.parameter],
                [ca.jacobian(self.ode_rhs, self.state)], func_opt)
            integrator = scipy.integrate.ode(f_ode, f_J)
            integrator.set_initial_value(x, t0)
            integrator.set_f_params(p)
            integrator.set_jac_params(p)
            integrator.set_integrator(opt['integrator'])
        else:
            if len(self.ode_rhs) > 0 and len(self.dae_eq) == 0 and len(self.alg_eq) == 0:
                problem = {
                    'ode': self.ode_rhs,
                    'x': self.state,
                    'p': self.parameter,
                    't': self.time}
            else:
                raise RuntimeError("model not handled")
            integrator = ca.integrator('f', 'cvodes', problem)

        # run sim loop
        for i in range(1, n):
            t = t0 + dt*i
            if opt['use_scipy']:
                integrator.integrate(t)
                x = integrator.y
            else:
                # need to add t0, tf to casadi interface, this is  slow
                # see https://github.com/casadi/casadi/issues/1592
                integrator = ca.integrator('f', 'cvodes', problem, {'t0': t, 'tf': t + dt})
                res = integrator(x0=x, p=p)
                x = res['xf']

            # compute output (this takes awhile, need to see how to speed it up)
            # it could be skipped all together or only computer when the user
            # asks for the variables for plotting after the simulation, but this
            # prevents the user from passing a control based on the output
            y = output_func(x, p, t)

            # store data
            data['x'][i, :] = np.array(ca.vertsplit(x))
            data['y'][i, :] = np.array(ca.vertsplit(y))

        return data
