"""
Scipy based model simulation.
"""
import time
from typing import Dict

import casadi as ca

import numpy as np

import scipy.integrate

from .model import HybridOde


def sim(model: HybridOde, options: Dict = None,  # noqa: too-complex
        user_callback=None) -> Dict[str, np.array]:
    """
    Simulates a Dae model.

    :model: The model to simulate
    :options: See opt dict below
    :user_callback: A routine to call after each integration step,
      f(t, x, y, m, p, c) -> ret   (ret < 0 means abort)
    """
    if model.f_x_rhs.shape[0] < 1:
        raise ValueError("there are no ODE equations to simulate, "
                         "check that the model is explicit")

    ic = {}
    for f in ['x', 'y', 'm', 'p']:
        ic[f] = []
        for x in ca.vertsplit(getattr(model, f)):
            start = model.prop[x.name()]['start']
            value = model.prop[x.name()]['value']
            if start is not None:
                ic[f].append(ca.reshape(start, x.numel(), 1))
            elif value is not None:
                ic[f].append(ca.reshape(value, x.numel(), 1))
            else:
                ic[f].append(ca.DM.zeros(x.numel(), 1))
                Warning("using default start value for", x.name())

        ic[f] = np.array([ic[f]], dtype=float).T

    # set options
    opt = {
        'x0': ic['x'],
        'p': ic['p'],
        't0': 0,
        'tf': 1,
        'dt': None,
        'integrator': 'dopri5',
        'atol': 1e-6,
        'rtol': 1e-6,
        'max_step': None,
        'record_event_times': True,
        'verbose': False,
    }
    if options is not None:
        for k in options.keys():
            if k in opt.keys():
                opt[k] = options[k]
            else:
                raise ValueError("unknown option {:s}".format(k))
    if opt['dt'] is None:
        opt['dt'] = opt['tf']/100
    if opt['max_step'] is None:
        opt['max_step'] = opt['dt']/2

    # create functions
    f_y = model.create_function_f_y()
    f_c = model.create_function_f_c()
    f_m = model.create_function_f_m()
    f_x_rhs = model.create_function_f_x_rhs()
    f_J = model.create_function_f_J()
    f_i = model.create_function_f_i()

    # initialize sim loop
    t0 = opt['t0']
    tf = opt['tf']
    x = opt['x0']
    ng = np.zeros(model.ng.shape[0])
    nu = np.zeros(model.nu.shape[0])
    m = ic['m']
    p = opt['p']
    y0 = ic['y']
    pre_c = np.array(f_c(t0, x, y0, m, p, ng, nu))
    c = pre_c
    y = f_y(t0, x, m, p, c, ng, nu)
    dt = opt['dt']
    t_vect = np.arange(t0, tf, dt)
    n = len(t_vect)
    data = {
        't': [],
        'x': [],
        'm': [],
        'y': [],
        'c': [],
    }

    # create integrator
    integrator = scipy.integrate.ode(f_x_rhs, f_J)
    integrator.set_integrator(
        opt['integrator'],
        first_step=opt['max_step'],
        atol=opt['atol'],
        rtol=opt['rtol'],
        max_step=opt['max_step'],
    )

    # try to catch events with sol out, (root finding)
    def sol_out(t, x):
        c = np.array(f_c(t, x, y, m, p, ng, nu))
        if np.any(c != pre_c):
            # print('event', t)
            return -1
        return 0
    if opt['integrator'] in ['dopri5', 'dopri853']:
        integrator.set_solout(sol_out)

    # run sim loop
    i = 0
    dt_f_i = 0
    dt_integrate = 0
    integrator.set_initial_value(opt['x0'], t0)

    while i < n:
        t = integrator.t

        # resample noise
        ng = np.random.randn(model.ng.shape[0])
        nu = np.random.randn(model.nu.shape[0])

        # call reinit
        start = time.time()
        x = f_i(t, x, y, m, p, c, pre_c, ng, nu)
        dt_f_i += (time.time() - start)

        # setup next continuous integration step
        integrator.set_initial_value(x, t)
        integrator.set_f_params(y, m, p, c, ng, nu)
        integrator.set_jac_params(y, m, p, c, ng, nu)

        # integrate
        t_goal = t0 + i*dt
        start = time.time()
        integrator.integrate(t_goal)
        dt_integrate += (time.time() - start)
        x = integrator.y

        # compute new conditions
        pre_c = c
        c = np.array(f_c(t, x, y, m, p, ng, nu))

        # compute output
        y = f_y(t, x, m, p, c, ng, nu)

        # compute discrete states
        m = f_m(t, x, y, m, p, c, pre_c, ng, nu)

        # store data
        if opt['record_event_times'] or (integrator.t - t_goal == 0):
            data['t'].append(t)
            data['x'].append(ca.vertsplit(x))
            data['y'].append(ca.vertsplit(y))
            data['m'].append(ca.vertsplit(m))
            data['c'].append(ca.vertsplit(c))
            if user_callback is not None:
                user_callback(t, x, y, m, p, c)

        # increment time goal if reached
        if integrator.t - t_goal == 0:
            # update discrete states
            # TODO: make this use sampling
            i += 1

    for k in data.keys():
        data[k] = np.array(data[k])

    data['labels'] = {}
    for field in ['x', 'y', 'c', 'm']:
        data['labels'][field] = [x.name() for x in ca.vertsplit(getattr(model, field))]

    if opt['verbose']:
        print('dt_f_i\t\t\t:', dt_f_i)
        print('dt_integrate\t:', dt_integrate)
    return data
