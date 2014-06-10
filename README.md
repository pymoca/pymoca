#pymola

A python/modelica based simulation environment.

[![Build Status](https://travis-ci.org/jgoppert/pymola.svg)](https://travis-ci.org/jgoppert/pymola)
[![Coverage Status](https://img.shields.io/coveralls/jgoppert/pymola.svg)](https://coveralls.io/r/jgoppert/pymola)

##Roadmap

### Current Tasks

* Parser: I haven't decided on final parser, currently 3 in the running

    1. ply  
        * parser type: LALR
        * good: fast, proven, lexer so it can parse tokens (avoid keywords as identifiers etc.)
        * bad: BNF grammar (haved to translate EBNF standard)
    1. grako
        * parser type: PEG
        * good: EBNF support, developer responsive
        * bad: no lexer
    1. parsimonious
        * parser type: PEG
        * good: EBNF support
        * bad: developer unresponsive, no lexer, doesn't integrate well with coverage testing, slight modificatin of EBNF when typing

    The current parsers all have built in unit tests that can be run with

        python setup.py test

    or
   
        nosetests

### TODO

* full hello world working prototype example with backend

### Completed Tasks

* Project setup.
* Unit testing for parsers.
* Parsing basic hello world example.
* Travis continuous integration testing setup.
* Coveralls coverage testing setup.

## Goals

### Backend Representation

We need to create a backend representation of the models that can generate the equations for simulation/ inverse simulation/ jacobians etc.

### Modelica Magic

If you have ever used fortran/cython magic, I would imagine that modelica magic would be similar. You would type modelica in one cell:

    %modelica
    model ball
      Real a;
    equation
      der(a) = a;
    end model ball

And then you can access the compiled modelica object via python.

    results = ball.simulate(tf=10)
    plot(results.a)

### Soft Real-Time Simulation

    sim = ode(ball.dynamics)
    while sim.successfull
        sim.integrate(sim.t + dt)
        do real-time stuff
        // wait on wall clock


### Analytical Jacobians

In order to linear control/ estimation etc, it is useful to have analytical jacobians. The backend for this could by sympy.

    import sympy
    ball_trimmed = ball.trim(a=1).
    states = sympy.Matrix([ball.a])
    inputs = sympy.Matrix([ball.u])
    f = ball_trimmed.dynamics([states]) # sympy matrix
    A = f.jacobian(states)
    B = f.jacobian(input)

Next, we can use the python control toolbox.

    import control
    ss = control.ss(A.subs(const),B.subs(const),
        C=np.eye(2),D=np.eye(2))
    control.bode(ss)

### Inverse Dynamics

The inverse dynamics should be able to be simulated with numpy and printed with the same interface as the standard dynamics.

    sim = ode(ball.inverse_dynamics)
    while sim.successfull
        sim.integrate(sim.t + dt)

<!--- vim:ts=4:sw=4:expandtab:
!-->
