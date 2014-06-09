pymola
======

A pure python modelica based object-oriented simulation environment.

##Roadmap

### DONE

* basic parser that can handle helloworld example

### TODO

#### backend representation

We need to create a backend representation of the models that can generate the equations for simulation/ inverse simulation/ jacobians etc.

#### modelica magic

If you have ever used fortran magic I would imagine that modelica magic would work the same way. You would do something like

    %modelica
    model ball
      Real a;
    equation
      der(a) = a;
    end model ball

in one cell, then you get the python object out to play with

    results = ball.simulate(tf=10)
    plot(results.a)

#### real-time simulation

    sim = ode(ball.dynamics)
    while sim.successfull
        sim.integrate(sim.t + dt)
        do real-time stuff
        // wait on wall clock


#### analytical jacobians

We want to create analytical jacobians that play well with sympy.

    ball_trimmed = ball.trim(a=1).
    states = [ball.a]
    inputs = [ball.u]
    f = ball_trimmed.dynamics([states]) # sympy matrix
    A = f.jacobian(states)
    B = f.jacobian(input)

Now we can use python control for linear analysis.

    ss = control.ss(A.subs(const),B.subs(const),
        C=np.eye(2),D=np.eye(2))
    control.bode(ss)

#### inverse dynamics

The inverse dynamics should be able to be simulated with numpy and printed with the same interface as the standard dynamics.

    sim = ode(ball.inverse_dynamics)
    while sim.successfull
        sim.integrate(sim.t + dt)

vim:ts=4:sw=4:expandtab
