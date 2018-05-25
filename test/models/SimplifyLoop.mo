model SimplifyLoop
    Real x;
    Real y[3];
    parameter Real p3 = 0.125;
    parameter Real p2 = 2 * p3;
    parameter Real p1 = 2 * p2;
    parameter Real p = 2 * p1;
    parameter Integer n = 1;
equation
    for i in 1:3*n loop
        y[i] = p * i * x;
    end for;
end SimplifyLoop;
