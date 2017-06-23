model Simplify
    constant Real c = 3.0;
    parameter Real p1 = 2.0;
    parameter Real p2 = 2 * p1;
    parameter Real p3;
    parameter Real p4 = 2 * p3;
    Real x;
    Real alias;
    Real y;
    Real _tmp;
equation
    der(x) = x + p1 + p2 + p3 + p4;
    alias = x;
    y = x + c + _tmp;
    _tmp = 0.1 * x;
end Simplify;
