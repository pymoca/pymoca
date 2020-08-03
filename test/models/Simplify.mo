model Simplify
    parameter String string_parameter = "test_p";
    constant String string_constant = "test_c";
    constant Real c = 3.0;
    parameter Real p1 = 2.0;
    parameter Real p2 = 2 * p1;
    parameter Real p3;
    parameter Real p4 = 2 * p3;
    Real x(min = 0, max = 3, nominal = 10);
    Real alias(min = 1, max = 2, nominal = 1, start = p3);
    Real y;
    Real _tmp;
    Real cst;
equation
    der(x) = x + p1 + p2 + p3 + p4;
    alias = x;
    y = x + c + _tmp + cst;
    _tmp = 0.1 * x;
    cst = 4;
end Simplify;
