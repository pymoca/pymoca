model SimplifiedVariables
    parameter Real a = 1;
    parameter Real b = a + 4;
    constant Real c = 2;
    constant Real d = c - 5;
    Real y;
    Real z;
    Real x;
equation
    y = - z + a;
    x = (3*y);
    z = 10;
end SimplifiedVariables;