model Test
    parameter Real c=10;
    Real x(start=1), v(start=1);
equation
    der(x) = v;
    der(v) = -c*x;
end Test;