model Test
    parameter Real c=10;
    Real x(start=1);
    Real v_x;
    Real y(start=2);
    Real v_y;
equation
    der(x) = v_x;
    der(v_x) = -c*x;
    der(y) = v_y;
    der(v_y) = -c*y;
end Test;