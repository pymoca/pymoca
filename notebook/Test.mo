model Test
    parameter Real c=10;
    Real x(start=0), v_x(start=0);
    Real y(start=0), v_y(start=0);
equation
    der(x) = v_x;
    der(y) = v_y;
    der(v_x) = -c*x^2;
    der(v_y) = -c*y^2;
end Test;