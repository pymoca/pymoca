model Spring
    Real x, v_x;
    parameter Real c = 0.1;
    parameter Real k = 2;
equation
    der(x) = v_x;
    der(v_x) = -k*x - c*v_x;
end Spring;
// vim: set noet fenc= ft=modelica ff=unix sts=0 sw=4 ts=4 :
