model Ball "This is a ball model"
    input Real f_x, f_y;
    parameter Real c = 1;
    output Real x;
    output Real y;
    output Real v_x;
    output Real v_y;
equation
    der(x) = v_x;
    der(y) = v_y;
    der(v_x) = -c*x + f_x;
    der(v_y) = -c*y + f_y;
end Ball;