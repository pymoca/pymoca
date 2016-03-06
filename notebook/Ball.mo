model Ball "This is a ball model"
    input Real f_x, f_y, f_z;
    parameter Real c = 1;
    output Real x, y, z, v_x, v_y, v_z, a_x, a_y, a_z;
equation
    der(x) = v_x;
    der(y) = v_y;
    der(z) = v_z;
    der(v_x) = a_x;
    der(v_y) = a_y;
    der(v_z) = a_z;
    a_x = -c*x + f_x*f_y;
    a_y = -c*y + f_y;
    a_z = -c*z + f_z;
    der(v_y) = a_y;
end Ball;