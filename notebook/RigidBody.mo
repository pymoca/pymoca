model RigidBody "body"
    input Real F_x "force x";
    input Real F_y "force y";
    input Real F_z "force z";
    input Real M_x "moment x";
    input Real M_y "moment y";
    input Real M_z "moment z";
    parameter Real m = 1 "mass";
    parameter Real Jx = 1 "Jx";
    parameter Real Jy = 1 "Jy";
    parameter Real Jz = 1 "Jz";
    output Real x, y, z;
    output Real v_x, v_y, v_z;
    output Real a_x, a_y, a_z;
    output Real q0, q1, q2, q3;
    output Real w_x, w_y, w_z;
    output Real alpha_x, alpha_y, alpha_z;
equation
    // translation
    der(x) = v_x;
    der(y) = v_y;
    der(z) = v_z;
    der(v_x) = a_x;
    der(v_y) = a_y;
    der(v_z) = a_z;
    m*a_x = F_x;
    m*a_y = F_y;
    m*a_z = F_z;
    // rotation
    der(q0) = w_x;
    der(q1) = w_y;
    der(q2) = w_z;
    der(q3) = w_z;
    der(w_x) = alpha_x;
    der(w_y) = alpha_y;
    der(w_z) = alpha_z;
    J_x*alpha_x = M_x;
    J_y*alpha_y = M_y;
    J_z*alpha_z = M_z;
end RigidBody;