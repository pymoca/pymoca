model RigidBody "the rigid body"
    parameter Real g=9.81; //gravitational acc.
    parameter Real c=0.90; //elasticity constant.
    parameter Real m=1.0; //mass
    input Real f_x;
    output Real x, v_x, a_x;
equation
    f_x = 1.0;
    der(x) = v_x;
    der(v_x) = a_x;
    f_x = m*a_x;
end RigidBody;

model Accelerometer "an accelerometer"
    Bias b_x;
    input Real a_x "true acceleration";
    output Real ma_x "measured acceleration";
equation
    connect(b_x.u, a_x);
    ma_x = b_x.y;
end Accelerometer;

model Aircraft "the aircraft"
    RigidBody body;
    Accelerometer accel;
equation
    connect(body.a_x, accel.a_x);
end Aircraft;

model Bias "bias model"
    parameter Real b = 0.0;
    input Real u;
    output Real y;
equation
    y = u + b;
end Bias;
