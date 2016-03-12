model RigidBody "the rigid body"
    parameter Real g=9.81; //gravitational acc.
    parameter Real c=0.90; //elasticity constant.
    input Real f_x, f_y, f_z;
    output Real x, y, z;
    output Real v_x, v_y, v_z;
    output Real a_x, a_y, a_z;
equation
    der(x) = v_x;
    der(y) = v_y;
    der(z) = v_z;
    der(v_x) = a_x;
    der(v_y) = a_y;
    der(v_z) = a_z;
    f_x = m*a_x;
    f_y = m*a_y;
    f_z = m*a_z;
end RigidBody;

model Accelerometer "an accelerometer"
    parameter Real b_x=1.0;
    parameter Real b_y=1.0;
    parameter Real b_z=1.0;
    input Real a_x, a_y, a_z;
    output Real ma_x, ma_y, ma_z;
equation
    ma_x = a_x + b_x;
    ma_y = a_y + b_y;
    ma_z = a_z + b_z;
end Accelerometer;

model Aircraft "the aircraft"
    RigidBody body;
    Accelerometer accel;
equation
    connect(body.a_x, accel.a_x);
    connect(body.a_y, accel.a_y);
    connect(body.a_z, accel.a_z);
end Aircraft;
