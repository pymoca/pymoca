model SpringSystem
    input Real u;
    output Real x(start=1), v_x(start=1);
    Spring spring;
    Damper damper;
equation
    spring.x = x;
    damper.v = v_x;
    der(x) = v_x;
    der(v_x) = spring.f + damper.f - u;
end SpringSystem;

model Spring
    Real x "displacement";
    Real f "force";
    parameter Real k = 2.0 "spring constant";
equation
    f = -k*x;
end Spring;

model Damper
    Real v "velocity";
    Real f "force";
    parameter Real c = 0.2 "damping constant";
equation
    f = -c*v;
end Damper;