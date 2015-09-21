model BouncingBall "the bouncing ball model"
    parameter Real g=9.81; //gravitational acc.
    parameter Real c=0.90; //elasticity constant.
    Real height(start=10), velocity(start=0);
equation
    der(height) = velocity;
    der(velocity) = -g;
    when height<0 then
        reinit(velocity, -c*velocity);
    end when;
end BouncingBall;
