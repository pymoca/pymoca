model Ball "This is a ball model"
    Real x;
    Real y;
equation
    der(x) = 1;
    der(y) = x;
end Ball;