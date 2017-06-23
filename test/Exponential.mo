model Exponential
    Real x(start = 0.0);
    input Real u;
equation
    der(x) = -x + u;
end Exponential;