model SimplifyVector
    Real x[2];
    output Real y[2];
equation
    der(x) = x;
    y = x;
end SimplifyVector;
