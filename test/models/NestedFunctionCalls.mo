function square_plus
    input Real u;
    output Real y;
algorithm
    y := u * u + u;
end square_plus;

model NestedFunctionCalls
    Real x;
equation
    der(x) = square_plus(square_plus(square_plus(square_plus(square_plus(square_plus(x))))));
end NestedFunctionCalls;
