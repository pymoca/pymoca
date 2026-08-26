function square_plus
    input Real u;
    output Real y;
algorithm
    y := u * u + u;
end square_plus;

model SharedBranchExpression
    Real x(start = 1.0);
    Real y;
equation
    der(x) = x;
    y = if x > 0.0
        then square_plus(square_plus(square_plus(square_plus(square_plus(square_plus(x))))))
        else -square_plus(square_plus(square_plus(square_plus(square_plus(square_plus(x))))));
end SharedBranchExpression;
