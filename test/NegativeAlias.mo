model NegativeAlias
    Real x(min = 0, max = 3, nominal = 10);
    Real alias(min = -2, max = -1, nominal = 1);
equation
    der(x) = x;
    alias = -x;
end NegativeAlias;
