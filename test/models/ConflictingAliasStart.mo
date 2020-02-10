model ConflictingAliasStart
    parameter Real p1;
    Real x(min = 0, max = 3, nominal = 10, start = p1);
    Real alias_neg(min = -2, max = -1, nominal = 1, start = p1);
    Real alias_pos(start = 4);
equation
    der(x) = x;
    alias_neg = -x;
    alias_pos = x;
end ConflictingAliasStart;
