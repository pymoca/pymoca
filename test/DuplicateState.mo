model DuplicateState
    Real x, y;
equation
    der(x) + der(y) = 1;
    der(x) = 2;
end DuplicateState;