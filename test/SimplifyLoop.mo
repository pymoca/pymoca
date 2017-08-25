model SimplifyLoop
    Real x;
    Real y[2];
equation
    for i in 1:2 loop
        y[i] = i * x;
    end for;
end SimplifyLoop;
