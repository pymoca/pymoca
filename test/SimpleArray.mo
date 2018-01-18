model SimpleArray
    Real a[3] = {1.0, 2.0, 3.0};
    constant Real b[4] = {2.7, 3.7, 4.7, 5.7};
    Real c[3](each min = 0.0);
    parameter Integer d_dim = 3;
equation
    c = a .+ b[1:d_dim];
end ArrayExpressions;
