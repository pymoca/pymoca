// Also see the Modelica specification, section 10.6
model MatrixExpressions
    Real A[3,3];
    Real b[3];
    Real c[3];
    Real d[3];
    constant Real C[2, 3] = fill(1.7, 2, 3);
    constant Real D[3, 2] = zeros(3, 2);
    constant Real E[2, 3] = ones(2, 3);
    constant Real I[5, 5] = identity(5);
    constant Real F[3, 3] = diagonal({1, 2, 3});
    constant Real G[3, 3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
equation
    A*b = c;
    transpose(A)*b = d;
    F[2, 3] = 0;
end MatrixExpressions;
