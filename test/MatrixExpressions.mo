// Also see the Modelica specification, section 10.6
model MatrixExpressions
    Real A[3,3];
    Real b[3];
    Real c[3];
    Real d[3];
    constant Real B[3] = linspace(1, 2, 3);
equation
    A*b = c;
    transpose(A)*b = d;
end MatrixExpressions;
