// Also see the Modelica specification, section 10.6
model ArrayExpressions
    Real a[3] = {1.0, 2.0, 3.0};
    constant Real b[4] = {2.7, 3.7, 4.7, 5.7}; // Can also be done with 2.7:5.7
    Real c[3];
    Real d[3];
    Real e[3];
    Real scalar_f = 1.3;
    Real g;
    constant Integer c_dim = 2;
    parameter Integer d_dim = 3;
    constant Real B[d_dim] = linspace(1, 2, 3);
    constant Real C[c_dim] = fill(1.7, c_dim);
    constant Real D[3] = zeros(d_dim);
    constant Real E[2] = ones(2);
equation
    // Array operators.
    c = a .+ b[1:d_dim].*e; // .+ is equal to + in this case

    // Calling a (scalar) function on an array maps the function to each element.
    d = sin(a ./ b[c_dim:4]);

    // Difference between .+ and +
    e = d .+ scalar_f; // Different shapes, so + is not allowed, only .+ is.

    // Sum.
    g = sum(c);
end ArrayExpressions;
