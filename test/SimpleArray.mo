model SimpleArray
    Real a[3] = {1.0, 2.0, 3.0};
    constant Real b[4] = {2.7, 3.7, 4.7, 5.7};
    Real c[3](each min = 0.0);
    Real[3] d;
    Real e[3];
    Real scalar_f = 1.3;
    Real g;
    output Real h;
    Real i[2, 3];
    constant Integer c_dim = 2;
    parameter Integer d_dim = 3;
    constant Real B[d_dim] = linspace(1, 2, 3);
equation
    // Array operators.
    c = a .+ b[1:d_dim].*e; // .+ is equal to + in this case

    // Calling a (scalar) function on an array maps the function to each element.
    d = sin(a ./ b[c_dim:4]);

    // Difference between .+ and +
    e = d .+ scalar_f; // Different shapes, so + is not allowed, only .+ is.

    // Sum.
    g = sum(c);

    // Indexing
    h = B[d_dim - 1];

    // Implicit transpose
    i[1, :] = ones(3);
    i[2, :] = transpose(ones(3));
end SimpleArray;
