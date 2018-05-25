// Also see the Modelica specification, section 10.6
model ArrayElement
    Real x;
end ArrayElement;

connector ArrayConnector
    Real y;
    flow Real w;
end ArrayConnector;

model NestedArrayExpressions
    parameter Integer n = 3;
    Real z[n];
end NestedArrayExpressions;

model ArrayExpressions
    Real a[3] = {1.0, 2.0, 3.0};
    constant Real b[4] = {2.7, 3.7, 4.7, 5.7}; // Can also be done with 2.7:5.7
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
    constant Real C[c_dim] = fill(1.7, c_dim);
    constant Real D[c_dim + 1] = zeros(d_dim);
    constant Real E[2] = ones(d_dim - 1);
    ArrayElement ar[c_dim + 1];
    ArrayConnector arc[c_dim];
    NestedArrayExpressions nested1;
    NestedArrayExpressions nested2[2];
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
    ar[2].x = scalar_f;

    // Nesting
    nested1.z = ones(3);
    // FIXME: We should be able to index with `nested2[1].z`, but we currently need the `[:]`.
    nested2[1].z[:] = {4, 5, 6};
    nested2[2].z[1] = 3;
    nested2[2].z[2] = 2;
    nested2[2].z[3] = 1;

    // Implicit transpose
    i[1, :] = ones(3);
    i[2, :] = transpose(ones(3));

    // Connecting
    connect(arc[1], arc[2]);
end ArrayExpressions;
