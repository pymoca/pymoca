model ArrayExpand
    parameter Real x[:, :] = {{ -5.326999,  54.050758,  0.000000},
                              { -1.0,        0.0,       0.0}};

    parameter Integer y[2, 2] = fill(-999, 2, 2);

    parameter Real z[2, 2] = fill(-999.0, 2, 2);
end ArrayExpand;
