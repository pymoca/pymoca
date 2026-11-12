// Array comprehension in a modification (issue #353).
model ForArrayModification
    parameter Integer n = 5;
    parameter Real H_b[n] = {0, 1, 2, 3, 4};
    Real H[n](
        min = cat(
            1,
            {max(H_b[1], H_b[2])},
            {max(H_b[i], max(H_b[i + 1], H_b[i + 2])) for i in 1:n - 2},
            {max(H_b[n - 1], H_b[n])}
        )
    );
equation
    H = H_b;
end ForArrayModification;
