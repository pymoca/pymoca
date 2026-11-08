// Array comprehensions whose range is only known through a modification of an enclosing component.
model ForArrayModifiedRange
    model Channel
        parameter Integer n = 2;
        parameter Real H_b[n];
        parameter Real p = 1;
        Real H[n](
            min = cat(
                1,
                {max(H_b[1], H_b[2])},
                {max(H_b[i], max(H_b[i + 1], H_b[i + 2])) for i in 1:n - 2},
                {max(H_b[n - 1], H_b[n])}
            ),
            nominal = {p * i for i in 1:n}
        );
    equation
        H = H_b;
    end Channel;
    Channel c(n = 5, H_b = {0, 3, 1, 4, 2}, p = 2);
    Channel c2(H_b = {1, 0});
end ForArrayModifiedRange;
