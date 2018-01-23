model Concat
    constant Real x = 2.0;
    parameter Integer n = 4;
    parameter Real c[n + 3] = cat(1, 0.0, x / 2, fill(x, n - 4) , fill(x, n - 3), fill(x, n - 2), {0.0, x / 2});
end Concat;
