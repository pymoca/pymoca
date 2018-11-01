model Logic
    Boolean a, b, c, d, e;
equation
    b = not a;
    c = a and b;
    d = a or b;
    e = not (1 or 1 or 0);
end Logic;
