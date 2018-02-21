model Alias
    Real x, y;
    output Real z;
    input Real w;
    Real u;
equation
    der(x) = 1;
    x = y;
    z = -y;
    u = w;
end Alias;