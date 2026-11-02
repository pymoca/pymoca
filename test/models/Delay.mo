model Delay
    Real x, y, z, w;
    parameter Real hour = 3600;
equation
    y = delay(x, 6 * hour);
    z = delay(x, 3600.0);
    w = delay(x, 6 * hour, 24 * hour);
end Delay;
