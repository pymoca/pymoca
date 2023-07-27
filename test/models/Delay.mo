model Delay
    Real x, y, z;
    parameter Real hour = 3600;
equation
    y = delay(x, 6 * hour);
    z = delay(x, 3600.0);
end Delay;
