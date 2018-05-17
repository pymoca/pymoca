model Delay
    Real x, y;
    parameter Real hour = 3600;
equation
    y = delay(x, 6 * hour);
end Delay;