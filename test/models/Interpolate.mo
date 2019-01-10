model Interpolate
    parameter Real xp[3] = {0.0, 1.0, 2.0};
    parameter Real yp[3] = {0.0, 1.0, 4.0};
    parameter Real zp[3, 3] = {{0.0, 0.0, 0.0}, {0.0, 1.0, 4.0}, {0.0, 2.0, 8.0}};
    Real x, y, z;
equation
    y = _pymoca_interp1d(xp, yp, x);
    z = _pymoca_interp2d(xp, yp, zp, x, y, "linear");
end Interpolate;