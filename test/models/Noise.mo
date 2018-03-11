model Noise
    Real x;
    Real dt;
    discrete Real t_last(start=0);
    discrete Real y;
equation
    der(x) = 1;
    dt = time - t_last;
    when (dt > 0.1) then
        t_last = time;
        y = x + noise_gaussian(0, 0.1) + noise_uniform(-0.01, 0.01);
    end when;
end Noise;