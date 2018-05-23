model DelayForLoop
  Real[2] x, y;
  input Real z[2];
  input Real delay_time;
equation
  for i in 1:2 loop
    x[i] = 5 * z[i];
    y[i] = delay(3 * x[i], delay_time);
  end for;
end DelayForLoop;
