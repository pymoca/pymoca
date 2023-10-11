model DelayForLoop
  Real[3] x, y, a, i;
  input Real z[3];
  parameter Real delay_time(fixed=true);
  parameter Real eps;
equation
  i = x .* a;
  for i in 2:3 loop
    x[i] = 5 * z[i] * eps;
    y[i] = delay(3 * a[i] * eps, delay_time);
  end for;
  a = x;
end DelayForLoop;
