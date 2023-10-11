model DelayForLoop
  constant Integer other = 3;
  constant Integer i = 3;
  constant Integer j = 3;
  Real[3] x, y, a;
  input Real z[3];
  parameter Real delay_time(fixed=true);
  parameter Real eps;
equation
  for i in 2:j loop
    x[i] = 5 * z[i] * eps;
    y[i] = delay(3 * a[i] * eps, delay_time);
    for j in i:3 loop
      a[i] = i * j;
    end for;
  end for;
  // a = x;
end DelayForLoop;
