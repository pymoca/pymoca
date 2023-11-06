// Test whether we properly
// - for i in 2:i <-- the last i is not the same as the first i
// - i in nested for loop in ForIndex _is_ that of the inner scope, e.g. for j in 1:i
// - We can apparently just redeclare builtin symbols/references like "sin", whatever we do take precedence

// TODO: We cannot define "cos" / "sin" etc at the _top_ level. Can we add a check for that? Or generally just
// add those symbols/functions to the root level of the tree from the get-go, and have the "already defined" check be more generic.

model DelayForLoop
  function cos
    input Real x;
    output Real y;
  algorithm
    y := 2 * x;
  end cos;

  constant Integer sin = 3;
  constant Integer i = 3;
  constant Integer j = 3;
  Real[3] x, y, a;
  input Real z[3];
  parameter Real delay_time(fixed=true);
  parameter Real eps;
equation
  for sin in 2:sin loop
    x[sin] = 5 * z[sin] * eps;
    y[sin] = delay(3 * a[sin] * eps, delay_time);
    for j in sin:3 loop
      a[sin] = sin * cos(j);
    end for;
  end for;
end DelayForLoop;
