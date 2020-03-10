model IterativeSimplification
  Real x(start = 0);
  Real z;
  Real f;
  Real g;
  Real h;

  equation

  der(x) = 10 + z;
  f = 0;
  g = 1;
  f = (z-h);
  h = g;
end IterativeSimplification;
