within Level1.Level2.Level3;

function f
  input Real x;
  output Real y;
algorithm
  y := x * 2.0;
end f;

model Function5
  Real a,b;
equation
  a = f(b);
end Function5;
