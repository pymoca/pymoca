model B
  Real X;
end B;

model C
  extends B;
end C;

model D
  C c;
end D;

model Wrong1
  extends D(c.x(nominal=2));
end Wrong1;

model Good1
  extends D(c.X(nominal=2));
end Good1;

model Wrong2
  B b(x=3);
end Wrong2;

model Good2
  B b(X=3);
end Good2;
