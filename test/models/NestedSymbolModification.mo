model B
  Real x;
end B;

model C
  extends B;
end C;

model D
  C c;
end D;

model E
  extends D(c.x(nominal=2));
end E;
