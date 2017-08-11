package P0
  constant Real p[2] = {1, 2};
end P0;

package P1
  constant Real p = 1;
end P1;

package P2
  constant Real p = 2;
end P2;

model a
  replaceable package m = P1;
end a;

model b
  package m2 = P2;
  extends a(redeclare package m = m2);
  Real x;
  parameter Real y = m.p;
  parameter Real z = P0.p[0];
  Real w;
equation
  x = m.p;
  w = m.p * 2.0;
end b;
