model Ball
  parameter Real c = 1;
  input Real F_x;
  Real x(start=1);
  Real v_x(start=1);
  Real y(start=2);
  Real v_y(start=1);
equation
  der(v_x) = F_x;
  der(x) = v_x;
  der(v_y) = -c*y;
  der(y) = v_y;
end Ball;