function CircleProperties
  input Real radius;
  output Real circumference;
  output Real area;
  output Real d;
  output Real e;
  output Real s1;
  output Real s2;
  output Real ignored;
protected
  Real diameter := radius*2;
algorithm
  circumference := 3.14159*diameter;
  area := 3.14159*radius^2;
  if area > 10 then
    d := 1;
    e := 10;
  else
    d := 2;
    e := area;
  end if;
  s1 := 1;
  s2 := 0;
  for i in 1:3 loop
    s1 := 2 * s1;
    s2 := s2 + 1;
  end for;
  ignored := 12;
end CircleProperties;

model FunctionCall
    Real r;
    output Real c, a, d, e, S1, S2;
equation
    (c, a, d, e, S1, S2) = CircleProperties(r);
end FunctionCall;
