function CircleProperties
  input Real radius;
  output Real circumference;
  output Real area;
protected
  Real diameter := radius*2;
algorithm
  circumference := 3.14159*diameter;
  area := 3.14159*radius^2;
end CircleProperties;

model FunctionCall
    Real r;
    output Real c, a;
equation
    (c, a) = CircleProperties(r);
end FunctionCall;
