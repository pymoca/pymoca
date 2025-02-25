// From https://github.com/modelica/ModelicaSpecification/blob/MCP/0031/RationaleMCP/0031/name-mapping.md
record _R1
  parameter Real 'p';
  Real[2] 'arr';
  Real 'x';
end _R1;

record _R2
  _R1[2] 'mm'('p' = {2.0, 3.0});
  _R1 'm'('p' = 4.0);
end _R2;

model 'ManglingTest'
  _R2 'root';
  Real 'y' = 'root'.'m'.'x';
equation
  'root'.'mm'[1].'x' = sum('root'.'mm'[1].'arr');
  'root'.'mm'[2].'x' = sum('root'.'mm'[2].'arr');
  'root'.'m'.'x' = sum('root'.'m'.'arr');
  der('root'.'mm'[1].'arr') = {'root'.'mm'[1].'p', 1.0};
  der('root'.'mm'[2].'arr') = {'root'.'mm'[2].'p', 1.0};
  der('root'.'m'.'arr') = {'root'.'m'.'p', 1.0};
end 'ManglingTest';
