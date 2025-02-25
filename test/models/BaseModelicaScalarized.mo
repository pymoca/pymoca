// From https://github.com/modelica/ModelicaSpecification/blob/MCP/0031/RationaleMCP/0031/name-mapping.md
model 'ManglingTest'
  parameter Real 'root.mm[1].p' = 2.0;
  Real 'root.mm[1].arr[1]';
  Real 'root.mm[1].arr[2]';
  Real 'root.mm[1].x' = 'root.mm[1].arr[1]' + 'root.mm[1].arr[2]';
  parameter Real 'root.mm[2].p' = 3.0;
  Real 'root.mm[2].arr[1]';
  Real 'root.mm[2].arr[2]';
  Real 'root.mm[2].x' = 'root.mm[2].arr[1]' + 'root.mm[2].arr[2]';
  parameter Real 'root.m.p' = 4.0;
  Real 'root.m.arr[1]';
  Real 'root.m.arr[2]';
  Real 'root.m.x' = 'root.m.arr[1]' + 'root.m.arr[2]';
  Real 'y' = 'root.m.x';
equation
  der('root.mm[1].arr[1]') = 'root.mm[1].p';
  der('root.mm[1].arr[2]') = 1.0;
  der('root.mm[2].arr[1]') = 'root.mm[2].p';
  der('root.mm[2].arr[2]') = 1.0;
  der('root.m.arr[1]') = 'root.m.p';
  der('root.m.arr[2]') = 1.0;
end 'ManglingTest';
