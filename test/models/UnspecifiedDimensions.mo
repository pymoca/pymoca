model A
  parameter Real x[:, :];
  parameter Real y[:, 2];
end A;

model B
  A a(x = {{ 88.3224,   281.642,  143.011},
           { 58.8183,  -24.9845,      0.0},
           {-1.45483,       0.0,      0.0}},
      y = {{1, 2},
           {3, 4},
           {5, 6}});
end B;
