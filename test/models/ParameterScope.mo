model NestedClass
    parameter Real p;
end NestedClass;

model ScopeTest
    parameter Real p = 1.0;
    NestedClass nc(p = p);
    Real x;
equation
    der(x) = nc.p;
end ScopeTest;
