// Example taken from Section 7.1 in the Modelica Specification (v3.3)
class A
    parameter Real a, b;
end A;

class B
    extends A(b=2);
end B;

class C
    extends B(a=1);
end C;

class C2
    B bcomp(b=3);
end C2;
