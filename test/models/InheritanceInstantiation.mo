// Example taken from Section 7.1 in the Modelica Specification (v3.3)
class A
    parameter Real a=0;
    parameter Integer b=0;
    Real v[b];
end A;

class B
    extends A(b=2);
end B;

class C
    extends B(a=1);
end C;

class C2
    B bcomp1(b=3);
    B bcomp2(b=4);
    C bcomp3;
end C2;
