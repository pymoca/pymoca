// Example taken from Section 4.5.3 in the Modelica Specification (v3.3)
class C1
    class Voltage = Real(nominal=1);
    Voltage v1, v2;
end C1;

class C2
    extends C1(Voltage(nominal=1000));
end C2;
