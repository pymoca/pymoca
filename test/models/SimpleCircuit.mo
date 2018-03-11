// Model based on example in:
// Introduction to Modeling and Simulation
// of Technical and Physical Systems, Fritzon 2011

type Voltage = Real(unit="V");
type Current = Real(unit="A");

connector Pin
    Real v;
    flow Real i;
end Pin;

model TwoPin
    Pin p, n;
    Real v;
    Real i;
equation
    v = p.v - n.v;
    0 = p.i + n.i;
    i = p.i;
end TwoPin;

model Resistor
    extends TwoPin;
    parameter Real R;
equation
    v = i*R;
end Resistor;

model Capacitor
    extends TwoPin;
    parameter Real C;
equation
    i = C * der(v);
end Capacitor;

model Inductor
    extends TwoPin;
    parameter Real L;
equation
    v = L * der(i);
end Inductor;

model VsourceAC
    extends TwoPin;
    parameter Real VA = 220;
    parameter Real f = 50;
    constant Real PI=3.14159;
equation
    v = VA*sin(2*PI*f*time);
end VsourceAC;

model Ground
    Pin p;
equation
    p.v = 0;
end Ground;

model SimpleCircuit
    input Real i;
    Resistor R1(R=10);
    Capacitor C(C=0.01);
    Resistor R2(R=100);
    Inductor L(L=0.1);
    VsourceAC AC;
    Ground G;
equation
    i = L.i;
    connect(AC.p, R1.p); // 1. Capacitor circuit
    connect(R1.n, C.p);  //   Wire 2
    connect(C.n, AC.n);  //   Wire 3
    connect(R1.p, R2.p); // 4. Inductor circuit
    connect(R2.n, L.p);  //   Wire 5
    connect(L.n, C.n);   //   Wire 6
    connect(AC.n, G.p);  // 7. Ground
end SimpleCircuit;
