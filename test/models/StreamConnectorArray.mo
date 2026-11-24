connector FluidPort
    flow Real m_flow;
    stream Real h_outflow;
end FluidPort;

model Source
    FluidPort port;
    parameter Real m = 3.0;
    parameter Real h0 = 2.0;
equation
    port.m_flow = -m;
    port.h_outflow = h0;
end Source;

model Pipe
    FluidPort a, b;
    Real h;
equation
    a.h_outflow = h;
    b.h_outflow = inStream(a.h_outflow);
end Pipe;

model StreamConnectorArray
    Source src1;
    Source src2(m = -3.0, h0 = 4.0);
    Pipe p[2];
    Real h_act[2];
equation
    connect(src1.port, p[1].a);
    connect(src2.port, p[2].a);
    for i in 1:2 loop
        p[i].h = 0.5 * i;
        h_act[i] = actualStream(p[i].a.h_outflow);
    end for;
end StreamConnectorArray;
