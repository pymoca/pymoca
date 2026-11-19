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

model Sink
    FluidPort port;
    output Real h_in;
    output Real h_act;
equation
    port.h_outflow = 0.5;
    h_in = inStream(port.h_outflow);
    h_act = actualStream(port.h_outflow);
end Sink;

model StreamConnector
    Source src;
    Sink snk;
equation
    connect(src.port, snk.port);
end StreamConnector;
