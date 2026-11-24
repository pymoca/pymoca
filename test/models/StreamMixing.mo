connector MixingPort
    flow Real m_flow;
    stream Real h_outflow;
end MixingPort;

model MixingSource
    MixingPort port;
    parameter Real m;
    parameter Real h0;
equation
    port.m_flow = -m;
    port.h_outflow = h0;
end MixingSource;

model MixingSink
    MixingPort port;
    output Real h_in;
equation
    port.h_outflow = 0.5;
    h_in = inStream(port.h_outflow);
end MixingSink;

model StreamMixing
    MixingSource src1(m = 2.0, h0 = 1.0);
    MixingSource src2(m = 1.0, h0 = 4.0);
    MixingSink snk;
    Real balance;
equation
    connect(src1.port, snk.port);
    connect(src2.port, snk.port);
    balance = src1.port.m_flow * actualStream(src1.port.h_outflow)
        + src2.port.m_flow * actualStream(src2.port.h_outflow)
        + snk.port.m_flow * actualStream(snk.port.h_outflow);
end StreamMixing;
