connector HQBase
    Real H;
    flow Real Q;
end HQBase;

connector HQZ
    extends HQBase;
    Real Z;
end HQZ;

model Channel
    replaceable connector port = HQBase;
    port up;
    HQZ down;
equation
    up.H = down.H;
    up.Q + down.Q = 0;
end Channel;

model ChannelZ
    extends Channel(redeclare connector port = HQZ);
end ChannelZ;
