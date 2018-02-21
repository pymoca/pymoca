// TODO: Uncertain about correctness of OMC. What do other compilers say?
// The specification disagrees. See "Recursive instantiation of components", line "This performs lookup for D in M" in the discussion of the example.
// Or is that only because it's a component, and not an extends?
connector HQBase
    Real H;
    flow Real Q;
end HQBase;

connector HQZ
    extends HQBase;
    Real Z;
end HQZ;

model Channel
    connector HQZ
      extends HQBase;
      Real A;
    end HQZ;

    replaceable connector port = HQBase;
    port up;
    HQZ down;
equation
    up.H = down.H;
    up.Q + down.Q = 0;
end Channel;

model ChannelZ
    Channel c(redeclare connector port = HQZ);
end ChannelZ;
