// Similar to RedeclarationScope, but now the class definitions are all
// _inside_ the class to be flattened (ChannelZ). They are therefore already
// in the instance tree, and so are the types of symbols in the Channel class.
model ChannelZ
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

    Channel c(redeclare connector port = HQZ);
end ChannelZ;
